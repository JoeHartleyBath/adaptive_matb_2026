"""Calibrate a personalised SVM MWL model for one participant.

Fits SelectKBest(k=35) + StandardScaler + SVC(linear, C=1.0) on the
participant's own MATB calibration XDF data and saves deployment artefacts.

Classifier selection (2026-03-27, sweep_scratch_models.py / selectk_sweep):
  SVM-linear (k=35, C=1.0) gives best AUC3 and stable P(HIGH) at baseline.
  SVM-RBF overfit.  LogReg showed P(HIGH)≈0 saturation at resting baseline.

Deployment artefacts (consumed by mwl_estimator.py):
    pipeline.pkl      — sklearn Pipeline(StandardScaler + SVC linear)
    selector.pkl      — SelectKBest(k=35)
    norm_stats.json   — {mean, std, n_classes}
    model_config.json — threshold (10-fold CV Youden J), diagnostic metadata

Usage
-----
    python scripts/session/calibrate_participant.py calibrate \\
        --xdf-dir   D:/data/physiology/P001 \\
        --pid       P001 \\
        --out-dir   D:/data/models/P001
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# src/ on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eeg.extract_features import (  # noqa: E402
    FIXED_BANDS,
    _build_region_map,
    _extract_feat,
    load_all_features,
)
from ml.pretrain_loader import (  # noqa: E402
    PretrainDataDir,
    calibration_norm_features,
)

from eeg.xdf_loader import (  # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _detect_level,
    _extract_all_blocks,
    _find_block_bounds,
    _find_stream,
    _load_eeg_metadata,
    _merge_eeg_streams,
    _parse_markers,
)

# Seconds from LabRecorder-start to MATB scenario t=0.
# LabRecorder begins recording before the MATB process starts, so the XDF
# contains a short lead-in before the first scenario event.  Override via
# the MATB_SCENARIO_OFFSET_S environment variable if needed.
_MATB_SCENARIO_OFFSET_S = 12.0

# ---------------------------------------------------------------------------
# Frozen model config (2026-04-10, selectk_sweep_s005_block01, PSELF S005)
# ---------------------------------------------------------------------------
_CAL_K = 35             # SelectKBest k — S005 +block01 selectk sweep (2026-04-10)
_CAL_C = 1.0            # SVM-linear C — SVM-linear sweep winner
SEED = 42
_LORO_MIN_J = 0.10         # minimum LORO Youden J for quality gate (warn if below)
_CV_N_SPLITS = 10          # stratified k-fold CV splits for threshold selection

# Region config — shared ANT Neuro NA-271 cap layout
_DEFAULT_REGION_CFG = Path(__file__).resolve().parent.parent / "config" / "eeg_feature_extraction.yaml"
_ANALYSIS_SRATE = 128.0



# ===================================================================
# calibrate subcommand — helpers
# ===================================================================

def _find_calibration_scenario(xdf_path: Path, scenarios_dir: Path) -> Path | None:
    """Return the scenario .txt for a calibration XDF, or None if not found.

    Derives the filename from the XDF stem, e.g.:
        sub-PSELF_ses-S001_task-matb_acq-cal_c1_physio.xdf
        → full_calibration_pself_c1.txt
    """
    stem = xdf_path.stem
    m_pid = re.search(r"sub-(\w+)_", stem)
    m_cond = re.search(r"acq-cal_(c\d+)", stem)
    if not m_pid or not m_cond:
        return None
    candidate = scenarios_dir / f"full_calibration_{m_pid.group(1).lower()}_{m_cond.group(1)}.txt"
    return candidate if candidate.exists() else None


def _parse_scenario_blocks(scenario_path: Path) -> list[tuple[float, float, str]]:
    """Parse a calibration scenario .txt and return (start_s, end_s, level) per block."""
    _SBLOCK_RE = re.compile(
        r"(?P<time>\d+:\d{2}:\d{2});labstreaminglayer;marker;"
        r"STUDY/V0/calibration_condition/\d+/block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<event>START|END)"
    )
    open_starts: dict[str, float] = {}
    blocks: list[tuple[float, float, str]] = []
    for line in scenario_path.read_text(encoding="utf-8").splitlines():
        m = _SBLOCK_RE.match(line.strip().split("|")[0])
        if not m:
            continue
        parts = m.group("time").split(":")
        t_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        level = m.group("level")
        if m.group("event") == "START":
            open_starts[level] = float(t_s)
        elif m.group("event") == "END" and level in open_starts:
            blocks.append((open_starts.pop(level), float(t_s), level))
    return blocks


def _load_xdf_block(
    xdf_path: Path,
    expected_channels: list[str],
) -> list[tuple[np.ndarray, str]] | None:
    """Load and preprocess one calibration XDF.

    Returns a list of ``(epochs, level)`` pairs — one per block — or None on
    failure so the caller can skip gracefully.

    Falls back to scenario-file timing when no MATB marker stream was recorded
    (e.g. if LabRecorder started before the OpenMATB LSL outlet existed).
    """
    import pyxdf

    print(f"  {xdf_path.name} ...", end=" ", flush=True)
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None

    eeg_stream = _merge_eeg_streams(streams)
    marker_stream = _find_stream(streams, "Markers")

    if eeg_stream is None:
        print("SKIPPED (no EEG stream)")
        return None

    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(expected_channels):
        print(f"SKIPPED (channel count {n_ch} != {len(expected_channels)})")
        return None

    from eeg import EegPreprocessor, extract_windows, slice_block

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    # Decimate to analysis srate if the amplifier ran at a higher rate (e.g. 500 → 128 Hz).
    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > _ANALYSIS_SRATE * 1.1:
            factor = int(round(actual_srate / _ANALYSIS_SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts = eeg_ts[::factor]
            print(f"(decimated {actual_srate:.0f}->{_ANALYSIS_SRATE:.0f} Hz, factor={factor}) ", end="", flush=True)

    # Preprocess once -- all blocks share the same filtered signal.
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0], prewarm=eeg_data[:, 0])
    preprocessed = preprocessor.process(eeg_data)

    # --- Primary: extract every labelled block from the marker stream ---
    markers = _parse_markers(marker_stream)
    block_specs: list[tuple[float, float, str]] = _extract_all_blocks(markers)

    # --- Fallback: reconstruct from scenario timing when markers not recorded ---
    if not block_specs:
        _SCENARIOS_DIR = _REPO_ROOT / "experiment" / "scenarios"
        scenario_path = _find_calibration_scenario(xdf_path, _SCENARIOS_DIR)
        if scenario_path is None:
            print("SKIPPED (no START marker)")
            return None
        scenario_blocks = _parse_scenario_blocks(scenario_path)
        if not scenario_blocks:
            print("SKIPPED (no START marker)")
            return None
        offset_s = float(os.environ.get("MATB_SCENARIO_OFFSET_S", str(_MATB_SCENARIO_OFFSET_S)))
        matb_t0 = eeg_ts[0] + offset_s
        block_specs = [(matb_t0 + s, matb_t0 + e, lvl) for s, e, lvl in scenario_blocks]
        print(f"(scenario fallback: offset={offset_s:.0f}s) ", end="", flush=True)

    # --- Extract epochs for each block ---
    results: list[tuple[np.ndarray, str]] = []
    for start_ts, end_ts, level in block_specs:
        start_idx = int(np.searchsorted(eeg_ts, start_ts))
        end_idx = int(np.searchsorted(eeg_ts, end_ts))
        block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
        epochs = extract_windows(block, WINDOW_CONFIG)
        if epochs.shape[0] > 0:
            results.append((epochs, level))

    if not results:
        print("SKIPPED (no windows)")
        return None

    summary = ", ".join(f"{lvl}:{e.shape[0]}" for e, lvl in results)
    print(f"OK  {len(results)} blocks [{summary}]")
    return results


def _load_rest_xdf_block(
    xdf_path: Path,
    expected_channels: list[str],
    settle_s: float = 5.0,
) -> np.ndarray | None:
    """Load a resting-baseline XDF and return windowed EEG epochs.

    Searches the Markers stream for ``STUDY/V0/rest/START`` and
    ``STUDY/V0/rest/END`` (emitted by ``rest_baseline.txt``), trims the
    first *settle_s* seconds after START to let the EEG settle, then
    windows the remainder.

    Returns
    -------
    epochs : np.ndarray, shape (n_windows, n_channels, n_samples), or None
        None is returned on any loading or parsing failure.
    """
    import pyxdf

    _REST_START = "STUDY/V0/rest/START"
    _REST_END = "STUDY/V0/rest/END"

    print(f"  {xdf_path.name} (rest) ...", end=" ", flush=True)
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None

    eeg_stream = _merge_eeg_streams(streams)
    marker_stream = _find_stream(streams, "Markers")

    if eeg_stream is None:
        print("SKIPPED (no EEG stream)")
        return None

    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(expected_channels):
        print(f"SKIPPED (channel count {n_ch} != {len(expected_channels)})")
        return None

    markers = _parse_markers(marker_stream)
    start_ts = end_ts = None
    for ts, text in markers:
        label = text.split("|")[0]
        if label == _REST_START:
            start_ts = ts
        elif label == _REST_END:
            end_ts = ts

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    # Decimate to analysis srate if the amplifier ran at a higher rate (e.g. 500 → 128 Hz).
    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > _ANALYSIS_SRATE * 1.1:
            factor = int(round(actual_srate / _ANALYSIS_SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts = eeg_ts[::factor]
            print(f"(decimated {actual_srate:.0f}->{_ANALYSIS_SRATE:.0f} Hz, factor={factor}) ", end="", flush=True)

    if start_ts is None or end_ts is None:
        # Fallback: markers not recorded — use full XDF duration minus settle.
        # The rest recording is the entire XDF (no other task mixed in).
        start_ts = eeg_ts[0]
        end_ts = eeg_ts[-1]
        print("(no REST markers — using full XDF) ", end="", flush=True)

    from eeg import EegPreprocessor, extract_windows, slice_block
    analysis_start_ts = start_ts + settle_s
    if analysis_start_ts >= end_ts:
        print("SKIPPED (rest block too short after settling)")
        return None

    start_idx = int(np.searchsorted(eeg_ts, analysis_start_ts))
    end_idx = int(np.searchsorted(eeg_ts, end_ts))

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0], prewarm=eeg_data[:, 0])
    preprocessed = preprocessor.process(eeg_data)

    block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
    epochs = extract_windows(block, WINDOW_CONFIG)
    if epochs.shape[0] == 0:
        print("SKIPPED (no windows)")
        return None

    print(f"OK  epochs={epochs.shape[0]}")

    # --- 25 Hz spike check (non-blocking, diagnostic only) ---
    # Computes the 25 Hz spike ratio across channels on the rest EEG.
    # A UK mains sub-harmonic (25 Hz = 50/2) can contaminate the beta band
    # and cause extreme FM_Beta z-scores during calibration and inference.
    # This check runs on the already-preprocessed signal and prints a loud
    # warning if the spike is present so the operator can address the source
    # before proceeding to calibration.
    try:
        import scipy.signal as _sp_sig
        _rest_seg = preprocessed[:, start_idx:end_idx]
        _nperseg  = min(int(_ANALYSIS_SRATE * 4), _rest_seg.shape[1] // 2)
        if _nperseg >= int(_ANALYSIS_SRATE * 2):
            _freqs, _pxx = _sp_sig.welch(
                _rest_seg, fs=_ANALYSIS_SRATE, nperseg=_nperseg,
                noverlap=_nperseg // 2, window="hann", axis=1,
            )
            _i25    = int(np.argmin(np.abs(_freqs - 25.0)))
            _fmask  = (
                ((_freqs >= 22.0) & (_freqs <= 24.0))
                | ((_freqs >= 26.0) & (_freqs <= 28.0))
            )
            _p25    = _pxx[:, _i25]
            _pfloor = _pxx[:, _fmask].mean(axis=1)
            _valid  = _pfloor > 0
            _ratios = np.where(_valid, _p25 / np.where(_valid, _pfloor, 1.0), np.nan)
            _p90    = float(np.nanpercentile(_ratios, 90))
            _worst  = int(np.nanargmax(_ratios))
            _wname  = expected_channels[_worst] if _worst < len(expected_channels) else str(_worst)
            if _p90 >= 3.0:
                print()
                print("!" * 72)
                print("!  WARNING: 25 Hz SPIKE IN REST EEG (p90 spike_ratio = "
                      f"{_p90:.1f}x, worst channel: {_wname})")
                print("!  This sits in the beta band and will contaminate FM_Beta")
                print("!  norm_stats and calibration features.")
                print("!  Likely cause: 50 Hz mains sub-harmonic from a powered")
                print("!  device (monitor, laptop charger, LED dimmer).")
                print("!  ACTION: identify and switch off the source, then")
                print("!  re-record the rest baseline before proceeding.")
                print("!" * 72)
                print()
            else:
                print(f"  [25 Hz check: p90 ratio={_p90:.2f} — OK]")
    except Exception:
        pass   # never block the rest of calibration on a diagnostic check

    return epochs


def _compute_resting_norm(
    resting_epochs: np.ndarray,
    srate: float,
    region_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean & std from resting-baseline epochs for norm_stats.json.

    Returns (mean, std) arrays of shape (n_features,).
    """
    X_rest, _ = _extract_feat(resting_epochs, srate, region_map)
    mean = X_rest.mean(axis=0)
    std = X_rest.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


# ===================================================================
# LORO CV helper
# ===================================================================

def _compute_loro_threshold(
    xdf_X_norm: list[np.ndarray],
    xdf_y: list[np.ndarray],
    cal_k: int,
    cal_c: float,
    seed: int,
) -> tuple[float | None, float | None, list[float]]:
    """Leave-One-Run-Out CV threshold for Youden J.

    Trains a fresh SelectKBest + StandardScaler + SVC on all-but-one XDFs,
    predicts p_high on the held-out XDF, then pools the held-out predictions
    across all folds and computes a single Youden J threshold.

    Returns (loro_thr, loro_j, fold_j_scores).  Returns (None, None, []) when
    any fold has fewer than 2 classes in train or test.
    """
    n = len(xdf_X_norm)
    pool_p: list[np.ndarray] = []
    pool_y: list[np.ndarray] = []
    fold_j: list[float] = []

    for i in range(n):
        train_X = np.concatenate([xdf_X_norm[j] for j in range(n) if j != i])
        train_y = np.concatenate([xdf_y[j] for j in range(n) if j != i])
        test_X = xdf_X_norm[i]
        test_y = xdf_y[i]

        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            return None, None, []

        k = min(cal_k, train_X.shape[1])
        sel = SelectKBest(f_classif, k=k)
        train_X_sel = sel.fit_transform(train_X, train_y)
        sc = StandardScaler()
        train_X_sc = sc.fit_transform(train_X_sel)
        svc = SVC(
            kernel="linear", C=cal_c,
            class_weight="balanced", probability=True, random_state=seed,
        )
        svc.fit(train_X_sc, train_y)

        test_X_sc = sc.transform(sel.transform(test_X))
        p_high_fold = svc.predict_proba(test_X_sc)[:, -1]

        # Per-fold J (diagnostic)
        y_bin_fold = (test_y == test_y.max()).astype(int)
        fpr_f, tpr_f, _ = roc_curve(y_bin_fold, p_high_fold)
        fold_j.append(float(np.max(tpr_f - fpr_f)))

        pool_p.append(p_high_fold)
        pool_y.append(y_bin_fold)

    # Pooled ROC over all held-out folds
    all_p = np.concatenate(pool_p)
    all_y = np.concatenate(pool_y)
    if len(np.unique(all_y)) < 2:
        return None, None, fold_j

    fpr, tpr, thr = roc_curve(all_y, all_p)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thr[best_idx]), float(j_scores[best_idx]), fold_j


# ===================================================================
# Stratified k-fold CV threshold helper
# ===================================================================

def _compute_kfold_threshold(
    cal_X_norm: np.ndarray,
    cal_y: np.ndarray,
    cal_k: int,
    cal_c: float,
    seed: int,
    n_splits: int = _CV_N_SPLITS,
) -> tuple[float | None, float | None]:
    """Stratified k-fold CV threshold for Youden J (primary threshold method).

    Pools held-out p_high predictions across all folds and returns the
    Youden-optimal threshold from the pooled ROC curve.

    Returns (kfold_thr, kfold_j).  Returns (None, None) when data has
    fewer than 2 classes or fewer samples than n_splits.
    """
    y_binary = (cal_y == cal_y.max()).astype(int)
    if len(np.unique(y_binary)) < 2 or len(cal_y) < n_splits:
        return None, None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pool_p = np.zeros(len(cal_y))

    for tr_idx, te_idx in cv.split(cal_X_norm, cal_y):
        Xtr = cal_X_norm[tr_idx]; ytr = cal_y[tr_idx]
        Xte = cal_X_norm[te_idx]

        k = min(cal_k, Xtr.shape[1])
        sel = SelectKBest(f_classif, k=k)
        Xtr_s = sel.fit_transform(Xtr, ytr)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_s)
        svc = SVC(
            kernel="linear", C=cal_c,
            class_weight="balanced", probability=True, random_state=seed,
        )
        svc.fit(Xtr_s, ytr)
        pool_p[te_idx] = svc.predict_proba(sc.transform(sel.transform(Xte)))[:, -1]

    fpr, tpr, thr = roc_curve(y_binary, pool_p)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thr[best_idx]), float(j_scores[best_idx])


# ===================================================================
# calibrate subcommand — main function
# ===================================================================

def calibrate(
    xdf_dir: Path,
    pid: str,
    out_dir: Path,
    resting_xdf: Path | None = None,
    region_cfg: Path = _DEFAULT_REGION_CFG,
    binary: bool = False,
) -> None:
    """Fit and save a personalised SVM MWL deployment model for one participant.

    Steps:
      1. Load participant's MATB calibration XDFs, extract features.
      2. Compute resting-baseline norm stats (from ``resting_xdf`` if given,
         else LOW block fallback).
      3. Z-normalise features; fit SelectKBest(k=35) + StandardScaler + SVC(linear, C=1.0).
      4. Compute Youden J threshold via 10-fold CV.
      5. Save: pipeline.pkl, selector.pkl, norm_stats.json, model_config.json.
    """
    from ml.dataset import LABEL_MAP
    from sklearn.preprocessing import StandardScaler

    # --- 1. Load calibration XDFs ---
    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    # Only use the latest version of each acquisition: exclude files whose stem
    # ends with _old1, _old2, etc.  Those are superseded recordings created when
    # LabRecorder was restarted mid-session; they may use a hardcoded fallback
    # offset with no Markers stream, producing mislabelled training windows.
    import re as _re
    _all_xdf = sorted(Path(xdf_dir).glob("*.xdf"))
    xdf_files = [f for f in _all_xdf if not _re.search(r"_old\d+$", f.stem) and "_acq-cal_" in f.stem]
    if not xdf_files:
        sys.exit(f"ERROR: No .xdf files found in {xdf_dir}")

    print(f"\nLoading {len(xdf_files)} calibration XDF(s) for {pid}:")
    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    xdf_epoch_groups: list[list[np.ndarray]] = []   # per-XDF list of epoch arrays
    xdf_label_groups: list[list[np.ndarray]] = []   # per-XDF list of label arrays
    xdf_files_loaded: list[Path] = []               # XDFs that yielded ≥1 block

    for xdf_path in xdf_files:
        results = _load_xdf_block(xdf_path, expected_channels)
        if results is None:
            continue
        xdf_epochs_this: list[np.ndarray] = []
        xdf_labels_this: list[np.ndarray] = []
        for epochs, level in results:
            label = LABEL_MAP[level]
            all_epochs.append(epochs)
            all_labels.append(np.full(epochs.shape[0], label, dtype=np.int64))
            xdf_epochs_this.append(epochs)
            xdf_labels_this.append(np.full(epochs.shape[0], label, dtype=np.int64))
        xdf_epoch_groups.append(xdf_epochs_this)
        xdf_label_groups.append(xdf_labels_this)
        xdf_files_loaded.append(xdf_path)

    if not all_epochs:
        sys.exit("ERROR: No valid calibration blocks loaded.")

    cal_epochs = np.concatenate(all_epochs)
    cal_labels_raw = np.concatenate(all_labels)

    if binary:
        # Binary: keep only LOW and HIGH, remap to 0/1
        keep = (cal_labels_raw == LABEL_MAP["LOW"]) | (cal_labels_raw == LABEL_MAP["HIGH"])
        cal_epochs = cal_epochs[keep]
        cal_labels_raw = cal_labels_raw[keep]
        cal_y = (cal_labels_raw == LABEL_MAP["HIGH"]).astype(np.int64)
        print(f"\n  Calibration (binary): {len(cal_y)} windows  "
              f"(LOW={int((cal_y == 0).sum())}, HIGH={int((cal_y == 1).sum())})")
    else:
        # 3-class: LOW=0, MODERATE=1, HIGH=2
        cal_y = cal_labels_raw
        print(f"\n  Calibration (3-class): {len(cal_y)} windows  "
              f"(LOW={int((cal_y == 0).sum())}, MODERATE={int((cal_y == 1).sum())}, HIGH={int((cal_y == 2).sum())})")
    # Per-XDF label vectors and window counts (post-filter) for LORO CV.
    _xdf_y_loro: list[np.ndarray] = []
    _xdf_window_counts: list[int] = []
    for _lbl_list in xdf_label_groups:
        _flat = np.concatenate(_lbl_list)
        if binary:
            _mask = (_flat == LABEL_MAP["LOW"]) | (_flat == LABEL_MAP["HIGH"])
            _flat = _flat[_mask]
            _xdf_y_loro.append((_flat == LABEL_MAP["HIGH"]).astype(np.int64))
        else:
            _xdf_y_loro.append(_flat.copy())
        _xdf_window_counts.append(len(_xdf_y_loro[-1]))
    # --- 3. Extract features ---
    ch_names = expected_channels
    region_map = _build_region_map(region_cfg, ch_names)
    cal_X, feat_names = _extract_feat(cal_epochs, _ANALYSIS_SRATE, region_map)

    # --- 4. Resting-baseline norm stats ---
    if resting_xdf is not None and resting_xdf.exists():
        print(f"\n  Resting baseline from: {resting_xdf.name}")
        rest_epochs = _load_rest_xdf_block(resting_xdf, expected_channels)
        if rest_epochs is not None:
            norm_mean, norm_std = _compute_resting_norm(
                rest_epochs, _ANALYSIS_SRATE, region_map)
        else:
            print("  WARNING: Resting XDF failed, using LOW block fallback")
            low_mask = cal_y == 0
            norm_mean, norm_std = _compute_resting_norm(
                cal_epochs[low_mask], _ANALYSIS_SRATE, region_map)
    else:
        # Fallback: use LOW-level calibration block as baseline proxy
        print("\n  No resting XDF provided — using LOW block as baseline")
        low_mask = cal_y == 0
        if low_mask.any():
            low_X, _ = _extract_feat(
                cal_epochs[low_mask], _ANALYSIS_SRATE, region_map)
            norm_mean = low_X.mean(axis=0)
            norm_std = low_X.std(axis=0)
            norm_std[norm_std < 1e-12] = 1.0
        else:
            norm_mean = cal_X.mean(axis=0)
            norm_std = cal_X.std(axis=0)
            norm_std[norm_std < 1e-12] = 1.0

    # --- 5. Calibration-normalise, select features, fit classifier ---
    cal_X_norm = (cal_X - norm_mean) / norm_std

    # Slice cal_X_norm into per-XDF arrays for LORO CV.
    _loro_split_pts = np.cumsum(_xdf_window_counts[:-1]).tolist() if len(_xdf_window_counts) > 1 else []
    xdf_X_norm_loro: list[np.ndarray] = list(np.split(cal_X_norm, _loro_split_pts))

    k = min(_CAL_K, cal_X_norm.shape[1])
    own_selector = SelectKBest(f_classif, k=k)
    cal_X_sel = own_selector.fit_transform(cal_X_norm, cal_y)
    own_sc = StandardScaler()
    cal_X_sc = own_sc.fit_transform(cal_X_sel)
    clf = SVC(
        kernel="linear", C=_CAL_C,
        class_weight="balanced", probability=True, random_state=SEED,
    )
    clf.fit(cal_X_sc, cal_y)
    deploy_selector = own_selector
    deploy_pipe = Pipeline([
        ("sc", own_sc),
        ("clf", clf),
    ])
    print(f"  Fitted SelectKBest(k={k}) + StandardScaler + SVC(linear, C={_CAL_C}).")

    # --- 6. Youden's J threshold (binary HIGH vs NOT-HIGH) ---
    # P(HIGH) is the last probability column — matches mwl_estimator.py behaviour
    # where p_high_col = n_classes - 1.
    # Use clf.predict_proba directly on cal_X_sc (already scaled) to avoid
    # double-scaling via deploy_pipe (which would apply own_sc a second time).
    p_high = clf.predict_proba(cal_X_sc)[:, -1]
    y_binary = (cal_y == LABEL_MAP["HIGH"]).astype(int)
    if len(np.unique(y_binary)) < 2:
        sys.exit(
            "ERROR: calibration data must contain both HIGH and non-HIGH windows "
            "to compute Youden J threshold.\n"
            "Check that all three difficulty levels (LOW, MODERATE, HIGH) are "
            "present in the XDFs and were loaded successfully."
        )
    fpr, tpr, thr = roc_curve(y_binary, p_high)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    train_youden_thr = float(thr[best_idx])
    train_youdens_j = float(j_scores[best_idx])

    # LORO CV (quality gate) + 10-fold CV (primary threshold method)
    loro_thr: float | None = None
    loro_j: float | None = None
    loro_fold_j: list[float] = []
    kfold_thr: float | None = None
    kfold_j: float | None = None
    threshold_method = "train_set"

    # 10-fold stratified CV — primary threshold
    kfold_thr, kfold_j = _compute_kfold_threshold(
        cal_X_norm, cal_y, _CAL_K, _CAL_C, SEED,
    )
    if kfold_thr is not None:
        youden_thr = kfold_thr
        youdens_j = kfold_j
        threshold_method = "10fold_cv"
    else:
        youden_thr = train_youden_thr
        youdens_j = train_youdens_j
        threshold_method = "train_set_fallback"
        print("  WARNING: 10-fold CV failed — falling back to training-set threshold")

    # LORO — quality gate only
    if len(xdf_files_loaded) >= 2:
        loro_thr, loro_j, loro_fold_j = _compute_loro_threshold(
            xdf_X_norm_loro, _xdf_y_loro, _CAL_K, _CAL_C, SEED,
        )
        if loro_j is not None and loro_j < _LORO_MIN_J:
            print(f"  WARNING: LORO J={loro_j:.4f} < {_LORO_MIN_J:.2f} "
                  f"— low cross-run transfer; model may not generalise across sessions")

    # --- 7. Save deployment artefacts ---
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy_pipe, out_dir / "pipeline.pkl")
    joblib.dump(deploy_selector, out_dir / "selector.pkl")

    norm_stats = {
        "mean": norm_mean.tolist(),
        "std": norm_std.tolist(),
        "n_classes": int(len(np.unique(cal_y))),
    }
    (out_dir / "norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2), encoding="utf-8")

    model_config = {
        "youden_threshold": round(youden_thr, 6),
        "youdens_j": round(youdens_j, 6),
        "threshold_method": threshold_method,
        "train_youden_threshold": round(train_youden_thr, 6),
        "train_youdens_j": round(train_youdens_j, 6),
        "kfold_youden_threshold": round(kfold_thr, 6) if kfold_thr is not None else None,
        "kfold_youdens_j": round(kfold_j, 6) if kfold_j is not None else None,
        "kfold_n_splits": _CV_N_SPLITS if kfold_thr is not None else None,
        "loro_youden_threshold": round(loro_thr, 6) if loro_thr is not None else None,
        "loro_youdens_j": round(loro_j, 6) if loro_j is not None else None,
        "loro_n_folds": len(loro_fold_j) if loro_fold_j else None,
        "loro_fold_j_scores": [round(v, 6) for v in loro_fold_j] if loro_fold_j else None,
        "n_classes": int(len(np.unique(cal_y))),
        "model_k": int(deploy_selector.k),
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8")

    # Brief summary for session log
    cal_preds = clf.predict(cal_X_sc)
    cal_acc = float(np.mean(cal_preds == cal_y))
    print(f"\n  Calibration accuracy: {cal_acc:.1%}")
    print(f"  Train-set threshold:  {train_youden_thr:.4f}  (J={train_youdens_j:.4f})")
    if kfold_thr is not None:
        print(f"  10-fold CV threshold: {kfold_thr:.4f}  (J={kfold_j:.4f})")
    if loro_thr is not None:
        print(f"  LORO threshold:       {loro_thr:.4f}  (J={loro_j:.4f}, "
              f"folds={len(loro_fold_j)}, fold_J={[round(v,3) for v in loro_fold_j]})  [quality gate]")
    print(f"  Deployed threshold:   {youden_thr:.4f}  [method={threshold_method}]")
    print(f"  Saved deployment artefacts to {out_dir}/")
    print(f"    pipeline.pkl, selector.pkl, norm_stats.json, model_config.json")


# ===================================================================
# CLI entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a per-participant SVM MWL deployment model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- calibrate ----
    cal = sub.add_parser(
        "calibrate",
        help="Fit personalised SVM model for one participant.",
    )
    cal.add_argument("--xdf-dir", type=Path, required=True,
                      help="Directory with calibration .xdf files for this participant.")
    cal.add_argument("--pid", type=str, required=True,
                      help="Participant identifier (e.g. P001).")
    cal.add_argument("--out-dir", type=Path, required=True,
                      help="Where to save pipeline.pkl, selector.pkl, norm_stats.json.")
    cal.add_argument("--resting-xdf", type=Path, default=None,
                      help="Optional resting-baseline .xdf for norm stats.")
    cal.add_argument("--region-cfg", type=Path, default=_DEFAULT_REGION_CFG,
                      help="EEG region YAML config.")
    cal.add_argument("--binary", action="store_true",
                      help="Train binary (LOW vs HIGH) instead of 3-class.")

    args = parser.parse_args()

    calibrate(
        xdf_dir=args.xdf_dir,
        pid=args.pid,
        out_dir=args.out_dir,
        resting_xdf=args.resting_xdf,
        region_cfg=args.region_cfg,
        binary=args.binary,
    )


if __name__ == "__main__":
    main()
