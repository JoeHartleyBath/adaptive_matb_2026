"""Train a group LogReg model and calibrate it for a single participant.

Two subcommands
---------------
    train-group   Build (or skip if up-to-date) the group LogReg model from
                  all available pretrain participants.  Run OFFLINE, before
                  any session.

    calibrate     Load the pre-built group model, compute resting-baseline
                  norm stats, warm-start fine-tune on the participant's
                  calibration XDFs, and save deployment artefacts.  Run
                  DURING session — must be fast (no group training).

Deployment artefacts (consumed by mwl_estimator.py)
---------------------------------------------------
    pipeline.pkl    joblib — sklearn Pipeline (StandardScaler + LogReg)
    selector.pkl    joblib — SelectKBest (k=30)
    norm_stats.json {"mean": [...], "std": [...]}  (54 floats each)

Usage
-----
    # Offline — train / update the group model
    python scripts/calibrate_participant_logreg.py train-group \\
        --data-dir  D:/adaptive_matb_2026_data/processed/pretrain/continuous \\
        --model-dir D:/adaptive_matb_2026_data/models/group

    # During session — calibrate one participant
    python scripts/calibrate_participant_logreg.py calibrate \\
        --group-dir D:/adaptive_matb_2026_data/models/group \\
        --xdf-dir   D:/adaptive_matb_2026_data/raw/training/P001 \\
        --pid       P001 \\
        --out-dir   D:/adaptive_matb_2026_data/models/P001
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# src/ on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
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

# Reuse XDF helpers from the dataset builder
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from build_mwl_training_dataset import (  # noqa: E402
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
# Frozen model config (from dc_logreg_hyperparameter_plateau)
# ---------------------------------------------------------------------------
_LOGREG_K = 30             # SelectKBest k (f_classif)
_LOGREG_C = 0.003          # ElasticNet regularisation strength (joint sweep winner)
_LOGREG_PENALTY = "elasticnet"  # penalty (joint sweep winner)
_LOGREG_L1_RATIO = 0.5     # ElasticNet mixing (requires solver=saga)
_WARM_C = 0.1              # weak warm-start C (WS-weak winner)
SEED = 42

# Region config — shared ANT Neuro NA-271 cap layout
_DEFAULT_REGION_CFG = Path(__file__).resolve().parent.parent / "config" / "eeg_feature_extraction.yaml"
_QC_CONFIG = _REPO_ROOT / "config" / "pretrain_qc.yaml"
_ANALYSIS_SRATE = 128.0


def _load_exclude() -> set[str]:
    """Load excluded participant IDs from pretrain QC config."""
    cfg = yaml.safe_load(_QC_CONFIG.read_text())
    excluded = cfg.get("excluded_participants") or {}
    return set(excluded.keys())


# ===================================================================
# train-group subcommand
# ===================================================================

def _is_group_model_current(model_dir: Path, n_available: int) -> bool:
    """Return True if a saved group model exists with the right participant count."""
    meta_path = model_dir / "group_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        return meta.get("n_participants", 0) == n_available
    except (json.JSONDecodeError, OSError):
        return False


def train_group(
    data_dir: Path,
    model_dir: Path,
    region_cfg: Path = _DEFAULT_REGION_CFG,
) -> None:
    """Train the group LogReg on ALL available pretrain participants.

    Lazy: skips training if the saved model already has the same
    participant count.  Re-trains automatically when new participants
    are added to the data directory.

    Saves to *model_dir*:
        group_pipeline.pkl   — Pipeline(StandardScaler + LogReg)
        group_selector.pkl   — SelectKBest(k=30)
        group_meta.json      — {n_participants, pids, built_at}
    """
    data = PretrainDataDir(data_dir)
    exclude = _load_exclude()
    all_pids = [p for p in data.available_pids() if p not in exclude]

    if not all_pids:
        sys.exit("ERROR: No participants found after exclusion.")

    # Lazy check — skip if model already matches
    if _is_group_model_current(model_dir, len(all_pids)):
        print(f"Group model already up-to-date ({len(all_pids)} participants). "
              "Skipping training.")
        return

    print(f"Training group LogReg on {len(all_pids)} participants")
    print(f"  Config: K={_LOGREG_K}, C={_LOGREG_C}, {_LOGREG_PENALTY}(l1_ratio={_LOGREG_L1_RATIO}), StandardScaler")
    print(f"  Data:   {data_dir}")
    print()

    # --- Load & extract features (uses disk cache) ---
    ch_names = data.channel_names()
    region_map = _build_region_map(region_cfg, ch_names)
    feat_data = load_all_features(data, all_pids, _ANALYSIS_SRATE, region_map)

    # --- Calibration-normalise every participant (ADR-0004) ---
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for pid in all_pids:
        d = feat_data[pid]
        X_norm = calibration_norm_features(
            d["task_X"], d["fix_X"], d["forest_X"], d["forest_bidx"],
        )
        X_parts.append(X_norm)
        y_parts.append(d["task_y"])

    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)
    print(f"\n  Total windows: {len(y_all)}  "
          f"(class 0: {(y_all == 0).sum()}, class 1: {(y_all == 1).sum()})")

    # --- Feature selection ---
    t0 = time.time()
    selector = SelectKBest(f_classif, k=_LOGREG_K)
    X_sel = selector.fit_transform(X_all, y_all)

    # --- Pipeline: StandardScaler + LogReg ---
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            C=_LOGREG_C, solver="saga",
            l1_ratio=_LOGREG_L1_RATIO, max_iter=2000,
            class_weight="balanced", random_state=SEED)),
    ])
    pipe.fit(X_sel, y_all)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    # --- Save artefacts ---
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "group_pipeline.pkl")
    joblib.dump(selector, model_dir / "group_selector.pkl")

    meta = {
        "n_participants": len(all_pids),
        "pids": sorted(all_pids),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "K": _LOGREG_K,
            "C": _LOGREG_C,
            "penalty": _LOGREG_PENALTY,
            "l1_ratio": _LOGREG_L1_RATIO,
            "scaler": "standard",
            "seed": SEED,
        },
        "n_windows": int(len(y_all)),
        "n_features_raw": int(X_all.shape[1]),
    }
    (model_dir / "group_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n  Saved group model to {model_dir}")
    print(f"  Participants: {len(all_pids)}  Features selected: {_LOGREG_K}")
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

    # Preprocess once — all blocks share the same filtered signal.
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
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
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
    epochs = extract_windows(block, WINDOW_CONFIG)
    if epochs.shape[0] == 0:
        print("SKIPPED (no windows)")
        return None

    print(f"OK  epochs={epochs.shape[0]}")
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
# calibrate subcommand — main function
# ===================================================================

def calibrate(
    group_dir: Path,
    xdf_dir: Path,
    pid: str,
    out_dir: Path,
    resting_xdf: Path | None = None,
    region_cfg: Path = _DEFAULT_REGION_CFG,
) -> None:
    """Load group model, fine-tune for one participant, save deployment artefacts.

    Steps:
      1. Load pre-built group model (group_pipeline.pkl, group_selector.pkl).
      2. Load participant's calibration XDFs → extract features.
      3. Compute resting-baseline norm stats (from resting XDF or from
         the LOW-level calibration block as fallback).
      4. Calibration-normalise the task features.
      5. Warm-start LogReg from group weights, refit with C=0.1.
      6. Save: pipeline.pkl, selector.pkl, norm_stats.json.
    """
    from ml.dataset import LABEL_MAP

    # --- 1. Load group model ---
    group_pipe_path = group_dir / "group_pipeline.pkl"
    group_sel_path = group_dir / "group_selector.pkl"
    for p in (group_pipe_path, group_sel_path):
        if not p.exists():
            sys.exit(f"ERROR: Group model not found: {p}\n"
                     "       Run 'train-group' first.")

    group_pipe: Pipeline = joblib.load(group_pipe_path)
    group_selector: SelectKBest = joblib.load(group_sel_path)
    group_clf = group_pipe.named_steps["clf"]
    group_sc = group_pipe.named_steps["sc"]
    print(f"Loaded group model from {group_dir}")

    # --- 2. Load calibration XDFs ---
    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    xdf_files = sorted(Path(xdf_dir).glob("*.xdf"))
    if not xdf_files:
        sys.exit(f"ERROR: No .xdf files found in {xdf_dir}")

    print(f"\nLoading {len(xdf_files)} calibration XDF(s) for {pid}:")
    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for xdf_path in xdf_files:
        results = _load_xdf_block(xdf_path, expected_channels)
        if results is None:
            continue
        for epochs, level in results:
            label = LABEL_MAP[level]
            all_epochs.append(epochs)
            all_labels.append(np.full(epochs.shape[0], label, dtype=np.int64))

    if not all_epochs:
        sys.exit("ERROR: No valid calibration blocks loaded.")

    cal_epochs = np.concatenate(all_epochs)
    cal_labels_raw = np.concatenate(all_labels)

    # Remap to binary (0=LOW, 1=HIGH; drop MODERATE if present)
    keep = (cal_labels_raw == LABEL_MAP["LOW"]) | (cal_labels_raw == LABEL_MAP["HIGH"])
    cal_epochs = cal_epochs[keep]
    cal_labels_raw = cal_labels_raw[keep]
    cal_y = (cal_labels_raw == LABEL_MAP["HIGH"]).astype(np.int64)

    print(f"\n  Calibration: {len(cal_y)} windows  "
          f"(LOW={int((cal_y == 0).sum())}, HIGH={int((cal_y == 1).sum())})")

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

    # --- 5. Calibration-normalise and warm-start fine-tune ---
    cal_X_norm = (cal_X - norm_mean) / norm_std
    cal_X_sel = group_selector.transform(cal_X_norm)
    cal_X_sc = group_sc.transform(cal_X_sel)

    clf = LogisticRegression(
        C=_WARM_C, max_iter=2000, warm_start=True,
        class_weight="balanced", random_state=SEED,
    )
    clf.classes_ = group_clf.classes_.copy()
    clf.coef_ = group_clf.coef_.copy()
    clf.intercept_ = group_clf.intercept_.copy()
    clf.fit(cal_X_sc, cal_y)

    # Build deployment pipeline (same scaler as group, new clf)
    deploy_pipe = Pipeline([
        ("sc", group_sc),
        ("clf", clf),
    ])

    # --- 6. Save deployment artefacts ---
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy_pipe, out_dir / "pipeline.pkl")
    joblib.dump(group_selector, out_dir / "selector.pkl")

    norm_stats = {
        "mean": norm_mean.tolist(),
        "std": norm_std.tolist(),
    }
    (out_dir / "norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2), encoding="utf-8")

    # Brief summary for session log
    cal_probs = clf.predict_proba(cal_X_sc)[:, 1]
    cal_acc = float(np.mean((cal_probs >= 0.5).astype(int) == cal_y))
    print(f"\n  Calibration accuracy: {cal_acc:.1%}")
    print(f"  Saved deployment artefacts to {out_dir}/")
    print(f"    pipeline.pkl, selector.pkl, norm_stats.json")


# ===================================================================
# CLI entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train group LogReg and calibrate per-participant deployment model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train-group ----
    tg = sub.add_parser(
        "train-group",
        help="Build (or skip if current) the group LogReg model.",
    )
    tg.add_argument("--data-dir", type=Path, required=True,
                     help="Pretrain continuous HDF5 directory.")
    tg.add_argument("--model-dir", type=Path, required=True,
                     help="Where to save group_pipeline.pkl etc.")
    tg.add_argument("--region-cfg", type=Path, default=_DEFAULT_REGION_CFG,
                     help="EEG region YAML config.")

    # ---- calibrate ----
    cal = sub.add_parser(
        "calibrate",
        help="Fine-tune group model for one participant (fast).",
    )
    cal.add_argument("--group-dir", type=Path, required=True,
                      help="Directory with group_pipeline.pkl etc.")
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

    args = parser.parse_args()

    if args.command == "train-group":
        train_group(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            region_cfg=args.region_cfg,
        )
    elif args.command == "calibrate":
        calibrate(
            group_dir=args.group_dir,
            xdf_dir=args.xdf_dir,
            pid=args.pid,
            out_dir=args.out_dir,
            resting_xdf=args.resting_xdf,
            region_cfg=args.region_cfg,
        )


if __name__ == "__main__":
    main()
