"""_tmp_calibrate_from_jsonl.py

One-off offline training for PDRY01 / S001 using the JSONL calibration
recordings captured on 2026-04-22.

Context
-------
The normal calibrate_participant.py CLI only reads XDFs.  During the PDRY01
dry run, calibration blocks were written to two JSONL files because
LabRecorder was absent after the --start-phase 4 resume.  This script
bridges that gap by re-implementing the XDF-loading step against JSONL.

What it does
------------
1. Parses the two JSONL files into pyxdf-compatible stream dicts so the
   shared preprocessing helpers (_merge_eeg_streams, _parse_markers, etc.)
   can be used without modification.
2. Applies the same EEG preprocessing + windowing + feature extraction
   pipeline as the live calibration path.
3. Loads the existing rest baseline XDF for normalisation stats.
4. Runs the frozen scratch model: SelectKBest(k=35) + StandardScaler
   + SVC(linear, C=1.0) — identical to calibrate_participant.py.
5. Computes LORO CV threshold (2 JSONL files → 2 folds).
6. Saves deployment artefacts to OUT_DIR.

Usage
-----
    python scripts/_tmp_calibrate_from_jsonl.py

No arguments — paths are resolved from config/paths.yaml.
Edit JSONL_CAL_FILES / REST_XDF / OUT_DIR below if your layout differs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Repo root + sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from build_mwl_training_dataset import (  # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _extract_all_blocks,
    _find_stream,
    _load_eeg_metadata,
    _merge_eeg_streams,
    _parse_markers,
)
import calibrate_participant as _cal  # noqa: E402
from eeg import EegPreprocessor, extract_windows, slice_block  # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402
from ml.dataset import LABEL_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen model constants — must match calibrate_participant.py
# ---------------------------------------------------------------------------
_CAL_K: int = 35
_CAL_C: float = 1.0
_ANALYSIS_SRATE: float = 128.0
_LORO_MIN_J: float = 0.10
SEED: int = 42

# ---------------------------------------------------------------------------
# Paths — resolved from config/paths.yaml then hardened below
# ---------------------------------------------------------------------------
_paths_cfg = yaml.safe_load(
    (_REPO_ROOT / "config" / "paths.yaml").read_text(encoding="utf-8")
)
_DATA_ROOT = Path(_paths_cfg["data_root"])

JSONL_CAL_FILES: list[Path] = [
    _DATA_ROOT / "physiology/PDRY01/S001/lsl_recording_20260422T160202.jsonl",
    _DATA_ROOT / "physiology/PDRY01/S001/lsl_recording_20260422T161256.jsonl",
]
REST_XDF: Path = (
    _DATA_ROOT
    / "physiology/sub-PDRY01/ses-S001/physio"
    / "sub-PDRY01_ses-S001_task-matb_acq-rest_physio.xdf"
)
OUT_DIR: Path = _DATA_ROOT / "models/PDRY01/S001"
_REGION_CFG: Path = _REPO_ROOT / "config" / "eeg_feature_extraction.yaml"


# ---------------------------------------------------------------------------
# JSONL → pyxdf-style stream dicts
# ---------------------------------------------------------------------------

def _load_jsonl_streams(jsonl_path: Path) -> tuple[list[dict], float | None]:
    """Parse one JSONL recording into a list of pyxdf-compatible stream dicts.

    The returned dicts mirror the structure that pyxdf.load_xdf() produces,
    so the shared helpers (_merge_eeg_streams, _find_stream, _parse_markers)
    work unmodified:
      - stream["info"]["type"][0]   — e.g. "EEG" or "Markers"
      - stream["time_series"]       — (n_samples, n_ch) float32 for EEG;
                                       list-of-lists for marker streams
      - stream["time_stamps"]       — (n_samples,) float64

    Also returns the header's ``lsl_start_time`` (Python local_clock() at
    recording start) so that callers can compute an EEG→local-clock offset.
    The eego amp and OpenMATB use different clock origins; normalising the EEG
    timestamps against this value puts them in the same domain as the OpenMATB
    marker timestamps.
    """
    stream_meta: dict[str, dict] = {}
    stream_samples: dict[str, list] = {}
    stream_ts: dict[str, list] = {}
    lsl_start_time: float | None = None

    print(f"  Parsing {jsonl_path.name} ...", flush=True)
    with open(jsonl_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            if rtype == "header":
                lsl_start_time = rec.get("lsl_start_time")
            elif rtype == "stream_info":
                key = rec["stream_key"]
                stream_meta[key] = rec["info"]
                stream_samples[key] = []
                stream_ts[key] = []
            elif rtype == "sample":
                key = rec.get("stream_key", "")
                if key in stream_samples:
                    stream_samples[key].append(rec["sample"])
                    stream_ts[key].append(rec["timestamp"])

    streams: list[dict] = []
    for key, info in stream_meta.items():
        ts_arr = np.array(stream_ts[key], dtype=np.float64)
        raw_samples = stream_samples[key]

        # Irregular (srate == 0.0) → marker / event stream; keep as list-of-lists
        # so _parse_markers can index sample[0] for the string value.
        is_irregular = float(info.get("nominal_srate", 0.0)) == 0.0
        if is_irregular:
            data_arr = raw_samples  # list of [marker_string]
        else:
            n_ch = int(info.get("channel_count", 1))
            if raw_samples:
                data_arr = np.array(raw_samples, dtype=np.float32)  # (n, n_ch)
            else:
                data_arr = np.empty((0, n_ch), dtype=np.float32)

        # Wrap scalar info values in lists to match pyxdf convention.
        stream_info = {
            "name": [str(info.get("name", key))],
            "type": [str(info.get("type", ""))],
            "source_id": [str(info.get("source_id", ""))],
            "channel_count": [str(info.get("channel_count", 1))],
            "nominal_srate": [str(info.get("nominal_srate", 0.0))],
        }

        # The JSONL writer only stores basic stream metadata — the full LSL
        # channel descriptor XML is not saved.  For ANT eego streams the layout
        # is always: 64 electrode ("ref") channels + 1 TRG + 1 CNT = 66 total.
        # Without a descriptor, _merge_eeg_streams falls back to taking all 66
        # channels per amp, yielding 132 instead of the expected 128.
        # Inject a synthetic descriptor so the ref-channel filter works.
        if str(info.get("type", "")) == "EEG" and int(info.get("channel_count", 0)) == 66:
            stream_info["desc"] = [{
                "channels": [{
                    "channel": (
                        [{"type": ["ref"]} for _ in range(64)]
                        + [{"type": ["TRG"]}, {"type": ["CNT"]}]
                    )
                }]
            }]

        streams.append({
            "info": stream_info,
            "time_series": data_arr,
            "time_stamps": ts_arr,
        })

        n_samp = len(ts_arr)
        print(f"    {info.get('name', key)} ({info.get('type', '?')}): {n_samp} samples")

    return streams, lsl_start_time


# ---------------------------------------------------------------------------
# JSONL → (epochs, level) pairs   [mirrors _load_xdf_block]
# ---------------------------------------------------------------------------

def _load_jsonl_block(
    jsonl_path: Path,
    expected_channels: list[str],
) -> list[tuple[np.ndarray, str]] | None:
    """Load and preprocess one calibration JSONL, return (epochs, level) pairs.

    Mirrors calibrate_participant._load_xdf_block but reads JSONL instead
    of XDF.  Returns None on any failure so the caller can skip gracefully.
    """
    streams, lsl_start_time = _load_jsonl_streams(jsonl_path)

    eeg_stream = _merge_eeg_streams(streams)

    # Prefer the OpenMATB stream over eego hardware trigger streams — there are
    # typically three "Markers"-type LSL streams in a JSONL recording:
    # two eego TRG streams (hardware triggers) and one OpenMATB event stream.
    # _find_stream returns the first match, which is never OpenMATB.
    marker_stream = next(
        (s for s in streams
         if s["info"]["type"][0] == "Markers"
         and "openmatb" in s["info"]["name"][0].lower()),
        _find_stream(streams, "Markers"),
    )

    if eeg_stream is None:
        print(f"  SKIPPED — no EEG stream in {jsonl_path.name}")
        return None

    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(expected_channels):
        print(f"  SKIPPED — channel count {n_ch} != {len(expected_channels)}")
        return None

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T  # (n_ch, n_samp)
    eeg_ts = np.array(eeg_stream["time_stamps"], dtype=np.float64)

    if len(eeg_ts) < 2:
        print(f"  SKIPPED — too few EEG samples ({len(eeg_ts)})")
        return None

    # Decimate to analysis srate when amp ran at a higher rate (e.g. 500 → 128 Hz).
    actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
    if actual_srate > _ANALYSIS_SRATE * 1.1:
        factor = int(round(actual_srate / _ANALYSIS_SRATE))
        eeg_data = eeg_data[:, ::factor]
        eeg_ts = eeg_ts[::factor]
        print(f"  Decimated {actual_srate:.0f} → {_ANALYSIS_SRATE:.0f} Hz (factor={factor})")

    # Clock-domain correction -----------------------------------------------
    # The eego amplifier and OpenMATB use different clock origins; pyxdf
    # normalises these in XDF recordings but we only have raw timestamps here.
    # Strategy: the JSONL header records lsl_start_time = pylsl.local_clock()
    # at the moment recording began.  The first EEG sample in the JSONL was
    # pulled from the inlet at approximately that same moment (small buffer lag
    # aside).  Shifting EEG timestamps by (lsl_start_time - eeg_ts[0]) maps
    # them into the same local_clock() domain that OpenMATB markers use.
    if lsl_start_time is not None:
        eeg_clock_offset = lsl_start_time - eeg_ts[0]
        eeg_ts = eeg_ts + eeg_clock_offset
        print(f"  Clock correction applied: {eeg_clock_offset:+.3f} s  "
              f"(EEG ts range now [{eeg_ts[0]:.3f}, {eeg_ts[-1]:.3f}])")
    else:
        print("  WARNING: no lsl_start_time in JSONL header — timestamps may be misaligned")

    # Preprocess (bandpass + notch + CAR — same config as live path).
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    # Extract labelled calibration blocks from the marker stream.
    markers = _parse_markers(marker_stream)
    block_specs = _extract_all_blocks(markers)  # list of (start_ts, end_ts, level)

    if not block_specs:
        print(f"  SKIPPED — no START/END markers found in {jsonl_path.name}")
        return None

    results: list[tuple[np.ndarray, str]] = []
    for start_ts, end_ts, level in block_specs:
        start_idx = int(np.searchsorted(eeg_ts, start_ts))
        end_idx = int(np.searchsorted(eeg_ts, end_ts))
        block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
        epochs = extract_windows(block, WINDOW_CONFIG)
        if epochs.shape[0] > 0:
            results.append((epochs, level))

    if not results:
        print(f"  SKIPPED — no windows extracted")
        return None

    summary = ", ".join(f"{lvl}:{e.shape[0]}" for e, lvl in results)
    print(f"  OK  {len(results)} blocks [{summary}]")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Validate inputs exist before doing any heavy work
    # ------------------------------------------------------------------
    missing = [p for p in JSONL_CAL_FILES + [REST_XDF] if not p.exists()]
    if missing:
        print("ERROR: Input file(s) not found:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    region_map = _build_region_map(_REGION_CFG, expected_channels)

    # ------------------------------------------------------------------
    # 1. Load calibration blocks from each JSONL file
    # ------------------------------------------------------------------
    print("\n=== Loading calibration blocks ===")
    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    jsonl_epoch_groups: list[list[np.ndarray]] = []
    jsonl_label_groups: list[list[np.ndarray]] = []

    for jsonl_path in JSONL_CAL_FILES:
        print(f"\n[{jsonl_path.name}]")
        results = _load_jsonl_block(jsonl_path, expected_channels)
        if results is None:
            continue
        eg: list[np.ndarray] = []
        lg: list[np.ndarray] = []
        for epochs, level in results:
            label = LABEL_MAP[level]
            all_epochs.append(epochs)
            all_labels.append(np.full(epochs.shape[0], label, dtype=np.int64))
            eg.append(epochs)
            lg.append(np.full(epochs.shape[0], label, dtype=np.int64))
        jsonl_epoch_groups.append(eg)
        jsonl_label_groups.append(lg)

    if not all_epochs:
        sys.exit("ERROR: No valid calibration blocks loaded from any JSONL file.")

    cal_epochs = np.concatenate(all_epochs)
    cal_y = np.concatenate(all_labels)
    print(
        f"\nTotal windows: {len(cal_y)}"
        f"  (LOW={int((cal_y == 0).sum())}, "
        f"MOD={int((cal_y == 1).sum())}, "
        f"HIGH={int((cal_y == 2).sum())})"
    )

    # ------------------------------------------------------------------
    # 2. Extract features
    # ------------------------------------------------------------------
    print("\n=== Extracting features ===")
    cal_X, _feat_names = _extract_feat(cal_epochs, _ANALYSIS_SRATE, region_map)
    print(f"  Feature matrix: {cal_X.shape}")

    # ------------------------------------------------------------------
    # 3. Resting-baseline normalisation stats
    # ------------------------------------------------------------------
    print(f"\n=== Resting baseline ===")
    print(f"  {REST_XDF.name}")
    rest_epochs = _cal._load_rest_xdf_block(REST_XDF, expected_channels)
    if rest_epochs is not None:
        norm_mean, norm_std = _cal._compute_resting_norm(
            rest_epochs, _ANALYSIS_SRATE, region_map
        )
        print("  Using rest XDF for normalisation.")
    else:
        print("  WARNING: Rest XDF failed — falling back to LOW-block proxy.")
        low_mask = cal_y == LABEL_MAP["LOW"]
        if not low_mask.any():
            sys.exit("ERROR: No LOW windows and REST XDF failed — cannot normalise.")
        low_X, _ = _extract_feat(cal_epochs[low_mask], _ANALYSIS_SRATE, region_map)
        norm_mean = low_X.mean(axis=0)
        norm_std = low_X.std(axis=0)
        norm_std[norm_std < 1e-12] = 1.0

    # ------------------------------------------------------------------
    # 4. Normalise, select features, train
    # ------------------------------------------------------------------
    print("\n=== Training ===")
    cal_X_norm = (cal_X - norm_mean) / norm_std

    # Build per-JSONL arrays for LORO CV.
    window_counts = [sum(len(lv) for lv in lg) for lg in jsonl_label_groups]
    split_pts = np.cumsum(window_counts[:-1]).tolist() if len(window_counts) > 1 else []
    jsonl_X_norm_loro: list[np.ndarray] = list(np.split(cal_X_norm, split_pts))
    jsonl_y_loro: list[np.ndarray] = [
        np.concatenate(lg) for lg in jsonl_label_groups
    ]

    k = min(_CAL_K, cal_X_norm.shape[1])
    own_selector = SelectKBest(f_classif, k=k)
    cal_X_sel = own_selector.fit_transform(cal_X_norm, cal_y)
    own_sc = StandardScaler()
    cal_X_sc = own_sc.fit_transform(cal_X_sel)
    clf = SVC(
        kernel="linear",
        C=_CAL_C,
        class_weight="balanced",
        probability=True,
        random_state=SEED,
    )
    clf.fit(cal_X_sc, cal_y)
    deploy_pipe = Pipeline([("sc", own_sc), ("clf", clf)])
    print(f"  SelectKBest(k={k}) + StandardScaler + SVC(linear, C={_CAL_C}) — done.")

    # ------------------------------------------------------------------
    # 5. Youden's J threshold
    # ------------------------------------------------------------------
    p_high = clf.predict_proba(cal_X_sc)[:, -1]
    y_binary = (cal_y == LABEL_MAP["HIGH"]).astype(int)
    if len(np.unique(y_binary)) < 2:
        sys.exit(
            "ERROR: Need both HIGH and non-HIGH windows to compute threshold.\n"
            "Check that all three levels were successfully loaded."
        )
    fpr, tpr, thr = roc_curve(y_binary, p_high)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    train_youden_thr = float(thr[best_idx])
    train_youdens_j = float(j_scores[best_idx])

    # LORO CV threshold (2 JSONL files = 2 folds).
    loro_thr: float | None = None
    loro_j: float | None = None
    loro_fold_j: list[float] = []
    threshold_method = "train_set"

    if len(jsonl_X_norm_loro) >= 2:
        print("\n  Running LORO CV threshold ...")
        loro_thr, loro_j, loro_fold_j = _cal._compute_loro_threshold(
            jsonl_X_norm_loro, jsonl_y_loro, k, _CAL_C, SEED
        )
        if loro_thr is not None and loro_j is not None and loro_j >= _LORO_MIN_J:
            youden_thr = loro_thr
            youdens_j = loro_j
            threshold_method = "loro"
        else:
            youden_thr = train_youden_thr
            youdens_j = train_youdens_j
            threshold_method = "train_set_fallback"
            reason = f"LORO J={loro_j:.4f}" if loro_j is not None else "LORO returned null"
            print(
                f"  WARNING: LORO threshold rejected ({reason} < min {_LORO_MIN_J:.2f})"
                f" — using train-set threshold."
            )
    else:
        youden_thr = train_youden_thr
        youdens_j = train_youdens_j

    # ------------------------------------------------------------------
    # 6. Save deployment artefacts
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy_pipe, OUT_DIR / "pipeline.pkl")
    joblib.dump(own_selector, OUT_DIR / "selector.pkl")

    norm_stats = {
        "mean": norm_mean.tolist(),
        "std": norm_std.tolist(),
        "n_classes": int(len(np.unique(cal_y))),
    }
    (OUT_DIR / "norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2), encoding="utf-8"
    )

    model_config = {
        "youden_threshold": round(youden_thr, 6),
        "youdens_j": round(youdens_j, 6),
        "threshold_method": threshold_method,
        "train_youden_threshold": round(train_youden_thr, 6),
        "train_youdens_j": round(train_youdens_j, 6),
        "loro_youden_threshold": round(loro_thr, 6) if loro_thr is not None else None,
        "loro_youdens_j": round(loro_j, 6) if loro_j is not None else None,
        "loro_n_folds": len(loro_fold_j) if loro_fold_j else None,
        "loro_fold_j_scores": (
            [round(v, 6) for v in loro_fold_j] if loro_fold_j else None
        ),
        "n_classes": int(len(np.unique(cal_y))),
        "model_k": k,
        "source": "jsonl_offline",
        "jsonl_files": [p.name for p in JSONL_CAL_FILES],
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )

    cal_preds = clf.predict(cal_X_sc)
    cal_acc = float(np.mean(cal_preds == cal_y))

    print(f"\n=== Results ===")
    print(f"  Calibration accuracy: {cal_acc:.1%}")
    print(f"  Train-set threshold:  {train_youden_thr:.4f}  (J={train_youdens_j:.4f})")
    if loro_thr is not None:
        print(
            f"  LORO threshold:       {loro_thr:.4f}  (J={loro_j:.4f}, "
            f"folds={len(loro_fold_j)}, fold_J={[round(v, 3) for v in loro_fold_j]})"
        )
    print(f"  Deployed threshold:   {youden_thr:.4f}  [method={threshold_method}]")
    print(f"\n  Saved to {OUT_DIR}/")
    print(f"    pipeline.pkl  selector.pkl  norm_stats.json  model_config.json")

    # ------------------------------------------------------------------
    # 7. Offline P(HIGH) analysis
    # ------------------------------------------------------------------
    _offline_phigh_analysis(
        p_high=p_high,
        cal_y=cal_y,
        jsonl_label_groups=jsonl_label_groups,
        youden_thr=youden_thr,
        threshold_method=threshold_method,
        out_dir=OUT_DIR,
    )


def _offline_phigh_analysis(
    p_high: np.ndarray,
    cal_y: np.ndarray,
    jsonl_label_groups: list[list[np.ndarray]],
    youden_thr: float,
    threshold_method: str,
    out_dir: Path,
) -> None:
    """Print P(HIGH) stats per level and save a two-panel figure.

    Panel 1 — time series: P(HIGH) per window in recording order, coloured
    by true level, with threshold line and block-boundary markers.  A 5-window
    rolling mean approximates the smoothed signal the live estimator would see.
    Blocks where the rolling mean stays above threshold are labelled as
    "adaptation ON" (reduce difficulty); blocks below are "adaptation OFF".

    Panel 2 — distribution: violin plot of P(HIGH) per difficulty level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

    INV_LABEL = {v: k for k, v in LABEL_MAP.items()}
    COLORS = {"LOW": "#2196F3", "MODERATE": "#FF9800", "HIGH": "#E53935"}

    # --- Reconstruct per-block layout in recording order ---
    block_records: list[tuple[str, int, int]] = []  # (level, start_win, end_win)
    offset = 0
    for lg in jsonl_label_groups:
        for labels in lg:
            n = len(labels)
            block_records.append((INV_LABEL[int(labels[0])], offset, offset + n))
            offset += n

    n_wins = len(p_high)
    win_idx = np.arange(n_wins)

    # 5-window rolling mean (~1.25 s at step=0.25 s) to mimic live smoothing
    _K = 5
    kernel = np.ones(_K) / _K
    p_smooth = np.convolve(p_high, kernel, mode="same")

    # --- Console summary table ---
    print("\n=== Offline P(HIGH) by difficulty level ===")
    print(f"  Threshold: {youden_thr:.4f}  [{threshold_method}]")
    print(f"  {'Level':<10}  {'N':>5}  {'Mean':>6}  {'Med':>6}  {'Std':>6}  {'%>thr':>7}  {'Adapt?':>7}")
    print(f"  {'-'*58}")
    for level in ["LOW", "MODERATE", "HIGH"]:
        mask = cal_y == LABEL_MAP[level]
        ph = p_high[mask]
        pct = 100.0 * (ph > youden_thr).mean()
        adapt = "ON" if pct >= 50 else "off"
        print(
            f"  {level:<10}  {len(ph):>5}  {ph.mean():>6.3f}  "
            f"{float(np.median(ph)):>6.3f}  {ph.std():>6.3f}  {pct:>6.1f}%  {adapt:>7}"
        )

    # Per-block "adaptation state" (majority vote: >50% windows above threshold)
    print(f"\n=== Simulated adaptation per block (threshold={youden_thr:.4f}) ===")
    print(f"  {'#':>3}  {'Level':<10}  {'%>thr':>7}  {'State':>8}  {'Correct?':>9}")
    print(f"  {'-'*48}")
    for i, (lvl, s, e) in enumerate(block_records):
        ph_blk = p_high[s:e]
        pct = 100.0 * (ph_blk > youden_thr).mean()
        state = "ADAPT ON" if pct >= 50 else "adapt off"
        correct = (lvl == "HIGH" and state == "ADAPT ON") or (lvl != "HIGH" and state == "adapt off")
        symbol = "✓" if correct else "✗"
        print(f"  {i+1:>3}  {lvl:<10}  {pct:>6.1f}%  {state:>9}  {symbol:>9}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    fig.suptitle(
        f"PDRY01 — Offline P(HIGH) on calibration data\n"
        f"Threshold = {youden_thr:.4f} [{threshold_method}]   "
        f"3-class SVC   {n_wins} windows",
        fontsize=11,
    )

    # Panel 1: scatter coloured by true level + rolling mean + threshold
    for level, color in COLORS.items():
        mask = cal_y == LABEL_MAP[level]
        ax1.scatter(win_idx[mask], p_high[mask], c=color, s=5, alpha=0.45,
                    label=level, zorder=2)
    ax1.plot(win_idx, p_smooth, color="black", lw=1.3, alpha=0.75,
             label=f"{_K}-win rolling mean", zorder=3)
    ax1.axhline(youden_thr, color="red", lw=1.8, ls="--",
                label=f"Threshold = {youden_thr:.3f}", zorder=4)

    # Block boundaries + level labels
    for i, (lvl, s, e) in enumerate(block_records):
        if i > 0:
            ax1.axvline(s, color="gray", lw=0.7, ls=":", zorder=1)
        mid = (s + e) / 2
        # Shaded band per block
        ax1.axvspan(s, e, alpha=0.06, color=COLORS[lvl], zorder=0)
        # Level initial at top
        ax1.text(mid, 1.03, lvl[0], ha="center", va="bottom", fontsize=8,
                 color=COLORS[lvl], transform=ax1.get_xaxis_transform(),
                 fontweight="bold")

    # Mark JSONL file boundaries
    file_lens = [sum(len(lv) for lv in lg) for lg in jsonl_label_groups]
    cumlen = 0
    for fi, fl in enumerate(file_lens[:-1]):
        cumlen += fl
        ax1.axvline(cumlen, color="#333333", lw=1.8, ls="--", alpha=0.5, zorder=4)
        ax1.text(cumlen + 2, 0.96, f"JSONL {fi+2}", fontsize=8, color="#333333", va="top")

    ax1.set_ylim(-0.03, 1.08)
    ax1.set_xlim(0, n_wins - 1)
    ax1.set_ylabel("P(HIGH)", fontsize=10)
    ax1.set_xlabel("")
    ax1.legend(loc="upper right", fontsize=8, markerscale=3)
    ax1.set_title(
        "P(HIGH) per window in recording order  "
        "(L/M/H shading = block level,  red dashed = adaptation threshold)",
        fontsize=9,
    )
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax1.grid(axis="y", lw=0.4, alpha=0.4)

    # Panel 2: per-block fraction above threshold (bar chart = simulated trigger rate)
    x_blk = np.arange(len(block_records))
    bar_cols = [COLORS[lvl] for lvl, _, _ in block_records]
    pcts = [
        (p_high[s:e] > youden_thr).mean()
        for _, s, e in block_records
    ]
    bars = ax2.bar(x_blk, pcts, color=bar_cols, width=0.7, alpha=0.85, edgecolor="white")
    ax2.axhline(0.5, color="red", lw=1.5, ls="--", label="50% majority vote")
    # Add % label inside each bar
    for bar, pct in zip(bars, pcts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            min(pct + 0.03, 0.95),
            f"{pct:.0%}",
            ha="center", va="bottom", fontsize=7,
        )

    ax2.set_xticks(x_blk)
    ax2.set_xticklabels(
        [f"{lvl[:3]}\nB{i+1}" for i, (lvl, _, _) in enumerate(block_records)],
        fontsize=8,
    )
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Fraction\nP(HIGH) > thr", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title(
        "Simulated adaptation trigger rate per block  "
        "(>50% → ADAPT ON = reduce difficulty)",
        fontsize=9,
    )
    # Legend patches for level colours
    legend_patches = [
        mpatches.Patch(color=c, label=l) for l, c in COLORS.items()
    ]
    ax2.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="red", ls="--", lw=1.5, label="50% threshold")
    ], fontsize=8, loc="upper right")

    plt.tight_layout()
    fig_path = out_dir / "phigh_calibration_analysis.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {fig_path}")


if __name__ == "__main__":
    main()
