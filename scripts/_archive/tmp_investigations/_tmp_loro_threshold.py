"""LORO (Leave-One-Run-Out) threshold analysis for PSELF S005 block01 model.

Changes from v1:
  - Include recovered block_01 data (END marker known, START inferred at END-59s)
  - MODERATE blocks dropped — binary LOW vs HIGH only
  - This mirrors a cleaner training regime and tests whether LORO generalises

Asks: if we had used 2-fold LORO CV to set the Youden J threshold instead of
fitting it on the training data, how would the threshold and adaptation
behaviour have differed?

The two "runs" are the calibration recordings C1 and C2 (each ~9 min, each
containing all three MWL levels in counterbalanced order).

LORO procedure
--------------
  Fold A: train on C1 features, evaluate threshold on C2 predictions
  Fold B: train on C2 features, evaluate threshold on C1 predictions
  LORO threshold = mean of {threshold_A, threshold_B}
  Final deployed model = full-data retrain (C1+C2) with the LORO threshold

Comparison
----------
  Threshold A  : original calibrate_participant.py (training-set Youden J)
  Threshold B  : LORO threshold (held-out Youden J, averaged across folds)

Both are then applied to the adaptation XDF via the EMA + hold + cooldown
state machine to show how much the assist-on% would have changed.

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_loro_threshold.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pyxdf
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import yaml
from build_mwl_training_dataset import (
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _merge_eeg_streams,
)
from eeg import EegPreprocessor, extract_windows, slice_block
from eeg.online_features import OnlineFeatureExtractor
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Config — must match calibrate_participant.py exactly
# ---------------------------------------------------------------------------
SRATE            = 128.0
CAL_K            = 40
CAL_C            = 1.0
SEED             = 42
BLOCK_DURATION_S = 59  # scenario: block_01 END fires at t=59s, START missing

MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PSELF")
PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")
ADAPT_XDF = PHYSIO / "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]
FEAT_CFG = _REPO / "config" / "eeg_feature_extraction.yaml"

# Norm stats from the original calibration (rest-XDF baseline)
_ns       = json.loads((MODEL_DIR / "norm_stats.json").read_text())
NORM_MEAN = np.array(_ns["mean"])
NORM_STD  = np.array(_ns["std"])
NORM_STD[NORM_STD < 1e-12] = 1.0

# Deployed threshold (training-set Youden J)
_mc = json.loads((MODEL_DIR / "model_config.json").read_text())
DEPLOYED_THRESHOLD = float(_mc["youden_threshold"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_RE     = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")
BLOCK_IDX_RE = re.compile(r"block_(?P<idx>\d+)/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")


def _load_eeg(xdf_path: Path):
    """Load XDF → (eeg_data [n_ch, n], eeg_ts), decimated to SRATE."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        raise RuntimeError(f"No EEG in {xdf_path.name}")
    data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    ts   = np.array(eeg_stream["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            factor = int(round(actual / SRATE))
            data, ts = data[:, ::factor], ts[::factor]
    return data, ts


def _preprocess(eeg_data: np.ndarray) -> np.ndarray:
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(eeg_data.shape[0])
    return pp.process(eeg_data)


def _extract_features(epochs: np.ndarray) -> np.ndarray:
    """(n_win, n_ch, win_s) → (n_win, n_feats)"""
    extractor = OnlineFeatureExtractor(CH_NAMES, srate=SRATE, region_cfg=FEAT_CFG)
    return np.array([extractor.compute(w) for w in epochs])


def _get_markers(xdf_path: Path) -> list[tuple[float, str]]:
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    m = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    if m is None:
        return []
    return list(zip(m["time_stamps"], [s[0] for s in m["time_series"]]))


def _parse_blocks(markers: list) -> list[tuple[float, float, str]]:
    """Return (t_start, t_end, level) for every complete block (both START+END present)."""
    opens: dict[str, float] = {}
    blocks = []
    for ts, ev in markers:
        hit = BLOCK_RE.search(ev)
        if not hit:
            continue
        lv = hit.group("level")
        if hit.group("ev") == "START":
            opens[lv] = ts
        elif hit.group("ev") == "END" and lv in opens:
            blocks.append((opens.pop(lv), ts, lv))
    return blocks


def _recover_block01(markers: list, eeg_ts: np.ndarray) -> tuple[float, float, str] | None:
    """Infer block_01 time range from its END marker when START was not recorded.

    LabRecorder sometimes subscribes after the MATB LSL outlet emits block_01/START.
    The END marker is always captured.  We back-calculate START as END - BLOCK_DURATION_S
    and clamp to the first available EEG sample.

    Returns (t_start_clamped, t_end, level) or None if END not found.
    """
    for ts, ev in markers:
        hit = BLOCK_IDX_RE.search(ev)
        if hit and int(hit.group("idx")) == 1 and hit.group("ev") == "END":
            level   = hit.group("level")
            t_end   = ts
            t_start = max(eeg_ts[0], t_end - BLOCK_DURATION_S)
            print(f"    block_01 RECOVERED  {level:<10}  "
                  f"start={t_start:.1f}  end={t_end:.1f}  "
                  f"({t_end - t_start:.0f}s available)")
            return t_start, t_end, level
    return None


def _windows_in_range(preprocessed, eeg_ts, t_start, t_end):
    i0 = int(np.searchsorted(eeg_ts, t_start))
    i1 = int(np.searchsorted(eeg_ts, t_end))
    block = slice_block(preprocessed, i0, i1, WINDOW_CONFIG)
    return extract_windows(block, WINDOW_CONFIG)


def _fit_model(X_norm: np.ndarray, y: np.ndarray):
    """Fit SelectKBest + StandardScaler + SVC on already-normalised features.

    Returns (selector, pipeline, X_sel_scaled, clf) for threshold computation.
    """
    k   = min(CAL_K, X_norm.shape[1])
    sel = SelectKBest(f_classif, k=k)
    X_sel = sel.fit_transform(X_norm, y)
    sc  = StandardScaler()
    X_sc = sc.fit_transform(X_sel)
    clf = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
              probability=True, random_state=SEED)
    clf.fit(X_sc, y)
    pipe = Pipeline([("sc", sc), ("clf", clf)])
    return sel, pipe, X_sc, clf


def _youden_threshold(clf, sel, X_norm_test: np.ndarray, y_test: np.ndarray, pipe):
    """Compute Youden J threshold on held-out data.

    Applies sel.transform then pipeline (StandardScaler + SVC) to X_norm_test.
    y_test must be binary (0=LOW, 1=HIGH).
    """
    X_sel  = sel.transform(X_norm_test)
    X_sc   = pipe.named_steps["sc"].transform(X_sel)
    p_high = pipe.named_steps["clf"].predict_proba(X_sc)[:, -1]
    fpr, tpr, thr_arr = roc_curve(y_test, p_high)
    j      = tpr - fpr
    best_i = int(np.argmax(j))
    return float(thr_arr[best_i]), float(j[best_i]), p_high


def _simulate_scheduler(
    ph_vals: np.ndarray,
    threshold: float,
    alpha: float   = 0.05,
    hysteresis: float = 0.02,
    t_hold_s: float   = 3.0,
    cooldown_s: float = 15.0,
    step_s: float     = 0.25,
) -> tuple[np.ndarray, int, int]:
    """EMA + hold-timer + cooldown state machine. Returns (on_flags, n_on, n_off)."""
    ema = None
    zone = None
    zone_entry = 0.0
    assist_on = False
    cooldown_end = 0.0
    on_flags = []
    n_on = n_off = 0
    for i, v in enumerate(ph_vals):
        t = i * step_s
        ema = float(v) if ema is None else alpha * float(v) + (1.0 - alpha) * ema
        new_zone = (
            "above" if ema > threshold + hysteresis else
            "below" if ema < threshold - hysteresis else
            "dead"
        )
        if new_zone != zone:
            zone = new_zone
            zone_entry = t
        hold_s = t - zone_entry
        if t >= cooldown_end:
            if zone == "above" and hold_s >= t_hold_s and not assist_on:
                assist_on = True
                cooldown_end = t + cooldown_s
                zone_entry = t
                n_on += 1
            elif zone == "below" and hold_s >= t_hold_s and assist_on:
                assist_on = False
                cooldown_end = t + cooldown_s
                zone_entry = t
                n_off += 1
        on_flags.append(assist_on)
    return np.array(on_flags, dtype=bool), n_on, n_off


def _phigh_from_xdf(xdf_path: Path, selector, pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Return (lsl_timestamps, p_high) for every sliding window in an XDF."""
    data, ts = _load_eeg(xdf_path)
    pp       = _preprocess(data)
    extractor = OnlineFeatureExtractor(CH_NAMES, srate=SRATE, region_cfg=FEAT_CFG)
    step = int(WINDOW_CONFIG.step_s * SRATE)
    win  = int(WINDOW_CONFIG.window_s * SRATE)
    n    = pp.shape[1]
    ts_list, ph_list = [], []
    for start in range(win, n - step, step):
        window    = pp[:, start - win : start]
        feats     = extractor.compute(window)
        feats_z   = (feats - NORM_MEAN) / NORM_STD
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        ts_list.append(ts[start])
        ph_list.append(float(proba[-1]))
    return np.array(ts_list), np.array(ph_list)


# ---------------------------------------------------------------------------
# Load both calibration runs
# ---------------------------------------------------------------------------

CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf",
]

print("Loading calibration XDFs and extracting features...")
print("(block_01 recovered from END marker; MODERATE blocks dropped)")
print()

run_X: list[np.ndarray] = []  # normalised feature arrays per run (binary LOW/HIGH)
run_y: list[np.ndarray] = []  # binary label arrays: 0=LOW, 1=HIGH
run_labels = ["C1", "C2"]

for xdf_path in CAL_XDFS:
    acq = xdf_path.stem.split("acq-")[1].replace("_physio", "")
    print(f"  {acq} ...", flush=True)
    data, ts = _load_eeg(xdf_path)
    pp       = _preprocess(data)
    markers  = _get_markers(xdf_path)

    # Recorded blocks (have both START+END) + recovered block_01
    blocks = _parse_blocks(markers)
    b01    = _recover_block01(markers, ts)
    if b01 is not None:
        blocks.append(b01)

    feats_list, label_list = [], []
    for t_s, t_e, level in blocks:
        # Drop MODERATE
        if level == "MODERATE":
            continue
        wins = _windows_in_range(pp, ts, t_s, t_e)
        if wins.shape[0] == 0:
            continue
        feats = _extract_features(wins)
        # Binary: LOW=0, HIGH=1
        lbl = 1 if level == "HIGH" else 0
        feats_list.append(feats)
        label_list.append(np.full(len(feats), lbl, dtype=np.int64))
        print(f"    {level:<10}  n={len(feats)}")

    if not feats_list:
        raise RuntimeError(f"No LOW/HIGH blocks loaded from {xdf_path.name}")

    X_raw  = np.concatenate(feats_list)
    X_norm = (X_raw - NORM_MEAN) / NORM_STD
    y_bin  = np.concatenate(label_list)
    run_X.append(X_norm)
    run_y.append(y_bin)
    print(f"  {acq}  total={len(y_bin)}  LOW={int((y_bin==0).sum())}  HIGH={int((y_bin==1).sum())}")
    print()

# Combined dataset (all cal data)
X_all = np.concatenate(run_X)
y_all = np.concatenate(run_y)

# ---------------------------------------------------------------------------
# Original deployed model: training-set Youden J
# ---------------------------------------------------------------------------

print("Fitting full-data model (training-set threshold)...")
sel_full, pipe_full, X_sc_full, clf_full = _fit_model(X_all, y_all)

# Training-set threshold — already binary so y_all IS y_binary
p_high_train = clf_full.predict_proba(X_sc_full)[:, -1]
fpr_t, tpr_t, thr_t = roc_curve(y_all, p_high_train)
j_t    = tpr_t - fpr_t
best_t = int(np.argmax(j_t))
train_thr = float(thr_t[best_t])
train_j   = float(j_t[best_t])

print(f"  Training-set threshold = {train_thr:.4f}  (J={train_j:.4f})")
print(f"  Deployed threshold     = {DEPLOYED_THRESHOLD:.4f}  "
      f"({'matches' if abs(train_thr - DEPLOYED_THRESHOLD) < 0.05 else 'differs — deployed used 3-class+MODERATE'})")
print()

# ---------------------------------------------------------------------------
# LORO CV: 2 folds (train on one run, evaluate threshold on the other)
# ---------------------------------------------------------------------------

print("Running LORO CV (2 folds)...")
print()

fold_thresholds = []
fold_j_scores   = []

for fold, (train_idx, test_idx) in enumerate([(0, 1), (1, 0)]):
    train_label = run_labels[train_idx]
    test_label  = run_labels[test_idx]
    X_tr = run_X[train_idx];  y_tr = run_y[train_idx]
    X_te = run_X[test_idx];   y_te = run_y[test_idx]

    # Check we have all three classes in both train and test
    for split_name, y_split in [("train", y_tr), ("test", y_te)]:
        classes = np.unique(y_split)
        if len(classes) < 3:
            print(f"  WARNING: fold {fold+1} {split_name} has only {len(classes)} class(es): {classes}")

    sel_f, pipe_f, _, clf_f = _fit_model(X_tr, y_tr)
    thr_f, j_f, p_high_te   = _youden_threshold(clf_f, sel_f, X_te, y_te, pipe_f)
    fold_thresholds.append(thr_f)
    fold_j_scores.append(j_f)

    # Per-class p_high on held-out set (binary)
    class_stats = {}
    for lv, lbl in [("LOW", 0), ("HIGH", 1)]:
        mask = y_te == lbl
        if mask.any():
            ph = p_high_te[mask]
            class_stats[lv] = (ph.mean(), (ph > thr_f).mean() * 100)

    print(f"  Fold {fold+1}: train={train_label} ({len(y_tr)} win)  "
          f"test={test_label} ({len(y_te)} win)  "
          f"[LOW={int((y_te==0).sum())} HIGH={int((y_te==1).sum())}]")
    print(f"    Held-out Youden J threshold: {thr_f:.4f}  (J={j_f:.4f})")
    for lv in ["LOW", "HIGH"]:
        if lv in class_stats:
            m, pct = class_stats[lv]
            print(f"    {lv:<10}  mean_p_high={m:.3f}  %>threshold={pct:.1f}%")
    print()

loro_threshold = float(np.mean(fold_thresholds))
loro_j_mean    = float(np.mean(fold_j_scores))

print(f"LORO threshold (mean of 2 folds): {loro_threshold:.4f}  "
      f"(J range: {min(fold_j_scores):.4f}–{max(fold_j_scores):.4f})")
print(f"Training-set threshold           : {train_thr:.4f}")
print(f"Delta (LORO - train)             : {loro_threshold - train_thr:+.4f}")
print()

# ---------------------------------------------------------------------------
# Per-class calibration stats using the full-data model
# Evaluate at BOTH thresholds to show what differs
# ---------------------------------------------------------------------------

print("Full-data model: per-class p_high at both thresholds")
print(f"  {'Level':<10}  {'mean_p_high':>11}  {'LORO %':>8}  {'train %':>8}")
print("  " + "-" * 44)
for lv, lbl in [("LOW", 0), ("HIGH", 1)]:
    mask = y_all == lbl
    if mask.any():
        ph = p_high_train[mask]
        pct_loro  = (ph > loro_threshold).mean() * 100
        pct_train = (ph > train_thr).mean()      * 100
        print(f"  {lv:<10}  {ph.mean():11.3f}  {pct_loro:7.1f}%  {pct_train:7.1f}%")
print()

# ---------------------------------------------------------------------------
# Apply full-data model to adaptation XDF at both thresholds
# ---------------------------------------------------------------------------

print("Applying full-data model to adaptation XDF...")
print("(This processes the full ~8 min recording — may take ~2 min)")
print()

lsl_ts, ph_adapt = _phigh_from_xdf(ADAPT_XDF, sel_full, pipe_full)

on_train, n_on_train, n_off_train = _simulate_scheduler(ph_adapt, train_thr)
on_loro,  n_on_loro,  n_off_loro  = _simulate_scheduler(ph_adapt, loro_threshold)

assist_pct_train = on_train.mean() * 100
assist_pct_loro  = on_loro.mean()  * 100

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("=" * 65)
print("  LORO THRESHOLD COMPARISON SUMMARY")
print("=" * 65)
print(f"  {'Metric':<40}  {'Train-set':>10}  {'LORO':>10}")
print("  " + "-" * 63)
print(f"  {'Youden J threshold':<40}  {train_thr:10.4f}  {loro_threshold:10.4f}")
print(f"  {'Youden J score':<40}  {train_j:10.4f}  {loro_j_mean:10.4f}")
print(f"  {'Threshold delta (LORO - train)':<40}  {'':10}  {loro_threshold - train_thr:+10.4f}")
print()
print(f"  {'Adaptation XDF: mean p_high':<40}  {ph_adapt.mean():10.3f}")
print(f"  {'Adaptation XDF: %>threshold (no smoother)':<40}  "
      f"{(ph_adapt > train_thr).mean()*100:9.1f}%  "
      f"{(ph_adapt > loro_threshold).mean()*100:9.1f}%")
print(f"  {'Assist ON % (EMA + hold + cooldown)':<40}  "
      f"{assist_pct_train:9.1f}%  {assist_pct_loro:9.1f}%")
print(f"  {'assist_on toggle count':<40}  {n_on_train:10d}  {n_on_loro:10d}")
print(f"  {'assist_off toggle count':<40}  {n_off_train:10d}  {n_off_loro:10d}")
print()
print("  Interpretation:")
if abs(loro_threshold - train_thr) < 0.02:
    print("  Thresholds are very similar — training-set Youden J was stable.")
elif loro_threshold > train_thr:
    print(f"  LORO threshold is HIGHER by {loro_threshold - train_thr:.4f}.")
    print("  Training-set threshold was optimistically low (overfit to training data).")
    print("  With LORO, assist would have fired LESS often during adaptation.")
else:
    print(f"  LORO threshold is LOWER by {train_thr - loro_threshold:.4f}.")
    print("  Training-set threshold was pessimistically high.")
    print("  With LORO, assist would have fired MORE often during adaptation.")
print("=" * 65)
