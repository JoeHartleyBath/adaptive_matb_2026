"""Sweep block warm-up duration for PSELF S005 calibration data.

For each warm-up duration in WARMUP_DURATIONS the first N seconds of every
training block are discarded before fitting the SVM.  The question is whether
removing the initial transient improves model quality and/or adaptation
behaviour.

Dataset foundation: +block_01 recovered (same logic as _tmp_retrain_with_block01.py)
  — block_01 START is inferred as block_01_end_ts - 59s.

NOTE: For PSELF S005 the MODERATE and HIGH blocks had identical task
difficulty due to a staircase calibration error.  A 3-class classifier
would be asked to separate an indistinguishable pair, muddying every
metric.  This script therefore uses a BINARY  LOW vs HIGH  classifier
(MODERATE windows are discarded).  This gives a clean measure of the
warmup effect on the only separable contrast in the data.

Metrics reported per warm-up value
-----------------------------------
  • n windows per class (LOW / HIGH only)
  • 5-fold stratified CV balanced accuracy  (binary)
  • Train balanced accuracy  (binary)
  • ROC-AUC  (binary: P(HIGH))
  • Youden-J threshold
  • Per-class mean P(overload)  and  % above threshold

Applied to adaptation XDF
--------------------------
  • Mean / SD of raw P(overload)
  • % windows > threshold  (no smoother)
  • % time  with assist ON  (EMA smoother + hold/cooldown state machine)
  • Number of ON/OFF transitions

Outputs
-------
  results/figures/warmup_sweep_pself_s005.png  — 4-panel metric figure
  (console)  full comparison table
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor, extract_windows, slice_block
from eeg.online_features import OnlineFeatureExtractor
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Study constants — match calibrate_participant.py exactly
# ---------------------------------------------------------------------------
SRATE            = 128.0
CAL_K            = 40
CAL_C            = 1.0
SEED             = 42
BLOCK_DURATION_S = 59   # old scenario: END marker fires at t=59s

WARMUP_DURATIONS = [0, 5, 10, 15, 20, 25, 30]   # seconds to discard per block

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
model_dir = Path(r"C:\data\adaptive_matb\models\PSELF")
physio    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")
feat_cfg  = _REPO / "config" / "eeg_feature_extraction.yaml"

meta     = yaml.safe_load(open(_REPO / "config" / "eeg_metadata.yaml"))
ch_names = meta["channel_names"]

ns        = json.load(open(model_dir / "norm_stats.json"))
norm_mean = np.array(ns["mean"])
norm_std  = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0

cal_xdfs = [
    physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf",
    physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf",
]
adapt_xdf   = physio / "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"
control_xdf = physio / "sub-PSELF_ses-S005_task-matb_acq-control_physio.xdf"

OUT_FIG = _REPO / "results" / "figures" / "warmup_sweep_pself_s005_binary.png"

# Binary mode: MODERATE discarded; classifier is LOW (0) vs HIGH (1)
BINARY_LABEL_MAP = {"LOW": 0, "HIGH": 1}

# ---------------------------------------------------------------------------
# Marker / XDF helpers
# ---------------------------------------------------------------------------
_BLOCK_RE = re.compile(
    r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)"
)


def _load_raw_eeg(xdf_path: Path):
    """Return (eeg_data [n_ch, n_samp], eeg_ts) from XDF, decimated to 128 Hz."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        raise RuntimeError(f"No EEG in {xdf_path.name}")
    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts   = np.array(eeg_stream["time_stamps"])
    if len(eeg_ts) > 1:
        actual = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual > SRATE * 1.1:
            f = int(round(actual / SRATE))
            eeg_data, eeg_ts = eeg_data[:, ::f], eeg_ts[::f]
    return eeg_data, eeg_ts


def _preprocess(eeg_data):
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(eeg_data.shape[0])
    return pp.process(eeg_data)


def _get_markers(xdf_path: Path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    m = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    if m is None:
        return []
    return list(zip(m["time_stamps"], [s[0] for s in m["time_series"]]))


def _parse_recorded_blocks(markers):
    """Blocks that have both START + END markers."""
    opens, blocks = {}, []
    for ts, ev in markers:
        hit = _BLOCK_RE.search(ev)
        if not hit:
            continue
        lv = hit.group("level")
        if hit.group("ev") == "START":
            opens[lv] = ts
        elif hit.group("ev") == "END" and lv in opens:
            blocks.append((opens.pop(lv), ts, lv))
    return blocks


def _get_block01_info(markers):
    """Return (end_ts, level_str) for block_01, or (None, None)."""
    for ts, ev in markers:
        hit = re.search(r"block_01/(\w+)/END", ev)
        if hit:
            return ts, hit.group(1)
    return None, None


# ---------------------------------------------------------------------------
# Feature extraction for one block (handles warmup offset)
# ---------------------------------------------------------------------------

def _extract_block_features(preprocessed, eeg_ts, t_start, t_end, warmup_s):
    """Extract feature windows from [t_start + warmup_s, t_end].

    Returns (features [n_win, n_feat],).  n_win may be 0.
    """
    effective_start = t_start + warmup_s
    if effective_start >= t_end:
        return np.empty((0,), dtype=np.float32)
    i0 = int(np.searchsorted(eeg_ts, effective_start))
    i1 = int(np.searchsorted(eeg_ts, t_end))
    # Direct slice — bypasses slice_block's hardcoded 30s warmup so that
    # warmup_s above is the sole warmup applied.
    block_data = preprocessed[:, i0: min(i1, preprocessed.shape[1])]
    windows = extract_windows(block_data, WINDOW_CONFIG)    # (n_win, n_ch, win_samp)
    if windows.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    return np.array([extractor.compute(w) for w in windows])  # (n_win, n_feat)


# ---------------------------------------------------------------------------
# Pre-load and preprocess all calibration XDFs (done once, reused across sweep)
# ---------------------------------------------------------------------------

class _CalibrationData:
    """Holds preprocessed EEG + block time ranges for one calibration XDF."""
    def __init__(self, xdf_path: Path):
        self.path = xdf_path
        self.acq  = xdf_path.stem.split("acq-")[1].replace("_physio", "")
        print(f"  Loading & preprocessing {self.acq} ...", flush=True)
        eeg_raw, self.eeg_ts = _load_raw_eeg(xdf_path)
        self.preprocessed = _preprocess(eeg_raw)
        markers = _get_markers(xdf_path)

        # blocks 02–09 (have both START + END)
        self.blocks = _parse_recorded_blocks(markers)

        # block_01 recovery
        b01_end, b01_level = _get_block01_info(markers)
        if b01_end is not None:
            b01_start = b01_end - BLOCK_DURATION_S
            t_avail   = max(self.eeg_ts[0], b01_start)
            self.blocks.append((t_avail, b01_end, b01_level))
            recovered_s = b01_end - t_avail
            print(f"    block_01 RECOVERED {b01_level:<10}  ({recovered_s:.0f}s available)")
        else:
            print(f"    block_01: no END marker — skipped")

        print(f"    total blocks available: {len(self.blocks)}")


print("\n=== Loading calibration data ===")
cal_data = [_CalibrationData(p) for p in cal_xdfs]


# ---------------------------------------------------------------------------
# For each warmup value: build dataset, fit SVM, evaluate
# ---------------------------------------------------------------------------

def _build_dataset(warmup_s: float):
    """Build binary (X, y) — LOW=0, HIGH=1 — discarding MODERATE windows."""
    epochs, labels = [], []
    for cd in cal_data:
        for t_s, t_e, level in cd.blocks:
            if level not in BINARY_LABEL_MAP:
                continue   # discard MODERATE
            feats = _extract_block_features(cd.preprocessed, cd.eeg_ts, t_s, t_e, warmup_s)
            if feats.ndim < 2 or feats.shape[0] == 0:
                continue
            lbl = BINARY_LABEL_MAP[level]
            epochs.append(feats)
            labels.append(np.full(len(feats), lbl, dtype=np.int64))
    if not epochs:
        raise RuntimeError(f"No windows for warmup={warmup_s}s")
    return np.concatenate(epochs), np.concatenate(labels)


def _fit_evaluate(X_raw, y, warmup_s):
    """Normalise + fit binary SVM (LOW=0, HIGH=1) + compute metrics."""
    X = (X_raw - norm_mean) / norm_std

    n_feat = X.shape[1]
    k = min(CAL_K, n_feat)

    pipe_cv = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                    probability=True, random_state=SEED)),
    ])

    # Guard: need at least 5 samples per class for 5-fold CV
    min_class_n = int(np.bincount(y).min())
    n_splits = min(5, min_class_n)
    if n_splits < 2:
        cv_ba = float("nan")
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        cv_ba = cross_val_score(pipe_cv, X, y, cv=cv,
                                scoring="balanced_accuracy").mean()

    # Full fit for threshold + AUC
    sel = SelectKBest(f_classif, k=k)
    X_sel = sel.fit_transform(X, y)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_sel)
    clf   = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                probability=True, random_state=SEED)
    clf.fit(X_sc, y)

    # Binary: class index 1 == HIGH
    p_high    = clf.predict_proba(X_sc)[:, 1]
    auc       = roc_auc_score(y, p_high)
    fpr, tpr, thr_arr = roc_curve(y, p_high)
    j_score   = tpr - fpr
    best_i    = int(np.argmax(j_score))
    threshold = float(thr_arr[best_i])
    youden_j  = float(j_score[best_i])
    train_ba  = balanced_accuracy_score(y, clf.predict(X_sc))

    # Per-class stats (LOW and HIGH only)
    class_stats = {}
    for lv, lbl in BINARY_LABEL_MAP.items():
        mask = y == lbl
        if mask.any():
            ph = p_high[mask]
            class_stats[lv] = {
                "mean_ph":   float(ph.mean()),
                "pct_above": float((ph > threshold).mean() * 100),
                "n":         int(mask.sum()),
            }

    # Deploy pipeline (sel → sc → clf)
    deploy_pipe = Pipeline([("sc", sc), ("clf", clf)])

    return {
        "warmup_s":   warmup_s,
        "n":          len(y),
        "cv_ba":      cv_ba,
        "train_ba":   train_ba,
        "auc":        auc,
        "threshold":  threshold,
        "youden_j":   youden_j,
        "class_stats": class_stats,
        "_selector":  sel,
        "_pipeline":  deploy_pipe,
    }


# ---------------------------------------------------------------------------
# Apply fitted model to a full XDF (no warmup at inference time)
# ---------------------------------------------------------------------------

def _predict_xdf(xdf_path: Path, selector, pipeline):
    """Return (lsl_ts, p_high) for every window in the XDF.

    For the binary classifier p_high = P(class=HIGH) = predict_proba[:, 1].
    """
    eeg_raw, eeg_ts = _load_raw_eeg(xdf_path)
    preprocessed = _preprocess(eeg_raw)
    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    step  = int(WINDOW_CONFIG.step_s * SRATE)
    win   = int(WINDOW_CONFIG.window_s * SRATE)
    n     = preprocessed.shape[1]
    ts_list, ph_list = [], []
    for start in range(win, n - step, step):
        window = preprocessed[:, start - win: start]
        feats  = extractor.compute(window)
        feats_z   = (feats - norm_mean) / norm_std
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba = pipeline.predict_proba(feats_sel)[0]
        ts_list.append(eeg_ts[start])
        ph_list.append(float(proba[1]))   # binary: index 1 == HIGH
    return np.array(ts_list), np.array(ph_list)


def _simulate_scheduler(ph_vals, threshold,
                         alpha=0.05, hysteresis=0.02,
                         t_hold_s=3.0, cooldown_s=15.0, step_s=0.25):
    """EMA + hold-timer + cooldown state machine.  Returns (on_flags, ema, n_on, n_off)."""
    ema = None; zone = None; zone_entry = None
    assist = False; cooldown_end = 0.0
    on_flags, ema_trace = [], []
    n_on = n_off = 0
    for i, v in enumerate(ph_vals):
        t = i * step_s
        ema = float(v) if ema is None else alpha * float(v) + (1.0 - alpha) * ema
        new_zone = (
            "above" if ema > threshold + hysteresis else
            "below" if ema < threshold - hysteresis else "dead"
        )
        if new_zone != zone:
            zone = new_zone; zone_entry = t
        hold = t - zone_entry
        if t >= cooldown_end:
            if zone == "above" and hold >= t_hold_s and not assist:
                assist = True;  cooldown_end = t + cooldown_s; zone_entry = t; n_on += 1
            elif zone == "below" and hold >= t_hold_s and assist:
                assist = False; cooldown_end = t + cooldown_s; zone_entry = t; n_off += 1
        on_flags.append(assist)
        ema_trace.append(ema)
    return np.array(on_flags), np.array(ema_trace), n_on, n_off


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

print("\n=== Fitting models for each warm-up duration ===")
results = []

for ws in WARMUP_DURATIONS:
    print(f"\n  warmup={ws}s ...", flush=True)
    try:
        X, y = _build_dataset(ws)
    except RuntimeError as exc:
        print(f"    SKIPPED: {exc}")
        continue
    counts = {lv: int((y == lbl).sum()) for lv, lbl in BINARY_LABEL_MAP.items()}
    print(f"    windows → {counts}", flush=True)
    r = _fit_evaluate(X, y, ws)
    results.append(r)
    print(f"    CV BA={r['cv_ba']:.4f}  train BA={r['train_ba']:.4f}  "
          f"AUC={r['auc']:.4f}  thr={r['threshold']:.4f}")


# ---------------------------------------------------------------------------
# Apply each model to adaptation XDF
# ---------------------------------------------------------------------------

print("\n=== Applying models to adaptation XDF ===")
adapt_stats = []
for r in results:
    ws = r["warmup_s"]
    print(f"  warmup={ws}s ...", flush=True)
    ts, ph = _predict_xdf(adapt_xdf, r["_selector"], r["_pipeline"])
    on, ema, n_on, n_off = _simulate_scheduler(ph, r["threshold"])
    adapt_stats.append({
        "warmup_s":    ws,
        "ph_mean":     float(ph.mean()),
        "ph_std":      float(ph.std()),
        "pct_above":   float((ph > r["threshold"]).mean() * 100),
        "assist_pct":  float(on.mean() * 100),
        "n_on":        n_on,
        "n_off":       n_off,
    })


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _h(text, width=76):
    print("\n" + "=" * width)
    print("  " + text)
    print("=" * width)


_h("WARM-UP SWEEP  —  PSELF S005  —  BINARY: LOW vs HIGH  (MODERATE discarded)")
print("  NOTE: MODERATE blocks excluded — MOD/HIGH had identical difficulty (staircase error)")

# — training metrics table —
col_w = 10
hdr = (
    f"  {'warmup_s':<10}"
    f"  {'n_total':>{col_w}}"
    f"  {'n_LOW':>{col_w}}"
    f"  {'n_HIGH':>{col_w}}"
    f"  {'CV_BA':>{col_w}}"
    f"  {'Train_BA':>{col_w}}"
    f"  {'AUC':>{col_w}}"
    f"  {'Youden_J':>{col_w}}"
    f"  {'threshold':>{col_w}}"
)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in results:
    cs = r["class_stats"]
    print(
        f"  {r['warmup_s']:<10}"
        f"  {r['n']:>{col_w}}"
        f"  {cs.get('LOW',  {}).get('n', 0):>{col_w}}"
        f"  {cs.get('HIGH', {}).get('n', 0):>{col_w}}"
        f"  {r['cv_ba']:>{col_w}.4f}"
        f"  {r['train_ba']:>{col_w}.4f}"
        f"  {r['auc']:>{col_w}.4f}"
        f"  {r['youden_j']:>{col_w}.4f}"
        f"  {r['threshold']:>{col_w}.4f}"
    )

# — per-class mean p_high at own threshold —
_h("PER-CLASS  mean P(HIGH)  and  % > threshold  (at own threshold)")
print(f"  {'warmup_s':<10}", end="")
for lv in ["LOW", "HIGH"]:
    print(f"  {lv+'_mean':>12}  {lv+'_%>thr':>12}", end="")
print()
print("  " + "-" * 54)
for r in results:
    print(f"  {r['warmup_s']:<10}", end="")
    for lv in ["LOW", "HIGH"]:
        cs = r["class_stats"].get(lv, {"mean_ph": float("nan"), "pct_above": float("nan")})
        print(f"  {cs['mean_ph']:>12.3f}  {cs['pct_above']:>11.1f}%", end="")
    print()

# — adaptation behaviour table —
_h("ADAPTATION BEHAVIOUR  (EMA α=0.05 | hold=3s | cooldown=15s | hyst=0.02)")
ahdr = (
    f"  {'warmup_s':<10}"
    f"  {'ph_mean':>10}"
    f"  {'ph_std':>8}"
    f"  {'%>thr':>8}"
    f"  {'assist%':>8}"
    f"  {'n_on':>6}"
    f"  {'n_off':>6}"
)
print(ahdr)
print("  " + "-" * (len(ahdr) - 2))
for a in adapt_stats:
    print(
        f"  {a['warmup_s']:<10}"
        f"  {a['ph_mean']:>10.3f}"
        f"  {a['ph_std']:>8.3f}"
        f"  {a['pct_above']:>7.1f}%"
        f"  {a['assist_pct']:>7.1f}%"
        f"  {a['n_on']:>6}"
        f"  {a['n_off']:>6}"
    )

# — delta from baseline (warmup=0) —
_h("DELTA FROM BASELINE  (warmup=0s)")
r0 = results[0]
a0 = adapt_stats[0]
dhdr = (
    f"  {'warmup_s':<10}"
    f"  {'Δ CV_BA':>10}"
    f"  {'Δ AUC':>8}"
    f"  {'Δ thr':>8}"
    f"  {'Δ assist%':>10}"
    f"  {'Δ n_on':>8}"
)
print(dhdr)
print("  " + "-" * (len(dhdr) - 2))
for r, a in zip(results, adapt_stats):
    ws = r["warmup_s"]
    print(
        f"  {ws:<10}"
        f"  {r['cv_ba']   - r0['cv_ba']:>+10.4f}"
        f"  {r['auc']     - r0['auc']:>+8.4f}"
        f"  {r['threshold']- r0['threshold']:>+8.4f}"
        f"  {a['assist_pct'] - a0['assist_pct']:>+10.1f}"
        f"  {a['n_on']    - a0['n_on']:>+8}"
    )


# ---------------------------------------------------------------------------
# Figure: 4-panel summary plot
# ---------------------------------------------------------------------------

ws_vals = [r["warmup_s"] for r in results]
cv_ba   = [r["cv_ba"]    for r in results]
auc_    = [r["auc"]      for r in results]
thr_    = [r["threshold"] for r in results]
n_tot   = [r["n"]        for r in results]
n_low   = [r["class_stats"].get("LOW",  {"n": 0})["n"] for r in results]
n_high  = [r["class_stats"].get("HIGH", {"n": 0})["n"] for r in results]

ast_pct  = [a["assist_pct"] for a in adapt_stats]
ph_mean  = [a["ph_mean"]    for a in adapt_stats]
n_on_arr = [a["n_on"]       for a in adapt_stats]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    "Warm-up sweep  —  PSELF S005  —  binary: LOW vs HIGH  (MODERATE discarded)",
    fontsize=11,
)

# Panel 1: CV balanced accuracy + AUC
ax = axes[0, 0]
ax.plot(ws_vals, cv_ba, "o-", color="steelblue", label="CV balanced acc")
ax.plot(ws_vals, auc_,  "s--", color="darkorange", label="ROC-AUC")
ax.axhline(cv_ba[0], linestyle=":", color="steelblue", alpha=0.4, linewidth=0.8)
ax.axhline(auc_[0],  linestyle=":", color="darkorange", alpha=0.4, linewidth=0.8)
ax.set_xlabel("Warm-up (s)", fontsize=9)
ax.set_ylabel("Score", fontsize=9)
ax.set_title("Model quality", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# Panel 2: window counts per class
ax = axes[0, 1]
ax.plot(ws_vals, n_low,  "o-", color="steelblue", label="LOW")
ax.plot(ws_vals, n_high, "^-", color="firebrick",  label="HIGH")
ax.set_xlabel("Warm-up (s)", fontsize=9)
ax.set_ylabel("Training windows", fontsize=9)
ax.set_title("Windows per class  (MODERATE excluded)", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# Panel 3: Youden threshold
ax = axes[1, 0]
ax.plot(ws_vals, thr_, "D-", color="purple")
ax.axhline(thr_[0], linestyle=":", color="purple", alpha=0.4, linewidth=0.8)
ax.set_xlabel("Warm-up (s)", fontsize=9)
ax.set_ylabel("P(overload) threshold", fontsize=9)
ax.set_title("Youden-J threshold", fontsize=9)
ax.tick_params(labelsize=8)

# Panel 4: adaptation behaviour
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(ws_vals, ast_pct,  "o-", color="green",   label="assist ON %")
ax2.plot(ws_vals, n_on_arr, "s--", color="teal",   label="n ON events")
ax.axhline(ast_pct[0], linestyle=":", color="green", alpha=0.4, linewidth=0.8)
ax.set_xlabel("Warm-up (s)", fontsize=9)
ax.set_ylabel("Assist ON (%)", fontsize=9, color="green")
ax2.set_ylabel("ON events", fontsize=9, color="teal")
ax.set_title("Adaptation behaviour", fontsize=9)
ax.tick_params(axis="y", colors="green", labelsize=8)
ax2.tick_params(axis="y", colors="teal", labelsize=8)
ax.tick_params(axis="x", labelsize=8)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

fig.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=150)
plt.close(fig)
print(f"\nFigure saved: {OUT_FIG}")
