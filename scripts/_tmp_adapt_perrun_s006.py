"""Test per-run LOW normalisation on S006 adaptation condition.

Steps
-----
1. Load C1 + C2, compute per-run LOW norms, fit full model (SelectKBest(35) →
   StandardScaler → SVC linear, same hypers as calibrate.py).
2. Load adaptation XDF: recover block_01 LOW (EEG starts 55 s in), use
   block_06 LOW as the second LOW block → compute adaptation LOW norms.
3. Drift report: compare per-run-normalised feature means C2 vs adaptation.
4. Continuous inference on full adaptation XDF, normalised by adapt LOW norms.
5. Replay EMA scheduler with LORO threshold (0.0328 from Strategy C).
6. Plot timeline figure (background blocks, p_high, EMA, threshold, assist-ON).

Run:
    $env:MATB_SCENARIO_OFFSET_S = "0.943"
    .venv\\Scripts\\python.exe scripts/_tmp_adapt_perrun_s006.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import calibrate_participant as _cal_mod
from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG
from eeg import EegPreprocessor, extract_windows, slice_block
from eeg.extract_features import _build_region_map, _extract_feat
from eeg.online_features import OnlineFeatureExtractor
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Config — match calibrate_participant.py exactly
# ---------------------------------------------------------------------------
import pyxdf

SRATE       = 128.0
CAL_K       = 35
CAL_C       = 1.0
SEED        = 42
BLOCK_DUR_S = 59.0          # adaptation scenario block duration
LORO_THRESHOLD = 0.3663     # 10-fold CV Youden threshold (replaces LORO pooled threshold)

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PSELF")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf",
]
ADAPT_XDF = PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-adaptation_physio.xdf"
TAGS = ["C1", "C2"]

LOW_LBL  = LABEL_MAP["LOW"]
HIGH_LBL = LABEL_MAP["HIGH"]
MOD_LBL  = LABEL_MAP["MODERATE"]

_ADAPT_BLOCK_RE = re.compile(
    r"STUDY/V0/adaptive_automation/\d+/block_(?P<num>\d+)/(?P<level>LOW|MODERATE|HIGH)/(?P<event>START|END)"
)

# ---------------------------------------------------------------------------
# Generic EEG helpers (shared with _tmp_retrain_with_block01.py pattern)
# ---------------------------------------------------------------------------

def _merge_eeg(streams):
    """Return first EEG stream with > 4 channels (same logic as build_mwl)."""
    from build_mwl_training_dataset import _merge_eeg_streams
    return _merge_eeg_streams(streams)


def load_eeg_from_xdf(xdf_path: Path):
    """Return (preprocessed [n_ch, n], eeg_ts) decimated and filtered."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg(streams)
    if eeg_stream is None:
        raise RuntimeError(f"No EEG in {xdf_path.name}")
    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts   = np.array(eeg_stream["time_stamps"])
    if len(eeg_ts) > 1:
        actual = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual > SRATE * 1.1:
            f = int(round(actual / SRATE))
            eeg_data = eeg_data[:, ::f]
            eeg_ts   = eeg_ts[::f]
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(eeg_data.shape[0])
    preprocessed = pp.process(eeg_data)
    return preprocessed, eeg_ts


def _norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def windows_from_ts_range(preprocessed, eeg_ts, t_start, t_end):
    """Extract windowed epochs from an LSL timestamp range."""
    i0 = int(np.searchsorted(eeg_ts, t_start))
    i1 = int(np.searchsorted(eeg_ts, t_end))
    block = slice_block(preprocessed, i0, i1, WINDOW_CONFIG)
    return extract_windows(block, WINDOW_CONFIG)


def get_adapt_markers(xdf_path: Path) -> list[tuple[float, str]]:
    """Return [(timestamp, marker_text)] from the OpenMATB stream."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    m = next((s for s in streams if "MATB" in str(s["info"]["name"])), None)
    if m is None:
        return []
    return [(float(ts), str(x[0])) for ts, x in zip(m["time_stamps"], m["time_series"])]


def parse_adapt_blocks(markers) -> list[tuple[float, float, str]]:
    """Return [(start_ts, end_ts, level)] for fully-labeled adaptation blocks."""
    opens: dict[str, float] = {}
    result = []
    for ts, ev in markers:
        mm = _ADAPT_BLOCK_RE.match(ev.split("|")[0])
        if not mm:
            continue
        num   = mm.group("num")
        level = mm.group("level")
        key   = f"{num}_{level}"
        if mm.group("event") == "START":
            opens[key] = ts
        elif mm.group("event") == "END" and key in opens:
            result.append((opens.pop(key), ts, level))
    return result


# ---------------------------------------------------------------------------
# Step 1: Load C1 + C2, compute per-run LOW norms, build training data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 — Load calibration runs (C1, C2) with per-run LOW norm")
print("=" * 70)

region_map = _build_region_map(FEAT_CFG, CH_NAMES)
feat_names: list[str] = []

xdf_raw_X: list[np.ndarray] = []
xdf_y:     list[np.ndarray] = []
xdf_low_X: list[np.ndarray] = []

for xdf_path, tag in zip(CAL_XDFS, TAGS):
    print(f"  {tag}: ", end="", flush=True)
    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)
    if results is None:
        print("FAILED"); sys.exit(1)

    all_feats, all_labels, low_feats = [], [], []
    for epochs, level_str in results:
        lbl = LABEL_MAP[level_str]
        X_raw, names = _extract_feat(epochs, SRATE, region_map)
        if not feat_names:
            feat_names = names
        all_feats.append(X_raw)
        all_labels.extend([lbl] * len(X_raw))
        if lbl == LOW_LBL:
            low_feats.append(X_raw)

    X_all = np.concatenate(all_feats)
    y_all = np.array(all_labels, dtype=np.int64)
    X_low = np.concatenate(low_feats) if low_feats else X_all

    xdf_raw_X.append(X_all)
    xdf_y.append(y_all)
    xdf_low_X.append(X_low)

n_feat = len(feat_names)
print(f"\nFeature vector length: {n_feat}")

# Per-run LOW norms for C1 and C2
cal_norms = [_norm_stats(Xl) for Xl in xdf_low_X]
for (mn, _), tag in zip(cal_norms, TAGS):
    print(f"  {tag} LOW norm mean RMS: {float(np.sqrt(np.mean(mn**2))):.4f}")

# Build combined per-run-normalised training set
X_train = np.concatenate([(X - mn) / st for X, (mn, st) in zip(xdf_raw_X, cal_norms)])
y_train = np.concatenate(xdf_y)
print(f"\nTraining set: n={len(y_train)}  "
      f"LOW={int((y_train==LOW_LBL).sum())}  "
      f"MOD={int((y_train==MOD_LBL).sum())}  "
      f"HIGH={int((y_train==HIGH_LBL).sum())}")

# ---------------------------------------------------------------------------
# Step 2: Fit full model on combined per-run-normalised data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — Fit model")
print("=" * 70)

sel = SelectKBest(f_classif, k=min(CAL_K, n_feat))
X_sel = sel.fit_transform(X_train, y_train)
sc  = StandardScaler()
X_sc = sc.fit_transform(X_sel)
clf = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
          probability=True, random_state=SEED)
clf.fit(X_sc, y_train)

deploy_pipe = Pipeline([("sc", sc), ("clf", clf)])

# Training-set Youden threshold (for reference)
p_high_train = clf.predict_proba(X_sc)[:, -1]
y_bin_train  = (y_train == HIGH_LBL).astype(int)
fpr_tr, tpr_tr, thr_tr = roc_curve(y_bin_train, p_high_train)
j_tr = tpr_tr - fpr_tr
best_tr = int(np.argmax(j_tr))
train_threshold = float(thr_tr[best_tr])
train_j         = float(j_tr[best_tr])
train_auc       = roc_auc_score(y_bin_train, p_high_train)
print(f"  Training-set J={train_j:.3f}  AUC={train_auc:.3f}  threshold={train_threshold:.4f}")
print(f"  LORO threshold (Strategy C): {LORO_THRESHOLD:.4f}")

# ---------------------------------------------------------------------------
# Step 3: Load adaptation XDF, extract labeled blocks + LOW blocks
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — Load adaptation XDF")
print("=" * 70)

adapt_markers = get_adapt_markers(ADAPT_XDF)
adapt_blocks  = parse_adapt_blocks(adapt_markers)   # fully-labeled [(s, e, level)]

print(f"  Recorded labeled blocks (START+END both present):")
for s, e, lv in adapt_blocks:
    print(f"    {lv:<10}  {s:.2f} to {e:.2f}  dur={e-s:.0f}s")

# Recover block_01 LOW (only END marker present, EEG starts during this block)
b01_end_ts = next(
    (float(ts) for ts, ev in adapt_markers
     if re.search(r"block_01/LOW/END", ev.split("|")[0])),
    None,
)
print(f"\n  block_01/LOW END ts: {b01_end_ts}")

# Preprocess full adaptation EEG (once — used for both norm and inference)
print("\n  Preprocessing adaptation EEG ... ", end="", flush=True)
preprocessed_adapt, adapt_eeg_ts = load_eeg_from_xdf(ADAPT_XDF)
t_eeg_start = adapt_eeg_ts[0]
print(f"done  ({len(adapt_eeg_ts)} samples, {adapt_eeg_ts[-1]-adapt_eeg_ts[0]:.1f}s)")

# Extract raw features per labeled block (for drift check)
extractor = OnlineFeatureExtractor(CH_NAMES, srate=SRATE, region_cfg=FEAT_CFG)

adapt_block_X: list[np.ndarray] = []
adapt_block_y: list[np.ndarray] = []
adapt_low_raw: list[np.ndarray] = []   # raw features of LOW windows

for t_s, t_e, level in adapt_blocks:
    wins = windows_from_ts_range(preprocessed_adapt, adapt_eeg_ts, t_s, t_e)
    if wins.shape[0] == 0:
        print(f"    WARN: no windows for {level} block {t_s:.0f}-{t_e:.0f}")
        continue
    X_raw, _ = _extract_feat(wins, SRATE, region_map)
    lbl = LABEL_MAP[level]
    adapt_block_X.append(X_raw)
    adapt_block_y.append(np.full(len(X_raw), lbl, dtype=np.int64))
    if lbl == LOW_LBL:
        adapt_low_raw.append(X_raw)

# Recover block_01 LOW
if b01_end_ts is not None:
    t_b01_start = max(t_eeg_start, b01_end_ts - BLOCK_DUR_S)
    wins = windows_from_ts_range(preprocessed_adapt, adapt_eeg_ts, t_b01_start, b01_end_ts)
    if wins.shape[0] > 0:
        X_raw, _ = _extract_feat(wins, SRATE, region_map)
        adapt_low_raw.insert(0, X_raw)      # prepend (chronologically first)
        adapt_block_X.insert(0, X_raw)
        adapt_block_y.insert(0, np.full(len(X_raw), LOW_LBL, dtype=np.int64))
        avail_s = b01_end_ts - t_b01_start
        print(f"  block_01 LOW RECOVERED: {avail_s:.0f}s available, {wins.shape[0]} windows")
    else:
        print("  block_01 LOW: no windows after clamping to EEG start")

if not adapt_low_raw:
    print("ERROR: no LOW windows in adaptation — cannot compute per-run norm")
    sys.exit(1)

X_adapt_low_all = np.concatenate(adapt_low_raw)
adapt_norm_mean, adapt_norm_std = _norm_stats(X_adapt_low_all)
print(f"\n  Adaptation LOW norm: n_windows={len(X_adapt_low_all)}")
print(f"  Adaptation LOW norm mean RMS: {float(np.sqrt(np.mean(adapt_norm_mean**2))):.4f}")
print(f"  C2       LOW norm mean RMS:   {float(np.sqrt(np.mean(cal_norms[1][0]**2))):.4f}")

# ---------------------------------------------------------------------------
# Step 4: Drift report — C2 vs adaptation (per-run-LOW normalised)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — Drift: C2 vs Adaptation (per-run LOW normalised)")
print("=" * 70)

# C2 per-run-normalised features (already in xdf_raw_X[1] / cal_norms[1])
c2_mn, c2_st = cal_norms[1]
C2_norm = (xdf_raw_X[1] - c2_mn) / c2_st
y_C2    = xdf_y[1]

# Adaptation per-run-normalised features (labeled blocks only)
if adapt_block_X:
    X_adapt_labeled = np.concatenate(adapt_block_X)
    y_adapt_labeled = np.concatenate(adapt_block_y)
    X_adapt_labeled_norm = (X_adapt_labeled - adapt_norm_mean) / adapt_norm_std
else:
    X_adapt_labeled_norm = np.empty((0, n_feat))
    y_adapt_labeled = np.empty(0, dtype=np.int64)

print(f"  {'Feature':<25}  {'C2_LOW_m':>9}  {'Ad_LOW_m':>9}  {'D_LOW':>8}  {'C2_HI_m':>9}  {'Ad_HI_m':>9}  {'D_HIGH':>8}")
print("  " + "-" * 90)

DRIFT_FEATURES = [
    "FM_Beta", "Cen_Beta", "Par_HjAct", "Cen_Engagement",
    "Cen_SpEnt", "Par_SpEnt", "Occ_SpEnt",
]
for fname in DRIFT_FEATURES:
    fi = feat_names.index(fname) if fname in feat_names else None
    if fi is None:
        continue

    c2_low_mu  = float(C2_norm[y_C2 == LOW_LBL,  fi].mean()) if (y_C2 == LOW_LBL).any()  else float("nan")
    c2_hi_mu   = float(C2_norm[y_C2 == HIGH_LBL, fi].mean()) if (y_C2 == HIGH_LBL).any() else float("nan")

    ad_low_mu  = float(X_adapt_labeled_norm[y_adapt_labeled == LOW_LBL,  fi].mean()) \
                 if (y_adapt_labeled == LOW_LBL).any()  else float("nan")
    ad_hi_mu   = float(X_adapt_labeled_norm[y_adapt_labeled == HIGH_LBL, fi].mean()) \
                 if (y_adapt_labeled == HIGH_LBL).any() else float("nan")

    d_low  = ad_low_mu  - c2_low_mu
    d_high = ad_hi_mu   - c2_hi_mu

    print(f"  {fname:<25}  {c2_low_mu:+9.3f}  {ad_low_mu:+9.3f}  {d_low:+8.3f}"
          f"  {c2_hi_mu:+9.3f}  {ad_hi_mu:+9.3f}  {d_high:+8.3f}")

# Summary: overall mean L2 drift across all features and all shared levels
all_level_deltas = []
for lbl in [LOW_LBL, MOD_LBL, HIGH_LBL]:
    c2_mask  = y_C2 == lbl
    ad_mask  = y_adapt_labeled == lbl
    if not c2_mask.any() or not ad_mask.any():
        continue
    delta = C2_norm[c2_mask].mean(axis=0) - X_adapt_labeled_norm[ad_mask].mean(axis=0)
    all_level_deltas.append(delta)

if all_level_deltas:
    mean_abs_drift = float(np.mean([np.abs(d) for d in all_level_deltas]))
    max_abs_drift  = float(np.max([np.abs(d).max() for d in all_level_deltas]))
    print(f"\n  Mean|D| across all features+levels: {mean_abs_drift:.3f} Z-units")
    print(f"  Max |D| (worst single feature):     {max_abs_drift:.3f} Z-units")

# ---------------------------------------------------------------------------
# Step 5: Continuous inference on full adaptation XDF
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 — Continuous inference on adaptation XDF")
print("=" * 70)

step_samp = int(WINDOW_CONFIG.step_s  * SRATE)
win_samp  = int(WINDOW_CONFIG.window_s * SRATE)
n_samp    = preprocessed_adapt.shape[1]

ts_list: list[float] = []
ph_list: list[float] = []

for start in range(win_samp, n_samp - step_samp + 1, step_samp):
    window    = preprocessed_adapt[:, start - win_samp : start]
    feats     = extractor.compute(window)
    feats_z   = (feats - adapt_norm_mean) / adapt_norm_std
    feats_sel = sel.transform(feats_z[np.newaxis, :])
    proba     = deploy_pipe.predict_proba(feats_sel)[0]
    ts_list.append(float(adapt_eeg_ts[start]))
    ph_list.append(float(proba[-1]))

lsl_ts = np.array(ts_list)
ph_arr = np.array(ph_list)
t_rel  = lsl_ts - lsl_ts[0]

print(f"  Inferred {len(ph_arr)} windows")
print(f"  p_high  mean={ph_arr.mean():.3f}  std={ph_arr.std():.3f}  "
      f"min={ph_arr.min():.3f}  max={ph_arr.max():.3f}")
print(f"  % > LORO thr ({LORO_THRESHOLD:.4f}): {(ph_arr > LORO_THRESHOLD).mean()*100:.1f}%")
print(f"  % > train thr ({train_threshold:.4f}): {(ph_arr > train_threshold).mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# Step 6: EMA scheduler simulation
# ---------------------------------------------------------------------------

def simulate_full(ph_vals, threshold,
                  alpha=0.05, hysteresis=0.02,
                  t_hold_s=3.0, cooldown_s=15.0, step_s=None):
    step_s = step_s or float(WINDOW_CONFIG.step_s)
    ema = None; zone = None; zone_entry = None
    assist_on = False; cooldown_end = 0.0
    on_flags, ema_trace = [], []
    n_on = n_off = 0
    for i, v in enumerate(ph_vals):
        t = i * step_s
        ema = float(v) if ema is None else alpha * float(v) + (1.0 - alpha) * ema
        new_zone = ("above" if ema > threshold + hysteresis else
                    "below" if ema < threshold - hysteresis else "dead")
        if new_zone != zone:
            zone = new_zone; zone_entry = t
        hold_s = t - zone_entry
        if t >= cooldown_end:
            if zone == "above" and hold_s >= t_hold_s and not assist_on:
                assist_on = True; cooldown_end = t + cooldown_s; zone_entry = t; n_on += 1
            elif zone == "below" and hold_s >= t_hold_s and assist_on:
                assist_on = False; cooldown_end = t + cooldown_s; zone_entry = t; n_off += 1
        on_flags.append(assist_on)
        ema_trace.append(ema)
    return np.array(on_flags), np.array(ema_trace), n_on, n_off


on_flags, ema_trace, n_on, n_off = simulate_full(ph_arr, LORO_THRESHOLD)
pct_on = 100.0 * on_flags.sum() / len(on_flags)
print(f"\n  Assist ON: {pct_on:.1f}%  ({n_on} activations, {n_off} deactivations)")
print(f"  (EMA alpha=0.05, hold=3s, cooldown=15s, hysteresis=0.02)")

# ---------------------------------------------------------------------------
# Step 7: Plot figure
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7 — Plotting")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_LEVEL_COLOURS = {
    "HIGH":     "tab:red",
    "MODERATE": "tab:orange",
    "LOW":      "tab:blue",
}


def _shade_assist(ax, t, assist_on):
    dt    = float(np.median(np.diff(t))) if len(t) > 1 else 0.25
    in_r  = False; start = 0.0
    for i in range(len(t)):
        if assist_on[i] and not in_r:
            start = t[i] - dt / 2; in_r = True
        elif not assist_on[i] and in_r:
            ax.axvspan(start, t[i - 1] + dt / 2, color="green", alpha=0.15, zorder=1)
            in_r = False
    if in_r:
        ax.axvspan(start, t[-1] + dt / 2, color="green", alpha=0.15, zorder=1)


fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                         gridspec_kw={"height_ratios": [3, 1]})

ax_main, ax_drift = axes

# ---- Main subplot: p_high + EMA + blocks + assist ----
t0_lsl = lsl_ts[0]

for t_s, t_e, level in adapt_blocks:
    xs = t_s - t0_lsl; xe = t_e - t0_lsl
    colour = _LEVEL_COLOURS.get(level, "grey")
    ax_main.axvspan(xs, xe, color=colour, alpha=0.08, zorder=0)
    ax_main.text((xs + xe) / 2, 1.03, level,
                 ha="center", va="bottom", fontsize=6.5, color=colour,
                 transform=ax_main.get_xaxis_transform())

# Also shade recovered block_01 LOW
if b01_end_ts is not None:
    t_b01_start_plot = max(t_eeg_start, b01_end_ts - BLOCK_DUR_S)
    xs = t_b01_start_plot - t0_lsl; xe = b01_end_ts - t0_lsl
    ax_main.axvspan(xs, xe, color=_LEVEL_COLOURS["LOW"], alpha=0.08, zorder=0)
    ax_main.text((xs + xe) / 2, 1.03, "LOW*",
                 ha="center", va="bottom", fontsize=6.5,
                 color=_LEVEL_COLOURS["LOW"],
                 transform=ax_main.get_xaxis_transform())

_shade_assist(ax_main, t_rel, on_flags)

ax_main.plot(t_rel, ph_arr, color="0.70", alpha=0.5, linewidth=0.5,
             label="raw P(overload)")
ax_main.plot(t_rel, ema_trace, color="0.15", linewidth=1.0, label="smoothed EMA")
ax_main.axhline(LORO_THRESHOLD, color="crimson", linestyle="--", linewidth=0.9,
                alpha=0.8, label=f"LORO thr {LORO_THRESHOLD:.3f}")
ax_main.axhline(train_threshold, color="steelblue", linestyle=":", linewidth=0.8,
                alpha=0.7, label=f"Train thr {train_threshold:.3f}")

ax_main.set_ylabel("MWL  P(overload)", fontsize=9)
ax_main.set_ylim(-0.05, 1.10)
ax_main.set_xlim(t_rel[0], t_rel[-1])
ax_main.tick_params(labelsize=7)

handles, labels_l = ax_main.get_legend_handles_labels()
handles.append(Patch(facecolor="green", alpha=0.15, label="assist ON"))
ax_main.legend(handles=handles, fontsize=7, loc="upper right")

dur_min = (t_rel[-1] - t_rel[0]) / 60
ax_main.set_title(
    f"S006 Adaptation — per-run LOW norm, 10-fold CV thr={LORO_THRESHOLD:.3f}  |  "
    f"{dur_min:.1f} min  |  assist ON {pct_on:.0f}%  ({n_on} trigger↑, {n_off} trigger↓)",
    fontsize=10,
)

# ---- Lower subplot: per-block mean p_high bar chart ----
block_labels_plot: list[str] = []
block_p_means:     list[float] = []
block_colours:     list[str] = []

# Include recovered block_01 in bars
all_blocks_for_bar = []
if b01_end_ts is not None:
    t_b01_plot = max(t_eeg_start, b01_end_ts - BLOCK_DUR_S)
    all_blocks_for_bar.append((t_b01_plot, b01_end_ts, "LOW"))
all_blocks_for_bar.extend(adapt_blocks)

for idx, (t_s, t_e, level) in enumerate(all_blocks_for_bar, start=1):
    xs = t_s - t0_lsl; xe = t_e - t0_lsl
    mask = (t_rel >= xs) & (t_rel < xe)
    if not mask.any():
        continue
    block_labels_plot.append(f"B{idx}\n{level[:3]}")
    block_p_means.append(float(ph_arr[mask].mean()))
    block_colours.append(_LEVEL_COLOURS.get(level, "grey"))

x_pos = np.arange(len(block_labels_plot))
ax_drift.bar(x_pos, block_p_means, color=block_colours, alpha=0.7, edgecolor="k", linewidth=0.4)
ax_drift.axhline(LORO_THRESHOLD, color="crimson", linestyle="--", linewidth=0.8, alpha=0.8)
ax_drift.set_xticks(x_pos)
ax_drift.set_xticklabels(block_labels_plot, fontsize=7)
ax_drift.set_ylabel("mean p_high", fontsize=8)
ax_drift.set_xlabel("Block", fontsize=8)
ax_drift.set_ylim(0, 1)
ax_drift.tick_params(labelsize=7)
ax_drift.set_title("Mean P(overload) per block", fontsize=9)

fig.tight_layout()

out_fig = _REPO / "results" / "figures" / "adaptation_s006__perrun_norm_10fold_threshold.png"
out_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_fig, dpi=150)
plt.close(fig)
print(f"  Saved: {out_fig}")
print("\n--- Done ---")
