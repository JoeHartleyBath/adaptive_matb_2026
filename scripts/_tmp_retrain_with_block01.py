"""Retrain S005 model including recovered block_01 data and compare metrics.

Block_01 START was not captured in the XDF (LabRecorder subscribed too late).
We recover it by inferring the start timestamp from the known END marker:
    block_01_start_ts = block_01_end_ts - 59s  (old scenario: 1-min blocks)

For cal_c2, block_01 started ~5.8s before EEG recording began, so only
~53s of EEG is available — still enough for several training windows.

Outputs a side-by-side comparison of:
  - n windows per class
  - cross-validated balanced accuracy
  - ROC-AUC
  - Youden J threshold
  - per-class hit rate at threshold
"""
import sys, json, re, joblib, pyxdf
import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor, extract_windows, slice_block
from eeg.online_features import OnlineFeatureExtractor
from ml.dataset import LABEL_MAP
import yaml

# ---------------------------------------------------------------------------
# Constants — match calibrate_participant.py exactly
# ---------------------------------------------------------------------------
SRATE    = 128.0
CAL_K    = 40
CAL_C    = 1.0
SEED     = 42
BLOCK_DURATION_S = 59  # old scenario: END marker fires at t=59s (START at t=0)

model_dir  = Path(r"C:\data\adaptive_matb\models\PSELF")
physio     = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")
rest_xdf   = physio / "sub-PSELF_ses-S005_task-matb_acq-rest_physio.xdf"

meta      = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names  = meta["channel_names"]
feat_cfg  = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

# Load existing norm stats (from rest XDF, fitted during original calibration)
ns        = json.load(open(model_dir / "norm_stats.json"))
norm_mean = np.array(ns["mean"])
norm_std  = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eeg_from_xdf(xdf_path: Path):
    """Return (eeg_data [n_ch, n], eeg_ts) from an XDF, decimated to 128 Hz."""
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


def preprocess(eeg_data):
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(eeg_data.shape[0])
    return pp.process(eeg_data)


def extract_features(epochs):
    """epochs: (n_windows, n_ch, win_samples) → (n_windows, n_feats)."""
    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    return np.array([extractor.compute(w) for w in epochs])


def windows_from_ts_range(preprocessed, eeg_ts, t_start, t_end):
    """Extract windowed epochs from a time range."""
    i0 = int(np.searchsorted(eeg_ts, t_start))
    i1 = int(np.searchsorted(eeg_ts, t_end))
    block = slice_block(preprocessed, i0, i1, WINDOW_CONFIG)
    return extract_windows(block, WINDOW_CONFIG)   # (n_win, n_ch, win_samp)


def get_markers(xdf_path: Path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    m = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    if m is None:
        return []
    return list(zip(m["time_stamps"], [s[0] for s in m["time_series"]]))


BLOCK_RE = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")

def parse_recorded_blocks(markers):
    """Return list of (t_start, t_end, level) for blocks that have both START+END."""
    opens = {}
    blocks = []
    for ts, ev in markers:
        m = BLOCK_RE.search(ev)
        if not m:
            continue
        lv = m.group("level")
        if m.group("ev") == "START":
            opens[lv] = ts
        elif m.group("ev") == "END" and lv in opens:
            blocks.append((opens.pop(lv), ts, lv))
    return blocks


def get_block01_end_ts(markers):
    """Return timestamp of block_01/*/END marker."""
    for ts, ev in markers:
        if re.search(r"block_01/\w+/END", ev):
            return ts
    return None


# ---------------------------------------------------------------------------
# Build epoch datasets
# ---------------------------------------------------------------------------

cal_files = [
    (physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf",
     r"C:\adaptive_matb_2026\experiment\scenarios\full_calibration_pself_c1.txt"),
    (physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf",
     r"C:\adaptive_matb_2026\experiment\scenarios\full_calibration_pself_c2.txt"),
]

epochs_orig   = []   # without block_01
labels_orig   = []
epochs_full   = []   # with block_01 recovered
labels_full   = []

for xdf_path, _ in cal_files:
    acq = xdf_path.stem.split("acq-")[1].replace("_physio", "")
    print(f"\nLoading {acq} ...", flush=True)

    eeg_data, eeg_ts = load_eeg_from_xdf(xdf_path)
    preprocessed = preprocess(eeg_data)
    markers = get_markers(xdf_path)

    # --- Recorded blocks (blocks 02–09, same as original calibration) ---
    rec_blocks = parse_recorded_blocks(markers)
    for t_s, t_e, level in rec_blocks:
        wins = windows_from_ts_range(preprocessed, eeg_ts, t_s, t_e)
        if wins.shape[0] == 0:
            continue
        feats = extract_features(wins)
        lbl   = LABEL_MAP[level]
        epochs_orig.append(feats)
        labels_orig.append(np.full(len(feats), lbl, dtype=np.int64))
        epochs_full.append(feats)
        labels_full.append(np.full(len(feats), lbl, dtype=np.int64))
        print(f"  block recorded  {level:<10} n={len(feats)}")

    # --- Recover block_01 from known END timestamp ---
    b01_end = get_block01_end_ts(markers)
    if b01_end is None:
        print(f"  block_01: no END marker found — skipping recovery")
        continue

    # Determine level from END marker text
    b01_ev = next(ev for ts, ev in markers if re.search(r"block_01/\w+/END", ev))
    b01_level = re.search(r"block_01/(\w+)/END", b01_ev).group(1)
    b01_start = b01_end - BLOCK_DURATION_S   # scenario had END at 0:00:59

    # Clamp to EEG recording range
    t_avail = max(eeg_ts[0], b01_start)
    wins = windows_from_ts_range(preprocessed, eeg_ts, t_avail, b01_end)
    if wins.shape[0] == 0:
        print(f"  block_01 recovery: no windows available")
        continue
    feats = extract_features(wins)
    lbl   = LABEL_MAP[b01_level]
    epochs_full.append(feats)
    labels_full.append(np.full(len(feats), lbl, dtype=np.int64))
    recovered_s = b01_end - t_avail
    print(f"  block_01 RECOVERED {b01_level:<10} n={len(feats)}  ({recovered_s:.0f}s available)")


# Stack
X_orig = np.concatenate(epochs_orig)
y_orig = np.concatenate(labels_orig)
X_full = np.concatenate(epochs_full)
y_full = np.concatenate(labels_full)

print(f"\nDataset sizes:")
print(f"  ORIGINAL  total={len(y_orig)}  LOW={int((y_orig==0).sum())}  MOD={int((y_orig==1).sum())}  HIGH={int((y_orig==2).sum())}")
print(f"  +BLOCK01  total={len(y_full)}  LOW={int((y_full==0).sum())}  MOD={int((y_full==1).sum())}  HIGH={int((y_full==2).sum())}")


# ---------------------------------------------------------------------------
# Load rest-based norm stats (same as original — identical baseline)
# ---------------------------------------------------------------------------

def fit_and_evaluate(X_raw, y, label):
    """Normalise, fit scratch SVM, compute metrics. Returns result dict + fitted objects."""
    X_norm = (X_raw - norm_mean) / norm_std

    # Cross-validated balanced accuracy (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe_cv = Pipeline([
        ("sel", SelectKBest(f_classif, k=min(CAL_K, X_norm.shape[1]))),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                    probability=True, random_state=SEED)),
    ])
    cv_ba = cross_val_score(pipe_cv, X_norm, y,
                            cv=cv, scoring="balanced_accuracy").mean()

    # Full-fit for threshold + AUC
    sel = SelectKBest(f_classif, k=min(CAL_K, X_norm.shape[1]))
    X_sel = sel.fit_transform(X_norm, y)
    sc  = StandardScaler()
    X_sc = sc.fit_transform(X_sel)
    clf = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
              probability=True, random_state=SEED)
    clf.fit(X_sc, y)

    # Build deploy pipeline (matches calibrate_participant.py)
    deploy_pipe = Pipeline([("sc", sc), ("clf", clf)])

    p_high    = clf.predict_proba(X_sc)[:, -1]
    y_binary  = (y == LABEL_MAP["HIGH"]).astype(int)
    auc       = roc_auc_score(y_binary, p_high)
    fpr, tpr, thr_arr = roc_curve(y_binary, p_high)
    j         = tpr - fpr
    best_i    = int(np.argmax(j))
    threshold = float(thr_arr[best_i])
    youdens_j = float(j[best_i])
    ba_train  = balanced_accuracy_score(y, clf.predict(X_sc))

    # Per-class mean p_high at threshold
    class_stats = {}
    for lv, lbl in LABEL_MAP.items():
        mask = y == lbl
        if mask.any():
            ph = p_high[mask]
            class_stats[lv] = (ph.mean(), (ph > threshold).mean() * 100)

    return {
        "label":      label,
        "n":          len(y),
        "cv_ba":      cv_ba,
        "train_ba":   ba_train,
        "auc":        auc,
        "threshold":  threshold,
        "youdens_j":  youdens_j,
        "class_stats": class_stats,
        # fitted objects for downstream inference
        "_selector":  sel,
        "_pipeline":  deploy_pipe,
    }


print("\nFitting ORIGINAL model (blocks 02-09 only) ...")
r_orig = fit_and_evaluate(X_orig, y_orig, "ORIGINAL (no block_01)")

print("Fitting +BLOCK01 model ...")
r_full = fit_and_evaluate(X_full, y_full, "+BLOCK01 recovered")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  COMPARISON")
print("=" * 65)

hdr = f"  {'Metric':<30}  {'ORIGINAL':>12}  {'+BLOCK01':>12}  {'delta':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

rows = [
    ("n windows",       r_orig["n"],          r_full["n"],          ""),
    ("CV balanced acc", r_orig["cv_ba"],       r_full["cv_ba"],      "delta"),
    ("Train bal acc",   r_orig["train_ba"],    r_full["train_ba"],   "delta"),
    ("ROC-AUC",         r_orig["auc"],         r_full["auc"],        "delta"),
    ("Youden J",        r_orig["youdens_j"],   r_full["youdens_j"],  "delta"),
    ("Threshold",       r_orig["threshold"],   r_full["threshold"],  "delta"),
]
for name, v_orig, v_full, mode in rows:
    if mode == "delta":
        delta_s = f"{v_full - v_orig:+.4f}"
        print(f"  {name:<30}  {v_orig:>12.4f}  {v_full:>12.4f}  {delta_s:>8}")
    else:
        print(f"  {name:<30}  {v_orig:>12}  {v_full:>12}")

print()
print("  Per-class mean p_high  (@ each model's own threshold)")
print(f"  {'Level':<10}  {'orig_mean':>10}  {'orig_%>thr':>10}  {'full_mean':>10}  {'full_%>thr':>10}")
print("  " + "-" * 60)
for lv in ["LOW", "MODERATE", "HIGH"]:
    o = r_orig["class_stats"].get(lv, (float("nan"), float("nan")))
    f = r_full["class_stats"].get(lv, (float("nan"), float("nan")))
    print(f"  {lv:<10}  {o[0]:>10.3f}  {o[1]:>9.1f}%  {f[0]:>10.3f}  {f[1]:>9.1f}%")


# ---------------------------------------------------------------------------
# Apply both models to all four XDFs
# ---------------------------------------------------------------------------

def phigh_series_from_xdf(xdf_path, selector, pipeline):
    """Return p_high array for every window in the XDF."""
    eeg_data, eeg_ts = load_eeg_from_xdf(xdf_path)
    preprocessed = preprocess(eeg_data)
    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    step = int(WINDOW_CONFIG.step_s * SRATE)
    win  = int(WINDOW_CONFIG.window_s * SRATE)
    n_samp = preprocessed.shape[1]
    ph_list = []
    for start in range(win, n_samp - step, step):
        window    = preprocessed[:, start - win : start]
        feats     = extractor.compute(window)
        feats_z   = (feats - norm_mean) / norm_std
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        ph_list.append(float(proba[-1]))
    return np.array(ph_list)


def simulate_scheduler(ph_vals, threshold,
                       alpha=0.05, hysteresis=0.02,
                       t_hold_s=3.0, cooldown_s=15.0, step_s=0.25):
    """Run EMA + hold-timer + cooldown state machine. Returns (on_flags, n_events)."""
    ema = None; zone = None; zone_entry = None
    assist_on = False; cooldown_end = 0.0
    on_flags = []
    n_on = 0; n_off = 0
    for i, v in enumerate(ph_vals):
        t = i * step_s
        if ema is None:
            ema = float(v)
        else:
            ema = alpha * float(v) + (1.0 - alpha) * ema
        new_zone = ("above" if ema > threshold + hysteresis else
                    "below" if ema < threshold - hysteresis else "dead")
        if new_zone != zone:
            zone = new_zone; zone_entry = t
        hold_s = t - zone_entry
        if t >= cooldown_end:
            if zone == "above" and hold_s >= t_hold_s and not assist_on:
                assist_on = True; cooldown_end = t + cooldown_s
                zone_entry = t; n_on += 1
            elif zone == "below" and hold_s >= t_hold_s and assist_on:
                assist_on = False; cooldown_end = t + cooldown_s
                zone_entry = t; n_off += 1
        on_flags.append(assist_on)
    return np.array(on_flags), n_on, n_off


acqs = [
    ("cal_c1",     physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf"),
    ("cal_c2",     physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf"),
    ("adaptation", physio / "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"),
    ("control",    physio / "sub-PSELF_ses-S005_task-matb_acq-control_physio.xdf"),
]

thr_orig = r_orig["threshold"]
thr_full = r_full["threshold"]
sel_orig = r_orig["_selector"];  pip_orig = r_orig["_pipeline"]
sel_full = r_full["_selector"];  pip_full = r_full["_pipeline"]

print("\n\n" + "=" * 75)
print("  APPLIED TO ALL FOUR XDFs")
print(f"  ORIGINAL threshold={thr_orig:.4f}   +BLOCK01 threshold={thr_full:.4f}")
print("=" * 75)

header = (
    f"  {'Recording':<13}"
    f"  {'orig_mean':>9}  {'orig_%>thr':>10}  {'orig_assist%':>12}"
    f"  {'new_mean':>9}  {'new_%>thr':>10}  {'new_assist%':>12}"
    f"  {'delta_mean':>10}  {'delta_assist':>12}"
)
print(header)
print("  " + "-" * (len(header) - 2))

for acq, xdf_path in acqs:
    print(f"  {acq:<13} ...", end=" ", flush=True)
    ph_o = phigh_series_from_xdf(xdf_path, sel_orig, pip_orig)
    ph_f = phigh_series_from_xdf(xdf_path, sel_full, pip_full)

    on_o, _, _ = simulate_scheduler(ph_o, thr_orig)
    on_f, _, _ = simulate_scheduler(ph_f, thr_full)

    pct_o  = (ph_o > thr_orig).mean() * 100
    pct_f  = (ph_f > thr_full).mean() * 100
    ast_o  = on_o.mean() * 100
    ast_f  = on_f.mean() * 100

    print(
        f"\r  {acq:<13}"
        f"  {ph_o.mean():9.3f}  {pct_o:9.1f}%  {ast_o:11.1f}%"
        f"  {ph_f.mean():9.3f}  {pct_f:9.1f}%  {ast_f:11.1f}%"
        f"  {ph_f.mean()-ph_o.mean():+10.3f}  {ast_f-ast_o:+11.1f}%"
    )

print()
print(f"  Notes:")
print(f"  - '%>thr' = raw p_high above threshold (no smoother)")
print(f"  - 'assist%' = % of windows where EMA-smoothed p_high would keep assist ON")
print(f"    (EMA α=0.05, hold=3s, cooldown=15s, hysteresis=0.02)")


# ---------------------------------------------------------------------------
# Figure: adaptation timeline with +block_01 model
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_LEVEL_COLOURS = {"HIGH": "tab:red", "MODERATE": "tab:orange", "LOW": "tab:blue"}


def _phigh_with_ts(xdf_path, selector, pipeline):
    """Return (lsl_timestamps, p_high) for every window in the XDF."""
    eeg_data, eeg_ts = load_eeg_from_xdf(xdf_path)
    preprocessed = preprocess(eeg_data)
    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    step = int(WINDOW_CONFIG.step_s * SRATE)
    win  = int(WINDOW_CONFIG.window_s * SRATE)
    n_samp = preprocessed.shape[1]
    ts_list, ph_list = [], []
    for start in range(win, n_samp - step, step):
        window    = preprocessed[:, start - win : start]
        feats     = extractor.compute(window)
        feats_z   = (feats - norm_mean) / norm_std
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        ts_list.append(eeg_ts[start])   # LSL timestamp at window end
        ph_list.append(float(proba[-1]))
    return np.array(ts_list), np.array(ph_list)


def _simulate_full(ph_vals, threshold,
                   alpha=0.05, hysteresis=0.02,
                   t_hold_s=3.0, cooldown_s=15.0, step_s=0.25):
    """Like simulate_scheduler but also returns the EMA trace."""
    ema = None; zone = None; zone_entry = None
    assist_on = False; cooldown_end = 0.0
    on_flags, ema_trace = [], []
    n_on = 0; n_off = 0
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


def _shade_assist(ax, t, assist_on):
    dt = float(np.median(np.diff(t)))
    in_region = False; start = 0.0
    for i in range(len(t)):
        if assist_on[i] and not in_region:
            start = t[i] - dt / 2; in_region = True
        elif not assist_on[i] and in_region:
            ax.axvspan(start, t[i - 1] + dt / 2, color="green", alpha=0.15, zorder=1)
            in_region = False
    if in_region:
        ax.axvspan(start, t[-1] + dt / 2, color="green", alpha=0.15, zorder=1)


adapt_xdf = physio / "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"
print("\nGenerating figure with +block_01 model ...", flush=True)

lsl_ts, ph_new = _phigh_with_ts(adapt_xdf, sel_full, pip_full)
on_new, ema_new, n_on_new, _ = _simulate_full(ph_new, thr_full)

# Scene-relative time: t=0 at first window
t0 = lsl_ts[0]
t_rel = lsl_ts - t0

# Block backgrounds from adaptation XDF markers
adapt_markers = get_markers(adapt_xdf)
adapt_blocks  = parse_recorded_blocks(adapt_markers)   # [(lsl_start, lsl_end, level)]

fig, ax = plt.subplots(figsize=(14, 5))

# Block backgrounds
for blk_ts, blk_te, level in adapt_blocks:
    xs = blk_ts - t0; xe = blk_te - t0
    colour = _LEVEL_COLOURS.get(level, "grey")
    ax.axvspan(xs, xe, color=colour, alpha=0.07, zorder=0)
    ax.text((xs + xe) / 2, 1.02, level,
            ha="center", va="bottom", fontsize=6, color=colour, alpha=0.8,
            transform=ax.get_xaxis_transform())

_shade_assist(ax, t_rel, on_new)

ax.plot(t_rel, ph_new,  color="0.65", alpha=0.5, linewidth=0.5, label="raw P(overload)")
ax.plot(t_rel, ema_new, color="0.2",  linewidth=1.0,            label="smoothed")
ax.axhline(thr_full, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="threshold")

ax.set_xlabel("Scenario time (s)", fontsize=9)
ax.set_ylabel("MWL  P(overload)",  fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(t_rel[0], t_rel[-1])
ax.tick_params(labelsize=7)

handles, labels_leg = ax.get_legend_handles_labels()
handles.append(Patch(facecolor="green", alpha=0.15, label="assist ON"))
ax.legend(handles=handles, fontsize=7, loc="upper right")

pct_on = 100.0 * on_new.sum() / len(on_new)
dur_min = (t_rel[-1] - t_rel[0]) / 60
ax.set_title(
    f"MWL Adaptation Timeline  (+block01 model, thr={thr_full:.3f})  —  "
    f"{dur_min:.1f} min  |  assist ON {pct_on:.0f}% ({on_new.sum()}/{len(on_new)} ticks)",
    fontsize=10,
)
fig.tight_layout()

out_fig = Path(r"C:\adaptive_matb_2026\results\figures") / "adaptation_pself__fig01__mwl_timeline_new_model.png"
out_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_fig, dpi=150)
plt.close(fig)
print(f"Saved: {out_fig}")

# ---------------------------------------------------------------------------
# Cache computed arrays so the plot can be regenerated quickly
# ---------------------------------------------------------------------------
cache_dir = Path(r"C:\adaptive_matb_2026\results") / "_tmp_new_model_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    cache_dir / "adapt_inference.npz",
    ph_new=ph_new,
    ema_new=ema_new,
    on_new=on_new.astype(np.uint8),
    t_rel=t_rel,
    lsl_ts=lsl_ts,
)
json.dump(
    {
        "threshold": float(thr_full),
        "adapt_blocks": [[float(s), float(e), lv] for s, e, lv in adapt_blocks],
        "adapt_xdf": str(adapt_xdf),
        "t0": float(t0),
    },
    open(cache_dir / "meta.json", "w"),
    indent=2,
)
print(f"Cache saved: {cache_dir}")
