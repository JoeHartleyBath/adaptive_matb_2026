"""Feature ablation test: effect of removing artifact-confounded features.

Conditions (S005 +block01, MODERATE dropped → binary LOW/HIGH):
  FULL      – all ~54 features (baseline)
  -Skew     – drop *_Skew  (×4)
  -Kurt     – drop *_Kurt  (×4)
  -ZCR      – drop *_ZCR   (×4)
  -FAA      – drop FAA     (×1)
  -ALL      – drop all of the above (×13)

Produces per-group comparison table + adaptation timeline figure with one
panel per condition saved to results/figures/.
"""

import json
import re
import sys
import pyxdf
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
# Constants — match calibrate_participant.py / _tmp_retrain_with_block01.py
# ---------------------------------------------------------------------------
SRATE            = 128.0
CAL_K            = 40
CAL_C            = 1.0
SEED             = 42
BLOCK_DURATION_S = 59   # old scenario: END fires at t=59s

model_dir = Path(r"C:\data\adaptive_matb\models\PSELF")
physio    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")

meta      = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names  = meta["channel_names"]
feat_cfg  = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

ns        = json.load(open(model_dir / "norm_stats.json"))
norm_mean = np.array(ns["mean"])
norm_std  = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0

BLOCK_RE = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_eeg_from_xdf(xdf_path: Path):
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


def get_markers(xdf_path: Path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    m = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    if m is None:
        return []
    return list(zip(m["time_stamps"], [s[0] for s in m["time_series"]]))


def parse_recorded_blocks(markers):
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
    for ts, ev in markers:
        if re.search(r"block_01/\w+/END", ev):
            return ts
    return None


def windows_from_ts_range(preprocessed, eeg_ts, t_start, t_end):
    i0 = int(np.searchsorted(eeg_ts, t_start))
    i1 = int(np.searchsorted(eeg_ts, t_end))
    block = slice_block(preprocessed, i0, i1, WINDOW_CONFIG)
    return extract_windows(block, WINDOW_CONFIG)


# ---------------------------------------------------------------------------
# Build feature matrix — block01 included, all three levels initially
# Re-use one extractor instance to keep feature names consistent
# ---------------------------------------------------------------------------

_extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)


def extract_feats(epochs: np.ndarray) -> np.ndarray:
    return np.array([_extractor.compute(w) for w in epochs])


cal_files = [
    physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf",
    physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf",
]

all_epochs: list[np.ndarray] = []
all_labels: list[np.ndarray] = []

for xdf_path in cal_files:
    acq = xdf_path.stem.split("acq-")[1].replace("_physio", "")
    print(f"\nLoading {acq} ...", flush=True)

    eeg_data, eeg_ts = load_eeg_from_xdf(xdf_path)
    preprocessed     = preprocess(eeg_data)
    markers          = get_markers(xdf_path)

    # Recorded blocks (02–09)
    for t_s, t_e, level in parse_recorded_blocks(markers):
        wins = windows_from_ts_range(preprocessed, eeg_ts, t_s, t_e)
        if wins.shape[0] == 0:
            continue
        feats = extract_feats(wins)
        all_epochs.append(feats)
        all_labels.append(np.full(len(feats), LABEL_MAP[level], dtype=np.int64))
        print(f"  recorded    {level:<10}  n={len(feats)}")

    # Recover block_01 from END timestamp
    b01_end = get_block01_end_ts(markers)
    if b01_end is None:
        print("  block_01: no END marker — skipping")
        continue
    b01_ev    = next(ev for ts, ev in markers if re.search(r"block_01/\w+/END", ev))
    b01_level = re.search(r"block_01/(\w+)/END", b01_ev).group(1)
    b01_start = b01_end - BLOCK_DURATION_S
    t_avail   = max(eeg_ts[0], b01_start)
    wins      = windows_from_ts_range(preprocessed, eeg_ts, t_avail, b01_end)
    if wins.shape[0] > 0:
        feats = extract_feats(wins)
        all_epochs.append(feats)
        all_labels.append(np.full(len(feats), LABEL_MAP[b01_level], dtype=np.int64))
        print(f"  block_01 RECOVERED {b01_level:<10}  n={len(feats)}  ({b01_end - t_avail:.0f}s)")

X_all = np.concatenate(all_epochs)
y_all = np.concatenate(all_labels)

# Feature names (available after first compute())
feat_names: list[str] = list(_extractor.feature_names)  # type: ignore[arg-type]
print(f"\nFeatures: {len(feat_names)}")

# ---------------------------------------------------------------------------
# Drop MODERATE → binary LOW / HIGH
# ---------------------------------------------------------------------------
keep  = y_all != LABEL_MAP["MODERATE"]
X_bin = X_all[keep]
y_bin = y_all[keep]

n_low  = int((y_bin == LABEL_MAP["LOW"]).sum())
n_high = int((y_bin == LABEL_MAP["HIGH"]).sum())
print(f"Binary dataset (MODERATE dropped): total={len(y_bin)}  LOW={n_low}  HIGH={n_high}")

# ---------------------------------------------------------------------------
# Feature masks — one per ablation group
# ---------------------------------------------------------------------------
def _mask_drop(feat_names: list[str], drop: list[str]) -> np.ndarray:
    """Boolean mask: True = keep."""
    drop_set = set(drop)
    return np.array([f not in drop_set for f in feat_names])

skew_feats = [f for f in feat_names if f.endswith("_Skew")]
kurt_feats = [f for f in feat_names if f.endswith("_Kurt")]
zcr_feats  = [f for f in feat_names if f.endswith("_ZCR")]
faa_feats  = [f for f in feat_names if f == "FAA"]
all_feats  = skew_feats + kurt_feats + zcr_feats + faa_feats

print(f"\nAblation groups:")
print(f"  Skew : {skew_feats}")
print(f"  Kurt : {kurt_feats}")
print(f"  ZCR  : {zcr_feats}")
print(f"  FAA  : {faa_feats}")
print(f"  ALL  : {len(all_feats)} features")

CONDITIONS: list[tuple[str, np.ndarray]] = [
    ("FULL",  np.ones(len(feat_names), dtype=bool)),
    ("-Skew", _mask_drop(feat_names, skew_feats)),
    ("-Kurt", _mask_drop(feat_names, kurt_feats)),
    ("-ZCR",  _mask_drop(feat_names, zcr_feats)),
    ("-FAA",  _mask_drop(feat_names, faa_feats)),
    ("-ALL",  _mask_drop(feat_names, all_feats)),
]


# ---------------------------------------------------------------------------
# Fit and evaluate
# ---------------------------------------------------------------------------

def fit_and_evaluate(X_raw: np.ndarray, y: np.ndarray,
                     feat_mask: np.ndarray, label: str) -> dict:
    X_sub  = X_raw[:, feat_mask]
    X_norm = (X_sub - norm_mean[feat_mask]) / norm_std[feat_mask]

    # 5-fold CV balanced accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe_cv = Pipeline([
        ("sel", SelectKBest(f_classif, k=min(CAL_K, X_norm.shape[1]))),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                    probability=True, random_state=SEED)),
    ])
    cv_ba = cross_val_score(pipe_cv, X_norm, y,
                            cv=cv, scoring="balanced_accuracy").mean()

    # Full fit for threshold derivation
    sel    = SelectKBest(f_classif, k=min(CAL_K, X_norm.shape[1]))
    X_sel  = sel.fit_transform(X_norm, y)
    sc     = StandardScaler()
    X_sc   = sc.fit_transform(X_sel)
    clf    = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                 probability=True, random_state=SEED)
    clf.fit(X_sc, y)

    deploy_pipe = Pipeline([("sc", sc), ("clf", clf)])

    # Metrics — P(HIGH) = last class probability
    p_high   = clf.predict_proba(X_sc)[:, -1]
    y_binary = (y == LABEL_MAP["HIGH"]).astype(int)
    auc      = roc_auc_score(y_binary, p_high)
    fpr, tpr, thr_arr = roc_curve(y_binary, p_high)
    j        = tpr - fpr
    best_i   = int(np.argmax(j))
    threshold = float(thr_arr[best_i])
    youdens_j = float(j[best_i])
    ba_train  = balanced_accuracy_score(y, clf.predict(X_sc))

    class_stats: dict[str, tuple[float, float]] = {}
    for lv, lbl in LABEL_MAP.items():
        mask = y == lbl
        if mask.any():
            ph = p_high[mask]
            class_stats[lv] = (float(ph.mean()), float((ph > threshold).mean() * 100))

    sub_names     = [n for n, k in zip(feat_names, feat_mask) if k]
    selected_names = [sub_names[i] for i in np.where(sel.get_support())[0]]

    return {
        "label":           label,
        "n":               len(y),
        "n_feats_in":      int(feat_mask.sum()),
        "n_feats_sel":     len(selected_names),
        "selected_names":  selected_names,
        "cv_ba":           cv_ba,
        "train_ba":        ba_train,
        "auc":             auc,
        "threshold":       threshold,
        "youdens_j":       youdens_j,
        "class_stats":     class_stats,
        "_feat_mask":      feat_mask,
        "_selector":       sel,
        "_pipeline":       deploy_pipe,
    }


results: dict[str, dict] = {}
for cond_name, mask in CONDITIONS:
    print(f"Fitting {cond_name} ({int(mask.sum())} features) ...", flush=True)
    results[cond_name] = fit_and_evaluate(X_bin, y_bin, mask, cond_name)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

cond_names = [c for c, _ in CONDITIONS]
r_full     = results["FULL"]

print("\n" + "=" * 100)
print("  FEATURE ABLATION  —  S005 +block01, binary LOW/HIGH")
print("=" * 100)

col_w = 11
hdr_cols = "".join(f"  {c:>{col_w}}" for c in cond_names)
hdr_delta = "".join(f"  {'Δ'+c:>{col_w}}" for c in cond_names[1:])
print(f"  {'Metric':<24}{hdr_cols}{hdr_delta}")
print("  " + "-" * (24 + (col_w + 2) * len(cond_names) + (col_w + 2) * (len(cond_names) - 1)))

def _row(metric_key: str, label: str, fmt: str = ".4f") -> None:
    vals = [results[c][metric_key] for c in cond_names]
    line = f"  {label:<24}"
    for v in vals:
        line += f"  {v:{col_w}{fmt}}"
    for v in vals[1:]:
        delta = v - vals[0]
        line += f"  {delta:>+{col_w}.4f}"
    print(line)

def _row_int(metric_key: str, label: str) -> None:
    vals = [results[c][metric_key] for c in cond_names]
    line = f"  {label:<24}"
    for v in vals:
        line += f"  {v:>{col_w}}"
    line += "  " + " " * ((col_w + 2) * (len(cond_names) - 1) - 2)
    print(line)

_row_int("n",           "n windows")
_row_int("n_feats_in",  "features in")
_row_int("n_feats_sel", "features selected")
_row("cv_ba",      "CV balanced acc")
_row("train_ba",   "Train bal acc")
_row("auc",        "ROC-AUC")
_row("youdens_j",  "Youden J")
_row("threshold",  "Threshold")

print()
print("  Per-class train mean p_high  (@ each model's own threshold)")
print(f"  {'Level':<10}" + "".join(f"  {c+'_mean':>{col_w}}  {c+'_%>thr':>{col_w}}" for c in cond_names))
print("  " + "-" * (10 + (col_w * 2 + 4) * len(cond_names)))
for lv in ["LOW", "HIGH"]:
    line = f"  {lv:<10}"
    for c in cond_names:
        s = results[c]["class_stats"].get(lv, (float("nan"), float("nan")))
        line += f"  {s[0]:>{col_w}.3f}  {s[1]:>{col_w-1}.1f}%"
    print(line)

print()
for c in cond_names:
    print(f"  {c:<8} selected ({results[c]['n_feats_sel']}): {results[c]['selected_names']}")


# ---------------------------------------------------------------------------
# Apply to all four XDFs — mean p_high, raw threshold hits, adaptation %
# ---------------------------------------------------------------------------

def phigh_series(xdf_path: Path, result: dict) -> np.ndarray:
    feat_mask = result["_feat_mask"]
    selector  = result["_selector"]
    pipeline  = result["_pipeline"]

    eeg_data, eeg_ts = load_eeg_from_xdf(xdf_path)
    preprocessed     = preprocess(eeg_data)
    step = int(WINDOW_CONFIG.step_s * SRATE)
    win  = int(WINDOW_CONFIG.window_s * SRATE)
    n_samp = preprocessed.shape[1]
    ph_list = []
    for start in range(win, n_samp - step, step):
        window    = preprocessed[:, start - win : start]
        feats     = _extractor.compute(window)
        feats_sub = feats[feat_mask]
        feats_z   = (feats_sub - norm_mean[feat_mask]) / norm_std[feat_mask]
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        ph_list.append(float(proba[-1]))
    return np.array(ph_list)


def simulate_scheduler(ph_vals: np.ndarray, threshold: float,
                       alpha: float = 0.05, hysteresis: float = 0.02,
                       t_hold_s: float = 3.0, cooldown_s: float = 15.0,
                       step_s: float = 0.25) -> tuple[np.ndarray, int, int]:
    ema = None
    zone: str | None = None
    zone_entry = 0.0
    assist_on = False
    cooldown_end = 0.0
    on_flags: list[bool] = []
    n_on = n_off = 0
    for i, v in enumerate(ph_vals):
        t = i * step_s
        ema = float(v) if ema is None else alpha * float(v) + (1.0 - alpha) * ema
        new_zone = ("above" if ema > threshold + hysteresis else
                    "below" if ema < threshold - hysteresis else "dead")
        if new_zone != zone:
            zone = new_zone
            zone_entry = t
        hold_s = t - zone_entry
        if t >= cooldown_end:
            if zone == "above" and hold_s >= t_hold_s and not assist_on:
                assist_on = True; cooldown_end = t + cooldown_s; zone_entry = t; n_on += 1
            elif zone == "below" and hold_s >= t_hold_s and assist_on:
                assist_on = False; cooldown_end = t + cooldown_s; zone_entry = t; n_off += 1
        on_flags.append(assist_on)
    return np.array(on_flags), n_on, n_off


acqs = [
    ("cal_c1",     physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf"),
    ("cal_c2",     physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf"),
    ("adaptation", physio / "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"),
    ("control",    physio / "sub-PSELF_ses-S005_task-matb_acq-control_physio.xdf"),
]

print("\n\n" + "=" * 110)
print("  APPLIED TO ALL FOUR XDFs")
print("  thresholds: " + "   ".join(f"{c}={results[c]['threshold']:.4f}" for c in cond_names))
print("=" * 110)

# Header: Recording | cond1_mean cond1_ast% | cond2_mean cond2_ast% | ...
_cw = 7
hdr_xdf = f"  {'Recording':<13}"
for c in cond_names:
    hdr_xdf += f"  {(c+'_mn'):>{_cw}}  {(c+'_ast%'):>{_cw+1}}"
print(hdr_xdf)
print("  " + "-" * (13 + ((_cw + _cw + 5) * len(cond_names)) + 2))

for acq, xdf_path in acqs:
    print(f"  {acq:<13} ...", end=" ", flush=True)
    row_data: dict[str, tuple[float, float]] = {}
    for c in cond_names:
        ph  = phigh_series(xdf_path, results[c])
        on, _, _ = simulate_scheduler(ph, results[c]["threshold"])
        row_data[c] = (float(ph.mean()), float(on.mean() * 100))
    line = f"\r  {acq:<13}"
    for c in cond_names:
        mn, ast = row_data[c]
        line += f"  {mn:>{_cw}.3f}  {ast:>{_cw}.1f}%"
    print(line)

print()
print("  Notes:")
print("  - MODERATE excluded from training; inference runs over all windows.")
print("  - 'ast%' = % windows where EMA scheduler keeps assist ON (α=0.05, hold=3s, cooldown=15s)")


# ---------------------------------------------------------------------------
# Figure: two-panel adaptation timeline (FULL top, CLEAN bottom)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_LEVEL_COLOURS = {"HIGH": "tab:red", "MODERATE": "tab:orange", "LOW": "tab:blue"}


def _phigh_with_ts(xdf_path: Path, result: dict):
    """Return (relative_time, lsl_t0, p_high, ema, on_flags).

    lsl_t0 is the absolute LSL timestamp of the first window end, used to
    convert block marker timestamps to the same scene-relative axis.
    """
    feat_mask = result["_feat_mask"]
    selector  = result["_selector"]
    pipeline  = result["_pipeline"]
    threshold = result["threshold"]

    eeg_data, eeg_ts = load_eeg_from_xdf(xdf_path)
    preprocessed     = preprocess(eeg_data)
    step   = int(WINDOW_CONFIG.step_s * SRATE)
    win    = int(WINDOW_CONFIG.window_s * SRATE)
    n_samp = preprocessed.shape[1]
    ts_list: list[float] = []
    ph_list: list[float] = []
    for start in range(win, n_samp - step, step):
        window    = preprocessed[:, start - win : start]
        feats     = _extractor.compute(window)
        feats_sub = feats[feat_mask]
        feats_z   = (feats_sub - norm_mean[feat_mask]) / norm_std[feat_mask]
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        ts_list.append(float(eeg_ts[start]))
        ph_list.append(float(proba[-1]))

    ph_arr  = np.array(ph_list)
    lsl_t0  = ts_list[0]
    t_rel   = np.array(ts_list) - lsl_t0
    on_arr, ema_arr = _simulate_with_ema(ph_arr, threshold)
    return t_rel, lsl_t0, ph_arr, ema_arr, on_arr


def _simulate_with_ema(ph_vals: np.ndarray, threshold: float,
                       alpha: float = 0.05, hysteresis: float = 0.02,
                       t_hold_s: float = 3.0, cooldown_s: float = 15.0,
                       step_s: float = 0.25):
    ema = None
    zone: str | None = None
    zone_entry = 0.0
    assist_on = False
    cooldown_end = 0.0
    on_flags: list[bool] = []
    ema_trace: list[float] = []
    for i, v in enumerate(ph_vals):
        t = i * step_s
        ema = float(v) if ema is None else alpha * float(v) + (1.0 - alpha) * ema
        new_zone = ("above" if ema > threshold + hysteresis else
                    "below" if ema < threshold - hysteresis else "dead")
        if new_zone != zone:
            zone = new_zone
            zone_entry = t
        hold_s = t - zone_entry
        if t >= cooldown_end:
            if zone == "above" and hold_s >= t_hold_s and not assist_on:
                assist_on = True; cooldown_end = t + cooldown_s; zone_entry = t
            elif zone == "below" and hold_s >= t_hold_s and assist_on:
                assist_on = False; cooldown_end = t + cooldown_s; zone_entry = t
        on_flags.append(assist_on)
        ema_trace.append(ema)
    return np.array(on_flags), np.array(ema_trace)


def _shade_assist(ax, t_rel, on_flags):
    dt = float(np.median(np.diff(t_rel)))
    in_region = False
    start_t = 0.0
    for i in range(len(t_rel)):
        if on_flags[i] and not in_region:
            start_t = t_rel[i] - dt / 2
            in_region = True
        elif not on_flags[i] and in_region:
            ax.axvspan(start_t, t_rel[i - 1] + dt / 2,
                       color="green", alpha=0.15, zorder=1)
            in_region = False
    if in_region:
        ax.axvspan(start_t, t_rel[-1] + dt / 2,
                   color="green", alpha=0.15, zorder=1)


adapt_xdf = physio / "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"
print("\nGenerating figure ...", flush=True)

adapt_markers = get_markers(adapt_xdf)
adapt_blocks  = parse_recorded_blocks(adapt_markers)

# Compute inference for every condition; cache lsl_t0 from the first one.
fig_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
lsl_t0_ref: float | None = None
for c in cond_names:
    t_rel, lsl_t0, ph, ema, on = _phigh_with_ts(adapt_xdf, results[c])
    fig_data[c] = (t_rel, ph, ema, on)
    if lsl_t0_ref is None:
        lsl_t0_ref = lsl_t0
assert lsl_t0_ref is not None

_PANEL_COLOURS = [
    "steelblue", "darkorange", "forestgreen", "purple", "saddlebrown", "crimson",
]

n_panels = len(cond_names)
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
if n_panels == 1:
    axes = [axes]

for ax, c, colour in zip(axes, cond_names, _PANEL_COLOURS):
    t_rel, ph, ema, on = fig_data[c]
    thr    = results[c]["threshold"]
    pct_on = 100.0 * on.sum() / len(on)

    _shade_assist(ax, t_rel, on)
    ax.plot(t_rel, ph,  color="0.70", alpha=0.45, linewidth=0.5, label="raw P(overload)")
    ax.plot(t_rel, ema, color=colour, linewidth=1.0, label="smoothed")
    ax.axhline(thr, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="threshold")
    ax.set_ylabel("P(overload)", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=7)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="green", alpha=0.15, label="assist ON"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_title(
        f"{c}  (thr={thr:.3f}, CV-BA={results[c]['cv_ba']:.3f}, "
        f"AUC={results[c]['auc']:.3f}, feats={results[c]['n_feats_in']})  "
        f"—  assist ON {pct_on:.0f}%",
        fontsize=8,
    )

# Block-level background shading — markers share LSL clock; offset = lsl_t0_ref.
for blk_ts, blk_te, level in adapt_blocks:
    xs = blk_ts - lsl_t0_ref
    xe = blk_te - lsl_t0_ref
    col = _LEVEL_COLOURS.get(level, "grey")
    for ax_p in axes:
        ax_p.axvspan(xs, xe, color=col, alpha=0.06, zorder=0)
    axes[0].text((xs + xe) / 2, 1.02, level,
                 ha="center", va="bottom", fontsize=6, color=col, alpha=0.8,
                 transform=axes[0].get_xaxis_transform())

axes[-1].set_xlabel("Scenario time (s)", fontsize=9)
fig.suptitle(
    "Feature Ablation — Adaptation Timeline  (S005 +block01, binary LOW/HIGH)",
    fontsize=10, y=1.01,
)
fig.tight_layout()

out_fig = (Path(r"C:\adaptive_matb_2026\results\figures")
           / "adaptation_pself__fig02__feature_ablation.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_fig}")
