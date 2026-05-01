"""Sweep SelectKBest k values on S005 +block01, new 40-feature set.

Uses the cleaned feature set (no Skew/Kurt/ZCR/FAA/FM_Delta) extracted from
cal_c1 + cal_c2 with block_01 recovered. MODERATE dropped → binary LOW/HIGH.

Prints a table of CV-BA, ROC-AUC, Youden J, and threshold for every k from
1 to n_features, and saves a CV-BA vs k figure.
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
# Constants
# ---------------------------------------------------------------------------
SRATE            = 128.0
CAL_C            = 1.0
SEED             = 42
BLOCK_DURATION_S = 59

model_dir = Path(r"C:\data\adaptive_matb\models\PSELF")
physio    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")

meta     = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names = meta["channel_names"]
feat_cfg = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

ns        = json.load(open(model_dir / "norm_stats.json"))
norm_mean = np.array(ns["mean"])
norm_std  = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0

BLOCK_RE = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")

# ---------------------------------------------------------------------------
# I/O helpers (identical to other _tmp_ scripts)
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
# Build dataset
# ---------------------------------------------------------------------------
_extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)

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

    for t_s, t_e, level in parse_recorded_blocks(markers):
        wins = windows_from_ts_range(preprocessed, eeg_ts, t_s, t_e)
        if wins.shape[0] == 0:
            continue
        feats = np.array([_extractor.compute(w) for w in wins])
        all_epochs.append(feats)
        all_labels.append(np.full(len(feats), LABEL_MAP[level], dtype=np.int64))
        print(f"  {level:<10}  n={len(feats)}")

    b01_end = get_block01_end_ts(markers)
    if b01_end is not None:
        b01_ev    = next(ev for ts, ev in markers if re.search(r"block_01/\w+/END", ev))
        b01_level = re.search(r"block_01/(\w+)/END", b01_ev).group(1)
        t_avail   = max(eeg_ts[0], b01_end - BLOCK_DURATION_S)
        wins      = windows_from_ts_range(preprocessed, eeg_ts, t_avail, b01_end)
        if wins.shape[0] > 0:
            feats = np.array([_extractor.compute(w) for w in wins])
            all_epochs.append(feats)
            all_labels.append(np.full(len(feats), LABEL_MAP[b01_level], dtype=np.int64))
            print(f"  {b01_level:<10}  n={len(feats)}  (block_01 recovered, {b01_end - t_avail:.0f}s)")

X_all = np.concatenate(all_epochs)
y_all = np.concatenate(all_labels)

feat_names: list[str] = list(_extractor.feature_names)  # type: ignore[arg-type]
N_FEATS = len(feat_names)
print(f"\nTotal features: {N_FEATS}")

# Drop MODERATE
keep  = y_all != LABEL_MAP["MODERATE"]
X_bin = X_all[keep]
y_bin = y_all[keep]
print(f"Binary dataset: total={len(y_bin)}  "
      f"LOW={int((y_bin==0).sum())}  HIGH={int((y_bin==2).sum())}")

# Normalise using rest-based stats — trim to current feature count if needed
mn = norm_mean[:N_FEATS]
sd = norm_std[:N_FEATS]
X_norm = (X_bin - mn) / sd

# ---------------------------------------------------------------------------
# Sweep k
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

k_values = list(range(1, N_FEATS + 1))
rows: list[dict] = []

print(f"\nSweeping k = 1 … {N_FEATS}  (5-fold CV each) ...", flush=True)

for k in k_values:
    pipe = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                    probability=True, random_state=SEED)),
    ])
    cv_ba = cross_val_score(pipe, X_norm, y_bin,
                            cv=cv, scoring="balanced_accuracy").mean()

    # Full fit for AUC + Youden J + threshold
    from sklearn.feature_selection import SelectKBest as _SKB
    sel   = _SKB(f_classif, k=k).fit(X_norm, y_bin)
    X_sel = sel.transform(X_norm)
    sc    = StandardScaler().fit(X_sel)
    X_sc  = sc.transform(X_sel)
    clf   = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                probability=True, random_state=SEED).fit(X_sc, y_bin)

    p_high   = clf.predict_proba(X_sc)[:, -1]
    y_binary = (y_bin == LABEL_MAP["HIGH"]).astype(int)
    auc      = roc_auc_score(y_binary, p_high)
    fpr, tpr, thr_arr = roc_curve(y_binary, p_high)
    j        = tpr - fpr
    best_i   = int(np.argmax(j))

    selected = [feat_names[i] for i in np.where(sel.get_support())[0]]

    rows.append({
        "k":         k,
        "cv_ba":     cv_ba,
        "auc":       auc,
        "youdens_j": float(j[best_i]),
        "threshold": float(thr_arr[best_i]),
        "selected":  selected,
    })
    print(f"  k={k:>2}  CV-BA={cv_ba:.4f}  AUC={auc:.4f}  J={j[best_i]:.4f}",
          flush=True)

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
best_cvba = max(rows, key=lambda r: r["cv_ba"])
best_auc  = max(rows, key=lambda r: r["auc"])

print("\n" + "=" * 72)
print("  SelectKBest sweep  —  S005 +block01, binary LOW/HIGH, 40 features")
print("=" * 72)
print(f"  {'k':>3}  {'CV-BA':>8}  {'AUC':>7}  {'YoudenJ':>8}  {'Threshold':>10}")
print("  " + "-" * 50)
for r in rows:
    flags = ""
    if r["k"] == best_cvba["k"]:
        flags += " ← best CV-BA"
    if r["k"] == best_auc["k"] and r["k"] != best_cvba["k"]:
        flags += " ← best AUC"
    print(f"  {r['k']:>3}  {r['cv_ba']:>8.4f}  {r['auc']:>7.4f}"
          f"  {r['youdens_j']:>8.4f}  {r['threshold']:>10.4f}{flags}")

print(f"\n  Best CV-BA: k={best_cvba['k']}  ({best_cvba['cv_ba']:.4f})")
print(f"  Best AUC:   k={best_auc['k']}   ({best_auc['auc']:.4f})")
print(f"\n  Features selected at k={best_cvba['k']}:")
for name in best_cvba["selected"]:
    print(f"    {name}")

# ---------------------------------------------------------------------------
# Figure: CV-BA and AUC vs k
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ks     = [r["k"]         for r in rows]
cv_bas = [r["cv_ba"]     for r in rows]
aucs   = [r["auc"]       for r in rows]
js     = [r["youdens_j"] for r in rows]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1 = axes[0]
ax1.plot(ks, cv_bas, "o-", color="steelblue", markersize=4, linewidth=1.2,
         label="CV balanced acc")
ax1.plot(ks, aucs,   "s--", color="darkorange", markersize=4, linewidth=1.0,
         label="ROC-AUC (train)")
ax1.axvline(best_cvba["k"], color="steelblue", linestyle=":", alpha=0.6,
            label=f"best CV-BA (k={best_cvba['k']})")
ax1.axvline(best_auc["k"],  color="darkorange", linestyle=":", alpha=0.6,
            label=f"best AUC (k={best_auc['k']})")
ax1.set_ylabel("Score", fontsize=9)
ax1.set_ylim(0.5, 1.0)
ax1.legend(fontsize=8)
ax1.set_title(
    f"SelectKBest sweep  —  S005 +block01, binary LOW/HIGH  "
    f"({N_FEATS} features in)",
    fontsize=9,
)
ax1.grid(axis="y", alpha=0.3)

ax2 = axes[1]
ax2.plot(ks, js, "^-", color="forestgreen", markersize=4, linewidth=1.2,
         label="Youden J (train)")
ax2.axvline(best_cvba["k"], color="steelblue", linestyle=":", alpha=0.6)
ax2.set_xlabel("k (features selected)", fontsize=9)
ax2.set_ylabel("Youden J", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)
ax2.set_xticks(range(1, N_FEATS + 1, max(1, N_FEATS // 20)))

fig.tight_layout()

out_fig = (Path(r"C:\adaptive_matb_2026\results\figures")
           / "selectk_sweep_s005_block01.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {out_fig}")
