"""Quick check: top features by F-score across workload levels."""
import json, numpy as np, re, sys
from pathlib import Path
from scipy.stats import f_oneway
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pyxdf, yaml
from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor
from eeg.online_features import OnlineFeatureExtractor
from ml.dataset import LABEL_MAP

SRATE = 128.0
WIN   = int(WINDOW_CONFIG.window_s * SRATE)
STEP  = int(WINDOW_CONFIG.step_s  * SRATE)
BD    = 59
BLOCK_RE = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")
B01_RE   = re.compile(r"block_01/(?P<level>LOW|MODERATE|HIGH)/END")

feat_cfg = Path("config/eeg_feature_extraction.yaml")
meta     = yaml.safe_load(open("config/eeg_metadata.yaml"))
ch_names = meta["channel_names"]
ns       = json.load(open(r"C:\data\adaptive_matb\models\PSELF\norm_stats.json"))
orig_mean = np.array(ns["mean"]); orig_std = np.array(ns["std"])
physio   = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")

# Feature names
ext0 = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
ext0.compute(np.zeros((len(ch_names), WIN)))
feat_names = ext0.feature_names
print(f"Total features: {len(feat_names)}")


def load_eeg(xdf_path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            f = int(round(actual / SRATE)); data, ts = data[:, ::f], ts[::f]
    pp = EegPreprocessor(PREPROCESSING_CONFIG); pp.initialize_filters(data.shape[0])
    return pp.process(data), ts


def extract_feats(prep, ets):
    ext = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    n = prep.shape[1]; ts_l, f_l = [], []
    for s in range(WIN, n - STEP, STEP):
        ts_l.append(ets[s]); f_l.append(ext.compute(prep[:, s - WIN : s]))
    return np.array(ts_l), np.array(f_l)


def get_markers(xdf_path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    m = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    return list(zip(m["time_stamps"], [r[0] for r in m["time_series"]])) if m else []


def parse_blocks(markers):
    opens = {}; blocks = []
    for ts, ev in markers:
        m = BLOCK_RE.search(ev)
        if not m: continue
        lv, et = m.group("level"), m.group("ev")
        if et == "START": opens[lv] = (ts, ev)
        elif et == "END" and lv in opens: blocks.append((opens.pop(lv)[0], ts, lv))
    return sorted(blocks, key=lambda b: b[0])


print("Extracting calibration features ...", flush=True)
Xf = []; yf = []
for xdf in [physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf",
            physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf"]:
    prep, ets = load_eeg(xdf); tsall, feats = extract_feats(prep, ets)
    mk = get_markers(xdf)
    for t_s, t_e, lv in parse_blocks(mk):
        mask = (tsall >= t_s) & (tsall <= t_e)
        w = feats[mask]
        if len(w): Xf.append(w); yf.extend([LABEL_MAP[lv]] * len(w))
    b01e = next((ts for ts, ev in mk if B01_RE.search(ev)), None)
    if b01e:
        b01ev = next(ev for ts, ev in mk if B01_RE.search(ev))
        b01lv = B01_RE.search(b01ev).group("level")
        tav = max(ets[0], b01e - BD)
        mask = (tsall >= tav) & (tsall <= b01e)
        w = feats[mask]
        if len(w): Xf.append(w); yf.extend([LABEL_MAP[b01lv]] * len(w))

X = np.vstack(Xf); y = np.array(yf)
Xz = (X - orig_mean) / orig_std
print(f"X: {X.shape},  classes: {np.bincount(y)}")

# F-scores
sel = SelectKBest(f_classif, k=40).fit(Xz, y)
selected_idx = set(sel.get_support(indices=True))
fscores = sel.scores_
order = np.argsort(fscores)[::-1]

print()
print("Top 25 features by F-score (* = in top-40 SelectKBest):")
print(f"  {'rank':<5} {'F':>8}  {'in40':>5}  feature")
print("  " + "-" * 65)
for rank, i in enumerate(order[:25], 1):
    flag = "  *" if i in selected_idx else "   "
    print(f"  {rank:<5} {fscores[i]:8.1f} {flag}  {feat_names[i]}")

# Also show selected but low-F features (potential noise selectees)
print()
print("Top-40 selected features ranked by F-score:")
sel_ordered = sorted(selected_idx, key=lambda i: fscores[i], reverse=True)
print(f"  {'rank':<5} {'F':>8}  feature")
print("  " + "-" * 55)
for rank, i in enumerate(sel_ordered, 1):
    print(f"  {rank:<5} {fscores[i]:8.1f}  {feat_names[i]}")
