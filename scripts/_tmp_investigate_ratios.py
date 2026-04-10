"""Investigate why classic θ/α ratios have low F-scores vs Hjorth complexity."""
import json, re, sys, numpy as np
from pathlib import Path
from scipy.stats import f_oneway

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pyxdf, yaml
from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor
from eeg.online_features import OnlineFeatureExtractor
from ml.dataset import LABEL_MAP

SRATE = 128.0
WIN   = int(WINDOW_CONFIG.window_s * SRATE)
STEP  = int(WINDOW_CONFIG.step_s * SRATE)
BD    = 59
BLOCK_RE = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")
B01_RE   = re.compile(r"block_01/(?P<level>LOW|MODERATE|HIGH)/END")

meta     = yaml.safe_load(open("config/eeg_metadata.yaml"))
ch_names = meta["channel_names"]
feat_cfg = Path("config/eeg_feature_extraction.yaml")
physio   = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")


def load_eeg(p):
    streams, _ = pyxdf.load_xdf(str(p))
    eeg = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            f = int(round(actual / SRATE)); data, ts = data[:, ::f], ts[::f]
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    return pp.process(data), ts


def extract_feats(prep, ets):
    ext = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    n = prep.shape[1]; tl, fl = [], []
    for s in range(WIN, n - STEP, STEP):
        tl.append(ets[s]); fl.append(ext.compute(prep[:, s - WIN : s]))
    # get names after first pass
    ext2 = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    ext2.compute(np.zeros((len(ch_names), WIN)))
    return np.array(tl), np.array(fl), ext2.feature_names


def get_markers(p):
    streams, _ = pyxdf.load_xdf(str(p))
    m = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    return list(zip(m["time_stamps"], [r[0] for r in m["time_series"]])) if m else []


def parse_blocks(mk):
    opens = {}; blocks = []
    for ts, ev in mk:
        m = BLOCK_RE.search(ev)
        if not m: continue
        lv, et = m.group("level"), m.group("ev")
        if et == "START": opens[lv] = (ts, ev)
        elif et == "END" and lv in opens: blocks.append((opens.pop(lv)[0], ts, lv))
    return sorted(blocks, key=lambda b: b[0])


print("Loading calibration data ...", flush=True)
Xf = []; yf = []; names = None
for xdf in [physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf",
            physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf"]:
    prep, ets = load_eeg(xdf)
    tsall, feats, names = extract_feats(prep, ets)
    mk = get_markers(xdf)
    for t_s, t_e, lv in parse_blocks(mk):
        mask = (tsall >= t_s) & (tsall <= t_e); w = feats[mask]
        if len(w): Xf.append(w); yf.extend([LABEL_MAP[lv]] * len(w))
    b01e = next((ts for ts, ev in mk if B01_RE.search(ev)), None)
    if b01e:
        b01ev = next(ev for ts, ev in mk if B01_RE.search(ev))
        b01lv = B01_RE.search(b01ev).group("level")
        tav = max(ets[0], b01e - BD)
        mask = (tsall >= tav) & (tsall <= b01e); w = feats[mask]
        if len(w): Xf.append(w); yf.extend([LABEL_MAP[b01lv]] * len(w))

X = np.vstack(Xf); y = np.array(yf)
LEVELS = ["LOW", "MODERATE", "HIGH"]

print(f"  {len(X)} windows, classes: {np.bincount(y)}")
print()

# ------------------------------------------------------------------
# Table 1: per-level means + F-score for key features
# ------------------------------------------------------------------
focus = [
    "FM_Theta", "FM_Alpha", "Par_Alpha", "Occ_Alpha",
    "FM_Theta_Alpha", "FM_Theta_Beta", "Cen_Theta_Beta", "Cen_Engagement",
    "Cen_HjComp", "Occ_HjComp", "Cen_HjAct", "Occ_HjAct",
]
print("Per-level means (raw feature value, not z-scored):")
print(f"  {'feature':<22}  {'F':>7}    LOW      MOD     HIGH    delta(H-L)")
print("  " + "-" * 72)
for fn in focus:
    if fn not in names:
        print(f"  {fn}: NOT FOUND"); continue
    i = names.index(fn)
    col = X[:, i]
    means = {lv: col[y == LABEL_MAP[lv]].mean() for lv in LEVELS}
    stds  = {lv: col[y == LABEL_MAP[lv]].std()  for lv in LEVELS}
    fstat = f_oneway(*[col[y == LABEL_MAP[lv]] for lv in LEVELS]).statistic
    delta = means["HIGH"] - means["LOW"]
    print(f"  {fn:<22}  {fstat:7.1f}  {means['LOW']:7.3f}  {means['MODERATE']:7.3f}  {means['HIGH']:7.3f}  {delta:+.3f}")

# ------------------------------------------------------------------
# Table 2: within-class variance vs between-class variance for ratios
# (explains WHY F is low even if direction is correct)
# ------------------------------------------------------------------
print()
print("Signal-to-noise (between / within std) for ratio features:")
print(f"  {'feature':<22}  {'between_std':>12}  {'within_std':>11}  {'SNR':>7}")
print("  " + "-" * 58)
ratio_feats = ["FM_Theta_Alpha", "FM_Theta_Beta", "Cen_Theta_Beta", "Cen_Engagement",
               "Cen_HjComp", "Occ_HjComp"]
for fn in ratio_feats:
    if fn not in names: continue
    i = names.index(fn)
    col = X[:, i]
    class_means = np.array([col[y == LABEL_MAP[lv]].mean() for lv in LEVELS])
    between = class_means.std()
    within  = np.sqrt(np.mean([col[y == LABEL_MAP[lv]].var() for lv in LEVELS]))
    print(f"  {fn:<22}  {between:12.4f}  {within:11.4f}  {between/within:7.3f}")

# ------------------------------------------------------------------
# Table 3: All F-scores — full ranking confirms position of ratios
# ------------------------------------------------------------------
print()
fscores = np.array([f_oneway(*[X[:, i][y == LABEL_MAP[lv]] for lv in LEVELS]).statistic
                    for i in range(X.shape[1])])
order = np.argsort(fscores)[::-1]
print("Full feature ranking (all 54):")
print(f"  {'rank':<5} {'F':>8}  feature")
print("  " + "-" * 42)
for rank, i in enumerate(order, 1):
    marker = " <-- RATIO/INDEX" if names[i] in ["FM_Theta_Alpha","FM_Theta_Beta","Cen_Theta_Beta","Cen_Engagement","FAA"] else ""
    print(f"  {rank:<5} {fscores[i]:8.1f}  {names[i]}{marker}")
