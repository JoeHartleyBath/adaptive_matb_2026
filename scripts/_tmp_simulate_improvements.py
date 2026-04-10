"""Simulate three potential improvements to the MWL pipeline using S005 data.

Simulation 1 — Baseline refresh:
  Re-normalise using features computed from the in-session LOW block
  (block_01/LOW, first 56s of adaptation) as a proxy for what a
  fresh rest recording taken immediately before the task would give.

Simulation 2 — Adaptive z-score:
  Apply a causal exponentially-weighted running mean/std that slowly
  tracks feature drift across the session (τ=120s), simulating online
  normalisation.

Simulation 3 — Feature selection stability:
  Keep only the subset of the original 40 selected features that are
  consistent across both calibration sessions (feature means similar
  in sign/direction for LOW vs HIGH in both cals).

All three compare per-block p_high to the original model.

Run:
    .venv\\Scripts\\Activate.ps1
    python scripts/_tmp_simulate_improvements.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyxdf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG, _merge_eeg_streams  # noqa
from eeg import EegPreprocessor, extract_windows, slice_block                                   # noqa
from eeg.online_features import OnlineFeatureExtractor                                          # noqa
from ml.dataset import LABEL_MAP                                                                # noqa
import yaml

# ---------------------------------------------------------------------------
# Constants (must match calibrate_participant.py)
# ---------------------------------------------------------------------------
SRATE = 128.0
STEP  = int(WINDOW_CONFIG.step_s * SRATE)
WIN   = int(WINDOW_CONFIG.window_s * SRATE)

physio    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")
model_dir = Path(r"C:\data\adaptive_matb\models\PSELF")
adapt_xdf = physio / "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"
cal1_xdf  = physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf"
cal2_xdf  = physio / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf"

meta      = yaml.safe_load(open(REPO / "config" / "eeg_metadata.yaml"))
ch_names  = meta["channel_names"]
feat_cfg  = REPO / "config" / "eeg_feature_extraction.yaml"

ns        = json.load(open(model_dir / "norm_stats.json"))
orig_mean = np.array(ns["mean"])
orig_std  = np.array(ns["std"])

cache = json.load(open(REPO / "results" / "_tmp_new_model_cache" / "meta.json"))
t0_adapt = float(cache["t0"])

# Load retrain results (selector + pipeline from +block_01 model)
# We re-import the retrain script to get the fitted objects.
# To avoid re-running the full script, we serialise from the cache path.
# If those cached objects don't exist yet, we import them via exec.
print("Loading +block_01 model from retrain script ...", flush=True)
import importlib, types, runpy
# Capture sel_full and pip_full without side-effects by monkeypatching sys.exit
import unittest.mock as mock
# Simplest: just re-run the retrain script up to the fit step via direct import
# of its building blocks (same pattern as _tmp_analyse_new_model.py).
# We duplicate a minimal fit here to keep this script self-contained.
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

CAL_K = 40; CAL_C = 1.0; SEED = 42
BLOCK_RE  = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")
B01_RE    = re.compile(r"block_01/(?P<level>LOW|MODERATE|HIGH)/END")
BLOCK_DURATION_S = 59


# ---------------------------------------------------------------------------
# Helper: EEG loading and feature extraction
# ---------------------------------------------------------------------------

def load_eeg(xdf_path):
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
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(eeg_data.shape[0])
    preprocessed = pp.process(eeg_data)
    return preprocessed, eeg_ts


def features_for_window(ext, data):
    """data: (n_ch, WIN_samp) → 1-D feature vector."""
    return ext.compute(data)


def extract_all_features(preprocessed, eeg_ts):
    """Extract feature matrix for every step w/ timestamps."""
    ext = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    n_samp = preprocessed.shape[1]
    ts_list, feat_list = [], []
    for start in range(WIN, n_samp - STEP, STEP):
        window = preprocessed[:, start - WIN: start]
        ts_list.append(eeg_ts[start])
        feat_list.append(ext.compute(window))
    return np.array(ts_list), np.array(feat_list)


def get_markers(xdf_path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    m = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    return list(zip(m["time_stamps"], [r[0] for r in m["time_series"]])) if m else []


def parse_blocks(markers):
    opens = {}; blocks = []
    for ts, ev in markers:
        m = BLOCK_RE.search(ev)
        if not m: continue
        lv, ev_type = m.group("level"), m.group("ev")
        if ev_type == "START":
            opens[lv] = (ts, ev)
        elif ev_type == "END" and lv in opens:
            blocks.append((opens.pop(lv)[0], ts, lv))
    return sorted(blocks, key=lambda b: b[0])


# ---------------------------------------------------------------------------
# Build +block_01 training dataset (minimal re-run)
# ---------------------------------------------------------------------------

def windows_in_range(feats_all, ts_all, t_start, t_end):
    mask = (ts_all >= t_start) & (ts_all <= t_end)
    return feats_all[mask]


print("\nExtracting calibration features ...", flush=True)
all_feats = []; all_labels = []

for xdf_path in (cal1_xdf, cal2_xdf):
    prep, ets = load_eeg(xdf_path)
    ts_all, feats_all = extract_all_features(prep, ets)
    markers = get_markers(xdf_path)

    # Recorded blocks
    for t_s, t_e, lv in parse_blocks(markers):
        w = windows_in_range(feats_all, ts_all, t_s, t_e)
        if len(w):
            all_feats.append(w); all_labels.extend([LABEL_MAP[lv]] * len(w))

    # Recover block_01
    b01_end = next((ts for ts, ev in markers if B01_RE.search(ev)), None)
    if b01_end is not None:
        b01_ev = next(ev for ts, ev in markers if B01_RE.search(ev))
        b01_lv = B01_RE.search(b01_ev).group("level")
        b01_start = b01_end - BLOCK_DURATION_S
        t_avail = max(ets[0], b01_start)
        w = windows_in_range(feats_all, ts_all, t_avail, b01_end)
        if len(w):
            all_feats.append(w); all_labels.extend([LABEL_MAP[b01_lv]] * len(w))

X_train = np.vstack(all_feats)
y_train = np.array(all_labels)
print(f"  Training: {len(X_train)} windows, {np.bincount(y_train)} per class")

# Fit original normalisation + selector
X_z_orig = (X_train - orig_mean) / orig_std
sel = SelectKBest(f_classif, k=CAL_K).fit(X_z_orig, y_train)
sc  = StandardScaler().fit(sel.transform(X_z_orig))
clf = SVC(C=CAL_C, kernel="rbf", probability=True, random_state=SEED)
clf.fit(sc.transform(sel.transform(X_z_orig)), y_train)
pipe_orig = Pipeline([("sc", sc), ("clf", clf)])
selected_idx = sel.get_support(indices=True)
print(f"  Selected feature indices[:10]: {selected_idx[:10]}")


# ---------------------------------------------------------------------------
# Load adaptation session: all features, all timestamps
# ---------------------------------------------------------------------------

print("\nLoading adaptation XDF ...", flush=True)
prep_adapt, ets_adapt = load_eeg(adapt_xdf)
ts_adapt, feats_adapt = extract_all_features(prep_adapt, ets_adapt)
t_rel_adapt = ts_adapt - t0_adapt

markers_adapt = get_markers(adapt_xdf)
blocks_adapt  = parse_blocks(markers_adapt)

# Recover block_01/LOW
b01_end_adapt = next((ts for ts, ev in markers_adapt if B01_RE.search(ev)), None)
if b01_end_adapt is not None:
    b01_ev_a   = next(ev for ts, ev in markers_adapt if B01_RE.search(ev))
    b01_lv_a   = B01_RE.search(b01_ev_a).group("level")
    b01_start_a = b01_end_adapt - BLOCK_DURATION_S
    t_avail_a   = max(ets_adapt[0], b01_start_a)
    blocks_adapt_full = [(t_avail_a, b01_end_adapt, b01_lv_a)] + blocks_adapt
    blocks_adapt_full.sort(key=lambda b: b[0])
else:
    blocks_adapt_full = blocks_adapt

print(f"  {len(ts_adapt)} windows over "
      f"{t_rel_adapt[0]:.1f}s–{t_rel_adapt[-1]:.1f}s")


# ---------------------------------------------------------------------------
# Scoring function: given norm + selector → p_high array
# ---------------------------------------------------------------------------

def score(feats, norm_mean, norm_std, selector, pipeline):
    X_z   = (feats - norm_mean) / norm_std
    X_sel = selector.transform(X_z)
    return pipeline.predict_proba(X_sel)[:, -1]


# ---------------------------------------------------------------------------
# Per-block summary helper
# ---------------------------------------------------------------------------

LEVEL_ORDER = ["LOW", "MODERATE", "HIGH"]

def block_table(ph_vals, t_rel, blocks_full, label):
    print(f"\n  {label}")
    print(f"  {'blk':<4} {'level':<10} {'mean_ph':>8}  {'%>thr':>7}  bar")
    print(f"  {'-'*50}")
    thr = 0.354
    by_level = {lv: [] for lv in LEVEL_ORDER}
    for i, (t_s, t_e, lv) in enumerate(blocks_full, 1):
        t_s_rel = t_s - t0_adapt; t_e_rel = t_e - t0_adapt
        mask = (t_rel >= t_s_rel) & (t_rel <= t_e_rel)
        ph_blk = ph_vals[mask]
        if len(ph_blk) == 0: continue
        m = ph_blk.mean(); pct = (ph_blk > thr).mean() * 100
        bar = "#" * int(pct / 5)
        print(f"  {i:<4} {lv:<10} {m:8.3f}  {pct:6.1f}%  {bar}")
        by_level[lv].append(m)
    print(f"  {'':4} {'---level means---':}")
    for lv in LEVEL_ORDER:
        vals = by_level[lv]
        if vals:
            print(f"  {'':4} {lv:<10} {np.mean(vals):8.3f}  (n={len(vals)} blocks)")


# ===========================================================================
# SIMULATION 1 — Baseline refresh (norm from block_01/LOW features)
# ===========================================================================

print("\n" + "=" * 65)
print("SIMULATION 1: Baseline refresh using block_01/LOW as norm source")
print("=" * 65)

b01_mask = (ts_adapt >= t_avail_a) & (ts_adapt <= b01_end_adapt)
feats_b01 = feats_adapt[b01_mask]
print(f"  block_01/LOW features: {len(feats_b01)} windows for normalisation")

refresh_mean = feats_b01.mean(axis=0)
refresh_std  = feats_b01.std(axis=0).clip(min=1e-6)

# Refit selector on training data with refreshed norm
X_z_refresh = (X_train - refresh_mean) / refresh_std
sel_r = SelectKBest(f_classif, k=CAL_K).fit(X_z_refresh, y_train)
sc_r  = StandardScaler().fit(sel_r.transform(X_z_refresh))
clf_r = SVC(C=CAL_C, kernel="rbf", probability=True, random_state=SEED)
clf_r.fit(sc_r.transform(sel_r.transform(X_z_refresh)), y_train)
pipe_r = Pipeline([("sc", sc_r), ("clf", clf_r)])

ph_orig    = score(feats_adapt, orig_mean,    orig_std,    sel,   pipe_orig)
ph_refresh = score(feats_adapt, refresh_mean, refresh_std, sel_r, pipe_r)

block_table(ph_orig,    t_rel_adapt, blocks_adapt_full, "Original normalisation (rest XDF, ~50min before)")
block_table(ph_refresh, t_rel_adapt, blocks_adapt_full, "Refreshed normalisation (block_01/LOW, in-session)")

print(f"\n  HIGH–LOW mean p_high gap:")
def hlgap(ph, blocks_full):
    lv_vals = {lv: [] for lv in LEVEL_ORDER}
    for t_s, t_e, lv in blocks_full:
        mask = (t_rel_adapt >= t_s - t0_adapt) & (t_rel_adapt <= t_e - t0_adapt)
        if mask.sum(): lv_vals[lv].append(ph[mask].mean())
    return np.mean(lv_vals["HIGH"]) - np.mean(lv_vals["LOW"])

print(f"  Original:  Δ(HIGH-LOW) = {hlgap(ph_orig, blocks_adapt_full):+.3f}")
print(f"  Refreshed: Δ(HIGH-LOW) = {hlgap(ph_refresh, blocks_adapt_full):+.3f}")


# ===========================================================================
# SIMULATION 2 — Adaptive z-score (causal drift correction, τ=120s)
# ===========================================================================

print("\n" + "=" * 65)
print("SIMULATION 2: Causal adaptive z-score (τ=120s drift correction)")
print("=" * 65)

TAU_S = 120.0
alpha_ema = WINDOW_CONFIG.step_s / TAU_S   # ≈ 0.002 per 250ms window

# Initialise running stats from original norm
run_mean = orig_mean.copy()
run_m2   = (orig_std ** 2).copy()   # running variance
ph_adaptive = np.empty(len(feats_adapt))

for i, feat_row in enumerate(feats_adapt):
    # Score with current running stats BEFORE updating (causal)
    rn_std = np.sqrt(run_m2).clip(min=1e-6)
    X_z    = (feat_row - run_mean) / rn_std
    X_sel  = sel.transform(X_z[np.newaxis, :])
    ph_adaptive[i] = pipe_orig.predict_proba(X_sel)[0, -1]

    # Update running mean and variance (EMA)
    run_mean = alpha_ema * feat_row + (1.0 - alpha_ema) * run_mean
    run_m2   = alpha_ema * (feat_row - run_mean) ** 2 + (1.0 - alpha_ema) * run_m2

print(f"  EMA α={alpha_ema:.4f} per window  (τ≈{TAU_S:.0f}s)")
block_table(ph_orig,     t_rel_adapt, blocks_adapt_full, "Original (fixed norm)")
block_table(ph_adaptive, t_rel_adapt, blocks_adapt_full, "Adaptive z-score (τ=120s)")
print(f"\n  HIGH–LOW mean p_high gap:")
print(f"  Original:  Δ(HIGH-LOW) = {hlgap(ph_orig, blocks_adapt_full):+.3f}")
print(f"  Adaptive:  Δ(HIGH-LOW) = {hlgap(ph_adaptive, blocks_adapt_full):+.3f}")


# ===========================================================================
# SIMULATION 3 — Feature selection stability across calibration sessions
# ===========================================================================

print("\n" + "=" * 65)
print("SIMULATION 3: Keep only features stable across both cal sessions")
print("=" * 65)

# Compute per-level mean for each calibration session individually
def per_level_means(xdf_path, recover_b01=True):
    prep, ets = load_eeg(xdf_path)
    ts_a, feats_a = extract_all_features(prep, ets)
    markers = get_markers(xdf_path)
    lv_feats = {lv: [] for lv in LEVEL_ORDER}
    for t_s, t_e, lv in parse_blocks(markers):
        w = windows_in_range(feats_a, ts_a, t_s, t_e)
        if len(w): lv_feats[lv].append(w)
    if recover_b01:
        b01_e = next((ts for ts, ev in markers if B01_RE.search(ev)), None)
        if b01_e is not None:
            ev_txt = next(ev for ts, ev in markers if B01_RE.search(ev))
            lv = B01_RE.search(ev_txt).group("level")
            ta = max(ets[0], b01_e - BLOCK_DURATION_S)
            w = windows_in_range(feats_a, ts_a, ta, b01_e)
            if len(w): lv_feats[lv].append(w)
    return {lv: np.vstack(v).mean(axis=0) for lv, v in lv_feats.items() if v}

print("  Computing per-level means for each calibration session ...")
means1 = per_level_means(cal1_xdf)
means2 = per_level_means(cal2_xdf)

# A feature is "stable" if HIGH-LOW has the same sign in both sessions
high_low_1 = means1["HIGH"] - means1["LOW"]
high_low_2 = means2["HIGH"] - means2["LOW"]
same_sign   = (np.sign(high_low_1) == np.sign(high_low_2))

# Intersect with originally selected features
stable_and_selected = selected_idx[same_sign[selected_idx]]
unstable_and_selected = selected_idx[~same_sign[selected_idx]]
print(f"  Original selected features: {len(selected_idx)}")
print(f"  Of those, consistent across cals: {len(stable_and_selected)}")
print(f"  Of those, inconsistent: {len(unstable_and_selected)}")

# Rebuild model keeping only stable features (manual selector mask)
stable_mask = np.zeros(feats_adapt.shape[1], dtype=bool)
stable_mask[stable_and_selected] = True
n_stable = stable_mask.sum()

# Refit scaler+clf on stable features only
X_stable_train = X_z_orig[:, stable_mask]
sc_st  = StandardScaler().fit(X_stable_train)
clf_st = SVC(C=CAL_C, kernel="rbf", probability=True, random_state=SEED)
clf_st.fit(sc_st.transform(X_stable_train), y_train)

# Score adaptation with stable features
X_z_a       = (feats_adapt - orig_mean) / orig_std
ph_stable   = clf_st.predict_proba(sc_st.transform(X_z_a[:, stable_mask]))[:, -1]
# Also score with only inconsistent features as a contrast
X_z_un      = X_z_orig[:, ~stable_mask & np.isin(np.arange(feats_adapt.shape[1]), selected_idx)]
if X_z_un.shape[1] > 0:
    sc_un  = StandardScaler().fit(X_z_un)
    clf_un = SVC(C=CAL_C, kernel="rbf", probability=True, random_state=SEED)
    clf_un.fit(sc_un.transform(X_z_un), y_train)
    X_z_a_un = X_z_orig[:, stable_mask ^ np.isin(np.arange(feats_adapt.shape[1]), selected_idx)]

block_table(ph_orig,   t_rel_adapt, blocks_adapt_full, "Original (all 40 features)")
block_table(ph_stable, t_rel_adapt, blocks_adapt_full,
            f"Stable-only ({n_stable} features consistent across both cals)")
print(f"\n  HIGH–LOW mean p_high gap:")
print(f"  Original:    Δ(HIGH-LOW) = {hlgap(ph_orig,   blocks_adapt_full):+.3f}")
print(f"  Stable only: Δ(HIGH-LOW) = {hlgap(ph_stable, blocks_adapt_full):+.3f}")

print("\nDone.")
