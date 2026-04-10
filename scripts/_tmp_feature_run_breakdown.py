"""Per-feature breakdown: C1 vs C2, LOW vs HIGH (binary, block_01 recovered).

For each of the 54 features, shows:
  - Mean (Z-normed) for C1-LOW, C1-HIGH, C2-LOW, C2-HIGH
  - Discrimination direction: HIGH_mean - LOW_mean per run
  - Whether the sign agrees or flips between runs
  - SelectKBest f-score per run, and whether the same feature is in the top-40

Goal: diagnose why LORO CV collapses (near-zero J on held-out run).

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_feature_run_breakdown.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyxdf
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from build_mwl_training_dataset import (
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _merge_eeg_streams,
)
from eeg import EegPreprocessor, extract_windows, slice_block
from eeg.extract_features import FIXED_BANDS, _build_region_map, _extract_feat
from sklearn.feature_selection import SelectKBest, f_classif

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE            = 128.0
CAL_K            = 40
BLOCK_DURATION_S = 59

PHYSIO   = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PSELF")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

_ns       = json.loads((MODEL_DIR / "norm_stats.json").read_text())
NORM_MEAN = np.array(_ns["mean"])
NORM_STD  = np.array(_ns["std"])
NORM_STD[NORM_STD < 1e-12] = 1.0

CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf",
]

BLOCK_RE     = re.compile(r"block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")
BLOCK_IDX_RE = re.compile(r"block_(?P<idx>\d+)/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)")

# ---------------------------------------------------------------------------
# Helpers (replicated from _tmp_loro_threshold.py — no module import)
# ---------------------------------------------------------------------------

def _load_eeg(xdf_path: Path):
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


def _parse_blocks(markers):
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


def _recover_block01(markers, eeg_ts):
    for ts, ev in markers:
        hit = BLOCK_IDX_RE.search(ev)
        if hit and int(hit.group("idx")) == 1 and hit.group("ev") == "END":
            level   = hit.group("level")
            t_end   = ts
            t_start = max(eeg_ts[0], t_end - BLOCK_DURATION_S)
            return t_start, t_end, level
    return None


def _windows_in_range(preprocessed, eeg_ts, t_start, t_end):
    i0 = int(np.searchsorted(eeg_ts, t_start))
    i1 = int(np.searchsorted(eeg_ts, t_end))
    block = slice_block(preprocessed, i0, i1, WINDOW_CONFIG)
    return extract_windows(block, WINDOW_CONFIG)


# ---------------------------------------------------------------------------
# Load both runs — raw features (pre-norm) and Z-normed features
# ---------------------------------------------------------------------------

region_map = _build_region_map(FEAT_CFG, CH_NAMES)

# run_data[run_idx][class_label] -> raw feature array (n_win, n_feats)
run_raw:  list[dict[int, np.ndarray]] = []

feat_names: list[str] = []

for xdf_path in CAL_XDFS:
    acq = xdf_path.stem.split("acq-")[1].replace("_physio", "")
    print(f"Loading {acq} ...", flush=True)
    data, ts = _load_eeg(xdf_path)
    pp       = _preprocess(data)
    markers  = _get_markers(xdf_path)
    blocks   = _parse_blocks(markers)
    b01      = _recover_block01(markers, ts)
    if b01 is not None:
        blocks.append(b01)

    class_feats: dict[int, list] = {0: [], 1: []}
    for t_s, t_e, level in blocks:
        if level == "MODERATE":
            continue
        wins = _windows_in_range(pp, ts, t_s, t_e)
        if wins.shape[0] == 0:
            continue
        X, names = _extract_feat(wins, SRATE, region_map)
        if not feat_names:
            feat_names = names
        lbl = 1 if level == "HIGH" else 0
        class_feats[lbl].append(X)
        print(f"  {level:<10}  n={len(X)}")

    run_raw.append({
        lbl: np.concatenate(arrs) for lbl, arrs in class_feats.items() if arrs
    })
    print()

n_feat = len(feat_names)

# ---------------------------------------------------------------------------
# Build per-run, per-class statistics
# ---------------------------------------------------------------------------
# Shape containers: (n_runs=2, n_classes=2, n_features)
means_raw = np.full((2, 2, n_feat), np.nan)
stds_raw  = np.full((2, 2, n_feat), np.nan)

# Also Z-normed versions
means_z = np.full((2, 2, n_feat), np.nan)

for r, class_feats in enumerate(run_raw):
    for c, X in class_feats.items():
        means_raw[r, c] = X.mean(axis=0)
        stds_raw[r, c]  = X.std(axis=0)
        X_z = (X - NORM_MEAN) / NORM_STD
        means_z[r, c]   = X_z.mean(axis=0)

# Discrimination direction: HIGH_mean - LOW_mean (Z-normed)
disc_z = means_z[:, 1, :] - means_z[:, 0, :]  # (2, n_feat)

# Sign agreement: +1 if both runs agree in sign, -1 if they flip
sign_c1 = np.sign(disc_z[0])
sign_c2 = np.sign(disc_z[1])
agrees  = (sign_c1 == sign_c2)                  # (n_feat,)

# Direction-flip magnitude: abs difference in discriminability
disc_diff = np.abs(disc_z[0] - disc_z[1])       # (n_feat,)

# ---------------------------------------------------------------------------
# SelectKBest f-scores per run
# ---------------------------------------------------------------------------
k = CAL_K

def _fscores_for_run(class_feats: dict[int, np.ndarray]) -> np.ndarray:
    X = np.concatenate([class_feats[0], class_feats[1]])
    y = np.concatenate([np.zeros(len(class_feats[0])), np.ones(len(class_feats[1]))])
    X_z = (X - NORM_MEAN) / NORM_STD
    sel = SelectKBest(f_classif, k=k)
    sel.fit(X_z, y)
    return sel.scores_, sel.get_support()

fscores_c1, mask_c1 = _fscores_for_run(run_raw[0])
fscores_c2, mask_c2 = _fscores_for_run(run_raw[1])

overlap_mask = mask_c1 & mask_c2
n_overlap    = overlap_mask.sum()

# ---------------------------------------------------------------------------
# Print: full feature table sorted by direction-flip magnitude
# ---------------------------------------------------------------------------

order = np.argsort(disc_diff)[::-1]   # largest disagreement first

print("=" * 110)
print(f"  PER-FEATURE BREAKDOWN: C1 vs C2 (Z-normed, LOW and HIGH only, block_01 recovered)")
print(f"  n(C1): LOW={len(run_raw[0][0])}  HIGH={len(run_raw[0][1])}")
print(f"  n(C2): LOW={len(run_raw[1][0])}  HIGH={len(run_raw[1][1])}")
print("=" * 110)
print(f"  {'Feature':<25}  "
      f"{'C1_LOW':>8}  {'C1_HI':>8}  {'C1_disc':>8}  "
      f"{'C2_LOW':>8}  {'C2_HI':>8}  {'C2_disc':>8}  "
      f"{'disc_diff':>9}  {'sign':>5}  {'k40_c1':>7}  {'k40_c2':>7}")
print("  " + "-" * 108)

for i in order:
    nm = feat_names[i]
    sign_str = "AGREE" if agrees[i] else "FLIP "
    print(f"  {nm:<25}  "
          f"{means_z[0, 0, i]:8.3f}  {means_z[0, 1, i]:8.3f}  {disc_z[0, i]:8.3f}  "
          f"{means_z[1, 0, i]:8.3f}  {means_z[1, 1, i]:8.3f}  {disc_z[1, i]:8.3f}  "
          f"{disc_diff[i]:9.3f}  {sign_str}  "
          f"{'Y' if mask_c1[i] else '-':>7}  {'Y' if mask_c2[i] else '-':>7}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

n_flip  = int((~agrees).sum())
n_agree = int(agrees.sum())

# Absolute mean disc per run (sign-corrected magnitude)
abs_disc_c1 = np.abs(disc_z[0])
abs_disc_c2 = np.abs(disc_z[1])

# Global offset shift: how much did the overall feature distribution shift
# between runs (both classes combined)?
all_c1 = np.concatenate([run_raw[0][0], run_raw[0][1]])
all_c2 = np.concatenate([run_raw[1][0], run_raw[1][1]])
all_c1_z = (all_c1 - NORM_MEAN) / NORM_STD
all_c2_z = (all_c2 - NORM_MEAN) / NORM_STD
global_shift = all_c2_z.mean(axis=0) - all_c1_z.mean(axis=0)  # (n_feat,)

print()
print("=" * 75)
print("  SUMMARY")
print("=" * 75)
print(f"  Features with same discrimination sign across runs : {n_agree}/{n_feat}")
print(f"  Features with FLIPPED discrimination sign          : {n_flip}/{n_feat}")
print(f"  SelectKBest k=40 overlap (C1 ∩ C2)                : {n_overlap}/{k}")
print()
print(f"  Mean |disc| C1: {abs_disc_c1.mean():.3f}   Max |disc| C1: {abs_disc_c1.max():.3f}")
print(f"  Mean |disc| C2: {abs_disc_c2.mean():.3f}   Max |disc| C2: {abs_disc_c2.max():.3f}")
print()

# Global shift: features with largest absolute mean shift between runs
shift_order = np.argsort(np.abs(global_shift))[::-1]
print(f"  Top 10 features by global distribution shift (C2_mean_z − C1_mean_z):")
print(f"  {'Feature':<25}  {'shift':>8}  {'C1_all_mean_z':>14}  {'C2_all_mean_z':>14}")
print("  " + "-" * 65)
for i in shift_order[:10]:
    print(f"  {feat_names[i]:<25}  {global_shift[i]:8.3f}  "
          f"{all_c1_z[:, i].mean():14.3f}  {all_c2_z[:, i].mean():14.3f}")

print()
print("  Top 10 features by discrimination-direction FLIP magnitude:")
print(f"  {'Feature':<25}  {'C1_disc':>8}  {'C2_disc':>8}  {'disc_diff':>9}  {'k40_c1':>7}  {'k40_c2':>7}")
print("  " + "-" * 70)
for i in order[:10]:
    if not agrees[i]:
        tag = "FLIP"
    else:
        tag = "    "
    print(f"  {feat_names[i]:<25}  {disc_z[0, i]:8.3f}  {disc_z[1, i]:8.3f}  "
          f"{disc_diff[i]:9.3f}  "
          f"{'Y' if mask_c1[i] else '-':>7}  {'Y' if mask_c2[i] else '-':>7}  {tag}")

print()

# Feature-type breakdown of flips
from collections import Counter

type_map = {}
for nm in feat_names:
    if "HjAct" in nm or "HjMob" in nm or "HjComp" in nm:
        t = "Hjorth"
    elif "SpEnt" in nm:
        t = "SpEntropy"
    elif "PeEnt" in nm:
        t = "PeEntropy"
    elif "Skew" in nm or "Kurt" in nm or "ZCR" in nm:
        t = "Stats"
    elif "1fSlope" in nm:
        t = "Aperiodic"
    elif "wPLI" in nm:
        t = "wPLI"
    elif nm in ("FAA", "Cen_Engagement", "FM_Theta_Alpha",
                "FM_Theta_Beta", "Cen_Theta_Beta"):
        t = "Ratio"
    else:
        t = "Bandpower"
    type_map[nm] = t

flip_by_type = Counter(type_map[feat_names[i]] for i in range(n_feat) if not agrees[i])
all_by_type  = Counter(type_map[nm] for nm in feat_names)

print("  Direction flips by feature type:")
print(f"  {'Type':<15}  {'flipped':>8}  {'total':>7}  {'flip%':>7}")
print("  " + "-" * 40)
for ft in sorted(all_by_type):
    total   = all_by_type[ft]
    flipped = flip_by_type.get(ft, 0)
    print(f"  {ft:<15}  {flipped:8d}  {total:7d}  {100*flipped/total:6.0f}%")

print("=" * 75)
