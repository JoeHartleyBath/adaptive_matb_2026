"""Block-level feature analysis: C1 vs C2 for PSELF S006.

For each feature and each workload level (LOW / MODERATE / HIGH):
  - Per-block mean Z-normed value (each run has 3 blocks per level)
  - Cross-run level mean and std
  - Mann-Whitney U test: are C1 and C2 windows drawn from the same distribution?
  - Effect size (rank-biserial r)

Also shows per-block time-course for the 13 sign-flipping features to
distinguish between-run shift from within-run temporal drift.

Run:
    $env:MATB_SCENARIO_OFFSET_S = "0.943"
    .venv\\Scripts\\Activate.ps1
    python scripts/_tmp_block_level_features_s006.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import mannwhitneyu

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import calibrate_participant as _cal_mod
from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG
from eeg.extract_features import _build_region_map, _extract_feat
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE   = 128.0
CAL_K   = 35
FDR_Q   = 0.05    # significance threshold after Benjamini-Hochberg correction
N_TOP   = 13      # how many features to show per-block time-course for

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PSELF")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

_ns       = json.loads((MODEL_DIR / "norm_stats.json").read_text())
NORM_MEAN = np.array(_ns["mean"])
NORM_STD  = np.array(_ns["std"])
NORM_STD[NORM_STD < 1e-12] = 1.0

CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf",
]

LEVEL_NAMES = {LABEL_MAP["LOW"]: "LOW", LABEL_MAP.get("MODERATE", 1): "MOD", LABEL_MAP["HIGH"]: "HIGH"}
LEVELS = [LABEL_MAP["LOW"], LABEL_MAP.get("MODERATE", 1), LABEL_MAP["HIGH"]]

# Known sign-flipping features from diagnosis (same with corrected offset)
FLIP_FEATURES = {"FM_Beta", "Occ_SpEnt", "Par_HjAct", "Cen_Beta", "Cen_SpEnt",
                 "Par_SpEnt", "Cen_Engagement", "Fro_HjMob", "wPLI_Fro_Par_Th",
                 "FM_Theta_Alpha", "wPLI_Fro_Cen_Th", "wPLI_Fro_Occ_Al", "Occ_PeEnt"}

# ---------------------------------------------------------------------------
# Load — using _load_xdf_block exactly as calibrate.py does
# ---------------------------------------------------------------------------
region_map = _build_region_map(FEAT_CFG, CH_NAMES)
feat_names: list[str] = []

# run_blocks[run_idx] = list of (X_norm: ndarray, level_int: int) — one per block
run_blocks: list[list[tuple[np.ndarray, int]]] = []

for xdf_idx, xdf_path in enumerate(CAL_XDFS):
    tag = "C1" if "c1" in xdf_path.stem else "C2"
    print(f"Loading {tag} ... ", end="", flush=True)
    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)
    if results is None:
        print("FAILED"); sys.exit(1)
    print(f"OK ({len(results)} blocks)")

    blocks_this: list[tuple[np.ndarray, int]] = []
    for epochs, level_str in results:
        X, names = _extract_feat(epochs, SRATE, region_map)
        if not feat_names:
            feat_names = names
        X_norm = (X - NORM_MEAN) / NORM_STD
        blocks_this.append((X_norm, LABEL_MAP[level_str]))
    run_blocks.append(blocks_this)

n_feat = len(feat_names)
print(f"Features: {n_feat}\n")

# ---------------------------------------------------------------------------
# Build per-level window arrays for C1 and C2
# (all windows of that level pooled across the 3 blocks)
# ---------------------------------------------------------------------------
# run_level_X[run_idx][level_int] -> (n_windows, n_features)
run_level_X: list[dict[int, np.ndarray]] = []
for blocks in run_blocks:
    by_level: dict[int, list] = {}
    for X_norm, lbl in blocks:
        by_level.setdefault(lbl, []).append(X_norm)
    run_level_X.append({lbl: np.concatenate(arrs) for lbl, arrs in by_level.items()})

# ---------------------------------------------------------------------------
# Per-block mean per feature (for time-course analysis)
# run_block_means[run_idx][level_int] -> list of per-block mean arrays (n_features,)
# ---------------------------------------------------------------------------
run_block_means: list[dict[int, list[np.ndarray]]] = []
for blocks in run_blocks:
    by_level_blocks: dict[int, list] = {}
    for X_norm, lbl in blocks:
        by_level_blocks.setdefault(lbl, []).append(X_norm.mean(axis=0))
    run_block_means.append(by_level_blocks)

# ---------------------------------------------------------------------------
# BH-corrected Mann-Whitney U across all features × levels
# ---------------------------------------------------------------------------
# p_vals[level][feat_idx]
p_raw: dict[int, np.ndarray] = {}
u_stat: dict[int, np.ndarray] = {}
effect_r: dict[int, np.ndarray] = {}

for lbl in LEVELS:
    X1 = run_level_X[0].get(lbl)
    X2 = run_level_X[1].get(lbl)
    p_lbl = np.ones(n_feat)
    u_lbl = np.zeros(n_feat)
    r_lbl = np.zeros(n_feat)
    if X1 is not None and X2 is not None:
        n1, n2 = len(X1), len(X2)
        for fi in range(n_feat):
            res = mannwhitneyu(X1[:, fi], X2[:, fi], alternative="two-sided")
            p_lbl[fi] = res.pvalue
            u_lbl[fi] = res.statistic
            r_lbl[fi] = 1 - 2 * res.statistic / (n1 * n2)  # rank-biserial r
    p_raw[lbl] = p_lbl
    u_stat[lbl] = u_lbl
    effect_r[lbl] = r_lbl

# Benjamini-Hochberg per level
def _bh(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    rank  = np.empty(m, dtype=int)
    rank[order] = np.arange(1, m + 1)
    threshold = rank / m * q
    sig = p <= threshold
    # BH: reject if p[i] <= (rank[i]/m)*q AND all lower-ranked are also rejected
    max_sig_rank = 0
    for k in range(m - 1, -1, -1):
        if sig[order[k]]:
            max_sig_rank = k + 1
            break
    reject = np.zeros(m, dtype=bool)
    reject[order[:max_sig_rank]] = True
    return reject

sig_bh: dict[int, np.ndarray] = {lbl: _bh(p_raw[lbl]) for lbl in LEVELS}

# ---------------------------------------------------------------------------
# Summary table: shift in level-mean between C1 and C2
# ---------------------------------------------------------------------------
print("=" * 120)
print("TABLE 1 — PER-LEVEL MEAN SHIFT C2 − C1  (Z-normed)  |  * = significant after BH (q=0.05)")
print("=" * 120)
print(f"  {'Feature':<30}  "
      f"{'LOW shift':>10}  {'LOW |r|':>7}  {'LOW*':>5}  "
      f"{'MOD shift':>10}  {'MOD |r|':>7}  {'MOD*':>5}  "
      f"{'HI shift':>10}  {'HI |r|':>7}  {'HI*':>5}  "
      f"{'flip':>5}")
print("  " + "-" * 118)

# Sort: features with any significant shift first, then by max |shift|
low_lbl  = LABEL_MAP["LOW"]
mod_lbl  = LABEL_MAP.get("MODERATE", 1)
high_lbl = LABEL_MAP["HIGH"]

shifts = {}
for fi in range(n_feat):
    row = {}
    for lbl in LEVELS:
        X1 = run_level_X[0].get(lbl)
        X2 = run_level_X[1].get(lbl)
        if X1 is not None and X2 is not None:
            row[lbl] = float(X2[:, fi].mean() - X1[:, fi].mean())
        else:
            row[lbl] = float("nan")
    shifts[fi] = row

order = sorted(range(n_feat),
               key=lambda i: (
                   -int(any(sig_bh[lbl][i] for lbl in LEVELS)),
                   -max(abs(shifts[i][lbl]) for lbl in LEVELS),
               ))

for fi in order:
    nm = feat_names[fi]
    is_flip = nm in FLIP_FEATURES
    parts = []
    for lbl in [low_lbl, mod_lbl, high_lbl]:
        sh = shifts[fi][lbl]
        r  = abs(float(effect_r[lbl][fi]))
        s  = "*" if sig_bh[lbl][fi] else ""
        parts += [f"{sh:+10.3f}", f"{r:7.3f}", f"{s:>5}"]
    flip_str = "FLIP" if is_flip else ""
    print(f"  {nm:<30}  {'  '.join(parts)}  {flip_str:>5}")

# ---------------------------------------------------------------------------
# Summary: how many features shifted significantly per level?
# ---------------------------------------------------------------------------
print()
print("Significant shifts (BH q=0.05):")
for lbl in LEVELS:
    n_sig = int(sig_bh[lbl].sum())
    n_flip_sig = int(sum(1 for fi in range(n_feat) if sig_bh[lbl][fi] and feat_names[fi] in FLIP_FEATURES))
    print(f"  {LEVEL_NAMES[lbl]:<10}: {n_sig:3d} / {n_feat}  "
          f"(of which {n_flip_sig} are in the 13 sign-flipping features)")

# ---------------------------------------------------------------------------
# TABLE 2: Level-mean absolute values C1 vs C2 for the sign-flipping features
# Shows WHICH class drives the flip (is HIGH shifting, LOW, or both?)
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("TABLE 2 — SIGN-FLIPPING FEATURES: level means C1 vs C2")
print("          Are HIGH values shifting, LOW values, or both?")
print("=" * 100)
print(f"  {'Feature':<25}  "
      f"{'C1_LOW':>8}  {'C2_LOW':>8}  {'Δ_LOW':>8}  "
      f"{'C1_MOD':>8}  {'C2_MOD':>8}  {'Δ_MOD':>8}  "
      f"{'C1_HI':>8}  {'C2_HI':>8}  {'Δ_HI':>8}  "
      f"{'disc_C1':>8}  {'disc_C2':>8}")
print("  " + "-" * 98)

flip_indices = [fi for fi, nm in enumerate(feat_names) if nm in FLIP_FEATURES]
flip_indices.sort(key=lambda i: abs(run_level_X[0][high_lbl][:, i].mean() -
                                     run_level_X[0][low_lbl][:, i].mean()), reverse=True)

for fi in flip_indices:
    nm = feat_names[fi]
    def _m(run, lbl): return float(run_level_X[run].get(lbl, np.full((1, n_feat), np.nan))[:, fi].mean())
    c1_low, c2_low = _m(0, low_lbl), _m(1, low_lbl)
    c1_mod, c2_mod = _m(0, mod_lbl), _m(1, mod_lbl)
    c1_hi,  c2_hi  = _m(0, high_lbl), _m(1, high_lbl)
    disc_c1 = c1_hi - c1_low
    disc_c2 = c2_hi - c2_low
    print(f"  {nm:<25}  "
          f"{c1_low:8.3f}  {c2_low:8.3f}  {c2_low-c1_low:+8.3f}  "
          f"{c1_mod:8.3f}  {c2_mod:8.3f}  {c2_mod-c1_mod:+8.3f}  "
          f"{c1_hi:8.3f}  {c2_hi:8.3f}  {c2_hi-c1_hi:+8.3f}  "
          f"{disc_c1:8.3f}  {disc_c2:8.3f}")

# ---------------------------------------------------------------------------
# TABLE 3: Per-block time-course for each sign-flipping feature
# Shows temporal drift within each run
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("TABLE 3 — PER-BLOCK TIME-COURSE (sign-flipping features)")
print("  Each column = one block in temporal order (H/L/M = workload level)")
print("  Value = block mean Z-normed feature, Δ = C2_block − C1_block (same level)")
print("=" * 100)

# Build ordered block list per run: [(level_int, block_mean_array), ...]
# Temporal order = order returned by _load_xdf_block
for fi in flip_indices:
    nm = feat_names[fi]
    print(f"\n  {nm}")
    header_c1 = "  C1  "
    header_c2 = "  C2  "
    rows_c1 = []
    rows_c2 = []
    for run_idx, (blocks, tag) in enumerate(zip(run_blocks, ["C1", "C2"])):
        row_parts = []
        for blk_idx, (X_norm, lbl) in enumerate(blocks):
            blk_mean = float(X_norm[:, fi].mean())
            lv = LEVEL_NAMES.get(lbl, "?")
            row_parts.append(f"{lv}:{blk_mean:+.2f}")
        if run_idx == 0:
            rows_c1 = row_parts
        else:
            rows_c2 = row_parts

    # Group by level to show C1 vs C2 blocks for same level
    by_level_c1: dict[int, list] = {}
    by_level_c2: dict[int, list] = {}
    for X_norm, lbl in run_blocks[0]:
        by_level_c1.setdefault(lbl, []).append(float(X_norm[:, fi].mean()))
    for X_norm, lbl in run_blocks[1]:
        by_level_c2.setdefault(lbl, []).append(float(X_norm[:, fi].mean()))

    print(f"    {'Run':<5}  " + "  ".join(f"blk{i+1:02d}" for i in range(9)))
    print(f"    C1     " + "  ".join(f"{lv}:{rows_c1[i].split(':')[1]}" for i, lv in enumerate(rows_c1)))
    print(f"    C2     " + "  ".join(f"{lv}:{rows_c2[i].split(':')[1]}" for i, lv in enumerate(rows_c2)))
    print()

    # Per-level block-by-block comparison
    for lbl in LEVELS:
        lv = LEVEL_NAMES[lbl]
        b1 = by_level_c1.get(lbl, [])
        b2 = by_level_c2.get(lbl, [])
        if not b1 or not b2:
            continue
        deltas = [f"{b2[k]-b1[k]:+.2f}" for k in range(min(len(b1), len(b2)))]
        print(f"    Δ({lv}): " + "  ".join(deltas) +
              f"   mean_Δ={np.mean([b2[k]-b1[k] for k in range(min(len(b1),len(b2)))]):.3f}")

print("\n--- End of analysis ---")
