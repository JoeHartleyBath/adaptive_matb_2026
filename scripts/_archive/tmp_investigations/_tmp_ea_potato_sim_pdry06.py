"""Offline simulation: Euclidean Alignment + Euclidean Potato on a participant.

Fits both techniques on calibration XDFs (cal_c1 + cal_c2), then re-runs
inference on control + adaptation under 4 pipeline variants:

    baseline  — existing pipeline, no modifications
    +EA       — Euclidean Alignment applied to each window before features
    +Potato   — Euclidean Potato artifact gate with hold-last-good fallback
    +Both     — EA + Potato combined

Output figures:
    results/figures/ea_potato_mwl_<pid>_<sid>.png      MWL timelines
    results/figures/ea_potato_features_<pid>_<sid>.png  Feature drift

Console prints per-condition statistics (% artifacts, mean P(HIGH), feature std).

Run (defaults to PDRY06 S001):
    .venv/Scripts/python.exe scripts/_tmp_ea_potato_sim_pdry06.py
    .venv/Scripts/python.exe scripts/_tmp_ea_potato_sim_pdry06.py --pid PSELF --sid S005
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import scipy.linalg
import yaml
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import argparse

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor
from eeg.online_features import OnlineFeatureExtractor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser()
_ap.add_argument("--pid", default="PDRY06", help="Participant ID (e.g. PDRY06, PSELF)")
_ap.add_argument("--sid", default="S001",   help="Session ID (e.g. S001, S005)")
ARGS = _ap.parse_args()

PID = ARGS.pid.upper()
SID = ARGS.sid.upper()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE     = 128.0
WINDOW_S  = 2.0
STEP_S    = 0.25
WIN_SAMP  = int(WINDOW_S * SRATE)   # 256
STEP_SAMP = int(STEP_S   * SRATE)   # 32

# After this many consecutive artifact windows (5 s), reset last-good P to 0.5
HOLD_RESET_N = 20

# Potato semi-dynamic update rate (small fixed alpha = slow drift of reference)
POTATO_UPDATE_ALPHA = 0.01

_DATA_ROOT = Path(r"C:\data\adaptive_matb")
MODEL_DIR = _DATA_ROOT / "models" / PID
PHYSIO    = _DATA_ROOT / "physiology" / f"sub-{PID}" / f"ses-{SID}" / "physio"
META_PATH = _REPO / "config" / "eeg_metadata.yaml"
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"
OUT_MWL   = _REPO / "results" / "figures" / f"ea_potato_mwl_{PID.lower()}_{SID.lower()}.png"
OUT_FEAT  = _REPO / "results" / "figures" / f"ea_potato_features_{PID.lower()}_{SID.lower()}.png"

TARGET_FEATURES = ["FM_Theta", "FM_Beta", "Par_Alpha"]

_pfx = f"sub-{PID}_ses-{SID}_task-matb_acq"
CAL_FILES = [
    PHYSIO / f"{_pfx}-cal_c1_physio.xdf",
    PHYSIO / f"{_pfx}-cal_c2_physio.xdf",
]
SIM_FILES = {
    "control":    PHYSIO / f"{_pfx}-control_physio.xdf",
    "adaptation": PHYSIO / f"{_pfx}-adaptation_physio.xdf",
}

# ---------------------------------------------------------------------------
# Load model artefacts
# ---------------------------------------------------------------------------
pipeline = joblib.load(MODEL_DIR / "pipeline.pkl")
selector = joblib.load(MODEL_DIR / "selector.pkl")

with open(MODEL_DIR / "norm_stats.json") as f:
    ns = json.load(f)
with open(MODEL_DIR / "model_config.json") as f:
    model_cfg = json.load(f)

norm_mean  = np.array(ns["mean"], dtype=np.float64)
norm_std   = np.array(ns["std"],  dtype=np.float64)
norm_std[norm_std < 1e-12] = 1.0
threshold  = float(model_cfg["youden_threshold"])
n_classes  = int(ns.get("n_classes", 3))
p_high_col = n_classes - 1

meta     = yaml.safe_load(META_PATH.read_text())
ch_names = meta["channel_names"]

extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=FEAT_CFG)
# Warm-up call to populate feature_names
_dummy = extractor.compute(np.zeros((len(ch_names), WIN_SAMP)))
feat_names   = extractor.feature_names
feat_indices = {fn: feat_names.index(fn) for fn in TARGET_FEATURES if fn in feat_names}

print(f"Model: {n_classes}-class  threshold={threshold:.4f}")
print(f"Features: {len(feat_names)} total | target indices: {feat_indices}\n")


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------
def load_and_filter(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XDF, decimate to SRATE, apply EegPreprocessor.

    Returns (C×N float64, timestamps).
    """
    print(f"  {path.name} ...", flush=True)
    streams, _ = pyxdf.load_xdf(str(path))
    eeg  = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            fac  = int(round(actual / SRATE))
            data = data[:, ::fac]
            ts   = ts[::fac]
            print(f"    decimated ×{fac} → {SRATE:.0f} Hz")
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    data = pp.process(data).astype(np.float64)
    print(f"    {data.shape[1]} samples ({data.shape[1]/SRATE:.0f} s)")
    return data, ts


# ---------------------------------------------------------------------------
# Phase 1: Build calibration references (EA + Potato)
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 1: Building calibration references")
print("=" * 60)

cal_windows: list[np.ndarray] = []
for path in CAL_FILES:
    if not path.exists():
        print(f"  NOT FOUND: {path.name}")
        sys.exit(1)
    data_cal, _ = load_and_filter(path)
    n = data_cal.shape[1]
    for start in range(WIN_SAMP, n, STEP_SAMP):
        cal_windows.append(data_cal[:, start - WIN_SAMP : start].copy())

print(f"\n  {len(cal_windows)} calibration windows from {len(CAL_FILES)} XDFs")

# Stack to (N, C, T) — pyriemann convention
cal_stack = np.stack(cal_windows)   # (N, 128, 256)

print("  Estimating covariances (LWF regularisation) ...", flush=True)
cov_estimator = Covariances(estimator="lwf")
cal_covs = cov_estimator.fit_transform(cal_stack)   # (N, 128, 128)
print(f"  Covariances computed: {cal_covs.shape}")

# ---- Euclidean Alignment reference ----
R_mean = cal_covs.mean(axis=0)
cond_num = np.linalg.cond(R_mean)
print(f"\n  R_mean condition number : {cond_num:.1f}")
print("  Computing R^(-1/2) via scipy.linalg.fractional_matrix_power ...",
      flush=True)
R_invsqrt = scipy.linalg.fractional_matrix_power(R_mean, -0.5).real
# Sanity: R_invsqrt @ R_mean @ R_invsqrt should be close to identity
residual = np.linalg.norm(R_invsqrt @ R_mean @ R_invsqrt - np.eye(R_mean.shape[0]))
print(f"  R_invsqrt sanity ||R^(-1/2) R R^(-1/2) - I|| = {residual:.4f}")

# ---- Riemannian Potato ----
# Using metric='euclid' (Frobenius-norm Potato) — Barthélemy et al. 2019
# benchmark both Riemannian and Euclidean variants. The Riemannian mean
# requires iterative Karcher-flow O(N × C³ × n_iter): with ~4000 windows
# at C=128, that is prohibitively slow (~hours). The Euclidean Potato uses
# the arithmetic mean + Frobenius z-score and is instant. For large
# amplitude artifacts (as flagged SPIKE in PDRY06), both variants detect
# the same events.  Riemannian can be re-enabled by changing metric here.
print("\n  Fitting Euclidean Potato (metric='euclid', z_th=2.0) ...", flush=True)
potato = Potato(metric="euclid", threshold=2.0).fit(cal_covs)
ref_trace = float(np.trace(potato._mdm.covmeans_[0]))
print(f"  Potato fitted.  Reference matrix trace: {ref_trace:.4f}")


# ---------------------------------------------------------------------------
# Phase 2: Offline inference with 4 variants
# ---------------------------------------------------------------------------
def _classify(window: np.ndarray) -> tuple[float, np.ndarray]:
    """Feature extraction + SVC classification for one window.

    Returns (P(HIGH), raw_features_vector).
    """
    feats     = extractor.compute(window)
    feats_z   = (feats - norm_mean) / norm_std
    feats_sel = selector.transform(feats_z[np.newaxis, :])
    proba     = pipeline.predict_proba(feats_sel)[0]
    return float(proba[p_high_col]), feats


def run_inference(data: np.ndarray, ts: np.ndarray, label: str) -> dict:
    """Slide windows over *data* and compute 4 inference variants.

    Returns dict with keys:
        t, baseline, ea, potato, both, flags,
        feat_base (N × n_target), feat_ea (N × n_target)
    """
    n = data.shape[1]
    t_vals     = []
    p_baseline = []
    p_ea       = []
    p_potato   = []
    p_both     = []
    flags      = []   # 1 = artifact flagged by Potato, 0 = clean
    feat_base_rows: list[list[float]] = []
    feat_ea_rows:   list[list[float]] = []

    last_good_potato  = 0.5
    consec_bad_potato = 0
    last_good_both    = 0.5
    consec_bad_both   = 0

    print(f"\n  {label}: {n} samples ({n/SRATE:.0f} s) → "
          f"{(n - WIN_SAMP) // STEP_SAMP} windows ...", flush=True)

    for start in range(WIN_SAMP, n - STEP_SAMP, STEP_SAMP):
        window = data[:, start - WIN_SAMP : start]

        # --- Riemannian Potato classification ---
        cov_w    = cov_estimator.transform(window[np.newaxis, :, :])  # (1,C,C)
        is_clean = bool(potato.predict(cov_w)[0] == 1)
        if is_clean:
            potato.partial_fit(cov_w, alpha=POTATO_UPDATE_ALPHA)
        flags.append(0 if is_clean else 1)

        # --- Baseline ---
        p_b, raw_b = _classify(window)
        p_baseline.append(p_b)

        # --- +EA ---
        window_ea     = R_invsqrt @ window   # (C, T)
        p_e, raw_e    = _classify(window_ea)
        p_ea.append(p_e)

        # --- +Potato: hold-last-good over baseline P ---
        if is_clean:
            last_good_potato  = p_b
            consec_bad_potato = 0
            p_potato.append(p_b)
        else:
            consec_bad_potato += 1
            if consec_bad_potato >= HOLD_RESET_N:
                last_good_potato = 0.5
            p_potato.append(last_good_potato)

        # --- +Both: hold-last-good over EA P ---
        if is_clean:
            last_good_both  = p_e
            consec_bad_both = 0
            p_both.append(p_e)
        else:
            consec_bad_both += 1
            if consec_bad_both >= HOLD_RESET_N:
                last_good_both = 0.5
            p_both.append(last_good_both)

        t_vals.append(start / SRATE)

        # Collect raw target features for drift figure
        fi_b = [raw_b[feat_indices[fn]] for fn in TARGET_FEATURES if fn in feat_indices]
        fi_e = [raw_e[feat_indices[fn]] for fn in TARGET_FEATURES if fn in feat_indices]
        feat_base_rows.append(fi_b)
        feat_ea_rows.append(fi_e)

    t_arr  = np.array(t_vals)
    flags  = np.array(flags, dtype=np.uint8)
    n_art  = int(flags.sum())
    pct    = 100.0 * n_art / len(flags)

    # Artifact run-length statistics
    runs = []
    cnt  = 0
    for f in flags:
        if f == 1:
            cnt += 1
        else:
            if cnt > 0:
                runs.append(cnt)
                cnt = 0
    if cnt > 0:
        runs.append(cnt)

    print(f"    {len(flags)} windows | {n_art} artifacts ({pct:.1f}%)")
    if runs:
        print(f"    run lengths: max={max(runs)}  "
              f"median={int(np.median(runs))}  "
              f"95th={np.percentile(runs, 95):.1f}  "
              f"(×{STEP_S:.2f}s each)")
    print(f"    baseline  mean P(HIGH) = {np.mean(p_baseline):.3f}  "
          f"std = {np.std(p_baseline):.3f}")
    print(f"    +EA       mean P(HIGH) = {np.mean(p_ea):.3f}  "
          f"std = {np.std(p_ea):.3f}")
    print(f"    +Potato   mean P(HIGH) = {np.mean(p_potato):.3f}  "
          f"std = {np.std(p_potato):.3f}")
    print(f"    +Both     mean P(HIGH) = {np.mean(p_both):.3f}  "
          f"std = {np.std(p_both):.3f}")

    fn_list = [fn for fn in TARGET_FEATURES if fn in feat_indices]
    fb  = np.array(feat_base_rows)
    fe  = np.array(feat_ea_rows)
    if fb.size and fn_list:
        print("    Feature drift (raw std):")
        for col, fn in enumerate(fn_list):
            print(f"      {fn:<12}  baseline std={fb[:, col].std():.5f}  "
                  f"EA std={fe[:, col].std():.5f}")

    return dict(
        t         = t_arr,
        baseline  = np.array(p_baseline),
        ea        = np.array(p_ea),
        potato    = np.array(p_potato),
        both      = np.array(p_both),
        flags     = flags,
        feat_base = np.array(feat_base_rows),
        feat_ea   = np.array(feat_ea_rows),
    )


print("\n" + "=" * 60)
print("PHASE 2: Offline inference (4 variants)")
print("=" * 60)

results: dict[str, dict] = {}
for cond, path in SIM_FILES.items():
    if not path.exists():
        print(f"\n  NOT FOUND: {path.name} — skipping")
        continue
    data_sim, ts_sim = load_and_filter(path)
    results[cond] = run_inference(data_sim, ts_sim, cond)


# ---------------------------------------------------------------------------
# Figure 1: MWL timelines
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#555555",
    "ea":       "#2980b9",
    "potato":   "#e67e22",
    "both":     "#27ae60",
}
LABELS = {
    "baseline": "Baseline",
    "ea":       "+Euclidean Alignment",
    "potato":   "+Riemannian Potato",
    "both":     "+Both",
}

conds = [c for c in ("control", "adaptation") if c in results]
n_conds = len(conds)

fig1, axes1 = plt.subplots(n_conds, 1, figsize=(16, 4.5 * n_conds),
                            constrained_layout=True)
if n_conds == 1:
    axes1 = [axes1]

fig1.suptitle(
    f"{PID} {SID} — Offline MWL: 4 pipeline variants\n"
    "Calibration reference: cal_c1 + cal_c2  |  "
    "Red shading = Euclidean Potato artifact",
    fontsize=12,
)

for ax, cond in zip(axes1, conds):
    r   = results[cond]
    t   = r["t"]
    pct = 100.0 * r["flags"].mean()

    # Shade artifact windows
    t_step = float(t[1] - t[0]) if len(t) > 1 else STEP_S
    for i, f in enumerate(r["flags"]):
        if f == 1:
            ax.axvspan(t[i], t[i] + t_step, color="#e74c3c",
                       alpha=0.12, linewidth=0)

    for key in ("baseline", "ea", "potato", "both"):
        ax.plot(t, r[key], lw=1.1, color=COLORS[key],
                label=LABELS[key], alpha=0.92)

    ax.axhline(threshold, color="black", lw=1.0, ls="--",
               label=f"Threshold ({threshold:.3f})", alpha=0.7)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(HIGH MWL)")
    ax.set_xlabel("Time (s from recording start)")
    ax.set_title(f"{cond.capitalize()}  |  "
                 f"{pct:.1f}% windows flagged by Potato")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

OUT_MWL.parent.mkdir(parents=True, exist_ok=True)
fig1.savefig(OUT_MWL, dpi=150)
print(f"\nFigure 1 saved: {OUT_MWL}")


# ---------------------------------------------------------------------------
# Figure 2: Feature drift — baseline vs EA
# ---------------------------------------------------------------------------
fn_list = [fn for fn in TARGET_FEATURES if fn in feat_indices]
n_feats = len(fn_list)

fig2, axes2 = plt.subplots(
    n_feats, n_conds,
    figsize=(9 * n_conds, 3.5 * n_feats),
    constrained_layout=True,
)
# Normalise to always be (n_feats, n_conds) grid
if n_feats == 1 and n_conds == 1:
    axes2 = [[axes2]]
elif n_feats == 1:
    axes2 = [list(axes2)]
elif n_conds == 1:
    axes2 = [[ax] for ax in axes2]

fig2.suptitle(
    f"{PID} {SID} — Raw feature drift: Baseline vs +Euclidean Alignment\n"
    "Calibration reference: cal_c1 + cal_c2",
    fontsize=12,
)

for row, fn in enumerate(fn_list):
    col_idx = fn_list.index(fn)
    for col, cond in enumerate(conds):
        ax   = axes2[row][col]
        r    = results[cond]
        t    = r["t"]
        vals_b = r["feat_base"][:, col_idx]
        vals_e = r["feat_ea"][:, col_idx]
        std_b  = vals_b.std()
        std_e  = vals_e.std()
        ax.plot(t, vals_b, lw=0.8, color="#555555",
                alpha=0.85, label="Baseline")
        ax.plot(t, vals_e, lw=0.8, color="#2980b9",
                alpha=0.85, label="+EA")
        ax.set_title(
            f"{fn} — {cond.capitalize()}\n"
            f"std  baseline={std_b:.5f}  EA={std_e:.5f}",
            fontsize=9,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Raw feature value")
        if row == 0 and col == 0:
            ax.legend(fontsize=8, framealpha=0.85)

OUT_FEAT.parent.mkdir(parents=True, exist_ok=True)
fig2.savefig(OUT_FEAT, dpi=150)
print(f"Figure 2 saved: {OUT_FEAT}\n")
print("Done.")
