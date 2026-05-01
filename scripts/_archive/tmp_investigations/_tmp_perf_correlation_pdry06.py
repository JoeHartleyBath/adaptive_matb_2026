"""_tmp_perf_correlation_pdry06.py

Per-block offline P(HIGH) vs task performance for the PDRY06 adaptation run.

Parses the adaptive_automation_pdry06_c1_8min.txt scenario (L H M H H L H M,
eight 1-min blocks) and aligns it to the adaptation XDF via the
STUDY/V0/adaptive_automation/1/START marker.  For each block:

  - offline P(HIGH) mean and SD (fresh-filter re-inference on adaptation XDF)
  - MATB tracking: mean center_deviation (lower = better tracking)
  - MATB sysmon: HIT rate (HITs / (HITs + FAs))

Reports Spearman ρ across the 8 blocks for:
  ρ(P(HIGH), tracking_deviation)   — expect positive if model tracks difficulty
  ρ(P(HIGH), sysmon_hit_rate)      — expect negative (higher difficulty → fewer hits)

Saves:
  results/figures/perf_correlation_pdry06.png

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_perf_correlation_pdry06.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams  # noqa: E402
from eeg import EegPreprocessor  # noqa: E402
from eeg.online_features import OnlineFeatureExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SRATE    = 128.0
WINDOW_S = 2.0
STEP_S   = 0.25

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
SESSION_DIR = Path(r"C:\data\adaptive_matb\openmatb\PDRY06\S001\sessions\2026-04-28")
SCENARIO  = Path(r"C:\adaptive_matb_2026\experiment\scenarios\adaptive_automation_pdry06_c1_8min.txt")
OUT_FIG   = Path(r"C:\adaptive_matb_2026\results\figures\perf_correlation_pdry06.png")

ADAPT_XDF  = PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf"
ADAPT_CSV  = SESSION_DIR / "11_260428_144307.csv"

meta      = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names: list[str] = meta["channel_names"]
feat_cfg  = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

pipeline  = joblib.load(MODEL_DIR / "pipeline.pkl")
selector  = joblib.load(MODEL_DIR / "selector.pkl")
with open(MODEL_DIR / "norm_stats.json") as f:
    ns = json.load(f)
with open(MODEL_DIR / "model_config.json") as f:
    model_cfg = json.load(f)
norm_mean = np.array(ns["mean"], dtype=np.float64)
norm_std  = np.array(ns["std"],  dtype=np.float64)
norm_std[norm_std < 1e-12] = 1.0
n_classes = int(ns.get("n_classes", 3))
threshold = float(model_cfg["youden_threshold"])
print(f"Model: {n_classes}-class  threshold={threshold:.4f}")

# ---------------------------------------------------------------------------
# Parse scenario file → (start_s, end_s, level) per block
# ---------------------------------------------------------------------------

_BLK_RE = re.compile(
    r"(?P<hh>\d+):(?P<mm>\d{2}):(?P<ss>\d{2});labstreaminglayer;marker;"
    r"STUDY/V0/adaptive_automation/\d+/block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)"
)

def parse_scenario(path: Path) -> list[tuple[float, float, str]]:
    opens: dict[str, float] = {}
    blocks: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _BLK_RE.match(line.strip().split("|")[0])
        if not m:
            continue
        t_s = int(m["hh"]) * 3600 + int(m["mm"]) * 60 + int(m["ss"])
        lv, ev = m["level"], m["ev"]
        if ev == "START":
            opens[lv] = float(t_s)
        elif ev == "END" and lv in opens:
            blocks.append((opens.pop(lv), float(t_s), lv))
    return sorted(blocks, key=lambda b: b[0])

scenario_blocks = parse_scenario(SCENARIO)
print(f"\nScenario: {len(scenario_blocks)} blocks")
for i, (s, e, lv) in enumerate(scenario_blocks, 1):
    print(f"  Block {i}: {lv:<9}  {s:.0f}s – {e:.0f}s")

# ---------------------------------------------------------------------------
# Load adaptation XDF and extract block boundaries from markers
# ---------------------------------------------------------------------------

print("\nLoading adaptation XDF ...")
streams_adapt, _ = pyxdf.load_xdf(str(ADAPT_XDF))

# Parse all block boundary markers from the OpenMATB stream in the XDF.
# These give us exact LSL timestamps for each block — more reliable than
# a fixed offset because block_01/START may be missing (LabRecorder restart race).
_BLK_MARKER_RE = re.compile(
    r"STUDY/V0/adaptive_automation/\d+/(?P<blk>block_\d+)/(?P<level>LOW|MODERATE|HIGH)/(?P<ev>START|END)"
)

matb_marker_stream = next(
    (s for s in streams_adapt if s["info"]["name"][0] == "OpenMATB"), None
)

# Build blocks from XDF markers (use scenario file only as fallback for t0)
xdf_blocks: list[tuple[float, float, str]] = []    # (start_lsl, end_lsl, level)
_opens: dict[str, float] = {}
if matb_marker_stream is not None:
    for ts, sample in zip(matb_marker_stream["time_stamps"], matb_marker_stream["time_series"]):
        m = _BLK_MARKER_RE.search(str(sample[0]).split("|")[0])
        if not m:
            continue
        lv, ev = m["level"], m["ev"]
        if ev == "START":
            _opens[lv] = float(ts)
        elif ev == "END":
            # If START wasn't captured (e.g. block_01 before LabRecorder reconnected),
            # infer START from scenario duration (blocks are exactly 60 s in this scenario).
            t_end = float(ts)
            t_start = _opens.pop(lv, t_end - 60.0)
            xdf_blocks.append((t_start, t_end, lv))

# Sort by start time
xdf_blocks.sort(key=lambda b: b[0])
print(f"  Found {len(xdf_blocks)} blocks in XDF markers (scenario has {len(scenario_blocks)})")

# If we got fewer XDF blocks than expected, fill from scenario + estimated t0
if len(xdf_blocks) < len(scenario_blocks) and len(xdf_blocks) > 0:
    # Estimate t0 from the first XDF block whose scenario offset we know
    first_xdf_t, _, first_xdf_lv = xdf_blocks[0]
    sc_match = next(
        (sc for sc in scenario_blocks if sc[2] == first_xdf_lv), None
    )
    if sc_match is not None:
        matb_t0 = first_xdf_t - sc_match[0]
        # Fill any missing blocks using scenario offsets
        if len(xdf_blocks) == len(scenario_blocks) - 1:
            missing_sc = scenario_blocks[0]   # assume first block was cut
            xdf_blocks.insert(0, (matb_t0 + missing_sc[0], matb_t0 + missing_sc[1], missing_sc[2]))
            print(f"  Inferred missing block 1 ({missing_sc[2]}) from scenario offset")

# Use XDF blocks if available, otherwise fall back to scenario + fixed offset
if len(xdf_blocks) == len(scenario_blocks):
    aligned_blocks = [(s, e, lv) for s, e, lv in xdf_blocks]
    print("  Using XDF marker timestamps for block alignment.")
else:
    # Last resort: fixed offset
    eeg_tmp = _merge_eeg_streams(streams_adapt)
    matb_t0 = float(np.array(eeg_tmp["time_stamps"])[0]) + 12.0
    aligned_blocks = [(matb_t0 + s, matb_t0 + e, lv) for s, e, lv in scenario_blocks]
    print(f"  WARNING: Using fixed +12s offset (t0={matb_t0:.3f}).")

for i, (s, e, lv) in enumerate(aligned_blocks, 1):
    print(f"  Block {i}: {lv:<9}  LSL {s:.1f} – {e:.1f}")

# ---------------------------------------------------------------------------
# Preprocess and infer on adaptation XDF
# ---------------------------------------------------------------------------

eeg_s = _merge_eeg_streams(streams_adapt)
eeg_data = np.array(eeg_s["time_series"], dtype=np.float32).T
eeg_ts   = np.array(eeg_s["time_stamps"])
if len(eeg_ts) > 1:
    actual = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
    if actual > SRATE * 1.1:
        fac      = int(round(actual / SRATE))
        eeg_data = eeg_data[:, ::fac]
        eeg_ts   = eeg_ts[::fac]

print(f"  EEG: {eeg_data.shape[1]} samp ({eeg_data.shape[1]/SRATE:.0f}s), {eeg_data.shape[0]} ch")

pp = EegPreprocessor(PREPROCESSING_CONFIG)
pp.initialize_filters(eeg_data.shape[0])
filtered = pp.process(eeg_data)

print("  Inferring P(HIGH) per 2s window ...", flush=True)
ext  = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
step = int(STEP_S   * SRATE)
win  = int(WINDOW_S * SRATE)
n    = filtered.shape[1]
ph_all: list[float] = []
ts_all: list[float] = []
for s in range(win, n - step, step):
    f   = ext.compute(filtered[:, s - win : s])
    fz  = (f - norm_mean) / norm_std
    p   = pipeline.predict_proba(selector.transform(fz[np.newaxis, :]))[0]
    ph_all.append(float(p[-1]))
    ts_all.append(float(eeg_ts[s]))      # timestamp at window end

ph_arr = np.array(ph_all)
ts_arr = np.array(ts_all)
print(f"  Total windows: {len(ph_arr)}  mean P(HIGH)={ph_arr.mean():.3f}")

# ---------------------------------------------------------------------------
# Load performance CSV
# ---------------------------------------------------------------------------

df = pd.read_csv(ADAPT_CSV)
df_perf = df[df["type"] == "performance"].copy()
df_track = df_perf[(df_perf["module"] == "track") &
                   (df_perf["address"] == "center_deviation")].copy()
df_track["value"]   = pd.to_numeric(df_track["value"], errors="coerce")
df_track = df_track.dropna(subset=["value"])

df_sysmon = df_perf[(df_perf["module"] == "sysmon") &
                    (df_perf["address"] == "signal_detection")].copy()

# ---------------------------------------------------------------------------
# Aggregate statistics per block
# ---------------------------------------------------------------------------

print("\nPer-block statistics:")
print(f"\n  {'Blk':>3} {'Level':>9}  {'N_win':>5}  {'P(H)_mean':>10}  {'P(H)>thr':>9}  "
      f"{'Track_dev':>10}  {'Sysmon_hit%':>11}")
print("  " + "-" * 75)

LEVEL_COLOR = {"LOW": "#4c78a8", "MODERATE": "#f58518", "HIGH": "#e45756"}

block_stats = []
for i, (t_start, t_end, lv) in enumerate(aligned_blocks, 1):

    # P(HIGH) windows in this block
    mask_ph = (ts_arr >= t_start) & (ts_arr < t_end)
    ph_blk  = ph_arr[mask_ph]

    # Tracking deviation
    mask_tr = (df_track["logtime"] >= t_start) & (df_track["logtime"] < t_end)
    trk_vals = df_track.loc[mask_tr, "value"].to_numpy()
    trk_mean = float(trk_vals.mean()) if len(trk_vals) > 0 else float("nan")

    # Sysmon hit rate
    mask_sy  = (df_sysmon["logtime"] >= t_start) & (df_sysmon["logtime"] < t_end)
    sy_vals  = df_sysmon.loc[mask_sy, "value"].tolist()
    n_hits   = sy_vals.count("HIT")
    n_total  = len(sy_vals)
    hit_rate = (n_hits / n_total * 100) if n_total > 0 else float("nan")

    row = dict(
        block=i, level=lv,
        n_win=len(ph_blk),
        ph_mean=(ph_blk.mean() if len(ph_blk) > 0 else float("nan")),
        ph_above=(np.mean(ph_blk > threshold) * 100 if len(ph_blk) > 0 else float("nan")),
        track_dev=trk_mean,
        hit_rate=hit_rate,
    )
    block_stats.append(row)
    print(f"  {i:>3} {lv:>9}  {row['n_win']:>5}  {row['ph_mean']:>10.3f}  "
          f"{row['ph_above']:>8.1f}%  {trk_mean:>10.2f}  {hit_rate:>10.1f}%")

# ---------------------------------------------------------------------------
# Spearman correlations
# ---------------------------------------------------------------------------

ph_means   = np.array([r["ph_mean"]  for r in block_stats])
trk_devs   = np.array([r["track_dev"] for r in block_stats])
hit_rates  = np.array([r["hit_rate"]  for r in block_stats])

# Encode true label for rank comparison: LOW=0, MODERATE=1, HIGH=2
LABEL_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
true_ranks = np.array([LABEL_RANK[r["level"]] for r in block_stats])

valid_trk  = np.isfinite(ph_means) & np.isfinite(trk_devs)
valid_hit  = np.isfinite(ph_means) & np.isfinite(hit_rates)

print("\nSpearman correlations:")
if valid_trk.sum() >= 4:
    r_trk, p_trk = spearmanr(ph_means[valid_trk], trk_devs[valid_trk])
    print(f"  ρ(offline P(HIGH), tracking_deviation): {r_trk:+.3f}  p={p_trk:.3f}")
if valid_hit.sum() >= 4:
    r_hit, p_hit = spearmanr(ph_means[valid_hit], hit_rates[valid_hit])
    print(f"  ρ(offline P(HIGH), sysmon_hit_rate):    {r_hit:+.3f}  p={p_hit:.3f}")

r_lbl, p_lbl = spearmanr(ph_means[np.isfinite(ph_means)], true_ranks[np.isfinite(ph_means)])
print(f"  ρ(offline P(HIGH), true_label_rank):    {r_lbl:+.3f}  p={p_lbl:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

n_blocks = len(block_stats)
x        = np.arange(n_blocks)
blk_lbls = [f"B{r['block']}\n{r['level'][:3]}" for r in block_stats]
bar_cols  = [LEVEL_COLOR[r["level"]] for r in block_stats]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: offline P(HIGH) per block
ax1 = axes[0]
bars1 = ax1.bar(x, ph_means, color=bar_cols, edgecolor="k", lw=0.5, alpha=0.85)
ax1.axhline(threshold, color="k", ls="--", lw=0.8, alpha=0.5, label=f"threshold={threshold:.3f}")
ax1.set_ylabel("Mean P(HIGH)")
ax1.set_ylim(0, 1.1)
ax1.set_title("PDRY06 Adaptation — Offline P(HIGH) vs Task Performance per Block")
ax1.legend(fontsize=8)
for bar, val in zip(bars1, ph_means):
    if np.isfinite(val):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

# Panel 2: tracking center_deviation
ax2 = axes[1]
bars2 = ax2.bar(x, trk_devs, color=bar_cols, edgecolor="k", lw=0.5, alpha=0.85)
ax2.set_ylabel("Mean centre deviation (px)")
val_label = f"ρ(P(H), track_dev)={r_trk:+.2f} (p={p_trk:.2f})" if valid_trk.sum() >= 4 else ""
ax2.set_title(f"Tracking deviation  ({val_label})")
for bar, val in zip(bars2, trk_devs):
    if np.isfinite(val):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

# Panel 3: sysmon hit rate
ax3 = axes[2]
bars3 = ax3.bar(x, hit_rates, color=bar_cols, edgecolor="k", lw=0.5, alpha=0.85)
ax3.set_ylabel("Sysmon HIT rate (%)")
ax3.set_ylim(0, 115)
val_label2 = f"ρ(P(H), hit_rate)={r_hit:+.2f} (p={p_hit:.2f})" if valid_hit.sum() >= 4 else ""
ax3.set_title(f"Sysmon hit rate  ({val_label2})")
ax3.set_xticks(x)
ax3.set_xticklabels(blk_lbls, fontsize=9)
for bar, val in zip(bars3, hit_rates):
    if np.isfinite(val):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 2,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

# Legend
legend_patches = [
    mpatches.Patch(facecolor=LEVEL_COLOR[lv], edgecolor="k", label=lv)
    for lv in ("LOW", "MODERATE", "HIGH")
]
fig.legend(handles=legend_patches, loc="upper right",
           bbox_to_anchor=(0.99, 0.98), fontsize=9, framealpha=0.9)

plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT_FIG}")
