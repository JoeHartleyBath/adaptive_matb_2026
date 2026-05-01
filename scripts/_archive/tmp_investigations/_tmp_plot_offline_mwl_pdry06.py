"""Offline MWL timeline plot for PDRY06 — control and adaptation conditions.

Re-runs inference on the XDF EEG for both conditions and plots a timeline
in the same style as plot_adaptation_session.py.  For the adaptation
condition, the live MWL stream (recorded in the XDF) is overlaid so the
online/offline divergence is directly visible.

Block boundaries are extracted from the OpenMATB markers stream inside
each XDF so no external session CSV alignment is needed.

Run:
    .\.venv\Scripts\python.exe scripts\_tmp_plot_offline_mwl_pdry06.py
"""
import sys, json, re
import joblib
import numpy as np
import pyxdf
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor
from eeg.online_features import OnlineFeatureExtractor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE    = 128.0
WINDOW_S = 2.0
STEP_S   = 0.25

MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
META_PATH = Path(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml")
FEAT_CFG  = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")
OUT_FIG   = Path(r"C:\adaptive_matb_2026\results\figures\offline_mwl_pdry06.png")

pipeline  = joblib.load(MODEL_DIR / "pipeline.pkl")
selector  = joblib.load(MODEL_DIR / "selector.pkl")
with open(MODEL_DIR / "norm_stats.json") as f:
    ns = json.load(f)
with open(MODEL_DIR / "model_config.json") as f:
    model_cfg = json.load(f)
norm_mean = np.array(ns["mean"])
norm_std  = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0
threshold = model_cfg["youden_threshold"]
n_classes = int(ns.get("n_classes", 3))

meta     = yaml.safe_load(open(META_PATH))
ch_names = meta["channel_names"]
extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=FEAT_CFG)

print(f"Model: {n_classes}-class  threshold={threshold:.4f}\n")

# ---------------------------------------------------------------------------
# Marker parsing helpers
# ---------------------------------------------------------------------------
_LEVEL_RE = re.compile(r"/block_(\d+)/(\w+)$")

def _parse_xdf_blocks(streams, eeg_ts_start: float) -> list[dict]:
    """Extract block start/end from the OpenMATB markers stream in an XDF.

    Returns list of dicts: {level, block_num, start_s, end_s}
    where times are seconds relative to eeg_ts_start (XDF EEG sample 0).
    """
    marker_stream = next(
        (s for s in streams
         if s["info"]["type"][0] == "Markers"
         and s["info"]["name"][0] == "OpenMATB"),
        None,
    )
    if marker_stream is None:
        return []

    starts: dict[str, tuple[float, str, int]] = {}
    ends:   dict[str, float] = {}

    for ts, sample in zip(marker_stream["time_stamps"], marker_stream["time_series"]):
        raw = str(sample[0]).split("|", 1)[0].strip()
        body = raw.removeprefix("STUDY/V0/")
        if body.endswith("/START"):
            base = body[:-len("/START")]
            m = _LEVEL_RE.search(base)
            if m:
                starts[base] = (ts, m.group(2).upper(), int(m.group(1)))
        elif body.endswith("/END"):
            base = body[:-len("/END")]
            ends[base] = ts

    blocks = []
    for base, (start_ts, level, block_num) in starts.items():
        end_ts = ends.get(base)
        if end_ts is None or end_ts <= start_ts:
            continue
        blocks.append(dict(
            level=level,
            block_num=block_num,
            start_s=start_ts - eeg_ts_start,
            end_s=end_ts   - eeg_ts_start,
        ))
    blocks.sort(key=lambda b: b["start_s"])
    return blocks


# ---------------------------------------------------------------------------
# Offline inference
# ---------------------------------------------------------------------------
def offline_inference(xdf_path: Path) -> dict:
    """Load XDF, run offline inference, return dict with time/p_high arrays."""
    print(f"  Loading {xdf_path.name} ...", flush=True)
    streams, _ = pyxdf.load_xdf(str(xdf_path))

    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        raise RuntimeError(f"No EEG stream in {xdf_path.name}")

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T  # (n_ch, n)
    eeg_ts   = np.array(eeg_stream["time_stamps"])
    eeg_t0   = eeg_ts[0]

    # Decimate to analysis srate
    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > SRATE * 1.1:
            factor = int(round(actual_srate / SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts   = eeg_ts[::factor]
            print(f"    decimated {actual_srate:.0f}→{SRATE:.0f} Hz (×{factor})")

    # Filter
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    filtered = preprocessor.process(eeg_data)

    # Slide windows
    step   = int(STEP_S * SRATE)
    win    = int(WINDOW_S * SRATE)
    n_samp = filtered.shape[1]
    t_vals, p_highs = [], []

    for start in range(win, n_samp - step, step):
        window    = filtered[:, start - win : start]
        feats     = extractor.compute(window)
        feats_z   = (feats - norm_mean) / norm_std
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        t_vals.append(start / SRATE)           # seconds from XDF start
        p_highs.append(float(proba[-1]))

    t_arr  = np.array(t_vals)
    ph_arr = np.array(p_highs)

    # EMA smoothing — same α=0.05 used by MwlAdaptationScheduler live
    _ALPHA = 0.05
    ph_smooth = np.empty_like(ph_arr)
    ph_smooth[0] = ph_arr[0]
    for i in range(1, len(ph_arr)):
        ph_smooth[i] = _ALPHA * ph_arr[i] + (1 - _ALPHA) * ph_smooth[i - 1]

    # Pull live MWL stream if present
    mwl_stream = next((s for s in streams if s["info"]["type"][0] == "MWL"), None)
    live_t = live_ph = live_assist = None
    if mwl_stream is not None:
        live_t  = np.array(mwl_stream["time_stamps"]) - eeg_t0
        mwl_data = np.array(mwl_stream["time_series"])  # (n, 3)
        live_ph = mwl_data[:, 0]
        print(f"    live MWL stream: n={len(live_ph)}, mean={live_ph.mean():.3f}")

    # Pull AdaptationEvents if present (for assist state)
    adapt_stream = next(
        (s for s in streams if s["info"]["name"][0] == "AdaptationEvents"), None
    )

    # Blocks
    blocks = _parse_xdf_blocks(streams, eeg_t0)
    level_counts = {}
    for b in blocks:
        level_counts[b["level"]] = level_counts.get(b["level"], 0) + 1
    print(f"    offline: n={len(ph_arr)}, mean={ph_arr.mean():.3f}, "
          f"pct>{threshold:.3f}={(ph_arr > threshold).mean()*100:.1f}%")
    print(f"    blocks: {level_counts}")

    return dict(t=t_arr, p_high=ph_arr, p_high_smooth=ph_smooth,
                live_t=live_t, live_ph=live_ph,
                blocks=blocks)


# ---------------------------------------------------------------------------
# Run inference for both conditions
# ---------------------------------------------------------------------------
print("=== Control ===")
ctrl = offline_inference(
    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf"
)
print("\n=== Adaptation ===")
adap = offline_inference(
    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf"
)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
_LEVEL_COLOURS = {
    "HIGH":     "tab:red",
    "MODERATE": "tab:orange",
    "LOW":      "tab:blue",
}


def _draw_blocks(ax, blocks: list[dict]) -> None:
    for blk in blocks:
        colour = _LEVEL_COLOURS.get(blk["level"], "grey")
        ax.axvspan(blk["start_s"], blk["end_s"], color=colour, alpha=0.07, zorder=0)
        ax.text(
            (blk["start_s"] + blk["end_s"]) / 2, 1.02,
            blk["level"],
            ha="center", va="bottom", fontsize=7,
            color=colour, alpha=0.9,
            transform=ax.get_xaxis_transform(),
        )


def _draw_panel(ax, data: dict, title: str, show_live: bool = False) -> None:
    t        = data["t"]
    p_high   = data["p_high"]
    p_smooth = data["p_high_smooth"]

    _draw_blocks(ax, data["blocks"])

    # Offline raw p_high (faint)
    ax.plot(t, p_high, color="0.75", lw=0.5, alpha=0.6, label="offline raw p_high")

    # Offline EMA-smoothed (matches live smoother α=0.05)
    ax.plot(t, p_smooth, color="0.2", lw=1.0, label="offline smoothed (α=0.05)")

    # Optional: live MWL overlay
    if show_live and data["live_t"] is not None:
        ax.plot(data["live_t"], data["live_ph"],
                color="steelblue", lw=0.9, alpha=0.85,
                label="live p_high (XDF MWL stream)")

    # Threshold
    ax.axhline(threshold, color="red", lw=1.0, ls="--", alpha=0.8,
               label=f"threshold ({threshold:.3f})")

    pct_raw    = (p_high   > threshold).mean() * 100
    pct_smooth = (p_smooth > threshold).mean() * 100
    live_note = ""
    if show_live:
        if data["live_ph"] is not None:
            live_note = f"  |  live {(data['live_ph'] > threshold).mean()*100:.0f}% above thr"
        else:
            live_note = "  |  no live stream (estimator not run for this condition)"
    ax.set_title(
        f"{title}  —  offline raw {pct_raw:.0f}%  /  smoothed {pct_smooth:.0f}% above threshold"
        + live_note,
        fontsize=9,
    )
    ax.set_ylabel("p_high", fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=7)

    # Legend
    handles = [
        plt.Line2D([], [], color="0.75", lw=1.0, alpha=0.6, label="offline raw p_high"),
        plt.Line2D([], [], color="0.2",  lw=1.2,             label="offline smoothed (α=0.05)"),
        plt.Line2D([], [], color="red",  lw=1.0, ls="--",    label=f"threshold ({threshold:.3f})"),
    ]
    if show_live:
        if data["live_t"] is not None:
            handles.append(
                plt.Line2D([], [], color="steelblue", lw=1.2, label="live p_high (XDF MWL stream)")
            )
        else:
            handles.append(
                plt.Line2D([], [], color="steelblue", lw=1.2, ls=":", alpha=0.4,
                           label="live p_high — not recorded (control condition)")
            )
    for level, colour in _LEVEL_COLOURS.items():
        handles.append(Patch(facecolor=colour, alpha=0.2, label=level))
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

_draw_panel(axes[0], ctrl, "Control  (no assistance)", show_live=False)
axes[0].set_xlabel("")

_draw_panel(axes[1], adap, "Adaptation  (MWL-driven assistance)", show_live=True)
axes[1].set_xlabel("Time from XDF recording start (s)", fontsize=8)

fig.suptitle("PDRY06 — Offline MWL Re-inference on XDF EEG", fontsize=11, fontweight="bold")
fig.tight_layout()

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=150)
plt.close(fig)
print(f"\nFigure saved: {OUT_FIG}")
