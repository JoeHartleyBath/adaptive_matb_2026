"""Replay the adaptation scheduler logic offline on calibration XDFs.

Computes p_high from EEG (matching the online pipeline) then runs the full
EMA + threshold + hysteresis + hold-timer + cooldown state machine to show
when assist_on / assist_off would have triggered — and whether that aligns
with the known block boundaries.

Usage:  python _tmp_replay_adapt_logic_on_cal.py
"""
import sys, json, re, joblib, pyxdf
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor
from eeg.online_features import OnlineFeatureExtractor
import yaml

# ---------------------------------------------------------------------------
# Model + config
# ---------------------------------------------------------------------------
SRATE     = 128.0
WINDOW_S  = 2.0
STEP_S    = 0.25

model_dir  = Path(r"C:\data\adaptive_matb\models\PSELF")
pipeline   = joblib.load(model_dir / "pipeline.pkl")
selector   = joblib.load(model_dir / "selector.pkl")
ns         = json.load(open(model_dir / "norm_stats.json"))
mc         = json.load(open(model_dir / "model_config.json"))
norm_mean  = np.array(ns["mean"])
norm_std   = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0

meta      = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names  = meta["channel_names"]
feat_cfg  = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

physio = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")

# Scheduler parameters (from MwlAdaptationConfig)
THRESHOLD  = mc["youden_threshold"]
ALPHA      = 0.05
HYSTERESIS = 0.02
T_HOLD_S   = 3.0
COOLDOWN_S = 15.0

print(f"Model threshold (Youden J): {THRESHOLD:.4f}")
print(f"EMA α={ALPHA}  hysteresis={HYSTERESIS}  hold={T_HOLD_S}s  cooldown={COOLDOWN_S}s\n")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def compute_phigh_series(xdf_path: Path):
    """Return (timestamps, p_high_array) computed offline from EEG."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        raise RuntimeError(f"No EEG streams in {xdf_path.name}")

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts   = np.array(eeg_stream["time_stamps"])

    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > SRATE * 1.1:
            factor = int(round(actual_srate / SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts   = eeg_ts[::factor]

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    filtered = preprocessor.process(eeg_data)

    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)

    step   = int(STEP_S * SRATE)
    win    = int(WINDOW_S * SRATE)
    n_samp = filtered.shape[1]

    ph_list  = []
    ts_list  = []
    for start in range(win, n_samp - step, step):
        window    = filtered[:, start - win : start]
        feats     = extractor.compute(window)
        feats_z   = (feats - norm_mean) / norm_std
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        ph_list.append(float(proba[-1]))
        # Assign timestamp to middle of the window
        ts_list.append(eeg_ts[start - win // 2])

    return np.array(ts_list), np.array(ph_list)


def parse_blocks(streams):
    """Extract sorted block (idx, level, t_start, t_end) from OpenMATB marker stream."""
    mstream = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
    if mstream is None:
        return []
    events = list(zip(mstream["time_stamps"], [s[0] for s in mstream["time_series"]]))
    blocks = {}
    for ts, ev in events:
        m = re.search(r"block_(\d+)/(\w+)/(START|END)", ev)
        if not m:
            continue
        idx, level, which = m.groups()
        if which == "START":
            blocks.setdefault(idx, {})["start"] = (ts, level)
        else:
            blocks.setdefault(idx, {})["end"] = (ts, level)
    result = []
    for idx, d in sorted(blocks.items()):
        t_s, lev = d.get("start", (None, d.get("end", (None, "?"))[1]))
        t_e, _   = d.get("end", (None, None))
        if t_s is not None and t_e is not None:
            result.append((idx, lev, t_s, t_e))
    return result


def simulate_scheduler(ph_ts, ph_vals):
    """Run the full adaptation scheduler state machine on a p_high series.

    Returns list of (timestamp, event_str) for assist_on / assist_off.
    """
    ema         = None
    zone        = None
    zone_entry  = None
    assist_on   = False
    cooldown_end = 0.0
    events_out  = []

    for ts, v in zip(ph_ts, ph_vals):
        # Update EMA
        if ema is None:
            ema = float(v)
        else:
            ema = ALPHA * float(v) + (1.0 - ALPHA) * ema

        # Determine zone
        if ema > THRESHOLD + HYSTERESIS:
            new_zone = "above"
        elif ema < THRESHOLD - HYSTERESIS:
            new_zone = "below"
        else:
            new_zone = "dead"

        if new_zone != zone:
            zone = new_zone
            zone_entry = ts

        hold_s = ts - zone_entry if zone_entry is not None else 0.0
        in_cooldown = ts < cooldown_end

        if in_cooldown:
            continue

        if zone == "above" and hold_s >= T_HOLD_S and not assist_on:
            assist_on = True
            cooldown_end = ts + COOLDOWN_S
            zone_entry = ts
            events_out.append((ts, "assist_on"))

        elif zone == "below" and hold_s >= T_HOLD_S and assist_on:
            assist_on = False
            cooldown_end = ts + COOLDOWN_S
            zone_entry = ts
            events_out.append((ts, "assist_off"))

    return events_out


# ---------------------------------------------------------------------------
# Main: run on cal_c1 and cal_c2
# ---------------------------------------------------------------------------

for acq in ["cal_c1", "cal_c2"]:
    xdf_path = physio / f"sub-PSELF_ses-S005_task-matb_acq-{acq}_physio.xdf"
    print("=" * 70)
    print(f"  {acq.upper()}  —  {xdf_path.name}")
    print("=" * 70)

    print("  Computing p_high from EEG ...", end=" ", flush=True)
    ph_ts, ph_vals = compute_phigh_series(xdf_path)
    print(f"done ({len(ph_vals)} windows)")

    # Block structure
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    blocks = parse_blocks(streams)
    t0 = ph_ts[0]

    print(f"\n  Block structure (times relative to first EEG window at {t0:.1f}):")
    for idx, level, t_s, t_e in blocks:
        mask = (ph_ts >= t_s) & (ph_ts < t_e)
        ph   = ph_vals[mask]
        if len(ph) == 0:
            print(f"    block_{idx:>2} {level:<10}  t+{t_s-t0:6.1f} – t+{t_e-t0:6.1f}s  NO WINDOWS")
            continue
        pct = (ph > THRESHOLD).mean() * 100
        print(
            f"    block_{idx:>2} {level:<10}  t+{t_s-t0:6.1f} – t+{t_e-t0:6.1f}s"
            f"  n={len(ph):4d}  mean_ph={ph.mean():.3f}  raw>{THRESHOLD:.3f}={pct:.0f}%"
        )

    # Simulate scheduler
    sched_events = simulate_scheduler(ph_ts, ph_vals)

    print(f"\n  Simulated adaptation events (using Youden threshold {THRESHOLD:.4f}):")
    if not sched_events:
        print("    (no assist_on or assist_off events)")
    for ts, ev in sched_events:
        # Find which block this falls in
        block_label = "?"
        for idx, level, t_s, t_e in blocks:
            if t_s <= ts < t_e:
                block_label = f"block_{idx}/{level}"
                break
        print(f"    t+{ts - t0:6.1f}s  {ev:<12}  [{block_label}]")

    # Per-block % time assist would be ON
    print(f"\n  Per-block % time assist_on would be active:")
    # Simulate assist_on state at each timestamp
    ema        = None
    zone       = None
    zone_entry = None
    assist_on  = False
    cooldown_end = 0.0
    on_flags   = []
    for ts, v in zip(ph_ts, ph_vals):
        if ema is None:
            ema = float(v)
        else:
            ema = ALPHA * float(v) + (1.0 - ALPHA) * ema
        new_zone = (
            "above" if ema > THRESHOLD + HYSTERESIS else
            "below" if ema < THRESHOLD - HYSTERESIS else
            "dead"
        )
        if new_zone != zone:
            zone = new_zone
            zone_entry = ts
        hold_s = ts - zone_entry if zone_entry is not None else 0.0
        if ts >= cooldown_end:
            if zone == "above" and hold_s >= T_HOLD_S and not assist_on:
                assist_on = True
                cooldown_end = ts + COOLDOWN_S
                zone_entry = ts
            elif zone == "below" and hold_s >= T_HOLD_S and assist_on:
                assist_on = False
                cooldown_end = ts + COOLDOWN_S
                zone_entry = ts
        on_flags.append(assist_on)
    on_flags = np.array(on_flags)

    for idx, level, t_s, t_e in blocks:
        mask = (ph_ts >= t_s) & (ph_ts < t_e)
        pct_on = on_flags[mask].mean() * 100 if mask.any() else float("nan")
        ph     = ph_vals[mask]
        print(
            f"    block_{idx:>2} {level:<10}  assist_ON {pct_on:5.1f}%"
            f"  (mean_ph={ph.mean():.3f})"
        )

    print()
