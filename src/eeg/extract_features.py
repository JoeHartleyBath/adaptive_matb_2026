"""EEG feature extraction functions for MWL classification."""
from __future__ import annotations

import hashlib
from itertools import permutations
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

# Regions used for time-domain features (Hjorth, entropy, stats)
_TEMPORAL_REGIONS = ["FrontalMidline", "Parietal", "Central", "Occipital"]
_PE_N_PATTERNS_M3 = len(list(permutations(range(3))))


def _save_feature_cache(
    cache_path: Path,
    key: str,
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    feat_names: list[str],
) -> None:
    """Write features to compressed .npz cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cache_key":  np.array(key),
        "pids":       np.array(sorted(X_by.keys())),
        "feat_names": np.array(feat_names),
    }
    for pid, X in X_by.items():
        arrays[f"X_{pid}"] = X
    for pid, y in y_by.items():
        arrays[f"y_{pid}"] = y
    np.savez_compressed(cache_path, **arrays)



def estimate_iaf(
    epochs: np.ndarray,   # (N, C, T)
    srate: float,
    search_lo: float = 7.0,
    search_hi: float = 14.0,
) -> float:
    """Return IAF as Hz: peak of grand-mean PSD in the search window.

    Falls back to 10.0 Hz if no clear peak is found (PSD is flat or monotone).
    """
    freqs, psd = welch(epochs, fs=srate,
                       nperseg=min(256, epochs.shape[-1]), axis=-1)
    # grand mean across epochs and channels
    mean_psd = psd.mean(axis=(0, 1))  # (F,)
    mask = (freqs >= search_lo) & (freqs <= search_hi)
    if not mask.any():
        return 10.0
    peak_idx = np.argmax(mean_psd[mask])
    iaf = float(freqs[mask][peak_idx])
    return iaf


def iaf_bands(iaf: float) -> dict[str, tuple[float, float]]:
    """Return band edges relative to IAF.

    Delta  : fixed [1, 4]
    Theta  : [IAF-4, IAF-1]
    Alpha  : [IAF-1, IAF+3]
    Beta   : fixed [12, 30]
    Gamma  : fixed [30, 45]
    """
    theta_lo = max(1.0, iaf - 4.0)
    theta_hi = max(theta_lo + 0.5, iaf - 1.0)
    alpha_lo = theta_hi
    alpha_hi = iaf + 3.0
    return {
        "Delta": (1.0,     4.0),
        "Theta": (theta_lo, theta_hi),
        "Alpha": (alpha_lo, alpha_hi),
        "Beta":  (12.0, 30.0),
        "Gamma": (30.0, 45.0),
    }


# ---------------------------------------------------------------------------
# Region map
# ---------------------------------------------------------------------------

def _build_region_map(
    cfg_path: Path,
    ch_names: list[str],
) -> dict[str, np.ndarray]:
    cfg = yaml.safe_load(cfg_path.read_text())
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    region_map: dict[str, np.ndarray] = {}
    for region, channels in cfg["regions"].items():
        idxs = [ch_idx[c] for c in channels if c in ch_idx]
        if idxs:
            region_map[region] = np.array(idxs, dtype=np.intp)
    return region_map


# ---------------------------------------------------------------------------
# Time-domain feature helpers (all vectorised over epoch axis N)
# ---------------------------------------------------------------------------

def _region_signals(
    epochs: np.ndarray,              # (N, C, T)
    region_map: dict[str, np.ndarray],
    regions: list[str],
) -> dict[str, np.ndarray]:
    """Return mean-across-channels signal (N, T) for each requested region."""
    out: dict[str, np.ndarray] = {}
    for r in regions:
        idx = region_map.get(r)
        if idx is not None and len(idx) > 0:
            out[r] = epochs[:, idx, :].mean(axis=1)  # (N, T)
    return out


def _hjorth_batch(signals: np.ndarray) -> np.ndarray:
    """Hjorth Activity, Mobility, Complexity. (N, T) -> (N, 3)."""
    dx  = np.diff(signals, axis=1)    # (N, T-1)
    d2x = np.diff(dx,      axis=1)    # (N, T-2)
    act  = signals.var(axis=1) + 1e-12
    mob  = np.sqrt(dx.var(axis=1)  / act  + 1e-12)
    mob2 = np.sqrt(d2x.var(axis=1) / (dx.var(axis=1) + 1e-12) + 1e-12)
    comp = mob2 / (mob + 1e-12)
    return np.column_stack([np.log(act), mob, comp])  # Activity as log for scale


def _spectral_entropy_batch(
    psd_region: np.ndarray,   # (N, F)  -- already averaged over channels
    freqs: np.ndarray,
    fmax: float = 40.0,
) -> np.ndarray:              # (N,)
    """Shannon entropy of normalised PSD, normalised to [0, 1]."""
    mask = freqs <= fmax
    p = psd_region[:, mask]
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    H = -np.sum(p * np.log2(p + 1e-12), axis=1)
    H_max = np.log2(float(mask.sum()))
    return H / (H_max + 1e-12)


def _perm_entropy_batch(signals: np.ndarray) -> np.ndarray:
    """Permutation entropy, m=3, delay=1, normalised. (N, T) -> (N,)."""
    N, T = signals.shape
    n_pats = T - 2
    i0 = np.arange(n_pats)
    # Embed: (N, n_pats, 3)
    windows = np.stack([signals[:, i0], signals[:, i0 + 1], signals[:, i0 + 2]], axis=2)
    ranks = np.argsort(windows, axis=2).astype(np.int32)           # (N, n_pats, 3)
    codes = ranks[:, :, 0] * 9 + ranks[:, :, 1] * 3 + ranks[:, :, 2]  # (N, n_pats)
    H_max = np.log2(float(_PE_N_PATTERNS_M3))
    H = np.empty(N)
    for i in range(N):
        _, cnts = np.unique(codes[i], return_counts=True)
        p = cnts / cnts.sum()
        H[i] = -np.sum(p * np.log2(p + 1e-12))
    return np.clip(H / H_max, 0.0, 1.0)


def _stats_batch(signals: np.ndarray) -> np.ndarray:
    """Skewness, kurtosis, zero-crossing rate. (N, T) -> (N, 3)."""
    sk  = sp_skew(signals, axis=1)
    ku  = sp_kurtosis(signals, axis=1)
    zcr = (np.diff(np.sign(signals), axis=1) != 0).mean(axis=1)
    return np.column_stack([sk, ku, zcr])


# ---------------------------------------------------------------------------
# Bandpower helpers
# ---------------------------------------------------------------------------

def _band_power(
    psd: np.ndarray,       # (N, C, F)  -- pre-computed for all channels
    freqs: np.ndarray,
    ch_idx: np.ndarray,    # channel indices for this region
    lo: float,
    hi: float,
) -> np.ndarray:           # (N,)  linear power, mean across channels and freq bins
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.ones(psd.shape[0])
    return psd[:, ch_idx, :][:, :, mask].mean(axis=(1, 2)) + 1e-12


# ---------------------------------------------------------------------------
# 1/f aperiodic slope helpers
# ---------------------------------------------------------------------------

def _aperiodic_slope_batch(
    psd_region: np.ndarray,  # (N, F)  -- mean across channels for one region
    freqs: np.ndarray,
    exclude_lo: float = 7.0,
    exclude_hi: float = 14.0,
    fit_lo: float = 1.0,
    fit_hi: float = 40.0,
) -> np.ndarray:  # (N,)  -- aperiodic slope (negative = steeper 1/f)
    """Fit 1/f slope in log10-log10 space, excluding alpha peak region."""
    mask = ((freqs >= fit_lo) & (freqs <= fit_hi) &
            ~((freqs >= exclude_lo) & (freqs <= exclude_hi)))
    if mask.sum() < 3:
        return np.zeros(psd_region.shape[0])
    log_f = np.log10(freqs[mask])              # (F',)
    log_p = np.log10(psd_region[:, mask] + 1e-20)  # (N, F')
    # Vectorised OLS: slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2)
    n  = float(log_f.shape[0])
    sx  = log_f.sum()
    sxx = (log_f ** 2).sum()
    sy  = log_p.sum(axis=1)
    sxy = (log_p * log_f[None, :]).sum(axis=1)
    denom = n * sxx - sx * sx + 1e-12
    slope = (n * sxy - sx * sy) / denom
    return slope.astype(np.float32)


# ---------------------------------------------------------------------------
# wPLI (weighted Phase Lag Index) helpers
# ---------------------------------------------------------------------------

def _wpli_batch(
    sig1: np.ndarray,   # (N, T)  region-mean signal 1
    sig2: np.ndarray,   # (N, T)  region-mean signal 2
    srate: float,
    lo: float,
    hi: float,
) -> np.ndarray:        # (N,)  wPLI values in [0, 1]
    """Compute wPLI between two region-averaged signals in a frequency band."""
    nyq = srate / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if lo >= hi:
        return np.zeros(sig1.shape[0], dtype=np.float32)
    b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
    f1 = filtfilt(b, a, sig1, axis=1)
    f2 = filtfilt(b, a, sig2, axis=1)
    # Analytic signal via Hilbert
    h1 = hilbert(f1, axis=1)
    h2 = hilbert(f2, axis=1)
    # Cross-spectral density (instantaneous)
    csd = h1 * np.conj(h2)
    im_csd = np.imag(csd)
    # wPLI = |mean(|Im(S)| * sign(Im(S)))| / mean(|Im(S)|)
    abs_im = np.abs(im_csd)
    num = np.abs(np.mean(abs_im * np.sign(im_csd), axis=1))
    den = np.mean(abs_im, axis=1) + 1e-12
    return (num / den).astype(np.float32)


def extract_features(
    epochs: np.ndarray,    # (N, C, T)
    psd: np.ndarray,       # (N, C, F)
    freqs: np.ndarray,
    region_map: dict[str, np.ndarray],
    bands: dict[str, tuple[float, float]],
    srate: float = 128.0,
) -> tuple[np.ndarray, list[str]]:
    """Return (N, ~54) float32 feature matrix and feature names.

    Feature groups:
      1. Bandpower (spectral, log) — 13 features
      2. Hjorth parameters (per temporal region) — 4 × 3 = 12
      3. Spectral entropy (per temporal region) — 4
      4. Permutation entropy m=3 (per temporal region) — 4
      5. Statistics: skewness, kurtosis, ZCR (per temporal region) — 4 × 3 = 12
      6. Aperiodic 1/f slope (per temporal region) — 4
      7. wPLI connectivity (FM↔Par, FM↔Cen, Cen↔Par × θ/α) — 5
    """
    features: list[np.ndarray] = []
    names:    list[str]        = []

    # ------------------------------------------------------------------
    # 1. Bandpower features (log scale)
    # ------------------------------------------------------------------
    def P(region: str, band: str) -> np.ndarray:
        idx = region_map.get(region)
        if idx is None or len(idx) == 0:
            return np.ones(psd.shape[0])
        lo, hi = bands[band]
        return _band_power(psd, freqs, idx, lo, hi)

    delta_fm  = P("FrontalMidline", "Delta")
    theta_fm  = P("FrontalMidline", "Theta")
    alpha_fm  = P("FrontalMidline", "Alpha")
    beta_fm   = P("FrontalMidline", "Beta")

    alpha_fl  = P("FrontalLeft",  "Alpha")
    alpha_fr  = P("FrontalRight", "Alpha")

    theta_c   = P("Central", "Theta")
    alpha_c   = P("Central", "Alpha")
    beta_c    = P("Central", "Beta")

    alpha_par = P("Parietal", "Alpha")
    alpha_occ = P("Occipital", "Alpha") if "Occipital" in region_map else np.ones(psd.shape[0])

    # Absolute log-power
    for val, nm in [
        (theta_fm,  "FM_Theta"),
        (alpha_fm,  "FM_Alpha"),
        (delta_fm,  "FM_Delta"),
        (beta_fm,   "FM_Beta"),
        (alpha_par, "Par_Alpha"),
        (alpha_occ, "Occ_Alpha"),
        (beta_c,    "Cen_Beta"),
        (theta_c,   "Cen_Theta"),
    ]:
        features.append(np.log(val))
        names.append(nm)

    # Ratio / composite features
    features.append(np.log(alpha_fr) - np.log(alpha_fl))                  # FAA
    features.append(np.log(beta_c  / (alpha_c + theta_c + 1e-12)))        # Engagement
    features.append(np.log(theta_fm / (alpha_fm + 1e-12)))                # FM_Theta/Alpha
    features.append(np.log(theta_fm / (beta_fm  + 1e-12)))                # FM_Theta/Beta
    features.append(np.log(theta_c  / (beta_c   + 1e-12)))                # Cen_Theta/Beta
    names.extend(["FAA", "Cen_Engagement", "FM_Theta_Alpha", "FM_Theta_Beta", "Cen_Theta_Beta"])

    # ------------------------------------------------------------------
    # 2–5. Time-domain features per temporal region
    # ------------------------------------------------------------------
    t_regions = [r for r in _TEMPORAL_REGIONS if r in region_map]
    region_sigs = _region_signals(epochs, region_map, t_regions)

    for region in t_regions:
        sig = region_sigs[region]    # (N, T)
        r   = region[:3]             # short prefix for feature names

        # Region mean PSD for spectral entropy
        ch_idx  = region_map[region]
        psd_r   = psd[:, ch_idx, :].mean(axis=1)  # (N, F)

        # 2. Hjorth
        h = _hjorth_batch(sig)                              # (N, 3)
        features.extend([h[:, 0], h[:, 1], h[:, 2]])
        names.extend([f"{r}_HjAct", f"{r}_HjMob", f"{r}_HjComp"])

        # 3. Spectral entropy
        se = _spectral_entropy_batch(psd_r, freqs)          # (N,)
        features.append(se)
        names.append(f"{r}_SpEnt")

        # 4. Permutation entropy
        pe = _perm_entropy_batch(sig)                       # (N,)
        features.append(pe)
        names.append(f"{r}_PeEnt")

        # 5. Skewness, kurtosis, ZCR
        st = _stats_batch(sig)                              # (N, 3)
        features.extend([st[:, 0], st[:, 1], st[:, 2]])
        names.extend([f"{r}_Skew", f"{r}_Kurt", f"{r}_ZCR"])

    # ------------------------------------------------------------------
    # 6. Aperiodic 1/f slope (per temporal region)
    # ------------------------------------------------------------------
    for region in t_regions:
        ch_idx = region_map[region]
        psd_r  = psd[:, ch_idx, :].mean(axis=1)  # (N, F)
        slope  = _aperiodic_slope_batch(psd_r, freqs)
        r      = region[:3]
        features.append(slope)
        names.append(f"{r}_1fSlope")

    # ------------------------------------------------------------------
    # 7. wPLI connectivity features
    # ------------------------------------------------------------------
    _wpli_pairs = [
        ("FrontalMidline", "Parietal",  "Theta"),   # FM↔Par theta
        ("FrontalMidline", "Parietal",  "Alpha"),   # FM↔Par alpha
        ("FrontalMidline", "Central",   "Theta"),   # FM↔Cen theta
        ("Central",        "Parietal",  "Alpha"),   # Cen↔Par alpha
        ("FrontalMidline", "Occipital", "Alpha"),   # FM↔Occ alpha
    ]
    all_sigs: dict[str, np.ndarray] = {}
    for r1, r2, band_name in _wpli_pairs:
        if r1 not in region_map or r2 not in region_map:
            features.append(np.zeros(epochs.shape[0], dtype=np.float32))
            names.append(f"wPLI_{r1[:3]}_{r2[:3]}_{band_name[:2]}")
            continue
        # Get or cache region-mean signals
        for rr in (r1, r2):
            if rr not in all_sigs:
                idx = region_map[rr]
                all_sigs[rr] = epochs[:, idx, :].mean(axis=1)  # (N, T)
        lo, hi = bands[band_name]
        w = _wpli_batch(all_sigs[r1], all_sigs[r2], srate, lo, hi)
        features.append(w)
        short = f"wPLI_{r1[:3]}_{r2[:3]}_{band_name[:2]}"
        names.append(short)

    return np.column_stack(features).astype(np.float32), names


# ---------------------------------------------------------------------------
# Fixed fallback bands (canonical location — other scripts may duplicate)
# ---------------------------------------------------------------------------

FIXED_BANDS = {
    "Delta":   (1.0,  4.0),
    "Theta":   (4.0,  7.5),
    "Alpha":   (7.5, 12.0),
    "Beta":   (12.0, 30.0),
    "Gamma":  (30.0, 45.0),
}


# ---------------------------------------------------------------------------
# Full-pipeline feature extraction (epochs → cached feature matrices)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from ml.pretrain_loader import PretrainDataDir

_DEFAULT_NORM_CACHE = (
    _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"
)


def _remap(labels: np.ndarray) -> np.ndarray:
    """Map raw task labels (e.g. 0, 2) to contiguous 0/1."""
    unique = sorted(set(labels.tolist()))
    lmap = {old: new for new, old in enumerate(unique)}
    return np.array([lmap[lb] for lb in labels], dtype=np.int64)


def _extract_feat(
    epochs: np.ndarray,
    srate: float,
    region_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Extract feature matrix from (N, C, T) epochs."""
    if epochs.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32), []
    freqs, psd = welch(
        epochs, fs=srate, nperseg=min(256, epochs.shape[-1]), axis=-1
    )
    return extract_features(epochs, psd, freqs, region_map, FIXED_BANDS,
                            srate=srate)


def _stress_labels_from_conditions(
    conditions: list[str],
    task_bidx: np.ndarray,
) -> np.ndarray:
    """Derive per-window binary stress labels from per-block condition names.

    conditions : one string per task block, sorted by onset
                 (e.g. 'HighStress_HighCog1022_Task')
    task_bidx  : per-window block index (0–3), from load_task_epochs

    Returns (N_windows,) int64 array — 1 = HighStress, 0 = LowStress.
    """
    block_stress: list[int] = []
    for cond in conditions:
        if "HighStress" in cond:
            block_stress.append(1)
        elif "LowStress" in cond:
            block_stress.append(0)
        else:
            raise ValueError(f"Cannot determine stress level from: {cond}")

    stress_y = np.empty(len(task_bidx), dtype=np.int64)
    for bidx, s_label in enumerate(block_stress):
        mask = task_bidx == bidx
        stress_y[mask] = s_label
    return stress_y


# ---------------------------------------------------------------------------
# Norm-cache helpers
# ---------------------------------------------------------------------------

def _norm_cache_key(
    dataset_path: Path, pids: list[str], win_tag: str = "",
) -> str:
    """Hash of manifest mtime + pid list + feature version (+ window tag)."""
    manifest = dataset_path / "manifest.json"
    mtime = str(manifest.stat().st_mtime) if manifest.exists() else "missing"
    raw = "|".join([mtime, str(sorted(pids)), "norm_v1", win_tag])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_norm_cache(
    cache_path: Path, expected_key: str
) -> dict[str, dict] | None:
    """Load cached features or return None if stale/missing."""
    if not cache_path.exists():
        return None
    try:
        npz = np.load(cache_path, allow_pickle=False)
        if str(npz["cache_key"]) != expected_key:
            return None
        pids = list(npz["pids"])
        feat_names = list(npz["feat_names"])
        data: dict[str, dict] = {}
        for pid in pids:
            stress_key = f"{pid}_task_stress_y"
            data[pid] = dict(
                task_X=npz[f"{pid}_task_X"],
                task_y=npz[f"{pid}_task_y"],
                task_stress_y=npz[stress_key] if stress_key in npz else None,
                task_bidx=npz[f"{pid}_task_bidx"],
                forest_X=npz[f"{pid}_forest_X"],
                forest_bidx=npz[f"{pid}_forest_bidx"],
                fix_X=npz[f"{pid}_fix_X"],
                feat_names=feat_names,
            )
        return data
    except Exception:
        return None


def _save_norm_cache(
    cache_path: Path, key: str, data: dict[str, dict], feat_names: list[str]
) -> None:
    """Save features to compressed .npz cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cache_key": np.array(key),
        "pids": np.array(sorted(data.keys())),
        "feat_names": np.array(feat_names),
    }
    for pid, d in data.items():
        arrays[f"{pid}_task_X"] = d["task_X"]
        arrays[f"{pid}_task_y"] = d["task_y"]
        arrays[f"{pid}_task_stress_y"] = d["task_stress_y"]
        arrays[f"{pid}_task_bidx"] = d["task_bidx"]
        arrays[f"{pid}_forest_X"] = d["forest_X"]
        arrays[f"{pid}_forest_bidx"] = d["forest_bidx"]
        arrays[f"{pid}_fix_X"] = d["fix_X"]
    np.savez_compressed(cache_path, **arrays)


def load_all_features(
    data_dir: PretrainDataDir,
    pids: list[str],
    srate: float,
    region_map: dict[str, np.ndarray],
    cache_path: Path | None = None,
    win_tag: str = "",
) -> dict[str, dict]:
    """Load & extract features for task, forest, and fixation blocks.

    Uses a disk cache so re-runs skip the expensive extraction step.

    Parameters
    ----------
    cache_path : Path, optional
        Override the default feature cache location.  Useful when sweeping
        window configurations (each needs its own cache file).
    win_tag : str, optional
        Extra string folded into the cache key so different window configs
        produce distinct keys even when stored in separate files.

    Returns
    -------
    dict keyed by pid, each containing:
        task_X       (N_task, F)  feature matrix
        task_y       (N_task,)    binary labels {0, 1}
        task_bidx    (N_task,)    block temporal index 0–3
        forest_X     (N_for, F)   features
        forest_bidx  (N_for,)     block index 0–3
        fix_X        (N_fix, F)   features
        feat_names   list[str]
    """
    effective_cache = cache_path or _DEFAULT_NORM_CACHE
    key = _norm_cache_key(data_dir.root, pids, win_tag=win_tag)
    cached = _load_norm_cache(effective_cache, key)
    if cached is not None:
        feat_names = next(iter(cached.values()))["feat_names"]
        print(f"Loaded from cache  ({len(cached)} participants, "
              f"{len(feat_names)} features)  [key={key}]")
        return cached

    data: dict[str, dict] = {}
    for i, pid in enumerate(pids):
        print(f"  [{i + 1:>2}/{len(pids)}] {pid} ...", end=" ", flush=True)

        # Task
        task_ep, task_lab_raw, task_bidx = data_dir.load_task_epochs(pid)
        task_y = _remap(task_lab_raw)
        task_X, feat_names = _extract_feat(task_ep, srate, region_map)

        # Stress labels (derived from condition names)
        conditions = data_dir.load_task_conditions(pid)
        task_stress_y = _stress_labels_from_conditions(conditions, task_bidx)

        # Forest
        forest_ep, forest_bidx = data_dir.load_forest_epochs(pid)
        forest_X, _ = _extract_feat(forest_ep, srate, region_map)

        # Fixation
        fix_ep = data_dir.load_fixation_epochs(pid)
        fix_X, _ = _extract_feat(fix_ep, srate, region_map)

        data[pid] = dict(
            task_X=task_X,
            task_y=task_y,
            task_stress_y=task_stress_y,
            task_bidx=task_bidx,
            forest_X=forest_X,
            forest_bidx=forest_bidx,
            fix_X=fix_X,
            feat_names=feat_names,
        )
        print(
            f"task={task_X.shape[0]:>4}  "
            f"forest={forest_X.shape[0]:>4}  "
            f"fix={fix_X.shape[0]:>3}"
        )

    _save_norm_cache(effective_cache, key, data, feat_names)
    print(f"  Cached to {effective_cache.relative_to(_REPO_ROOT)}")
    return data
