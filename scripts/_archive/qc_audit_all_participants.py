"""QC audit: compute EEG quality metrics for all 47 participants.

Applies the same criteria documented in config/pretrain_qc.yaml uniformly
to every participant, including currently-excluded ones.  Prints a full
table so exclusion decisions can be reviewed for consistency.

Run with:
    C:\\vr_tsst_2025\\.venv\\Scripts\\python.exe scripts/_qc_audit_all_participants.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from eeg.eeg_windower import WindowConfig  # noqa: E402
from ml.pretrain_loader import PretrainDataDir  # noqa: E402

warnings.filterwarnings("ignore")

DATASET_DIR = Path("C:/vr_tsst_2025/output/matb_pretrain/continuous")

# Flag thresholds — derived from criteria stated in pretrain_qc.yaml
RMS_XFOLD_THRESH   = 5.0   # > 5× cohort median RMS → EEG_QUALITY
COND_RATIO_THRESH  = 2.0   # > 2× RMS between conditions → EEG_QUALITY
ALPHA_RATIO_THRESH = 5.0   # > 5× alpha power between conditions → EEG_QUALITY
BALANCE_THRESH     = 0.70  # min/max class ratio < 0.70 → CLASS_IMBALANCE
CHANCE_THRESH      = 0.50  # within-participant AUC < 0.50 → LABEL_ERROR / no signal
LOW_CV_THRESH      = 0.10  # epoch-RMS CV < 0.10 → FLAT (low variance)

# Non-overlapping windows for QC — eliminates train/test leakage
# from the 75%-overlapping default step (0.5 s window, 2.0 s step).
_QC_WIN = WindowConfig(window_s=2.0, step_s=2.0, srate=128.0)

print(f"Loading from {DATASET_DIR} ...")
data_dir = PretrainDataDir(DATASET_DIR)
srate = data_dir.srate()
pids  = data_dir.available_pids()
print(f"Found {len(pids)} participants: {' '.join(pids)}\n")

rows: list[tuple] = []
for pid in pids:
    ep, lbl, blk = data_dir.load_task_epochs(pid, win=_QC_WIN)
    N, C, T = ep.shape

    # Binary remap
    uniq = sorted(set(lbl.tolist()))
    lmap = {v: i for i, v in enumerate(uniq)}
    y = np.array([lmap[v] for v in lbl], dtype=np.int64)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    balance = min(n0, n1) / max(n0, n1) if max(n0, n1) > 0 else 0.0

    # Epoch RMS
    rms = np.sqrt((ep ** 2).mean(axis=(1, 2)))
    med_rms = float(np.median(rms))
    rms_cv = float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0
    rms0 = float(np.median(rms[y == 0])) if n0 > 0 else np.nan
    rms1 = float(np.median(rms[y == 1])) if n1 > 0 else np.nan
    cond_ratio = (max(rms0, rms1) / min(rms0, rms1)
                  if min(rms0, rms1) > 0 else np.nan)

    # Alpha power ratio between conditions (Welch on channel-mean signal)
    freqs, psd = welch(ep.mean(axis=1), fs=srate,
                       nperseg=min(256, T), axis=-1)   # (N, F)
    amask = (freqs >= 8) & (freqs < 13)
    alpha = psd[:, amask].mean(axis=1)
    a0 = float(np.median(alpha[y == 0])) if n0 > 0 else np.nan
    a1 = float(np.median(alpha[y == 1])) if n1 > 0 else np.nan
    alpha_ratio = (max(a0, a1) / min(a0, a1)
                   if min(a0, a1) > 0 else np.nan)

    # Within-participant repeated-CV AUC (PSD features, per-participant z-score)
    # Non-overlapping epochs + repeated shuffled folds → stable separability.
    X = StandardScaler().fit_transform(psd.reshape(N, -1))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_aucs: list[float] = []
    for tr, te in cv.split(X, y):
        if len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(C=0.01, max_iter=500, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        fold_aucs.append(
            float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        )
    within_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    within_auc_std = float(np.std(fold_aucs)) if fold_aucs else 0.0

    # Drift check: compare AUC on early vs. late blocks
    unique_blks = sorted(set(blk.tolist()))
    drift_flag = False
    if len(unique_blks) >= 2:
        early_mask = blk == unique_blks[0]
        late_mask = blk == unique_blks[-1]
        if early_mask.sum() > 5 and late_mask.sum() > 5:
            rms_early = float(np.median(rms[early_mask]))
            rms_late = float(np.median(rms[late_mask]))
            drift_ratio = max(rms_early, rms_late) / min(rms_early, rms_late)
            drift_flag = drift_ratio > 3.0

    rows.append((pid, N, n0, n1, balance, med_rms, rms_cv, cond_ratio,
                 alpha_ratio, within_auc, within_auc_std, drift_flag))
    print(f"  {pid} done  (N={N}, rms_cv={rms_cv:.3f}, "
          f"within_AUC={within_auc:.3f}±{within_auc_std:.3f})")

cohort_med_rms = float(np.median([r[5] for r in rows]))
cohort_med_cv  = float(np.median([r[6] for r in rows]))
print(f"\nCohort median RMS: {cohort_med_rms:.5f}")
print()

# Print full table
header = (f"{'PID':<6} {'N':>5} {'n0':>5} {'n1':>5} {'bal':>5} "
          f"{'medRMS':>8} {'rmsCV':>7} {'xCohort':>8} {'condRatio':>10} "
          f"{'alphaRatio':>11} {'withinAUC':>10} {'aucSD':>6}  flags")
print(header)
print("-" * 133)

flagged: list[tuple[str, list[str]]] = []

for (pid, N, n0, n1, bal, mrms, rcv, cratio,
     aratio, wauc, wauc_sd, drift) in rows:
    rms_x = mrms / cohort_med_rms
    flags: list[str] = []
    if rms_x > RMS_XFOLD_THRESH:
        flags.append("RMS-OUTLIER")
    if rcv < LOW_CV_THRESH:
        flags.append("FLAT")
    if not np.isnan(cratio) and cratio > COND_RATIO_THRESH:
        flags.append("COND-AMP")
    if not np.isnan(aratio) and aratio > ALPHA_RATIO_THRESH:
        flags.append("ALPHA-RATIO")
    if bal < BALANCE_THRESH:
        flags.append("CLASS-IMBAL")
    if wauc < CHANCE_THRESH:
        flags.append("BELOW-CHANCE")
    if drift:
        flags.append("DRIFT")

    flag_str = " ".join(flags) if flags else "ok"
    print(f"{pid:<6} {N:>5} {n0:>5} {n1:>5} {bal:>5.2f} "
          f"{mrms:>8.5f} {rcv:>7.3f} {rms_x:>7.1f}x {cratio:>10.2f}x "
          f"{aratio:>11.1f}x {wauc:>10.3f} {wauc_sd:>6.3f}  {flag_str}")

    if flags:
        flagged.append((pid, flags))

# --- Save to CSV ---
csv_path = _REPO_ROOT / "results" / "qc_audit.csv"
csv_rows = []
for (pid, N, n0, n1, bal, mrms, rcv, cratio, aratio, wauc, wauc_sd, drift) in rows:
    rms_x = mrms / cohort_med_rms
    flags: list[str] = []
    if rms_x > RMS_XFOLD_THRESH:
        flags.append("RMS-OUTLIER")
    if rcv < LOW_CV_THRESH:
        flags.append("FLAT")
    if not np.isnan(cratio) and cratio > COND_RATIO_THRESH:
        flags.append("COND-AMP")
    if not np.isnan(aratio) and aratio > ALPHA_RATIO_THRESH:
        flags.append("ALPHA-RATIO")
    if bal < BALANCE_THRESH:
        flags.append("CLASS-IMBAL")
    if wauc < CHANCE_THRESH:
        flags.append("BELOW-CHANCE")
    if drift:
        flags.append("DRIFT")
    csv_rows.append({
        "pid": pid, "N": N, "n0": n0, "n1": n1, "balance": f"{bal:.2f}",
        "med_rms": f"{mrms:.5f}", "rms_cv": f"{rcv:.3f}",
        "x_cohort": f"{rms_x:.1f}",
        "cond_ratio": f"{cratio:.2f}", "alpha_ratio": f"{aratio:.1f}",
        "within_auc": f"{wauc:.3f}", "auc_sd": f"{wauc_sd:.3f}",
        "flags": " ".join(flags) if flags else "ok",
    })

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Saved {len(csv_rows)} rows to {csv_path}")
print()

print()
print("=== Participants with QC flags (candidates for exclusion) ===")
if flagged:
    for pid, fl in flagged:
        print(f"  {pid}: {' '.join(fl)}")
else:
    print("  none")
print(f"\n{len(flagged)}/{len(rows)} participants flagged")
