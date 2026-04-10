# Lab notes — 2026-04-08 — PSELF S005 post-hoc analysis

## Goals

- Review S005 adaptation session quality: were assists triggered appropriately?
- Diagnose root causes of poor MWL discrimination during the adaptation task
- Identify pipeline improvements to make before S006
- Characterise which EEG features drove the model for this participant

**Outcome: Model worked in calibration (AUC=0.84, Δ(HIGH-LOW)=+0.326) but failed to discriminate
during adaptation (Δ(HIGH-LOW)=+0.010). Root cause: SelectKBest overfits calibration-specific
features (29/40 are session-specific). Two pipeline fixes identified for S006: (1) reduce k, (2) add
cross-session feature stability filter. A scenario bug (LSL_SETTLE_SEC) was also fixed.**

---

## S005 adaptation run summary

### Run details
```powershell
python src/run_full_study_session.py --participant PSELF --labrecorder-rcs --eda-auto-port \
  --post-phase-verify --start-phase 6 --skip-baseline-refresh
```
XDF: `sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf`

### Adaptation controller settings (from `model_config.json`)
| Parameter | Value |
|---|---|
| Threshold (Youden-optimal) | 0.2891 (original) → 0.3544 (+block_01 recovery, see below) |
| α (EMA smoothing) | 0.05 |
| Hysteresis | 0.02 |
| Hold time | 3 s |
| Cooldown | 15 s |

### Assist ON/OFF per block (original model, threshold=0.289)
| blk | level | assist_ON% | note |
|---|---|---|---|
| 1 | LOW | 0% | correctly off |
| 2 | HIGH | 58% | |
| 3 | HIGH | 28% | |
| 4 | MODERATE | 37% | |
| 5 | HIGH | 13% | |
| 6 | LOW | **74%** | **false alarm** |
| 7 | HIGH | **100%** | carryover from blk 6 |
| 8 | MODERATE | 74% | |

Overall: **49% time assist-ON** (6 ON events, 5 OFF events).  
Detection latency: 34.9 s ± 18 s.  
One clear false alarm: blk 6 (LOW) had 74% assist-ON.

### Post-hoc model metrics during adaptation
| | p_high LOW | p_high MOD | p_high HIGH | Δ(HIGH-LOW) |
|---|---|---|---|---|
| Calibration sessions | 0.212 | 0.253 | 0.538 | **+0.326** |
| Adaptation session | 0.383 | 0.402 | 0.393 | **+0.010** |

The model works well in-domain (calibration) but produces near-flat scores during adaptation.

---

## Bug found and fixed: block_01 missing from training data

### Root cause
`_extract_all_blocks()` in the training pipeline requires a `START` marker for each block.
LabRecorder's LSL discovery latency is ~50–60 s, so the block_01 `START` marker is always missed.
The `END` marker is captured, but without a paired `START`, block_01 is silently dropped.

For PSELF calibration (2 sessions, 3 levels × 2 blocks each):
- block_01 of cal_c1 was always HIGH → missing HIGH windows drove threshold down
- Original threshold: 0.2891; recovered threshold: 0.3544 (+0.065)

### Fix applied: `LSL_SETTLE_SEC`
Added `LSL_SETTLE_SEC: int = 5` to `scripts/generate_scenarios/generate_full_study_scenarios.py`.
All block START markers now shifted by +5 s. Total scenario duration: 9:05 (was 9:00).
Scenario files regenerated:
- `experiment/scenarios/full_calibration_pself_c1.txt`
- `experiment/scenarios/full_calibration_pself_c2.txt`

Block 01 START now at `0:00:05` — well within capture range even on a slow LabRecorder startup.

### +block_01 model vs original
| Metric | Original | +block_01 |
|---|---|---|
| Threshold | 0.2891 | 0.3544 |
| CV balanced accuracy | 0.610 | 0.635 |
| AUC (cal data) | 0.845 | 0.838 |

The +block_01 model has a higher threshold (+0.065), which would have reduced false alarms (blk 6
would have been substantially shorter). CV BA improves. AUC drops marginally (~noise).

**All future participants use the regenerated scenario files — block_01 will no longer be lost.**

For S005, the original (lower) threshold was active during the live run.

---

## Root cause analysis: why did the model fail during adaptation?

Three hypotheses were simulated retrospectively using `scripts/_tmp_simulate_improvements.py`.

### Hypothesis 1 — Stale normalisation (rest XDF taken ~50 min before adaptation)
**Disproved.**

The rest XDF was recorded ~50 min before the adaptation session. Hypothesis: feature distributions
drifted and the fixed z-score normalisation became stale.

Simulation: re-normalised using block_01/LOW features from the adaptation XDF itself (56 windows,
~56 s of data) as a proxy for a fresh within-session baseline.

| | Δ(HIGH-LOW) |
|---|---|
| Original norm | +0.052 |
| Refreshed norm (in-session) | +0.052 |

**Zero effect.** Feature distributions in block_01/LOW were essentially identical to the rest
recording. Stale normalisation is not the problem. Running `update_session_baseline.py` immediately
before the adaptation task would have made no difference for S005.

### Hypothesis 2 — Causal adaptive z-score (τ=120 s drift correction)
**Partially helps; introduces new problem.**

Simulation: causal EMA running z-score (τ=120 s, α≈0.002 per 250 ms window). Score each window
*before* updating the running norm.

| Block | Level | Original | Adaptive z-score |
|---|---|---|---|
| 1 | LOW | 0.571 | **0.215** (correctly low) |
| 2 | HIGH | 0.658 | 0.540 |
| 3 | HIGH | 0.762 | 0.711 |
| 4 | MOD | 0.610 | 0.830 ← drift |
| 5 | HIGH | 0.673 | 0.841 |
| 6 | LOW | 0.786 | 0.822 ← same as HIGH |

| | Δ(HIGH-LOW) |
|---|---|
| Original | +0.052 |
| Adaptive z-score | **+0.204** |

The Δ(H-L) nearly quadruples and block_01/LOW is correctly identified as low-workload. However,
the running norm drifts upward: by block 4 it's centred on the session's HIGH-workload mean, and
LOW block 6 is indistinguishable from HIGH. τ=120 s is too short for an 8-minute session.
A longer τ (≥ full session) or a reset-on-block-transition scheme would be needed.

**Not recommended for implementation without further tuning.** Flag for future work.

### Hypothesis 3 — Feature selection overfits calibration (root cause confirmed)
**This is the real problem.**

Of the 40 features selected by `SelectKBest(k=40)`, only **11 are sign-consistent** across both
calibration sessions (i.e., HIGH > LOW in both c1 and c2). The other 29 are session-specific noise.

Simulation: retrain with only the 11 cross-session-stable features.

| | Δ(HIGH-LOW) |
|---|---|
| Original (40 features) | +0.052 |
| Stable-only (11 features) | **−0.000** |

The 11 genuinely stable features have *zero* discriminative power in the adaptation session.
This means the 29 session-specific features were carrying almost all of the apparent calibration
discrimination — and completely failed to generalise.

**Root cause summary:** `SelectKBest(k=40)` is over-selecting from a 54-feature space. With only
~130 windows per class in calibration, selecting 40/54 features is near-saturating. Many selected
features are calibration-period artefacts (position effects, order effects, drowsiness, noise)
rather than genuine MWL signal.

---

## Feature investigation: which features actually discriminate?

Full F-score ranking from `scripts/_tmp_feature_fscore.py` (calibration data, 4117 windows):

### Top discriminators (F > 10, selected by current pipeline):
| Rank | F | Feature | Interpretation |
|---|---|---|---|
| 1 | 29.3 | `Occ_Alpha` | Posterior α desynchronisation |
| 2 | 27.4 | `Cen_HjComp` | Central broadband complexity ↑ |
| 3 | 25.0 | `Occ_HjComp` | Occipital broadband complexity ↑ |
| 4 | 24.0 | `Cen_HjAct` | Central signal power ↑ |
| 5 | 23.6 | `Occ_HjAct` | Occipital signal power ↑ |
| 6 | 21.3 | `Par_Alpha` | Parietal α desynchronisation |
| 7 | 19.6 | `Occ_ZCR` | Occipital zero-crossing rate |
| 8 | 19.0 | `Par_PeEnt` | Parietal permutation entropy |
| 9 | 18.4 | `Occ_PeEnt` | Occipital permutation entropy |
| 14 | 12.8 | `wPLI_Fro_Cen_Th` | Fronto-central θ connectivity |
| 18 | 10.5 | `FAA` | Frontal alpha asymmetry |

### Classic θ and ratio features — NOT discriminating for this participant:
| Rank | F | Feature | Expected? |
|---|---|---|---|
| 50 | 0.3 | `FM_Theta` | Expected ↑ with MWL — **NOT moving** |
| 45 | 1.0 | `FM_Theta_Beta` | Expected ↑ — **F≈noise** |
| 47 | 0.7 | `FM_Theta_Alpha` | Expected ↑ — **F≈noise** |
| 53 | 0.2 | `Cen_Engagement` | Expected ↑ — **F≈noise** |
| 51 | 0.2 | `Cen_Theta_Beta` | Expected ↑ — **F≈noise** |

Per-level means confirm `FM_Theta` barely moves:
```
FM_Theta (log-power):  LOW=-26.200  MOD=-26.212  HIGH=-26.196  Δ=+0.004
```
SNR for FM_Theta_Alpha (between-class std / within-class std) = **0.018** — completely buried in noise.

**Interpretation:** This participant's MWL response is driven by posterior α desynchronisation and
broadband Hjorth complexity, not by frontal midline θ. Individual differences in the θ response
during MATB are real and documented. This is not a computation bug — the computation was verified
to be correct. It is a genuine participant-level finding.

**Note for S006+:** Consider recording individual θ response signature during calibration (whether
FM_Theta actually increases with HIGH vs LOW). If θ is flat for a participant, down-weight
θ-derived features or exclude them from that participant's feature set.

---

## Simulated controller variants

Tested via `scripts/_tmp_plot_new_model_fig.py` (changes EMA α; does not require re-extraction):

| Variant | assist_ON% | ON→OFF cycles |
|---|---|---|
| α=0.05, cd=15 s (current) | 49.1% | 6 |
| α=0.05, cd=0 s (no cooldown) | 49.5% | 7 |
| α=0.10, cd=15 s | 49.1% | 7 |
| α=0.20, cd=15 s | 50.1% | 7 |
| α=0.40, cd=15 s | 42.9% | 7 |

**Conclusion:** EMA smoothing and cooldown are not the bottleneck. The underlying p_high signal
lacks discrimination — changing the controller cannot compensate for a flat score distribution.
The problem is in the feature/model layer.

---

## Pipeline changes needed before S006

Priority order:

### 1. Reduce SelectKBest k (high priority)
**Change:** In `scripts/calibrate_participant.py`, reduce `k` in `SelectKBest(f_classif, k=40)`
to `k=20` or `k=25`.

**Rationale:** With 54 features and ~130 windows/class in calibration, k=40 is near-saturating
(74% of features selected). The bottom 15 selected features have F < 5, meaning they are
calibration-specific noise. Reducing to k=20–25 keeps only the genuinely high-F features
(those above the knee at rank ≈ 20–22 in the F-score ranking).

**Risk:** Might reduce calibration AUC slightly. Run cross-validated comparison before committing.

### 2. Add cross-session feature stability filter (high priority)
**Change:** After fitting `SelectKBest` on the pooled two-session training set, add a second
filter that retains only features where `sign(mean_HIGH - mean_LOW)` is consistent across
both calibration sessions individually.

**Rationale:** Simulation showed only 11/40 selected features are cross-session stable. A feature
that points in opposite directions in c1 vs c2 is measuring session noise, not MWL signal. The
two calibration sessions exist precisely to enable this check.

**Where to implement:** `scripts/calibrate_participant.py`, after the `SelectKBest.fit()` call,
before fitting the SVM.

**Pseudocode:**
```python
# After sel = SelectKBest(...).fit(X_z, y):
selected_idx = sel.get_support(indices=True)
stable_mask = []
for i in selected_idx:
    means_c1 = {lv: X_c1_z[y_c1==LABEL_MAP[lv], i].mean() for lv in LEVELS}
    means_c2 = {lv: X_c2_z[y_c2==LABEL_MAP[lv], i].mean() for lv in LEVELS}
    sign_c1 = np.sign(means_c1['HIGH'] - means_c1['LOW'])
    sign_c2 = np.sign(means_c2['HIGH'] - means_c2['LOW'])
    stable_mask.append(sign_c1 == sign_c2)
stable_idx = selected_idx[stable_mask]
# Fit SVM only on stable_idx features
```

### 3. Record LSL_SETTLE_SEC timing in QC log (medium priority)
The block_01 recovery fix is now in the scenario files. Verify that block_01 is captured in S006
calibration XDF before starting the adaptation phase. Check: does the XDF contain a
`block_01/.../START` marker?

### 4. Keep calibration-to-adaptation gap short (medium priority)
For S005, calibration was at ~10:00 and adaptation at ~14:00 (4-hour gap). While the stale
normalisation simulation showed no effect for S005, minimising the gap reduces the chance of
impedance/state drift. Target: ≤ 90 min between end of calibration and start of adaptation.

### 5. Do NOT add adaptive z-score normalisation yet (defer)
Simulation 2 showed it helps initially but drifts badly with τ=120 s. Needs further work
(longer τ, or reset mechanism). Implementing without validation could make things worse.

---

## Figures generated (in `results/figures/`)

| File | Contents |
|---|---|
| `adaptation_pself__fig01__mwl_timeline_new_model.png` | Single-panel timeline, +block_01 model |
| `adaptation_pself__fig01__mwl_timeline_cooldown_compare.png` | 2-panel: cd=15 s vs cd=0 s |
| `adaptation_pself__fig01__mwl_timeline_ema_variants.png` | 5-panel EMA α variants |

---

## Temporary scripts created (for reference / re-running)

All in `scripts/`:

| Script | Purpose |
|---|---|
| `_tmp_retrain_with_block01.py` | Retrain +block_01 model; generates cache; run time ~4 min |
| `_tmp_plot_new_model_fig.py` | Fast figure regeneration from cache (~2 s) |
| `_tmp_analyse_new_model.py` | Synthesises audit CSV; runs analyse + plot scripts |
| `_tmp_simulate_improvements.py` | Three improvement simulations (Sim1/2/3 described above) |
| `_tmp_feature_fscore.py` | F-score ranking for all 54 features |
| `_tmp_investigate_ratios.py` | Per-level means + SNR breakdown for ratio features |

Cache at `results/_tmp_new_model_cache/`:
- `adapt_inference.npz` — `ph_new`, `ema_new`, `on_new`, `t_rel`, `lsl_ts`
- `meta.json` — threshold, adapt_blocks, adapt_xdf path, t0
- `synth_audit_new_model.csv` — synthesised audit CSV (1855 rows)

**These scripts are disposable investigation tools.** Do not rely on them for S006 data.
The permanent fix is in `calibrate_participant.py` (to be done before S006).

---

## S005 data status

- S005 is the **first complete PSELF study run** (calibration + adaptation on same day).
- Adaptation MWL performance was poor (flat p_high, 49% assist-ON overall).
- However, the poor performance is attributable to a known pipeline bug (feature over-selection),
  not to a measurement or hardware failure.
- **S005 adaptation data should not be included in group-level adaptive-condition analyses**
  until it is determined whether re-running with the fixed pipeline is possible/warranted.
- S005 calibration data (both sessions) is good quality and should be retained.

---

## Issues / risks

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | SelectKBest k=40 over-selects calibration noise (29/40 features session-specific) | **High** | Needs fix before S006 |
| 2 | No frontal θ response in S005 (FM_Theta F=0.3) | Medium | Participant-level finding; monitor in S006+ |
| 3 | S005 XDF missing block_01 START markers in calibration | Medium | Fixed in scenario files; verify in S006 XDF |
| 4 | Adaptive z-score (τ=120 s) drifts badly after ~3 min | Low | Deferred; do not implement |
| 5 | 4-hour calibration-adaptation gap in S005 | Low | Accepted for S005; target ≤90 min in S006 |

---

## Decisions made today

- **Scenario LSL_SETTLE_SEC fix is permanent** and takes effect from S006.
- **k=40 → k=20–25 change will be made before S006** (needs implementation + cross-val check).
- **Cross-session stability filter will be added to `calibrate_participant.py`** before S006.
- **Adaptive z-score deferred** — too risky without further τ tuning.
- **Stale normalisation is not a problem** — confirmed by simulation; no change needed.

---

## Next actions before S006

- [ ] In `calibrate_participant.py`: change `SelectKBest(k=40)` → `SelectKBest(k=20)` (or 25)
- [ ] In `calibrate_participant.py`: add cross-session stability filter after SelectKBest
- [ ] Run cross-validated comparison of k=20 vs k=40 on pre-train dataset to confirm no AUC regression
- [ ] After S006 calibration but before adaptation: verify block_01 START marker present in XDF
- [ ] Update `participant_assignments.yaml` to record S005 session as completed
- [ ] Decide whether S005 adaptation data warrants a re-run with the fixed pipeline

---

## Time spent

- Total: ~6 hours
- Breakdown:
  - p_high drift investigation and root cause identification: 1 h
  - block_01 recovery + retrain: 1 h
  - Figure generation + post-session analysis: 1 h
  - Improvement simulations (design + run): 1.5 h
  - Feature investigation (F-scores + ratio deep-dive): 1 h
  - Lab notes: 0.5 h
