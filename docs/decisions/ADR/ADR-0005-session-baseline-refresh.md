# ADR-0005: Session-level resting baseline refresh before adaptation blocks

## Status

Accepted — 2026-03-27

## Context

ADR-0004 established that EEG features are z-scored relative to a resting-state
baseline recorded once at the start of the calibration phase.  That baseline
mean/std is frozen in `norm_stats.json` and applied to all subsequent task
epochs.

During self-pilot testing (sub-PSELF, 2026-03-26/27), empirical measurement
revealed that this single-session baseline fails within the same study visit
when calibration and adaptation are separated by ~30 minutes:

### Measured temporal drift (sub-PSELF, ses-S001)

| Session | Gap from cal end | N windows | Mean z (sel. features) | Std z | >3σ outliers |
|---|---|---|---|---|---|
| cal_c1 (training) | 0 s | 2 043 | −0.18 | 3.15 | 4.5% |
| adaptation (held-out) | 1 778 s (~29.6 min) | 1 845 | +0.37 | 5.59 | **16.4%** |

**The adaptation session had 3.6× more feature-window pairs exceeding 3σ
above the resting baseline than the calibration session.**  This is the
direct cause of P(HIGH) saturation observed in the held-out eval:

| Level | P(HIGH) mean — cal session | P(HIGH) mean — adaptation |
|---|---|---|
| LOW | 0.155 | 0.751 |
| MODERATE | 0.248 | 0.939 |
| HIGH | 0.551 | 0.997 |

The model's discrimination AUC dropped from 0.889 on training data to 0.702
on the held-out adaptation data, and the P(HIGH) values for LOW and MODERATE
blocks became practically useless as absolute thresholds.

LSL clock analysis confirms the sessions were **contiguous within the same
recording visit**:

- cal_c1: LSL t = 103 035 – 103 591  (9.3 min)
- adaptation: LSL t = 105 370 – 105 875  (8.4 min)
- Gap: **1 778.5 s (29.6 min)** of inter-session time (questionnaires + setup)

The drift is therefore not an artefact of cross-day recording.  It occurs
within a single two-hour lab visit, most likely driven by:

1. **Electrode impedance creep** — gel partitioning and minor cap movement
   over 30 min alter the contact impedance and thereby the absolute signal
   amplitude/spectral shape.
2. **Genuine state change** — 30+ min of prior cognitive effort leaves the
   participant in a physiologically different state than at session start.
3. **Alpha rebound suppression** carrying over from intensive calibration.

A `--no-norm` control confirmed the baseline is load-bearing: without it,
P(HIGH) collapses to 0.000 for all windows (AUC 0.571).  The normalisation
cannot simply be removed.

## Decision

Add a **session-level baseline refresh step** to the adaptation session
protocol:

1. At the start of each adaptation session, record a fresh **60-second
   eyes-open resting baseline** via a short scenario marker segment.
2. After recording, run `scripts/session/update_session_baseline.py --xdf <xdf>
   --model-dir <model>` to recompute `norm_mean` and `norm_std` from this
   fresh resting data and overwrite `norm_stats.json`.
3. `mwl_estimator.py` then loads the updated `norm_stats.json` as usual —
   no changes to the inference loop.

The feature **selector** (`selector.pkl`) and classifier **weights**
(`pipeline.pkl`) are not changed.  Only the normalisation reference shifts.

## Rationale

1. **Directly addresses the measured root cause.**  The drift is in the
   feature distribution relative to resting state.  Re-anchoring the
   resting reference at the start of the adaptation session removes the
   confound.

2. **Consistent with ADR-0004's conceptual design.**  ADR-0004 chose
   calibration-based normalisation because it provides a "pure resting
   state" reference with no task contamination.  This ADR extends that
   principle across session boundaries — the reference is still resting
   state, just a more temporally proximate one.

3. **Minimal protocol change.**  60 s of eyes-open rest is already familiar
   to participants (they completed ~120 s at the start of calibration).
   It requires no additional equipment or instructions beyond what is
   already present.

4. **Empirically motivated.**  The decision is based on measured data from
   a self-pilot (sub-PSELF), not a theoretical concern.  The 3.6× increase
   in >3σ outliers and the AUC degradation from 0.889→0.702 provide
   quantitative justification.

5. **Conservative scope.**  The fix is confined to a single JSON file
   overwrite and a short new utility script.  It does not require
   retraining, does not change the model architecture, and is fully
   reversible (original norm_stats.json can be kept as a backup).

## Alternatives considered

### Rolling z-baseline (adaptive online normalisation)

A rolling backward-looking window of N samples was simulated with window
sizes 10–103.  All windows reduced AUC vs the fixed baseline (best: w=103
gave AUC=0.697 vs 0.702 fixed).  The failure mode is structural: in a
sustained HIGH-workload block, the rolling baseline itself shifts upward,
erasing the signal of interest.

### Accept saturation; use relative changes

The ranking is preserved (AUC=0.702 > 0.5), so relative P(HIGH) changes
are meaningful.  However, absolute thresholding (e.g. trigger assistance
when P(H) > 0.7) becomes impossible, and the adaptation controller would
need to be redesigned around rank-based signals.  This is a viable
fallback but adds complexity to the controller and complicates
interpretation in the thesis.

### Session-level linear recalibration of classifier weights

Fit a new LogReg head on the fresh resting data only.  Rejected: resting
data contains only one class (no workload variation), so weights cannot be
updated from it alone.

## Consequences

**Positive:**
- Removes the primary source of P(HIGH) saturation observed in self-pilot.
- Preserves the clean conceptual framing of ADR-0004 (resting reference,
  no task contamination).
- Allows the same absolute P(HIGH) scale to be used for threshold-based
  adaptation control.
- Adds only ~60 s to the adaptation session setup, well within protocol
  time budget.

**Negative:**
- Introduces a procedural step that must be executed correctly before each
  adaptation session.  A missed or corrupted baseline refresh would silently
  undermine model performance.
- `norm_stats.json` is now a mutable run-time artefact, not a static training
  output.  This must be reflected in data management documentation.

**Mitigation:**
- `update_session_baseline.py` will print a clear confirmation with the
  computed mean/std and a sanity flag if the norms look implausible (e.g.
  mean feature z relative to the old baseline > 3).
- The original pre-adaptation `norm_stats.json` will be backed up as
  `norm_stats_pretrain.json` before overwrite.

## Implementation

- New utility script: `scripts/session/update_session_baseline.py`
- No changes to `pipeline.pkl`, `selector.pkl`, or model training code.
- `mwl_estimator.py` already reads `norm_stats.json` on startup — no
  changes needed if the file is updated before the estimator is launched.

## References

- ADR-0004: `docs/decisions/ADR/ADR-0004-causal-normalisation-strategy.md`
- Self-pilot data: `sub-PSELF_ses-S001_task-matb_acq-adaptation_physio_old1.xdf`
- Self-pilot drift analysis: `docs/lab-notes/2026-03-27.md` (to be written)
- Shenoy et al. (2013) — adaptation strategies for EEG-based BCIs under
  non-stationarity: general reference supporting within-session recalibration.
