# Design choice: Log-scale event rates with symmetric difficulty extrapolation

**Decision**  
All MATB subtask event rates and resman drain are scaled exponentially (log scale)
over the difficulty range d ∈ [−0.8, +1.8], with `DEFAULT_DELTA = 0.8` so that LOW
and HIGH blocks are exactly ±0.8 from the staircase-converged d_final.

---

## Background

The staircase calibration converges a scalar `d_final ∈ [0, 1]` representing
individual workload threshold.  Calibration and adaptation scenario files
pre-schedule three workload levels per participant:

- **LOW** = d_final − delta
- **MODERATE** = d_final
- **HIGH** = d_final + delta

Prior to this change, the system used linear interpolation (`_lerp`) for all
parameters, `delta = 0.30`, and hard clamps `d_low = max(0, …)` / `d_high = min(1, …)`.

---

## Problems with the previous design

1. **Ceiling collapse (critical):** When `d_final = 1.0`, `min(1.0, 1.0 + 0.30) = 1.0`
   → MODERATE = HIGH.  The MWL classifier was trained with no HIGH vs MODERATE
   separation, which was the root cause of the S005 adaptation failure.

2. **Asymmetric floor:** `max(0.0, d_final − 0.30)` clamped LOW to d=0 for easy
   participants, losing separation at the bottom of the range.

3. **Event rates not properly modulated:** The old formula `int(d / (slot / 60))`
   saturated for sysmon and gave identical counts at MODERATE and HIGH for
   ceiling participants.

4. **Unequal H/L fold-change:** A linear scale gives ~3x for easy participants
   (near d=0) but only ~1.5x for ceiling participants (near d=1).  This
   imbalance means the classifier sees different-quality training data
   depending on where each participant's staircase converged.

---

## Design

### Difficulty range

| Anchor | d value | Meaning |
|---|---|---|
| Absolute minimum | −0.8 | LOW level for a floor-converging participant (d_final = 0) |
| Staircase usable range | [0, 1] | Where d_final can converge |
| Absolute maximum | +1.8 | HIGH level for a ceiling participant (d_final = 1) |
| Total range | 2.6 | D_LOG_MAX − D_LOG_MIN |

### Delta

`DEFAULT_DELTA = 0.80`

This choice is locked by two boundary conditions simultaneously:
- Floor participant: d_final = 0 → d_LOW = 0 − 0.8 = **−0.8** (the absolute minimum)
- Ceiling participant: d_final = 1 → d_HIGH = 1 + 0.8 = **+1.8** (the absolute maximum)

### Log (exponential) scale for event rates

For a given subtask, the rate at difficulty d is:

```
rate(d) = MIN_RATE × (MAX_RATE / MIN_RATE)^t
t = clamp((d − D_LOG_MIN) / D_LOG_RANGE, 0, 1)
```

This guarantees that the HIGH/LOW fold-change is the same for *every* participant,
regardless of where in [0, 1] their staircase converged:

```
fold = (MAX_RATE/MIN_RATE)^(2 × delta / D_LOG_RANGE)
```

### Subtask anchors

| Subtask | d = −0.8 | d = +1.8 | H/L fold | Physical limit |
|---|---|---|---|---|
| Sysmon lights /block | 1 | 18 | ~6x | Channels independent; no hard cap |
| Sysmon scales /block | 1 | 18 | ~6x | Channels independent; no hard cap |
| Comms /block | 1 | 2 | 2x | Serial prompts: 19 s/slot → max 2 per 54 s |
| Pump failures /block | 1 | 4 | 4x | 11 s/slot (10 s failure + 1 s) → max 4 per 54 s |
| Drain ml/min | 50 | 400 | ~8x | Capped below naive-pump sustainable max (600 ml/min) |
| track_update_ms | 50 ms | 10 ms | — | Log scale over [−0.8, +1.8]; no emergency floor needed |
| joystick_force | 3.0 | 1.0 | — | Log scale over [−0.8, +1.8]; 1.0 = compensation limit |

The 54 s effective window = 60 s block − 2 × 3 s edge buffer (prevents events
from straddling EEG epoch boundaries at block transitions).

### Comms note

Comms is physically limited to 1–2 prompts per block.  This is acceptable
because each prompt (18 s auditory task) interacts with the other subtasks —
1 vs 2 concurrent prompts produces a meaningful workload difference even if
the count difference is small.

### Sysmon event rate reference

Kim et al. (2019) used 37.5 sysmon failures/min; the proposed maximum of
18/block ≈ 20/min is well within established practice.

### Drain cap at 400 ml/min

The original drain max of 1200 ml/min was set as the "pump-network physical max",
but this assumed participants run pumps 5 and 6 (finite reserve replenishment),
which requires expert knowledge not in the instructions.  A naive participant who
only runs the two infinite-source pumps (2: E→A, 4: F→B) has a sustainable max
inflow of 600 ml/min per tank.  Drain exceeds this at d ≥ 1.23, creating an
impossible task and causing disengagement.  The cap was reduced to 400 ml/min,
leaving ≥200 ml/min net positive flow at all levels for a naive pump strategy.

### Tracking aligned to log scale

Tracking originally used linear interpolation over d ∈ [0, 1] (easy→hard),
inconsistent with the [−0.8, +1.8] range used for all other subtasks.  When d
exceeded 1.0, tracking extrapolated into physically impossible territory
(joystick_force < 0, update_ms < 1 ms).  Tracking now uses the same
t = (d − D_LOG_MIN) / D_LOG_RANGE normalisation: d=−0.8 → 50 ms / force 3.0,
d=+1.8 → 10 ms / force 1.0 (compensation limit).

---

## Verification table (delta = 0.8, log scale)

| Participant | Level | d | track_ms | force | drain | lights | comms | pumps |
|---|---|---|---|---|---|---|---|---|
| Floor d_fin=0.0 | LOW | −0.80 | 50 ms | 3.00 | 50 ml/m | 1 | 1 | 1 |
| | MOD | +0.00 | 38 ms | 2.38 | 95 ml/m | 2 | 1 | 2 |
| | HIGH | +0.80 | 18 ms | 1.77 | 224 ml/m | 6 | 2 | 2 |
| Typical d_fin=0.5 | LOW | −0.30 | 44 ms | 2.69 | 71 ml/m | 2 | 1 | 1 |
| | MOD | +0.50 | 30 ms | 2.00 | 141 ml/m | 4 | 1 | 2 |
| | HIGH | +1.30 | 14 ms | 1.31 | 280 ml/m | 9 | 2 | 3 |
| Ceiling d_fin=1.0 | LOW | +0.20 | 42 ms | 2.46 | 107 ml/m | 3 | 1 | 2 |
| | MOD | +1.00 | 22 ms | 1.62 | 211 ml/m | 7 | 2 | 3 |
| | HIGH | +1.80 | 10 ms | 1.00 | 400 ml/m | 18 | 2 | 4 |

H/L fold: sysmon ~6x, drain ~8x, comms 2x, pumps 2–4x.

---

## Code locations

| File | Change |
|---|---|
| `src/adaptation/difficulty_state.py` | Replaced linear `_lerp` + `_resman_leak` with `_log_rate` + `_log_drain`; new constants `_D_LOG_MIN/MAX`, `_SYSMON/COMMS/PUMP/DRAIN` anchors; tracking aligned to log scale over [−0.8, +1.8]; drain capped at 400 ml/min; `DifficultyState` bounds relaxed to `d_min ≤ d_max` |
| `scripts/generate_scenarios/generate_full_study_scenarios.py` | `DEFAULT_DELTA = 0.80`; `compute_level_difficulties` symmetric (no clamps on d_low or d_high); `--d-final` validation widened to [−0.8, +1.8] |
| `scripts/generate_scenarios/generate_adaptive_automation_scenarios.py` | `DEFAULT_DELTA = 0.80` |
| `tests/test_staircase_simulation.py` | `test_resman_leak_key_values` → `test_log_drain_key_values` testing log endpoints (50→400) and monotonicity |

---

**Alternatives considered**  
- *Linear scale, delta = 0.30:* Simple, but gives 1.5–3x H/L varying by participant.
  Rejected: unequal training data quality across participants.
- *Linear scale, delta = 0.50:* Slightly better separation (~3x) but d_min = −0.50
  for floor participants and still unequal across the cohort.
- *Log scale, delta = 0.30:* Consistent ~2x H/L for all participants.  Rejected:
  too small a separation for reliable MWL classification.
- *Log scale, delta = 0.8, max events = 18×6 = 108:* Would give 6x for all
  participants but requires 18 events/block at the ceiling — judged impractical
  (one sysmon failure every 3 s) until confirmed by Kim et al. (2019) reference.

**Implications**  
- Scenario files must be regenerated with the new parameters before use.
- The staircase can now converge outside [0, 1] if `d_min`/`d_max` are set
  accordingly; log-scale values are clamped at endpoints outside [−0.8, +1.8].
- The drain formula is no longer comparable to the pilot scenario drain values;
  pilot data cannot be directly pooled with study data for drain-based features.

**Status**  
Current (revisable — pending first study-participant run to verify task feel)

**References**  
- Kim, S. et al. (2019). [sysmon failure rate reference, 37.5/min]
- `docs/lab-notes/2026-04-08_pself_s005_post-hoc_analysis.md` (root-cause audit leading to this change)
