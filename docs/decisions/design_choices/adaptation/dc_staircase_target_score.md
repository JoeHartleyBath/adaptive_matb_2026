# Design choice: Staircase convergence target = 0.90 (not 0.70)

**Decision**  
The staircase calibration targets `cursor_in_target = 0.90` at MODERATE difficulty,
not 0.70.  This ensures that the HIGH level (MODERATE + delta=0.80) lands at
approximately 0.68 in-target — appropriately challenging but not impossible.

---

## Background

The staircase converges `d_final` such that a participant achieves `target_score`
tracking performance at MODERATE difficulty.  With `DEFAULT_DELTA = 0.80`,
participants then experience LOW at `d_final − 0.8` and HIGH at `d_final + 0.8`.

The original target of 0.70 was inherited from early pilot work and was not
re-evaluated when delta was increased from 0.30 to 0.80.

---

## Problem

With delta=0.80, tracking drops by approximately 0.22 between MODERATE and HIGH
(empirical data from PSELF, 2026-04-09):

| d | cursor_in_target |
|---|---|
| 0.5 | 0.890 |
| 0.7 | 0.904 |
| 0.9 | 0.918 |
| 1.8 | 0.675 |

The drop from d=0.9 (MODERATE at ceiling) to d=1.8 (HIGH at ceiling) is −0.24.
If the staircase targeted 0.70, a participant would converge at a d where tracking
is 0.70, and then HIGH would place them at approximately 0.70 − 0.22 = **0.48**
in-target — nearly half the time outside the target zone.  This is overwhelming
and causes task disengagement, not elevated workload.

More concretely: PSELF converged at d=1.0 (tracking ~92%) and found d=1.8 (HIGH)
hard but manageable (67.5% in-target).  If the staircase had targeted 0.70 instead,
PSELF's MODERATE would have been near d=1.7 and HIGH would have been off the scale.

---

## Design

**Target 0.90 means:**  
- MODERATE ≈ 90% in-target for all participants  
- HIGH ≈ 90% − 22% = **68%** in-target for all participants  
- 68% is challenging but achievable (empirically validated on PSELF)

**Applies uniformly:** because `_log_rate` gives approximately the same absolute
drop in tracking performance across the 0.8 delta regardless of where d_final sits.
This makes the difficulty gap between MODERATE and HIGH consistent across the
participant population.

The 0.70 figure is now the target *at HIGH*, not at MODERATE — which is the
correct interpretation of "70% performance threshold".

---

## Code locations

| File | Change |
|---|---|
| `src/adaptation/adaptation_scheduler.py` | `target_score: float = 0.90` (was 0.70) |
| `src/adaptation/staircase_controller.py` | Default `target_score=0.90`; docstring example updated |

Tests in `tests/test_staircase_simulation.py` all pass explicit `target_score=0.70`
and are unaffected by the default change.

---

**Alternatives considered**  
- *Keep 0.70:* Correct target for a fixed, non-adaptive task.  Inappropriate
  when HIGH = MODERATE + 0.80, as it pushes HIGH into the impossible zone.
- *Use 0.80:* Would give HIGH ≈ 0.58 in-target.  Still too hard for most
  participants and lacks the empirical grounding that 0.90 has.

**Implications**  
- The staircase will now converge at a higher d_final for all participants
  compared to the old 0.70 target (participants need to demonstrate 90% before
  the difficulty stops rising).
- Ceiling participants like PSELF will converge at d_max=1.0 as before (since
  tracking is ~92% there, above 0.90), so no change for ceiling cases.
- Weaker participants will converge at a lower d_final than they would have with
  0.70, because the staircase will keep pushing higher until they drop below 90%
  rather than stopping at 70%.  This is the desired behaviour.

**Status**  
Current — empirically validated on PSELF 2026-04-09

**References**  
- PSELF session data: `C:\data\adaptive_matb\openmatb\PSELF\STEST\sessions\2026-04-09\`
- DC-13: Log-scale event rates (sets delta=0.80, which this decision depends on)
