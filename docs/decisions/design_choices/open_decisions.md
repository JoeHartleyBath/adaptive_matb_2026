# Open / unresolved design questions

Questions that have been raised but not yet resolved.
When a decision is made, move the entry to the main decision index and create a
DC-nn document.

---

## OD-01 — Resman tank reset between calibration blocks

**Question:** Should the scenario generator emit resman tank-reset commands at
each block boundary so every block starts from the same tank level (2500 ml)?

**Background:**  
The full-study calibration scenario is 9 continuous 1-minute blocks with no
restarts.  Resman runs continuously, so tanks drain across block boundaries.
A HIGH block with high drain can leave tanks depleted at the start of the
following block, regardless of its target difficulty level.

Observed in PSELF 2026-04-09 (run 8, `full_calibration_pself_c1`):
- Block 6 (HIGH, drain=400 ml/min) depleted tanks by end of block
- Block 7 (MODERATE, drain=211 ml/min) inherited empty tanks → resman
  tolerance collapsed to ResA=50%, ResB=10%
- Block 7 composite score: 66.5% (HIGH-like) despite being a MODERATE block

**Options considered:**

| Option | Pro | Con |
|---|---|---|
| No reset (current) | Resman state is a real consequence of prior difficulty; ecologically valid | Confounds block-level labels for MWL classifier training |
| Reset tanks at block start | Clean isolated samples; each block represents its label faithfully | Introduces an artificial discontinuity; slightly less naturalistic |

**Current stance:** No reset for now.  
The argument for keeping it: tank depletion *is* correlated with difficulty
over the session as a whole, and forcing a reset could mask ecologically valid
MWL dynamics.  The concern is that individual MODERATE blocks with depleted
tanks look HIGH in the data.  Whether this is a problem depends on whether
the EEG signal also looks HIGH in those blocks (it might, if the participant
was genuinely stressed by the depleted state).

**Revisit when:** analysing the first real participant's calibration data to
check whether resman-depleted MODERATE blocks cluster with HIGH in EEG space.

**Files affected if reset is added:**  
`scripts/generate_scenarios/generate_full_study_scenarios.py` — emit
`resman;tank-a-level;2500` / `resman;tank-b-level;2500` at `t = block_start`
for blocks 2–9.
