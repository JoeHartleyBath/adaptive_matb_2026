# Design choice: Include a training phase before calibration blocks

**Decision**  
Include an explicit training/familiarisation phase (training blocks) before any calibration blocks.

**Rationale**  
- Reduces confounding from early learning and control familiarisation when interpreting calibration-block performance, subjective workload, and EEG measures.
- Makes the protocol more robust to between-participant differences in prior exposure to MATB-style task batteries.
- Creates a consistent operator procedure and session flow that can be implemented deterministically in scenarios.

**Alternatives considered**  
- No training phase (start immediately with calibration blocks).
- Training as an optional, operator-discretion step (variable exposure).

**Implications**  
- Training blocks are treated as non-calibration (not used for primary analyses/modeling in v0).
- The training phase must be explicitly represented in scenarios and markers so that training vs calibration data can be separated deterministically.

**Status**  
Final

**References**  
- Internal: `docs/pilot/PILOT_STUDY_SPEC_V0.md`
- Internal: `docs/pilot/PILOT_BUILD_PLAN_V0.md`
- Internal: `docs/contracts/training_scenario_contract_v0.md`
