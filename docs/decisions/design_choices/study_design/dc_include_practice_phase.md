# Design choice: Include a practice phase before calibration blocks

**Decision**  
Include an explicit practice/familiarisation phase (practice blocks) before any calibration blocks.

**Rationale**  
- Reduces confounding from early learning and control familiarisation when interpreting calibration-block performance, subjective workload, and EEG measures.
- Makes the protocol more robust to between-participant differences in prior exposure to MATB-style task batteries.
- Creates a consistent operator procedure and session flow that can be implemented deterministically in scenarios.

**Alternatives considered**  
- No practice phase (start immediately with calibration blocks).
- practice as an optional, operator-discretion step (variable exposure).

**Implications**  
- practice blocks are treated as non-calibration (not used for primary analyses/modeling in v0).
- The practice phase must be explicitly represented in scenarios and markers so that practice vs calibration data can be separated deterministically.

**Status**  
Final

**References**  
- Internal: `docs/pilot/pilot_study_spec.md`
- Internal: `docs/pilot/pilot_build_plan.md`
- Internal: `docs/contracts/practice_scenario_contract_v0.md`
