# Design choice: Full-study calibration structure (2×9 min; 1-min blocks)

**Decision**  
For the full study (post-Pilot 1), calibration will consist of **two 9-minute conditions**, where each condition contains **nine 1-minute blocks**: **3×LOW**, **3×MODERATE**, and **3×HIGH**.

**Rationale**  
- Aligns calibration duration/structure to the supervisor-agreed full-study protocol.
- Keeps workload labels single-level at the block level (each 1-minute block has one workload level), supporting clean label-to-window mapping.
- Provides repeated exposure per workload level within a fixed total calibration time (18 minutes), supporting within-session reliability checks.

**Alternatives considered**  
- Three longer blocks (e.g., 3×5 min, one per level) for simpler implementation.
- Fewer, longer mixed-level conditions (risking transition ambiguity and/or state carryover effects).

**Implications**  
- This structure is **not implemented for Pilot 1**. Pilot 1 remains on the current 3×5:00 calibration blocks to avoid destabilising pilot readiness artifacts.
- Marker/segment naming must support repeated blocks per workload level (e.g., include condition and minute-block identifiers).
- Verification and summary scripts must be updated to handle repeated per-level blocks and aggregation rules.

**Status**  
Final (locked; implement after Pilot 1)

**References**  
- Internal: [docs/pilot/pilot_study_spec.md](../../pilot/pilot_study_spec.md) (Pilot 1 structure)
- Internal: [docs/decisions/open_decisions.md](../../open_decisions.md) (OD-07 context)
