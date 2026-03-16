# Design choice: Use three workload levels (low/moderate/high)

**Decision**  
Use more than two workload levels—specifically `LOW`, `MODERATE`, and `HIGH`—to reduce inverted-U ambiguity in workload–performance/physiology relationships.

**Rationale**  
- Two-level designs can be hard to interpret if effects are non-monotonic (e.g., inverted-U), because “higher vs lower” does not establish where the levels fall on the curve.
- Three ordered levels provide a minimal, defensible ordinal ladder that can distinguish monotonic trends from non-monotonic patterns in performance, subjective workload, and EEG-derived measures.
- A three-level scheme aligns cleanly with block-level labeling and counterbalancing (e.g., Latin-square over 3 levels), improving interpretability and reproducibility.

**Alternatives considered**  
- Two workload levels (low/high) only.
- More than three levels (finer-grained), at the cost of longer sessions or reduced per-level data.
- Continuous difficulty modulation without discrete level definitions.

**Implications**  
- Protocol, scenarios, and markers must represent exactly three workload level identifiers (`LOW`, `MODERATE`, `HIGH`) wherever workload levels are referenced.
- Analysis and QC assumptions should treat workload level as an ordered factor with three levels (not binary).

**Status**  
Final

**References**  
- Internal: `docs/pilot/pilot_build_plan.md`
- Internal: `docs/pilot/pilot_study_spec.md`
- Internal: `docs/decisions/2026-01-28_methods_workload_levels_and_operationalisation.md`
