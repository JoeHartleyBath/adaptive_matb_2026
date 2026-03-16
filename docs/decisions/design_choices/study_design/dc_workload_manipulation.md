# Design choice: Workload manipulation via difficulty and event rate

**Decision**  
Manipulate mental workload by varying task difficulty and event/event-rate structure within a fixed multitask set, rather than by adding/removing tasks or introducing autopilot assistance.

**Rationale**  
- Isolates changes in workload from changes in task identity or task presence, reducing confounds between workload and task-switching or strategy shifts.
- Preserves a consistent behavioural and perceptual context across workload levels, supporting clearer interpretation of physiological and performance-based workload markers.
- Enables graded and temporally precise workload modulation (e.g., via pacing, density, or control difficulty), which shows better alignment with continuous physiological measures and adaptive control.
- Avoids the interpretational ambiguity introduced by autopilot or task removal, where reduced workload may reflect changes in agency, engagement, or vigilance rather than demand intensity.

**Alternatives considered**  
- Adding or removing tasks across workload levels.
- Autopilot or task automation as a primary workload manipulation.
- Discrete task-set reconfiguration between conditions.

**Implications**  
Workload levels are defined relative to demand intensity within a stable task set; all adaptive logic, labelling strategies, and analyses must interpret workload changes as arising from difficulty or event-rate modulation rather than task presence or automation.

**Status**  
Final (foundational)

**References**  
- Internal: `docs/pilot/pilot_study_spec.md`
- Internal: `docs/decisions/2026-01-28_methods_workload_levels_and_operationalisation.md`
