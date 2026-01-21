# TODO — Counterbalancing is not participant ID

**Problem**: Participant IDs must not be conflated with counterbalancing order, especially across pilot vs main study.

**Decision (deferred)**: Counterbalancing will be driven by a study-specific index (e.g. `cb_index`), not `participant_id`.

**Action later**: Define assignment table / rule before first main-study participant.

**Risk if ignored**: Order imbalance, reuse errors, irreproducible assignment.
