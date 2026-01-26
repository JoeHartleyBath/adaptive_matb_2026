# ADR-0002: Prefer wrapper-run-per-block to guarantee clean task state

## Status

Accepted — 2026-01-26

## Context

We want a pilot protocol with multiple task blocks (familiarisation, training, then three retained difficulty blocks in a counterbalanced order). A desirable simplification would be to embed multiple “runs”/blocks into a single OpenMATB scenario file (similar to the vendor example scenarios), reducing the number of scenario files and reducing wrapper logic.

However, the core experimental requirement is that each block begins from a known, comparable baseline state (especially the **Resource Management** task, but also other tasks with accumulating state).

### What the vendor documentation supports (and what it does not)

The vendor wiki describes scenario “actions” primarily as `start` / `stop` / `pause` / `resume` / `hide` / `show` and notes that actions execute the corresponding plugin method (example: sending `pause` to `resman` calls `Resman.pause()` in `resman.py`). There is no documented scenario-level `reset` action.

- Scenario actions are “limited in number” and “most used actions are `start` and `stop`” (no mention of reset): `docs/openmatb/wiki/How-to-build-a-scenario-file.md` (Actions section).
- “When a plugin is sent an action instruction (e.g., `start`), … execute the corresponding method in the plugin python file.”: `docs/openmatb/wiki/How-to-build-a-scenario-file.md` (Actions section).
- The `resman` and `track` wiki pages document parameter manipulation (e.g., `tank-*-level`, `pump-*-state`, `automaticsolver`), but do not describe a canonical block reset mechanism: `docs/openmatb/wiki/The-resources-management-task-(resman).md`, `docs/openmatb/wiki/The-tracking-task-(track).md`.

### What the vendor code does

OpenMATB parses the scenario file into a time-ordered list of events. For each event, it either:
- calls a plugin method (action), or
- sets a plugin parameter (parameter update).

Critically, plugins are instantiated **once per scenario load**, not once per block.

Evidence (vendor code):
- Scenario parsing & plugin instantiation (plugins created once per scenario): `src/python/vendor/openmatb/core/scenario.py` (`Scenario.__init__`, see the `self.plugins = {...}()` comprehension).
- Event execution (action => method call; parameter => `set_parameter`): `src/python/vendor/openmatb/core/scheduler.py` (`Scheduler.execute_one_event`).
- Base plugin lifecycle (no reset semantics in `start()` / `stop()`): `src/python/vendor/openmatb/plugins/abstractplugin.py` (`AbstractPlugin.start()` / `stop()`).

### Why a single multi-block scenario cannot guarantee clean state today

Within a single scenario run, repeating `stop` → `start` does not re-instantiate plugins and does not restore non-parameter internal state.

Concrete examples:
- `resman` mutates tank levels and pump states over time and also uses internal attributes not represented as scenario parameters (e.g., `wait_before_leak`). Those will carry over between blocks unless the entire application/plugin instance is restarted.
  - Evidence: `src/python/vendor/openmatb/plugins/resman.py` (`Resman.compute_next_plugin_state`, `wait_before_leak`).
- `track` maintains internal generator/state for cursor motion and response timing. This state is not fully expressible as scenario parameters.
  - Evidence: `src/python/vendor/openmatb/plugins/track.py` (cursor path generator and response-time accumulation).

Because the vendor scenario/action surface does not include a documented reset mechanism, and because plugin instances persist within a scenario, a single multi-block scenario would require additional vendor-side behavior to be confident that each block starts from an identical baseline.

## Decision

Keep the current approach:

- Each block is run as a separate OpenMATB process invocation via the repo wrapper (`src/python/run_openmatb.py`).
- The wrapper selects the scenario file for each block (training + retained blocks) and restarts OpenMATB between blocks.

This guarantees a fresh plugin instantiation and baseline state for each block without relying on undocumented or incomplete “reset via scenario actions/parameters”.

## Consequences

Positive:
- Stronger experimental control: each block starts from a clean baseline state.
- Lower risk than vendor modifications (no need to add/maintain custom “reset” actions in OpenMATB).
- Easier validation/debugging: each block produces its own session log/manifest.

Negative:
- More wrapper logic and more session outputs (multiple runs per participant session).
- Slightly longer runtime overhead due to process restarts.

## Alternatives considered

1) Single scenario with multiple blocks and in-scenario `stop`/`start` cycles
- Rejected: cannot guarantee full state reset with current vendor action/parameter surface.

2) Add a vendor “reset” action per plugin (e.g., `resman.reset`, `track.reset`)
- Not pursued for pilot: increases risk and maintenance burden in vendor code; would need dedicated verification.

## References

- Wrapper orchestrating per-block runs: `src/python/run_openmatb.py`.
- Vendor scenario mechanics:
  - `src/python/vendor/openmatb/core/scenario.py`
  - `src/python/vendor/openmatb/core/scheduler.py`
  - `src/python/vendor/openmatb/plugins/abstractplugin.py`
- Vendor wiki (scenario actions / parameters; questionnaires):
  - `docs/openmatb/wiki/How-to-build-a-scenario-file.md`
  - `docs/openmatb/wiki/Write-a-questionnaire.md`
  - `docs/openmatb/wiki/The-resources-management-task-(resman).md`
  - `docs/openmatb/wiki/The-tracking-task-(track).md`
