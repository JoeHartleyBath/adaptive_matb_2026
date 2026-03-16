# experiment/

Experiment asset files for the adaptive MATB 2026 study.

## Contents

- `scenarios/` — OpenMATB scenario `.txt` files. These are the **source of truth** for
  the Pilot 1 task sequence. Do **not** regenerate them casually; changes here directly
  affect what participants experience. Use `scripts/generate_pilot_scenarios.py` to
  produce new files, then review carefully before replacing any locked scenario.

- `instructions/` — Participant-facing instruction screen text files. These are copied
  into the OpenMATB include tree at runtime by `src/run_openmatb.py`. The filenames are
  bound to the scenario reference strings (e.g. `1_welcome.txt`); do not rename without
  updating the corresponding scenario files.

## Status

Scenario files under `scenarios/` are **locked** as of Pilot 1 (March 2026).
Regeneration requires explicit sign-off and a new commit with a clear rationale.
