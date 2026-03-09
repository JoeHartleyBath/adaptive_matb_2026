# adaptive_matb_2026

Solo PhD research repository for the 2026 adaptive MATB study: experiment code, analysis pipelines, and paper-ready outputs.

## Repository structure

- `config/` — participant assignments, path configuration (local `paths.yaml` is gitignored)
- `docs/` — decisions, lab notes, pilot specs, study/paper documentation
  - `docs/decisions/` — ADRs, committed design choices, open decisions
  - `docs/pilot/` — pilot study spec, build plan, session checklist
  - `docs/contracts/` — scenario and interface contracts
  - `docs/lab-notes/` — dated exploratory notes (not commitments)
  - `docs/openmatb/` — OpenMATB integration documentation
- `scenarios/` — locked OpenMATB scenario files (source of truth for pilot; do not regenerate casually)
- `instructions/` — participant-facing task instruction screen text files (used by scenarios at runtime)
- `scripts/` — standalone helper and utility scripts
- `src/python/` — production source code
  - `src/python/run_openmatb.py` — main session runner (entrypoint)
  - `src/python/adaptation/` — online staircase calibration logic
  - `src/python/eeg/` — real-time EEG preprocessing pipeline
  - `src/python/verification/` — post-run verification scripts
  - `src/python/vendor/openmatb/` — OpenMATB git submodule
- `analysis/` — notebooks and analysis reports (currently unpopulated; used post-Pilot 1)
- `results/` — derived outputs (figures, metrics, tables; currently unpopulated)

## Running a session

```powershell
cd C:\adaptive_matb_2026
.\.venv\Scripts\Activate.ps1

# Fixed-block pilot (LOW → MODERATE → HIGH), full physiology:
python src/python/run_openmatb.py --pilot1 --calibration-trend --summarise-performance --eda-port COM5 --participant PSELF --seq-id SEQ1

# Staircase pilot:
python src/python/run_openmatb.py --pilot1 --adaptation --only-scenario adaptation_skeleton.txt --eda-port COM5 --participant PSELF --seq-id SEQ1
```

VS Code launch configs for both self-pilot variants are in `.vscode/launch.json` (look for `PSELF:` entries).

Full operator procedure: [`docs/pilot/PILOT_SESSION_CHECKLIST.md`](docs/pilot/PILOT_SESSION_CHECKLIST.md)

## Data boundaries

This repository stores only shareable research code and derived artifacts.

- No raw or identifiable participant data (PII)
- No large datasets (use external storage; see [`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md))
- No model checkpoints or training snapshots
- Session outputs go to `C:\data\adaptive_matb\` (external; gitignored by path)

## Working rules

Solo PhD trunk-based workflow. Small, focused commits directly to `main` are fine for incremental work. Use a short-lived branch only for substantial changes that could temporarily break things. Full rules: [`docs/REPO_RULES.md`](docs/REPO_RULES.md)

## Decision logging

- `docs/decisions/design_choices/` — committed, locked design choices
- `docs/decisions/open_decisions.md` — active unresolved questions (reviewed after each pilot phase)
- `docs/decisions/ADR/` — architectural decision records
- `docs/lab-notes/` — exploratory thinking and narrative (not commitments)

## Naming conventions

[`docs/STYLEGUIDE.md`](docs/STYLEGUIDE.md) is the canonical source.

## OpenMATB documentation

OpenMATB runs as a git submodule at `src/python/vendor/openmatb/`. Study-specific docs:

- Index: [`docs/openmatb/index.md`](docs/openmatb/index.md)
- Setup and run: [`docs/openmatb/SETUP_AND_RUN.md`](docs/openmatb/SETUP_AND_RUN.md)
- Instrumentation and event schema: [`docs/openmatb/INSTRUMENTATION_POINTS.md`](docs/openmatb/INSTRUMENTATION_POINTS.md)
- Closed-loop adaptation design: [`docs/openmatb/ADAPTATION_DESIGN.md`](docs/openmatb/ADAPTATION_DESIGN.md)
