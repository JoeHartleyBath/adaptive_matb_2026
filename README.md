# adaptive_matb_2026

Solo PhD research repository for the 2026 adaptive MATB study: experiment code, analysis pipelines, and paper-ready outputs.

## Repository structure

- `config/` — participant assignments, path configuration (local `paths.yaml` is gitignored)
- `experiment/` — locked experiment asset files
  - `experiment/scenarios/` — OpenMATB scenario `.txt` files (source of truth for Pilot 1; do not regenerate casually)
  - `experiment/instructions/` — participant-facing task instruction screen text files (copied into OpenMATB at runtime)
- `docs/` — decisions, lab notes, pilot specs, study/paper documentation
  - `docs/decisions/` — ADRs, committed design choices, open decisions
  - `docs/pilot/` — pilot study spec, build plan, session checklist
  - `docs/contracts/` — scenario and interface contracts
  - `docs/lab-notes/` — dated exploratory notes (not commitments)
  - `docs/openmatb/` — OpenMATB integration documentation
- `scripts/` — standalone operational and data-pipeline scripts
- `src/` — production source code (installed as editable package via `pip install -e .`)
  - `src/run_openmatb.py` — main session runner (entrypoint)
  - `src/adaptation/` — online staircase calibration logic
  - `src/eeg/` — real-time EEG preprocessing pipeline
  - `src/ml/` — MWL model definition (EEGNet) and dataset loader
  - `src/performance/` — session performance summary and export scripts
  - `src/vendor/openmatb/` — OpenMATB git submodule
- `tests/` — test and verification scripts
  - `tests/verification/` — post-run session verification scripts
- `analysis/` — notebooks and analysis reports (currently unpopulated; used post-Pilot 1)
- `results/` — derived outputs (figures, metrics, tables; currently unpopulated)

## Running a session

```powershell
cd C:\adaptive_matb_2026
.\.venv\Scripts\Activate.ps1

# Fixed-block pilot (LOW → MODERATE → HIGH), full physiology:
python src/run_openmatb.py --pilot1 --calibration-trend --summarise-performance --eda-port COM5 --participant PSELF --seq-id SEQ1

# Staircase pilot:
python src/run_openmatb.py --pilot1 --adaptation --only-scenario adaptation_skeleton.txt --eda-port COM5 --participant PSELF --seq-id SEQ1
```

VS Code launch configs for both self-pilot variants are in `.vscode/launch.json` (look for `PSELF:` entries).

Full operator procedure: [`docs/pilot/pilot_session_checklist.md`](docs/pilot/pilot_session_checklist.md)

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

OpenMATB runs as a git submodule at `src/vendor/openmatb/`. Study-specific docs:

- Index: [`docs/openmatb/index.md`](docs/openmatb/index.md)
- Setup and run: [`docs/openmatb/SETUP_AND_RUN.md`](docs/openmatb/SETUP_AND_RUN.md)
- Instrumentation and event schema: [`docs/openmatb/INSTRUMENTATION_POINTS.md`](docs/openmatb/INSTRUMENTATION_POINTS.md)
- Closed-loop adaptation design: [`docs/openmatb/ADAPTATION_DESIGN.md`](docs/openmatb/ADAPTATION_DESIGN.md)
