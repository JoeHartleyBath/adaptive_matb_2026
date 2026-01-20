# adaptive_matb_2026

Research repository for the 2026 adaptive MATB study: experiment code, analysis pipelines, and paper-ready outputs.

## Repository structure

- `analysis/` — notebooks and analysis reports
- `archive/` — deprecated/legacy artifacts kept for reference
- `assets/` — static assets (e.g., images, stimuli metadata)
- `config/` — configuration files (study/app/analysis)
- `docs/` — decisions, lab notes, study/paper documentation
- `results/` — derived outputs (figures, metrics, tables, model cards)
- `scripts/` — one-off and helper scripts (data munging, utilities)
- `src/` — source code
	- `src/matlab/`, `src/python/`, `src/r/`
- `unity/` — Unity project and experiment runtime assets

## Data boundaries

This repository is intentionally scoped to shareable research code and derived artifacts.

- No raw or identifiable participant data (PII) is stored here.
- No large datasets are stored here (use external storage and document access separately).
- No model checkpoints or training snapshots are stored here.

## Working rules

- Branching: protect `main`; work on short-lived topic branches (e.g., `feat/...`, `fix/...`, `chore/...`).
- PRs: changes land via pull request with a brief description and review when feasible.
- Commit rhythm: commit small, coherent units of work; write messages that explain intent and scope.

## Naming conventions

Naming and style conventions are defined in `docs/STYLEGUIDE.md` (canonical source).