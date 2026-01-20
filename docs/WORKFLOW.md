# Workflow

This document describes the day-to-day working rules for this repository.

## Branching

- `main` is protected/stable.
- Work happens on short-lived branches.
- Branch name format:
  - `<type>/<short_slug>`
  - Types: `feat/`, `fix/`, `docs/`, `chore/`, `refactor/`, `analysis/`, `unity/`
  - Slug: lowercase + snake_case or kebab-case (pick one and stay consistent within a branch name).
- Examples:
  - `analysis/rt_qc_summary`
  - `feat/adaptive_thresholding`

## Pull requests

- Merge to `main` via PR (no direct pushes).
- PRs should include:
  - Purpose and scope (what changed, why).
  - How to reproduce / validate (commands, notebook path, or steps).
  - Links to relevant docs (ADR, lab note, issue).
- Keep PRs small enough to review quickly; split large work into sequenced PRs.

## Commit messages (Conventional Commits)

Use Conventional Commits:

- Format: `<type>(optional-scope): <summary>`
- Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`
- Use imperative present tense; keep summaries short.
- Examples:
  - `feat(adaptation): add threshold update rule`
  - `fix(parser): handle missing event timestamps`
  - `docs: add workflow and style guide`

## Code vs notebooks

- Prefer code in `src/` (reusable, testable) and keep notebooks thin.
- Notebooks (`analysis/notebooks/`) are for:
  - exploration, QA, and narrative results
  - calling well-named functions from `src/`
- Notebook rules:
  - Use a dated filename (`YYYY-MM-DD_<topic>.ipynb`).
  - Clear inputs/outputs at the top (data source location, parameters).
  - No embedded raw/identifiable participant data.

## Results placement

- `results/` contains derived, shareable outputs only:
  - `results/figures/` — figures for review/paper
  - `results/tables/` — analysis tables (CSV)
  - `results/metrics/` — aggregated metrics (JSON/CSV)
  - `results/model_cards/` — model documentation (MD)
- Avoid committing large binaries or anything that could contain PII.

## Definition of done: analysis run

An analysis run is “done” when:

- Inputs are documented (data location, filters/exclusions, parameters).
- The run is reproducible from repo code/config (no manual, undocumented steps).
- Outputs are written to `results/` with meaningful names.
- A short note exists linking the run to context:
  - a dated notebook and/or a lab note entry in `docs/lab-notes/`
  - any design choices captured as an ADR in `docs/decisions/` when appropriate.
- Sensitive/raw data is not committed (PII, raw exports, large artifacts, checkpoints).
