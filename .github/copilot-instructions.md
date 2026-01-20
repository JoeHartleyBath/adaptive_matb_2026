
# Copilot instructions (adaptive_matb_2026)

## Canonical docs

- Style & naming: see `docs/STYLEGUIDE.md` (canonical).
- How we work: see `docs/WORKFLOW.md`.

## Constraints

- Do not create new top-level folders.
- Do not write to ignored paths (respect `.gitignore` and data boundaries).
- Prefer editing existing files over creating new ones.
- If creating a file is necessary, choose the correct existing directory (e.g., `docs/`, `src/`, `analysis/`, `results/`, `unity/`, `scripts/`) and include a brief rationale in the PR text (or, if no PR, in the response describing the change).

## Data boundaries

- Never add raw/identifiable participant data (PII), large datasets, or model checkpoints to git.
