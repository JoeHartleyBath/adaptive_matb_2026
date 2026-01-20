
# Copilot instructions (adaptive_matb_2026)

## Canonical docs

- Style & naming: see `docs/STYLEGUIDE.md` (canonical).
- How we work: see `docs/WORKFLOW.md`.

## Constraints

- Do not create new top-level folders.
- Do not write to ignored paths (respect `.gitignore` and data boundaries).
- Prefer editing existing files over creating new ones.
- If creating a file is necessary, choose the correct existing directory (e.g., `docs/`, `src/`, `analysis/`, `results/`, `scripts/`) and include a brief rationale in the PR text (or, if no PR, in the response describing the change).

## Data boundaries

- Never add raw/identifiable participant data (PII), large datasets, or model checkpoints to git.

## Workflow rules (mandatory)

- Always propose changes as a new branch with a clear name (e.g. docs/…, feat/…, fix/…).
- Default branch for new work is `main` unless explicitly told otherwise.
- Commits must follow Conventional Commits.
- Prefer small, focused commits that touch only related files.
- Do not batch unrelated changes into a single commit.
- After creating a branch and committing, instruct the user to open a PR to `main`.

## File placement rules (mandatory)

- Documentation goes in `docs/`.
- Repo-wide rules go in `docs/REPO_RULES.md`.
- Naming and formatting rules live in `docs/STYLEGUIDE.md`.
- Do not create new top-level directories.
- Do not write to ignored paths.

## Third party integrations and submodules

- For third-party integration (e.g., OpenMATB submodule), generate:
  * integration docs in docs/
  * step-by-step instructions for initialization and updates
  * licensing and compliance notes
  * example usage with real-time MWL extension
- When reviewing submodules, comment on:
  * correct path setup (.gitmodules)
  * clone/update instructions
  * licensing details and directory placement


