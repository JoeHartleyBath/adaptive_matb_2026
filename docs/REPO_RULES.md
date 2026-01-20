# Repo rules (non‑negotiables)

This document is the “repo constitution” for **adaptive_matb_2026**. If a change conflicts with anything below, it must be handled as an explicit exception via an ADR.

## 1) Data boundary (hard stop)

- Never commit raw participant data.
- Never commit identifiable participant data (PII) or quasi-identifiers.
- Never commit large datasets/binaries.
- Never commit model checkpoints / training snapshots.

See `docs/DATA_MANAGEMENT.md` for the approved external-data workflow.

## 2) Naming conventions (enforced)

- Filenames and directories must follow the canonical spec in `docs/STYLEGUIDE.md`.
- Prohibited patterns include: spaces, CamelCase/PascalCase, and `final_final`-style suffix chains.

## 3) Folder rules (separation of concerns)

- Do not create new top-level folders.
- Use existing locations:
  - `docs/` documentation, ADRs, lab notes
  - `src/` reusable code
  - `analysis/` notebooks + analysis reports
  - `results/` derived, shareable artifacts only
  - `unity/` Unity runtime
  - `scripts/` one-off helpers
- Do not write to ignored paths (respect `.gitignore`).

## 4) Main branch protection

- No direct pushes to `main`.
- All changes land via PR.

## 5) Commit discipline

- Commit early and regularly (small, coherent units).
- Use Conventional Commits (see `docs/WORKFLOW.md`).

## 6) Exceptions (ADR-required)

If you need to violate a rule:

- Propose an exception via an ADR in `docs/decisions/`.
- State: what rule is being violated, why it’s necessary, scope/duration, risk controls, and rollback plan.
- Do not merge exception work until the ADR is accepted.
