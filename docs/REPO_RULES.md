This document is the “repo constitution” for adaptive_matb_2026. If a change conflicts with anything below, it must be handled as an explicit exception via an ADR.

1) Data boundary (hard stop)

Never commit raw participant data.

Never commit identifiable participant data (PII) or quasi-identifiers.

Never commit large datasets/binaries.

Never commit model checkpoints / training snapshots.

See docs/DATA_MANAGEMENT.md for the approved external-data workflow.

Rationale: Ethical, legal, and reproducibility constraints override convenience.

2) Naming conventions (enforced)

Filenames and directories must follow the canonical spec in docs/STYLEGUIDE.md.

Prohibited patterns include: spaces, CamelCase/PascalCase, and final_final-style suffix chains.

Rationale: Naming consistency protects future analysis, scripting, and paper-writing.

3) Folder rules (separation of concerns)

Do not create new top-level folders.

Use existing locations:

docs/ documentation, ADRs, lab notes

src/ reusable code

analysis/ notebooks + analysis reports

results/ derived, shareable artifacts only

scripts/ one-off helpers

Do not write to ignored paths (respect .gitignore).

Rationale: Stable structure matters more than convenience in research repos.

4) Main branch policy (solo-optimised)

main is the stable reference branch.

Direct commits to main are allowed for:

small fixes

exploratory analysis

documentation updates

Use short-lived feature branches for:

substantial refactors

new pipelines or modules

changes that may temporarily break the repo

Rationale: This is a solo research project; PR-only workflows add friction without review benefit.

5) Commit discipline

Commit early and regularly (small, coherent units).

Prefer Conventional Commits (feat:, fix:, refactor:, docs:), but clarity takes priority over strict compliance.

Do not batch unrelated changes into a single commit.

Rationale: Commit history is a research artifact for future debugging and writing.

6) Exceptions (ADR-required)

If you need to violate a rule:

Propose an exception via an ADR in docs/decisions/.

State:

what rule is being violated

why it’s necessary

scope and duration

risk controls

rollback plan

Do not treat ADRs as bureaucracy — they exist to protect future-you.

Rationale: Exceptions are allowed, but must be explicit and traceable.