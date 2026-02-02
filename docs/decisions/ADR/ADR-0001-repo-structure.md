# ADR-0001: Repository structure and data boundaries

## Status

Accepted — 2026-01-20

## Context

This project (“adaptive_matb_2026”) combines:

- An experimental runtime (implementation details may evolve; keep runtime code discoverable and separate from analysis).
- Analysis workflows (notebooks and reports) that turn collected logs into derived metrics and figures (`analysis/`, `results/`).
- Multi-language source code used for utilities and analysis (`src/` with `python/`, `matlab/`, `r/`).
- Study/paper documentation, lab notes, and design rationale (`docs/`).

We need a repository layout that:

- Keeps experiment runtime, analysis, and documentation discoverable and maintainable.
- Ensures derived outputs are reproducible from code/config rather than “mystery files”.
- Prevents accidental inclusion of sensitive data (raw/identifiable participant data) and large artifacts that do not belong in git.

## Decision

Use a **single monorepo** with **strict separation of concerns** across top-level directories:

- `docs/` — decisions, lab notes, study/paper documentation
- `src/` — reusable source code (Python/MATLAB/R)
- `analysis/` — notebooks and analysis reports
- `results/` — derived, shareable artifacts (figures, tables, metrics, model cards)

Additionally, enforce explicit **data boundaries**:

- Do not commit raw participant data or any identifiable participant information (PII).
- Do not commit large datasets or large binary artifacts.
- Do not commit model checkpoints/training snapshots.

Only derived, shareable outputs belong in `results/` (e.g., aggregated metrics, figures, tables, and model cards) when they do not contain PII.

## Consequences

- Positive: one canonical place for experiment code + analysis + paper/study documentation; simpler onboarding and cross-referencing.
- Positive: clearer provenance—derived artifacts in `results/` trace back to code in `src/`/`analysis/` and configuration in `config/`.
- Positive: reduced risk of leaking sensitive data by explicitly excluding raw/identifiable data from version control.

- Negative: repository can grow if outputs are not curated; requires discipline around what is stored in `results/`.
- Negative: external storage is required for raw data and large artifacts; workflows must document where data lives and how to access it.

Mitigations:

- Keep `results/` limited to lightweight, non-sensitive, publication-appropriate derivatives.
- Document data locations and access procedures outside the repository (e.g., internal storage paths and permissions) in project documentation.
