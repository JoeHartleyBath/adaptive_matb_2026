# Style & naming guide (canonical)

This is the enforceable naming specification for this repository.

## Global rules (apply everywhere)

- Use **lowercase** and **snake_case** for filenames and directories: `lowercase_words_joined_by_underscores`.
- Use ISO dates: `YYYY-MM-DD`.
- Keep names short and specific; include only the minimum needed identifiers.

### Exception: standard / must-read docs

To make “must-read” governance docs visually distinctive (similar to `README.md`), a small, curated set of repository-level policy/standard documents in `docs/` MAY use **SCREAMING_SNAKE_CASE**.

Requirements for this exception:

- Only use this for repo-wide standards/policies/checklists (not general prose).
- The filename MUST still be snake_case (underscores only), just in all caps.
- The set MUST be small and stable; do not introduce new SCREAMING_SNAKE_CASE docs without a clear reason.
- These docs SHOULD be linked from `README.md` (or another obvious index) so they remain discoverable.

### Prohibited patterns

- No spaces.
- No CamelCase / PascalCase in filenames.
- No ambiguous suffix chains like `final`, `final_final`, `final2`, `new`, `temp`.
- Avoid punctuation beyond `_` and `-` (prefer `_`).

## Location-specific rules

### Scripts (`scripts/`)

- Pattern: `scripts/<topic>_<action>.<ext>`
- Include language-appropriate extension: `.py`, `.m`, `.r`, `.ps1`.
- Example (good): `scripts/matb_extract_events.py`

### Notebooks (`analysis/notebooks/`)

- Pattern: `analysis/notebooks/YYYY-MM-DD_<topic>.ipynb`
- Notebook names are dated to support a chronological research log.
- Example (good): `analysis/notebooks/2026-01-20_adaptive_thresholding.ipynb`

### Figures (`results/figures/`)

- Pattern: `results/figures/<analysis>__<figure_id>__<brief_desc>.<ext>`
- Use a double-underscore separator `__` between logical parts.
- Preferred formats: `.png` for quick review, `.pdf`/`.svg` for publication.
- Example (good): `results/figures/performance__fig03__rt_by_condition.pdf`

### Tables (`results/tables/`)

- Pattern: `results/tables/<analysis>__<table_id>__<brief_desc>.csv`
- Example (good): `results/tables/performance__tab01__subject_summary.csv`

### Model cards (`results/model_cards/`)

- Pattern: `results/model_cards/<model_name>__<variant>__YYYY-MM-DD.md`
- Example (good): `results/model_cards/rt_predictor__baseline__2026-01-20.md`

### Decisions (`docs/decisions/`) and lab notes (`docs/lab-notes/`)

- Decisions pattern: `docs/decisions/YYYY-MM-DD_<short_slug>.md`
- Lab notes pattern: `docs/lab-notes/YYYY-MM-DD.md` (or `YYYY-MM-DD_<short_slug>.md` if needed)
- Examples (good): `docs/decisions/2026-01-20_data_boundaries.md`, `docs/lab-notes/2026-01-20.md`

## Examples (good vs bad)

- Good: `results/metrics/performance__v1__rt_summary.json`
- Bad: `results/metrics/PerformanceFinal_FINAL.json` (CamelCase + `final_final`)

- Good: `analysis/notebooks/2026-01-20_attention_checks.ipynb`
- Bad: `analysis/notebooks/Jan 20 2026 Attention Checks.ipynb` (spaces + non-ISO date)

- Good: `scripts/preprocess_matb_logs.py`
- Bad: `scripts/PreprocessMATBLogs.py` (CamelCase)

If a file violates these rules, rename it in the same PR that introduces it.
