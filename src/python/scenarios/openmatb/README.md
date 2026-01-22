# Repo-managed OpenMATB scenarios

Purpose: keep **study-owned** scenario files in this repo (under `src/python/`) instead of editing the upstream/vendor scenario directory.

Why:

- Keeps `src/python/vendor/openmatb/` closer to upstream and easier to update.
- Makes it unambiguous which scenarios are *ours* vs shipped examples.

## How OpenMATB finds scenarios

OpenMATB normally loads scenarios from:

- `src/python/vendor/openmatb/includes/scenarios/`

In this repo, we point `scenario_path` (in `src/python/vendor/openmatb/config.ini`) at a **relative path** that traverses from the vendor scenario folder to this directory.

Recommended (Windows + cross-platform safe): use forward slashes and a relative path.

Example `scenario_path` value (conceptual):

- `../../../..` climbs from `vendor/openmatb/includes/scenarios/` up to `src/python/`
- then append `scenarios/openmatb/<your-file>.txt`

## Rules

- Scenario files here must be deterministic and versioned.
- Do not store any run outputs here (those must stay outside git per `docs/DATA_MANAGEMENT.md`).
- Use clear filenames that encode purpose/version, e.g. `pilot_v0_low.txt`, `pilot_v0_moderate.txt`, `pilot_v0_high.txt`.
