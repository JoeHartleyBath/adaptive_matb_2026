# Data management

This repository intentionally stores **code, documentation, and small derived artifacts** only. All **large data** and any **sensitive/raw exports** must live outside git.

## 1) Where large data must live

Large datasets and any raw exports must be stored in external/internal storage (e.g., an encrypted lab drive, institutional storage, or a secured server share) and referenced locally via configuration.

Examples of content that must stay **outside** the repo:

- Raw task logs as exported from acquisition systems
- Any files containing participant identifiers or quasi-identifiers
- Large video/audio/screen recordings
- Model checkpoints / training snapshots
- Large intermediate artifacts (multi-GB parquet/hdf5, etc.)

## 2) Recommended external folder layout

Create a single project root on your data drive (path varies by machine), e.g.:

- `D:/adaptive_matb_2026_data/`
  - `raw/` — immutable raw exports (read-only after ingest)
  - `interim/` — scratch/intermediate products (safe to delete/rebuild)
  - `processed/` — cleaned/standardized datasets (no PII)
  - `derived/` — derived datasets used by figures/tables (no PII)
  - `checks/` — QA summaries (no PII)
  - `docs/` — non-sensitive run manifests (optional)

Notes:

- Treat `raw/` as append-only and never edit in place.
- If you need a manifest, store it in `derived/` or `docs/` and keep it non-sensitive.

## 3) Local path configuration

Local machine paths are configured via:

- `config/paths.yaml` — **local-only**, ignored by git
- `config/paths.example.yaml` — **tracked**, safe template for others

Workflow:

1. Copy `config/paths.example.yaml` to `config/paths.yaml`.
2. Edit `config/paths.yaml` to point to your local data root (external/internal drive path).
3. Never commit `config/paths.yaml`.

## 4) Rules for derived artifacts

**Allowed in `results/` (tracked):**

- Small, shareable, non-sensitive outputs: figures (`.png/.pdf/.svg`), tables (`.csv`), aggregated metrics (`.json/.csv`), model cards (`.md`).
- Outputs must not include raw/identifiable participant data.

**Must stay external (not tracked):**

- Anything large (multi-GB artifacts, bulky intermediate datasets)
- Any output that could contain PII or re-identification risk
- Checkpoints, training snapshots, or serialized model weights

Rule of thumb: if it’s not safe to email to a collaborator without special handling, it does not belong in git.

## 5) OpenMATB session logs

OpenMATB writes session outputs under `src/python/vendor/openmatb/` (submodule):

- `sessions/` (CSV session logs)
- `last_scenario_errors.log` (scenario validation summary)

These files must remain untracked (ignored by git). Confirm with:

```powershell
git -C src/python/vendor/openmatb check-ignore -v sessions/ last_scenario_errors.log
```
