# OpenMATB — setup and run (Windows-first)

Related docs:
- [OVERVIEW.md](OVERVIEW.md)
- [INSTRUMENTATION_POINTS.md](INSTRUMENTATION_POINTS.md)

OpenMATB lives at: `src/python/vendor/openmatb/`

## 1) Create a virtual environment

From the repo root:

```powershell
cd src/python/vendor/openmatb

# Validated on Windows with Python 3.10
py -3.10 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Notes:
- Upstream `main.py` contains a Unix shebang pointing at `.venv/…`; on Windows this is not used, but keeping the venv named `.venv` matches upstream docs.

## 2) Install dependencies

```powershell
cd src/python/vendor/openmatb
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Dependencies are pinned in [src/python/vendor/openmatb/requirements.txt](../../src/python/vendor/openmatb/requirements.txt).

### Common Windows pitfalls

- `pyparallel` install failures:
  - `pyparallel==0.2.2` can fail to build/install on some Windows setups.
  - If you do not need the physical parallel-port trigger path, a pragmatic local workaround is:

```powershell
python -m pip install pyglet==1.5.31 rstr==3.1.0 pylsl==1.16.1
```

  - In that case, the `parallelport` plugin will not function (and may display an error), but the rest of OpenMATB can still run.

- `pylsl` import issues:
  - If `pylsl` imports but cannot find a runtime library, reinstalling `pylsl` or using a compatible Python version typically resolves it.

## 3) Run OpenMATB

Recommended: use the repo wrapper (supported entrypoint in this repo).

```powershell
cd src/python/vendor/openmatb
.\.venv\Scripts\Activate.ps1

python ..\..\run_openmatb.py --participant P001 --session S001
```

Notes:
- The wrapper sets `OPENMATB_OUTPUT_ROOT` / `OPENMATB_OUTPUT_SUBDIR` and also injects `OPENMATB_REPO_COMMIT` / `OPENMATB_SUBMODULE_COMMIT`.
- This repo’s OpenMATB logger writes a `.manifest.json` alongside the CSV and requires those git commit env vars; running `python main.py` directly will typically error unless you set them yourself.

### English-only (study constraint)

OpenMATB reads its locale from `language` in [src/python/vendor/openmatb/config.ini](../../src/python/vendor/openmatb/config.ini).

- Set `language=en_EN`.
- Do not delete or modify French assets.
- Do not commit local `config.ini` changes.

Controls:
- `Esc` prompts exit.
- `P` pauses (via a modal dialog).

## 4) Smoke test mode (minimal confidence run)

Because OpenMATB is scenario-driven, the “smoke test” is simply running a short built-in scenario and verifying:
- the window opens
- tasks appear
- a session log is written

Options that require no new files:
- Use the default scenario and exit after a few seconds with `Esc`.
- Switch to a shorter built-in scenario by editing `scenario_path` in [src/python/vendor/openmatb/config.ini](../../src/python/vendor/openmatb/config.ini) (e.g., `basic.txt`), then run OpenMATB (see above).

Study-owned scenarios (recommended for this repo):

- Store repo-managed scenarios under `scenarios/`.
- Point `scenario_path` in [src/python/vendor/openmatb/config.ini](../../src/python/vendor/openmatb/config.ini) at a relative path that traverses from `vendor/openmatb/includes/scenarios/` to `scenarios/`.
  - Use forward slashes for portability.
  - Example pattern: `../../../..` (to `src/python/`) + `/scenarios/openmatb/<scenario>.txt`.

## 5) Where output/logs are written

OpenMATB writes logs outside the repo by default (Windows default):

`C:/data/adaptive_matb/openmatb/`

When you run via the repo wrapper (recommended), logs are further organized under:

`C:/data/adaptive_matb/openmatb/<participant>/<session>/`

Within that folder:
- Session CSV logs: `.../sessions/YYYY-MM-DD/<n>_<timestamp>.csv`
- Session manifest: `.../sessions/YYYY-MM-DD/<n>_<timestamp>.manifest.json`
- Scenario validation errors: `.../last_scenario_errors.log`

The CSV columns are written by `core.logger.Logger`:
- Source: [src/python/vendor/openmatb/core/logger.py](../../src/python/vendor/openmatb/core/logger.py)

## 6) Redirect logs to the external data root (recommended)

Per repo policy, large runs/logs should live outside git (see [docs/DATA_MANAGEMENT.md](../DATA_MANAGEMENT.md)).

Use the repo wrapper to enforce IDs and set the correct output layout. Example (PowerShell):

```powershell
cd src/python/vendor/openmatb

# Activate your OpenMATB venv (recommended)
.\.venv\Scripts\Activate.ps1

# Required
python ..\..\run_openmatb.py --participant P001 --session S001

# Optional: override the data root
# python ..\..\run_openmatb.py --participant P001 --session S001 --output-root C:\data\adaptive_matb
```

Notes:
- The wrapper sets `OPENMATB_OUTPUT_ROOT` (default: `C:\data\adaptive_matb`) and `OPENMATB_OUTPUT_SUBDIR=openmatb/P001/S001`.
- If you run `python main.py` directly, you may still get external output (defaults to `C:\data\adaptive_matb\openmatb`), but you will typically need to set `OPENMATB_REPO_COMMIT` and `OPENMATB_SUBMODULE_COMMIT` to avoid logger startup errors.

## 7) LSL streaming

OpenMATB includes a Lab Streaming Layer (LSL) *outlet* plugin:
- Plugin source: [src/python/vendor/openmatb/plugins/labstreaminglayer.py](../../src/python/vendor/openmatb/plugins/labstreaminglayer.py)

Confirm outputs are ignored by git (inside the OpenMATB submodule):

```powershell
git -C src/python/vendor/openmatb check-ignore -v sessions/ last_scenario_errors.log .venv/
```
