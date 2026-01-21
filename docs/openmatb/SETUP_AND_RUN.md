# OpenMATB — setup and run (Windows-first)

Related docs:
- [OVERVIEW.md](OVERVIEW.md)
- [INSTRUMENTATION_POINTS.md](INSTRUMENTATION_POINTS.md)
- Repo data boundary rules: [docs/DATA_MANAGEMENT.md](../DATA_MANAGEMENT.md)

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
pip install pyglet==1.5.26 rstr==3.1.0 pylsl==1.16.1
```

  - In that case, the `parallelport` plugin will not function (and may display an error), but the rest of OpenMATB can still run.

- `pylsl` import issues:
  - If `pylsl` imports but cannot find a runtime library, reinstalling `pylsl` or using a compatible Python version typically resolves it.

## 3) Run OpenMATB

```powershell
cd src/python/vendor/openmatb
.\.venv\Scripts\Activate.ps1

python main.py
```

### Switch language (English/French)

OpenMATB reads its locale from `language` in [src/python/vendor/openmatb/config.ini](../../src/python/vendor/openmatb/config.ini).

Study policy (this repo):
- Use English-only for UI/audio/questionnaires.
- Do not delete or modify French assets; they remain available upstream.
- Do not commit local `config.ini` tweaks.

- English: `language=en_EN`
- French: `language=fr_FR`

After changing `config.ini`, restart OpenMATB.

Controls:
- `Esc` prompts exit.
- `P` pauses (via a modal dialog).

## 4) Smoke test mode (minimal confidence run)

Because OpenMATB is scenario-driven, the “smoke test” is running a short scenario and verifying:

Procedure (no new files; do not commit local config changes):

```powershell
cd src/python/vendor/openmatb
.\.venv\Scripts\Activate.ps1

# Ensure config.ini points at the built-in basic scenario (adjust if your repo uses a different relative path)
# scenario_path=includes/scenarios/basic.txt

python main.py
```

- `last_scenario_errors.log` ends with “No error”.

## 5) Where output/logs are written

By default OpenMATB writes to a local `sessions/` folder *relative to its working directory*:
- Session CSV logs: [src/python/vendor/openmatb/sessions/](../../src/python/vendor/openmatb/sessions/)
- Scenario validation errors: [src/python/vendor/openmatb/last_scenario_errors.log](../../src/python/vendor/openmatb/last_scenario_errors.log)

The CSV columns are written by `core.logger.Logger`:
- Source: [src/python/vendor/openmatb/core/logger.py](../../src/python/vendor/openmatb/core/logger.py)

## 6) Confirm logs remain untracked

This repository must not commit raw session logs. For the smoke test, OpenMATB should still write logs locally, but git must ignore them.

Verify ignore rules inside the submodule:

```powershell
git -C src/python/vendor/openmatb check-ignore -v sessions/ last_scenario_errors.log
```

## 7) LSL streaming

OpenMATB includes a Lab Streaming Layer (LSL) *outlet* plugin:
- Plugin source: [src/python/vendor/openmatb/plugins/labstreaminglayer.py](../../src/python/vendor/openmatb/plugins/labstreaminglayer.py)

Two modes exist:
- Push explicit markers (`marker` parameter).
- Stream the full CSV rows by enabling `streamsession=True`.

This is scenario-controlled; see [INSTRUMENTATION_POINTS.md](INSTRUMENTATION_POINTS.md).
