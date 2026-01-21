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

### English-only (study constraint)

OpenMATB reads its locale from `language` in [src/python/vendor/openmatb/config.ini](../../src/python/vendor/openmatb/config.ini).

- Set `language=en_EN`.
- Do not delete or modify French assets.
- Do not commit local `config.ini` changes.

Controls:
- `Esc` prompts exit.
- `P` pauses (via a modal dialog).

## 4) Smoke test mode (minimal confidence run)

Procedure (no new files; do not commit local config changes):

```powershell
cd src/python/vendor/openmatb
.\.venv\Scripts\Activate.ps1

# Ensure config.ini points at the built-in basic scenario (adjust if your repo uses a different relative path)
# scenario_path=includes/scenarios/basic.txt

python main.py
```

Pass criteria (basic smoke test):
- Tasks run for about 60 seconds then stop.
- Communications audio is English.
- NASA-TLX appears after the task block.
- A CSV is written under `sessions/YYYY-MM-DD/`.
- `last_scenario_errors.log` ends with “No error”.

Confirm outputs are ignored by git (inside the OpenMATB submodule):

```powershell
git -C src/python/vendor/openmatb check-ignore -v sessions/ last_scenario_errors.log
```
