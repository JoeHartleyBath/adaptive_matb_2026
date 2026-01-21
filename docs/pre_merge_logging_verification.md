# Pre-merge logging verification (OpenMATB)

Date: 2026-01-21  
Host OS: Windows  
Scope: **verification only** (no product-code changes intended; local-only run configuration tweaks were reverted)

This document is a pre-merge gate for the branch `fix/run-manifest-compliance`.

## Check 1 — Submodule pointer hygiene (PASS/FAIL)

Goal: ensure the OpenMATB submodule gitlink in the parent repo points to a commit that is reachable on a remote branch (or otherwise reviewable via PR), so merging does not create a dangling submodule reference.

**Evidence**

- Parent repo HEAD: `26c01a296ac321b47ad050857724b4b9c092f15a`
- Submodule gitlink at `src/python/vendor/openmatb`: `e3f77226907424e9711e95967c118c959fa01903`
- Submodule remote: `origin https://github.com/JoeHartleyBath/OpenMATB.git`

Commands run:

- `git ls-tree HEAD src/python/vendor/openmatb`
- `git -C src/python/vendor/openmatb fetch origin`
- `git -C src/python/vendor/openmatb branch -r --contains e3f77226907424e9711e95967c118c959fa01903`

Result:

- `git branch -r --contains ...` returned **no remote branches**.

**Status: FAIL**

**Merge recommendation**: do not merge until the submodule commit `e3f7722…` is pushed to a remote branch and/or has a PR on the upstream repo, or the parent repo gitlink is moved back to a reachable upstream commit.

## Check 2 — Single-run artifact verification (PASS/FAIL)

Goal: run one short scenario via the wrapper and confirm external artifacts exist and are well-formed:

- CSV event log written under `C:\data\adaptive_matb\openmatb\…`
- Manifest JSON written **adjacent to the CSV**
- Manifest includes required top-level keys and `ended_at_local` is non-null after clean exit

### Run configuration

Wrapper invocation:

- Working dir: `src/python/vendor/openmatb`
- Python: `.venv\\Scripts\\python.exe`
- Command:
  - `..\\..\\run_openmatb.py --participant PTEST --session STEST --output-root C:\\data\\adaptive_matb`

Scenario used (short): `includes/scenarios/mwe/issue_19_callsign_A.txt`

Note (local-only): `config.ini` contained a UTF-8 BOM (`EF BB BF`) which caused `configparser.MissingSectionHeaderError` (BOM prevents `#` from being recognized as a comment prefix). For the purpose of this verification run only, the BOM was stripped and the scenario was set to the short `mwe/issue_19_callsign_A.txt`, then reverted.

### Artifacts produced

External output base:

- `C:\\data\\adaptive_matb\\openmatb\\PTEST\\STEST`

Session artifacts:

- CSV: `C:\\data\\adaptive_matb\\openmatb\\PTEST\\STEST\\sessions\\2026-01-21\\1_260121_151522.csv`
- Manifest: `C:\\data\\adaptive_matb\\openmatb\\PTEST\\STEST\\sessions\\2026-01-21\\1_260121_151522.manifest.json`

CSV header / metadata spot check:

- Header: `logtime,scenario_time,type,module,address,value`
- Includes `version` row and absolute `scenario_path` row.

Manifest spot check:

- `started_at_local`: `2026-01-21T15:15:22`
- `ended_at_local`: `2026-01-21T15:15:31` (non-null)
- `repo_commit`: `26c01a296ac321b47ad050857724b4b9c092f15a`
- `submodule_commit`: `e3f77226907424e9711e95967c118c959fa01903`
- `participant_id`: `PTEST`
- `session_id`: `STEST`
- `scenario_name`: `issue_19_callsign_A`
- `lsl_enabled`: `false`
- `output_dir` and `event_log_path`: absolute paths under `C:\\data\\adaptive_matb\\openmatb\\…`

Required key presence check:

- Verified the manifest contains these top-level keys:
  `manifest_version`, `created_at_local`, `started_at_local`, `ended_at_local`,
  `repo_commit`, `submodule_commit`, `scenario_name`, `participant_id`, `session_id`,
  `lsl_enabled`, `output_dir`, `event_log_path`.

**Status: PASS**

## Overall gate

- Check 1 (submodule reachability): **FAIL**
- Check 2 (artifacts + manifest compliance): **PASS**

**Overall: DO NOT MERGE** until Check 1 is resolved.
