# Two-run no-overwrite verification (OpenMATB logging)

Date: 2026-01-21  
Host OS: Windows  
Scope: verification only (no product-code changes)

This report verifies that two non-replay runs with different participant/session IDs write distinct outputs and do not overwrite prior run artifacts.

## Run #1 (PTEST1 / STEST1)

### Invocation

- Working dir: `C:\phd_projects\adaptive_matb_2026\src\python\vendor\openmatb`
- Command:
  - `.\.venv\Scripts\python.exe ..\..\run_openmatb.py --participant PTEST1 --session STEST1`

### Evidence (paths + rules)

- Outputs are outside the repo (external root): `C:\data\adaptive_matb` (outside `C:\phd_projects\adaptive_matb_2026`)
- Absolute output directory (record):
  - `C:\data\adaptive_matb\openmatb\PTEST1\STEST1`
- Filenames / directories encode IDs:
  - Directory path includes `PTEST1` and `STEST1`.

### Artifact files (existence + adjacency)

- CSV (record):
  - Path: `C:\data\adaptive_matb\openmatb\PTEST1\STEST1\sessions\2026-01-21\1_260121_153234.csv`
  - Size (bytes): 104757
  - Modified time (local): 2026-01-21 15:32:51
  - SHA-256: `A5ED9D4F7D273FA05A481CBAC6024C2FE1DB264195FDD6229A1DA27815682000`
- Manifest (record, adjacent + same stem):
  - Path: `C:\data\adaptive_matb\openmatb\PTEST1\STEST1\sessions\2026-01-21\1_260121_153234.manifest.json`
  - Size (bytes): 1613
  - Modified time (local): 2026-01-21 15:32:51
  - SHA-256: `F4D1925EB67C3A2FAB78D974645584D7990609BCD0C0C7F7F2732054E8AD4AAA`

### Manifest content checks

- `started_at_local`: `2026-01-21T15:32:34`
- `ended_at_local`: `2026-01-21T15:32:51`
- Timestamp ordering: `started_at_local < ended_at_local` (PASS)
- File modified times align with timestamps: CSV/manifest modified time equals `ended_at_local` seconds (PASS)
- `participant_id`: `PTEST1`
- `session_id`: `STEST1`
- `scenario_name`: `default`
- `lsl_enabled`: `false`
- `output_dir` (absolute): `C:\data\adaptive_matb\openmatb\PTEST1\STEST1`
- `event_log_path` (absolute, exists, points to CSV):
  - `C:\data\adaptive_matb\openmatb\PTEST1\STEST1\sessions\2026-01-21\1_260121_153234.csv`

### Commit identifiers (record)

- `repo_commit`: `b74a8606b8a542e507335bae01e42e22bfd19b0a`
- `submodule_commit`: `e3f77226907424e9711e95967c118c959fa01903`

### Run #1 snapshot (for overwrite detection)

| File | Path | Size (bytes) | Modified time (local) | SHA-256 |
|---|---|---:|---|---|
| CSV | `C:\data\adaptive_matb\openmatb\PTEST1\STEST1\sessions\2026-01-21\1_260121_153234.csv` | 104757 | 2026-01-21 15:32:51 | `A5ED9D4F7D273FA05A481CBAC6024C2FE1DB264195FDD6229A1DA27815682000` |
| Manifest | `C:\data\adaptive_matb\openmatb\PTEST1\STEST1\sessions\2026-01-21\1_260121_153234.manifest.json` | 1613 | 2026-01-21 15:32:51 | `F4D1925EB67C3A2FAB78D974645584D7990609BCD0C0C7F7F2732054E8AD4AAA` |

## Run #2 (PTEST2 / STEST2)

### Invocation

- Working dir: `C:\phd_projects\adaptive_matb_2026\src\python\vendor\openmatb`
- Command:
  - `.\.venv\Scripts\python.exe ..\..\run_openmatb.py --participant PTEST2 --session STEST2`

### Evidence (distinct output dir)

- Absolute output directory (record):
  - `C:\data\adaptive_matb\openmatb\PTEST2\STEST2`
- Distinct from Run #1 output directory (PASS)

### Artifact files (existence + adjacency)

- CSV (record):
  - Path: `C:\data\adaptive_matb\openmatb\PTEST2\STEST2\sessions\2026-01-21\1_260121_153359.csv`
  - Size (bytes): 127122
  - Modified time (local): 2026-01-21 15:34:08
  - SHA-256: `2624842FD362DCA4EE0C00E67E407EF812AC2836B93AF7B3F67317890BD71FC2`
- Manifest (record, adjacent + same stem):
  - Path: `C:\data\adaptive_matb\openmatb\PTEST2\STEST2\sessions\2026-01-21\1_260121_153359.manifest.json`
  - Size (bytes): 1613
  - Modified time (local): 2026-01-21 15:34:08
  - SHA-256: `3C283A4751811E983A9455683CC80DDBDBA6B1BF3E667BACB7E65310470EA200`

### Manifest content checks

- `started_at_local`: `2026-01-21T15:33:59`
- `ended_at_local`: `2026-01-21T15:34:08`
- Timestamp ordering: `started_at_local < ended_at_local` (PASS)
- File modified times align with timestamps: CSV/manifest modified time equals `ended_at_local` seconds (PASS)
- `participant_id`: `PTEST2`
- `session_id`: `STEST2`
- `scenario_name`: `default`
- `lsl_enabled`: `false`
- `output_dir` (absolute): `C:\data\adaptive_matb\openmatb\PTEST2\STEST2`
- `event_log_path` (absolute, exists, points to CSV):
  - `C:\data\adaptive_matb\openmatb\PTEST2\STEST2\sessions\2026-01-21\1_260121_153359.csv`

### Commit identifiers (record)

- `repo_commit`: `b74a8606b8a542e507335bae01e42e22bfd19b0a`
- `submodule_commit`: `e3f77226907424e9711e95967c118c959fa01903`

## Overwrite verdict

**PASS**

- Run #2 output directory is distinct from Run #1.
- Run #1 CSV + manifest overwrite check after Run #2:
  - Same path as snapshot (PASS)
  - Same size as snapshot (PASS)
  - Same modified time as snapshot (PASS)
  - Same SHA-256 hash as snapshot (PASS)

## Fixes

- None (no overwrite observed; no code changes made)
