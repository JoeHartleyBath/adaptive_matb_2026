# Participant Assignment System

## Overview

The participant assignment system tracks completed sessions and participant condition
assignments to minimise human error during data collection.

---

## Participant Namespaces

Three run tiers are kept strictly separate. IDs never overlap.

| Namespace | Range | Launcher | Who | Data fate |
|-----------|-------|----------|-----|-----------|
| `PDRY01`–`PDRY05` | 5 slots | Direct Python invocation | Lab colleagues/friends; non-target-population; **pre-ethics** | Never analysed. Throwaway. |
| `PSELF` | 1 slot | Direct Python invocation | Researcher self-test | Never analysed. |
| `DEV` | 1 slot | Manual | Development / debugging | Never analysed. |
| `PPILOT01`–`PPILOT10` | 10 slots | `session_start_PILOT.bat` | Target-population participants; formal protocol; **post-ethics** | Excluded from study analysis; used for protocol QA. |
| `P001`–`P030` | 30 slots | `session_start_STUDY.bat` | Consented target-population participants | **Included in analysis.** |

> **Why the dry-run / pilot distinction matters:** a pilot participant requires ethics
> approval and recruitment from the target population. `PDRY` slots are for informal
> runs with lab colleagues *before* ethics approval — to verify hardware, software, and
> protocol flow. Keep these namespaces separate so pilot and study data are never
> contaminated.

---

## Fields per participant entry

```yaml
P012:
  adaptation_first: true      # determines condition order (adaptation→control or control→adaptation)
  sessions_completed: []      # auto-populated after each run
  last_run: null              # auto-populated after each run
```

There is no `sequence` field. Calibration block order is determined automatically
by participant rank and condition via the 6-template system in
`scripts/generate_scenarios/generate_full_study_scenarios.py`.

---

## Running a session

### Study or pilot participant (using launcher)

```
Double-click  →  STUDY SESSION  (or PILOT SESSION)
Enter number  →  e.g.  12   (builds P012 or PPILOT12 automatically)
Press Enter to confirm  →  session starts
```

### Dry-run or self-test participant (direct Python)

```bash
python src/run_full_study_session.py --participant PDRY01
python src/run_full_study_session.py --participant PSELF
```

Session is auto-incremented from `sessions_completed` length (S001, S002, …).

---

## After a session

`config/participant_assignments.yaml` is updated automatically:

```yaml
P012:
  adaptation_first: true
  sessions_completed: [S001]          # appended
  last_run: '2026-04-02T14:30:45'    # timestamp added
```

This prevents session number conflicts and serves as the audit log.

---

## Adding participants mid-study

Edit `config/participant_assignments.yaml` directly. Required fields:
- `adaptation_first` — alternated true/false in sequence with existing participants
- `sessions_completed: []`
- `last_run: null`

---

## Troubleshooting

**"Participant not found in assignments file"**
→ Add the participant entry to `config/participant_assignments.yaml` before running.

**Want to re-run a session that was aborted?**
→ Use `--start-phase N` to resume from the last completed phase.  
→ Or manually edit `sessions_completed` to roll back if needed.
