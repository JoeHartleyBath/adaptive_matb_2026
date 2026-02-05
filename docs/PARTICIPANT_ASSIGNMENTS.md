# Participant Assignment System

## Overview

The participant assignment system manages sequence assignments and tracks completed sessions to minimize human error during data collection.

## Quick Start

### 1. Generate Assignments (Before Data Collection)

Generate counterbalanced assignments for your study:

```bash
# Generate 10 participants per sequence (30 total)
python scripts/generate_participant_assignments.py --n-per-sequence 10

# Or start from a specific number
python scripts/generate_participant_assignments.py --n-per-sequence 10 --start 101

# Or create custom assignments
python scripts/generate_participant_assignments.py \
    --participant-ids P001 P002 P003 \
    --sequences SEQ1 SEQ2 SEQ3
```

This creates `config/participant_assignments.yaml` with:
- Counterbalanced sequence assignments
- Session tracking structure
- Last-run timestamps

### 2. Run Data Collection

Simply press play in VS Code on `run_openmatb.py` or run:

```bash
python src/python/run_openmatb.py
```

**The script will:**
1. Show recently run participants (if any)
2. Prompt for participant selection (number or list selection)
3. **Automatically look up** the assigned sequence
4. **Auto-increment** session number (S001, S002, etc.)
5. Show confirmation prompt
6. Track completion after successful run

**Example interaction:**
```
=== OpenMATB Session Setup ===

Recent participants:
  1. P003 (SEQ3, 1 sessions)
  2. P001 (SEQ1, 1 sessions)

Enter participant number or selection (1-5): 2

==================================================
  Participant: P002
  Sequence:    SEQ2
  Session:     S001
==================================================

Proceed with this configuration? (y/n): y
```

### 3. View Assignment Status

Check `config/participant_assignments.yaml` to see:
- Which participants have run
- Completed sessions for each
- Last run timestamps

```yaml
participants:
  P001:
    last_run: '2026-02-05T14:30:00.123456'
    sequence: SEQ1
    sessions_completed: [S001]
  P002:
    last_run: null
    sequence: SEQ2
    sessions_completed: []
```

## Features

### High Priority (Implemented)
- ✅ **Automatic sequence lookup** - No manual sequence entry
- ✅ **Session tracking** - Knows what sessions are completed
- ✅ **Confirmation prompt** - Review before running
- ✅ **Recent participants list** - Quick selection from last 5
- ✅ **Auto-increment sessions** - S001 → S002 for re-runs
- ✅ **Assignment generator** - Create counterbalanced assignments

### Error Prevention
- ❌ **Can't mix up sequences** - Looked up automatically
- ❌ **Can't duplicate sessions** - Auto-incremented
- ❌ **Can't forget assignment** - Loaded from file
- ✅ **Confirmation check** - See exactly what will run

### Testing Modes

**Dry-run (skip assignment updates):**
```bash
python src/python/run_openmatb.py --skip-assignment-update
```

**Manual override (for special cases):**
```bash
python src/python/run_openmatb.py --participant P999 --seq-id SEQ1 --session S001
```

## Workflow for Pilot Study

1. **Before first session:**
   ```bash
   # Generate assignments for your expected N
   python scripts/generate_participant_assignments.py --n-per-sequence 10
   ```

2. **For each data collection session:**
   - Press play in VS Code on `run_openmatb.py`
   - Enter participant number (or select from recent)
   - Confirm the displayed configuration
   - Run the study

3. **After data collection:**
   - `config/participant_assignments.yaml` serves as audit log
   - Shows who ran, when, and which sessions completed

## Merge Additional Participants

If you need to add more participants mid-study:

```bash
# Add 5 more per sequence (preserves existing data)
python scripts/generate_participant_assignments.py --n-per-sequence 5 --start 11 --merge
```

## Troubleshooting

**"Participant not found in assignments"**
- Enter sequence manually, or add to assignments file first

**"Overriding assigned sequence"**
- You're manually specifying a different sequence than assigned
- Confirm this is intentional

**Want to re-run a failed session?**
- Use `--session` to specify the session explicitly
- Or remove it from `sessions_completed` in the yaml file

## File Locations

- Assignments: `config/participant_assignments.yaml`
- Generator script: `scripts/generate_participant_assignments.py`
- Runner script: `src/python/run_openmatb.py`
- Data output: `C:\data\adaptive_matb\openmatb\{participant}\{session}\`
