# ADR-0006: SPACE-gated instruction timing in familiarisation scenario

## Status

Accepted — 2026-04-22

## Context

The familiarisation scenario (`pilot_practice_intro.txt`) shows participants an
instruction slide for each task sub-section (welcome, sysmon, tracking,
communications, resman, all-together).  Each section is followed by a 40-second
task-demo window.

The intent is that each participant gets a full 40 seconds of task time
regardless of how long they take to read the instructions.  The instructions
overlay blocks the task, and SPACE dismisses it and starts the task window.

### Original behaviour (broken)

`BlockingPlugin.update()` in `abstractplugin.py` set `self.blocking = False`
the moment the **last (only) slide was displayed**, not when it was dismissed.
This caused the scenario clock to unpause immediately on slide appearance,
before the participant had interacted at all.  Consequently:

- For a participant who takes 15 s to read a slide, only ~25 s of task
  interaction remained in their 40-second window.
- For a fast reader (5 s) they got ~35 s.
- Reading time varied unpredictably across participants, confounding the
  amount of task exposure during familiarisation.

The original OpenMATB comment said this was intentional — to allow a timed
`instructions;stop` event in the scenario to fire while a slide was still
visible.  Our familiarisation scenario does **not** use timed `instructions;stop`
events, so this rationale does not apply.

## Decision

Remove the `if len(self.slides) == 0: self.blocking = False` line from the
"show slide" branch of `BlockingPlugin.update()`.

Move `self.blocking = False` into the `else` branch (when SPACE is pressed and
no slides remain), just after `stop()` or `hide()`.

This means the scenario clock stays paused for the **entire duration** the
instruction slide is on screen — from when it appears until the participant
presses SPACE.

The 40-second demo window therefore begins at SPACE press, not at slide
appearance.  All participants receive the same task exposure regardless of
reading speed.

## Consequences

- **Familiarisation timing is consistent** across participants.
- The change is in vendor code (`src/vendor/openmatb/`) and diverges from
  upstream OpenMATB.  The diff is small (3 lines).
- Any scenario that relies on the old behaviour — showing a slide and then
  firing a **timed** `instructions;stop` to dismiss it automatically — would
  no longer work as intended (the clock wouldn't advance to reach that event
  while the slide is showing).  No such scenario exists in this repository.
- `Genericscales` (questionnaire plugin) inherits `BlockingPlugin`.
  Questionnaires stop when all scales are answered, not via SPACE, so they
  never reach the `else` branch by SPACE-press.  Unaffected.

## Files changed

- `src/vendor/openmatb/plugins/abstractplugin.py` — `BlockingPlugin.update()`
