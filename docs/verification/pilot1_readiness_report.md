# Pilot 1 readiness report (strict audit)

Date: 2026-01-28

This report is intentionally strict and skeptical: it lists what is *actually* implemented and observable today vs what remains open at the decision layer.

Canonical sources used (per instruction):
1) `docs/decisions/open_decisions.md`
2) `docs/design_choices/**/dc_*.md`
3) Contract docs explicitly marked locked/canonical (none found)

Important governance note:
- There are **no** DC files under `docs/design_choices/**/dc_*.md` in this repo.
- DC-like files exist under `docs/decisions/design_choices/**/dc_*.md` and were reviewed as *de-facto* design constraints, but this path mismatch means the “canonical sources only” rule is currently self-contradictory.
- Action: create a single canonical DC location (see checklist).

---

## 1) Pilot 1 scope

Pilot 1 = Task & protocol validation (behavioural + subjective only), focused on:
- Finalise practice/training structure (OD-03)
- Finalise calibration block semantics (OD-02) at the label/marker level (surface gaps even if calibration blocks are not fully separate yet)
- Finalise event-rate scaling defaults (OD-04)
- Finalise subjective ratings between calibration blocks (OD-05): collect in Pilot 1 or defer
- Finalise performance measurement definition (OD-06)
- Must produce deterministic logs/markers/manifests and performance summaries needed to close Pilot-1 ODs

---

## 2) Canonical requirements inventory

### Open Decisions (ODs)

From `docs/decisions/open_decisions.md`:

- **OD-02 (Pilot 1): Calibration block semantics**
  - Current pilot default: discrete single-level blocks (`LOW|MODERATE|HIGH`)
  - Evidence needed: label clarity, deterministic marker definition, transition effects
- **OD-03 (Pilot 1): Practice/training structure**
  - Current pilot default: three short blocks (~5 min each) LOW→MOD→HIGH + quiet breaks
  - Evidence needed: performance stabilisation, subjective confusion/fatigue
- **OD-04 (Pilot 1): Event-rate scaling**
  - Current pilot default: ~6× total rate LOW→HIGH (example 3 vs 18 events/min)
  - Evidence needed: subjective separation, tolerability, failure/abort rates
- **OD-05 (Pilot 1): Subjective ratings between calibration blocks**
  - Current pilot default: short-form ratings after each calibration block
  - Evidence needed: usefulness vs burden/disruption; logging quality
- **OD-06 (Pilot 1): Performance measurement definition**
  - Current pilot default: log raw events + compute standard per-subtask summaries
  - Evidence needed: stability, sensitivity, redundancy, interpretability
- **OD-07 (Pilot 2): Calibration structure for per-participant ML personalisation**
  - Not required for Pilot 1 (but keep semantics compatible)

### Design Choices (DCs)

No DCs were found at `docs/design_choices/**/dc_*.md`.

Reviewed (non-canonical *by path*, but likely intended DCs) under `docs/decisions/design_choices/study_design/`:
- `dc_study_paradigm_multitask_matb.md` (Final)
- `dc_task_set.md` (Final)
- `dc_three_workload_levels.md` (Final)
- `dc_workload_manipulation.md` (Final)
- `dc_include_practice_phase.md` (Final)

Interpretation for Pilot 1: even if the file location is wrong, these impose concrete constraints that Pilot 1 runs should not violate (task set stability, exactly three workload identifiers, explicit practice phase, etc.).

### Locked/canonical contracts

No contract docs are explicitly marked locked/canonical.
`docs/contracts/training_scenario_contract_v0.md` exists but is **draft** and **not canonical**.

---

## 3) Decision-to-implementation audit table

Legend for “Gap”:
- **Enforced**: code/verification will fail if violated
- **Observable**: logged/derivable, but not enforced
- **Neither**: not enforced and not reliably observable

| Source (OD/DC) | Requirement / pilot default | Where implemented / enforced | Evidence artifact produced | Verified? | Gap | Action needed |
|---|---|---|---|---:|---|---|
| `docs/decisions/open_decisions.md` OD-03 | Practice = 3×5min blocks LOW→MOD→HIGH (plus breaks) | `scenarios/pilot_practice_low|moderate|high.txt` markers `STUDY/V0/TRAINING/T1..T3/*`; playlist in `src/python/run_openmatb.py:_get_playlist()` | CSV marker rows (`labstreaminglayer/marker`), manifest `scenario_name` per file | No (dynamic) / Yes (static parse) | Observable (partially enforced by scenario artifacts existing) | [DECISION] confirm whether `pilot_practice_intro.txt` is part of training; [DOC] specify marker for intro or explicitly exclude intro from analyses |
| `docs/decisions/open_decisions.md` OD-02 | Calibration blocks are single-level labels | Scenario markers currently use `calibration/<LEVEL>` not `CALIBRATION/<LEVEL>`; filenames are `pilot_calibration_<level>.txt` | CSV markers: `STUDY/V0/calibration/<LEVEL>/START|...` and `/END|...` | No | Observable but semantically ambiguous | [DECISION] lock terminology: are these “calibration” or “calibration” for Pilot 1? [DOC] define marker namespace and how it maps to block semantics |
| `docs/decisions/open_decisions.md` OD-04 | Default event-rate scaling ≈6× LOW→HIGH | Actual schedule encoded in committed scenarios; generator `src/python/generate_pilot_scenarios.py` uses non-seeded randomness (non-deterministic generation) | Scenario files + derived counts (see below); CSV confirms observed schedule; verify harness can compare intended vs observed comm times | No | Observable, not enforced at decision layer | [DECISION] reconcile OD-04 default (6×) with actual implemented totals (currently ~5.1× by counts); [DOC] record per-task allocation policy |
| `docs/decisions/open_decisions.md` OD-05 | Subjective ratings after each calibration block (or defer) | Implemented via `genericscales` in `scenarios/pilot_calibration_*.txt` with markers `TLX/calibration_<LEVEL>/*` | CSV markers for TLX START/END; questionnaire responses depend on OpenMATB logging semantics; potential `performance` rows from module `genericscales` | No | Observable for presence/timing; value logging unverified | [DECISION] keep TLX in Pilot 1 or defer; [VERIFICATION] confirm questionnaire values appear in CSV deterministically; [CODE] extend summariser if values are not captured |
| `docs/decisions/open_decisions.md` OD-06 | Compute per-subtask summary metrics deterministically | `src/python/summarise_openmatb_performance.py` produces `*.performance_summary.json` from CSV `type==performance` rows; runner flag `--summarise-performance` | `*.manifest.json.performance_summary.json` including per-segment summaries + derived KPIs | Partially (script exists; runtime depends on CSV content) | Enforced only if you run summariser; otherwise neither | [VERIFICATION] ensure OpenMATB emits required `performance` rows; [DOC] define “Pilot 1 minimum KPI set” |
| `docs/decisions/design_choices/.../dc_three_workload_levels.md` | Exactly three workload identifiers (`LOW|MODERATE|HIGH`) | Scenarios embed these strings; runner seq-id maps to these | Scenario text + markers | Yes (static parse) | Enforced by artifact content; not programmatic | [DOC] move DCs to canonical path or clarify canonical location |
| `docs/decisions/design_choices/.../dc_task_set.md` | Full multitask set (sysmon/track/comms/resman + scheduling) | Scenarios start these plugins; intro hides/shows tasks | Scenario text + CSV events/performance rows | Yes (static parse) | Observable | [VERIFICATION] confirm all tasks log events as expected per run |
| `docs/decisions/design_choices/.../dc_include_practice_phase.md` | Practice is explicit + separable by markers | Training scenarios have TRAINING markers; intro lacks markers | CSV marker rows for TRAINING; none for intro | Partial | Observable | [DECISION] whether intro requires markers; [DOC] clarify what “training phase” includes |

### Concrete implemented event-rate evidence (current repo scenarios)

Derived from the committed scenario files (counts over 5:00 blocks):

- `pilot_calibration_low.txt`: sysmon failures=10, comm prompts=4, resman pump failures=2, total=16
- `pilot_calibration_moderate.txt`: sysmon=30, comm=11, resman=6, total=47
- `pilot_calibration_high.txt`: sysmon=50, comm=20, resman=12, total=82

That implies HIGH/LOW total ratio = 82/16 ≈ 5.125×, **not** 6×.

Communications minimum spacing (by scheduled timestamps) is ≥14s in MOD/HIGH and ≥31s in LOW (so it satisfies any ≥8s constraint).

---

## 4) Exactly what to do next (Pilot 1 readiness checklist)


2) [DECISION] (OD-02) Lock marker semantics for “calibration vs calibration”: either rename markers to `CALIBRATION/<LEVEL>` or explicitly state that Pilot 1 uses `calibration/<LEVEL>` markers as calibration proxies. Unblocks: OD-02.
3) [DECISION] (OD-03) Decide whether `pilot_practice_intro.txt` is part of the training phase and whether it needs study markers. Unblocks: OD-03 + analysis segmentation.
4) [DECISION] (OD-04) Decide the Pilot 1 *default* scaling rule (6× vs current ~5.1×) and whether “total rate” is defined as sysmon+comms+resman only, or includes other event classes. Unblocks: OD-04.
5) [DECISION] (OD-04) Decide event allocation policy across subtasks (fixed per-task counts vs proportional scaling). Unblocks: OD-04.
6) [DECISION] Tracking difficulty mapping: lock the parameters used for LOW/MOD/HIGH (e.g., `track.taskupdatetime`, `track.joystickforce`) and whether they are part of the workload manipulation or held constant. Unblocks: OD-04 + DC workload manipulation interpretation.
7) [VERIFICATION] Run `src/python/verification/verify_pilot.py` on a real attended run (1 participant) and archive only derived summaries/notes (not raw). Confirm:
   - markers match scenario text after token substitution
   - segments durations match 300s block
   - comm prompt times match intended schedule within tolerance
   Unblocks: OD-02/03/04.
8) [VERIFICATION] Confirm TLX responses are logged deterministically in the CSV (module/address/value semantics). Unblocks: OD-05.
9) [VERIFICATION] Confirm `*.performance_summary.json` is produced for each scenario and contains stable keys (especially per-segment summaries and derived KPIs). Unblocks: OD-06.
10) [DECISION] (OD-06) Choose Pilot 1 “minimal KPI set” to report per block/subtask (and what is QC-only). Unblocks: OD-06.

---

## 5) Pending decisions that must be resolved before running Pilot 1

These are written as proposed OD entries (exact wording + closure criteria).

### Proposed OD: Tracking difficulty manipulation parameters (Pilot 1)

**Decision question**  
Which tracking parameters define `LOW`, `MODERATE`, and `HIGH` in Pilot 1 (e.g., `taskupdatetime`, `joystickforce`, any other track dynamics)? Are these parameters part of the workload manipulation or held constant while only event rates vary?

**What will close it**  
- A table mapping level → parameter values, written into a DC file
- Verified in scenario text + observed in logs (if logged)

### Proposed OD: Event allocation policy across subtasks (Pilot 1)

**Decision question**  
When increasing total event rate with workload, how is that increase distributed across subtasks (sysmon vs comms vs resman)?

**What will close it**  
- A written rule (e.g., fixed per-task counts; proportional scaling; caps)
- A verifier that computes per-task rates from scenarios/CSV and reports them deterministically

### Proposed OD: Marker/segment naming + repetition policy (Pilot 1)

**Decision question**  
What is the canonical marker namespace and repetition policy for multiple blocks?
- Do we encode block index (`B1/B2/B3`) or workload level (`LOW/MOD/HIGH`) in marker names?
- How do we represent multiple segments of the same type across a session without ambiguity?

**What will close it**  
- A canonical marker naming spec written into a DC or locked contract
- Verification confirms markers are unique per segment within a session and match manifests

### Proposed OD: Subjective ratings collected in Pilot 1? (OD-05 tightening)

**Decision question**  
Do we collect subjective workload ratings in Pilot 1?
- If yes: which instrument (full TLX / short-form TLX / single-item), and how are responses logged in CSV + manifest?
- If no: confirm that Pilot 1 closes OD-05 by deferring and documenting rationale.

**What will close it**  
- A locked selection (instrument + timing)
- Demonstrated deterministic logging of responses (raw values retrievable from CSV)

### Proposed OD: Event-rate scaling default reconciliation (OD-04 tightening)

**Decision question**  
OD-04 default states ~6× separation LOW→HIGH; current committed scenarios implement ~5.125× (82 vs 16 counted events). Which is the Pilot 1 default?

**What will close it**  
- A written scaling rule + regenerated scenarios (or explicit acceptance of current) with verification outputs showing intended counts

---

## 6) Minimal code changes required

Only changes required to unblock deterministic Pilot 1 evidence generation and verification were applied:

1) `src/python/verification/verify_pilot.py`
   - Fix repo-root resolution and `sys.path` so it runs from repo root without manual PYTHONPATH.
   - Add TLX segment expectations matching current scenario markers (`TLX/calibration_<LEVEL>`).
   - Ensure Pilot 1 runs produce performance summaries by passing `--summarise-performance` to the runner.

2) `src/python/verification/verify_pilot_scenarios.py` (new)
   - Add missing static verifier + shared parsing utilities required by `verify_pilot.py`.

No other code changes are strictly required to *run* Pilot 1, but additional changes may be required after verification of TLX response logging and performance-row availability.

---

## 7) Risks / failure modes if we run Pilot 1 now

- Canonical-source ambiguity: DCs are not in the declared canonical location, so “what is locked” is unclear.
- OD-04 mismatch: current implemented event-rate ratio is ~5.1×, but OD-04 default says ~6×.
- `generate_pilot_scenarios.py` is non-deterministic (random without a fixed seed), so regeneration may silently change schedules.
- Intro scenario has no study markers (analysis segmentation may ignore/lose it).
- TLX value logging in CSV is unverified (markers exist, but responses may be missing or hard to parse).
- Performance summary relies on `type==performance` rows; if OpenMATB doesn’t emit these consistently for all subtasks, OD-06 evidence will be incomplete.
