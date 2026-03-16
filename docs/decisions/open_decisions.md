# Open design decisions

This file tracks **explicitly open design questions** that must be resolved
before they can become committed design choices.

Rules:
- Open decisions do NOT belong in `design_choices/`
- Each item must state what evidence or constraint will close it
- When a decision is closed, it is removed from this file and promoted to
  `design_choices/<category>/dc_*.md`
  - Open decisions must be reviewed after each pilot phase.
- Any OD older than one pilot cycle must be either closed, split, or explicitly deferred.

## Open decisions index (status snapshot)

- OD-02: Calibration block semantics — Pilot 1
- OD-03: Practice block structure — Pilot 1
- OD-04: Event-rate scaling — Pilot 1
- OD-05: Subjective ratings during calibration — Pilot 1
- OD-06: Performance metrics — Pilot 1 → Pilot 2
- OD-07: Calibration structure for ML — Pilot 2
- OD-08: Event scheduling policy — Pilot 1
- OD-09: Tracking difficulty operationalisation — Pilot 1
- OD-10: Experience sampling during calibration — **pre-Pilot 1**
- OD-11: MODERATE placement — Pilot 1

Note: the **full-study calibration structure is locked** (post-Pilot 1). See the committed design choice:
- `docs/decisions/design_choices/study_design/dc_full_study_calibration_structure.md`

---

## Design note: Proposed multi-phase piloting structure (non-canonical)

This is a **working design note**, not a committed decision.
It records a plausible piloting decomposition to guide current reasoning
and avoid premature convergence, without locking protocol details.

### Proposed phases

**Pilot 1 — Task & protocol validation**
- Scope: practice structure, calibration structure, event-rate scaling,
  performance metrics, subjective ratings.
- Data: behavioural + subjective only.
- Outcome: close OD-02 through OD-07 where possible.

**Pilot 2 — Physiological & ML feasibility**
- Scope: signal quality, marker alignment, general model training,
  per-participant personalisation using calibration blocks.
- Data: physiology during calibration blocks only (optionally no adaptation).
- Outcome: verify ML feasibility before adaptation.

**Pilot 3 — Adaptation validation**
- Scope: real-time adaptation behaviour, responsiveness, stability, safety.
- Data: full closed-loop runs using locked task + calibration design.
- Outcome: validate adaptation logic without changing task semantics.

### Open questions tied to this note
- Is calibration-only data collection required to train the general model?
- Can Pilot 2 and Pilot 3 be merged without risk?
- Do ethics/time constraints permit three distinct pilot phases?

This note should be revisited when deciding whether to formalise
OD-08 (Pilot phase decomposition).



---


### OD-02: Calibration block semantics (single-level vs mixed-level blocks)

**Decision question**  
Should calibration data be collected in blocks that contain a single, stable workload level, or in blocks that include multiple workload levels or sequences within the same block?

**Why this is open**  
This choice determines the fundamental semantics of calibration labels and their alignment to EEG windows. Mixed-level blocks may improve efficiency but risk ambiguity in label assignment and transition effects; single-level blocks are simpler but may require more total time.

**Current pilot default**  
Discrete calibration blocks, each containing a single workload level (`LOW`, `MODERATE`, or `HIGH`).

**Options under consideration**
- Single-level blocks only (one workload level per block).
- Mixed-level or sequential blocks (multiple workload levels within a block).

**What pilot evidence should resolve this**
- Label clarity and ease of window assignment.
- Impact of transitions on physiological signals.
- Feasibility of deterministic marker definition.

**Decision trigger**  
Lock before finalising scenario and marker contracts.
**Closure phase:** Pilot 1 (task & protocol validation)

### OD-03: Practice (training) block structure prior to calibration blocks

**Decision question**  
What training block structure (count, duration, ordering) is sufficient to ensure task familiarity without inducing unnecessary fatigue?

**Why this is open**  
The literature reports high variability in MATB(-II) training practices with no clear optimal duration, and pilot data is needed to assess learning curves, fatigue, and participant burden under the current setup.

**Current pilot default**  
Three short training blocks (~5 min each) spanning increasing workload levels (LOW → MODERATE → HIGH), separated by brief quiet breaks.

**Options under consideration**  
- Fewer, longer blocks (e.g., 2 × 7–8 min).
- More, shorter blocks (e.g., 4 × 3–4 min).
- Alternative ordering (e.g., fixed MODERATE only, or randomized).

**What pilot evidence should resolve this**  
- Stabilisation of performance metrics across training blocks.
- Subjective reports of confusion, overload, or fatigue.
- Absence of strong learning effects bleeding into calibration blocks.

**Decision trigger**  
Lock after first pilot run with ≥N participants once performance and subjective measures stabilise across calibration blocks.
**Closure phase:** Pilot 1 

**Downstream impact if changed**  
Affects scenario timing, total session duration, and operator run procedures, but does not invalidate calibration data or contracts.

### OD-04: Event rate scaling between LOW and HIGH workload

**Decision question**  
What total event-rate separation between LOW and HIGH workload levels produces reliable workload differentiation without causing excessive failure, frustration, or dropout?

**Why this is open**  
Although prior work reports wide variability in absolute event rates, and reviews suggest a common ≈6× increase between low and high workload levels, tolerability and effective separation are highly task- and population-dependent and must be confirmed in pilot data.

**Current pilot default**  
Target a total event rate of approximately 6× (5.1x) between LOW and HIGH workload levels (e.g., 3 vs 18 events/min), applied as a multiplicative rule across scenario schedules.

**Options under consideration**  
- Smaller separation (e.g., 3×–4×) to improve tolerability.
- Larger separation (>6×) to maximise condition differentiation.
- Participant-specific or performance-adaptive scaling based on training performance (e.g., Bowers et al., 2014).
- Non-multiplicative targets defined per task rather than by total rate.

**What pilot evidence should resolve this**  
- Separation of subjective workload ratings across levels.
- Task performance stability and failure/abort rates.
- Signs of overload or disengagement at the HIGH level.

**Decision trigger**  
Lock after pilot evaluation once a scaling rule reliably separates workload levels without excessive failure or participant burden.
**Closure phase:** Pilot 1 (behavioural + subjective sufficiency)

**Downstream impact if changed**  
Affects scenario schedules, workload labels, and analysis assumptions, but does not invalidate calibration data if labels remain ordinal and logged consistently.

**References**  
- Pontiggia et al., 2024 (review summarising event-rate ranges and common multiplicative scaling patterns)
- Bowers et al., 2014


### OD-05: Subjective ratings between calibration blocks

**Decision question**  
Should calibration blocks be followed by subjective ratings, and if so, what minimal set of questions provides useful information without excessive time burden or task-flow disruption?

**Why this is open**  
Subjective ratings during calibration may provide manipulation checks and auxiliary labels, but they introduce time cost and potential disruption. Pilot data is required to determine whether the added value outweighs these costs under the current protocol.

**Current pilot default**  
Short-form subjective ratings collected immediately after each calibration block.

**Options under consideration**  

**Primary instrument choice**
- **A: Full NASA-TLX after each calibration block**  
  High construct coverage; high burden and disruption.
- **B: Short-form TLX (1–3 items) after each block** *(recommended provisional)*  
  e.g. Mental Demand, Effort, Temporal Demand.
- **C: Single-item workload rating after each block**  
  Minimal burden; reduced construct resolution.
- **D: No ratings during calibration**  
  Rely on task-defined labels only; no subjective manipulation check.

**Secondary subjective items (optional / additive)**  
If subjective ratings are included, should non-TLX items be added?

- None (TLX-only)
- TLX + single stress or arousal item
- TLX + valence/arousal pair
- Custom single-item workload only (no TLX)

**Constraints**
- Total rating time per block ≤ 30–45 seconds.
- All items must refer explicitly to the immediately preceding block.
- Non-TLX items, if included, are auxiliary only (not primary model labels).

**What pilot evidence should resolve this**  
- **Usefulness:** ratings show ordered separation (LOW < MODERATE < HIGH) with acceptable variance.
- **Burden:** completion time, participant feedback, missingness.
- **Disruption:** observable performance reset or learning effects following ratings.
- **Downstream value:** ratings add interpretive or QC value beyond task-defined workload labels.

**Decision trigger**  
Lock after first pilot run with ≥N participants once:
- workload separation is confirmed or rejected, and
- burden/disruption is judged acceptable or excessive.
**Closure phase:** Pilot 1

**Operational notes (if included)**  
- Ratings administered within 15–30s after block end.
- Logged with timestamp, block ID, and instrument version.
- Instruction: “Rate the workload you experienced during the immediately preceding block only.”

**Downstream impact if changed**  
Affects session duration, operator procedure, and availability of auxiliary subjective labels; does not affect workload manipulation, scenario structure, or core contracts.

**References**  
- Internal: `docs/pilot/pilot_build_plan.md`
- Internal: `docs/pilot/pilot_study_spec.md`

### OD-06: Performance measurement definition for MATB(-II) blocks

**Decision question**  
What performance metrics should be computed and logged per block (and per subtask) to support workload validation, QC, and downstream analyses without overfitting or unnecessary complexity?

**Why this is open**  
MATB(-II) offers multiple performance signals across subtasks (accuracy, RT, error rates, tracking deviation, response latency), but the utility and stability of these metrics under the current task configuration must be validated in pilot data.

**Current pilot default**  
Log all available raw task events and compute standard per-subtask summary metrics at block level.

**Options under consideration**

**Primary performance metrics**
- **A: Subtask-specific metrics only**  
  - Tracking: RMS deviation / error integral  
  - SYSMON / COMM: accuracy, misses, false alarms  
  - Resource management: stability / violations  
- **B: Composite performance index**  
  - Weighted or normalised aggregate across subtasks.
- **C: Minimal performance checks only**  
  - Use performance solely for QC/manipulation checks, not modelling.

**Temporal resolution**
- Block-level aggregates only.
- Block-level + coarse rolling summaries.
- Fine-grained time-resolved performance aligned to EEG windows.

**Role of performance in analysis**
- Manipulation check only.
- Auxiliary correlates (not labels).
- Used for adaptive logic (future / not v0).

**Constraints**
- Metrics must be computable deterministically from logs.
- Definitions must be stable across workload levels.
- Performance metrics are **not** primary ML labels in v0.

**What pilot evidence should resolve this**  
- Metric stability across participants.
- Sensitivity to workload level without ceiling/floor effects.
- Redundancy across metrics (correlation / collinearity).
- Interpretability for reporting and supervision.

**Decision trigger**  
Lock after pilot analysis once:
- a minimal, stable metric set is identified, and
- unnecessary or unstable metrics are eliminated.
**Closure phase:** Pilot 1 (behavioural), with optional confirmation in Pilot 2

**Downstream impact if changed**  
Affects analysis scripts, reporting, and interpretation; does not affect task design, scenarios, or workload manipulation.

**References**  
- Internal: `docs/pilot/pilot_study_spec.md`
- Pontiggia et al., 2024 (review of MATB(-II) performance measures and workload sensitivity)


### OD-07: Calibration block structure for per-participant ML personalisation

**Dependency**  
This decision assumes OD-02 resolves to the use of discrete, single-level calibration blocks (i.e., each block contains one stable workload level).

**Decision question**  
Given the locked full-study calibration structure, what *subset/aggregation strategy* and *total calibration evidence* is sufficient to support per-participant model personalisation prior to adaptation, without excessive session length or fatigue?

**Why this is open**  
The calibration phase is intended to provide participant-specific data for ML personalisation. The full-study block structure is now locked, but it is unvalidated with respect to model stability, calibration curve quality, and downstream adaptation performance. Pilot data is required to determine sufficiency and efficiency.

**Locked full-study structure (post-Pilot 1)**  
Two 9-minute calibration conditions, each containing nine 1-minute blocks: 3×`LOW`, 3×`MODERATE`, 3×`HIGH`.

**Current Pilot 1 default (implementation-stable)**  
Three calibration blocks of 5 minutes each, one per workload level (`LOW`, `MODERATE`, `HIGH`), ordered using a Latin square across participants.

**Options under consideration**

**Using the locked structure for personalisation**
- Use all 18 minutes (all blocks) for personalisation.
- Use only the first condition (9 minutes) to reduce burden.
- Use a balanced subset (e.g., 1–2 minutes per level per condition) if diminishing returns are observed.

**Aggregation / labels**
- Treat each 1-minute block as an independent labelled segment.
- Aggregate repeated minutes within each level to form more stable per-level features.


**Workload coverage**
- Use all three levels (default).
- If needed for efficiency: focus on `LOW` and `HIGH` only (drop `MODERATE`) while preserving protocol clarity.

**Constraints**
- Total calibration duration should remain ≤ ~20 minutes.
- Calibration blocks use the same task set as calibration/adaptation blocks.
- Calibration data may be used for per-participant model personalisation, but not for primary outcome analysis.
- Calibration blocks are treated as independent units; within-block workload transitions are out of scope for this decision (see OD-02).

**What pilot evidence should resolve this**  
- **Model stability:** personalised model parameters converge or stabilise with available calibration data.
- **Separability:** calibration data yields separable workload representations at the participant level.
- **Efficiency:** additional calibration time yields diminishing returns for model performance.
- **Burden:** participant fatigue or disengagement during calibration remains acceptable.

**Decision trigger**  
Lock after pilot analysis once:
- personalised model performance and stability plateau with a given calibration structure, and
- session duration and participant burden are acceptable.
**Closure phase:** Pilot 2 (physiology + ML feasibility)

**Downstream impact if changed**  
Affects scenario timing, marker structure, calibration data volume, and ML personalisation procedures; may require updates to calibration-related contracts.

**References**  

### OD-08: Event scheduling policy across subtasks and workload levels

**Decision question**  
How should workload difficulty map to event counts and event timing across SYSMON, COMM, and RESMAN (and continuous tracking parameters), including whether there is a global event budget and whether cross-task event overlap is allowed?

**Why this is open**  
This policy determines what “LOW/MODERATE/HIGH” actually means operationally (event frequency, concurrency, and interruption load). If left implicit in code, it becomes hard to audit, tune, or justify, and may silently conflict with OD-04 (event-rate scaling).

**Current implementation (as-is, v0)**  
- **No global event budget.** Each task independently schedules events from the same difficulty scalar; total event rate is the sum of per-task rates.
- **Cross-task overlap allowed.** Tasks schedule independently; events can collide in time.
- **Block length fixed:** 300s.
- **TRACK:** no discrete events; difficulty sets continuous control parameters at t=0 (e.g., `taskupdatetime`, `joystickforce`).
- **SYSMON:** event count derived from an occupancy model using `alerttimeout` + refractory (≈11s slot), capped at 27; schedules *both* lights failures and scale failures at `events_N` each.
  - Approx: LOW=5 (10 total), MOD=15 (30 total), HIGH=25 (50 total).
- **COMM:** event count derived from occupancy model using mean prompt duration + refractory (≈14s slot), capped at 21.
  - Approx: LOW=4, MOD=11, HIGH=20.
- **RESMAN:** pump failures `num_failures = int(13.33*difficulty - 0.66)` with `single_duration=15s`.
  - Approx: LOW=1, MOD=6, HIGH=12.
- **Within-task timing:** `distribute_events` places events by drawing random delays that sum to remaining free time, then advances time by event duration.

**Options under consideration**
- **A: Keep emergent total event rate** (no global budget; current approach) and tune difficulty mappings per task.
- **B: Enforce a global event budget** per block and allocate across tasks (fixed proportions or weights), reducing drift and overlap.
- **C: Add an overlap limiter** (soft or hard) so events from different tasks do not occur within X seconds of each other.
- **D: Define “event rate” reporting scope**: count only discrete events with duration vs also include t=0 parameter changes.

**What pilot evidence should resolve this (Pilot 1)**  
- **Observable event-rate report:** per-task and total event counts/min for each workload level (computed from scenario text or logs).
- **Manipulation check:** subjective workload separation LOW < MOD < HIGH without excessive variance.
- **Tolerability:** acceptable failure/abort/timeout rates; no consistent overload at HIGH.
- **Overlap sanity:** quantify collision rate (e.g., % events within 1s across tasks); verify not pathological.

**Decision trigger**  
Lock a v1 event scheduling policy after Pilot 1 once event-rate reports, workload separation, and tolerability are acceptable.

**Downstream impact if changed**  
Affects scenario generation, workload operationalisation, and any “event-rate scaling” claims (OD-04); may require updating scenario generator and verification scripts.

**References**  
- Internal: (add the generator script path and any relevant DC/OD links once decided)

### OD-09: Tracking difficulty operationalisation (continuous dynamics vs perturbation events)

**Decision question**  
How should tracking workload be manipulated across `LOW/MODERATE/HIGH`:
- by **continuous control dynamics parameters** (e.g., update rate, control attenuation) held constant within a block, or
- by **discrete perturbation / disturbance events** (and/or inclusion in a global event-budget / overlap policy)?

**Why this is open**  
Tracking is the only continuous-control subtask. Treating it differently may be correct, but it changes what “event-rate scaling” means across workload levels and may affect interpretability of workload manipulation (sustained control demand vs interruption load). The current parameter mapping is plausible but unvalidated for sensitivity, tolerability, and interaction with concurrent events from other subtasks.

**Current pilot default (implemented)**  
Parameter-based tracking difficulty set at block start (t=0) using a clamped difficulty scalar `d ∈ [0,1]`:
- `taskupdatetime_ms` decreases with difficulty (higher update bandwidth)
- `joystickforce` decreases with difficulty (increased control challenge)
No discrete tracking events are scheduled; tracking runs continuously for the full block.

**Options under consideration**
- **A: Keep parameter-based continuous dynamics** (current default).
- **B: Add discrete perturbation events** (e.g., occasional disturbances) while keeping baseline dynamics constant.
- **C: Hybrid** (mild dynamics scaling + perturbations at higher levels).
- **D: Include tracking in a global workload budget** (treat tracking “difficulty units” as part of an overall workload allocation model).

**Constraints**
- Tracking manipulation must remain deterministic and auditable per block.
- Must produce clear, monotonic difficulty ordering in at least one performance metric (e.g., deviation/RMSE) without excessive ceiling/floor effects.
- Must not require within-block label changes in v0 (unless OD-02 changes).

**What pilot evidence should resolve this (Pilot 1)**  
- **Sensitivity:** tracking performance (e.g., RMSE/mean deviation) separates by level (LOW < MOD < HIGH) without saturation.
- **Tolerability:** participants can maintain engagement; failure/abort/frustration not excessive at HIGH.
- **Interpretability:** tracking difficulty changes are understandable and justifyable as sustained control demand (vs interruption load).
- **Interaction:** the chosen tracking manipulation does not become meaningless due to concurrent overlaps from SYSMON/COMM/RESMAN events.

**Decision trigger**  
Lock after Pilot 1 once tracking difficulty shows reliable separation + tolerability under the full multitask context.

**Downstream impact if changed**  
Affects scenario generation parameters, tracking difficulty reporting, workload manipulation claims, and any adaptation policy that uses tracking difficulty as a control lever.

**References**  
- Pontiggia et al., 2024 (review of MATB(-II) workload manipulation practices; tracking often treated as dynamics-driven)
- (Add specific MATB-II tracking manipulation primary sources if needed)
### OD-10: Experience sampling during calibration (high-frequency subjective labels)

**Decision question**  
During the calibration phase, do we collect high-frequency subjective workload ratings (e.g., every ~10s), or keep calibration uninterrupted and rely on block-level labels (with optional post-block ratings)?

**Why this is open**  
Experience sampling could provide time-resolved subjective labels, but it introduces frequent interruptions that may alter workload, disrupt task performance, and contaminate the calibration data used for ML personalisation. This choice materially affects protocol validity and must be resolved before Pilot 1.

**Current pilot default**  
No within-block experience sampling. Calibration uses block-level workload labels, with optional brief post-block ratings (see OD-05).

**Options under consideration**
- **A: No experience sampling (uninterrupted calibration)** *(default)*  
- **B: Sparse sampling** (e.g., once per minute or fixed anchor points)  
- **C: High-frequency sampling** (e.g., every 10–20s)  
- **D: Event-triggered or adaptive sampling** (risk of circularity)

**Constraints**
- Sampling, if used, must be ultra-brief and logged deterministically.
- Sampling must not require within-block workload level changes (unless OD-02 changes).
- Calibration should remain representative of uninterrupted task performance.

**What evidence resolves this decision**

**Literature evidence**
- Experience sampling and EMA literature consistently show that frequent probes act as an intervention, increasing cognitive load and altering task engagement.
- Workload and neuroergonomics studies typically avoid high-frequency probing during continuous control tasks, instead using block-level or post-condition ratings to preserve task validity.
- Reviews of MATB(-II) protocols emphasise uninterrupted task execution during workload manipulation, with subjective measures collected between blocks rather than within blocks (e.g., Pontiggia et al., 2024).

**Conclusion from literature:**  
High-frequency experience sampling during calibration is likely to contaminate both performance and workload, undermining the purpose of calibration. Sparse or post-block ratings are preferred.

**Pilot evidence (if required)**
- Optional: verify that the absence of within-block sampling does not reduce interpretability of calibration data.
- No pilot testing of high-frequency sampling is required unless explicitly revisiting this decision.

**Decision trigger**  
Lock **before Pilot 1** based on literature review; do not include high-frequency experience sampling in Pilot 1 calibration.

**Downstream impact if changed**  
Would alter calibration protocol timing, task interruption profile, subjective label semantics, and ML personalisation validity.

**Status**  
Open → expected to close pre-Pilot 1 via literature.

### OD-11: Placement of MODERATE between LOW and HIGH (difficulty mapping shape)

**Decision question**  
When scaling difficulty across `LOW`, `MODERATE`, `HIGH`, should `MODERATE` be:
- **A: Midpoint** (approximately halfway between LOW and HIGH on the operational difficulty scale), or
- **B: Biased** (closer to LOW or closer to HIGH), and if biased, in which direction?

**Why this is open**  
The placement of `MODERATE` determines whether the three-level design captures a meaningful ordinal ladder or collapses into “easy vs hard with a fuzzy middle”. It affects:
- manipulation checks (subjective separation),
- performance ceiling/floor effects,
- tolerability,
- ML separability and calibration curve quality.

A midpoint is simple and symmetric, but may be too close to LOW (weak separation) or too close to HIGH (excess fatigue/overload) depending on task dynamics.

**Current pilot default**  
Linear midpoint mapping for any parameter scaled by a single difficulty scalar (i.e., `MODERATE` ≈ halfway between `LOW` and `HIGH`), unless task-specific caps/occupancy models distort the effective midpoint.

**Options under consideration**
- **A: Linear midpoint**  
  `MODERATE` set at 0.5 (or parameter midpoint).
- **B: Ease-in (MOD closer to LOW)**  
  Conservative moderate to reduce early overload and preserve tolerability.
- **C: Ease-out (MOD closer to HIGH)**  
  Aggressive moderate to ensure clear separation from LOW.
- **D: Task-specific moderate placement**  
  Allow different mappings per subtask (risk: interpretability drift).

**Constraints**
- `MODERATE` must remain ordinal and defensible across subtasks.
- Mapping must be auditable (explicit formula/lookup table), not “tuned by feel”.
- Must not introduce within-block level changes (unless OD-02 changes).

**What evidence resolves this decision**

**Literature evidence**
- Three-level designs are commonly used to avoid two-level ambiguity, but absolute placement is not standardised; reporting and manipulation checks are emphasised more than prescriptive midpoints.

**Pilot evidence (Pilot 1)**
- **Subjective separation:** LOW < MOD < HIGH reliably (not just LOW vs HIGH).
- **Performance sensitivity:** MOD should not be at ceiling (≈LOW) or floor (≈HIGH).
- **Tolerability:** MOD should not cause overload/frustration comparable to HIGH.
- **Event overlap sanity:** MOD should not create pathological concurrency patterns (if event-driven).

**Decision trigger**  
Lock after Pilot 1 once a mapping yields:
- ordered subjective ratings,
- non-saturated performance metrics,
- acceptable tolerability.

**Downstream impact if changed**  
Affects scenario parameterisation for all scaled subtasks, event-rate totals (OD-04), and any calibration/personalisation assumptions relying on separable levels.

**References**  
- (Add Pontiggia et al., 2024 and any primary sources if they explicitly discuss multi-level scaling practices.)
