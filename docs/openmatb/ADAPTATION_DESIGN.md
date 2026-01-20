# Closed-loop adaptation design (MWL → MATB difficulty)

Related docs:
- [OVERVIEW.md](OVERVIEW.md)
- [INSTRUMENTATION_POINTS.md](INSTRUMENTATION_POINTS.md)

Constraint: treat OpenMATB as upstream; do not edit its source to add adaptation.

## Design goals

- Keep adaptation logic *separate* from task rendering and task mechanics.
- Make every adaptation decision observable (logged with timestamps + params).
- Prevent oscillatory behavior and “surprising” task changes.

## Proposed architecture (clean separation)

### Two-process mental model

- **MWL estimator (external process)**
  - Reads EEG, computes workload (MWL) continuously.
  - Publishes a time-stamped MWL metric via LSL (or another IPC channel).

- **Task policy + actuator (in the OpenMATB loop, but not in upstream code)**
  - Runs alongside OpenMATB’s scheduler tick.
  - Consumes MWL estimates (latest value + uncertainty).
  - Computes a difficulty action.
  - Applies changes via the existing parameter-setting surface (`plugin.set_parameter`).

Why this split:
- Keeps EEG models modular and swappable.
- Avoids entangling neurophysiology code with UI/task code.

### Recommended scaffolding (in our repo, future work)

Implement these as *our* modules (not inside the submodule):

- `MwlEstimatorClient`
  - Reads an LSL stream (float MWL value + timestamp).
  - Maintains a short ring buffer.

- `MwlSmoother`
  - Exponential moving average (EMA) or median filter over a short window.

- `AdaptationPolicy`
  - Decides when to change difficulty.
  - Implements hysteresis, cooldown, and max step sizes.

- `ParameterActuator`
  - Applies changes by calling `plugin.set_parameter(key, value)`.
  - Emits an adaptation event log row (CSV/LSL) for traceability.

Integration strategy without upstream edits:
- Create a new *wrapper entry point* (in our code) that constructs the OpenMATB `Window` and a custom `Scheduler` subclass.
- In the overridden `update(dt)`, call `super().update(dt)` then run `policy.tick(...)`.

The hook point this relies on is stable and central:
- `Scheduler.update(dt)` in [src/python/vendor/openmatb/core/scheduler.py](../../src/python/vendor/openmatb/core/scheduler.py)

## Control policy recommendations (avoid spaghetti + oscillation)

### Baseline policy (robust default)

- **Smoothing**: compute a smoothed MWL $\hat{w}(t)$ (EMA with $\alpha \in [0.05, 0.2]$).
- **Dual-threshold hysteresis**:
  - increase difficulty when $\hat{w}(t) < \theta_{low}$ for $T_{hold}$
  - decrease difficulty when $\hat{w}(t) > \theta_{high}$ for $T_{hold}$
  - with $\theta_{low} < \theta_{high}$
- **Cooldown**: after any change, block further changes for $T_{cool}$ seconds.
- **Step-size limits**: limit parameter deltas per step (and per minute).

This prevents rapid toggling when MWL hovers near a threshold.

### Practical defaults (starting point)

- $T_{hold}$: 2–5 s (depending on estimator latency)
- $T_{cool}$: 10–30 s
- Max change rate: e.g., 1 difficulty step per 15–30 s

## What parameters are safest to adapt in MATB-style multitask settings

Principle: adapt *rate/pressure* variables more than *interface geometry* variables.

Safer (typically less perceptually jarring):
- Update rates / pacing:
  - Task-level `taskupdatetime` (ms)
  - Stimulus prompt frequency / event density via scenario-controlled triggers
- Time pressure variables:
  - System monitoring failure timeout (`alerttimeout`)
  - Communications response windows / overdue timers
- Stochastic load variables (carefully bounded):
  - Frequency of failures (sysmon), pump failures (resman), radio prompts (communications)

Riskier (can cause user surprise or confound motor demands):
- Big layout changes (`taskplacement`, showing/hiding tasks)
- Sudden changes in tracking dynamics (cursor/target behavior)
- Changing control mappings (keys/joystick axes)

Reference for many parameter names and semantics:
- [src/python/vendor/openmatb/parameters.csv](../../src/python/vendor/openmatb/parameters.csv)

## Risks and confounds

- **Oscillation**: MWL ↔ difficulty positive feedback loops; mitigated by hysteresis/cooldowns.
- **User surprise / strategy disruption**: abrupt changes can induce stress or re-orienting costs.
- **Non-stationarity**: fatigue, learning, and EEG drift change MWL calibration over time.
- **Confounds**:
  - stress/arousal vs workload
  - motion artifacts (especially with joystick tracking)
  - estimator latency may shift apparent cause/effect

Mitigations:
- Log every adaptation action with timestamps and parameter deltas.
- Include “adaptation disabled” control conditions.
- Consider per-subject calibration or adaptive baselining.

## Suggested evaluation metrics

Adaptation stability:
- Number of difficulty changes per minute
- Mean dwell time per difficulty level
- Oscillation index (e.g., fraction of changes that revert within $X$ seconds)

Task performance:
- Per-task performance metrics already produced by OpenMATB (`performance` rows)
- Composite performance (if using the `performance` plugin)

MWL-alignment:
- Correlation/lag between MWL estimates and task demands
- Predictive validity: does MWL predict upcoming errors/RT increases?

Human factors / usability:
- Self-report workload (e.g., NASA-TLX) vs adaptive policy behavior
- Surprise/annoyance ratings; perceived control
