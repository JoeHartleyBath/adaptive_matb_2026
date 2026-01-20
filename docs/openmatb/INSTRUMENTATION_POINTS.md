# OpenMATB — instrumentation points (events, LSL, MWL callback)

Related docs:
- [OVERVIEW.md](OVERVIEW.md)
- [SETUP_AND_RUN.md](SETUP_AND_RUN.md)
- [ADAPTATION_DESIGN.md](ADAPTATION_DESIGN.md)

Goal: identify the *best* hook points for timestamped event markers, LSL/file streaming, and injecting a real-time MWL metric callback — while treating OpenMATB as upstream/third-party.

## Current logging + timing model (what we get “for free”)

- Monotonic timestamps: `time.perf_counter()` is recorded as `logtime`.
- Scenario time: `scenario_time` is a float in seconds (advanced by pyglet `dt`).
- Event types already logged:
  - `event` (scenario events)
  - `input` (keyboard)
  - `state` (widgets/values)
  - `parameter` (initial parameter snapshot)
  - `performance` (task-specific performance metrics)

Primary source: [src/python/vendor/openmatb/core/logger.py](../../src/python/vendor/openmatb/core/logger.py)

## Ranked hook points (best → acceptable)

### 1) Central log write path (best for unified event stream)

Why it’s best:
- One place sees *all* logs (events, inputs, state, performance).
- Already supports optional LSL streaming of the same rows.
- Cleanest place to add wall-clock timestamps and/or schema normalization (in our wrapper, without editing upstream).

Where:
- `core.logger.Logger.write_row_queue()` and `Logger.write_single_slot()` in [src/python/vendor/openmatb/core/logger.py](../../src/python/vendor/openmatb/core/logger.py)

Essential excerpt (≤10 lines):
```python
row_dict = self.round_row(this_row)._asdict()
self.writer.writerow(row_dict)
if self.lsl is not None:
    self.lsl.push(';'.join([str(r) for r in row_dict.values()]))
```

Variables available:
- `row_dict` with fields: `logtime`, `scenario_time`, `type`, `module`, `address`, `value`
- `self.datetime` (session start wall-clock) is also available in the Logger instance

Recommended use:
- For file logging: leave as-is; parse CSV downstream.
- For LSL streaming: use the built-in LabStreamingLayer plugin to send either marker strings or full CSV rows.
- For future MWL integration: in our non-upstream wrapper, wrap/patch `Logger.write_single_slot` to also emit a separate “normalized” event packet (e.g., JSON) to our own sink.

### 2) Scenario event execution (best for protocol markers)

Why it’s good:
- Every scenario command funnels through one method.
- Ideal for markers like `task_start`, `task_stop`, `parameter_change`, `instruction_start`, etc.

Where:
- `Scheduler.execute_one_event(event)` in [src/python/vendor/openmatb/core/scheduler.py](../../src/python/vendor/openmatb/core/scheduler.py)

Essential excerpt (≤10 lines):
```python
plugin = self.plugins[event.plugin]
if len(event.command) == 1:
    getattr(plugin, event.command[0])()
elif len(event.command) == 2:
    getattr(plugin, 'set_parameter')(event.command[0], event.command[1])
logger.record_event(event)
```

Variables available:
- `event.plugin` (alias), `event.command` (method or parameter/value)
- `self.scenario_time` at time of execution
- `plugin` instance (so you can read `plugin.parameters`)

Recommended markers:
- `scenario_event` with `{plugin, command, scenario_time, logtime}`
- `task_state_change` derived from `start/stop/show/hide/pause/resume`

### 3) Per-plugin parameter updates (best for adaptation actuation auditing)

Why it’s useful:
- This is where difficulty updates land.
- Lets you log the “control surface” precisely.

Where:
- `AbstractPlugin.set_parameter(keys_str, value)` in [src/python/vendor/openmatb/plugins/abstractplugin.py](../../src/python/vendor/openmatb/plugins/abstractplugin.py)

Essential excerpt (≤10 lines):
```python
keys_list = keys_str.split('-')
dic = self.parameters
for key in keys_list[:-1]:
    dic = dic.setdefault(key, {})
dic[keys_list[-1]] = value
```

Variables available:
- `keys_str` (e.g., `taskupdatetime`, `scales-1-failure`)
- `value` (already validated by `Scenario.check_events`)
- `self.parameters` (full current task config)

Recommended use:
- For adaptation: treat `set_parameter` as the only supported “actuator API” to change difficulty online.

### 4) Inputs (keyboard + joystick) (best for response markers)

Where:
- Keyboard logging at the window layer:
  - `Window.on_key_press` / `Window.on_key_release` in [src/python/vendor/openmatb/core/window.py](../../src/python/vendor/openmatb/core/window.py)
- Task-specific responses (e.g., sysmon): plugin overrides `do_on_key` and logs performance
  - Example: [src/python/vendor/openmatb/plugins/sysmon.py](../../src/python/vendor/openmatb/plugins/sysmon.py)

Variables available:
- `keystr`, `state`, current `scenario_time`, plus task context inside the plugin

### 5) LSL marker outlet (good when you want explicit event markers)

Where:
- `Labstreaminglayer.update` and `Labstreaminglayer.push` in [src/python/vendor/openmatb/plugins/labstreaminglayer.py](../../src/python/vendor/openmatb/plugins/labstreaminglayer.py)

Key behavior:
- If `streamsession=True`, the logger pushes full CSV rows over LSL.
- If `marker` is set to a non-empty string, it pushes that string as a marker sample.

Variables available:
- Marker text is entirely caller-defined; best to send structured strings.

### 6) Hardware trigger (parallel port) (good for legacy EEG trigger boxes)

Where:
- `Parallelport.set_trigger_value` in [src/python/vendor/openmatb/plugins/parallelport.py](../../src/python/vendor/openmatb/plugins/parallelport.py)

Note:
- This requires `pyparallel` + hardware access; treat as optional.

## Recommended event schema (for EEG synchronization)

OpenMATB’s native CSV row format is:
- `logtime`, `scenario_time`, `type`, `module`, `address`, `value`

For EEG work, prefer a normalized schema (regardless of transport: CSV, LSL, or IPC):

```text
event_name: string
session_id: string/int
trial_id: string/int (optional; scenario-defined)
condition: string (optional; scenario-defined)

# timing
ts_monotonic_s: float   # perf_counter
ts_wall_iso: string     # datetime.now().isoformat(timespec='milliseconds')
scenario_time_s: float

# origin
module: string          # plugin alias or 'keyboard'
address: string         # parameter name / input key / widget metric
value: any              # scalar or JSON string

# task/adaptation context
state: object           # task state snapshot (optional)
difficulty: object      # difficulty parameters (optional)
response: object        # RT/accuracy metrics (optional)
```

Mapping guidance:
- `ts_monotonic_s` ← OpenMATB `logtime`
- `scenario_time_s` ← OpenMATB `scenario_time`
- `event_name` can be derived from `type` + `address` + `value` (or explicitly sent via marker plugin)

## What makes MWL injection tricky (so we design around it)

- OpenMATB does not include an LSL *inlet* for receiving MWL values; it is primarily an outlet/logger.
- The update loop is frame-driven (`pyglet` dt); adaptations must be robust to timing jitter.
- Scenario event times are integer seconds (scenario file format), so sub-second protocol changes need either:
  - more frequent scenario events (still quantized), or
  - a runtime control channel (recommended; see [ADAPTATION_DESIGN.md](ADAPTATION_DESIGN.md)).
