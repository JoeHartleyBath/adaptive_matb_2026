# OpenMATB (MATB-style battery) — overview

Status: OpenMATB is vendored as a third-party submodule at `src/python/vendor/openmatb/`.

Related docs:
- [SETUP_AND_RUN.md](SETUP_AND_RUN.md)
- [INSTRUMENTATION_POINTS.md](INSTRUMENTATION_POINTS.md)
- [ADAPTATION_DESIGN.md](ADAPTATION_DESIGN.md)
- [LICENSING.md](LICENSING.md)
- Repo data boundary rules: [docs/DATA_MANAGEMENT.md](../DATA_MANAGEMENT.md)

## What OpenMATB is

OpenMATB is an open-source re-implementation of the Multi-Attribute Task Battery (MATB) paradigm: multiple concurrent tasks (monitoring, tracking, communications, resource management) plus an optional scheduling timeline.

Upstream readme: [src/python/vendor/openmatb/README.md](../../src/python/vendor/openmatb/README.md)

## Where the code lives (high-level)

- Entry point: [src/python/vendor/openmatb/main.py](../../src/python/vendor/openmatb/main.py)
- Core runtime (clock, scenario parsing, scheduler, logging, window): [src/python/vendor/openmatb/core/](../../src/python/vendor/openmatb/core/)
- Tasks/plugins (sysmon, track, communications, resman, scheduling, …): [src/python/vendor/openmatb/plugins/](../../src/python/vendor/openmatb/plugins/)
- Scenario + assets (text scenarios, instruction pages, images, sounds): [src/python/vendor/openmatb/includes/](../../src/python/vendor/openmatb/includes/)

## Main entry points

- Normal run: `python main.py`
  - Creates the main window, then runs `Scheduler()`.
- Replay run: `python main.py -r [session_id]`
  - Runs `ReplayScheduler()` (replays a recorded session CSV).

See:
- [src/python/vendor/openmatb/main.py](../../src/python/vendor/openmatb/main.py)
- [src/python/vendor/openmatb/core/scheduler.py](../../src/python/vendor/openmatb/core/scheduler.py)
- [src/python/vendor/openmatb/core/replayscheduler.py](../../src/python/vendor/openmatb/core/replayscheduler.py)

## Experiment session flow (mental model)

### Narrative flow

1. `main.py` loads locale (from `config.ini`), then constructs the GUI window.
2. `Scheduler` initializes logging, constructs a custom `Clock`, loads a `Scenario`, and starts the pyglet event loop.
3. The `Scenario` parser reads a scenario text file and converts each line to an `Event(time, plugin, command...)`.
4. The `Scenario` instantiates all referenced plugins (tasks).
5. Each update tick:
   - scenario time advances
   - joystick state is polled
   - each active plugin updates at its own `taskupdatetime`
   - scheduled scenario `Event`s are executed (start/stop, show/hide, set parameters)
   - the logger writes events/inputs/states/performance to CSV (and optionally LSL)
6. When all plugins stop and no events remain, the program exits.

### “Mental model” diagram (modules + data flow)

```
   config.ini                 includes/scenarios/*.txt
      |                                 |
      v                                 v
  core.utils.get_conf_value()     core.scenario.Scenario
      |                                 |
      v                                 v
   core.window.Window  <---->  plugins/* (tasks)
      |                                 |
      v                                 v
       core.scheduler.Scheduler.update(dt)
        |   |        |         |
        |   |        |         +--> execute_one_event(Event) -> plugin.start/stop/set_parameter
        |   |        +--> plugin.update(scenario_time) -> state/performance logs
        |   +--> window + joystick inputs -> logger.record_input(...)
        +--> timekeeping: scenario_time + perf_counter timestamps
                     |
                     v
               core.logger.Logger  -> CSV sessions/*  (+ optional LSL streaming)
```

## How tasks are represented

Tasks are implemented as “plugins”, all inheriting from `AbstractPlugin`:
- Base lifecycle: `start()` → `update(scenario_time)` → `stop()`
- Each plugin has:
  - a `parameters` dict (task difficulty/behavior)
  - a `taskupdatetime` (ms) controlling update rate
  - key handlers (keyboard/joystick), plus optional automation

See base class: [src/python/vendor/openmatb/plugins/abstractplugin.py](../../src/python/vendor/openmatb/plugins/abstractplugin.py)

## Key configuration points (where task parameters live)

OpenMATB has three “layers” of configuration:

1) Global runtime config (`config.ini`)
- Location: [src/python/vendor/openmatb/config.ini](../../src/python/vendor/openmatb/config.ini)
- Controls: locale, screen, fullscreen, scenario selection, AOI highlighting, layout bounds.

2) Scenario file (protocol + parameter changes over time)
- Location: [src/python/vendor/openmatb/includes/scenarios/](../../src/python/vendor/openmatb/includes/scenarios/)
- Syntax: `HH:MM:SS;plugin_alias;command` or `HH:MM:SS;plugin_alias;parameter;value`
- Example: [src/python/vendor/openmatb/includes/scenarios/default.txt](../../src/python/vendor/openmatb/includes/scenarios/default.txt)

3) Plugin parameter dictionaries (defaults + validation)
- Each plugin defines defaults in code (its `__init__`) and adds a `validation_dict`.
- Reference table of many parameters: [src/python/vendor/openmatb/parameters.csv](../../src/python/vendor/openmatb/parameters.csv)

Practical guidance for adaptation work:
- Treat the scenario file as the “protocol timeline” (when tasks start/stop and when difficulty changes).
- Treat plugin parameters as the “control surface” (what we can safely adjust online).
- Treat the logger as the “ground truth event stream” for EEG synchronization.
