# run_logging_verification.md

This report verifies run manifest + primary event log compliance for the current codebase.

## Run manifest (JSON)

### Where it is written

- Manifest is written by `Logger._write_manifest()` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L47).
- It is written as JSON via `json.dump(...)` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L128).
- The manifest is written next to the session CSV: `self.path.with_suffix('.manifest.json')` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L51).
- Manifest writes are atomic: temp file + `os.replace(...)` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L124-L131).
- Missing commit hashes are treated as fatal (no silent fallback) in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L70-L79).

### Required fields

The manifest payload includes the following required keys:

- `repo_commit`: read from env `OPENMATB_REPO_COMMIT` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L70-L79). Exported by wrapper in [src/python/run_openmatb.py](../src/python/run_openmatb.py#L132-L133).
- `submodule_commit`: read from env `OPENMATB_SUBMODULE_COMMIT` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L71-L79). Exported by wrapper in [src/python/run_openmatb.py](../src/python/run_openmatb.py#L132-L133).
- `scenario_name`: derived from `CONFIG.get('Openmatb','scenario_path')` then `Path(...).stem` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L54-L60).
- `participant_id` and `session_id`: read from env in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L89-L90), and set by wrapper in [src/python/run_openmatb.py](../src/python/run_openmatb.py#L115-L116).
- `started_at_local`: alias of `created_at_local` written in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L84-L88).
- `ended_at_local`: initialized as `null` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L84-L88) and set on clean shutdown by `Logger.finalize()` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L134-L150).
- `lsl_enabled`: initialized `False` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L91), updated to `True` if LSL ever streams a row in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L255), and finalized in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L147-L150).
- `output_dir`: absolute output directory path in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L81-L83) (derived from `OUTPUT_BASE_DIR` in [src/python/vendor/openmatb/core/constants.py](../src/python/vendor/openmatb/core/constants.py#L60-L61)).
- `event_log_path`: absolute path to the CSV in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L81-L83).

## Primary event log (CSV)

### Where it is written

- Event log is a CSV written via `csv.DictWriter` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L8) with schema `logtime, scenario_time, type, module, address, value` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L20).
- CSV path is under external output dir `PATHS['SESSIONS']` in [src/python/vendor/openmatb/core/constants.py](../src/python/vendor/openmatb/core/constants.py#L63-L66) and used in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L36-L38).
- Manifest links to the event log via `event_log_path` in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L93).

### Timestamping and timebases

- Monotonic timebase: `time.perf_counter()` is written to `logtime` in record methods (e.g. [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L106)).
- Scenario-relative time: `scenario_time` is accumulated using pyglet `dt` in [src/python/vendor/openmatb/core/scheduler.py](../src/python/vendor/openmatb/core/scheduler.py#L81) and passed into the logger in [src/python/vendor/openmatb/core/scheduler.py](../src/python/vendor/openmatb/core/scheduler.py#L82).

## Failure modes

- If the process crashes, `ended_at_local` remains `null` because it is only set during `Logger.finalize()` on clean shutdown in [src/python/vendor/openmatb/core/logger.py](../src/python/vendor/openmatb/core/logger.py#L134-L150).

## Post-fix verification

Clean shutdown path calls `logger.finalize()` exactly once, immediately after logging the `end` marker, in [src/python/vendor/openmatb/core/scheduler.py](../src/python/vendor/openmatb/core/scheduler.py#L263-L264).
