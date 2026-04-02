"""generate_pilot_scenarios.py

⚠  DO NOT RE-RUN without explicit intent.

The committed scenario files under scenarios/ are the locked source of truth
for Pilot 1. This generator uses unseeded randomness — regenerating would
silently change event schedules and invalidate any prior runs.

Only re-run if deliberately regenerating scenarios (e.g. after a protocol
change), and immediately re-run the static verifier afterwards:
    python src/python/verification/verify_pilot_scenarios.py
"""
import sys
import os
from pathlib import Path
from random import randint, shuffle, random
import math

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "experiment" / "scenarios"
BLOCK_DURATION_SEC = 300

# Targeted Difficulty Config (0.0 - 1.0) to approximate 5/15/30 events
DIFFICULTIES = {
    "LOW": 0.2,
    "MODERATE": 0.55,
    "HIGH": 0.95
}

# --------------------------------------------------------------------------------
# VENDOR ALGORITHMS & MOCKS
# Copied/Adapted from src/python/vendor/openmatb/scenario_generator.py
# --------------------------------------------------------------------------------

class Event:
    def __init__(self, line, timestamp, plugin, command):
        self.line = line
        self.timestamp = timestamp
        self.plugin = plugin
        self.command = command
    
    @property
    def line_str(self):
        # Format: H:MM:SS;plugin;command;[arg]
        hours = int(self.timestamp // 3600)
        minutes = int((self.timestamp % 3600) // 60)
        seconds = int(self.timestamp % 60)
        t_str = f"{hours}:{minutes:02}:{seconds:02}"
        
        cmd_str = self.command
        if isinstance(self.command, list):
            cmd_str = ';'.join(map(str, self.command))
        elif isinstance(self.command, str):
            cmd_str = self.command
            
        return f"{t_str};{self.plugin};{cmd_str}"

# Standard Mock Plugins to avoid importing Pyglet/OpenMATB core
class MockPlugin:
    def __init__(self, params):
        self.parameters = params

plugins = {
    'track': MockPlugin({}),
    'sysmon': MockPlugin({
        'alerttimeout': 10000,
        'lights': {'1': {}, '2': {}},
        # Match vendor SysMon default: 4 scales (F1-F4) and 2 lights (F5-F6).
        'scales': {'1': {}, '2': {}, '3': {}, '4': {}}
    }),
    'communications': MockPlugin({}),
    'resman': MockPlugin({
        'pump': {
            # Match vendor resman defaults (see src/python/vendor/openmatb/plugins/resman.py).
            # The vendor scenario generator uses these pump flows to derive a *solvable* target-tank leakage.
            '1': {'flow': 800},
            '2': {'flow': 600},
            '3': {'flow': 800},
            '4': {'flow': 600},
            '5': {'flow': 600},
            '6': {'flow': 600},
            '7': {'flow': 400},
            '8': {'flow': 400},
        },
        'tank': {'A': {'target': 2500}, 'B': {'target': 2500}}
    })
}

# Copied Constants
EVENTS_REFRACTORY_DURATION = 1
# The vendor generator uses an average of ~13s, but in practice our comms audio
# can run longer; using a more conservative duration prevents overlapping prompts.
AVERAGE_AUDITORY_PROMPT_DURATION = 18
COMMUNICATIONS_TARGET_RATIO = 0.50
STEP_DURATION_SEC = BLOCK_DURATION_SEC # Override global for the vendor functions

# Guard against events firing at the very start/end of a block.
# This applies to distributed *demand* events (failures/prompts), not to task start/stop.
EVENT_EDGE_BUFFER_SEC = 5

# ResMan: we keep the vendor leak scaling logic, but apply a small offset to
# MODERATE/HIGH to make the task "hard but not impossible" without changing the
# overall block difficulty used for other tasks.
RESMAN_LEAK_OFFSET_THRESHOLD_DIFFICULTY = 0.50
RESMAN_LEAK_OFFSET_MODERATE_HIGH = -100
RESMAN_LEAK_OFFSET_HIGH_THRESHOLD_DIFFICULTY = 0.90
RESMAN_LEAK_OFFSET_HIGH_EXTRA = -100

# ResMan: vendor example scenarios recover a pump failure by explicitly
# scheduling `pump-*-state;off` 10s after `failure`.
RESMAN_PUMP_FAILURE_DURATION_SEC = 10

# Copied Functions
def part_duration_sec(duration_sec, part_left, duration_list=list()):
    """Vendor algorithm for splitting duration into parts."""
    MIN_PART_DURATION_SEC = 0
    if duration_sec == 0:
        return duration_list
    part_left = max(2, part_left)
    allowed_max_duration = int(duration_sec/(part_left-1))
    # Fix for cases where duration is tight
    if allowed_max_duration < MIN_PART_DURATION_SEC: allowed_max_duration = MIN_PART_DURATION_SEC
    n = randint(MIN_PART_DURATION_SEC, allowed_max_duration)
    return part_duration_sec(duration_sec - n, part_left-1, duration_list + [n])

def get_part_durations(duration_sec, part_number):
    try:
        # Retry loop for tight fits
        for _ in range(100):
            parts = part_duration_sec(duration_sec, part_number, [])
            if len(parts) == part_number:
                shuffle(parts)
                return parts
    except RecursionError:
        pass
    # Fallback
    return [int(duration_sec/part_number)] * part_number

def reduce(p, q):
    if p == q: return 1, 1
    x = max(p, q); y = min(p, q);
    while True:
        x %= y
        if x == 0: break
        if x < y: temp = x; x = y; y = temp;
    return int(p/y), int(q/y)

def choices(l, k, randomize):
    wl = list(l)
    shuffle(wl)
    nl = list()
    while len(nl) < k:
        if len(wl) == 0:
            wl = list(l)
            shuffle(wl)
        nl.append(wl.pop())
    if randomize == True:
        shuffle(nl)
    return nl


def choices_balanced_by_weight(items, weights, k: int, randomize: bool):
    """Return a length-k list by allocating counts ~proportional to weights.

    This is used to reduce variance in "impact" (e.g., ResMan pump failures),
    compared with naive uniform sampling.

    Implementation: largest-remainder allocation.
    - expected_i = w_i / sum(w) * k
    - count_i = floor(expected_i)
    - distribute remaining counts to largest fractional parts
    - shuffle the resulting list to randomize order
    """

    if k <= 0:
        return []
    if len(items) != len(weights):
        raise ValueError("items and weights must be same length")

    pairs = []
    for item, w in zip(items, weights):
        w = float(w)
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"invalid weight for {item}: {w}")
        pairs.append((item, w))

    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        # Fallback: behave like the existing choices() (uniform with repeats)
        return choices(items, k, randomize)

    expected = [(item, (w / total_w) * k) for item, w in pairs]
    counts = {item: int(math.floor(e)) for item, e in expected}
    remaining = int(k - sum(counts.values()))

    # Largest remainder distribution with randomized tie-breaking
    remainders = [(item, (e - math.floor(e))) for item, e in expected]
    shuffle(remainders)
    remainders.sort(key=lambda x: x[1], reverse=True)

    for item, _ in remainders[:remaining]:
        counts[item] += 1

    out = []
    for item, _ in pairs:
        out.extend([item] * counts[item])

    if randomize:
        shuffle(out)
    return out

def get_events_from_scenario(scenario_lines):
    return [l for l in scenario_lines if isinstance(l, Event)]

def get_task_current_state(scenario_lines, plugin):
    scenario_events = get_events_from_scenario(scenario_lines)
    task_events = [e for e in scenario_events if e.plugin==plugin]
    if len(task_events) == 0:
        return None
    cmd_keep_list = ['start', 'pause', 'resume']
    # Handle list commands (e.g. ['start']) vs str commands 'start'
    task_cmd_events = []
    for e in task_events:
        c = e.command if isinstance(e.command, list) else [e.command]
        if c[0] in cmd_keep_list:
            task_cmd_events.append(e)
            
    if len(task_cmd_events) > 0:
        return task_cmd_events[-1].command
    else:
        return None

def distribute_events(
    scenario_lines,
    start_sec,
    single_duration,
    cmd_list,
    plugin_name,
    start_buffer_sec: float = EVENT_EDGE_BUFFER_SEC,
    end_buffer_sec: float = EVENT_EDGE_BUFFER_SEC,
):
    # Constrain distributed events to occur within the buffered window.
    # Policy: ensure the *full event duration* fits before the end buffer.
    buffered_window_sec = float(STEP_DURATION_SEC) - float(start_buffer_sec) - float(end_buffer_sec)
    if buffered_window_sec < 0:
        buffered_window_sec = 0.0

    # If too many events to fit, trim to the maximum that can fit.
    # This should be rare, but keeps the invariant deterministic.
    if single_duration and single_duration > 0:
        max_events_fit = int(math.floor(buffered_window_sec / float(single_duration)))
        if max_events_fit < 0:
            max_events_fit = 0
        if len(cmd_list) > max_events_fit:
            cmd_list = list(cmd_list)[:max_events_fit]

    total_event_duration = len(cmd_list) * single_duration
    rest_sec = buffered_window_sec - total_event_duration
    
    # Cap if overloaded
    if rest_sec < 0:
        rest_sec = 0

    n = len(cmd_list) + 1
    random_delays = get_part_durations(rest_sec, n) if n > 1 else [rest_sec]
    random_delays = random_delays[:-1]

    onset_sec = float(start_sec) + float(start_buffer_sec)
    lastline = 0
    if len(scenario_lines) > 0 and isinstance(scenario_lines[-1], Event):
        lastline = scenario_lines[-1].line

    for previous_delay, cmd in zip(random_delays, cmd_list):
        lastline += 1
        onset_sec += previous_delay
        scenario_lines.append(Event(lastline, onset_sec, plugin_name, cmd))
        onset_sec += single_duration
    return scenario_lines


def distribute_resman_pump_failures(scenario_lines, start_sec, pump_ids):
    """Distribute resman pump failures across the block and schedule recovery.

    Vendor policy (see vendor includes/scenarios/basic.txt):
      - Set `pump-*-state` to `failure`
      - Exactly 10 seconds later, set `pump-*-state` back to `off`

    We also apply the standard refractory duration between failures.
    """

    failure_duration_sec = RESMAN_PUMP_FAILURE_DURATION_SEC
    single_event_duration = failure_duration_sec + EVENTS_REFRACTORY_DURATION

    buffered_window_sec = float(STEP_DURATION_SEC) - float(EVENT_EDGE_BUFFER_SEC) - float(EVENT_EDGE_BUFFER_SEC)
    if buffered_window_sec < 0:
        buffered_window_sec = 0.0

    if single_event_duration and single_event_duration > 0:
        max_events_fit = int(math.floor(buffered_window_sec / float(single_event_duration)))
        if max_events_fit < 0:
            max_events_fit = 0
        if len(pump_ids) > max_events_fit:
            pump_ids = list(pump_ids)[:max_events_fit]

    total_event_duration = len(pump_ids) * single_event_duration
    rest_sec = buffered_window_sec - total_event_duration
    if rest_sec < 0:
        rest_sec = 0

    n = len(pump_ids) + 1
    random_delays = get_part_durations(rest_sec, n) if n > 1 else [rest_sec]
    random_delays = random_delays[:-1]

    onset_sec = float(start_sec) + float(EVENT_EDGE_BUFFER_SEC)
    lastline = 0
    if len(scenario_lines) > 0 and isinstance(scenario_lines[-1], Event):
        lastline = scenario_lines[-1].line

    for previous_delay, pid in zip(random_delays, pump_ids):
        onset_sec += previous_delay

        lastline += 1
        scenario_lines.append(Event(lastline, onset_sec, 'resman', [f'pump-{pid}-state', 'failure']))

        lastline += 1
        scenario_lines.append(
            Event(lastline, onset_sec + failure_duration_sec, 'resman', [f'pump-{pid}-state', 'off'])
        )

        onset_sec += single_event_duration

    return scenario_lines

def add_scenario_phase(scenario_lines, task_difficulty_tuples, start_sec):
    # Modified slightly to remove 'resume' logic since we always start fresh
    start_line = 1
    
    # Start tasks
    for (plugin_name, difficulty) in task_difficulty_tuples:
        plugin = plugins[plugin_name]

        # Tracking difficulty: manipulate control-dynamics parameters
        # (1) taskupdatetime (ms): lower => higher update rate/bandwidth (harder)
        # (2) joystickforce (gain): lower => more attenuation (harder)
        # Keep targetproportion at OpenMATB defaults by not setting it here.
        if plugin_name == 'track':
            # Linear mapping from difficulty in [0,1] to parameter ranges.
            # Easy (0.0): slower updates, higher gain
            # Hard (1.0): faster updates, lower gain
            difficulty = max(0.0, min(1.0, float(difficulty)))

            taskupdatetime_easy_ms = 50
            taskupdatetime_hard_ms = 10
            taskupdatetime_ms = int(round(
                taskupdatetime_easy_ms
                + (taskupdatetime_hard_ms - taskupdatetime_easy_ms) * difficulty
            ))
            if taskupdatetime_ms < 1:
                taskupdatetime_ms = 1

            joystickforce_easy = 3
            joystickforce_hard = 1
            joystickforce = int(round(
                joystickforce_easy
                + (joystickforce_hard - joystickforce_easy) * difficulty
            ))
            if joystickforce < 1:
                joystickforce = 1

            # Ensure params are applied before start (even at identical timestamps)
            scenario_lines.append(Event(start_line, start_sec, plugin_name, ['taskupdatetime', taskupdatetime_ms]))
            scenario_lines.append(Event(start_line, start_sec, plugin_name, ['joystickforce', joystickforce]))
            scenario_lines.append(Event(start_line, start_sec, plugin_name, 'start'))
            continue

        # Default: start the task first, then schedule its events
        scenario_lines.append(Event(start_line, start_sec, plugin_name, 'start'))

        if plugin_name == 'sysmon':
            failure_duration_sec = plugin.parameters['alerttimeout'] / 1000 + EVENTS_REFRACTORY_DURATION
            single_failure_ratio = (failure_duration_sec) / (STEP_DURATION_SEC)
            max_events_N = int(1/single_failure_ratio)
            difficulty_events_N = int((difficulty)/single_failure_ratio)
            events_N = min(max_events_N, difficulty_events_N)

            # Lights
            light_names = [k for k,v in plugin.parameters['lights'].items()]
            light_list = choices(light_names, events_N, True)
            cmd_list = [[f'lights-{l}-failure', True] for l in light_list]
            scenario_lines = distribute_events(scenario_lines, start_sec, failure_duration_sec, cmd_list, plugin_name)

            # Scales
            scale_names = [k for k,v in plugin.parameters['scales'].items()]
            scale_list = choices(scale_names, events_N, True)
            cmd_list = [[f'scales-{s}-failure', True] for s in scale_list]
            scenario_lines = distribute_events(scenario_lines, start_sec, failure_duration_sec, cmd_list, plugin_name)

        elif plugin_name == 'communications':
            averaged_duration_sec = AVERAGE_AUDITORY_PROMPT_DURATION
            single_duration_sec = averaged_duration_sec + EVENTS_REFRACTORY_DURATION
            communication_ratio = difficulty
            single_event_ratio = single_duration_sec/STEP_DURATION_SEC
            max_event_num = int(STEP_DURATION_SEC / single_duration_sec)
            current_event_num = int(communication_ratio/single_event_ratio)
            event_num = min(max_event_num, current_event_num)

            n,d = reduce(COMMUNICATIONS_TARGET_RATIO*100,100)
            promptlist = ['own']*n+['other']*(d-n)
            
            if (event_num % d) == 0 and event_num>1:
                prompt_list = choices(promptlist, event_num, True)
                p = prompt_list.count('own') / len(prompt_list)
                while p != n/d:
                    prompt_list = choices(promptlist, event_num, True)
                    p = prompt_list.count('own') / len(prompt_list)
            else:
                prompt_list = choices(promptlist, event_num, True)

            cmd_list = [['radioprompt', p] for p in prompt_list]
            scenario_lines = distribute_events(scenario_lines, start_sec, single_duration_sec, cmd_list, plugin_name)

        elif plugin_name == 'resman':
            # Vendor Logic: Tank Loss
            pumps = plugin.parameters['pump']
            infinite_capacity = sum([p['flow'] for k,p in pumps.items() if k in ['2', '4']])
            finite_capacity = sum([p['flow'] for k,p in pumps.items() if k in ['5', '6']])
            total_capacity = infinite_capacity + finite_capacity
            maximum_single_leakage = int(total_capacity / 2)

            target_tank_letters = [k for k,t in plugin.parameters['tank'].items() if 'target' in t]
            for letter in target_tank_letters:
                # cmd = [f'tank-{letter}-lossperminute', int(maximum_single_leakage * difficulty)]
                # Warning: 'maximum_single_leakage * difficulty' might result in leaks > inputs at high difficulty.
                # Vendor code does exactly this.
                # OpenMATB parameter keys are case-sensitive (expects tank-a-..., tank-b-...)
                leak_per_min = int(maximum_single_leakage * difficulty)
                if difficulty >= RESMAN_LEAK_OFFSET_THRESHOLD_DIFFICULTY:
                    leak_per_min = max(0, leak_per_min + RESMAN_LEAK_OFFSET_MODERATE_HIGH)
                if difficulty >= RESMAN_LEAK_OFFSET_HIGH_THRESHOLD_DIFFICULTY:
                    leak_per_min = max(0, leak_per_min + RESMAN_LEAK_OFFSET_HIGH_EXTRA)
                cmd = [f"tank-{letter.lower()}-lossperminute", leak_per_min]
                scenario_lines.append(Event(start_line, start_sec, plugin_name, cmd))

            # ADDED: Pump Failure Generation
            # Goal: 6-fold increase from Low (0.2) to High (0.95).
            # Formula: events = 13.33 * diff - 0.66
            # Low (0.2) -> ~2 events
            # High (0.95) -> ~12 events
            # Moderate (0.55) -> ~7 events

            num_failures = int(13.33 * difficulty - 0.66)
            if num_failures < 0: num_failures = 0

            if num_failures > 0:
                pump_ids = list(pumps.keys())

                # Impact-balanced pump failures:
                # failing a pump removes its flow entirely (0 during failure), so higher-flow
                # pumps are more disruptive. We therefore allocate failures ~proportional to
                # 1/flow to reduce block-to-block variance in effective difficulty.
                pump_weights = []
                for pid in pump_ids:
                    flow = float(pumps[pid].get('flow', 0) or 0)
                    pump_weights.append(1.0 / flow if flow > 0 else 0.0)

                chosen_pumps = choices_balanced_by_weight(pump_ids, pump_weights, num_failures, True)
                scenario_lines = distribute_resman_pump_failures(scenario_lines, start_sec, chosen_pumps)

    return scenario_lines

# --------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------

def write_scenario(filename, lines, marker_name, include_tlx=False):
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by time
    events = get_events_from_scenario(lines)
    events.sort(key=lambda x: x.timestamp)
    
    with open(path, 'w') as f:
        f.write(f"# OpenMATB Scenario: {filename}\n")
        f.write("# Auto-generated Clone (Vendor Logic)\n\n")
        
        # Metadata tokens (supported by pilot run_openmatb.py wrapper)
        pid_tok = "${OPENMATB_PARTICIPANT}"
        sid_tok = "${OPENMATB_SESSION}"
        
        # Start marker: Format STUDY/V0/<MARKER>|pid=...|sid=...
        payload = f"pid={pid_tok}|sid={sid_tok}"
        # Careful: marker_name comes in without leading STUDY/V0/
        marker_str = f"STUDY/V0/{marker_name}/START|{payload}"
        
        f.write(f"0:00:00;labstreaminglayer;start\n")
        f.write(f"0:00:00;labstreaminglayer;marker;{marker_str}\n")
        f.write(f"0:00:00;scheduling;start\n")
        f.write(f"0:00:00;communications;voiceidiom;english\n")
        f.write(f"0:00:00;communications;voicegender;male\n")
        
        for e in events:
            f.write(e.line_str + '\n')
            
        # Stop All
        end_time = BLOCK_DURATION_SEC
        end_h = int(end_time//3600)
        end_m = int((end_time%3600)//60)
        end_s = int(end_time%60)
        
        # Stop task plugins
        for plugin in ['sysmon', 'track', 'communications', 'resman', 'scheduling']:
            f.write(f"{end_h}:{end_m:02}:{end_s:02};{plugin};stop\n")
        
        # End marker for the Block
        end_marker_str = f"STUDY/V0/{marker_name}/END|{payload}"
        f.write(f"{end_h}:{end_m:02}:{end_s:02};labstreaminglayer;marker;{end_marker_str}\n")
        
        if include_tlx:
            # TLX Injection: This plugin blocks execution until the user submits.
            # We derive the marker name from the block name.
            tlx_marker_name = f"TLX/{marker_name.replace('calibration/', 'calibration_')}"
            tlx_start_marker = f"STUDY/V0/{tlx_marker_name}/START|{payload}"
            
            f.write(f"\n# --- NASA-TLX (Blocking) ---\n")
            f.write(f"{end_h}:{end_m:02}:{end_s:02};labstreaminglayer;marker;{tlx_start_marker}\n")
            # Vendor genericscales plugin expects 'filename' (no create/load methods)
            f.write(f"{end_h}:{end_m:02}:{end_s:02};genericscales;filename;nasatlx_en.txt\n")
            f.write(f"{end_h}:{end_m:02}:{end_s:02};genericscales;start\n")
            
            # These lines execute AFTER the user submits the questionnaire
            tlx_end_marker = f"STUDY/V0/{tlx_marker_name}/END|{payload}"
            f.write(f"{end_h}:{end_m:02}:{end_s:02};labstreaminglayer;marker;{tlx_end_marker}\n")

        # Stop LSL last (keeps recording during TLX if present)
        f.write(f"{end_h}:{end_m:02}:{end_s:02};labstreaminglayer;stop\n")


def main():
    print("Generating Pilot Scenarios using Vendor Logic...")

    levels = ["LOW", "MODERATE", "HIGH"]
    practice_filenames = {
        "LOW": "pilot_practice_low.txt",
        "MODERATE": "pilot_practice_moderate.txt",
        "HIGH": "pilot_practice_high.txt",
    }
    calibration_filenames = {
        "LOW": "pilot_calibration_low.txt",
        "MODERATE": "pilot_calibration_moderate.txt",
        "HIGH": "pilot_calibration_high.txt",
    }
    
    # Generate Training Blocks (T1, T2, T3 fixed order: Low, Mod, High)
    for i, level in enumerate(levels):
        scen_lines = []
        difficulty = DIFFICULTIES[level]
        phase = (('track', difficulty),('sysmon', difficulty),
                 ('communications', difficulty),('resman', difficulty))
        
        scen_lines = add_scenario_phase(scen_lines, phase, 0)
        marker_name = f"TRAINING/T{i+1}"
        write_scenario(practice_filenames[level], scen_lines, marker_name, include_tlx=False)
        print(f"Generated Training {level} (Diff {difficulty})")

    # Generate calibration Blocks (Generic LOW, MODERATE, HIGH)
    for level in levels:
        scen_lines = []
        difficulty = DIFFICULTIES[level]
        phase = (('track', difficulty),('sysmon', difficulty),
                 ('communications', difficulty),('resman', difficulty))
        
        scen_lines = add_scenario_phase(scen_lines, phase, 0)
        marker_name = f"calibration/{level}"
        write_scenario(calibration_filenames[level], scen_lines, marker_name, include_tlx=True)
        print(f"Generated calibration {level} (Diff {difficulty})")


if __name__ == "__main__":
    main()
