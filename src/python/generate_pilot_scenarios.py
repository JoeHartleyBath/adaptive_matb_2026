import sys
import os
from pathlib import Path
from random import randint, shuffle, random

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
OUTPUT_DIR = Path('src/python/scenarios')
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
        'scales': {'1': {}, '2': {}}
    }),
    'communications': MockPlugin({}),
    'resman': MockPlugin({
        'pump': {
            '1': {'flow': 1000}, '2': {'flow': 1000}, '3': {'flow': 1000}, '4': {'flow': 1000},
            '5': {'flow': 600}, '6': {'flow': 600}, '7': {'flow': 600}, '8': {'flow': 600}
        },
        'tank': {'A': {'target': 2500}, 'B': {'target': 2500}}
    })
}

# Copied Constants
EVENTS_REFRACTORY_DURATION = 1
AVERAGE_AUDITORY_PROMPT_DURATION = 13
COMMUNICATIONS_TARGET_RATIO = 0.50
STEP_DURATION_SEC = BLOCK_DURATION_SEC # Override global for the vendor functions

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

def distribute_events(scenario_lines, start_sec, single_duration, cmd_list, plugin_name):
    total_event_duration = len(cmd_list) * single_duration
    rest_sec = STEP_DURATION_SEC - total_event_duration
    
    # Cap if overloaded
    if rest_sec < 0:
        rest_sec = 0

    n = len(cmd_list) + 1
    random_delays = get_part_durations(rest_sec, n) if n > 1 else [rest_sec]
    random_delays = random_delays[:-1]

    onset_sec = start_sec
    lastline = 0
    if len(scenario_lines) > 0 and isinstance(scenario_lines[-1], Event):
        lastline = scenario_lines[-1].line

    for previous_delay, cmd in zip(random_delays, cmd_list):
        lastline += 1
        onset_sec += previous_delay
        scenario_lines.append(Event(lastline, onset_sec, plugin_name, cmd))
        onset_sec += single_duration
    return scenario_lines

def add_scenario_phase(scenario_lines, task_difficulty_tuples, start_sec):
    # Modified slightly to remove 'resume' logic since we always start fresh
    start_line = 1
    
    # Start tasks
    for (plugin_name, difficulty) in task_difficulty_tuples:
        scenario_lines.append(Event(start_line, start_sec, plugin_name, 'start'))
        plugin = plugins[plugin_name]

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

        elif plugin_name == 'track':
            scenario_lines.append(Event(start_line, start_sec, plugin_name, ['targetproportion', 1 - difficulty]))

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
                cmd = [f'tank-{letter}-lossperminute', int(maximum_single_leakage * difficulty)]
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
                # Pick pumps randomly
                chosen_pumps = choices(pump_ids, num_failures, True)
                cmd_list = [[f'pump-{pid}-state', 'failure'] for pid in chosen_pumps]
                
                # Distribute with approximate 15s spacing/buffer
                scenario_lines = distribute_events(scenario_lines, start_sec, 15, cmd_list, plugin_name)

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
        seq_tok = "${OPENMATB_SEQ_ID}"
        
        # Start marker: Format STUDY/V0/<MARKER>|pid=...|sid=...|seq=...
        payload = f"pid={pid_tok}|sid={sid_tok}|seq={seq_tok}"
        # Careful: marker_name comes in without leading STUDY/V0/
        marker_str = f"STUDY/V0/{marker_name}/START|{payload}"
        
        f.write(f"0:00:00;labstreaminglayer;start\n")
        f.write(f"0:00:00;labstreaminglayer;marker;{marker_str}\n")
        f.write(f"0:00:00;scheduling;start\n")
        
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
            tlx_marker_name = f"TLX/{marker_name.replace('RETAINED/', 'RETAINED_')}"
            tlx_start_marker = f"STUDY/V0/{tlx_marker_name}/START|{payload}"
            
            f.write(f"\n# --- NASA-TLX (Blocking) ---\n")
            f.write(f"{end_h}:{end_m:02}:{end_s:02};labstreaminglayer;marker;{tlx_start_marker}\n")
            f.write(f"{end_h}:{end_m:02}:{end_s:02};genericscales;create\n")
            f.write(f"{end_h}:{end_m:02}:{end_s:02};genericscales;load;nasatlx_en.txt\n")
            f.write(f"{end_h}:{end_m:02}:{end_s:02};genericscales;start\n")
            
            # These lines execute AFTER the user submits the questionnaire
            tlx_end_marker = f"STUDY/V0/{tlx_marker_name}/END|{payload}"
            f.write(f"{end_h}:{end_m:02}:{end_s:02};labstreaminglayer;marker;{tlx_end_marker}\n")

        # Stop LSL last (keeps recording during TLX if present)
        f.write(f"{end_h}:{end_m:02}:{end_s:02};labstreaminglayer;stop\n")


def main():
    print("Generating Pilot Scenarios using Vendor Logic...")

    levels = ["LOW", "MODERATE", "HIGH"]
    
    # Generate Training Blocks (T1, T2, T3 fixed order: Low, Mod, High)
    for i, level in enumerate(levels):
        scen_lines = []
        difficulty = DIFFICULTIES[level]
        phase = (('track', difficulty),('sysmon', difficulty),
                 ('communications', difficulty),('resman', difficulty))
        
        scen_lines = add_scenario_phase(scen_lines, phase, 0)
        marker_name = f"TRAINING/T{i+1}"
        write_scenario(f"pilot_training_T{i+1}_{level}.txt", scen_lines, marker_name, include_tlx=False)
        print(f"Generated Training {level} (Diff {difficulty})")

    # Generate Retained Blocks (Generic LOW, MODERATE, HIGH)
    for level in levels:
        scen_lines = []
        difficulty = DIFFICULTIES[level]
        phase = (('track', difficulty),('sysmon', difficulty),
                 ('communications', difficulty),('resman', difficulty))
        
        scen_lines = add_scenario_phase(scen_lines, phase, 0)
        marker_name = f"RETAINED/{level}"
        write_scenario(f"pilot_retained_{level}.txt", scen_lines, marker_name, include_tlx=True)
        print(f"Generated Retained {level} (Diff {difficulty})")


if __name__ == "__main__":
    main()
