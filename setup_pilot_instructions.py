import os
from pathlib import Path

def create_instruction_files(base_path):
    """Creates the HTML instruction files for the familiarization phase."""
    instructions_dir = base_path / "instructions" / "pilot_en"
    instructions_dir.mkdir(parents=True, exist_ok=True)

    instructions = {
        "1_welcome.txt": """
<h1>Welcome to the Adaptive MATB Pilot</h1>
<center><p>This study involves performing multiple tasks simultaneously.</p></center>
<p>We will introduce each sub-task one by one so you can get familiar with them.</p>
<p>Press <b>SPACE</b> or click <b>NEXT</b> to continue.</p>
""",
        "2_sysmon.txt": """
<h1>Task 1: System Monitoring</h1>
<p>Watch the lights and scales at the top left.</p>
<ul>
    <li><b>Lights:</b> If a light turns OFF (green) or RED, click it or press F1-F6 to reset it.</li>
    <li><b>Scales:</b> If a pointer moves out of the center zone, click the scale or press F1-F6 to reset it.</li>
</ul>
<p>Press <b>SPACE</b> to try this task.</p>
""",
        "3_track.txt": """
<h1>Task 2: Tracking</h1>
<p>Use the joystick to keep the target circle tracking the green circle in the center.</p>
<ul>
    <li>Ideally, keep the target inside the center box.</li>
    <li>Move the joystick smoothly.</li>
</ul>
<p>Press <b>SPACE</b> to try this task.</p>
""",
        "4_comm.txt": """
<h1>Task 3: Communications</h1>
<p>Listen for your callsign: <b>NASA-504</b>.</p>
<ul>
    <li>When you hear "NASA-504", listen for the command (e.g., "Change frequency to 123.4").</li>
    <li>Set the frequency using the arrow keys and press ENTER.</li>
    <li>If you hear a different callsign, ignore it.</li>
</ul>
<p>Press <b>SPACE</b> to try this task.</p>
""",
        "5_resman.txt": """
<h1>Task 4: Resource Management</h1>
<p>Keep the fuel level in the main tanks (A & B) around 2500 units.</p>
<ul>
    <li>Turn pumps ON/OFF (Keys 1-8) to transfer fuel from reserve tanks.</li>
    <li>Don't let any tank run dry or overflow.</li>
</ul>
<p>Press <b>SPACE</b> to try this task.</p>
""",
        "6_all_tasks.txt": """
<h1>Putting it All Together</h1>
<p>Now all tasks will run simultaneously for a short practice period.</p>
<p>Do your best to attend to all of them.</p>
<p>Press <b>SPACE</b> to begin the full practice.</p>
"""
    }

    for filename, content in instructions.items():
        file_path = instructions_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
    
    print(f"Created {len(instructions)} instruction files in {instructions_dir}")

def create_scenario_file(base_path):
    """Creates the pilot_familiarisation.txt scenario file."""
    # Base path is now src/python/scenarios
    output_dir = base_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Note: OpenMATB timing is 0:00:00. This scenario is event-driven by the "instructions" plugin mostly,
    # but we use time to sequence the "demos".
    # Path to instructions relative to vendor/openmatb/includes/instructions
    # vendor/openmatb/includes/instructions -> (4 levels up) -> src/python -> assets/instructions
    inst_rel = "../../../../assets/instructions/pilot_en"
    
    scenario_content = f"""# Pilot Familiarization Scenario
# Introduces tasks one by one
# Format: TIME;EVENT;ACTION;ARGUMENTS

# --- Initialization ---
0:00:00;sysmon;start
0:00:00;sysmon;hide
0:00:00;track;start
0:00:00;track;hide
0:00:00;communications;start
0:00:00;communications;hide
0:00:00;resman;start
0:00:00;resman;hide

# --- Step 1: Welcome ---
0:00:01;instructions;filename;{inst_rel}/1_welcome.txt
0:00:01;instructions;start

# --- Step 2: SysMon Demo ---
# Wait for user to dismiss instructions, then show Sysmon
0:00:02;instructions;filename;{inst_rel}/2_sysmon.txt
0:00:02;instructions;start
0:00:02;sysmon;show
# Let them practice sysmon for 30 seconds alone
# (Technically instructions pauses, so 'resume' happens when they click Next)

# --- Step 3: Tracking Demo ---
0:00:03;sysmon;hide
0:00:03;instructions;filename;{inst_rel}/3_track.txt
0:00:03;instructions;start
0:00:03;track;show

# --- Step 4: Comm Demo ---
0:00:04;track;hide
0:00:04;instructions;filename;{inst_rel}/4_comm.txt
0:00:04;instructions;start
0:00:04;communications;show

# --- Step 5: ResMan Demo ---
0:00:05;communications;hide
0:00:05;instructions;filename;{inst_rel}/5_resman.txt
0:00:05;instructions;start
0:00:05;resman;show

# --- Step 6: All Together ---
0:00:06;resman;hide
0:00:06;instructions;filename;{inst_rel}/6_all_tasks.txt
0:00:06;instructions;start

# Show all for 2 minutes of practice
0:00:06;sysmon;show
0:00:06;track;show
0:00:06;communications;show
0:00:06;resman;show

# --- End ---
0:02:30;sysmon;stop
0:02:30;track;stop
0:02:30;communications;stop
0:02:30;resman;stop
"""
    
    output_path = output_dir / "pilot_familiarisation.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(scenario_content)
    
    print(f"Created scenario file at {output_path}")

if __name__ == "__main__":
    # Script root: c:\phd_projects\adaptive_matb_2026
    project_root = Path(__file__).resolve().parent
    
    # New assets location: src/python/assets/instructions
    assets_dir = project_root / "src" / "python" / "assets"
    
    # New scenario location: src/python/scenarios
    scenarios_dir = project_root / "src" / "python" / "scenarios"
    
    create_instruction_files(assets_dir)
    create_scenario_file(scenarios_dir)
