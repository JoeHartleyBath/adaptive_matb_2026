import subprocess
import sys
import shutil
import time
import os
import csv
import re
from collections import defaultdict
from pathlib import Path

# Config
PYTHON = sys.executable
RUN_SCRIPT = str(Path("src/python/run_openmatb.py").absolute())
OUTPUT_ROOT = str(Path("results/temp_verification").absolute())

SEQUENCES = ["SEQ1", "SEQ2", "SEQ3"]
SPEED = 60

# Contract Expectations
COUNTS_BY_DIFFICULTY = {
    "LOW": {"sysmon": 5, "communications": 5, "resman": 5},
    "MODERATE": {"sysmon": 15, "communications": 15, "resman": 10},
    "HIGH": {"sysmon": 30, "communications": 30, "resman": 30}
}
COMM_MIN_SPACING = 8.0

def run_verification():
    # Clean up
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    
    overall_success = True

    for seq in SEQUENCES:
        print(f"\n=== Verifying {seq} ===")
        participant = f"P_VERIFY_{seq}"
        session = f"S_{seq}"
        
        cmd = [
            PYTHON, RUN_SCRIPT,
            "--output-root", OUTPUT_ROOT,
            "--participant", participant,
            "--session", session,
            "--seq-id", seq,
            "--verification",
            "--speed", str(SPEED)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print(f"(Expected runtime ~{45/SPEED:.1f} minutes)")
        
        start = time.time()
        try:
            subprocess.run(cmd, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(f" TIMEOUT for {seq}")
            overall_success = False
            continue
        except subprocess.CalledProcessError as e:
            print(f" FAILED with code {e.returncode}")
            overall_success = False
            continue
            
        elapsed = time.time() - start
        print(f" Finished in {elapsed:.1f}s")
        
        # Verify Log
        if not verify_output(seq, participant, session):
            overall_success = False
            
    if overall_success:
        print("\n ALL SEQUENCES PASS")
        sys.exit(0)
    else:
        print("\n ONE OR MORE FAILURES")
        sys.exit(1)

def verify_output(seq, participant, session):
    # Find CSV
    session_dir = Path(OUTPUT_ROOT) / "openmatb" / participant / session / "sessions"
    if not session_dir.exists():
         print(f" Session dir not found: {session_dir}")
         return False

    csv_files = list(session_dir.rglob("*.csv"))
    if not csv_files:
        print(f" No CSV produced for {seq}")
        return False
        
    csv_path = csv_files[0]
    print(f"   Analyzing {csv_path.name}...")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if not rows:
        print(f" CSV is empty")
        return False

    success = True
    
    # ---------------------------------------------------------
    # 1. Marker Syntax Check (Requirement 4)
    # ---------------------------------------------------------
    markers = [row for row in rows if row["type"] == "trigger" or (row["type"] == "event" and row["module"] == "labstreaminglayer" and row["address"] == "marker")]
    
    # Template: STUDY/V0/...|pid=<P>|sid=<S>|seq=<SEQ>
    marker_regex = re.compile(rf"STUDY/V0/.+\|pid={participant}\|sid={session}\|seq={seq}")
    
    malformed_markers = []
    for m in markers:
        val = m["value"]
        if "STUDY/V0/" in val:
            if not marker_regex.match(val):
                malformed_markers.append(val)
                
    if malformed_markers:
        print(f" {seq}: Malformed markers found (Requirement 4 check):")
        for mm in malformed_markers[:3]: print(f"    {mm}")
        if len(malformed_markers) > 3: print("    ...")
        success = False
    else:
        print(f" {seq}: Marker syntax valid")

    # ---------------------------------------------------------
    # 2. Block Content & Rate Analysis (Requirements 1, 2, 3)
    # ---------------------------------------------------------
    # Define expected blocks map
    expected_blocks = {}
    
    # Training
    expected_blocks["TRAINING/T1"] = "LOW"
    expected_blocks["TRAINING/T2"] = "MODERATE"
    expected_blocks["TRAINING/T3"] = "HIGH"
    
    # Retained (Sequence dependent mapping)
    seq_map = {
        "SEQ1": ["LOW", "MODERATE", "HIGH"],
        "SEQ2": ["MODERATE", "HIGH", "LOW"],
        "SEQ3": ["HIGH", "LOW", "MODERATE"]
    }
    mapping = seq_map.get(seq, [])
    for i, difficulty in enumerate(mapping):
        expected_blocks[f"RETAINED/B{i+1}"] = difficulty

    # Traverse rows and isolate blocks
    current_block_name = None
    block_start_time = 0.0
    
    block_events = defaultdict(list) # block_name -> list of event rows
    
    for row in rows:
        val = row["value"]
        st = float(row["scenario_time"])
        
        # Check start
        if "STUDY/V0/" in val and "/START" in val:
            parts = val.split("|")[0].split("/") # STUDY, V0, TRAINING, T1, START
            
            found_block = None
            for b_key in expected_blocks:
                if b_key in val:
                    found_block = b_key
                    break
            
            if found_block:
                current_block_name = found_block
                block_start_time = st
                continue
                
        # Check end
        if "STUDY/V0/" in val and "/END" in val and current_block_name:
            if current_block_name in val:
                current_block_name = None
                continue
        
        # Accumulate events if inside a block
        if current_block_name:
            block_events[current_block_name].append(row)

    # Now analyze each expected block
    for block_name, difficulty in expected_blocks.items():
        if block_name not in block_events:
            print(f" {seq}: Block {block_name} not found or no events recorded")
            success = False
            continue
            
        rows_in_block = block_events[block_name]
        
        c_sysmon = 0
        c_comm = 0
        c_resman = 0
        
        comm_times = []
        has_track = False
        
        # AOI / Visibility Flags
        aoi_seen = {"sysmon": False, "track": False, "communications": False, "resman": False}
        
        for r in rows_in_block:
            mod = r["module"]
            addr = r["address"]
            val = r["value"]
            rtype = r["type"]
            t_sec = float(r["scenario_time"])
            
            # AOI Check
            if rtype == "aoi" and mod in aoi_seen:
                aoi_seen[mod] = True

            if mod == "track":
                has_track = True

            # Sysmon: Count failure ONSETS (events)
            if rtype == "event" and mod == "sysmon" and "lights" in addr and "failure" in addr:
               if val == "1" or val == "True":
                   c_sysmon += 1
                   
            # Comm: Count prompt ONSETS (events)
            if rtype == "event" and mod == "communications" and "radioprompt" in addr:
                c_comm += 1
                comm_times.append(t_sec)
            
            # Resman: Count failure ONSETS (events)
            if rtype == "event" and mod == "resman" and "pump" in addr and "state" in addr and val == "failure":
                c_resman += 1

        # Check counts (Requirement 1)
        expected = COUNTS_BY_DIFFICULTY[difficulty]
        
        errs = []
        if c_sysmon != expected["sysmon"]: errs.append(f"Sysmon {c_sysmon}!={expected['sysmon']}")  
        if c_comm != expected["communications"]: errs.append(f"Comm {c_comm}!={expected['communications']}")
        if c_resman != expected["resman"]: errs.append(f"Resman {c_resman}!={expected['resman']}")  
        
        # Requirement 5: AOI Check
        for task in ["sysmon", "communications", "resman"]:
            if not aoi_seen[task]:
                errs.append(f"Missing AOI (visuals) for {task}")

        # Requirement 6: Resman Tank Reset Check
        # Search for first tank-a-level parameter after block start
        tank_a_level = None
        for r in rows_in_block:
             if r["type"] == "parameter" and r["module"] == "resman" and r["address"] == "tank-a-level":
                 tank_a_level = float(r["value"])
                 break # Found the initial level
        
        # Note: If no parameter log found, it might mean Resman didn't re-log parameters.
        # But with the fix, start() calls log_all_parameters()
        if tank_a_level is not None:
            if tank_a_level != 2500:
                 errs.append(f"Resman Tank A level not reset! Found {tank_a_level}, expected 2500")
        else:
             # Only strictly require it if we expect Resman to be running
             if expected["resman"] > 0 and len(rows_in_block) > 0:
                  errs.append("Resman Tank A level parameter not found at start of block")

        # Requirement 7: Comms Manual Mode Check
        # Ensure no rapid frequency changes follow prompts (which would indicate auto-solver is ON)
        # We look for "radio_frequency" state changes.
        freq_changes = [float(r["scenario_time"]) for r in rows_in_block 
                        if r["type"] == "state" and "radio_frequency" in r["address"]]
        
        if c_comm > 0 and len(freq_changes) > c_comm * 5: 
             # Heuristic: if we see lots of freq changes, auto-solver is likely on.
             # In manual/unattended(no-auto) mode, there should be ZERO freq changes.
             errs.append(f"Comms appears to be auto-solving ({len(freq_changes)} freq changes detected). It should be manual.")

        
        if errs:
            print(f"❌ {seq}: {block_name} ({difficulty}) errors: {', '.join(errs)}")
            success = False
        else:
            print(f" {seq}: {block_name} ({difficulty}) counts & visuals correct")

        # Check Comm Spacing (Requirement 2)
        if len(comm_times) > 1:
            violations = []
            for k in range(1, len(comm_times)):
                diff = comm_times[k] - comm_times[k-1]
                if diff < (COMM_MIN_SPACING - 0.2): # 0.2s buffer
                    violations.append(f"{diff:.2f}s at {comm_times[k]:.1f}")
            if violations:
                 print(f" {seq}: {block_name} comm spacing violation: {violations}")
                 success = False

        # Check Task Presence (Requirement 3)
        if not has_track:
             print(f" {seq}: {block_name} missing track events")
             success = False

    return success

if __name__ == "__main__":
    run_verification()
