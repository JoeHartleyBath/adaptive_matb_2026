import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run the pilot study in fast-forward (dry run) mode.")
    parser.add_argument("--seq", choices=["SEQ1", "SEQ2", "SEQ3"], required=True, help="Sequence ID (SEQ1: L-M-H, SEQ2: M-H-L, SEQ3: H-L-M)")
    parser.add_argument("--pid", required=True, help="Participant ID")
    parser.add_argument("--sid", required=True, help="Session ID")
    parser.add_argument("--speed", type=int, default=60, help="Speed multiplier (default: 60x)")
    parser.add_argument("--output", default="results/fast_forward_runs", help="Output root directory")
    
    args = parser.parse_args()
    
    # Path setup
    repo_root = Path(__file__).resolve().parents[2]
    run_script = repo_root / "src" / "python" / "run_openmatb.py"
    output_abs = Path(args.output).resolve()
    
    print(f"--- PILOT FAST FORWARD RUN ---")
    print(f"Participant: {args.pid}")
    print(f"Session:     {args.sid}")
    print(f"Sequence:    {args.seq}")
    print(f"Speed:       {args.speed}x")
    print(f"Output:      {output_abs}")
    print(f"------------------------------")
    
    cmd = [
        sys.executable, str(run_script),
        "--participant", args.pid,
        "--session", args.sid,
        "--seq-id", args.seq,
        "--output-root", str(output_abs),
        "--speed", str(args.speed)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nExecution failed with code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    
    print(f"\nFast forward run completed successfully.")
    print(f"Logs available in: {output_abs}")

if __name__ == "__main__":
    main()
