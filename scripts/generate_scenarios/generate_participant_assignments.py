"""Generate participant assignments with counterbalanced sequence order.

#
# RETIRED — 2026-01: The SEQ counterbalancing system was removed when the
# calibration design changed from 3×5-min blocks (SEQ1/2/3) to 2×9-min
# continuous blocks ordered by a 6-template rank system baked into the
# scenario generator.  Assignments are now maintained manually in
# config/participant_assignments.yaml — see docs/PARTICIPANT_ASSIGNMENTS.md.
# This script is kept for historical reference only and should not be used.
#

Usage:
    python scripts/generate_participant_assignments.py --n-per-sequence 10
    python scripts/generate_participant_assignments.py --n-per-sequence 10 --start 1
    python scripts/generate_participant_assignments.py --participant-ids P001 P002 P003 --sequences SEQ1 SEQ2 SEQ3
"""

import argparse
import yaml
from pathlib import Path
import sys


def generate_counterbalanced_assignments(n_per_sequence: int, start_id: int = 1) -> dict:
    """Generate counterbalanced participant assignments.
    
    Args:
        n_per_sequence: Number of participants per sequence condition
        start_id: Starting participant number
        
    Returns:
        Dictionary of participant assignments
    """
    sequences = ["SEQ1", "SEQ2", "SEQ3"]
    assignments = {}
    
    participant_num = start_id
    for _ in range(n_per_sequence):
        for seq in sequences:
            pid = f"P{participant_num:03d}"
            assignments[pid] = {
                "sequence": seq,
                "sessions_completed": [],
                "last_run": None
            }
            participant_num += 1
    
    return assignments


def generate_custom_assignments(participant_ids: list, sequences: list) -> dict:
    """Generate assignments from specific participant IDs and sequences.
    
    Args:
        participant_ids: List of participant IDs
        sequences: List of sequence IDs (must match length of participant_ids)
        
    Returns:
        Dictionary of participant assignments
    """
    if len(participant_ids) != len(sequences):
        raise ValueError(f"Mismatch: {len(participant_ids)} participants but {len(sequences)} sequences")
    
    assignments = {}
    for pid, seq in zip(participant_ids, sequences):
        assignments[pid] = {
            "sequence": seq,
            "sessions_completed": [],
            "last_run": None
        }
    
    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Generate participant assignment file with counterbalanced sequences"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--n-per-sequence",
        type=int,
        help="Number of participants per sequence condition (generates N*3 participants)"
    )
    group.add_argument(
        "--participant-ids",
        nargs="+",
        help="Specific participant IDs to assign"
    )
    
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Sequence assignments (required with --participant-ids)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting participant number (default: 1)"
    )
    parser.add_argument(
        "--output",
        default="config/participant_assignments.yaml",
        help="Output file path (default: config/participant_assignments.yaml)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing assignments (preserves completed sessions)"
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.participant_ids and not args.sequences:
        print("ERROR: --sequences required when using --participant-ids", file=sys.stderr)
        return 1
    
    # Generate assignments
    if args.n_per_sequence:
        new_assignments = generate_counterbalanced_assignments(args.n_per_sequence, args.start)
        print(f"Generated {len(new_assignments)} participants ({args.n_per_sequence} per sequence)")
    else:
        new_assignments = generate_custom_assignments(args.participant_ids, args.sequences)
        print(f"Generated {len(new_assignments)} custom assignments")
    
    # Load existing file if merging
    output_path = Path(args.output)
    existing_data = {"participants": {}}
    
    if args.merge and output_path.exists():
        with open(output_path, 'r') as f:
            existing_data = yaml.safe_load(f) or {"participants": {}}
        print(f"Loaded {len(existing_data.get('participants', {}))} existing participants")
    
    # Merge: new assignments don't overwrite existing completed sessions
    for pid, assignment in new_assignments.items():
        if pid in existing_data["participants"]:
            # Preserve existing sessions and last_run
            assignment["sessions_completed"] = existing_data["participants"][pid].get("sessions_completed", [])
            assignment["last_run"] = existing_data["participants"][pid].get("last_run")
            print(f"  Preserved data for {pid}")
    
    existing_data["participants"].update(new_assignments)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(existing_data, f, default_flow_style=False, sort_keys=True)
    
    print(f"\nWrote {len(existing_data['participants'])} participants to {output_path}")
    
    # Show distribution
    seq_counts = {}
    for pid, data in existing_data["participants"].items():
        seq = data["sequence"]
        seq_counts[seq] = seq_counts.get(seq, 0) + 1
    
    print("\nSequence distribution:")
    for seq in sorted(seq_counts.keys()):
        print(f"  {seq}: {seq_counts[seq]} participants")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
