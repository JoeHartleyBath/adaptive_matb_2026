r"""Launch OpenMATB with repo-safe output paths.

This wrapper enforces that participant/session identifiers are provided and configures
OpenMATB to write session logs outside the git repo.

Usage (PowerShell):
  cd src/python/vendor/openmatb
        .\.venv\Scripts\Activate.ps1
    python ..\..\run_openmatb.py --participant P001 --session S001 --seq-id SEQ1

Environment variables (optional):
  OPENMATB_OUTPUT_ROOT   (default: C:\data\adaptive_matb)
  OPENMATB_PARTICIPANT / OPENMATB_PARTICIPANT_ID
  OPENMATB_SESSION / OPENMATB_SESSION_ID
    OPENMATB_SEQ_ID

OpenMATB uses:
  OPENMATB_OUTPUT_ROOT and OPENMATB_OUTPUT_SUBDIR
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import json
import os
import re
import signal
import subprocess
import shutil
import sys
import time
import yaml
from pathlib import Path
from typing import Optional

# Suppress liblsl's verbose C++ logging (must be set before pylsl is imported)
os.environ.setdefault("LSL_LOGLEVEL", "0")

# ---------------------------------------------------------------------------
# EDA subprocess management (global state for crash-safe cleanup)
# ---------------------------------------------------------------------------
_eda_subprocess: Optional[subprocess.Popen] = None


def _cleanup_eda_subprocess() -> None:
    """Terminate EDA subprocess if running. Called via atexit or signal handlers."""
    global _eda_subprocess
    if _eda_subprocess is not None and _eda_subprocess.poll() is None:
        print("Terminating EDA streamer subprocess...", file=sys.stderr)
        try:
            _eda_subprocess.terminate()
            _eda_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("EDA streamer did not terminate, killing...", file=sys.stderr)
            _eda_subprocess.kill()
            _eda_subprocess.wait()
        except Exception as e:
            print(f"Error cleaning up EDA subprocess: {e}", file=sys.stderr)
        _eda_subprocess = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM by cleaning up EDA subprocess then re-raising."""
    _cleanup_eda_subprocess()
    # Re-raise the signal to allow normal exit behavior
    sys.exit(128 + signum)


# Register cleanup handlers
atexit.register(_cleanup_eda_subprocess)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _start_eda_streamer(
    repo_root: Path,
    eda_port: str,
    eda_stream_name: str = "ShimmerEDA",
    health_check_timeout: float = 15.0,
) -> dict:
    """Start EDA streamer subprocess and verify LSL stream appears.
    
    Returns:
        dict with keys:
            - started: bool
            - pid: int or None
            - stream_name: str
            - stream_type: str
            - health_check_passed: bool
            - error: str or None
    """
    global _eda_subprocess
    
    result = {
        "started": False,
        "pid": None,
        "stream_name": eda_stream_name,
        "stream_type": "EDA",
        "health_check_passed": False,
        "error": None,
    }
    
    # Build command to run EDA streamer
    streamer_script = repo_root / "scripts" / "stream_shimmer_eda.py"
    if not streamer_script.exists():
        result["error"] = f"EDA streamer script not found: {streamer_script}"
        return result
    
    cmd = [
        sys.executable,
        str(streamer_script),
        "--port", eda_port,
        "--name", eda_stream_name,
    ]
    
    print(f"Starting EDA streamer: {' '.join(cmd)}")
    
    try:
        _eda_subprocess = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        result["started"] = True
        result["pid"] = _eda_subprocess.pid
        print(f"EDA streamer started (PID: {_eda_subprocess.pid})")
    except Exception as e:
        result["error"] = f"Failed to start EDA streamer: {e}"
        return result
    
    # Wait a moment for the subprocess to initialize
    time.sleep(2.0)
    
    # Check if subprocess crashed immediately
    if _eda_subprocess.poll() is not None:
        stderr_output = _eda_subprocess.stderr.read() if _eda_subprocess.stderr else ""
        result["error"] = f"EDA streamer exited immediately (code {_eda_subprocess.returncode}): {stderr_output}"
        result["started"] = False
        _eda_subprocess = None
        return result
    
    # Health check: resolve LSL stream
    print(f"Waiting for LSL stream '{eda_stream_name}' (timeout: {health_check_timeout}s)...")
    
    try:
        import pylsl
        
        start_time = time.time()
        streams = []
        
        while time.time() - start_time < health_check_timeout:
            # Look for streams with matching name
            streams = pylsl.resolve_byprop("name", eda_stream_name, timeout=2.0)
            if streams:
                break
            
            # Check if subprocess is still running
            if _eda_subprocess.poll() is not None:
                stderr_output = _eda_subprocess.stderr.read() if _eda_subprocess.stderr else ""
                result["error"] = f"EDA streamer crashed during health check: {stderr_output}"
                result["started"] = False
                _eda_subprocess = None
                return result
        
        if streams:
            result["health_check_passed"] = True
            print(f"✓ LSL stream '{eda_stream_name}' found ({len(streams)} stream(s))")
        else:
            result["error"] = f"LSL stream '{eda_stream_name}' not found within {health_check_timeout}s"
            
    except ImportError:
        result["error"] = "pylsl not installed; cannot verify EDA stream"
    except Exception as e:
        result["error"] = f"LSL health check failed: {e}"
    
    return result


def _stop_eda_streamer() -> None:
    """Stop EDA streamer subprocess."""
    _cleanup_eda_subprocess()


def _preflight_stream_check(
    expected_streams: list[dict],
    timeout_per_stream: float = 5.0,
    sample_test_duration: float = 2.0,
) -> dict:
    """Check for expected LSL streams before session starts.
    
    Args:
        expected_streams: List of dicts with 'name' and/or 'type' keys to search for.
                         e.g., [{"name": "ShimmerEDA", "type": "EDA"}, {"type": "EEG"}]
        timeout_per_stream: Seconds to wait for each stream.
        sample_test_duration: Seconds to collect samples for rate/quality estimation.
    
    Returns:
        dict with:
            - all_found: bool
            - all_streaming: bool
            - streams: list of dicts with name, type, found, info, sample_stats
            - warnings: list of warning messages
    """
    try:
        import pylsl
    except ImportError:
        print("WARNING: pylsl not installed, skipping stream check", file=sys.stderr)
        return {"all_found": False, "all_streaming": False, "streams": [], "warnings": [], "error": "pylsl not installed"}
    
    results = {
        "all_found": True,
        "all_streaming": True,
        "streams": [],
        "warnings": [],
        "error": None,
    }
    
    # Helper to suppress liblsl C++ logging during resolve calls
    def _resolve_quiet(prop: str, value: str, timeout: float):
        """Resolve LSL streams while suppressing C library logging."""
        old_stderr_fd = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 2)
                result = pylsl.resolve_byprop(prop, value, timeout=timeout)
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
        return result
    
    def _test_sample_flow(stream_info, duration: float) -> dict:
        """Pull samples from a stream to verify data flow and estimate quality."""
        import numpy as np
        
        stats = {
            "samples_received": 0,
            "measured_rate_hz": 0.0,
            "rate_error_pct": 0.0,
            "signal_range": None,
            "signal_std": None,
            "flat_signal": False,
            "time_correction_ms": 0.0,
            "streaming": False,
            "warnings": [],
        }
        
        try:
            inlet = pylsl.StreamInlet(stream_info, max_buflen=int(duration * 2))
            inlet.open_stream(timeout=2.0)
            
            # Get time correction
            try:
                tc = inlet.time_correction(timeout=1.0)
                stats["time_correction_ms"] = tc * 1000
                if abs(tc) > 0.1:  # >100ms clock difference
                    stats["warnings"].append(f"Large clock offset: {tc*1000:.1f}ms")
            except Exception:
                stats["warnings"].append("Could not get time correction")
            
            # Collect samples
            samples = []
            timestamps = []
            start_time = time.time()
            
            while time.time() - start_time < duration:
                sample, ts = inlet.pull_sample(timeout=0.1)
                if sample is not None:
                    samples.append(sample)
                    timestamps.append(ts)
            
            inlet.close_stream()
            
            n_samples = len(samples)
            stats["samples_received"] = n_samples
            
            if n_samples > 1:
                stats["streaming"] = True
                
                # Calculate actual sample rate
                elapsed = timestamps[-1] - timestamps[0]
                if elapsed > 0:
                    stats["measured_rate_hz"] = (n_samples - 1) / elapsed
                
                # Compare to nominal rate
                nominal = stream_info.nominal_srate()
                if nominal > 0 and stats["measured_rate_hz"] > 0:
                    stats["rate_error_pct"] = abs(stats["measured_rate_hz"] - nominal) / nominal * 100
                    if stats["rate_error_pct"] > 10:
                        stats["warnings"].append(
                            f"Sample rate mismatch: {stats['measured_rate_hz']:.1f}Hz vs nominal {nominal:.1f}Hz"
                        )
                
                # Signal quality checks (first channel)
                values = np.array([s[0] for s in samples])
                stats["signal_range"] = float(np.max(values) - np.min(values))
                stats["signal_std"] = float(np.std(values))
                
                # Check for flat/stuck signal
                if stats["signal_std"] < 1e-9:
                    stats["flat_signal"] = True
                    stats["warnings"].append("FLAT SIGNAL: No variation detected (sensor issue?)")
                
                # Check for NaN/Inf values
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    stats["warnings"].append("Invalid values detected (NaN/Inf)")
                
                # EDA-specific checks
                if stream_info.type() == "EDA":
                    mean_val = float(np.mean(values))
                    if mean_val < 0.01:
                        stats["warnings"].append(f"EDA very low ({mean_val:.4f} uS) - check electrode contact")
                    elif mean_val > 50:
                        stats["warnings"].append(f"EDA unusually high ({mean_val:.1f} uS) - check for artifacts")
                
                # EEG-specific checks
                if stream_info.type() == "EEG":
                    # Check for clipping or saturation
                    if stats["signal_range"] < 1:
                        stats["warnings"].append("EEG range very small - check impedances")
                    
            elif n_samples == 0:
                stats["warnings"].append("NO SAMPLES RECEIVED - stream may be stalled")
            else:
                stats["streaming"] = True  # Got at least 1 sample
                stats["warnings"].append("Too few samples for quality analysis")
                
        except Exception as e:
            stats["warnings"].append(f"Sample test failed: {e}")
        
        return stats
    
    for expected in expected_streams:
        stream_name = expected.get("name")
        stream_type = expected.get("type")
        label = stream_name or stream_type or "unknown"
        
        stream_result = {
            "name": stream_name,
            "type": stream_type,
            "found": False,
            "info": None,
            "sample_stats": None,
        }
        
        # Search by name first, then by type
        if stream_name:
            print(f"  Searching for stream '{stream_name}'...", end="", flush=True)
            streams = _resolve_quiet("name", stream_name, timeout_per_stream)
        elif stream_type:
            print(f"  Searching for stream type '{stream_type}'...", end="", flush=True)
            streams = _resolve_quiet("type", stream_type, timeout_per_stream)
        else:
            print(f"  Skipping stream with no name or type")
            continue
        
        if streams:
            stream_result["found"] = True
            info = streams[0]
            stream_result["info"] = {
                "name": info.name(),
                "type": info.type(),
                "channel_count": info.channel_count(),
                "nominal_srate": info.nominal_srate(),
                "source_id": info.source_id(),
            }
            print(f" Found: {info.name()}")
            
            # Test sample flow
            print(f"    Testing data flow ({sample_test_duration}s)...", end="", flush=True)
            sample_stats = _test_sample_flow(info, sample_test_duration)
            stream_result["sample_stats"] = sample_stats
            
            if sample_stats["streaming"]:
                print(f" {sample_stats['samples_received']} samples @ {sample_stats['measured_rate_hz']:.1f}Hz")
            else:
                print(f" NO SAMPLES!")
                results["all_streaming"] = False
            
            # Collect warnings
            for warning in sample_stats.get("warnings", []):
                results["warnings"].append(f"{label}: {warning}")
        else:
            stream_result["found"] = False
            results["all_found"] = False
            results["all_streaming"] = False
            print(f" Not found")
        
        results["streams"].append(stream_result)
    
    return results


def _check_labrecorder() -> dict:
    """Check if LabRecorder is running.
    
    Note: We can only detect if the process is running, not if it's actively recording.
    User must verify recording status manually.
    """
    result = {
        "running": False,
        "process_name": None,
        "warning": None,
        "recording_note": "Cannot verify recording status programmatically - check LabRecorder UI!",
    }
    
    try:
        # Check for LabRecorder process (Windows)
        if sys.platform == "win32":
            output = subprocess.check_output(
                ["tasklist", "/FI", "IMAGENAME eq LabRecorder.exe", "/FO", "CSV"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if "LabRecorder.exe" in output:
                result["running"] = True
                result["process_name"] = "LabRecorder.exe"
        else:
            # Unix: check with pgrep
            try:
                subprocess.check_call(
                    ["pgrep", "-x", "LabRecorder"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                result["running"] = True
                result["process_name"] = "LabRecorder"
            except subprocess.CalledProcessError:
                pass
    except Exception as e:
        result["warning"] = f"Could not check for LabRecorder: {e}"
    
    return result


def _run_preflight_checks(
    check_eeg: bool = True,
    check_eda: bool = True,
    eeg_stream_type: str = "EEG",
    eda_stream_name: str = "ShimmerEDA",
    timeout: float = 5.0,
) -> bool:
    """Run pre-flight LSL stream checks and prompt user to continue.
    
    Includes a retry loop so users can fix issues and re-check without restarting.
    
    Returns True if user confirms to proceed, False otherwise.
    """
    
    def _run_single_check():
        """Run one round of preflight checks. Returns (all_ok, has_warnings, lr_status, result)."""
        print("\n" + "=" * 60)
        print("PRE-FLIGHT STREAM CHECK")
        print("=" * 60)
        
        # Check LabRecorder first
        print("\n[1/2] Checking LabRecorder...")
        lr_status = _check_labrecorder()
        if lr_status["running"]:
            print(f"  LabRecorder is running")
        else:
            print(f"  LabRecorder NOT DETECTED")
            print(f"    Start LabRecorder before proceeding to ensure data is saved!")
        
        # Check streams
        expected_streams = []
        if check_eeg:
            expected_streams.append({"type": eeg_stream_type, "name": None})
        if check_eda:
            expected_streams.append({"name": eda_stream_name, "type": "EDA"})
        
        result = {"streams": [], "warnings": [], "all_found": True, "all_streaming": True}
        
        if not expected_streams:
            print("\n[2/2] No streams to check.")
        else:
            print(f"\n[2/2] Checking {len(expected_streams)} stream(s)...\n")
            result = _preflight_stream_check(
                expected_streams,
                timeout_per_stream=timeout,
                sample_test_duration=2.0,
            )
        
        # Summary
        print("\n" + "=" * 60)
        print("PREFLIGHT SUMMARY")
        print("=" * 60)
        
        all_ok = True
        
        # LabRecorder status
        if lr_status["running"]:
            print(f"  [OK] LabRecorder: RUNNING")
            print(f"        >>> Verify it is RECORDING and has all streams selected! <<<")
        else:
            print(f"  [!!] LabRecorder: NOT DETECTED (data may not be saved!)")
            all_ok = False
        
        # Stream status
        if expected_streams:
            for stream in result["streams"]:
                label = stream.get("name") or stream.get("type") or "unknown"
                if stream["found"]:
                    stats = stream.get("sample_stats", {})
                    if stats.get("streaming"):
                        rate = stats.get("measured_rate_hz", 0)
                        print(f"  [OK] {label}: STREAMING ({rate:.1f} Hz)")
                    else:
                        print(f"  [!!] {label}: FOUND but NO DATA")
                        all_ok = False
                else:
                    print(f"  [XX] {label}: NOT FOUND")
                    all_ok = False
        
        # Warnings
        has_warnings = bool(result.get("warnings"))
        if has_warnings:
            print("\n" + "-" * 40)
            print("SIGNAL QUALITY WARNINGS")
            print("-" * 40)
            for warning in result["warnings"]:
                print(f"  [!] {warning}")
            print("-" * 40)
        
        print("=" * 60)
        
        return all_ok, has_warnings, lr_status, result
    
    # Run checks in a retry loop
    while True:
        all_ok, has_warnings, lr_status, result = _run_single_check()
        
        if all_ok and not has_warnings:
            print("\nAll checks passed!")
            confirm = input("\nProceed to session setup? (y/n): ").strip().lower()
            if confirm in ("y", "yes", ""):
                return True
            elif confirm in ("n", "no"):
                return False
            # Any other input: re-run checks
            
        elif all_ok and has_warnings:
            print("\nStreams OK but there are signal quality warnings.")
            print("\nOptions:")
            print("  [y] Proceed to session setup")
            print("  [r] Re-check after fixing issues")
            print("  [n] Cancel and exit")
            
            confirm = input("\nChoice (y/r/n): ").strip().lower()
            if confirm in ("y", "yes"):
                return True
            elif confirm in ("n", "no"):
                return False
            elif confirm in ("r", "retry", ""):
                print("\n" + "~" * 60)
                print("Fix the issues and press Enter to re-check...")
                print("~" * 60)
                input()
                continue  # Re-run the check
            else:
                # Default to retry
                continue
                
        else:
            print("\nSome checks FAILED. Review the issues above.")
            print("\nOptions:")
            print("  [y] Proceed anyway (not recommended)")
            print("  [r] Re-check after fixing issues")
            print("  [n] Cancel and exit")
            
            confirm = input("\nChoice (y/r/n): ").strip().lower()
            if confirm in ("y", "yes"):
                return True
            elif confirm in ("n", "no"):
                return False
            elif confirm in ("r", "retry", ""):
                print("\n" + "~" * 60)
                print("Fix the issues and press Enter to re-check...")
                print("~" * 60)
                input()
                continue  # Re-run the check
            else:
                # Default to retry
                continue


def _get_env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _validate_id(value: str, *, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} is required")

    # Keep folder names safe and predictable
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{label} must be alphanumeric/underscore/dash (got: {value!r})")

    return value


def _load_assignments(repo_root: Path) -> dict:
    """Load participant assignments from config file."""
    assignments_path = repo_root / "config" / "participant_assignments.yaml"
    if not assignments_path.exists():
        return {"participants": {}}
    
    with open(assignments_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    return data if "participants" in data else {"participants": data}


def _save_assignments(repo_root: Path, assignments: dict, dry_run: bool = False) -> None:
    """Save participant assignments to config file."""
    if dry_run:
        return
    
    assignments_path = repo_root / "config" / "participant_assignments.yaml"
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(assignments_path, 'w', encoding='utf-8') as f:
        yaml.dump(assignments, f, default_flow_style=False, sort_keys=True)


def _get_recent_participants(assignments: dict, limit: int = 5) -> list[str]:
    """Get list of recent participants sorted by last_run timestamp."""
    participants = assignments.get("participants", {})
    
    # Filter participants with last_run timestamp
    recent = []
    for pid, data in participants.items():
        if data.get("last_run"):
            recent.append((pid, data["last_run"]))
    
    # Sort by timestamp descending
    recent.sort(key=lambda x: x[1], reverse=True)
    
    return [pid for pid, _ in recent[:limit]]


def _list_manifest_paths(sessions_dir: Path) -> set[Path]:
    if not sessions_dir.exists():
        return set()
    return {p.resolve() for p in sessions_dir.glob("**/*.manifest.json") if p.is_file()}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_seq_id_into_manifest(
    manifest_path: Path,
    *,
    seq_id: str,
    dry_run: bool,
    scenario_filename: str,
    abort_reason: Optional[str] = None,
) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["seq_id"] = seq_id
    manifest["unattended"] = False
    manifest["dry_run"] = dry_run
    if abort_reason:
        manifest["abort_reason"] = abort_reason
    if scenario_filename:
        manifest["scenario_name"] = Path(scenario_filename).stem
        openmatb_meta = manifest.get("openmatb")
        if not isinstance(openmatb_meta, dict):
            openmatb_meta = {}
            manifest["openmatb"] = openmatb_meta
        openmatb_meta["scenario_path"] = scenario_filename
    identifiers = manifest.get("identifiers")
    if not isinstance(identifiers, dict):
        identifiers = {}
        manifest["identifiers"] = identifiers
    identifiers["seq_id"] = seq_id

    _atomic_write_json(manifest_path, manifest)


def _get_playlist(seq_id: str, dry_run: bool, *, calibration_only: bool = False) -> list[str]:
    if dry_run:
        return ["pilot_dry_run_v0.txt"]

    playlist: list[str] = []
    if not calibration_only:
        # Fixed training sequence
        playlist.extend(
            [
                "pilot_practice_intro.txt",
                "pilot_practice_low.txt",
                "pilot_practice_moderate.txt",
                "pilot_practice_high.txt",
            ]
        )

    # calibration blocks based on counterbalancing sequence
    # SEQ1: Low -> Moderate -> High
    # SEQ2: Moderate -> High -> Low
    # SEQ3: High -> Low -> Moderate
    calibration_levels = {
        "SEQ1": ["LOW", "MODERATE", "HIGH"],
        "SEQ2": ["MODERATE", "HIGH", "LOW"],
        "SEQ3": ["HIGH", "LOW", "MODERATE"],
    }

    try:
        levels = calibration_levels[seq_id]
    except KeyError as exc:
        raise ValueError(f"Unknown sequence ID: {seq_id}") from exc

    for level in levels:
        playlist.append(f"pilot_calibration_{level.lower()}.txt")

    return playlist


def _stage_pilot_instruction_files(openmatb_dir: Path, repo_root: Path) -> None:
    """Copy repo-managed pilot instruction text files into OpenMATB includes.

    OpenMATB validates blocking-plugin filenames by requiring them to exist under
    includes/instructions/ or includes/questionnaires/.
    """

    source_dir = repo_root / "instructions"
    required_names = [
        "1_welcome.txt",
        "2_sysmon.txt",
        "3_track.txt",
        "4_comm.txt",
        "5_resman.txt",
        "6_all_tasks.txt",
    ]

    missing = [name for name in required_names if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required pilot instruction files under <repo>/instructions: " + ", ".join(missing)
        )

    # Keep pilot assets namespaced to avoid collisions with vendor examples.
    target_dir = openmatb_dir / "includes" / "instructions" / "pilot_en"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required_names:
        shutil.copyfile(source_dir / name, target_dir / name)


def _rewrite_scenario_paths_for_openmatb_includes(scenario_text: str) -> str:
    """Rewrite repo scenario text for OpenMATB runtime/validation compatibility."""

    # The vendor validator requires filenames for blocking plugins to be present under
    # includes/instructions/ or includes/questionnaires/.
    # Our repo scenarios may reference historical paths like ../../../../assets/…
    # Rewrite those to the namespaced include folder we populate.
    rewrites = [
        ("../../../../assets/instructions/pilot_en/", "pilot_en/"),
        ("..\\..\\..\\..\\assets\\instructions\\pilot_en\\", "pilot_en/"),
        ("..\\..\\..\\..\\assets\\instructions\\pilot_en/", "pilot_en/"),
        ("../../../../assets/instructions/pilot_en\\", "pilot_en/"),
    ]
    for old, new in rewrites:
        scenario_text = scenario_text.replace(old, new)

    # Normalize vendor parameter keys that are case-sensitive.
    # The ResMan plugin uses lowercase tank letters in parameter names
    # (e.g., tank-a-lossperminute). Some generated scenarios used tank-A-…
    # which fails OpenMATB validation.
    scenario_text = re.sub(
        r"\btank-([A-F])-",
        lambda m: f"tank-{m.group(1).lower()}-",
        scenario_text,
    )

    # Vendor genericscales plugin expects:
    #   <t>;genericscales;filename;<questionnaire.txt>
    #   <t>;genericscales;start
    # Some generated scenarios used genericscales;create and genericscales;load, which are invalid.
    scenario_text = re.sub(
        r"^\s*\d+:\d+:\d+;genericscales;create\s*$\r?\n?",
        "",
        scenario_text,
        flags=re.MULTILINE,
    )
    scenario_text = scenario_text.replace(";genericscales;load;", ";genericscales;filename;")
    return scenario_text


def _run_single_scenario(
    openmatb_dir: Path,
    scenario_filename: str,
    output_root_path: Path,
    participant: str,
    session: str,
    seq_id: str,
    args: argparse.Namespace,
    repo_commit: str,
    submodule_commit: str,
) -> tuple[int, Optional[Path]]:
    repo_root = Path(__file__).resolve().parents[2]

    try:
        _stage_pilot_instruction_files(openmatb_dir, repo_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2, None

    scenario_source_path = repo_root / "scenarios" / scenario_filename
    if not scenario_source_path.exists():
        print(f"Scenario file not found: {scenario_source_path}", file=sys.stderr)
        return 2, None

    scenario_target_path = openmatb_dir / "includes" / "scenarios" / scenario_filename
    scenario_target_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_text = scenario_source_path.read_text(encoding="utf-8")
    scenario_text = _rewrite_scenario_paths_for_openmatb_includes(scenario_text)
    scenario_target_path.write_text(scenario_text, encoding="utf-8")

    print(f"\n>>> Starting Scenario Block: {scenario_filename}")

    # Reset/Update environment for this specific run
    env = os.environ.copy()
    env["OPENMATB_OUTPUT_ROOT"] = str(output_root_path)
    env["OPENMATB_OUTPUT_SUBDIR"] = str(Path("openmatb") / participant / session)
    env["OPENMATB_PARTICIPANT"] = participant
    env["OPENMATB_SESSION"] = session
    env["OPENMATB_SEQ_ID"] = seq_id
    env["OPENMATB_REPO_COMMIT"] = repo_commit
    env["OPENMATB_SUBMODULE_COMMIT"] = submodule_commit

    # Calculate paths specifically for this run
    scenario_rel_path = scenario_filename
    sessions_dir = output_root_path / Path(env["OPENMATB_OUTPUT_SUBDIR"]) / "sessions"
    manifests_before = _list_manifest_paths(sessions_dir)

    # Generate bootstrap script dynamically for the scenario
    bootstrap = (
        "import sys\n"
        f"sys.argv.extend(['--speed', '{args.speed}'])\n"
        "import gettext, os, runpy\n"
        "from pathlib import Path\n"
        "\n"
        "# Wire --speed into the OpenMATB scenario clock (vendor does not parse argv for this).\n"
        "_speed = 1\n"
        "try:\n"
        "    if '--speed' in sys.argv:\n"
        "        i = sys.argv.index('--speed')\n"
        "        _speed = int(float(sys.argv[i + 1]))\n"
        "except Exception:\n"
        "    _speed = 1\n"
        "if _speed < 1:\n"
        "    _speed = 1\n"
        "try:\n"
        "    from core.clock import Clock\n"
        "    Clock._speed = _speed\n"
        "except Exception:\n"
        "    pass\n"
        "LOCALE_PATH = Path('.', 'locales')\n"
        "language_iso = [l for l in open('config.ini', 'r').readlines() if 'language=' in l][0].split('=')[-1].strip()\n"
        "language = gettext.translation('openmatb', LOCALE_PATH, [language_iso])\n"
        "language.install()\n"
        "from core.constants import CONFIG\n"
        "if not CONFIG.has_section('Openmatb'):\n"
        "    CONFIG.add_section('Openmatb')\n"
        f"CONFIG.set('Openmatb', 'scenario_path', {scenario_rel_path!r})\n"
        "# Some vendor plugins call self.log_manual_entry(), but AbstractPlugin only exposes self.logger.\n"
        "# Provide a compatibility shim to avoid runtime AttributeError.\n"
        "try:\n"
        "    from plugins.abstractplugin import AbstractPlugin\n"
        "    def _log_manual_entry(self, entry, key='manual'):\n"
        "        try:\n"
        "            self.logger.log_manual_entry(entry, key=key)\n"
        "        except Exception:\n"
        "            pass\n"
        "    AbstractPlugin.log_manual_entry = _log_manual_entry\n"
        "except Exception:\n"
        "    pass\n"
        "import core.event as event_mod\n"
        "_orig_parse = event_mod.Event.parse_from_string\n"
        "def _replace_tokens(text):\n"
        "    token_map = {\n"
        "        '${OPENMATB_PARTICIPANT}': os.environ.get('OPENMATB_PARTICIPANT', ''),\n"
        "        '${OPENMATB_SESSION}': os.environ.get('OPENMATB_SESSION', ''),\n"
        "        '${OPENMATB_SEQ_ID}': os.environ.get('OPENMATB_SEQ_ID', ''),\n"
        "    }\n"
        "    for key, value in token_map.items():\n"
        "        text = text.replace(key, value)\n"
        "    return text\n"
        "def _patched_parse(cls, line_id, line_str):\n"
        "    return _orig_parse(line_id, _replace_tokens(line_str))\n"
        "event_mod.Event.parse_from_string = classmethod(_patched_parse)\n"
        "from core.window import Window\n"
        "from core.modaldialog import ModalDialog\n"
        "from core.utils import get_conf_value\n"
        "from core.constants import REPLAY_MODE\n"
        "def _display_session_id(self):\n"
        "    if not REPLAY_MODE and get_conf_value('Openmatb', 'display_session_number'):\n"
        "        pid = os.environ.get('OPENMATB_PARTICIPANT') or 'UNKNOWN'\n"
        "        sid = os.environ.get('OPENMATB_SESSION') or 'UNKNOWN'\n"
        "        seq = os.environ.get('OPENMATB_SEQ_ID') or 'UNKNOWN'\n"
        "        msg = [f'Participant ID: {pid}', f'Session ID: {sid}', f'Sequence ID: {seq}']\n"
        "        self.modal_dialog = ModalDialog(self, msg, 'OpenMATB')\n"
        "Window.display_session_id = _display_session_id\n"
        "runpy.run_path('main.py', run_name='__main__')\n"
    )

    # Pass the modified environment
    proc = subprocess.Popen([sys.executable, "-c", bootstrap], cwd=str(openmatb_dir), env=env)
    exit_code = proc.wait()

    # Post-process manifests
    if exit_code != 0:
        manifests_after = _list_manifest_paths(sessions_dir)
        new_manifests = sorted(manifests_after - manifests_before)
        if len(new_manifests) == 1:
            _write_seq_id_into_manifest(
                new_manifests[0],
                seq_id=seq_id,
                dry_run=args.dry_run,
                scenario_filename=scenario_filename,
                abort_reason=f"exit_code_{exit_code}",
            )
        return exit_code, (new_manifests[0] if len(new_manifests) == 1 else None)

    manifests_after = _list_manifest_paths(sessions_dir)
    new_manifests = sorted(manifests_after - manifests_before)
    if len(new_manifests) != 1:
        print(
            f"WARNING: Expected 1 new manifest for {scenario_filename}, found {len(new_manifests)} in {sessions_dir}. "
            "Skipping manifest metadata injection.",
            file=sys.stderr,
        )
        return 2, None

    manifest_path = new_manifests[0]
    _write_seq_id_into_manifest(
        manifest_path,
        seq_id=seq_id,
        dry_run=args.dry_run,
        scenario_filename=scenario_filename,
        abort_reason=None,
    )

    if getattr(args, "summarise_performance", False):
        try:
            summariser = (repo_root / "src" / "python" / "summarise_openmatb_performance.py").resolve()
            subprocess.run(
                [sys.executable, str(summariser), "--manifest", str(manifest_path)],
                check=False,
                cwd=str(repo_root),
            )
        except Exception:
            # Performance summarization is best-effort and must not crash an attended run.
            pass

    # If OpenMATB reports scenario parsing/runtime errors, treat this run as failed.
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        errors_log = Path(manifest.get("paths", {}).get("scenario_errors_log", ""))

        def _errors_log_has_actionable_errors(path: Path) -> bool:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return False
            stripped = text.strip()
            if not stripped:
                return False
            # OpenMATB writes a sentinel line when there are no validation errors.
            if stripped.lower() in {"no error", "no errors"}:
                return False
            return True

        if errors_log and errors_log.exists() and _errors_log_has_actionable_errors(errors_log):
            per_run_errors_log = manifest_path.with_suffix(manifest_path.suffix + ".errors.log")
            try:
                shutil.copyfile(errors_log, per_run_errors_log)
                manifest.setdefault("paths", {})["scenario_errors_log"] = str(per_run_errors_log)
                _atomic_write_json(manifest_path, manifest)
            except Exception:
                per_run_errors_log = errors_log
            _write_seq_id_into_manifest(
                manifest_path,
                seq_id=seq_id,
                dry_run=args.dry_run,
                scenario_filename=scenario_filename,
                abort_reason="scenario_errors",
            )
            print(f"Scenario errors detected; see: {per_run_errors_log}", file=sys.stderr)
            return 2, manifest_path

        # Detect early termination even without explicit errors: if the CSV ends far before
        # the latest timestamp in the scenario file, treat it as an abort.
        csv_path_str = manifest.get("paths", {}).get("session_csv")
        if csv_path_str:
            csv_path = Path(csv_path_str)
        else:
            csv_path = None

        def _scenario_max_time_seconds(path: Path) -> int:
            max_sec = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split(";")
                if not parts:
                    continue
                time_str = parts[0]
                try:
                    h, m, s = time_str.split(":")
                    sec = int(h) * 3600 + int(m) * 60 + int(s)
                except Exception:
                    continue
                if sec > max_sec:
                    max_sec = sec
            return max_sec

        def _csv_max_scenario_time_seconds(path: Path) -> float:
            try:
                with open(path, "r", encoding="utf-8", newline="") as f_csv:
                    header = f_csv.readline()
                    if not header:
                        return 0.0
                    cols = [c.strip() for c in header.split(",")]
                    try:
                        idx = cols.index("scenario_time")
                    except ValueError:
                        return 0.0
                    max_t = 0.0
                    for row in f_csv:
                        parts = row.rstrip("\n").split(",")
                        if len(parts) <= idx:
                            continue
                        try:
                            t = float(parts[idx])
                        except Exception:
                            continue
                        if t > max_t:
                            max_t = t
                    return max_t
            except Exception:
                return 0.0

        if csv_path and csv_path.exists():
            expected_end = _scenario_max_time_seconds(scenario_source_path)
            observed_end = _csv_max_scenario_time_seconds(csv_path)
            # Allow some slack (UI pauses, logging granularity), but if we didn't even
            # reach ~90% of intended time, assume the run ended early.
            if expected_end >= 10 and observed_end < (expected_end * 0.9):
                _write_seq_id_into_manifest(
                    manifest_path,
                    seq_id=seq_id,
                    dry_run=args.dry_run,
                    scenario_filename=scenario_filename,
                    abort_reason=f"early_exit_observed_{observed_end:.3f}_expected_{expected_end}",
                )
                print(
                    f"Scenario ended early (observed scenario_time={observed_end:.2f}s, expected~{expected_end}s)",
                    file=sys.stderr,
                )
                return 2, manifest_path
    except Exception:
        # If we can't read the manifest or error log, do not crash the runner.
        pass
    return 0, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenMATB with external output paths.")
    parser.add_argument(
        "--participant",
        help="Participant ID (e.g., P001). Can also be set via OPENMATB_PARTICIPANT.",
    )
    parser.add_argument(
        "--session",
        help="Session ID (e.g., S001). Can also be set via OPENMATB_SESSION.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="External data root (default: C:\\data\\adaptive_matb or OPENMATB_OUTPUT_ROOT).",
    )
    parser.add_argument(
        "--openmatb-dir",
        default=None,
        help="Path to OpenMATB directory (default: <repo>/src/python/vendor/openmatb).",
    )

    parser.add_argument(
        "--seq-id",
        required=False,
        choices=("SEQ1", "SEQ2", "SEQ3"),
        help="calibration-order sequence ID (SEQ1/SEQ2/SEQ3). Can also be set via OPENMATB_SEQ_ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the deterministic dry-run scenario artifact.",
    )
    parser.add_argument(
        "--verification",
        action="store_true",
        help="Enable fast-forward and other automation-oriented behaviors (verification only).",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Fast-forward speed multiplier (ignored unless --verification is set).",
    )

    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run only the counterbalanced calibration blocks (skip practice/training).",
    )

    parser.add_argument(
        "--pilot1",
        action="store_true",
        help=(
            "Enable Pilot 1 checks: requires a calibration-stage .xdf (LabRecorder) so a run-level manifest "
            "can link physiology to OpenMATB outputs and run alignment QC."
        ),
    )

    parser.add_argument(
        "--xdf-path",
        default=None,
        help=(
            "Path to the LabRecorder .xdf file that contains OpenMATB markers + physiology for this run. "
            "Required for --pilot1."
        ),
    )

    parser.add_argument(
        "--skip-xdf-qc",
        action="store_true",
        help="Skip XDF↔CSV marker alignment QC (pilot verification).",
    )

    parser.add_argument(
        "--summarise-performance",
        action="store_true",
        help="Write a derived performance summary JSON next to each run manifest.",
    )
    parser.add_argument(
        "--skip-assignment-update",
        action="store_true",
        help="Don't update participant_assignments.yaml (dry-run for assignments).",
    )

    # EDA streamer integration (opt-in)
    parser.add_argument(
        "--with-eda",
        action="store_true",
        help=(
            "Spawn the Shimmer EDA-to-LSL streamer subprocess before scenarios. "
            "Requires --eda-port. The streamer is terminated when the runner exits."
        ),
    )
    parser.add_argument(
        "--eda-port",
        default=None,
        help="Shimmer Bluetooth COM port (e.g., COM5). Required when --with-eda is set.",
    )
    parser.add_argument(
        "--eda-stream-name",
        default="ShimmerEDA",
        help="LSL stream name for EDA data (default: ShimmerEDA).",
    )
    parser.add_argument(
        "--eda-health-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for EDA LSL stream before failing (default: 15).",
    )

    # Pre-flight stream checks
    parser.add_argument(
        "--check-streams",
        action="store_true",
        help=(
            "Run pre-flight LSL stream check before session setup. "
            "Verifies EEG and EDA streams are live before prompting for participant."
        ),
    )
    parser.add_argument(
        "--skip-stream-check",
        action="store_true",
        help="Skip pre-flight stream check (even if --pilot1 is set).",
    )
    parser.add_argument(
        "--eeg-stream-type",
        default="EEG",
        help="LSL stream type to search for EEG (default: EEG).",
    )

    args = parser.parse_args()

    if args.pilot1 and args.dry_run:
        print("NOTE: --pilot1 is ignored during --dry-run.", file=sys.stderr)
        args.pilot1 = False

    if args.speed != 1 and not args.verification:
        print(
            f"NOTE: Ignoring --speed={args.speed} because this is an attended run. "
            "Use --verification to enable fast-forward.",
            file=sys.stderr,
        )
        args.speed = 1

    # Validate EDA arguments
    if args.with_eda and not args.eda_port:
        print("ERROR: --with-eda requires --eda-port (e.g., --eda-port COM5)", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    
    # EDA subprocess state (will be populated if --with-eda is set)
    eda_info: Optional[dict] = None
    
    # Load participant assignments
    assignments = _load_assignments(repo_root)
    participants_data = assignments.get("participants", {})

    # Pre-flight stream check (before asking for participant info)
    # Run if --check-streams is set, or if --pilot1 is set (unless --skip-stream-check)
    should_check_streams = args.check_streams or (args.pilot1 and not args.skip_stream_check)
    
    if should_check_streams and not args.verification:
        # Don't check EDA if we're about to spawn it with --with-eda
        check_eda = not args.with_eda
        
        if not _run_preflight_checks(
            check_eeg=True,
            check_eda=check_eda,
            eeg_stream_type=args.eeg_stream_type,
            eda_stream_name=args.eda_stream_name,
            timeout=5.0,
        ):
            print("Session cancelled by user.", file=sys.stderr)
            return 1

    # Interactive prompts when arguments not provided
    participant_raw = args.participant or _get_env_first("OPENMATB_PARTICIPANT", "OPENMATB_PARTICIPANT_ID")
    session_raw = args.session or _get_env_first("OPENMATB_SESSION", "OPENMATB_SESSION_ID")
    seq_id = args.seq_id or _get_env_first("OPENMATB_SEQ_ID")

    # Interactive participant selection
    if participant_raw is None:
        print("\n=== OpenMATB Session Setup ===")
        
        # Show recent participants
        recent = _get_recent_participants(assignments, limit=5)
        if recent:
            print("\nRecent participants:")
            for i, pid in enumerate(recent, 1):
                pdata = participants_data[pid]
                seq = pdata.get("sequence", "?")
                completed = len(pdata.get("sessions_completed", []))
                print(f"  {i}. {pid} ({seq}, {completed} sessions)")
            print(f"  Or enter a participant number/ID")
        
        while not participant_raw:
            user_input = input("\nEnter participant number: ").strip()
            if not user_input:
                print("Participant selection cannot be empty.")
                continue
            
            # Check if selecting from recent list
            if user_input.isdigit() and 1 <= int(user_input) <= len(recent):
                participant_raw = recent[int(user_input) - 1]
            else:
                # Parse as participant number
                if not user_input.startswith('P'):
                    try:
                        num = int(user_input)
                        participant_raw = f"P{num:03d}"
                    except ValueError:
                        participant_raw = f"P{user_input}"
                else:
                    participant_raw = user_input
    
    # Look up sequence from assignments
    if participant_raw in participants_data:
        assigned_seq = participants_data[participant_raw].get("sequence")
        assigned_sessions = participants_data[participant_raw].get("sessions_completed", [])
        
        # Use assigned sequence if not overridden
        if seq_id is None:
            seq_id = assigned_seq
        elif seq_id != assigned_seq and assigned_seq:
            print(f"\nWARNING: Overriding assigned sequence {assigned_seq} with {seq_id}", file=sys.stderr)
        
        # Auto-increment session if needed
        if session_raw is None:
            next_session_num = len(assigned_sessions) + 1
            session_raw = f"S{next_session_num:03d}"
    else:
        # New participant not in assignments
        print(f"\nERROR: {participant_raw} not found in assignments file.", file=sys.stderr)
        print(f"Add to config/participant_assignments.yaml first, or use:", file=sys.stderr)
        print(f"  python scripts/generate_participant_assignments.py --participant-ids {participant_raw} --sequences SEQ1", file=sys.stderr)
        return 2
    
    # Confirmation prompt
    if not args.dry_run:
        print(f"\n{'='*50}")
        print(f"  Participant: {participant_raw}")
        print(f"  Sequence:    {seq_id}")
        print(f"  Session:     {session_raw}")
        print(f"{'='*50}")
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm not in ('y', 'yes'):
            print("Aborted by user.")
            return 1

    try:
        participant = _validate_id(participant_raw, label="participant")
        session = _validate_id(session_raw, label="session")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_root = args.output_root or _get_env_first("OPENMATB_OUTPUT_ROOT") or r"C:\data\adaptive_matb"
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        print(f"OPENMATB_OUTPUT_ROOT must be an absolute path (got: {output_root})", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    openmatb_dir = Path(args.openmatb_dir) if args.openmatb_dir else repo_root / "src" / "python" / "vendor" / "openmatb"

    if not openmatb_dir.exists():
        print(f"OpenMATB directory not found: {openmatb_dir}", file=sys.stderr)
        return 2

    # Check if session output already exists (prevent accidental overwriting)
    session_output_dir = output_root_path / "openmatb" / participant / session
    if session_output_dir.exists():
        existing_files = list(session_output_dir.glob("*"))
        if existing_files:
            print(f"\n{'!'*60}", file=sys.stderr)
            print(f"WARNING: Session output directory already contains data!", file=sys.stderr)
            print(f"  Path: {session_output_dir}", file=sys.stderr)
            print(f"  Files: {len(existing_files)} existing file(s)", file=sys.stderr)
            print(f"{'!'*60}", file=sys.stderr)
            print("\nOptions:")
            print("  [y] Continue (will ADD to existing data - may cause confusion)")
            print("  [n] Cancel and choose a different session ID")
            confirm = input("\nContinue with this session? (y/n): ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Aborted. Use a different session ID or remove existing data.")
                return 1

    def _git_rev_parse_head(cwd: Path) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.STDOUT)
        except Exception as exc:
            raise RuntimeError(f"Unable to determine git commit hash in {cwd}: {exc}") from exc
        return out.decode("utf-8", errors="replace").strip()

    try:
        repo_commit = _git_rev_parse_head(repo_root)
        submodule_commit = _git_rev_parse_head(openmatb_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        playlist = _get_playlist(seq_id, args.dry_run, calibration_only=bool(args.calibration_only))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    missing_scenarios: list[str] = []
    for scenario_filename in playlist:
        if not (repo_root / "scenarios" / scenario_filename).exists():
            missing_scenarios.append(scenario_filename)
    if missing_scenarios:
        print("Missing scenario files under <repo>/scenarios:", file=sys.stderr)
        for name in missing_scenarios:
            print(f" - {name}", file=sys.stderr)
        return 2

    print(f"Running sequence: {seq_id}")
    print(f"Playlist ({len(playlist)} scenarios):")
    for s in playlist:
        print(f" - {s}")

    # Start EDA streamer if requested (before scenarios)
    if args.with_eda:
        print("\n" + "="*60)
        print("Starting EDA streamer...")
        print("="*60)
        
        eda_info = _start_eda_streamer(
            repo_root=repo_root,
            eda_port=args.eda_port,
            eda_stream_name=args.eda_stream_name,
            health_check_timeout=args.eda_health_timeout,
        )
        
        if not eda_info["health_check_passed"]:
            print(f"\n!!! EDA HEALTH CHECK FAILED !!!", file=sys.stderr)
            print(f"Error: {eda_info.get('error', 'Unknown error')}", file=sys.stderr)
            print("\nCannot proceed without verified EDA stream.", file=sys.stderr)
            _stop_eda_streamer()
            return 2
        
        print(f"EDA streamer ready (PID: {eda_info['pid']}, stream: {eda_info['stream_name']})")
        print("="*60 + "\n")

    # Final launch checkpoint (only for real runs, not verification)
    if not args.verification and not args.dry_run:
        print("\n" + "=" * 60)
        print("FINAL LAUNCH CHECKLIST")
        print("=" * 60)
        print(f"  Participant:  {participant}")
        print(f"  Session:      {session}")
        print(f"  Sequence:     {seq_id}")
        print(f"  Scenarios:    {len(playlist)}")
        print("-" * 60)
        print("  Before pressing Enter, confirm:")
        print("    [ ] LabRecorder is RECORDING (red button pressed)")
        print("    [ ] All streams visible in LabRecorder (EEG, EDA, OpenMATB*)")
        print("    [ ] Participant is seated and ready")
        print("    [ ] Electrodes are attached and signal looks good")
        print("-" * 60)
        print("  * OpenMATB markers stream will appear when first scenario starts")
        print("=" * 60)
        
        launch = input("\nStart session now? (y/n): ").strip().lower()
        if launch not in ("y", "yes"):
            print("Session aborted by user at final checkpoint.")
            if args.with_eda:
                _stop_eda_streamer()
            return 1

    scenario_manifests: list[Path] = []

    for scenario_filename in playlist:
        exit_code, manifest_path = _run_single_scenario(
            openmatb_dir=openmatb_dir,
            scenario_filename=scenario_filename,
            output_root_path=output_root_path,
            participant=participant,
            session=session,
            seq_id=seq_id,
            args=args,
            repo_commit=repo_commit,
            submodule_commit=submodule_commit,
        )

        if exit_code != 0:
            print(f"\n!!! Scenario {scenario_filename} failed (code {exit_code}). Stopping sequence. !!!", file=sys.stderr)
            return exit_code

        if manifest_path:
            scenario_manifests.append(manifest_path)
        
        # Simple separation between blocks
        print(f"Scenario {scenario_filename} completed successfully.")
        # (Interactive UI and blocking dialogs are expected in attended mode)

    print("\nAll scenarios in playlist completed successfully.")

    if args.pilot1 and not args.xdf_path:
        # We require an explicit .xdf to create the run-level manifest and enforce QC.
        print("\n" + "-" * 50)
        print("XDF PATH REQUIRED")
        print("-" * 50)
        print("Stop LabRecorder recording now (if not already stopped).")
        print("Enter the path to the .xdf file that was just recorded.")
        print("-" * 50)
        
        while True:
            xdf_in = input("\nEnter LabRecorder .xdf path: ").strip()
            if not xdf_in:
                print("ERROR: --pilot1 requires an .xdf path (LabRecorder output).", file=sys.stderr)
                retry = input("Try again? (y/n): ").strip().lower()
                if retry not in ("y", "yes"):
                    return 2
                continue
            
            # Validate the path exists
            xdf_path = Path(xdf_in)
            if not xdf_path.exists():
                print(f"ERROR: File not found: {xdf_path}", file=sys.stderr)
                print("Check the path and try again.")
                continue
            
            if not xdf_path.suffix.lower() == ".xdf":
                print(f"WARNING: File does not have .xdf extension: {xdf_path.name}")
                confirm = input("Use this file anyway? (y/n): ").strip().lower()
                if confirm not in ("y", "yes"):
                    continue
            
            # Check file size (sanity check)
            file_size_mb = xdf_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 0.1:
                print(f"WARNING: XDF file is very small ({file_size_mb:.2f} MB) - may be incomplete.")
                confirm = input("Use this file anyway? (y/n): ").strip().lower()
                if confirm not in ("y", "yes"):
                    continue
            
            print(f"Using XDF: {xdf_path} ({file_size_mb:.1f} MB)")
            args.xdf_path = str(xdf_path)
            break

    # Write run-level manifest linking the playlist to OpenMATB outputs (and optional physiology XDF).
    try:
        session_root = output_root_path / "openmatb" / participant / session
        run_manifest_dir = session_root
        run_manifest_dir.mkdir(parents=True, exist_ok=True)

        scenario_rows = []
        for scenario_filename, manifest_path in zip(playlist, scenario_manifests, strict=False):
            row = {
                "scenario_filename": scenario_filename,
                "scenario_name": Path(scenario_filename).stem,
                "manifest_path": str(manifest_path),
            }
            try:
                manifest = _read_json(manifest_path)
                row["session_csv"] = str(manifest.get("paths", {}).get("session_csv", ""))
                row["lsl_enabled"] = bool(manifest.get("lsl_enabled"))
            except Exception:
                # Best-effort; don't crash an attended run if a manifest is malformed.
                pass
            scenario_rows.append(row)

        timestamp_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_manifest_path = run_manifest_dir / f"run_manifest_{timestamp_tag}.json"
        if run_manifest_path.exists():
            print(f"ERROR: Refusing to overwrite existing run manifest: {run_manifest_path}", file=sys.stderr)
            return 2

        run_manifest = {
            "schema": "pilot_run_manifest_v0",
            "created_at": datetime.now().isoformat(),
            "participant": participant,
            "session": session,
            "seq_id": seq_id,
            "mode": {
                "pilot1": bool(args.pilot1),
                "calibration_only": bool(args.calibration_only),
                "dry_run": bool(args.dry_run),
                "verification": bool(args.verification),
            },
            "repo": {
                "commit": repo_commit,
                "openmatb_submodule_commit": submodule_commit,
            },
            "openmatb": {
                "output_root": str(output_root_path),
                "output_subdir": str(Path("openmatb") / participant / session),
                "session_root": str(session_root),
                "openmatb_dir": str(openmatb_dir),
            },
            "physiology": {
                "xdf_path": str(args.xdf_path) if args.xdf_path else None,
                "recording_scope": "calibration_only",
                "expected_streams": {
                    "markers": {"name": "OpenMATB", "type": "Markers"},
                    "eda": {"type": "EDA"},
                },
                "eda_streamer": {
                    "managed_by_runner": bool(args.with_eda),
                    "started": eda_info["started"] if eda_info else False,
                    "pid": eda_info["pid"] if eda_info else None,
                    "stream_name": eda_info["stream_name"] if eda_info else args.eda_stream_name,
                    "stream_type": eda_info["stream_type"] if eda_info else "EDA",
                    "health_check_passed": eda_info["health_check_passed"] if eda_info else None,
                    "port": args.eda_port if args.with_eda else None,
                    "error": eda_info.get("error") if eda_info else None,
                } if args.with_eda else None,
            },
            "playlist": scenario_rows,
            "qc": {
                "xdf_alignment": {
                    "required": bool(args.pilot1),
                    "skipped": bool(args.skip_xdf_qc),
                    "status": "pending" if (args.pilot1 and not args.skip_xdf_qc) else ("skipped" if args.skip_xdf_qc else "not_required"),
                    "report_path": None,
                    "thresholds": {
                        "median_abs_ms": 20,
                        "p95_abs_ms": 50,
                        "drift_ms_per_min": 5,
                        "hard_fail_drift_ms_per_min": 20,
                    },
                }
            },
        }

        _atomic_write_json(run_manifest_path, run_manifest)
        print(f"Wrote run manifest: {run_manifest_path}")
    except Exception as exc:
        print(f"ERROR: Failed to write run-level manifest: {exc}", file=sys.stderr)
        return 2
    
    # Run XDF↔CSV marker alignment QC if --pilot1 and not --skip-xdf-qc
    if args.pilot1 and not args.skip_xdf_qc and args.xdf_path:
        print("\n" + "="*60)
        print("Running XDF↔CSV marker alignment QC...")
        print("="*60)
        
        try:
            # Import from sibling verification module
            import importlib.util
            qc_script = Path(__file__).parent / "verification" / "verify_xdf_alignment.py"
            spec = importlib.util.spec_from_file_location("verify_xdf_alignment", qc_script)
            verify_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verify_module)
            run_qc = verify_module.run_qc
            
            csv_paths = [Path(row["session_csv"]) for row in scenario_rows if row.get("session_csv")]
            qc_report_path = run_manifest_path.with_suffix(".qc_alignment.json")
            
            qc_report = run_qc(
                xdf_path=Path(args.xdf_path),
                csv_paths=csv_paths,
                thresholds=run_manifest["qc"]["xdf_alignment"]["thresholds"],
                output_path=qc_report_path,
            )
            
            qc_passed = qc_report.get("results", {}).get("passed", False)
            run_manifest["qc"]["xdf_alignment"]["status"] = "passed" if qc_passed else "failed"
            run_manifest["qc"]["xdf_alignment"]["report_path"] = str(qc_report_path)
            
            # Update run manifest with QC results
            _atomic_write_json(run_manifest_path, run_manifest)
            
            if not qc_passed:
                print("\n⚠️  XDF alignment QC FAILED")
                print("Fail reasons:")
                for reason in qc_report.get("results", {}).get("fail_reasons", []):
                    print(f"  - {reason}")
                print(f"\nSee QC report: {qc_report_path}")
            else:
                print("\n✓ XDF alignment QC PASSED")
                timing = qc_report.get("results", {}).get("timing", {})
                print(f"  Median error: {timing.get('median_abs_error_ms', 0):.1f} ms")
                print(f"  95th pct:     {timing.get('p95_abs_error_ms', 0):.1f} ms")
        
        except ImportError as exc:
            print(f"WARNING: Could not run XDF QC (missing dependency): {exc}", file=sys.stderr)
            print("Install with: pip install pyxdf", file=sys.stderr)
            run_manifest["qc"]["xdf_alignment"]["status"] = "skipped_missing_dep"
            _atomic_write_json(run_manifest_path, run_manifest)
        except Exception as exc:
            print(f"WARNING: XDF QC failed with error: {exc}", file=sys.stderr)
            run_manifest["qc"]["xdf_alignment"]["status"] = f"error: {exc}"
            _atomic_write_json(run_manifest_path, run_manifest)
    
    # Update participant assignments
    if not args.skip_assignment_update:
        # Ensure participant is in assignments
        if participant not in participants_data:
            participants_data[participant] = {
                "sequence": seq_id,
                "sessions_completed": [],
                "last_run": None
            }
        
        # Add session to completed list if not already there
        pdata = participants_data[participant]
        if session not in pdata.get("sessions_completed", []):
            if "sessions_completed" not in pdata:
                pdata["sessions_completed"] = []
            pdata["sessions_completed"].append(session)
        
        # Update last run timestamp
        pdata["last_run"] = datetime.now().isoformat()
        
        # Save assignments
        assignments["participants"] = participants_data
        _save_assignments(repo_root, assignments, dry_run=args.skip_assignment_update)
        print(f"Updated participant assignments: {participant} completed {session}")
    
    # Stop EDA streamer if we started it
    if args.with_eda:
        print("\nStopping EDA streamer...")
        _stop_eda_streamer()
        print("EDA streamer stopped.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
