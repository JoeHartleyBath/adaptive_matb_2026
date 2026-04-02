r"""Launch OpenMATB with repo-safe output paths.

This wrapper enforces that participant/session identifiers are provided and configures
OpenMATB to write session logs outside the git repo.

Usage (PowerShell):
  cd src/python/vendor/openmatb
        .\.venv\Scripts\Activate.ps1
    python ..\..\run_openmatb.py --participant P001 --session S001

Environment variables (optional):
  OPENMATB_OUTPUT_ROOT   (default: C:\data\adaptive_matb)
  OPENMATB_PARTICIPANT / OPENMATB_PARTICIPANT_ID
  OPENMATB_SESSION / OPENMATB_SESSION_ID

OpenMATB uses:
  OPENMATB_OUTPUT_ROOT and OPENMATB_OUTPUT_SUBDIR
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import shutil
import socket
import sys
import time
import yaml
from pathlib import Path
from typing import Optional

# Suppress liblsl's verbose C++ logging (must be set before pylsl is imported).
# Level semantics: DEBUG=5, INFO=1, WARNING=0, ERROR=-1, FATAL=-2.
# Setting to -3 suppresses everything including the spurious "Stream transmission
# broke off; re-connecting" ERROR messages that eego amps emit on normal
# keep-alive reconnects.  Real data problems are caught by the sample-flow check.
os.environ.setdefault("LSL_LOGLEVEL", "-3")

# ---------------------------------------------------------------------------
# Subprocess management (global state for crash-safe cleanup)
# ---------------------------------------------------------------------------
_eda_subprocess: Optional[subprocess.Popen] = None
_hr_subprocess: Optional[subprocess.Popen] = None
_lsl_recorder_subprocess: Optional[subprocess.Popen] = None
_lsl_recording_path: Optional[Path] = None
_mwl_subprocess: Optional[subprocess.Popen] = None

_labrecorder_rcs_recording_started: bool = False
_labrecorder_rcs_host: Optional[str] = None
_labrecorder_rcs_port: Optional[int] = None


def _read_pinned_pyglet_version(requirements_path: Path) -> Optional[str]:
    if not requirements_path.exists():
        return None
    try:
        for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("pyglet=="):
                return line.split("==", 1)[1].strip()
    except Exception:
        return None
    return None


def _ensure_openmatb_runtime_dependencies(openmatb_dir: Path) -> None:
    requirements_path = openmatb_dir / "requirements.txt"
    pinned_pyglet = _read_pinned_pyglet_version(requirements_path)

    if not pinned_pyglet:
        return

    try:
        installed_pyglet = importlib.metadata.version("pyglet")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "OpenMATB dependency check failed: pyglet is not installed.\n"
            f"Install pinned dependencies with:\n  {sys.executable} -m pip install -r {requirements_path}"
        ) from exc

    # Allow any patch release within the same minor series (e.g. 1.5.31 when pinned is 1.5.26).
    # Newer patch versions have Python 3.13 compatibility fixes; exact pinning breaks on newer Pythons.
    # We do the real capability check below (OrderedGroup) rather than relying on the version number.
    pinned_major_minor = ".".join(pinned_pyglet.split(".")[:2])
    installed_major_minor = ".".join(installed_pyglet.split(".")[:2])
    if installed_major_minor != pinned_major_minor:
        raise RuntimeError(
            "OpenMATB dependency check failed: incompatible pyglet version.\n"
            f"  Required: {pinned_major_minor}.x (pinned: {pinned_pyglet})\n"
            f"  Installed: {installed_pyglet}\n"
            f"Install pinned dependencies with:\n  {sys.executable} -m pip install -r {requirements_path}"
        )

    try:
        from pyglet.graphics import OrderedGroup  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "OpenMATB dependency check failed: pyglet does not expose OrderedGroup, which this OpenMATB build requires.\n"
            f"  Installed pyglet: {installed_pyglet}\n"
            f"  Try: {sys.executable} -m pip install 'pyglet>=1.5.26,<2.0'\n"
            f"  Or reinstall pinned deps: {sys.executable} -m pip install -r {requirements_path}"
        ) from exc


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


def _cleanup_hr_subprocess() -> None:
    """Terminate HR streamer subprocess if running. Called via atexit or signal handlers."""
    global _hr_subprocess
    if _hr_subprocess is not None and _hr_subprocess.poll() is None:
        print("Terminating HR streamer subprocess...", file=sys.stderr)
        try:
            _hr_subprocess.terminate()
            _hr_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("HR streamer did not terminate, killing...", file=sys.stderr)
            _hr_subprocess.kill()
            _hr_subprocess.wait()
        except Exception as e:
            print(f"Error cleaning up HR subprocess: {e}", file=sys.stderr)
        _hr_subprocess = None


def _cleanup_lsl_recorder_subprocess() -> None:
    """Terminate Python LSL recorder subprocess if running."""
    global _lsl_recorder_subprocess
    if _lsl_recorder_subprocess is not None and _lsl_recorder_subprocess.poll() is None:
        print("Stopping Python LSL recorder subprocess...", file=sys.stderr)
        try:
            _lsl_recorder_subprocess.terminate()
            _lsl_recorder_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("LSL recorder did not terminate, killing...", file=sys.stderr)
            _lsl_recorder_subprocess.kill()
            _lsl_recorder_subprocess.wait()
        except Exception as e:
            print(f"Error cleaning up LSL recorder subprocess: {e}", file=sys.stderr)
        _lsl_recorder_subprocess = None


def _cleanup_labrecorder_rcs_recording() -> None:
    """Best-effort stop of LabRecorder recording via RCS.

    This is intentionally best-effort: if LabRecorder is not running or the
    RCS port is closed, we do not block shutdown.
    """
    global _labrecorder_rcs_recording_started, _labrecorder_rcs_host, _labrecorder_rcs_port

    if not _labrecorder_rcs_recording_started:
        return

    host = _labrecorder_rcs_host or "127.0.0.1"
    port = int(_labrecorder_rcs_port or 22345)
    try:
        _labrecorder_send_rcs(host, port, ["stop"], timeout_s=1.5)
    except Exception:
        pass
    finally:
        _labrecorder_rcs_recording_started = False
        _labrecorder_rcs_host = None
        _labrecorder_rcs_port = None


def _cleanup_mwl_subprocess() -> None:
    """Terminate MWL estimator/simulated subprocess if running."""
    global _mwl_subprocess
    if _mwl_subprocess is not None and _mwl_subprocess.poll() is None:
        print("Terminating MWL subprocess...", file=sys.stderr)
        try:
            _mwl_subprocess.terminate()
            _mwl_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("MWL subprocess did not terminate, killing...", file=sys.stderr)
            _mwl_subprocess.kill()
            _mwl_subprocess.wait()
        except Exception as e:
            print(f"Error cleaning up MWL subprocess: {e}", file=sys.stderr)
        _mwl_subprocess = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM by cleaning up subprocesses then re-raising."""
    _cleanup_mwl_subprocess()
    _cleanup_eda_subprocess()
    _cleanup_hr_subprocess()
    _cleanup_lsl_recorder_subprocess()
    _cleanup_labrecorder_rcs_recording()
    # Re-raise the signal to allow normal exit behavior
    sys.exit(128 + signum)


# Register cleanup handlers
atexit.register(_cleanup_mwl_subprocess)
atexit.register(_cleanup_eda_subprocess)
atexit.register(_cleanup_hr_subprocess)
atexit.register(_cleanup_lsl_recorder_subprocess)
atexit.register(_cleanup_labrecorder_rcs_recording)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _wait_for_markers_outlet(timeout_s: float = 15.0, poll_interval_s: float = 0.5) -> bool:
    """Block until an OpenMATB Markers LSL outlet is visible on the network.

    MATB creates its Markers outlet during initialisation, before the scenario
    clock starts.  Waiting here gives a reliable gate before restarting
    LabRecorder: the new recording will begin while MATB is showing the
    block-start dialog, ensuring all scenario events are captured.

    Returns True if found, False on timeout.
    """
    print(
        f"[REC] Waiting for MATB Markers outlet (type=Markers, timeout={timeout_s:.0f}s)...",
        flush=True,
    )
    import pylsl
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        streams = pylsl.resolve_stream("type", "Markers", 1, poll_interval_s)
        if streams:
            print(f"[REC] Markers outlet found: '{streams[0].name()}' — restarting LabRecorder.", flush=True)
            return True
    print(
        f"[REC] WARNING: Markers outlet not found within {timeout_s:.0f}s — "
        "LabRecorder will not be restarted; block timing will use scenario fallback offset.",
        flush=True,
    )
    return False


def _wait_for_mwl_outlet(timeout_s: float = 60.0, poll_interval_s: float = 1.0) -> bool:
    """Block until the MWL LSL outlet is visible on the network, or timeout.

    The MWL estimator creates its outlet only after successfully connecting to
    both EEG streams and loading the model.  Polling here gives a hard gate
    before OpenMATB starts: the scenario will not begin until the estimator is
    live and ready to push samples.

    Returns True if found, False on timeout (subprocess may have crashed).
    """
    print(
        f"[MWL] Waiting for MWL estimator outlet (type=MWL, timeout={timeout_s:.0f}s)...",
        flush=True,
    )
    import pylsl
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # Check subprocess is still alive first
        if _mwl_subprocess is not None and _mwl_subprocess.poll() is not None:
            rc = _mwl_subprocess.poll()
            print(
                f"[MWL] ERROR: MWL estimator exited prematurely (code={rc}). "
                "Check EEG streams are live.",
                flush=True,
            )
            return False
        streams = pylsl.resolve_stream("type", "MWL", 1, poll_interval_s)
        if streams:
            print(f"[MWL] MWL outlet found: '{streams[0].name()}' — proceeding.", flush=True)
            return True
    print(
        f"[MWL] ERROR: MWL outlet not found within {timeout_s:.0f}s — aborting scenario.",
        flush=True,
    )
    return False


def _labrecorder_send_rcs(host: str, port: int, lines: list[str], timeout_s: float = 3.0) -> None:
    payload = "".join(f"{line.strip()}\n" for line in lines if line and line.strip()).encode("utf-8")
    if not payload:
        return
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.sendall(payload)


def _parse_labrecorder_required_stream_spec(spec: str) -> dict:
    """Parse a required-stream spec into an _preflight_stream_check entry.

    Supported formats:
      - "StreamName"
      - "StreamName (HOST)"
      - "StreamName::Type" (e.g., "OpenMATB::Markers")
      - "StreamName (HOST)::Type"

    Returns a dict compatible with _preflight_stream_check.
    """
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty required stream spec")

    name_part = raw
    type_part: Optional[str] = None
    if "::" in raw:
        left, right = raw.split("::", 1)
        name_part = left.strip()
        type_part = right.strip() or None

    hostname: Optional[str] = None
    name = name_part
    m = re.match(r"^(?P<name>.+?)\s*\((?P<host>[^)]+)\)\s*$", name_part)
    if m:
        name = (m.group("name") or "").strip()
        hostname = (m.group("host") or "").strip() or None

    if not name and not type_part:
        raise ValueError(f"Invalid required stream spec (need name and/or type): {spec}")

    return {
        "name": name or None,
        "type": type_part,
        "hostname": hostname,
        "min_count": 1,
    }


def _labrecorder_expected_xdf_path(
    root: Path,
    template: str,
    participant: str,
    session: str,
    task: str,
    acquisition: str,
    modality: str,
    run: Optional[str],
) -> Path:
    mapped = template
    mapped = mapped.replace("%p", participant)
    mapped = mapped.replace("%s", session)
    mapped = mapped.replace("%b", task)
    mapped = mapped.replace("%a", acquisition)
    mapped = mapped.replace("%m", modality)
    if run is not None:
        mapped = mapped.replace("%r", str(run))
        mapped = mapped.replace("%n", str(run))

    mapped = mapped.replace("/", "\\")
    return (root / mapped).resolve()


def _start_labrecorder_xdf_recording_rcs(
    *,
    host: str,
    port: int,
    root: Path,
    template: str,
    participant: str,
    session: str,
    task: str,
    acquisition: str,
    modality: str,
    run: Optional[str] = None,
    timeout_s: float = 3.0,
) -> dict:
    """Start LabRecorder XDF recording via RCS.

    Returns:
        dict with keys:
          - started: bool
          - expected_xdf_path: Path
          - error: str | None
    """
    global _labrecorder_rcs_recording_started, _labrecorder_rcs_host, _labrecorder_rcs_port

    expected_path = _labrecorder_expected_xdf_path(
        root=root,
        template=template,
        participant=participant,
        session=session,
        task=task,
        acquisition=acquisition,
        modality=modality,
        run=run,
    )

    filename_cmd = (
        "filename "
        f"{{root:{str(root)}}} "
        f"{{template:{template}}} "
        f"{{participant:{participant}}} "
        f"{{session:{session}}} "
        f"{{task:{task}}} "
        f"{{acquisition:{acquisition}}} "
        f"{{modality:{modality}}}"
    )
    if run is not None:
        filename_cmd += f" {{run:{run}}}"

    try:
        # Typical robust sequence: refresh → select all → set filename → start.
        _labrecorder_send_rcs(
            host,
            port,
            [
                "update",
                "select all",
                filename_cmd,
                "start",
            ],
            timeout_s=timeout_s,
        )
    except OSError as exc:
        return {
            "started": False,
            "expected_xdf_path": expected_path,
            "error": f"Unable to connect to LabRecorder RCS at {host}:{port}: {exc}",
        }

    _labrecorder_rcs_recording_started = True
    _labrecorder_rcs_host = host
    _labrecorder_rcs_port = port
    return {"started": True, "expected_xdf_path": expected_path, "error": None}


def _stop_labrecorder_xdf_recording_rcs(*, host: str, port: int, timeout_s: float = 3.0) -> dict:
    global _labrecorder_rcs_recording_started, _labrecorder_rcs_host, _labrecorder_rcs_port

    try:
        _labrecorder_send_rcs(host, port, ["stop"], timeout_s=timeout_s)
    except OSError as exc:
        return {"stopped": False, "error": f"Unable to stop LabRecorder via RCS at {host}:{port}: {exc}"}
    finally:
        _labrecorder_rcs_recording_started = False
        _labrecorder_rcs_host = None
        _labrecorder_rcs_port = None

    return {"stopped": True, "error": None}


def _start_eda_streamer(
    repo_root: Path,
    eda_port: Optional[str],
    eda_stream_name: str = "ShimmerEDA",
    health_check_timeout: float = 15.0,
    min_battery_pct: Optional[float] = None,
    auto_port: bool = False,
) -> dict:
    """Start EDA streamer subprocess and verify LSL stream appears.

    If min_battery_pct is set, runs a quick --probe-json before spawning the
    full streamer and blocks if battery is below the threshold.

    Returns:
        dict with keys:
            - started: bool
            - pid: int or None
            - stream_name: str
            - stream_type: str
            - battery_pct: float or None
            - health_check_passed: bool
            - error: str or None
    """
    global _eda_subprocess

    result = {
        "started": False,
        "pid": None,
        "stream_name": eda_stream_name,
        "stream_type": "EDA",
        "battery_pct": None,
        "health_check_passed": False,
        "error": None,
    }

    # Build command to run EDA streamer
    streamer_script = repo_root / "scripts" / "stream_shimmer_eda.py"
    if not streamer_script.exists():
        result["error"] = f"EDA streamer script not found: {streamer_script}"
        return result

    # --- Battery pre-check ---
    if min_battery_pct is not None:
        if auto_port:
            print("Probing Shimmer battery (auto-detecting port)...")
            probe_cmd = [sys.executable, str(streamer_script), "--auto-port", "--probe-json"]
        else:
            print(f"Probing Shimmer battery (port {eda_port})...")
            probe_cmd = [sys.executable, str(streamer_script), "--port", eda_port, "--probe-json"]
        batt_info = _probe_device_battery(probe_cmd, timeout=45.0)
        result["battery_pct"] = batt_info["battery_pct"]

        if batt_info["ok"]:
            batt = batt_info["battery_pct"]
            if batt is not None:
                level_str = f"{batt:.0f}%"
                if batt < min_battery_pct:
                    result["error"] = (
                        f"Shimmer battery too low: {level_str} "
                        f"(minimum required: {min_battery_pct:.0f}%). "
                        f"Charge device before starting session."
                    )
                    return result
                print(f"  Shimmer battery: {level_str}")
                if batt < min_battery_pct + 10:
                    print(
                        f"  WARNING: Shimmer battery is low ({level_str}). "
                        f"Consider charging before a long session.",
                        file=sys.stderr,
                    )
            else:
                print("  Shimmer battery level not reported by device.")
        else:
            print(
                f"  WARNING: Could not read Shimmer battery: {batt_info.get('error', 'unknown')}",
                file=sys.stderr,
            )

    if auto_port:
        cmd = [
            sys.executable,
            str(streamer_script),
            "--auto-port",
            "--name", eda_stream_name,
        ]
    else:
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


def _start_hr_streamer(
    repo_root: Path,
    hr_device: Optional[str],
    hr_name_prefix: str = "Polar",
    health_check_timeout: float = 20.0,
    enable_ecg: bool = True,
    min_battery_pct: Optional[float] = None,
) -> dict:
    """Start Polar HR streamer subprocess and verify LSL streams appear.

    If min_battery_pct is set, runs a quick --probe-json before spawning the
    full streamer and blocks if battery is below the threshold.

    Returns dict with keys:
        started, pid, stream_names, battery_pct, health_check_passed, error.
    """
    global _hr_subprocess

    # HR and RR appear immediately on BLE connection.  ECG requires an extra
    # ~30 s of PMD protocol negotiation, so it is excluded from the health
    # check and verified only during the preflight check (as advisory).
    required_stream_names = [f"{hr_name_prefix}HR", f"{hr_name_prefix}RR"]
    all_stream_names = required_stream_names.copy()
    if enable_ecg:
        all_stream_names.append(f"{hr_name_prefix}ECG")

    result: dict = {
        "started": False,
        "pid": None,
        "stream_names": all_stream_names,
        "battery_pct": None,
        "health_check_passed": False,
        "error": None,
    }

    streamer_script = repo_root / "scripts" / "stream_polar_hr.py"
    if not streamer_script.exists():
        result["error"] = f"HR streamer script not found: {streamer_script}"
        return result

    # --- Battery pre-check ---
    if min_battery_pct is not None:
        print(f"Probing Polar H10 battery...")
        probe_cmd = [sys.executable, str(streamer_script), "--probe-json"]
        if hr_device:
            probe_cmd += ["--device", hr_device]
        batt_info = _probe_device_battery(probe_cmd, timeout=30.0)
        result["battery_pct"] = batt_info["battery_pct"]

        if batt_info["ok"]:
            batt = batt_info["battery_pct"]
            if batt is not None:
                level_str = f"{batt:.0f}%"
                if batt < min_battery_pct:
                    result["error"] = (
                        f"Polar H10 battery too low: {level_str} "
                        f"(minimum required: {min_battery_pct:.0f}%). "
                        f"Charge device before starting session."
                    )
                    return result
                print(f"  Polar H10 battery: {level_str}")
                if batt < min_battery_pct + 10:
                    print(
                        f"  WARNING: Polar H10 battery is low ({level_str}). "
                        f"Consider charging before a long session.",
                        file=sys.stderr,
                    )
            else:
                print("  Polar H10 battery level not reported by device.")
        else:
            print(
                f"  WARNING: Could not read Polar H10 battery: {batt_info.get('error', 'unknown')}",
                file=sys.stderr,
            )

    cmd = [
        sys.executable,
        str(streamer_script),
        "--prefix", hr_name_prefix,
    ]
    if hr_device:
        cmd += ["--device", hr_device]
    if not enable_ecg:
        cmd.append("--no-ecg")

    print(f"Starting HR streamer: {' '.join(cmd)}")

    try:
        _hr_subprocess = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        result["started"] = True
        result["pid"] = _hr_subprocess.pid
        print(f"HR streamer started (PID: {_hr_subprocess.pid})")
    except Exception as e:
        result["error"] = f"Failed to start HR streamer: {e}"
        return result

    # Wait for BLE scan + connect (can take several seconds)
    time.sleep(3.0)

    if _hr_subprocess.poll() is not None:
        stderr_output = _hr_subprocess.stderr.read() if _hr_subprocess.stderr else ""
        result["error"] = f"HR streamer exited immediately (code {_hr_subprocess.returncode}): {stderr_output}"
        result["started"] = False
        _hr_subprocess = None
        return result

    # Health check: verify HR and RR appear on the LSL network (ECG excluded —
    # it takes ~30 s of BLE PMD negotiation and is checked as advisory in preflight).
    print(f"Waiting for LSL HR/RR streams (timeout: {health_check_timeout}s)...")

    try:
        import pylsl

        start_time = time.time()
        found_names: set[str] = set()

        while time.time() - start_time < health_check_timeout:
            if _hr_subprocess.poll() is not None:
                stderr_output = _hr_subprocess.stderr.read() if _hr_subprocess.stderr else ""
                result["error"] = f"HR streamer crashed during health check: {stderr_output}"
                result["started"] = False
                _hr_subprocess = None
                return result

            try:
                discovered = pylsl.resolve_streams(wait_time=1.0)
            except Exception:
                discovered = []

            for info in discovered:
                if info.name() in required_stream_names:
                    found_names.add(info.name())

            if found_names >= set(required_stream_names):
                break

        if found_names >= set(required_stream_names):
            result["health_check_passed"] = True
            print(f"\u2713 HR/RR LSL streams confirmed: {sorted(found_names)}")
        else:
            missing = set(required_stream_names) - found_names
            result["error"] = (
                f"HR/RR streams not found within {health_check_timeout}s. "
                f"Missing: {sorted(missing)}"
            )

    except ImportError:
        result["error"] = "pylsl not installed; cannot verify HR streams"
    except Exception as e:
        result["error"] = f"LSL health check failed: {e}"

    return result


def _stop_hr_streamer() -> None:
    """Stop HR streamer subprocess."""
    _cleanup_hr_subprocess()


def _start_python_lsl_recorder(
    *,
    repo_root: Path,
    output_root_path: Path,
    participant: str,
    session: str,
    eda_stream_name: str,
    hr_stream_prefix: Optional[str] = None,
) -> dict:
    """Start Python LSL recorder subprocess for Pilot 1.

    Returns dict with keys: started, pid, recording_path, error.
    """
    global _lsl_recorder_subprocess, _lsl_recording_path

    result = {
        "started": False,
        "pid": None,
        "recording_path": None,
        "error": None,
    }

    recorder_script = repo_root / "scripts" / "record_lsl_streams.py"
    if not recorder_script.exists():
        result["error"] = f"Recorder script not found: {recorder_script}"
        return result

    recordings_dir = output_root_path / "physiology" / participant / session
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    recording_path = recordings_dir / f"lsl_recording_{timestamp_tag}.jsonl"

    cmd = [
        sys.executable,
        str(recorder_script),
        "--out", str(recording_path),
        "--include-type", "Markers",
        "--include-type", "EEG",
        "--include-type", "EDA",
        "--include-name", "OpenMATB",
        "--include-name", eda_stream_name,
    ]
    # Include Polar HR streams when managed by the runner
    if hr_stream_prefix:
        cmd += [
            "--include-type", "HR",
            "--include-type", "RR",
            "--include-type", "ECG",
            "--include-name", f"{hr_stream_prefix}HR",
            "--include-name", f"{hr_stream_prefix}RR",
            "--include-name", f"{hr_stream_prefix}ECG",
        ]

    print(f"Starting Python LSL recorder: {' '.join(cmd)}")

    try:
        _lsl_recorder_subprocess = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _lsl_recording_path = recording_path
    except Exception as e:
        result["error"] = f"Failed to start Python LSL recorder: {e}"
        return result

    time.sleep(1.0)
    if _lsl_recorder_subprocess.poll() is not None:
        stderr_output = _lsl_recorder_subprocess.stderr.read() if _lsl_recorder_subprocess.stderr else ""
        result["error"] = f"Python LSL recorder exited immediately (code {_lsl_recorder_subprocess.returncode}): {stderr_output}"
        _lsl_recorder_subprocess = None
        _lsl_recording_path = None
        return result

    result["started"] = True
    result["pid"] = _lsl_recorder_subprocess.pid
    result["recording_path"] = str(recording_path)
    return result


def _stop_python_lsl_recorder() -> Optional[Path]:
    """Stop Python LSL recorder and return recording path if available."""
    global _lsl_recording_path
    _cleanup_lsl_recorder_subprocess()
    path = _lsl_recording_path
    _lsl_recording_path = None
    return path


def _probe_device_battery(cmd: list, timeout: float = 40.0) -> dict:
    """Run a --probe-json subprocess and extract battery info from its JSON output.

    Returns dict with keys: ok, battery_pct (float|None), firmware (str|None),
    device_name_or_address (str|None), error (str|None).
    """
    result: dict = {
        "ok": False,
        "battery_pct": None,
        "firmware": None,
        "device_label": None,
        "error": None,
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Parse first JSON-looking line from stdout
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                result["ok"]         = bool(data.get("ok", False))
                result["battery_pct"] = data.get("battery_pct")  # float or None
                result["firmware"]    = data.get("firmware")
                # Polar uses 'address', Shimmer uses 'port'
                result["device_label"] = data.get("address") or data.get("port")
                return result
        rc = proc.returncode
        stderr_snippet = (proc.stderr or "")[:500]
        result["error"] = f"Probe exited with code {rc} and no JSON output. stderr: {stderr_snippet}"
    except subprocess.TimeoutExpired:
        result["error"] = f"Probe timed out after {timeout:.0f}s"
    except json.JSONDecodeError as exc:
        result["error"] = f"JSON parse error from probe: {exc}"
    except Exception as exc:
        result["error"] = str(exc)
    return result


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
        return {"all_found": False, "all_streaming": False, "streams": [], "warnings": [], "info": [], "error": "pylsl not installed"}
    
    results = {
        "all_found": True,
        "all_streaming": True,
        "streams": [],
        "warnings": [],
        "info": [],
        "error": None,
    }
    
    # Helper to suppress liblsl C++ logging during resolve calls
    def _resolve_quiet(prop: str, value: str, timeout: float, minimum: int = 1):
        """Resolve LSL streams while suppressing C library logging.

        For type/name searches we use resolve_streams(wait_time=timeout) and
        filter manually rather than resolve_byprop, because resolve_byprop
        returns as soon as `minimum` streams respond — meaning a second amp
        that responds slightly later is silently dropped.  resolve_streams
        collects everything that announces itself within the full wait window.
        """
        old_stderr_fd = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 2)
                all_streams = pylsl.resolve_streams(wait_time=timeout)
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)

        if prop == "name":
            return [s for s in all_streams if s.name() == value]
        elif prop == "type":
            return [s for s in all_streams if s.type() == value]
        else:
            return [s for s in all_streams if getattr(s, prop, lambda: None)() == value]

    def _stream_hostname(info) -> str:
        try:
            host = info.hostname()
        except Exception:
            host = ""
        return (host or "").strip()

    from contextlib import contextmanager

    @contextmanager
    def _suppress_stderr():
        """Redirect fd 2 to /dev/null for the duration of the block."""
        old_fd = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 2)
                yield
        finally:
            os.dup2(old_fd, 2)
            os.close(old_fd)

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
            "info": [],
        }
        
        try:
            # Suppress fd 2 for the entire inlet lifetime so that liblsl's
            # background reconnect thread (which fires asynchronously, including
            # after close_stream() returns) never leaks C++ ERR messages to the
            # terminal.  The extra sleep after close() gives the thread time to
            # flush before we restore fd 2.
            with _suppress_stderr():
                inlet = pylsl.StreamInlet(stream_info, max_buflen=int(duration * 2))
                inlet.open_stream(timeout=2.0)

                # Time correction: measure twice to detect instability.
                try:
                    tc1 = inlet.time_correction(timeout=1.0)
                    time.sleep(0.3)
                    tc2 = inlet.time_correction(timeout=1.0)
                    tc = tc2
                    stats["time_correction_ms"] = tc * 1000
                    jitter_ms = abs(tc2 - tc1) * 1000
                    if jitter_ms > 10:
                        stats["warnings"].append(
                            f"Clock offset unstable: jitter={jitter_ms:.1f}ms (offset={tc*1000:.1f}ms)"
                        )
                except Exception:
                    stats["warnings"].append("Could not get time correction")

                # Collect samples
                samples = []
                timestamps = []
                start_time = time.time()
                # Exit early once 5 samples are confirmed — high-rate streams
                # (EEG, ECG at 130+ Hz) are verified in <0.1 s rather than
                # waiting the full window.  Low-rate streams (HR at 1 Hz) still
                # naturally take their full window before 5 samples arrive.
                while time.time() - start_time < duration:
                    sample, ts = inlet.pull_sample(timeout=0.1)
                    if sample is not None:
                        samples.append(sample)
                        timestamps.append(ts)
                    if len(samples) >= 5:
                        break

                inlet.close_stream()
                # Brief pause so the liblsl reconnect thread can flush any
                # pending log output before fd 2 is restored.
                time.sleep(0.3)

            n_samples = len(samples)
            stats["samples_received"] = n_samples

            # For low-rate streams (HR ~1 Hz) a single sample in the window is
            # sufficient to confirm the stream is live.
            if n_samples >= 1:
                stats["streaming"] = True
                
                # Calculate actual sample rate.
                # Guard against burst-reads: when early-exit fires on a high-rate
                # stream, all 5 samples may arrive in microseconds (buffered from
                # the LSL queue), giving a meaningless inter-sample span.  Only
                # compute and compare the rate when the timestamp spread is at
                # least 0.5 s — enough for a credible estimate.
                elapsed = timestamps[-1] - timestamps[0]
                if elapsed >= 0.5:
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
                # The ShimmerEDA stream publishes GSR_RAW in raw 12-bit ADC counts (0–4095).
                # Plausible range for connected electrodes: 100–3900.
                # Near 0 → shorted / saturation.  Near 4095 → disconnected / very dry skin.
                if stream_info.type() == "EDA":
                    mean_val = float(np.mean(values))
                    out_of_range = values[(values < 100) | (values > 3900)]
                    if len(out_of_range) > 0:
                        stats["warnings"].append(
                            f"EDA out of range [100–3900 ADC counts]: "
                            f"mean={mean_val:.0f}, min={float(np.min(values)):.0f}, "
                            f"max={float(np.max(values)):.0f} — check electrode contact"
                        )
                    else:
                        stats["info"].append(
                            f"EDA range OK: mean={mean_val:.0f} ADC counts, "
                            f"min={float(np.min(values)):.0f}, max={float(np.max(values)):.0f} [100–3900]"
                        )

                # HR-specific range gate (50–200 bpm)
                if stream_info.type() == "HR":
                    mean_hr = float(np.mean(values))
                    out_of_range = values[(values < 50) | (values > 200)]
                    if len(out_of_range) > 0:
                        stats["warnings"].append(
                            f"HR out of range [50–200 bpm]: {out_of_range.tolist()} "
                            f"(check sensor contact / motion artefact)"
                        )
                    else:
                        stats["info"].append(
                            f"HR range OK: mean={mean_hr:.0f} bpm, "
                            f"min={float(np.min(values)):.0f}, max={float(np.max(values)):.0f} [50–200 bpm]"
                        )

                # RR-specific range gate (500–2000 ms)
                if stream_info.type() == "RR":
                    mean_rr = float(np.mean(values))
                    out_of_range = values[(values < 500) | (values > 2000)]
                    if len(out_of_range) > 0:
                        stats["warnings"].append(
                            f"RR out of range [500–2000 ms]: {out_of_range.tolist()} "
                            f"(check sensor contact / motion artefact)"
                        )
                    else:
                        stats["info"].append(
                            f"RR range OK: mean={mean_rr:.0f} ms, "
                            f"min={float(np.min(values)):.0f}, max={float(np.max(values)):.0f} [500–2000 ms]"
                        )

                # EEG-specific checks
                # EEG: the flat-signal check (std < 1e-9 above) is sufficient to detect
                # a dead amp. An absolute range threshold is unreliable because units vary
                # across amps (V vs µV), so no additional range check is applied here.
                    
            elif n_samples == 0:
                stats["warnings"].append("NO SAMPLES RECEIVED - stream may be stalled")
            else:
                # n_samples == 1: treat as streaming if stream is low-rate or irregular.
                # At 1 Hz (PolarHR) or irregular (PolarRR) a single sample in 2s is normal.
                nominal = stream_info.nominal_srate()
                stats["streaming"] = True
                if nominal > 2.0:  # only warn for higher-rate streams
                    stats["warnings"].append("Too few samples for quality analysis")
                # Still run range checks on the single available sample.
                values = np.array([samples[0][0]])
                if stream_info.type() == "HR":
                    out_of_range = values[(values < 50) | (values > 200)]
                    if len(out_of_range) > 0:
                        stats["warnings"].append(
                            f"HR out of range [50–200 bpm]: {out_of_range.tolist()} "
                            f"(check sensor contact / motion artefact)"
                        )
                    else:
                        stats["info"].append(
                            f"HR range OK: {float(values[0]):.0f} bpm [50–200 bpm]"
                        )
                if stream_info.type() == "RR":
                    out_of_range = values[(values < 500) | (values > 2000)]
                    if len(out_of_range) > 0:
                        stats["warnings"].append(
                            f"RR out of range [500–2000 ms]: {out_of_range.tolist()} "
                            f"(check sensor contact / motion artefact)"
                        )
                    else:
                        stats["info"].append(
                            f"RR range OK: {float(values[0]):.0f} ms [500–2000 ms]"
                        )
                
        except Exception as e:
            stats["warnings"].append(f"Sample test failed: {e}")
        
        return stats
    
    for expected in expected_streams:
        stream_name = expected.get("name")
        stream_type = expected.get("type")
        min_count = expected.get("min_count", 1)
        expected_host = (expected.get("hostname") or "").strip() or None
        label = stream_name or stream_type or "unknown"

        # Search by name (unique) or type (may return multiple amps)
        if stream_name:
            print(f"  Searching for stream '{stream_name}'...", end="", flush=True)
            found_streams = _resolve_quiet("name", stream_name, timeout_per_stream)
            if expected_host:
                found_streams = [s for s in found_streams if _stream_hostname(s) == expected_host]
            streams_to_test = found_streams[:1]  # name searches always expect exactly 1
        elif stream_type:
            suffix = f" (expecting {min_count})" if min_count > 1 else ""
            print(f"  Searching for stream type '{stream_type}'{suffix}...", end="", flush=True)
            found_streams = _resolve_quiet("type", stream_type, timeout_per_stream)
            if expected_host:
                found_streams = [s for s in found_streams if _stream_hostname(s) == expected_host]
            streams_to_test = found_streams
        else:
            print(f"  Skipping stream with no name or type")
            continue

        is_advisory = expected.get("advisory", False)
        if len(streams_to_test) < min_count:
            print(f" Found {len(streams_to_test)}/{min_count} — NOT ENOUGH")
            results["all_found"] = False
            # Advisory streams that are absent do not block the session.
            if not is_advisory:
                results["all_streaming"] = False
            results["streams"].append({
                "name": stream_name or label,
                "type": stream_type,
                "found": False,
                "advisory": is_advisory,
                "info": None,
                "sample_stats": None,
            })
            continue

        n_found = len(streams_to_test)
        if n_found > 1:
            print(f" Found {n_found}")
        else:
            print(f" Found: {streams_to_test[0].name()}")

        # For type-based multi-stream searches use short labels (EEG-A, EEG-B …)
        # For name-based searches keep the stream's own name.
        use_short_labels = (stream_type is not None and stream_name is None and n_found > 1)
        letter_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for idx, info in enumerate(streams_to_test):
            if use_short_labels:
                stream_label = f"{stream_type}-{letter_labels[idx]}"
            else:
                stream_label = info.name() or label
            stream_result = {
                "name": stream_label,  # actual amp name for summary
                "type": stream_type,
                "found": True,
                "info": {
                    "name": info.name(),
                    "type": info.type(),
                    "channel_count": info.channel_count(),
                    "nominal_srate": info.nominal_srate(),
                    "source_id": info.source_id(),
                },
                "sample_stats": None,
            }

            # Test sample flow
            # Low-rate streams (≤2 Hz or irregular) get a 5s window so we collect
            # enough samples for a meaningful rate estimate.  High-rate streams use
            # the default 2s window but exit early once 5 samples arrive (typically
            # <0.1 s for EEG/ECG at 130+ Hz).
            # Advisory streams (e.g. PolarECG) are "nice to have" — no data is not
            # a blocker and adds zero extra wait time.
            # is_advisory was set in the not-found check above; re-read here for clarity.
            nominal_for_duration = info.nominal_srate()
            if nominal_for_duration == 0 or nominal_for_duration <= 2.0:
                adaptive_duration = max(sample_test_duration, 5.0)
            else:
                adaptive_duration = sample_test_duration

            print(f"    Testing {stream_label} data flow ({adaptive_duration:.0f}s)...", end="", flush=True)
            sample_stats = _test_sample_flow(info, adaptive_duration)
            stream_result["sample_stats"] = sample_stats
            stream_result["advisory"] = is_advisory

            if sample_stats["streaming"]:
                rate = sample_stats["measured_rate_hz"]
                rate_str = f"@ {rate:.1f}Hz" if rate > 0 else "(rate n/a — burst read)"
                print(f" {sample_stats['samples_received']} samples {rate_str}")
            elif is_advisory:
                print(f" no data yet (advisory — will not block session)")
            else:
                print(f" NO SAMPLES!")
                results["all_streaming"] = False

            for warning in sample_stats.get("warnings", []):
                results["warnings"].append(f"{stream_label}: {warning}")
            for info_msg in sample_stats.get("info", []):
                results["info"].append(f"{stream_label}: {info_msg}")

            results["streams"].append(stream_result)
    
    return results


def _run_preflight_checks(
    check_eeg: bool = True,
    check_eda: bool = True,
    check_hr: bool = True,
    check_joystick: bool = True,
    eeg_stream_type: str = "EEG",
    eeg_stream_count: int = 1,
    eda_stream_name: str = "ShimmerEDA",
    hr_stream_prefix: str = "Polar",
    timeout: float = 1.0,
    battery_data: Optional[list] = None,
    extra_required_streams: Optional[list[dict]] = None,
) -> bool:
    """Run pre-flight LSL stream checks and prompt user to continue.

    battery_data: optional list of dicts with keys:
        label (str), pct (float|None), warn_below (float)
    e.g. [{"label": "Shimmer EDA", "pct": 72.0, "warn_below": 30.0}, ...]
    Battery levels below warn_below are surfaced as [!] warnings.
    """

    def _run_single_check():
        """Run one round of preflight checks. Returns (all_ok, has_warnings, result)."""
        print("\n" + "=" * 60)
        print("PRE-FLIGHT STREAM CHECK")
        print("=" * 60)

        # Check streams
        expected_streams = []
        if check_eeg:
            expected_streams.append({"type": eeg_stream_type, "name": None, "min_count": eeg_stream_count})
        if check_eda:
            expected_streams.append({"name": eda_stream_name, "type": "EDA"})
        if check_hr:
            expected_streams.append({"name": f"{hr_stream_prefix}HR",  "type": "HR"})
            expected_streams.append({"name": f"{hr_stream_prefix}RR",  "type": "RR"})
            # ECG is advisory: the Polar H10 takes ~30 s of BLE negotiation before
            # ECG data flows, but HR/RR are live immediately.  ECG is recorded
            # opportunistically — its absence does not block the session.
            expected_streams.append({"name": f"{hr_stream_prefix}ECG", "type": "ECG", "advisory": True})

        if extra_required_streams:
            expected_streams.extend(extra_required_streams)
        
        result = {"streams": [], "warnings": [], "info": [], "all_found": True, "all_streaming": True}
        
        if not expected_streams:
            print("\nNo streams to check.")
        else:
            print(f"\nChecking {len(expected_streams)} stream(s)...\n")
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
        
        # Stream status
        if expected_streams:
            for stream in result["streams"]:
                label = stream.get("name") or stream.get("type") or "unknown"
                if stream["found"]:
                    stats = stream.get("sample_stats", {})
                    if stats.get("streaming"):
                        rate = stats.get("measured_rate_hz", 0)
                        rate_str = f"{rate:.1f} Hz" if rate > 0 else "confirmed"
                        print(f"  [OK] {label}: STREAMING ({rate_str})")
                    elif stream.get("advisory"):
                        # Advisory streams that haven't started flowing yet are
                        # informational only — they don't block the session.
                        print(f"  [--] {label}: PRESENT (no data yet — advisory, will not block)")
                    else:
                        print(f"  [!!] {label}: FOUND but NO DATA")
                        all_ok = False
                elif stream.get("advisory"):
                    print(f"  [--] {label}: NOT FOUND (advisory — will not block)")
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

        # Informational signal checks (range gates etc.) — never block or prompt retry
        if result.get("info"):
            print("\n" + "-" * 40)
            print("SIGNAL QUALITY CHECKS")
            print("-" * 40)
            for info_msg in result["info"]:
                print(f"  [i] {info_msg}")
            print("-" * 40)

        # Device battery — shown as [OK]/[!] based on level
        if battery_data:
            print("\n" + "-" * 40)
            print("DEVICE BATTERY")
            print("-" * 40)
            for entry in battery_data:
                label     = entry.get("label", "Device")
                pct       = entry.get("pct")
                warn_below = entry.get("warn_below", 30.0)
                if pct is None:
                    print(f"  [?] {label}: unknown (not reported)")
                elif pct < warn_below:
                    print(f"  [!] {label}: {pct:.0f}% — LOW (charge before session)")
                    has_warnings = True
                else:
                    print(f"  [OK] {label}: {pct:.0f}%")
            print("-" * 40)

        print("=" * 60)
        
        return all_ok, has_warnings, result

    # Joystick check (once, before the retry loop — hardware doesn't change between retries)
    if check_joystick:
        print("\n" + "=" * 60)
        print("JOYSTICK CHECK")
        print("=" * 60)
        try:
            import pyglet
            joysticks = pyglet.input.get_joysticks()
            if joysticks:
                print(f"  [OK] Joystick detected: {joysticks[0].device.name}")
            else:
                print("  [!!] No joystick detected — tracking task requires a joystick.")
                print("       Connect the joystick now, then press Enter to continue, or 'n' to cancel.")
                resp = input("  Continue without joystick? (y/n): ").strip().lower()
                if resp in ("n", "no"):
                    return False
        except Exception as exc:
            print(f"  [?] Joystick check skipped: {exc}")

    # EEG amplifier battery (manual — no LSL battery stream exists for the amp).
    # This is the only battery that cannot be checked automatically.
    print("\n" + "=" * 60)
    print("EEG AMPLIFIER BATTERY CHECK")
    print("=" * 60)
    print("  The ANTneuro amplifier battery cannot be checked automatically.")
    print("  Confirm the charge level in eego software or on the hardware LED.")
    while True:
        amp_ok = input("\n  EEG amplifier battery adequate? (y/n): ").strip().lower()
        if amp_ok in ("y", "yes"):
            break
        elif amp_ok in ("n", "no"):
            print("\nSession cancelled: EEG amplifier battery not confirmed.", file=sys.stderr)
            return False
        else:
            print("  Please enter y or n.")

    # Run stream checks in a retry loop
    while True:
        all_ok, has_warnings, result = _run_single_check()
        
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


def _save_assignments(repo_root: Path, assignments: dict, skip_write: bool = False) -> None:
    """Save participant assignments to config file."""
    if skip_write:
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


def _update_manifest_metadata(
    manifest_path: Path,
    *,
    scenario_filename: str,
    abort_reason: Optional[str] = None,
) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["unattended"] = False
    if abort_reason:
        manifest["abort_reason"] = abort_reason
    if scenario_filename:
        manifest["scenario_name"] = Path(scenario_filename).stem
        openmatb_meta = manifest.get("openmatb")
        if not isinstance(openmatb_meta, dict):
            openmatb_meta = {}
            manifest["openmatb"] = openmatb_meta
        openmatb_meta["scenario_path"] = scenario_filename

    _atomic_write_json(manifest_path, manifest)



def _block_dialog_lines(
    scenario_filename: str,
    block_index: int,
    participant: str,
    session: str,
) -> list[str]:
    """Build participant-friendly modal-dialog lines for a block start.

    Three message variants:
      - Familiarisation (practice intro): IDs only (no instructional text).
      - Practice blocks: reassuring, emphasises learning.
      - Calibration blocks: neutral, mentions TLX afterwards.

    Full IDs are shown only on the first block (block_index == 0);
    subsequent blocks get a compact one-line ID footer.
    """
    is_intro = "pilot_practice_intro" in scenario_filename
    is_practice = "pilot_practice" in scenario_filename and not is_intro
    is_rest = "rest_baseline" in scenario_filename

    if is_rest:
        # Rest baseline: just show IDs, no instructional body text.
        body = [
            "Rest baseline",
            "",
            "Fixation cross — 2 minutes.",
            "Press OK when the participant is ready.",
        ]
        body.append("")
        body.append(f"Participant: {participant}")
        body.append(f"Session: {session}")
        return body

    if is_intro:
        # Familiarisation phase — IDs only, no instructional body text.
        body = [
            f"Participant: {participant}",
            f"Session: {session}",
        ]
        return body

    if is_practice:
        body = [
            "Practice block",
            "",
            "This is a practice round to help you get",
            "comfortable with the tasks.",
            "Work at a steady pace \u2014 there is no scoring.",
            "",
            "Take a short break if you need one.",
            "Press OK when you are ready to begin.",
        ]
    else:
        body = [
            "Calibration block",
            "",
            "Please do your best, but remember",
            "there is no pass or fail.",
            "After this block you will complete",
            "a short questionnaire.",
            "",
            "Take a short break if you need one.",
            "Press OK when you are ready to begin.",
        ]

    # Separator before IDs
    body.append("")

    if block_index == 0:
        body.append(f"Participant: {participant}")
        body.append(f"Session: {session}")
    else:
        body.append(f"ID: {participant} \u00b7 {session}")

    return body


def _stage_pilot_instruction_files(openmatb_dir: Path, repo_root: Path) -> None:
    """Copy repo-managed pilot instruction text files into OpenMATB includes.

    OpenMATB validates blocking-plugin filenames by requiring them to exist under
    includes/instructions/ or includes/questionnaires/.
    """

    source_dir = repo_root / "experiment" / "instructions"
    required_names = [
        "1_welcome.txt",
        "2_sysmon.txt",
        "3_track.txt",
        "4_comm.txt",
        "5_resman.txt",
        "6_all_tasks.txt",
        "rest_fixation.txt",
    ]

    missing = [name for name in required_names if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required pilot instruction files under <repo>/experiment/instructions: " + ", ".join(missing)
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


def _cleanup_openmatb_staged_assets(openmatb_dir: Path, scenario_filename: str) -> None:
    """Remove wrapper-staged files from the vendor OpenMATB tree.

    The wrapper stages repo-managed assets into the vendor submodule at runtime
    (includes/scenarios + includes/instructions) to satisfy OpenMATB validation.
    We then remove them again so the submodule working tree does not remain dirty.
    """

    try:
        scenario_target_path = openmatb_dir / "includes" / "scenarios" / scenario_filename
        if scenario_target_path.exists():
            scenario_target_path.unlink()
    except Exception:
        pass

    try:
        pilot_instructions_dir = openmatb_dir / "includes" / "instructions" / "pilot_en"
        if pilot_instructions_dir.exists():
            shutil.rmtree(pilot_instructions_dir, ignore_errors=True)
    except Exception:
        pass


def _run_single_scenario(
    openmatb_dir: Path,
    scenario_filename: str,
    output_root_path: Path,
    participant: str,
    session: str,
    args: argparse.Namespace,
    repo_commit: str,
    submodule_commit: str,
    *,
    block_index: int = 0,
    adaptation_mode: bool = False,
    adaptation_seed: Optional[int] = None,
    mwl_adaptation_mode: bool = False,
    mwl_simulated_mode: Optional[str] = None,
    mwl_model_dir: Optional[str] = None,
    mwl_audit_csv: Optional[str] = None,
    mwl_threshold: Optional[float] = None,
) -> tuple[int, Optional[Path]]:
    repo_root = Path(__file__).resolve().parents[1]

    try:
        _stage_pilot_instruction_files(openmatb_dir, repo_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2, None

    scenario_source_path = repo_root / "experiment" / "scenarios" / scenario_filename
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
    env["OPENMATB_REPO_COMMIT"] = repo_commit
    env["OPENMATB_SUBMODULE_COMMIT"] = submodule_commit

    # Block-start dialog metadata
    env["OPENMATB_BLOCK_INDEX"] = str(block_index)
    dialog_lines = _block_dialog_lines(
        scenario_filename, block_index, participant, session,
    )
    env["OPENMATB_BLOCK_DIALOG_LINES"] = "\n".join(dialog_lines)
    # Suppress interactive dialog for automated / verification runs
    if getattr(args, "verification", False):
        env["OPENMATB_SUPPRESS_BLOCK_DIALOG"] = "1"

    # Calculate paths specifically for this run
    scenario_rel_path = scenario_filename
    sessions_dir = output_root_path / Path(env["OPENMATB_OUTPUT_SUBDIR"]) / "sessions"
    manifests_before = _list_manifest_paths(sessions_dir)

    # Precompute threshold kwarg for MwlAdaptationConfig bootstrap injection.
    _mwl_thr_kwarg = f", threshold={mwl_threshold!r}" if mwl_threshold is not None else ""

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
        "    if os.environ.get('OPENMATB_SUPPRESS_BLOCK_DIALOG') == '1':\n"
        "        return\n"
        "    if not REPLAY_MODE and get_conf_value('Openmatb', 'display_session_number'):\n"
        "        raw = os.environ.get('OPENMATB_BLOCK_DIALOG_LINES', '')\n"
        "        if raw:\n"
        "            msg = raw.split('\\n')\n"
        "        else:\n"
        "            pid = os.environ.get('OPENMATB_PARTICIPANT') or 'UNKNOWN'\n"
        "            sid = os.environ.get('OPENMATB_SESSION') or 'UNKNOWN'\n"
        "            msg = [f'Participant: {pid}', f'Session: {sid}']\n"
        "        self.modal_dialog = ModalDialog(self, msg, 'OpenMATB')\n"
        "Window.display_session_id = _display_session_id\n"
        # -------------------------------------------------------
        # Adaptation mode: monkey-patch Scheduler → AdaptationScheduler
        # -------------------------------------------------------
        + (
            "# --- Online staircase calibration ---\n"
            "try:\n"
            f"    import sys as _sys; _sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "    from adaptation.adaptation_scheduler import AdaptationScheduler, AdaptationConfig\n"
            f"    _adapt_cfg = AdaptationConfig(seed={adaptation_seed!r})\n"
            "    AdaptationScheduler._ADAPT_CFG = _adapt_cfg\n"
            "    import core as _core_mod, core.scheduler as _sched_mod\n"
            "    _sched_mod.Scheduler = AdaptationScheduler\n"
            "    _core_mod.Scheduler = AdaptationScheduler\n"
            "except Exception as _adapt_exc:\n"
            "    import traceback; traceback.print_exc()\n"
            "    raise RuntimeError(f'AdaptationScheduler injection failed: {_adapt_exc}') from _adapt_exc\n"
            if adaptation_mode else ""
        )
        # -------------------------------------------------------
        # MWL adaptation mode: monkey-patch Scheduler → MwlAdaptationScheduler
        # -------------------------------------------------------
        + (
            "# --- MWL-driven adaptation ---\n"
            "try:\n"
            f"    import sys as _sys; _sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "    from adaptation.mwl_adaptation_scheduler import MwlAdaptationScheduler, MwlAdaptationConfig\n"
            f"    _mwl_cfg = MwlAdaptationConfig(seed={adaptation_seed!r}"
            f", audit_csv_path={mwl_audit_csv!r}{_mwl_thr_kwarg})\n"
            "    MwlAdaptationScheduler._MWL_CFG = _mwl_cfg\n"
            "    import core as _core_mod, core.scheduler as _sched_mod\n"
            "    _sched_mod.Scheduler = MwlAdaptationScheduler\n"
            "    _core_mod.Scheduler = MwlAdaptationScheduler\n"
            "except Exception as _mwl_exc:\n"
            "    import traceback; traceback.print_exc()\n"
            "    raise RuntimeError(f'MwlAdaptationScheduler injection failed: {_mwl_exc}') from _mwl_exc\n"
            if mwl_adaptation_mode else ""
        )
        + "runpy.run_path('main.py', run_name='__main__')\n"
    )

    # Start MWL subprocess (simulated or real estimator) before OpenMATB
    global _mwl_subprocess
    if mwl_adaptation_mode and mwl_simulated_mode:
        mwl_cmd = [
            sys.executable, "-m", "mwl_simulated",
            "--mode", mwl_simulated_mode,
            "--rate", "4",
        ]
        print(f"[MWL] Starting simulated source: {' '.join(mwl_cmd)}", flush=True)
        _mwl_subprocess = subprocess.Popen(
            mwl_cmd,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
        )
        time.sleep(1.0)  # let LSL outlet register
    elif mwl_adaptation_mode and mwl_model_dir:
        _repo_root = Path(__file__).resolve().parent.parent
        mwl_cmd = [
            sys.executable, "-m", "mwl_estimator",
            "--model-dir", mwl_model_dir,
            "--eeg-config", str(_repo_root / "config" / "eeg_metadata.yaml"),
            "--region-config", str(_repo_root / "config" / "eeg_feature_extraction.yaml"),
        ]
        print(f"[MWL] Starting real estimator: {' '.join(mwl_cmd)}", flush=True)
        _mwl_subprocess = subprocess.Popen(
            mwl_cmd,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
        )
        # Block until the MWL outlet is visible — ensures both EEG streams are
        # connected and the estimator is ready before the scenario begins.
        if not _wait_for_mwl_outlet(timeout_s=90.0):
            _cleanup_mwl_subprocess()
            return 1, None

        # Restart LabRecorder recording now that the MWL outlet is on the network.
        # The session-level "select all" fired before the MWL estimator started,
        # so MWL was not included.  Stop + re-select + start ensures the XDF
        # captures the MWL stream for the full scenario.
        if getattr(args, "labrecorder_rcs", False) and _labrecorder_rcs_recording_started:
            lr_host = str(getattr(args, "labrecorder_host", "127.0.0.1"))
            lr_port = int(getattr(args, "labrecorder_port", 22345))
            print("[MWL] Restarting LabRecorder to include MWL outlet in XDF...", flush=True)
            try:
                _stop_labrecorder_xdf_recording_rcs(host=lr_host, port=lr_port)
                time.sleep(0.5)
                _start_labrecorder_xdf_recording_rcs(
                    host=lr_host,
                    port=lr_port,
                    root=Path(str(args.labrecorder_root)).resolve() if getattr(args, "labrecorder_root", None) else (output_root_path / "physiology"),
                    template=str(args.labrecorder_template),
                    participant=participant,
                    session=session,
                    task=str(args.labrecorder_task),
                    acquisition=str(args.labrecorder_acq),
                    modality=str(args.labrecorder_modality),
                    run=str(args.labrecorder_run) if getattr(args, "labrecorder_run", None) else None,
                )
                print("[MWL] LabRecorder restarted — MWL outlet will be recorded.", flush=True)
            except Exception as _lr_exc:
                print(f"[MWL] WARNING: LabRecorder restart failed: {_lr_exc}", flush=True)

    try:
        # Pass the modified environment
        proc = subprocess.Popen([sys.executable, "-c", bootstrap], cwd=str(openmatb_dir), env=env)

        # Restart LabRecorder now that the MATB Markers LSL outlet is live.
        # The session-level "select all" fired in main() before MATB launched, so
        # the Markers stream was absent and not included in the initial recording.
        # Stopping and restarting here ensures the XDF captures all scenario events.
        # MATB shows the block-start dialog before t=0, giving enough time for the
        # restart to complete before any scenario markers are sent.
        if getattr(args, "labrecorder_rcs", False) and _labrecorder_rcs_recording_started:
            if _wait_for_markers_outlet(timeout_s=15.0):
                lr_host = str(getattr(args, "labrecorder_host", "127.0.0.1"))
                lr_port = int(getattr(args, "labrecorder_port", 22345))
                try:
                    _stop_labrecorder_xdf_recording_rcs(host=lr_host, port=lr_port)
                    time.sleep(0.5)
                    _start_labrecorder_xdf_recording_rcs(
                        host=lr_host,
                        port=lr_port,
                        root=Path(str(args.labrecorder_root)).resolve() if getattr(args, "labrecorder_root", None) else (output_root_path / "physiology"),
                        template=str(args.labrecorder_template),
                        participant=participant,
                        session=session,
                        task=str(args.labrecorder_task),
                        acquisition=str(args.labrecorder_acq),
                        modality=str(args.labrecorder_modality),
                        run=str(args.labrecorder_run) if getattr(args, "labrecorder_run", None) else None,
                    )
                    print("[REC] LabRecorder restarted — MATB Markers stream will be recorded.", flush=True)
                except Exception as _lr_exc:
                    print(f"[REC] WARNING: LabRecorder restart failed: {_lr_exc}", flush=True)

        exit_code = proc.wait()
    finally:
        _cleanup_mwl_subprocess()
        # Best-effort cleanup: do not leave the vendor submodule dirty.
        _cleanup_openmatb_staged_assets(openmatb_dir, scenario_filename)

    # Post-process manifests
    if exit_code != 0:
        manifests_after = _list_manifest_paths(sessions_dir)
        new_manifests = sorted(manifests_after - manifests_before)
        if len(new_manifests) == 1:
            _update_manifest_metadata(
                new_manifests[0],
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
    _update_manifest_metadata(
        manifest_path,
        scenario_filename=scenario_filename,
        abort_reason=None,
    )

    if getattr(args, "summarise_performance", False):
        try:
            summariser = (repo_root / "src" / "performance" / "summarise_openmatb_performance.py").resolve()
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
            _update_manifest_metadata(
                manifest_path,
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
                _update_manifest_metadata(
                    manifest_path,
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


def _ensure_performance_summaries(repo_root: Path, scenario_manifests: list[Path]) -> None:
    """Best-effort: ensure *.performance_summary.json exists next to each scenario manifest."""

    summariser = (repo_root / "src" / "performance" / "summarise_openmatb_performance.py").resolve()
    if not summariser.exists():
        return

    for manifest_path in scenario_manifests:
        try:
            summary_path = Path(str(manifest_path) + ".performance_summary.json")
            if summary_path.exists():
                continue
            subprocess.run(
                [sys.executable, str(summariser), "--manifest", str(manifest_path)],
                check=False,
                cwd=str(repo_root),
            )
        except Exception:
            # Best-effort; never crash an attended run.
            continue


def _auto_export_pilot_results(
    *,
    repo_root: Path,
    output_root_path: Path,
    participant: str,
    session: str,
    run_manifest_path: Path,
    scenario_manifests: list[Path],
) -> None:
    """Best-effort external-only exports for operator-friendly readiness tracking."""

    try:
        _ensure_performance_summaries(repo_root, scenario_manifests)
    except Exception:
        pass

    # Export per-run results summary table
    try:
        import sys as _sys
        _perf_dir = str(Path(__file__).resolve().parent / "performance")
        if _perf_dir not in _sys.path:
            _sys.path.insert(0, _perf_dir)
        import export_pilot_performance_table as exporter

        session_root = output_root_path / "openmatb" / participant / session
        derived_dir = session_root / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        tag = run_manifest_path.stem.replace("run_manifest_", "")
        out_csv = derived_dir / f"pilot_results_summary_{tag}.csv"

        written = exporter.export_table(run_manifest_path=run_manifest_path, out_csv=out_csv)
        print(f"Wrote pilot results summary CSV: {written}")
    except Exception as exc:
        print(f"WARNING: Could not export pilot results summary CSV: {exc}", file=sys.stderr)

    # Update cohort-level status tables (Option A: always attempt)
    try:
        import sys as _sys
        _perf_dir = str(Path(__file__).resolve().parent / "performance")
        if _perf_dir not in _sys.path:
            _sys.path.insert(0, _perf_dir)
        import summarise_pilot_cohort_status as cohort

        blocks_path, status_path = cohort.summarise(output_root_path)
        print(f"Updated cohort block table: {blocks_path}")
        print(f"Updated cohort status table: {status_path}")
    except Exception as exc:
        print(f"WARNING: Could not update cohort status tables: {exc}", file=sys.stderr)


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
        help="Path to OpenMATB directory (default: <repo>/src/vendor/openmatb).",
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
        "--calibration-trend",
        action="store_true",
        help=(
            "Run calibration blocks in fixed order LOW → MODERATE → HIGH (for within-session trend checks), "
            "overriding counterbalancing order."
        ),
    )

    parser.add_argument(
        "--only-scenario",
        default=None,
        help=(
            "Run exactly one scenario file (from <repo>/scenarios), overriding the normal playlist. "
            "Example: --only-scenario pilot_calibration_high.txt"
        ),
    )

    parser.add_argument(
        "--pilot1",
        action="store_true",
        help=(
            "Enable Pilot 1 checks with physiology recording and marker-alignment QC. "
            "If --xdf-path is not provided, a Python LSL recorder is started automatically."
        ),
    )

    parser.add_argument(
        "--xdf-path",
        default=None,
        help=(
            "Path to a pre-recorded physiology artifact for this run (.xdf or .jsonl). "
            "Optional for --pilot1 when using auto Python recorder."
        ),
    )

    parser.add_argument(
        "--no-python-recorder",
        action="store_true",
        help="Disable auto-starting Python LSL recorder for --pilot1 when --xdf-path is omitted.",
    )

    parser.add_argument(
        "--labrecorder-rcs",
        action="store_true",
        help=(
            "Control LabRecorder recording via its Remote Control Socket (RCS) for --pilot1. "
            "When enabled, run_openmatb will auto-start/stop the XDF recording and auto-compute the expected --xdf-path. "
            "Requires LabRecorder to be running with RCSEnabled=1."
        ),
    )
    parser.add_argument(
        "--labrecorder-host",
        default="127.0.0.1",
        help="LabRecorder RCS host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--labrecorder-port",
        type=int,
        default=22345,
        help="LabRecorder RCS port (default: 22345)",
    )
    parser.add_argument(
        "--labrecorder-root",
        default=None,
        help=(
            "Root directory for LabRecorder recordings when using --labrecorder-rcs. "
            "Default: <output_root>/physiology"
        ),
    )
    parser.add_argument(
        "--labrecorder-template",
        default=r"sub-%p\\ses-%s\\physio\\sub-%p_ses-%s_task-%b_acq-%a_%m.xdf",
        help=(
            "BIDS-like path template relative to --labrecorder-root. "
            "Supports placeholders %%p %%s %%b %%a %%m (and optionally %%r/%%n if used with --labrecorder-run)."
        ),
    )
    parser.add_argument(
        "--labrecorder-task",
        default="matb",
        help="Value for %%b (task/block) in LabRecorder template (default: matb)",
    )
    parser.add_argument(
        "--labrecorder-acq",
        default="pilot1",
        help="Value for %%a (acquisition) in LabRecorder template (default: pilot1)",
    )
    parser.add_argument(
        "--labrecorder-modality",
        default="physio",
        help="Value for %%m (modality) in LabRecorder template (default: physio)",
    )
    parser.add_argument(
        "--labrecorder-run",
        default=None,
        help="Optional run index for %%r/%%n in LabRecorder template.",
    )

    parser.add_argument(
        "--labrecorder-required-stream",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Additional required LSL streams to check in preflight when preparing a LabRecorder session. "
            "Repeatable. SPEC formats: 'StreamName', 'StreamName (HOST)', or 'StreamName::Type'."
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
        help="Don't update participant_assignments.yaml.",
    )

    # EDA streamer
    parser.add_argument(
        "--eda-port",
        default=None,
        help="Shimmer Bluetooth COM port (e.g., COM5). Use --eda-auto-port to detect automatically.",
    )
    parser.add_argument(
        "--eda-auto-port",
        action="store_true",
        help="Auto-detect the Shimmer COM port by scanning available serial ports. Overrides --eda-port.",
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
    parser.add_argument(
        "--eda-min-battery",
        type=float,
        default=25.0,
        metavar="PCT",
        help="Minimum Shimmer battery %% required to start a session (default: 25). Set 0 to disable.",
    )

    # Polar H10 HR streamer
    parser.add_argument(
        "--hr-device",
        default=None,
        metavar="ADDRESS",
        help=(
            "BLE address of the Polar H10 (e.g. XX:XX:XX:XX:XX:XX). "
            "If omitted, the first Polar device found during BLE scanning is used."
        ),
    )
    parser.add_argument(
        "--hr-name-prefix",
        default="Polar",
        metavar="PREFIX",
        help="LSL stream name prefix for HR streams (default: Polar → PolarHR, PolarRR, PolarECG).",
    )
    parser.add_argument(
        "--hr-health-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for Polar HR LSL streams before failing (default: 20). BLE scan can take ~10s.",
    )
    parser.add_argument(
        "--hr-min-battery",
        type=float,
        default=20.0,
        metavar="PCT",
        help="Minimum Polar H10 battery %% required to start a session (default: 20). Set 0 to disable.",
    )
    parser.add_argument(
        "--hr-ecg",
        action="store_true",
        help="Enable raw ECG stream from the Polar H10 (HR+RR only by default; ECG causes BLE disconnect on some firmware).",
    )
    parser.add_argument(
        "--skip-stream-check",
        action="store_true",
        help="Skip pre-flight stream and sensor checks (for automated/verification runs only).",
    )
    parser.add_argument(
        "--eeg-stream-type",
        default="EEG",
        help="LSL stream type to search for EEG (default: EEG).",
    )
    parser.add_argument(
        "--eeg-stream-count",
        type=int,
        default=1,
        help="Number of EEG LSL streams to expect in preflight check (default: 1). Use 2 for dual-amp setups.",
    )

    # Online staircase calibration
    parser.add_argument(
        "--adaptation",
        action="store_true",
        help=(
            "Enable online staircase calibration. "
            "Injects AdaptationScheduler into the OpenMATB process; "
            "use with --only-scenario adaptation_skeleton.txt for a standalone test run."
        ),
    )
    parser.add_argument(
        "--adaptation-seed",
        type=int,
        default=0,
        metavar="SEED",
        help="RNG seed for staircase and Poisson event generators (default: 0).",
    )

    # MWL-driven adaptation
    parser.add_argument(
        "--mwl-adaptation",
        action="store_true",
        help=(
            "Enable MWL-driven adaptive difficulty. "
            "Mutually exclusive with --adaptation (staircase). "
            "Reads real-time MWL scores from an LSL stream and adjusts MATB difficulty."
        ),
    )
    parser.add_argument(
        "--mwl-simulated",
        default=None,
        metavar="MODE",
        help=(
            "Launch a simulated MWL source as a subprocess before OpenMATB. "
            "MODE is passed to mwl_simulated.py --mode (e.g. block, constant, sinusoid). "
            "Use with --mwl-adaptation for desk testing without EEG hardware."
        ),
    )
    parser.add_argument(
        "--mwl-model-dir",
        default=None,
        metavar="DIR",
        help=(
            "Path to model artefacts directory (pipeline.pkl, selector.pkl, norm_stats.json). "
            "When provided with --mwl-adaptation, launches the real MWL estimator as a subprocess."
        ),
    )
    parser.add_argument(
        "--mwl-audit-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path for the MWL adaptation audit CSV. "
            "If omitted, defaults to <output_root>/<participant>/<session>/mwl_audit.csv."
        ),
    )
    parser.add_argument(
        "--mwl-threshold",
        type=float,
        default=None,
        metavar="THR",
        help=(
            "MWL decision threshold (0-1). Overrides the MwlAdaptationConfig default (0.50). "
            "Set automatically from the participant's Youden J threshold in model_config.json."
        ),
    )

    args = parser.parse_args()

    extra_required_streams: list[dict] = []
    if getattr(args, "labrecorder_required_stream", None):
        for spec in args.labrecorder_required_stream:
            try:
                extra_required_streams.append(_parse_labrecorder_required_stream_spec(spec))
            except ValueError as exc:
                print(f"ERROR: invalid --labrecorder-required-stream {spec!r}: {exc}", file=sys.stderr)
                return 2

    if args.only_scenario and args.calibration_trend:
        print("ERROR: --only-scenario cannot be combined with --calibration-trend.", file=sys.stderr)
        return 2

    if args.adaptation and args.mwl_adaptation:
        print("ERROR: --adaptation and --mwl-adaptation are mutually exclusive.", file=sys.stderr)
        return 2

    if args.mwl_simulated and args.mwl_model_dir:
        print("ERROR: --mwl-simulated and --mwl-model-dir are mutually exclusive.", file=sys.stderr)
        return 2

    if (args.mwl_simulated or args.mwl_model_dir) and not args.mwl_adaptation:
        print("ERROR: --mwl-simulated / --mwl-model-dir require --mwl-adaptation.", file=sys.stderr)
        return 2

    # --- MWL artefact pre-check ---
    if args.mwl_adaptation and args.mwl_model_dir and not args.mwl_simulated:
        _model_dir = Path(args.mwl_model_dir)
        _required = ["pipeline.pkl", "selector.pkl", "norm_stats.json"]
        _missing = [f for f in _required if not (_model_dir / f).exists()]
        if _missing:
            print(
                f"ERROR: MWL model artefacts missing from {_model_dir}:\n"
                + "".join(f"  - {f}\n" for f in _missing)
                + "Run 'python scripts/calibrate_participant.py calibrate' first.",
                file=sys.stderr,
            )
            return 2

    if args.mwl_adaptation and args.mwl_audit_csv:
        _audit_parent = Path(args.mwl_audit_csv).parent
        if not _audit_parent.exists():
            try:
                _audit_parent.mkdir(parents=True, exist_ok=True)
                print(f"Created audit output directory: {_audit_parent}")
            except OSError as exc:
                print(f"ERROR: Cannot create audit CSV directory {_audit_parent}: {exc}", file=sys.stderr)
                return 2

    # --pilot1 implies a dual-amp EEG setup; bump the stream count unless the user
    # has explicitly requested a higher value via --eeg-stream-count.
    if args.pilot1 and args.eeg_stream_count < 2:
        args.eeg_stream_count = 2

    if args.speed != 1 and not args.verification:
        print(
            f"NOTE: Ignoring --speed={args.speed} because this is an attended run. "
            "Use --verification to enable fast-forward.",
            file=sys.stderr,
        )
        args.speed = 1

    repo_root = Path(__file__).resolve().parents[1]
    openmatb_dir = Path(args.openmatb_dir) if args.openmatb_dir else repo_root / "src" / "vendor" / "openmatb"

    if not openmatb_dir.exists():
        print(f"OpenMATB directory not found: {openmatb_dir}", file=sys.stderr)
        return 2

    try:
        _ensure_openmatb_runtime_dependencies(openmatb_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    
    # Subprocess state
    eda_info: Optional[dict] = None
    hr_info:  Optional[dict] = None
    lsl_recorder_info: Optional[dict] = None
    labrecorder_info: Optional[dict] = None
    physiology_recording_path: Optional[Path] = Path(args.xdf_path) if args.xdf_path else None

    # Load participant assignments
    assignments = _load_assignments(repo_root)
    participants_data = assignments.get("participants", {})

    # Preflight runs for every attended session (all sensors are always required).
    # Use --skip-stream-check only for automated/verification runs.
    run_preflight = not args.verification and not args.skip_stream_check

    # Start EDA streamer before preflight so the stream check can verify it is live.
    # If neither --eda-port nor --eda-auto-port is given, default to auto-detect.
    _eda_auto_port = getattr(args, "eda_auto_port", False) or not args.eda_port
    if run_preflight and (args.eda_port or _eda_auto_port):
        print("\n" + "=" * 60)
        print("Starting EDA streamer before preflight checks...")
        print("=" * 60)

        eda_info = _start_eda_streamer(
            repo_root=repo_root,
            eda_port=args.eda_port,
            eda_stream_name=args.eda_stream_name,
            health_check_timeout=args.eda_health_timeout,
            min_battery_pct=args.eda_min_battery if args.eda_min_battery > 0 else None,
            auto_port=_eda_auto_port,
        )

        if not eda_info["health_check_passed"]:
            print(f"\n[!] EDA streamer did not start: {eda_info.get('error', 'unknown error')}", file=sys.stderr)
            print("    EDA will appear as NOT FOUND in preflight — use [r] to retry after fixing.", file=sys.stderr)
            _stop_eda_streamer()
        else:
            print(f"EDA streamer ready (PID: {eda_info['pid']}, stream: {eda_info['stream_name']})")
    elif run_preflight:
        print("\n[!] --eda-port not provided — EDA streamer will not be started and EDA checks will be skipped.")

    # Same for HR streamer: start before preflight so the check can verify live streams.
    if run_preflight:
        print("\n" + "=" * 60)
        print("Starting Polar HR streamer before preflight checks...")
        print("=" * 60)

        hr_info = _start_hr_streamer(
            repo_root=repo_root,
            hr_device=args.hr_device,
            hr_name_prefix=args.hr_name_prefix,
            health_check_timeout=args.hr_health_timeout,
            enable_ecg=getattr(args, 'hr_ecg', False),
            min_battery_pct=args.hr_min_battery if args.hr_min_battery > 0 else None,
        )

        if not hr_info["health_check_passed"]:
            print(f"\n[!] HR streamer did not start: {hr_info.get('error', 'unknown error')}", file=sys.stderr)
            print("    HR/RR will appear as NOT FOUND in preflight — use [r] to retry after fixing.", file=sys.stderr)
            _stop_hr_streamer()
        else:
            print(f"HR streamer ready (PID: {hr_info['pid']}, streams: {hr_info['stream_names']})")

    if run_preflight:
        # Build battery data (structured: label, pct, warn_below)
        _battery_data: list[dict] = []
        if eda_info is not None:
            _battery_data.append({
                "label": "Shimmer EDA",
                "pct": eda_info.get("battery_pct"),
                "warn_below": args.eda_min_battery + 10.0,
            })
        if hr_info is not None:
            _battery_data.append({
                "label": "Polar H10",
                "pct": hr_info.get("battery_pct"),
                "warn_below": args.hr_min_battery + 10.0,
            })

        if not _run_preflight_checks(
            check_eeg=True,
            check_eda=eda_info is not None,
            check_hr=hr_info is not None,
            check_joystick=not getattr(args, "skip_joystick_check", False),
            eeg_stream_type=args.eeg_stream_type,
            eeg_stream_count=args.eeg_stream_count,
            eda_stream_name=args.eda_stream_name,
            hr_stream_prefix=args.hr_name_prefix,
            timeout=5.0,
            battery_data=_battery_data or None,
            extra_required_streams=extra_required_streams or None,
        ):
            print("Session cancelled by user.", file=sys.stderr)
            return 1

    # Interactive prompts when arguments not provided
    _participant_arg = args.participant or _get_env_first("OPENMATB_PARTICIPANT", "OPENMATB_PARTICIPANT_ID")
    _session_arg = args.session or _get_env_first("OPENMATB_SESSION", "OPENMATB_SESSION_ID")

    while True:
        participant_raw = _participant_arg
        session_raw = _session_arg

        # Interactive participant selection
        if participant_raw is None:
            print("\n=== OpenMATB Session Setup ===")

            # Show recent participants
            recent = _get_recent_participants(assignments, limit=5)
            if recent:
                print("\nRecent participants:")
                for i, pid in enumerate(recent, 1):
                    pdata = participants_data[pid]
                    completed = len(pdata.get("sessions_completed", []))
                    print(f"  {i}. {pid} ({completed} sessions)")
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
                    if not user_input.startswith('P'):
                        try:
                            num = int(user_input)
                            participant_raw = f"P{num:03d}"
                        except ValueError:
                            participant_raw = f"P{user_input}"
                    else:
                        participant_raw = user_input

        # Look up session number from assignments
        if participant_raw in participants_data:
            assigned_sessions = participants_data[participant_raw].get("sessions_completed", [])

            if session_raw is None:
                next_session_num = len(assigned_sessions) + 1
                session_raw = f"S{next_session_num:03d}"
        else:
            print(f"\nERROR: {participant_raw} not found in assignments file.", file=sys.stderr)
            print(f"Add to config/participant_assignments.yaml first.", file=sys.stderr)
            _participant_arg = None  # force re-entry
            continue

        # Confirmation prompt
        if not args.verification:
            print(f"\n{'='*50}")
            print(f"  Participant: {participant_raw}")
            print(f"  Session:     {session_raw}")
            print(f"{'='*50}")
            while True:
                confirm = input("\nProceed? ([y]es / [r] re-enter / [n] abort): ").strip().lower()
                if confirm in ('y', 'yes', 'r', 'retry', 're-enter', '', 'n', 'no'):
                    break
                print("  Please enter y, r, or n.")
            if confirm in ('y', 'yes'):
                break
            elif confirm in ('r', 'retry', 're-enter', ''):
                _participant_arg = None  # force interactive re-entry next iteration
                continue
            else:
                print("Aborted by user.")
                return 1
        else:
            break


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

    # Check if session output already exists (prevent accidental overwriting).
    # Loop so the operator can enter a corrected participant ID without restarting.
    while True:
        session_output_dir = output_root_path / "openmatb" / participant / session
        if not session_output_dir.exists():
            break
        existing_files = list(session_output_dir.glob("*"))
        if not existing_files:
            break
        # Directory exists and has files — warn and offer options.
        print(f"\n{'!'*60}", file=sys.stderr)
        print(f"!!! WARNING: Session output directory already contains data!", file=sys.stderr)
        print(f"  Participant : {participant}", file=sys.stderr)
        print(f"  Session     : {session}", file=sys.stderr)
        print(f"  Path        : {session_output_dir}", file=sys.stderr)
        print(f"  Files       : {len(existing_files)} existing file(s)", file=sys.stderr)
        print(f"{'!'*60}", file=sys.stderr)
        print("\nOptions:")
        print("  [c] Continue anyway  (appends to existing data — may cause confusion)")
        print("  [p] Enter a different Participant ID")
        print("  [n] Abort")
        choice = input("\nChoice (c / p / n): ").strip().lower()
        if choice in ("c", "continue"):
            break
        elif choice in ("n", "no"):
            print("Aborted.")
            return 1
        elif choice not in ("p",):
            print("  Please enter c, p, or n.")
            continue
        elif choice in ("p",):
            new_pid_raw = input("New Participant ID (e.g. P004 or 4): ").strip()
            if not new_pid_raw:
                print("No input — aborting.")
                return 1
            # Normalise exactly the same way as the main entry loop.
            if not new_pid_raw.startswith("P"):
                try:
                    new_pid_raw = f"P{int(new_pid_raw):03d}"
                except ValueError:
                    new_pid_raw = f"P{new_pid_raw}"
            try:
                participant = _validate_id(new_pid_raw, label="participant")
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                continue  # re-prompt
            # Re-derive session from assignments for the new participant.
            if participant in participants_data:
                assigned_sessions = participants_data[participant].get("sessions_completed", [])
                next_session_num = len(assigned_sessions) + 1
                session = f"S{next_session_num:03d}"
            else:
                print(f"\nERROR: {participant} not found in assignments file.", file=sys.stderr)
                print(f"Add them to config/participant_assignments.yaml first.", file=sys.stderr)
                continue  # re-prompt
            print(f"\n  -> Switched to: Participant={participant}  Session={session}")
            # Loop back to re-check the new directory.
        else:
            print("Aborted.")
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

    playlist: list[str] = []

    if args.only_scenario:
        # Allow both bare filenames and paths; we always resolve relative to <repo>/scenarios.
        requested = Path(str(args.only_scenario)).name
        playlist = [requested]
    elif args.calibration_trend:
        playlist = [
            "pilot_calibration_low.txt",
            "pilot_calibration_moderate.txt",
            "pilot_calibration_high.txt",
        ]

    missing_scenarios: list[str] = []
    for scenario_filename in playlist:
        if not (repo_root / "experiment" / "scenarios" / scenario_filename).exists():
            missing_scenarios.append(scenario_filename)
    if missing_scenarios:
        print("Missing scenario files under <repo>/experiment/scenarios:", file=sys.stderr)
        for name in missing_scenarios:
            print(f" - {name}", file=sys.stderr)
        return 2

    print(f"Playlist ({len(playlist)} scenarios):")
    for s in playlist:
        print(f" - {s}")

    # Start LabRecorder (XDF) recording via RCS for Pilot 1 when no physiology artifact is provided.
    if args.pilot1 and physiology_recording_path is None and args.labrecorder_rcs:
        print("\n" + "=" * 60)
        print("Starting LabRecorder XDF recording via RCS for Pilot 1...")
        print("=" * 60)

        lab_root = Path(str(args.labrecorder_root)).resolve() if args.labrecorder_root else (output_root_path / "physiology")
        lab_template = str(args.labrecorder_template)

        labrecorder_info = _start_labrecorder_xdf_recording_rcs(
            host=str(args.labrecorder_host),
            port=int(args.labrecorder_port),
            root=lab_root,
            template=lab_template,
            participant=participant,
            session=session,
            task=str(args.labrecorder_task),
            acquisition=str(args.labrecorder_acq),
            modality=str(args.labrecorder_modality),
            run=str(args.labrecorder_run) if args.labrecorder_run else None,
        )

        if not labrecorder_info.get("started"):
            print("\n!!! LABRECORDER RCS FAILED TO START RECORDING !!!", file=sys.stderr)
            print(f"Error: {labrecorder_info.get('error', 'Unknown error')}", file=sys.stderr)
            print("\nEnsure LabRecorder is running with RCSEnabled=1 and the correct RCSPort.", file=sys.stderr)
            if eda_info and eda_info.get("started"):
                _stop_eda_streamer()
            if hr_info and hr_info.get("started"):
                _stop_hr_streamer()
            return 2

        physiology_recording_path = Path(labrecorder_info["expected_xdf_path"])
        print(f"Expected XDF path: {physiology_recording_path}")
        print("=" * 60 + "\n")

    # Start Python recorder for Pilot 1 when an existing recording artifact is not supplied.
    if args.pilot1 and physiology_recording_path is None and not args.no_python_recorder:
        print("\n" + "=" * 60)
        print("Starting Python LSL recorder for Pilot 1...")
        print("=" * 60)

        lsl_recorder_info = _start_python_lsl_recorder(
            repo_root=repo_root,
            output_root_path=output_root_path,
            participant=participant,
            session=session,
            eda_stream_name=args.eda_stream_name,
            hr_stream_prefix=args.hr_name_prefix if (hr_info and hr_info.get("started")) else None,
        )

        if not lsl_recorder_info["started"]:
            print(f"\n!!! PYTHON LSL RECORDER FAILED TO START !!!", file=sys.stderr)
            print(f"Error: {lsl_recorder_info.get('error', 'Unknown error')}", file=sys.stderr)
            print("\nProvide --xdf-path or run without --pilot1.", file=sys.stderr)
            if eda_info and eda_info.get("started"):
                _stop_eda_streamer()
            return 2

        physiology_recording_path = Path(str(lsl_recorder_info["recording_path"]))
        print(f"Python recorder ready (PID: {lsl_recorder_info['pid']})")
        print(f"Recording to: {physiology_recording_path}")
        print("=" * 60 + "\n")

    # Final launch checkpoint (only for real runs, not verification)
    if not args.verification:
        print("\n" + "=" * 60)
        print(f"  Participant: {participant}  |  Session: {session}  |  {len(playlist)} scenarios")
        print("=" * 60)

        while True:
            launch = input("\nStart session now? (y/n): ").strip().lower()
            if launch in ("y", "yes"):
                break
            elif launch in ("n", "no"):
                print("Session aborted by user at final checkpoint.")
                if args.pilot1:
                    _stop_python_lsl_recorder()
                    if args.labrecorder_rcs and labrecorder_info and labrecorder_info.get("started"):
                        _stop_labrecorder_xdf_recording_rcs(host=str(args.labrecorder_host), port=int(args.labrecorder_port))
                if eda_info and eda_info.get("started"):
                    _stop_eda_streamer()
                return 1
            else:
                print("  Please enter y or n.")

    scenario_manifests: list[Path] = []

    for block_index, scenario_filename in enumerate(playlist):
        exit_code, manifest_path = _run_single_scenario(
            openmatb_dir=openmatb_dir,
            scenario_filename=scenario_filename,
            output_root_path=output_root_path,
            participant=participant,
            session=session,
            args=args,
            repo_commit=repo_commit,
            submodule_commit=submodule_commit,
            block_index=block_index,
            adaptation_mode=getattr(args, "adaptation", False),
            adaptation_seed=getattr(args, "adaptation_seed", 0),
            mwl_adaptation_mode=getattr(args, "mwl_adaptation", False),
            mwl_simulated_mode=getattr(args, "mwl_simulated", None),
            mwl_model_dir=getattr(args, "mwl_model_dir", None),
            mwl_audit_csv=getattr(args, "mwl_audit_csv", None),
            mwl_threshold=getattr(args, "mwl_threshold", None),
        )

        if exit_code != 0:
            print(f"\n!!! Scenario {scenario_filename} failed (code {exit_code}). Stopping sequence. !!!", file=sys.stderr)
            if args.pilot1:
                _stop_python_lsl_recorder()
            if eda_info and eda_info.get("started"):
                _stop_eda_streamer()
            return exit_code

        if manifest_path:
            scenario_manifests.append(manifest_path)
        
        # Simple separation between blocks
        print(f"Scenario {scenario_filename} completed successfully.")
        # (Interactive UI and blocking dialogs are expected in attended mode)

    print("\nAll scenarios in playlist completed successfully.")

    if args.pilot1 and args.labrecorder_rcs and labrecorder_info and labrecorder_info.get("started"):
        print("\nStopping LabRecorder XDF recording via RCS...")
        stop_result = _stop_labrecorder_xdf_recording_rcs(host=str(args.labrecorder_host), port=int(args.labrecorder_port))
        if not stop_result.get("stopped"):
            print(f"WARNING: LabRecorder stop command may have failed: {stop_result.get('error')}", file=sys.stderr)

        # Give LabRecorder a moment to finalize the file.
        if physiology_recording_path is not None:
            for _ in range(10):
                if physiology_recording_path.exists():
                    break
                time.sleep(0.5)

            if not physiology_recording_path.exists():
                print("\n!!! Expected XDF file not found after stopping LabRecorder !!!", file=sys.stderr)
                print(f"Expected: {physiology_recording_path}", file=sys.stderr)
                entered = input("Enter the actual .xdf path (or leave blank to abort): ").strip()
                if not entered:
                    if eda_info and eda_info.get("started"):
                        _stop_eda_streamer()
                    if hr_info and hr_info.get("started"):
                        _stop_hr_streamer()
                    return 2
                physiology_recording_path = Path(entered)

    if args.pilot1 and lsl_recorder_info and lsl_recorder_info.get("started"):
        print("\nStopping Python LSL recorder...")
        recorded_path = _stop_python_lsl_recorder()
        if recorded_path is not None:
            physiology_recording_path = recorded_path
            print(f"Saved recording artifact: {physiology_recording_path}")

    if args.pilot1 and physiology_recording_path is None:
        print(
            "ERROR: --pilot1 requires a physiology recording artifact. "
            "Provide --xdf-path, or allow auto recorder (omit --no-python-recorder).",
            file=sys.stderr,
        )
        if eda_info and eda_info.get("started"):
            _stop_eda_streamer()
        return 2

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
            "mode": {
                "pilot1": bool(args.pilot1),
                "calibration_only": bool(args.calibration_only),
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
                "xdf_path": str(physiology_recording_path) if (physiology_recording_path and physiology_recording_path.suffix.lower() == ".xdf") else None,
                "recording_path": str(physiology_recording_path) if physiology_recording_path else None,
                "recording_format": (physiology_recording_path.suffix.lower().lstrip(".") if physiology_recording_path else None),
                "recording_scope": "calibration_only",
                "expected_streams": {
                    "markers": {"name": "OpenMATB", "type": "Markers"},
                    "eda": {"type": "EDA"},
                },
                "eda_streamer": {
                    "managed_by_runner": True,
                    "started": eda_info["started"] if eda_info else False,
                    "pid": eda_info["pid"] if eda_info else None,
                    "stream_name": eda_info["stream_name"] if eda_info else args.eda_stream_name,
                    "stream_type": eda_info["stream_type"] if eda_info else "EDA",
                    "health_check_passed": eda_info["health_check_passed"] if eda_info else None,
                    "port": args.eda_port,
                    "error": eda_info.get("error") if eda_info else None,
                } if eda_info is not None else None,
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
    
    # Run recording↔CSV marker alignment QC if --pilot1 and not --skip-xdf-qc
    if args.pilot1 and not args.skip_xdf_qc and physiology_recording_path:
        print("\n" + "="*60)
        print("Running recording↔CSV marker alignment QC...")
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
                xdf_path=Path(physiology_recording_path),
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
            print(f"WARNING: Could not run marker-alignment QC (missing dependency): {exc}", file=sys.stderr)
            run_manifest["qc"]["xdf_alignment"]["status"] = "skipped_missing_dep"
            _atomic_write_json(run_manifest_path, run_manifest)
        except Exception as exc:
            print(f"WARNING: Marker-alignment QC failed with error: {exc}", file=sys.stderr)
            run_manifest["qc"]["xdf_alignment"]["status"] = f"error: {exc}"
            _atomic_write_json(run_manifest_path, run_manifest)

    # Automatic external-only exports (results summary + cohort status)
    # Option A: always attempt; failures are warnings only.
    if not args.verification:
        _auto_export_pilot_results(
            repo_root=repo_root,
            output_root_path=output_root_path,
            participant=participant,
            session=session,
            run_manifest_path=run_manifest_path,
            scenario_manifests=scenario_manifests,
        )
    
    # Update participant assignments
    if not args.skip_assignment_update:
        # Ensure participant is in assignments
        if participant not in participants_data:
            participants_data[participant] = {
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
        _save_assignments(repo_root, assignments, skip_write=args.skip_assignment_update)
        print(f"Updated participant assignments: {participant} completed {session}")
    
    # Stop EDA streamer if we started it
    if eda_info and eda_info.get("started"):
        print("\nStopping EDA streamer...")
        _stop_eda_streamer()
        print("EDA streamer stopped.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
