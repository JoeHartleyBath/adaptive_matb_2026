"""Stream Shimmer GSR3 EDA data to LSL.

This script connects to a Shimmer GSR3 device via Bluetooth and streams
electrodermal activity (EDA/GSR) data to Lab Streaming Layer (LSL) for
synchronization with OpenMATB markers and EEG data.

Requirements:
  pip install pyshimmer pylsl

Usage:
  python scripts/stream_shimmer_eda.py --port COM5
    python scripts/stream_shimmer_eda.py --auto-port --ports COM3,COM5
  python scripts/stream_shimmer_eda.py --port COM5 --name ShimmerEDA

The script creates an LSL outlet with:
  - Name: "ShimmerEDA" (configurable via --name)
  - Type: "EDA"
    - Channels: 1 (GSR_RAW in raw_counts)
    - Sample rate: device-reported (commonly 128.0 Hz)

Press Ctrl+C to stop streaming.
"""

from __future__ import annotations

import argparse
import multiprocessing
import signal
import sys
import time
from typing import Optional
from collections.abc import Mapping, Sequence
import serial

# Check dependencies
try:
    import pylsl
    HAS_PYLSL = True
except ImportError:
    HAS_PYLSL = False

try:
    from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, EChannelType
    from pyshimmer.dev.channels import ESensorGroup
    HAS_PYSHIMMER = True
except ImportError:
    HAS_PYSHIMMER = False


def _candidate_serial_ports(preferred_ports: Optional[list[str]] = None) -> list[str]:
    if preferred_ports:
        return preferred_ports

    try:
        from serial.tools import list_ports

        return [p.device for p in list_ports.comports()]
    except Exception:
        if sys.platform.startswith("win"):
            return [f"COM{i}" for i in range(1, 33)]
        return []


def _probe_shimmer_port(port: str, baudrate: int) -> tuple[bool, str]:
    serial_conn: Optional[serial.Serial] = None
    shimmer: Optional[ShimmerBluetooth] = None
    try:
        serial_conn = serial.Serial(port=port, baudrate=baudrate, timeout=None)
        shimmer = ShimmerBluetooth(serial_conn)
        shimmer.initialize()

        device_name = None
        firmware = None
        try:
            device_name = shimmer.get_device_name()
        except Exception:
            pass
        try:
            firmware = shimmer.get_firmware_version()
        except Exception:
            pass

        details = []
        if device_name is not None:
            details.append(f"name={device_name}")
        if firmware is not None:
            details.append(f"firmware={firmware}")
        detail_text = ", ".join(details) if details else "connected"
        return True, detail_text
    except Exception as exc:
        return False, str(exc)
    finally:
        if shimmer is not None:
            try:
                shimmer.stop_streaming()
            except Exception:
                pass
            try:
                shimmer.disconnect()
            except Exception:
                pass
        if serial_conn is not None:
            try:
                serial_conn.close()
            except Exception:
                pass


def _probe_worker(port: str, baudrate: int, queue: multiprocessing.Queue) -> None:
    ok, message = _probe_shimmer_port(port, baudrate)
    queue.put((ok, message))


def _probe_shimmer_port_with_timeout(port: str, baudrate: int, timeout_sec: float) -> tuple[bool, str]:
    queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    process = multiprocessing.Process(target=_probe_worker, args=(port, baudrate, queue), daemon=True)
    process.start()
    process.join(max(0.1, timeout_sec))

    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"timeout after {timeout_sec:.1f}s"

    if not queue.empty():
        try:
            return queue.get_nowait()
        except Exception:
            pass

    return False, "probe exited without result"


def _autodetect_shimmer_port(
    baudrate: int,
    timeout_sec: float,
    preferred_ports: Optional[list[str]] = None,
) -> Optional[str]:
    ports = _candidate_serial_ports(preferred_ports)
    if not ports:
        print("ERROR: No serial ports found to scan.", file=sys.stderr)
        return None

    print(f"Scanning {len(ports)} serial port(s) for Shimmer (timeout {timeout_sec:.1f}s/port)...")
    for port in ports:
        ok, message = _probe_shimmer_port_with_timeout(port, baudrate, timeout_sec)
        if ok:
            print(f"  {port}: Shimmer detected ({message})")
            return port
        print(f"  {port}: not Shimmer ({message})")

    print("ERROR: No valid Shimmer device found on scanned serial ports.", file=sys.stderr)
    return None


class ShimmerEDAStreamer:
    """Streams Shimmer GSR3 EDA data to LSL."""
    
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        stream_name: str = "ShimmerEDA",
        stream_type: str = "EDA",
        source_id: Optional[str] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.source_id = source_id or f"shimmer_gsr_{port}"
        
        self.serial_conn: Optional[serial.Serial] = None
        self.shimmer: Optional[ShimmerBluetooth] = None
        self.outlet: Optional[pylsl.StreamOutlet] = None
        self.running = False
        self.samples_sent = 0
        self.start_time = 0.0
        self.channel_index: dict[EChannelType, int] = {}
        self.extract_errors = 0
        self.battery: Optional[float] = None  # percentage (0–100), or None if unreadable

        # Shimmer GSR3 nominal sample rate
        self.sample_rate = 51.2

    def _safe_get(self, method_name: str):
        if self.shimmer is None:
            return None
        method = getattr(self.shimmer, method_name, None)
        if method is None:
            return None
        try:
            return method()
        except Exception:
            return None

    def _get_battery_pct(self) -> Optional[float]:
        """Attempt to read battery level as a percentage (0–100).

        Tries several method names in order (pyshimmer API varies by version).
        Returns None if the device does not expose battery level.
        """
        # Direct percentage-returning methods
        for method in ("get_battery_percent", "battery_percent", "get_battery_pct"):
            val = self._safe_get(method)
            if val is not None:
                try:
                    pct = float(val)
                    if 0.0 <= pct <= 100.0:
                        return round(pct, 1)
                except (TypeError, ValueError):
                    pass

        # Voltage-returning methods — convert LiPo range (3.2 V = 0 %, 4.2 V = 100 %)
        for method in ("get_vsense_batt", "get_battery_voltage", "get_battery"):
            val = self._safe_get(method)
            if val is not None:
                try:
                    v = float(val)
                    if 3.0 <= v <= 4.5:           # raw volts
                        pct = (v - 3.2) / 1.0 * 100.0
                    elif 3000 <= v <= 4500:        # millivolts
                        pct = (v - 3200) / 1000.0 * 100.0
                    else:
                        continue
                    return round(max(0.0, min(100.0, pct)), 1)
                except (TypeError, ValueError):
                    pass

        return None

    def describe_device(self) -> None:
        """Print device-reported details for verification."""
        if self.shimmer is None:
            return

        device_name = self._safe_get("get_device_name")
        firmware = self._safe_get("get_firmware_version")
        sample_rate = self._safe_get("get_sampling_rate")
        sensors = self._safe_get("get_data_types")

        print("Shimmer device details:")
        print(f"  port: {self.port}")
        print(f"  baudrate: {self.baudrate}")
        if device_name is not None:
            print(f"  device_name: {device_name}")
        if firmware is not None:
            print(f"  firmware_version: {firmware}")
        if sample_rate is not None:
            print(f"  sampling_rate_hz: {sample_rate}")
        if sensors is not None:
            print(f"  enabled_data_types: {sensors}")
        gsr_idx = self.channel_index.get(EChannelType.GSR_RAW)
        if gsr_idx is not None:
            print(f"  gsr_raw_channel_index: {gsr_idx}")
        if self.battery is not None:
            print(f"  battery: {self.battery:.0f}%")
        else:
            print(f"  battery: unknown (not reported by pyshimmer)")
        print("  streamer_channel: GSR_RAW (unit=raw_counts)")

    def probe_json(self) -> dict:
        """Return device info as a JSON-serialisable dict (for --probe-json mode)."""
        device_name = self._safe_get("get_device_name")
        firmware = self._safe_get("get_firmware_version")
        return {
            "ok": True,
            "device_name": str(device_name) if device_name is not None else None,
            "firmware": str(firmware) if firmware is not None else None,
            "port": self.port,
            "battery_pct": self.battery,
        }

    def _extract_gsr_value(self, packet) -> float:
        channels = getattr(packet, "channels", None)

        if isinstance(channels, Sequence) and EChannelType.GSR_RAW in channels:
            try:
                return float(packet[EChannelType.GSR_RAW])
            except Exception:
                pass

        if isinstance(channels, Mapping):
            if EChannelType.GSR_RAW in channels:
                return float(channels[EChannelType.GSR_RAW])
            if "GSR_RAW" in channels:
                return float(channels["GSR_RAW"])
            if "GSR" in channels:
                return float(channels["GSR"])

        if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
            gsr_idx = self.channel_index.get(EChannelType.GSR_RAW)
            if gsr_idx is not None and 0 <= gsr_idx < len(channels):
                return float(channels[gsr_idx])

        if isinstance(packet, Mapping):
            if EChannelType.GSR_RAW in packet:
                return float(packet[EChannelType.GSR_RAW])
            if "GSR_RAW" in packet:
                return float(packet["GSR_RAW"])
            if "GSR" in packet:
                return float(packet["GSR"])

        if isinstance(packet, Sequence) and not isinstance(packet, (str, bytes)):
            gsr_idx = self.channel_index.get(EChannelType.GSR_RAW)
            if gsr_idx is not None and 0 <= gsr_idx < len(packet):
                return float(packet[gsr_idx])

        gsr_attr = getattr(packet, "gsr_raw", None)
        if gsr_attr is None:
            gsr_attr = getattr(packet, "GSR_RAW", None)
        if gsr_attr is not None:
            return float(gsr_attr)

        raise ValueError("Unable to locate GSR_RAW in packet payload")
    
    def connect(self) -> bool:
        """Connect to Shimmer device and create LSL outlet."""
        print(f"Connecting to Shimmer on {self.port}...")
        
        try:
            self.serial_conn = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=None)
            self.shimmer = ShimmerBluetooth(self.serial_conn)
            self.shimmer.initialize()
            
            # pyshimmer set_sensors expects ESensorGroup values (not EChannelType).
            # Enable the GSR sensor group so packets include GSR_RAW channel data.
            self.shimmer.set_sensors([ESensorGroup.GSR])
            
            # Get actual sample rate
            actual_rate = self.shimmer.get_sampling_rate()
            if actual_rate:
                self.sample_rate = actual_rate

            data_types = self._safe_get("get_data_types")
            if isinstance(data_types, Sequence) and not isinstance(data_types, (str, bytes)):
                self.channel_index = {
                    channel: index
                    for index, channel in enumerate(data_types)
                    if isinstance(channel, EChannelType)
                }
            
            print(f"Connected to Shimmer (sample rate: {self.sample_rate} Hz)")
            # Attempt battery reading before describe so it can be displayed
            self.battery = self._get_battery_pct()
            self.describe_device()

        except Exception as e:
            print(f"ERROR: Failed to connect to Shimmer: {e}", file=sys.stderr)
            return False
        
        # Create LSL outlet
        print(f"Creating LSL outlet: {self.stream_name} (type={self.stream_type})")
        
        info = pylsl.StreamInfo(
            name=self.stream_name,
            type=self.stream_type,
            channel_count=1,
            nominal_srate=self.sample_rate,
            channel_format=pylsl.cf_float32,
            source_id=self.source_id,
        )
        
        # Add channel metadata
        channels = info.desc().append_child("channels")
        ch = channels.append_child("channel")
        ch.append_child_value("label", "GSR_RAW")
        ch.append_child_value("unit", "raw_counts")
        ch.append_child_value("type", "EDA")
        
        # Add device metadata
        device = info.desc().append_child("device")
        device.append_child_value("manufacturer", "Shimmer")
        device.append_child_value("model", "GSR3")
        device.append_child_value("port", self.port)
        device.append_child_value("signal_format", "raw")
        
        self.outlet = pylsl.StreamOutlet(info)
        print(f"LSL outlet created: {self.stream_name}")
        
        return True
    
    def _stream_callback(self, packet) -> None:
        """Callback for Shimmer data packets (uses DataPacket from pyshimmer)."""
        if not self.running or self.outlet is None:
            return
        
        try:
            gsr_value = self._extract_gsr_value(packet)
            self.outlet.push_sample([gsr_value])
            self.samples_sent += 1
        except Exception as e:
            self.extract_errors += 1
            if self.extract_errors <= 5:
                packet_type = type(packet).__name__
                channels = getattr(packet, "channels", None)
                channels_type = type(channels).__name__ if channels is not None else "None"
                print(
                    f"WARNING: Error extracting GSR from packet #{self.extract_errors} "
                    f"(packet={packet_type}, channels={channels_type}): {e}",
                    file=sys.stderr,
                )
    
    def start_streaming(self) -> None:
        """Start streaming data to LSL."""
        if self.shimmer is None:
            print("ERROR: Not connected to Shimmer", file=sys.stderr)
            return
        
        print("\nStarting EDA stream... (press Ctrl+C to stop)")
        self.running = True
        self.start_time = time.time()
        self.samples_sent = 0
        
        # Register callback and start streaming
        # pyshimmer uses add_stream_callback for data packets
        self.shimmer.add_stream_callback(self._stream_callback)
        self.shimmer.start_streaming()
        
        # Keep alive and show status
        try:
            while self.running:
                time.sleep(5.0)
                elapsed = time.time() - self.start_time
                rate = self.samples_sent / elapsed if elapsed > 0 else 0
                print(f"  Streamed {self.samples_sent} samples ({rate:.1f} Hz)")
        except KeyboardInterrupt:
            print("\nStopping...")
        
        self.stop()
    
    def stop(self) -> None:
        """Stop streaming and disconnect."""
        self.running = False
        
        if self.shimmer:
            try:
                self.shimmer.stop_streaming()
                self.shimmer.disconnect()
            except Exception:
                pass
            self.shimmer = None

        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        
        if self.samples_sent > 0:
            elapsed = time.time() - self.start_time
            print(f"\nStreamed {self.samples_sent} samples in {elapsed:.1f}s")
        
        print("Shimmer EDA streamer stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream Shimmer GSR3 EDA data to LSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/stream_shimmer_eda.py --port COM5
    python scripts/stream_shimmer_eda.py --auto-port
    python scripts/stream_shimmer_eda.py --auto-port --ports COM3,COM5
  python scripts/stream_shimmer_eda.py --port COM5 --name ShimmerEDA

The script creates an LSL stream with type="EDA" that can be recorded
alongside OpenMATB markers and EEG using LabRecorder.
""",
    )
    
    parser.add_argument(
        "--port",
        required=False,
        help="Serial port for Shimmer Bluetooth connection (e.g., COM5, /dev/tty.Shimmer-GSR3-RN42)",
    )
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Auto-detect Shimmer by scanning serial ports and selecting the first valid device.",
    )
    parser.add_argument(
        "--auto-port-timeout-sec",
        type=float,
        default=6.0,
        help="Per-port probe timeout in seconds for --auto-port scanning (default: 6.0).",
    )
    parser.add_argument(
        "--ports",
        default=None,
        help="Comma-separated port list for --auto-port scan order (e.g., COM3,COM5).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE if HAS_PYSHIMMER else 115200,
        help="Serial baud rate for Shimmer Bluetooth link (default: pyshimmer DEFAULT_BAUDRATE)",
    )
    parser.add_argument(
        "--name",
        default="ShimmerEDA",
        help="LSL stream name (default: ShimmerEDA)",
    )
    parser.add_argument(
        "--type",
        dest="stream_type",
        default="EDA",
        help="LSL stream type (default: EDA)",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="LSL source ID (default: shimmer_gsr_<port>)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: check dependencies and exit",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Connect to Shimmer, print device details, and exit.",
    )
    parser.add_argument(
        "--probe-json",
        action="store_true",
        help="Connect, print device info (including battery) as a single JSON line, and exit.",
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_PYLSL:
        print("ERROR: pylsl is required. Install with: pip install pylsl", file=sys.stderr)
        return 2
    
    if not HAS_PYSHIMMER:
        print("ERROR: pyshimmer is required. Install with: pip install pyshimmer", file=sys.stderr)
        return 2
    
    if args.test:
        print("Dependencies OK:")
        print(f"  pylsl: {HAS_PYLSL}")
        print(f"  pyshimmer: {HAS_PYSHIMMER}")
        return 0

    selected_port = args.port
    if args.auto_port and not selected_port:
        preferred_ports = None
        if args.ports:
            preferred_ports = [p.strip().upper() for p in args.ports.split(",") if p.strip()]

        selected_port = _autodetect_shimmer_port(
            args.baudrate,
            args.auto_port_timeout_sec,
            preferred_ports,
        )
        if selected_port is None:
            return 1
        print(f"Using auto-detected port: {selected_port}")

    if not selected_port:
        print("ERROR: Provide --port or use --auto-port.", file=sys.stderr)
        return 2
    
    # Create and start streamer
    streamer = ShimmerEDAStreamer(
        port=selected_port,
        baudrate=args.baudrate,
        stream_name=args.name,
        stream_type=args.stream_type,
        source_id=args.source_id,
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        streamer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if not streamer.connect():
        return 1

    if args.probe_json:
        import json as _json
        info = streamer.probe_json()
        print(_json.dumps(info))
        streamer.stop()
        return 0

    if args.probe:
        print("Probe successful. Exiting without starting stream.")
        streamer.stop()
        return 0

    streamer.start_streaming()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
