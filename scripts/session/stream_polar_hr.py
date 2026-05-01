"""Stream Polar H10 HR, RR intervals, and raw ECG to LSL.

This script connects to a Polar H10 heart-rate sensor via Bluetooth Low Energy
(BLE) and publishes three LSL outlets:

  {prefix}HR   type=HR    1 ch, ~1 Hz     – heart rate (beats/min, int32)
  {prefix}RR   type=RR    1 ch, irregular – RR intervals (ms, float32)
  {prefix}ECG  type=ECG   1 ch, 130 Hz    – raw ECG (µV, float32)

HR and RR arrive together via the standard Bluetooth GATT HR Measurement
characteristic (0x2A37).  ECG comes from Polar's proprietary PMD protocol.

Requirements:
  pip install bleak pylsl

Usage:
  python scripts/session/stream_polar_hr.py
  python scripts/session/stream_polar_hr.py --device XX:XX:XX:XX:XX:XX
  python scripts/session/stream_polar_hr.py --prefix PolarH10 --scan-timeout 15
  python scripts/session/stream_polar_hr.py --probe              # connect, print info, exit
  python scripts/session/stream_polar_hr.py --test               # check deps, exit

Press Ctrl+C to stop streaming.

Notes:
  - BLE address format differs by OS: on Windows, BLE addresses are shown as
    integers (e.g. 171981279273299) or XX:XX:XX:XX:XX:XX strings – bleak
    accepts both.
  - The Polar H10 must be worn against skin and paired/connectable over BLE.
  - Only one concurrent BLE client is supported by the H10 at a time.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import struct
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------
try:
    import pylsl
    HAS_PYLSL = True
except ImportError:
    HAS_PYLSL = False

try:
    import bleak  # noqa: F401 – import just to probe availability
    from bleak import BleakClient, BleakScanner
    from bleak.backends.device import BLEDevice
    HAS_BLEAK = True
except ImportError:
    HAS_BLEAK = False

# ---------------------------------------------------------------------------
# Polar H10 GATT UUIDs
# ---------------------------------------------------------------------------
# Standard BT HR Measurement characteristic
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
# Standard BT Device Information characteristics
FIRMWARE_REVISION_UUID = "00002a26-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_UUID     = "00002a19-0000-1000-8000-00805f9b34fb"

# Polar Measurement Data (PMD) proprietary service
PMD_SERVICE_UUID        = "fb005c80-02e7-f387-1cad-8acd2d8df0c8"
PMD_CONTROL_POINT_UUID  = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA_UUID           = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

# ECG stream start request: request type 0x02, ECG (0x00), start (0x01),
# 130 Hz (0x82 0x00), resolution 14-bit (0x01), sample_group 1 (0x01),
# channels 1 (0x0E 0x00)
_ECG_START_CMD = bytes([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

# Polar ECG nominal sample rate
ECG_SRATE_HZ = 130.0


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_hr_measurement(data: bytes) -> tuple[int, list[float]]:
    """Parse Bluetooth HR Measurement characteristic payload.

    Returns:
        hr_bpm: Heart rate in beats per minute.
        rr_ms:  List of RR interval values in milliseconds (may be empty).

    Flags byte layout (bit 0 = HR format, bit 4 = RR present):
      bit 0 = 0 → HR is uint8; = 1 → HR is uint16
      bit 4 = 1 → RR interval field is present (each entry is uint16 × 1000/1024 ms)
    """
    if not data:
        return 0, []

    flags = data[0]
    hr_format_16bit = bool(flags & 0x01)
    rr_present      = bool(flags & 0x10)

    offset = 1
    if hr_format_16bit:
        hr_bpm = struct.unpack_from("<H", data, offset)[0]
        offset += 2
    else:
        hr_bpm = data[offset]
        offset += 1

    # Skip energy expended field if present (bit 3)
    if flags & 0x08:
        offset += 2

    rr_ms: list[float] = []
    if rr_present:
        while offset + 1 < len(data):
            raw = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            rr_ms.append(raw * 1000.0 / 1024.0)

    return hr_bpm, rr_ms


def _parse_ecg_frame(data: bytes) -> list[float]:
    """Parse a Polar PMD ECG data frame into a list of µV sample values.

    Frame format (after 10-byte header):
      Each sample is 3 bytes, little-endian, 14-bit signed (sign-extended from bit 13).
    Returns list of float µV values.
    """
    if len(data) < 10:
        return []

    samples: list[float] = []
    idx = 10  # skip the 10-byte PMD frame header
    while idx + 2 < len(data):
        raw = data[idx] | (data[idx + 1] << 8) | (data[idx + 2] << 16)
        idx += 3
        # Sign-extend 14-bit value stored in lower 14 bits
        value = raw & 0x3FFF
        if value & 0x2000:          # bit 13 set → negative
            value -= 0x4000
        samples.append(float(value))

    return samples


# ---------------------------------------------------------------------------
# Streamer class
# ---------------------------------------------------------------------------

class PolarHRStreamer:
    """Connect to a Polar H10 over BLE and publish HR, RR, and ECG to LSL."""

    def __init__(
        self,
        device_address: Optional[str] = None,
        name_prefix: str = "Polar",
        source_id: Optional[str] = None,
        scan_timeout: float = 10.0,
        enable_ecg: bool = True,
    ):
        self.device_address = device_address
        self.name_prefix    = name_prefix
        self.source_id_base = source_id or f"polar_h10_{name_prefix.lower()}"
        self.scan_timeout   = scan_timeout
        self.enable_ecg     = enable_ecg

        # BLE state
        self._client: Optional[BleakClient] = None
        self._device: Optional[BLEDevice]   = None
        self._running                        = False

        # LSL outlets (created after device is found so source_id can use MAC)
        self._hr_outlet:  Optional[pylsl.StreamOutlet] = None
        self._rr_outlet:  Optional[pylsl.StreamOutlet] = None
        self._ecg_outlet: Optional[pylsl.StreamOutlet] = None

        # Counters for status reporting
        self.hr_samples_sent  = 0
        self.rr_samples_sent  = 0
        self.ecg_samples_sent = 0
        self.start_time       = 0.0

        # Device metadata (populated after connection)
        self.firmware: Optional[str] = None
        self.battery:  Optional[int] = None

    # ------------------------------------------------------------------
    # LSL outlet creation
    # ------------------------------------------------------------------

    def _create_outlets(self, mac: str) -> None:
        src = self.source_id_base

        # HR outlet – 1 channel, ~1 Hz, int32 (BPM)
        hr_info = pylsl.StreamInfo(
            name=f"{self.name_prefix}HR",
            type="HR",
            channel_count=1,
            nominal_srate=1.0,
            channel_format=pylsl.cf_int32,
            source_id=f"{src}_hr",
        )
        _ch = hr_info.desc().append_child("channels").append_child("channel")
        _ch.append_child_value("label", "HR")
        _ch.append_child_value("unit", "beats_per_min")
        _ch.append_child_value("type", "HR")
        _dev = hr_info.desc().append_child("device")
        _dev.append_child_value("manufacturer", "Polar")
        _dev.append_child_value("model", "H10")
        _dev.append_child_value("ble_address", mac)

        # RR outlet – 1 channel, irregular (0), float32 (ms)
        rr_info = pylsl.StreamInfo(
            name=f"{self.name_prefix}RR",
            type="RR",
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_float32,
            source_id=f"{src}_rr",
        )
        _ch = rr_info.desc().append_child("channels").append_child("channel")
        _ch.append_child_value("label", "RR")
        _ch.append_child_value("unit", "ms")
        _ch.append_child_value("type", "RR")
        _dev = rr_info.desc().append_child("device")
        _dev.append_child_value("manufacturer", "Polar")
        _dev.append_child_value("model", "H10")
        _dev.append_child_value("ble_address", mac)

        self._hr_outlet = pylsl.StreamOutlet(hr_info)
        self._rr_outlet = pylsl.StreamOutlet(rr_info)

        # ECG outlet – 1 channel, 130 Hz, float32 (µV)
        if self.enable_ecg:
            ecg_info = pylsl.StreamInfo(
                name=f"{self.name_prefix}ECG",
                type="ECG",
                channel_count=1,
                nominal_srate=ECG_SRATE_HZ,
                channel_format=pylsl.cf_float32,
                source_id=f"{src}_ecg",
            )
            _ch = ecg_info.desc().append_child("channels").append_child("channel")
            _ch.append_child_value("label", "ECG")
            _ch.append_child_value("unit", "microvolts")
            _ch.append_child_value("type", "ECG")
            _dev = ecg_info.desc().append_child("device")
            _dev.append_child_value("manufacturer", "Polar")
            _dev.append_child_value("model", "H10")
            _dev.append_child_value("ble_address", mac)
            self._ecg_outlet = pylsl.StreamOutlet(ecg_info)

    # ------------------------------------------------------------------
    # BLE callbacks
    # ------------------------------------------------------------------

    def _on_hr_measurement(self, _sender: int, data: bytearray) -> None:
        if not self._running:
            return
        try:
            hr_bpm, rr_list = _parse_hr_measurement(bytes(data))
        except Exception as exc:
            print(f"WARNING: HR parse error: {exc}", file=sys.stderr)
            return

        if self._hr_outlet:
            self._hr_outlet.push_sample([hr_bpm])
            self.hr_samples_sent += 1

        if self._rr_outlet and rr_list:
            for rr_ms in rr_list:
                self._rr_outlet.push_sample([rr_ms])
                self.rr_samples_sent += 1

    def _on_ecg_data(self, _sender: int, data: bytearray) -> None:
        if not self._running or self._ecg_outlet is None:
            return
        try:
            samples = _parse_ecg_frame(bytes(data))
        except Exception as exc:
            print(f"WARNING: ECG parse error: {exc}", file=sys.stderr)
            return

        for s in samples:
            self._ecg_outlet.push_sample([s])
        self.ecg_samples_sent += len(samples)

    def _on_pmd_control(self, _sender: int, data: bytearray) -> None:
        """Receive indicate from PMD control point (response to start/stop commands)."""
        if not data or len(data) < 3:
            return
        op     = data[0]
        mtype  = data[1]
        status = data[2]
        if status == 0x00:
            print(f"  PMD control: op=0x{op:02x} type=0x{mtype:02x} → OK")
        else:
            print(
                f"  WARNING: PMD control error: op=0x{op:02x} type=0x{mtype:02x} "
                f"status=0x{status:02x}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    async def _find_device(self) -> Optional[BLEDevice]:
        """Resolve BLE address or scan for first Polar H10."""
        if self.device_address:
            print(f"Connecting to specified BLE address: {self.device_address}")
            device = await BleakScanner.find_device_by_address(
                self.device_address, timeout=self.scan_timeout
            )
            if device is None:
                print(
                    f"ERROR: Device not found at address {self.device_address} "
                    f"within {self.scan_timeout:.0f}s.",
                    file=sys.stderr,
                )
            return device

        print(f"Scanning for Polar H10... (timeout {self.scan_timeout:.0f}s)")
        devices = await BleakScanner.discover(timeout=self.scan_timeout)
        for dev in devices:
            name = dev.name or ""
            if "Polar" in name and "H10" in name:
                print(f"Found: {dev.name!r} at {dev.address}")
                return dev
            # Tolerate "Polar H 10" or just "Polar H10" variants
            if name.upper().startswith("POLAR"):
                print(f"Found Polar device: {dev.name!r} at {dev.address} — using it")
                return dev

        print(
            f"ERROR: No Polar device found within {self.scan_timeout:.0f}s. "
            "Is the H10 worn and connectable?",
            file=sys.stderr,
        )
        return None

    # ------------------------------------------------------------------
    # Device information
    # ------------------------------------------------------------------

    async def _read_device_info(self, client: BleakClient) -> None:
        try:
            fw_bytes = await client.read_gatt_char(FIRMWARE_REVISION_UUID)
            self.firmware = fw_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            self.firmware = "unknown"

        try:
            batt_bytes = await client.read_gatt_char(BATTERY_LEVEL_UUID)
            self.battery = batt_bytes[0]
        except Exception:
            self.battery = None

    def describe_device(self) -> None:
        if self._device is None:
            print("No device connected.")
            return
        print("Polar H10 device details:")
        print(f"  name:        {self._device.name or 'unknown'}")
        print(f"  ble_address: {self._device.address}")
        if self.firmware is not None:
            print(f"  firmware:    {self.firmware}")
        if self.battery is not None:
            print(f"  battery:     {self.battery}%")
        print(f"  streams:")
        print(f"    {self.name_prefix}HR  (type=HR,  ~1 Hz,  beats/min)")
        print(f"    {self.name_prefix}RR  (type=RR,  irregular, ms)")
        if self.enable_ecg:
            print(f"    {self.name_prefix}ECG (type=ECG, 130 Hz, µV)")

    # ------------------------------------------------------------------
    # Main streaming loop
    # ------------------------------------------------------------------

    async def _stream_loop(self, probe_only: bool = False) -> None:
        self._device = await self._find_device()
        if self._device is None:
            return

        mac = self._device.address
        print(f"Connecting to {self._device.name!r} ({mac})...")

        async with BleakClient(self._device) as client:
            self._client = client

            await self._read_device_info(client)
            self.describe_device()

            if probe_only:
                return

            # Create LSL outlets now that we have the MAC
            self._create_outlets(mac)

            # Subscribe to HR Measurement
            await client.start_notify(HR_MEASUREMENT_UUID, self._on_hr_measurement)
            print(f"Subscribing to HR Measurement — LSL outlet: {self.name_prefix}HR")

            # Configure and subscribe to ECG via PMD.
            # Critical order for H10 firmware: arm both notification subscriptions
            # (PMD_DATA notify + PMD_CONTROL_POINT indicate) *before* writing the
            # start-measurement command.  The H10 checks that the CCC descriptors are
            # enabled before it will honour the request and begin pushing ECG frames.
            if self.enable_ecg:
                try:
                    # 1. Arm the data-frame subscription (notify)
                    await client.start_notify(PMD_DATA_UUID, self._on_ecg_data)
                    # 2. Arm the control-point subscription (indicate) so we receive
                    #    the measurement-start confirmation from the device.
                    await client.start_notify(PMD_CONTROL_POINT_UUID, self._on_pmd_control)
                    await asyncio.sleep(0.1)  # let CCC writes settle on the device
                    # 3. Send the start-measurement request
                    await client.write_gatt_char(
                        PMD_CONTROL_POINT_UUID, _ECG_START_CMD, response=True
                    )
                    print(f"Subscribing to PMD ECG data  — LSL outlet: {self.name_prefix}ECG")
                except Exception as exc:
                    print(
                        f"WARNING: ECG PMD setup failed: {exc} — ECG will not stream.",
                        file=sys.stderr,
                    )
                    self.enable_ecg = False

            self._running = True
            self.start_time = time.time()
            print(f"\nStreaming… (press Ctrl+C to stop)")

            try:
                while self._running:
                    await asyncio.sleep(5.0)
                    elapsed = time.time() - self.start_time
                    hr_rate  = self.hr_samples_sent  / elapsed if elapsed > 0 else 0.0
                    rr_count = self.rr_samples_sent
                    ecg_rate = self.ecg_samples_sent / elapsed if elapsed > 0 else 0.0

                    status = (
                        f"  HR: {self.hr_samples_sent} samples ({hr_rate:.2f} Hz)  "
                        f"RR: {rr_count} intervals"
                    )
                    if self.enable_ecg:
                        status += f"  ECG: {self.ecg_samples_sent} samples ({ecg_rate:.1f} Hz)"
                    print(status)

            except asyncio.CancelledError:
                pass
            finally:
                try:
                    await client.stop_notify(HR_MEASUREMENT_UUID)
                except Exception:
                    pass
                if self.enable_ecg:
                    try:
                        await client.stop_notify(PMD_DATA_UUID)
                    except Exception:
                        pass
                    try:
                        await client.stop_notify(PMD_CONTROL_POINT_UUID)
                    except Exception:
                        pass
                self._running = False

    async def _probe_loop(self) -> None:
        await self._stream_loop(probe_only=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        """Block until streaming completes or is interrupted."""
        asyncio.run(self._stream_loop())

    def probe(self) -> bool:
        """Connect, read device info, print details, disconnect. Returns True if ok."""
        asyncio.run(self._probe_loop())
        return self._device is not None

    def stop(self) -> None:
        """Signal the streaming loop to stop."""
        self._running = False

    def print_summary(self) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        print(f"\nSession summary ({elapsed:.1f}s):")
        print(f"  HR sent:  {self.hr_samples_sent} samples")
        print(f"  RR sent:  {self.rr_samples_sent} intervals")
        if self.enable_ecg:
            print(f"  ECG sent: {self.ecg_samples_sent} samples")
        print("Polar HR streamer stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream Polar H10 HR, RR intervals, and ECG to LSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/session/stream_polar_hr.py
  python scripts/session/stream_polar_hr.py --device XX:XX:XX:XX:XX:XX
  python scripts/session/stream_polar_hr.py --prefix PolarH10 --scan-timeout 15
  python scripts/session/stream_polar_hr.py --probe    # connect + print info, then exit
  python scripts/session/stream_polar_hr.py --test     # dependency check, then exit

Streams created (with default --prefix Polar):
  PolarHR   type=HR   1 ch  ~1 Hz       beats/min  int32
  PolarRR   type=RR   1 ch  irregular   ms         float32
  PolarECG  type=ECG  1 ch  130 Hz      µV         float32
""",
    )

    parser.add_argument(
        "--device",
        default=None,
        metavar="ADDRESS",
        help=(
            "BLE address of the Polar H10 (e.g. XX:XX:XX:XX:XX:XX). "
            "If omitted, the first Polar device found during scanning is used."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="Polar",
        metavar="PREFIX",
        help="Prefix for LSL stream names (default: Polar → PolarHR, PolarRR, PolarECG).",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        metavar="ID",
        help="Base LSL source ID; defaults to polar_h10_{prefix_lower}.",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="BLE scan/connect timeout in seconds (default: 10).",
    )
    parser.add_argument(
        "--no-ecg",
        action="store_true",
        help="Disable ECG streaming (HR and RR only).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Check dependencies and exit.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Connect to device, print firmware/battery info, then exit without streaming.",
    )
    parser.add_argument(
        "--probe-json",
        action="store_true",
        help="Connect, print device info (including battery) as a single JSON line, and exit.",
    )

    args = parser.parse_args()

    # Dependency checks
    if not HAS_PYLSL:
        print("ERROR: pylsl is required. Install with: pip install pylsl", file=sys.stderr)
        return 2

    if not HAS_BLEAK:
        print("ERROR: bleak is required. Install with: pip install bleak", file=sys.stderr)
        return 2

    if args.test:
        import bleak as bleak_mod
        print("Dependencies OK:")
        print(f"  pylsl:  {HAS_PYLSL}")
        print(f"  bleak:  {bleak_mod.__version__}")
        return 0

    streamer = PolarHRStreamer(
        device_address=args.device,
        name_prefix=args.prefix,
        source_id=args.source_id,
        scan_timeout=args.scan_timeout,
        enable_ecg=not args.no_ecg,
    )

    # Signal handling — set _running=False and let the async loop exit cleanly
    def _signal_handler(sig, frame):
        print("\nStopping…", file=sys.stderr)
        streamer.stop()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.probe_json:
        asyncio.run(streamer._probe_loop())
        if streamer._device is None:
            return 1
        import json as _json
        info = {
            "ok": True,
            "name": streamer._device.name or "unknown",
            "address": streamer._device.address,
            "firmware": streamer.firmware,
            "battery_pct": streamer.battery,
        }
        print(_json.dumps(info))
        return 0

    if args.probe:
        ok = streamer.probe()
        return 0 if ok else 1

    streamer.start_streaming()
    streamer.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
