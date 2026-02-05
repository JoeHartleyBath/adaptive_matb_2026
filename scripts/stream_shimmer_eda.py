"""Stream Shimmer GSR3 EDA data to LSL.

This script connects to a Shimmer GSR3 device via Bluetooth and streams
electrodermal activity (EDA/GSR) data to Lab Streaming Layer (LSL) for
synchronization with OpenMATB markers and EEG data.

Requirements:
  pip install pyshimmer pylsl

Usage:
  python scripts/stream_shimmer_eda.py --port COM5
  python scripts/stream_shimmer_eda.py --port COM5 --name ShimmerEDA

The script creates an LSL outlet with:
  - Name: "ShimmerEDA" (configurable via --name)
  - Type: "EDA"
  - Channels: 1 (GSR in microsiemens)
  - Sample rate: ~51.2 Hz (Shimmer GSR3 default)

Press Ctrl+C to stop streaming.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

# Check dependencies
try:
    import pylsl
    HAS_PYLSL = True
except ImportError:
    HAS_PYLSL = False

try:
    from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, EChannelType
    HAS_PYSHIMMER = True
except ImportError:
    HAS_PYSHIMMER = False


class ShimmerEDAStreamer:
    """Streams Shimmer GSR3 EDA data to LSL."""
    
    def __init__(
        self,
        port: str,
        stream_name: str = "ShimmerEDA",
        stream_type: str = "EDA",
        source_id: Optional[str] = None,
    ):
        self.port = port
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.source_id = source_id or f"shimmer_gsr_{port}"
        
        self.shimmer: Optional[ShimmerBluetooth] = None
        self.outlet: Optional[pylsl.StreamOutlet] = None
        self.running = False
        self.samples_sent = 0
        self.start_time = 0.0
        
        # Shimmer GSR3 nominal sample rate
        self.sample_rate = 51.2
    
    def connect(self) -> bool:
        """Connect to Shimmer device and create LSL outlet."""
        print(f"Connecting to Shimmer on {self.port}...")
        
        try:
            self.shimmer = ShimmerBluetooth(self.port)
            self.shimmer.initialize()
            
            # The GSR sensor is enabled by configuring the data types to include GSR_RAW
            # pyshimmer uses set_sensors() which expects EChannelType flags
            # For GSR3+, we need GSR_RAW channel
            self.shimmer.set_sensors([EChannelType.GSR_RAW, EChannelType.INTERNAL_ADC_13])
            
            # Get actual sample rate
            actual_rate = self.shimmer.get_sampling_rate()
            if actual_rate:
                self.sample_rate = actual_rate
            
            print(f"Connected to Shimmer (sample rate: {self.sample_rate} Hz)")
            
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
        ch.append_child_value("label", "GSR")
        ch.append_child_value("unit", "microsiemens")
        ch.append_child_value("type", "EDA")
        
        # Add device metadata
        device = info.desc().append_child("device")
        device.append_child_value("manufacturer", "Shimmer")
        device.append_child_value("model", "GSR3")
        device.append_child_value("port", self.port)
        
        self.outlet = pylsl.StreamOutlet(info)
        print(f"LSL outlet created: {self.stream_name}")
        
        return True
    
    def _stream_callback(self, packet) -> None:
        """Callback for Shimmer data packets (uses DataPacket from pyshimmer)."""
        if not self.running or self.outlet is None:
            return
        
        # Extract GSR value from packet
        # The packet is a DataPacket with channels dict
        try:
            # Try to get GSR_RAW channel
            if hasattr(packet, 'channels') and EChannelType.GSR_RAW in packet.channels:
                gsr_value = packet.channels[EChannelType.GSR_RAW]
            elif isinstance(packet, dict):
                # Fallback for dict-style packets
                gsr_value = packet.get(EChannelType.GSR_RAW, packet.get("GSR_RAW", packet.get("GSR", 0.0)))
            else:
                # Try accessing as attribute
                gsr_value = getattr(packet, 'gsr_raw', getattr(packet, 'GSR_RAW', 0.0))
            
            # Push to LSL
            self.outlet.push_sample([float(gsr_value)])
            self.samples_sent += 1
        except Exception as e:
            if self.samples_sent == 0:
                print(f"WARNING: Error extracting GSR from packet: {e}", file=sys.stderr)
    
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
  python scripts/stream_shimmer_eda.py --port COM5 --name ShimmerEDA

The script creates an LSL stream with type="EDA" that can be recorded
alongside OpenMATB markers and EEG using LabRecorder.
""",
    )
    
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port for Shimmer Bluetooth connection (e.g., COM5, /dev/tty.Shimmer-GSR3-RN42)",
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
    
    # Create and start streamer
    streamer = ShimmerEDAStreamer(
        port=args.port,
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
    
    streamer.start_streaming()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
