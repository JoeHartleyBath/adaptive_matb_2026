"""Quick EEG channel metadata inspector - run once, don't commit."""
import os
import pylsl
import xml.etree.ElementTree as ET

os.environ.setdefault("LSL_LOGLEVEL", "-3")


def _inspect_stream(info, idx):
    print("=" * 75)
    print("STREAM {}  name={!r}  source_id={!r}".format(idx, info.name(), info.source_id()))
    print("=" * 75)
    print("Type    :", info.type())
    print("Channels:", info.channel_count())
    print("Rate    :", info.nominal_srate(), "Hz")
    print("Format  :", info.channel_format())
    print()

    root = ET.fromstring(info.as_xml())
    channels = root.findall(".//channel")
    print("Channels in XML desc:", len(channels))
    print()

    if not channels:
        print("  (no channel metadata in XML descriptor)")
        print()
        return

    print("{:<5} {:<20} {:<12} {:<10}  {}".format("#", "Label", "Type", "Unit", "Location (X,Y,Z)"))
    print("-" * 75)
    missing_loc = 0
    for i, ch in enumerate(channels):
        label = (ch.findtext("label") or ch.findtext("name") or "").strip() or "CH{}".format(i + 1)
        ch_type = (ch.findtext("type") or "").strip()
        unit = (ch.findtext("unit") or "").strip()
        loc = ch.find("location")
        if loc is not None:
            x = loc.findtext("X") or loc.findtext("x") or "?"
            y = loc.findtext("Y") or loc.findtext("y") or "?"
            z = loc.findtext("Z") or loc.findtext("z") or "?"
            loc_str = "({}, {}, {})".format(x, y, z)
        else:
            loc_str = "NO LOCATION"
            missing_loc += 1
        print("{:<5} {:<20} {:<12} {:<10}  {}".format(i + 1, label, ch_type, unit, loc_str))
    print()
    print("Summary: {} channels total, {} missing location info".format(len(channels), missing_loc))
    print()


# Resolve ALL EEG streams - use resolve_streams+filter rather than
# resolve_byprop, which returns as soon as minimum=1 responds and silently
# drops any amp that answers slightly later.
all_streams = pylsl.resolve_streams(wait_time=5.0)
streams = [s for s in all_streams if s.type() == "EEG"]
if not streams:
    print("NO EEG STREAM FOUND")
    raise SystemExit(1)

print("Found {} EEG stream(s) on the network.\n".format(len(streams)))
for idx, info in enumerate(streams, start=1):
    _inspect_stream(info, idx)

if len(streams) > 1:
    total_ch = sum(s.channel_count() for s in streams)
    print(">>> {} separate streams found — eego is likely streaming each amp independently.".format(len(streams)))
    print(">>> Total channels across all streams: {}".format(total_ch))
    print(">>> To get a single combined stream, configure one amp as Master and the other as Slave in eego,")
    print(">>> then restart the LSL output so they appear as one merged stream.")
