"""Final verification table: log-scale design, delta=0.8, constrained comms/pump maxima."""
import math

D_MIN, D_MAX = -0.8, 1.8
TOTAL = D_MAX - D_MIN   # 2.6
EFF = 54.0              # schedulable window per 60-s block (60 - 2x3 guard)
DELTA = 0.8


def _t(d):
    return max(0.0, min(1.0, (d - D_MIN) / TOTAL))


def log_hz(min_hz, max_hz, d):
    return min_hz * (max_hz / min_hz) ** _t(d)


def track_ms(d):
    return max(1.0, 50.0 + (10.0 - 50.0) * d)


def joystick(d):
    return max(0.1, 3.0 + (1.0 - 3.0) * d)


# Log-scale event rates — anchored at d=-0.8 (min) and d=+1.8 (max)
MIN_HZ       = 1.0 / EFF   # 1 event/block at d=-0.8 (floor, all subtasks)
SYSMON_MAX   = 18.0 / EFF  # 18/block at d=+1.8  → ~6x H/L  (no physical cap)
COMMS_MAX    =  2.0 / EFF  #  2/block at d=+1.8  → serial prompt: physical max=floor(54/19)=2
PUMP_MAX     =  4.0 / EFF  #  4/block at d=+1.8  → pump slot 11s: physical max=floor(54/11)=4

# Drain: log scale, 50 ml/min at d=-0.8 -> 1200 ml/min at d=+1.8 (physical network max)
DRAIN_MIN, DRAIN_MAX = 50.0, 1200.0

def drain(d):
    return DRAIN_MIN * (DRAIN_MAX / DRAIN_MIN) ** _t(d)


def ev(min_hz, max_hz, d):
    return round(log_hz(min_hz, max_hz, d) * EFF)


# ── 1. Values at key d points ──────────────────────────────────────────────
print("=== Proposed values at key d points ===")
print(f"  {'d':>5}  {'track_ms':>8}  {'joystick':>8}  {'lights':>6}  {'comms':>5}  {'pumps':>5}  {'drain':>10}")
for d in [-0.8, 0.0, 0.5, 1.0, 1.8]:
    print(
        f"  {d:+5.2f}  {track_ms(d):8.0f}  {joystick(d):8.2f}"
        f"  {ev(MIN_HZ, SYSMON_MAX, d):6}  {ev(MIN_HZ, COMMS_MAX, d):5}"
        f"  {ev(MIN_HZ, PUMP_MAX, d):5}  {drain(d):10.0f}"
    )

# ── 2. Range within staircase calibration (d_final in [0,1]) ───────────────
print()
print("=== Subtask range WITHIN staircase calibration  (d_final converges between 0 and 1) ===")
print(f"  MOD level (d = d_final) spans d=0.0 to d=1.0")
print(f"  LOW level (d_final - 0.8) spans d=-0.8 to d=+0.2")
print(f"  HIGH level (d_final + 0.8) spans d=+0.8 to d=+1.8")
print()
print(f"  {'Subtask':<18}  {'at d=0.0':>10}  {'at d=1.0':>10}  {'abs min d=-0.8':>14}  {'abs max d=+1.8':>14}")

rows = [
    ("track_ms",        lambda d: f"{track_ms(d):.0f} ms"),
    ("joystick_force",  lambda d: f"{joystick(d):.2f}"),
    ("lights /block",   lambda d: f"{ev(MIN_HZ, SYSMON_MAX, d)}"),
    ("scales /block",   lambda d: f"{ev(MIN_HZ, SYSMON_MAX, d)}"),
    ("comms /block",    lambda d: f"{ev(MIN_HZ, COMMS_MAX, d)}"),
    ("pumps /block",    lambda d: f"{ev(MIN_HZ, PUMP_MAX, d)}"),
    ("drain ml/min",    lambda d: f"{drain(d):.0f}"),
]
for name, fn in rows:
    print(f"  {name:<18}  {fn(0.0):>10}  {fn(1.0):>10}  {fn(-0.8):>14}  {fn(1.8):>14}")

# ── 3. H/L separation inside each calibration session ─────────────────────
print()
print("=== H/L separation inside calibration session (delta=0.8) ===")
print(f"  {'Case':<22}  {'lights':^14}  {'comms':^12}  {'pumps':^12}  {'drain':^18}  {'track':^16}")
for label, df in [("Floor   d_fin=0.0", 0.0), ("Typical d_fin=0.65", 0.65), ("Ceiling d_fin=1.0", 1.0)]:
    dl, dh = df - DELTA, df + DELTA
    sl, sh = ev(MIN_HZ, SYSMON_MAX, dl), ev(MIN_HZ, SYSMON_MAX, dh)
    cl, ch = ev(MIN_HZ, COMMS_MAX, dl),  ev(MIN_HZ, COMMS_MAX, dh)
    pl, ph = ev(MIN_HZ, PUMP_MAX, dl),   ev(MIN_HZ, PUMP_MAX, dh)
    drl, drh = drain(dl), drain(dh)
    tl, th = track_ms(dl), track_ms(dh)
    print(
        f"  {label:<22}  {sl}->{sh} ({sh//max(sl,1)}x){'':<4}  "
        f"{cl}->{ch} ({ch//max(cl,1)}x){'':<4}  "
        f"{pl}->{ph} ({ph//max(pl,1)}x){'':<4}  "
        f"{drl:.0f}->{drh:.0f} ({drh//max(drl,1)}x){'':<2}  "
        f"{tl:.0f}ms->{th:.0f}ms"
    )
