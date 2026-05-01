"""
Full pre-implementation subtask range table.


Design constraints:
  - Full usable d range: [-0.3, 1.3]  (floor participant at d_fin=0  uses d=-0.3 as LOW;
                                        ceiling participant at d_fin=1 uses d=+1.3 as HIGH)
  - Minimum events at d=-0.3:  1 per 54-s block  ("task structure same for everyone")
  - Maximum events at d=+1.3:  6 per 54-s block  (6x total range)
  - delta = 0.30

Key insight:
  Linear scale -> absolute step size constant -> fold-change varies with position
  Log scale    -> fold-change constant for every participant regardless of d_final
"""
import math

EFF = 54.0
delta = 0.30
D_MIN, D_MAX = -0.3, 1.3   # full usable d range
total = D_MAX - D_MIN       # 1.6

MIN_HZ  = 1.0 / EFF        # 1 event / block
MAX_HZ  = 6.0 / EFF        # 6 events / block

# ── Linear anchors ──────────────────────────────────────────────
def fit_linear(lo, hi):
    """Return (a, b) for lerp(a,b,d) s.t. value(-0.3)=lo, value(1.3)=hi."""
    a = (1.3 * lo + 0.3 * hi) / 1.6
    b = (0.3 * lo + 1.3 * hi) / 1.6
    return a, b

def lin_rate(a, b, d):
    return max(0.0, a + (b - a) * d)

# ── Log (exponential) scale ──────────────────────────────────────
def log_rate(d):
    """rate = MIN_HZ * (MAX_HZ/MIN_HZ)^((d - D_MIN) / total)"""
    t = (d - D_MIN) / total
    return MIN_HZ * (MAX_HZ / MIN_HZ) ** t

def ev_lin(a, b, d):
    return max(1, round(lin_rate(a, b, d) * EFF))

def ev_log(d):
    return max(1, round(log_rate(d) * EFF))

a_lin, b_lin = fit_linear(MIN_HZ, MAX_HZ)

# ── Report ───────────────────────────────────────────────────────
print(f"Linear anchors: a={a_lin:.5f}  b={b_lin:.5f}")
print(f"Log scale:      rate(d) = {MIN_HZ:.5f} * ({MAX_HZ/MIN_HZ:.1f})^((d+0.3)/1.6)")
print()
print("Log-scale H/L fold change per participant type (CONSTANT regardless of d_fin):")
fold_log = (MAX_HZ / MIN_HZ) ** (2 * delta / total)
print(f"  fold = {MAX_HZ/MIN_HZ:.0f}^(2*{delta}/{total:.1f}) = {fold_log:.2f}x  (same for all participants)")
print()

cases = [
    ("Floor   d_fin=0.00", 0.00),
    ("Easy    d_fin=0.30", 0.30),
    ("Typical d_fin=0.65", 0.65),
    ("Ceiling d_fin=1.00", 1.00),
]

def track_ms(d):
    return max(1.0, 50.0 + (10.0 - 50.0) * d)

for scheme, ev_fn in [
    ("LINEAR scale", lambda d: ev_lin(a_lin, b_lin, d)),
    ("LOG scale   ", ev_log),
]:
    print(f"── {scheme}  (delta=0.30) ──────────────────────────────────")
    for label, d_fin in cases:
        d_l, d_m, d_h = d_fin - delta, d_fin, d_fin + delta
        el, em, eh = ev_fn(d_l), ev_fn(d_m), ev_fn(d_h)
        tm = track_ms(d_m)
        print(f"  {label}   LOW={el} MOD={em} HIGH={eh}  ratio={eh/el:.1f}x   (track@MOD={tm:.0f}ms)")
    print()
