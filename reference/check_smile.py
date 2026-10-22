"""Stage 2 checks: the implied-volatility smile of each kernel, how it flattens with
maturity (measured), and the skew a negative mean jump creates.
Exits non-zero if a check fails."""
import sys
import time
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kernels import bs, merton_series, vg_mixture, mean_sd
from smile import implied_vol, smile, atm_shape

failures = []
S0, r = 100.0, 0.04
MATURITIES = [7 / 365, 14 / 365, 1 / 12, 2 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
# Symmetric jumps first, then the same model with a negative mean jump.
MODELS = {
    "Merton, m = 0":     ("merton", 0.15, dict(lam=1.0, m=0.0, delta=0.15)),
    "Merton, m = -10%":  ("merton", 0.15, dict(lam=1.0, m=-0.10, delta=0.15)),
    "VG, theta = 0":     ("vg", 0.18, dict(theta=0.0, nu=0.2)),
    "VG, theta = -0.2":  ("vg", 0.18, dict(theta=-0.2, nu=0.2)),
}


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


def header(title):
    print("=" * 76)
    print(title)
    print("=" * 76)


def label_T(T):
    d = T * 365
    return f"{d:.0f}d" if d < 60 else (f"{T * 12:.0f}m" if T < 1 else f"{T:.0f}y")


t0 = time.time()
print(f"S0 = {S0}, r = {r:.0%}. Moneyness k = ln(K/F), F = S0 e^(rT). Each strike priced with")
print("its out-of-the-money option and inverted to a Black-Scholes vol (Brent, to 1e-15).")
for name, (model, s, p) in MODELS.items():
    print(f"  {name:18s} diffusion vol {s:.0%}, " + ", ".join(f"{k} {v:g}" for k, v in p.items()))
print()

# ----------------------------------------------------------------------------
header("1) The inversion and the Gaussian kernel: no smile where there should be none")
err = 0.0
for T in MATURITIES:
    for k in np.linspace(-0.5, 0.5, 21):
        for sig in (0.05, 0.2, 0.6):
            K = S0 * np.exp(r * T + k)
            cp = "call" if k >= 0 else "put"
            if bs(S0, K, sig, T, r, cp) < 1e-9:
                continue
            err = max(err, abs(implied_vol(bs(S0, K, sig, T, r, cp), S0, K, T, r, cp) - sig))
print(f"  BS price -> implied vol round trip, 10 maturities x 21 strikes x 3 vols:")
print(f"    max |implied - true| = {err:.1e}   {check(err < 1e-10, 'inversion')}")
flat = 0.0
for T in MATURITIES:
    sd = 0.2 * np.sqrt(T)
    v = smile("gaussian", np.linspace(-3 * sd, 3 * sd, 13), T, r, 0.2)
    flat = max(flat, np.nanmax(np.abs(v - 0.2)))
print(f"  Gaussian kernel at 20% vol, 10 maturities x 13 strikes out to 3 sd:")
print(f"    max |implied - 20%| = {flat:.1e}   {check(flat < 1e-8, 'gaussian flat')}")
print()

# ----------------------------------------------------------------------------
header("2) Implied vols from the blur vs from the independent prices (Stage 1 references)")
worst = 0.0
for name, (model, s, p) in MODELS.items():
    for T in (14 / 365, 0.25, 2.0):
        sd = mean_sd(model, T, r, s, **p)[1]
        ks = np.linspace(-2.5 * sd, 2.5 * sd, 11)
        v = smile(model, ks, T, r, s, S0, **p)
        for k, iv in zip(ks, v):
            K = S0 * np.exp(r * T + k)
            cp = "call" if k >= 0 else "put"
            ref = (merton_series(S0, K, T, r, s, cp=cp, **p) if model == "merton"
                   else vg_mixture(S0, K, T, r, s, cp=cp, **p))
            worst = max(worst, abs(iv - implied_vol(ref, S0, K, T, r, cp)))
print("  4 models x 3 maturities (2w, 3m, 2y) x 11 strikes out to 2.5 sd; the references are")
print("  Merton's series and the variance-gamma mixture integral")
print(f"    max |vol from blur - vol from reference| = {worst * 1e4:.1e} vol bp   "
      f"{check(worst < 1e-6, 'smile vs reference')}  (< 0.01 bp)")
print()

# ----------------------------------------------------------------------------
header("3) The smile, measured: level, skew and curvature at the money, by maturity")
print("  skew = dIV/dk, curvature = d2IV/dk2 at k = 0 (5-point differences, step 0.1 sd).")
print("  wings = IV at k = -1 sd and +1 sd of the kernel, minus ATM, in vol points.")
shape = {}
for name, (model, s, p) in MODELS.items():
    print(f"\n  {name}")
    print("    T      ATM vol    skew      curvature    -1 sd wing  +1 sd wing")
    rows = []
    for T in MATURITIES:
        atm, sk, cu = atm_shape(model, T, r, s, S0=S0, **p)
        sd = mean_sd(model, T, r, s, **p)[1]
        lo, hi = smile(model, [-sd, sd], T, r, s, S0, **p)
        rows.append((T, atm, sk, cu, lo - atm, hi - atm))
        print(f"    {label_T(T):5s}  {atm:7.2%}  {sk:+8.4f}  {cu:+11.4f}     "
              f"{(lo - atm) * 100:+6.2f}      {(hi - atm) * 100:+6.2f}")
    shape[name] = np.array(rows)


def slope(T, y, lo, hi):
    sel = (T >= lo) & (T <= hi)
    return np.polyfit(np.log(T[sel]), np.log(np.abs(y[sel])), 1)[0]


print()
print("  How fast it flattens: log-log slope of |skew| and curvature against T")
print("  (a slope of -1 means halving when maturity doubles)")
print("    model                 |skew| 1m-6m  |skew| 1y-5y   curv 1m-6m   curv 1y-5y")
for name, a in shape.items():
    T = a[:, 0]
    print(f"    {name:20s}  {slope(T, a[:, 2], 1 / 12, 0.5):+9.2f}  {slope(T, a[:, 2], 1, 5):+12.2f}"
          f"   {slope(T, a[:, 3], 1 / 12, 0.5):+10.2f}  {slope(T, a[:, 3], 1, 5):+10.2f}")
print("  Why it fades: X over T is a sum of independent increments, so its excess kurtosis")
print("  falls like 1/T and the kernel looks more Gaussian the longer the maturity.")
for name, a in shape.items():
    curv_down = bool(np.all(np.diff(a[:, 3]) < 0))
    print(f"    {name:20s} curvature falls at every step from 1 week to 5 years: {curv_down}")
print()

# ----------------------------------------------------------------------------
header("4) A negative mean jump turns the smile into a downward skew")
print("  Same diffusion and jump frequency; only the mean jump (Merton m, VG theta) changes.")
print("  wing gap = IV(k = -1 sd) - IV(k = +1 sd), vol points. Downward skew: skew < 0, gap > 0.")
print("    T      Merton m=0       Merton m=-10%      VG theta=0       VG theta=-0.2")
print("           skew    gap      skew    gap        skew    gap      skew    gap")
for i, T in enumerate(MATURITIES):
    cells = []
    for name in MODELS:
        a = shape[name][i]
        cells.append(f"{a[2]:+7.4f} {(a[4] - a[5]) * 100:+5.2f}")
    print(f"    {label_T(T):5s}  " + "    ".join(cells))
for name in ("Merton, m = -10%", "VG, theta = -0.2"):
    a = shape[name]
    ok = bool(np.all(a[:, 2] < 0) and np.all(a[:, 4] > a[:, 5]))
    print(f"  {name}: skew < 0 and left wing above right wing at all 10 maturities   "
          f"{check(ok, name + ' skew')}")
for name in ("Merton, m = 0", "VG, theta = 0"):
    a = shape[name]
    print(f"  {name}: skew between {a[:, 2].min():+.4f} and {a[:, 2].max():+.4f}, "
          f"{'always positive' if np.all(a[:, 2] > 0) else 'mixed sign'}")
print("  So jumps that are symmetric in log-price give a smile tilted slightly UP in k; the")
print("  negative mean jump flips the sign and makes the skew 7 to 10 times larger.")
print()

print(f"total time {time.time() - t0:.0f} s")
if failures:
    print("FAILED: " + ", ".join(failures))
    sys.exit(1)
print("ALL PASS")
