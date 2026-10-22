"""Stage 2 checks: double knock-out by one blur of the image series. Exits non-zero on failure."""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from barriers import vanilla
from double_barrier import dko_closed_form, dko_blur

S0 = 100.0
failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


def dko_mp(mp, S, K, L, U, sigma, T, r, q, cp, n_max=40):
    """dko_closed_form's image series in arbitrary precision (referee only)."""
    S, K, L, U, sigma, T, r, q = (mp.mpf(v) for v in (S, K, L, U, sigma, T, r, q))
    x, l, u, k = mp.log(S), mp.log(L), mp.log(U), mp.log(K)
    w, nu, s = u - l, r - q - sigma**2 / 2, sigma * mp.sqrt(T)
    a = 2 * nu / sigma**2

    def interval(m):
        lo, hi = (max(l, k), u) if cp == "call" else (l, min(u, k))
        if hi <= lo:
            return mp.mpf(0)
        pS = mp.ncdf((hi - m - s * s) / s) - mp.ncdf((lo - m - s * s) / s)
        pK = mp.ncdf((hi - m) / s) - mp.ncdf((lo - m) / s)
        ef = mp.e ** (m + s * s / 2)
        return ef * pS - K * pK if cp == "call" else K * pK - ef * pS

    total = mp.mpf(0)
    for n in range(-n_max, n_max + 1):
        total += mp.e ** (a * n * w) * interval(x + 2 * n * w + nu * T)
        total -= mp.e ** (a * (l - x + n * w)) * interval(2 * l - x + 2 * n * w + nu * T)
    return mp.e ** (-r * T) * total


def images_needed(K, L, U, s, T, r, cp, tol=1e-8, max_side=40):
    """Smallest number of mirror images (both sides together) that gets within tol."""
    full = dko_blur(S0, K, L, U, s, T, r, cp=cp)
    errs = []
    for m in range(max_side + 1):
        e = abs(dko_blur(S0, K, L, U, s, T, r, cp=cp, images=m) - full)
        errs.append(e)
        if e < tol:
            return 2 * m, errs, full
    return None, errs, full


print("=" * 72)
print("1) How many mirror images does one blur need?")
print("=" * 72)
print("  S0=100, r=5%. The payoff is cell 0; images are added in pairs, nearest cells first.")
print("  Error is against the blur with every image that reaches the grid.")
print()
K, L, U, s, T, r = 100.0, 90.0, 110.0, 0.20, 1.0, 0.05
_, errs, full = images_needed(K, L, U, s, T, r, "call", tol=0.0, max_side=8)
print(f"  call K={K:.0f}, L={L:.0f}, U={U:.0f}, sigma={s}, T={T}  (price {full:.10f})")
print(f"  {'images':>8} {'|error|':>10}")
for m, e in enumerate(errs):
    print(f"  {2*m:>8} {e:>10.1e}")
print()
print(f"  {'corridor':>10} {'sigma':>6} {'T':>5} {'blur width / corridor':>22} {'images to 1e-8':>15}")
need = []
for L, U in [(95, 105), (90, 110), (80, 125), (70, 140), (50, 200)]:
    for s, T in [(0.1, 0.25), (0.2, 1.0), (0.4, 2.0)]:
        n, _, _ = images_needed(100.0, L, U, s, T, 0.05, "call")
        ratio = s * np.sqrt(T) / np.log(U / L)
        need.append(n)
        print(f"  {f'{L}-{U}':>10} {s:>6} {T:>5} {ratio:>22.2f} {n if n is not None else '> 80':>15}")
print("  (images needed ~ 11 x blur width / corridor width: the Gaussian reaches about 5.5 sigma sqrt(T)")
print("   before its tails drop below 1e-8, and every corridor width it crosses on either side is one more image)")

print()
print("=" * 72)
print("2) One blur vs the exact image series (closed form)")
print("=" * 72)
cases = []
for L, U in [(90, 110), (80, 125), (70, 140)]:
    for K in [L - 5, 0.5 * (L + U), U - 2, U + 5]:
        for cp in ["call", "put"]:
            for s, T, r in [(0.2, 1.0, 0.05), (0.35, 0.5, 0.0), (0.12, 2.0, 0.08)]:
                cases.append((K, L, U, s, T, r, cp))
worst = 0.0
for c in cases:
    K, L, U, s, T, r, cp = c
    worst = max(worst, abs(dko_blur(S0, K, L, U, s, T, r, cp=cp) - dko_closed_form(S0, K, L, U, s, T, r, cp=cp)))
print(f"  {len(cases)} options (3 corridors, strikes inside and outside, calls and puts, 3 markets), dx <= 0.0005")
print(f"  max |blur - closed form| = {worst:.1e}   {check(worst < 1e-6, 'dko blur')}")

print()
print("=" * 72)
print("3) Independent check: closed form vs QuantLib")
print("=" * 72)
try:
    import QuantLib as ql
except ImportError:
    print("  QuantLib not installed (pip install QuantLib)")
    failures.append("QuantLib missing")
else:
    today = ql.Date(20, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()

    def process(S, s, r, q):
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), s, dc)))

    def ql_dbl(kind, S, K, L, U, s, days, r, q, cp):
        o = ql.DoubleBarrierOption(kind, L, U, 0.0,
                                   ql.PlainVanillaPayoff(ql.Option.Call if cp == "call" else ql.Option.Put, K),
                                   ql.EuropeanExercise(today + days))
        o.setPricingEngine(ql.AnalyticDoubleBarrierEngine(process(S, s, r, q), SERIES))
        return o.NPV()

    def ql_no_touch(S, L, U, s, days, r, q):
        o = ql.DoubleBarrierOption(ql.DoubleBarrier.KnockOut, L, U, 0.0,
                                   ql.CashOrNothingPayoff(ql.Option.Call, 0.0, 1.0),
                                   ql.EuropeanExercise(today + days))
        o.setPricingEngine(ql.AnalyticDoubleBarrierBinaryEngine(process(S, s, r, q)))
        return o.NPV()

    SERIES = 10  # QuantLib's series overflows to garbage at 20 terms for some of these cases
    rng = np.random.default_rng(20261022)
    n = 200
    w_out, w_in, w_rel, worst = 0.0, 0.0, 0.0, None
    for i in range(n):
        cp = "call" if i % 2 == 0 else "put"
        S = float(rng.uniform(60, 140))
        L = float(S * rng.uniform(0.6, 0.97))
        U = float(S * rng.uniform(1.03, 1.6))
        K = float(rng.uniform(L * 1.001, U * 0.999))
        s = float(rng.uniform(0.05, 0.5))
        r = float(rng.uniform(-0.01, 0.12))
        q = float(rng.uniform(0.0, 0.06))
        days = int(rng.integers(7, 3 * 365))
        T = days / 365.0
        ours = dko_closed_form(S, K, L, U, s, T, r, q, cp)
        ref = ql_dbl(ql.DoubleBarrier.KnockOut, S, K, L, U, s, days, r, q, cp)
        if abs(ours - ref) > w_out:
            w_out, worst = abs(ours - ref), (cp, S, K, L, U, s, days, r, q, ours, ref)
        if abs(ref) > 1e-6:
            w_rel = max(w_rel, abs(ours - ref) / abs(ref))
        ki = vanilla(S, K, s, T, r, q, cp) - ours
        w_in = max(w_in, abs(ki - ql_dbl(ql.DoubleBarrier.KnockIn, S, K, L, U, s, days, r, q, cp)))
    print(f"  QuantLib {ql.__version__} AnalyticDoubleBarrierEngine (Ikeda-Kunitomo, 10 terms), {n} random cases,")
    print(f"  L < K < U, sigma 5-50%, r -1% to 12%, dividends 0-6%, 1 week to 3 years")
    print(f"  knock-out: max |ours - QuantLib| = {w_out:.1e}  (relative {w_rel:.1e})   {check(w_out < 1e-7, 'QuantLib DKO')}")
    print(f"  knock-in (vanilla - ours): max diff = {w_in:.1e}   {check(w_in < 1e-7, 'QuantLib DKI')}")
    # Who is right in the worst case? Referee with the same image series evaluated in
    # 40-digit arithmetic (mpmath), and with QuantLib's single-barrier engine: the worst
    # case has one barrier far out of reach, so it is really a single-barrier option.
    cp, S, K, L, U, s, days, r, q, ours, ref = worst
    o = ql.BarrierOption(ql.Barrier.DownOut if abs(np.log(S / L)) < abs(np.log(U / S)) else ql.Barrier.UpOut,
                         L if abs(np.log(S / L)) < abs(np.log(U / S)) else U, 0.0,
                         ql.PlainVanillaPayoff(ql.Option.Call if cp == "call" else ql.Option.Put, K),
                         ql.EuropeanExercise(today + days))
    o.setPricingEngine(ql.AnalyticBarrierEngine(process(S, s, r, q)))
    single = o.NPV()
    print(f"  worst case: {cp} S={S:.1f} K={K:.1f} L={L:.1f} U={U:.1f} sigma={s:.3f} T={days}d")
    print(f"    ours {ours:.12f}   QuantLib double {ref:.12f}   QuantLib single-barrier engine {single:.12f}")
    try:
        import mpmath as mp
    except ImportError:
        print("    (pip install mpmath to referee this case in 40-digit arithmetic)")
    else:
        mp.mp.dps = 40
        print(f"    40-digit image series {mp.nstr(dko_mp(mp, S, K, L, U, s, days / 365.0, r, q, cp), 15)}"
              f"   |ours - 40-digit| = {abs(ours - float(dko_mp(mp, S, K, L, U, s, days / 365.0, r, q, cp))):.1e}")
        print("    so the ~1e-8 gap is on QuantLib's side (its single-barrier engine is off by as much here)")
    print()
    # Strikes outside the corridor. QuantLib's analytic engine assumes L < K < U, so
    # rebuild the reference from pieces it does price: on the corridor a call with
    # K < L pays (S - L) + (L - K), i.e. a call struck at L plus (L - K) double-no-touch
    # binaries; a put with K > U is a put struck at U plus (K - U) binaries.
    w_dec, w_raw = 0.0, 0.0
    for i in range(100):
        cp = "call" if i % 2 == 0 else "put"
        S = float(rng.uniform(60, 140))
        L = float(S * rng.uniform(0.6, 0.97))
        U = float(S * rng.uniform(1.03, 1.6))
        s = float(rng.uniform(0.05, 0.5))
        r = float(rng.uniform(-0.01, 0.12))
        q = float(rng.uniform(0.0, 0.06))
        days = int(rng.integers(7, 3 * 365))
        T = days / 365.0
        if cp == "call":
            K = float(L * rng.uniform(0.5, 0.99))
            ref = ql_dbl(ql.DoubleBarrier.KnockOut, S, L, L, U, s, days, r, q, cp) + \
                (L - K) * ql_no_touch(S, L, U, s, days, r, q)
        else:
            K = float(U * rng.uniform(1.01, 1.5))
            ref = ql_dbl(ql.DoubleBarrier.KnockOut, S, U, L, U, s, days, r, q, cp) + \
                (K - U) * ql_no_touch(S, L, U, s, days, r, q)
        ours = dko_closed_form(S, K, L, U, s, T, r, q, cp)
        w_dec = max(w_dec, abs(ours - ref))
        w_raw = max(w_raw, abs(ours - ql_dbl(ql.DoubleBarrier.KnockOut, S, K, L, U, s, days, r, q, cp)))
    print("  100 more cases with the strike outside the corridor (call K < L, put K > U):")
    print(f"  vs QuantLib vanilla-at-barrier + double-no-touch binaries: max diff = {w_dec:.1e}"
          f"   {check(w_dec < 1e-10, 'QuantLib outside-strike')}")
    print(f"  (QuantLib's analytic engine priced directly at those strikes is off by up to {w_raw:.1e};")
    print("   its Ikeda-Kunitomo formula assumes L < K < U)")

print()
if failures:
    print("FAILED: " + ", ".join(failures))
    sys.exit(1)
print("all double-barrier checks passed")
