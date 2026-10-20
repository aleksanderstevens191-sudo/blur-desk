"""Stage 1 checks for the single-barrier family. Exits non-zero if a check fails."""
import sys
import time
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from barriers import (
    TYPES, parse, vanilla, barrier_closed_form, mirror_price, vanilla_blur,
    stepping_prices,
)

S0 = 100.0
failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


# Contracts: every type, strike on both sides of the barrier, three markets.
MARKETS = [(0.20, 1.0, 0.05), (0.30, 0.5, 0.08), (0.15, 2.0, 0.01)]
BARRIERS = {"down": (90.0, [80.0, 100.0]), "up": (115.0, [100.0, 125.0])}


def contracts():
    for sigma, T, r in MARKETS:
        for kind in TYPES:
            H, Ks = BARRIERS[parse(kind)[0]]
            for K in Ks:
                yield kind, K, H, sigma, T, r


print("=" * 72)
print("1) Weighted mirror (b), one blur, vs Reiner-Rubinstein closed form")
print("=" * 72)
print(f"  S0={S0}; down barrier H=90 (K=80, 100), up barrier H=115 (K=100, 125)")
print("  grid anchored so the barrier is a node; strikes fall between nodes")
for dx in [0.002, 0.001, 0.0005]:
    errs = [abs(mirror_price(k, S0, K, H, s, T, r, dx=dx) - barrier_closed_form(k, S0, K, H, s, T, r))
            for k, K, H, s, T, r in contracts()]
    print(f"  dx={dx:<7} max |blur - closed form| over 48 options: {max(errs):.2e}")
dx = 0.0005
print()
print(f"  dx={dx}, by type (max over 2 strikes x 3 markets):")
print(f"  {'type':<20} {'max error':>10}   {'worst case (sigma, T, r, K)':<28} check < 1e-6")
rows = {}
for k, K, H, s, T, r in contracts():
    e = abs(mirror_price(k, S0, K, H, s, T, r, dx=dx) - barrier_closed_form(k, S0, K, H, s, T, r))
    if e >= rows.get(k, (-1,))[0]:
        rows[k] = (e, (s, T, r, K))
for k in TYPES:
    e, case = rows[k]
    print(f"  {k:<20} {e:>10.2e}   {str(case):<28} {check(e < 1e-6, f'(b) {k}')}")
print()
print("  vanilla calls with the same sampler (no barrier), for scale:")
for K in [80.0, 100.0, 107.3]:
    e = abs(vanilla_blur(S0, K, 0.2, 1.0, 0.05, dx=dx) - vanilla(S0, K, 0.2, 1.0, 0.05))
    print(f"    K={K:<6} |blur - BS| = {e:.1e}")

print()
print("=" * 72)
print("2) In + out = vanilla")
print("=" * 72)
cf_par, blur_par = 0.0, 0.0
for s, T, r in MARKETS:
    for ud, (H, Ks) in BARRIERS.items():
        for K in Ks:
            for cp in ["call", "put"]:
                v = vanilla(S0, K, s, T, r, cp=cp)
                ki = barrier_closed_form(f"{ud}-and-in {cp}", S0, K, H, s, T, r)
                ko = barrier_closed_form(f"{ud}-and-out {cp}", S0, K, H, s, T, r)
                cf_par = max(cf_par, abs(ki + ko - v))
                bi = mirror_price(f"{ud}-and-in {cp}", S0, K, H, s, T, r)
                bo = mirror_price(f"{ud}-and-out {cp}", S0, K, H, s, T, r)
                blur_par = max(blur_par, abs(bi + bo - vanilla_blur(S0, K, s, T, r, cp=cp, anchor=H)))
print(f"  closed forms (knock-ins from their own formulas): max |in + out - BS| = {cf_par:.1e}"
      f"   {check(cf_par < 1e-10, 'closed-form parity')}")
print(f"  weighted-mirror blurs:                          max |in + out - vanilla blur, same grid| = {blur_par:.1e}"
      f"   {check(blur_par < 1e-10, 'blur parity')}")

print()
print("=" * 72)
print("3) Plain mirror (a), no weight: right only when r = sigma^2/2")
print("=" * 72)
print("  down-and-out call, K=100, H=90, sigma=0.2 (sigma^2/2 = 0.02), T=1")
print(f"  {'r':>6} {'2r/s^2-1':>9} {'closed form':>12} {'plain (a)':>11} {'error (a)':>10} {'weighted (b)':>13} {'error (b)':>10}")
for r in [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]:
    cf = barrier_closed_form("down-and-out call", S0, 100, 90, 0.2, 1.0, r)
    a = mirror_price("down-and-out call", S0, 100, 90, 0.2, 1.0, r, weighted=False)
    b = mirror_price("down-and-out call", S0, 100, 90, 0.2, 1.0, r)
    print(f"  {r:>6.2f} {2*r/0.04-1:>9.2f} {cf:>12.6f} {a:>11.6f} {a-cf:>10.1e} {b:>13.6f} {b-cf:>10.1e}")
print()
print("  every type, market 1 (r=0.05, sigma=0.2), largest plain-mirror error:")
worst_a, at_half = 0.0, 0.0
for k in TYPES:
    H, Ks = BARRIERS[parse(k)[0]]
    ea = max(abs(mirror_price(k, S0, K, H, 0.2, 1.0, 0.05, weighted=False)
                 - barrier_closed_form(k, S0, K, H, 0.2, 1.0, 0.05)) for K in Ks)
    eh = max(abs(mirror_price(k, S0, K, H, 0.2, 1.0, 0.02, weighted=False)
                 - barrier_closed_form(k, S0, K, H, 0.2, 1.0, 0.02)) for K in Ks)
    worst_a, at_half = max(worst_a, ea), max(at_half, eh)
    print(f"    {k:<20} r=0.05: {ea:.2e}    r=0.02 (= sigma^2/2): {eh:.1e}")
print(f"  {check(worst_a > 1e-2 and at_half < 1e-6, 'plain mirror')}: fails by up to "
      f"${worst_a:.2f} when r != sigma^2/2, exact (< 1e-6) when r = sigma^2/2")

print()
print("=" * 72)
print("4) Step by step (c): blur dt, zero past the barrier, repeat")
print("=" * 72)
s, T, r = 0.2, 1.0, 0.05
print(f"  sigma={s}, T={T}, r={r}; monitored at T/N, 2T/N, ..., T; error vs the continuous closed form")
print(f"  {'type':<20} {'K':>5} {'closed':>9} | {'no shift: N=25':>14} {'100':>9} {'400':>9} | "
      f"{'BGK shift: N=25':>15} {'100':>9} {'400':>9}")
t0 = time.perf_counter()
table = {}
for ud, (H, Ks) in BARRIERS.items():
    for cp in ["call", "put"]:
        kinds = [f"{ud}-and-out {cp}", f"{ud}-and-in {cp}"]
        for K in Ks:
            for N in [25, 100, 400]:
                for sh in [False, True]:
                    p = stepping_prices(kinds, S0, K, H, s, T, r, N, shift=sh)
                    for kd in kinds:
                        table[(kd, K, N, sh)] = p[kd]
for k in TYPES:
    H, Ks = BARRIERS[parse(k)[0]]
    for K in Ks:
        cf = barrier_closed_form(k, S0, K, H, s, T, r)
        e = [table[(k, K, N, sh)] - cf for sh in [False, True] for N in [25, 100, 400]]
        print(f"  {k:<20} {K:>5.0f} {cf:>9.5f} | {e[0]:>14.2e} {e[1]:>9.2e} {e[2]:>9.2e} | "
              f"{e[3]:>15.2e} {e[4]:>9.2e} {e[5]:>9.2e}")
print(f"  ({time.perf_counter() - t0:.1f}s)")
ko_over = all(table[(k, K, N, False)] >= barrier_closed_form(k, S0, K, H, s, T, r) - 1e-9
              for k in TYPES if "out" in k
              for H, Ks in [BARRIERS[parse(k)[0]]] for K in Ks for N in [25, 100, 400])
print(f"  {check(ko_over, 'stepping overprices knock-outs')}: without the shift every knock-out is "
      f"overpriced (a path can cross and come back between monitoring dates)")
shift_better = all(abs(table[(k, K, 400, True)] - barrier_closed_form(k, S0, K, H, s, T, r))
                   <= abs(table[(k, K, 400, False)] - barrier_closed_form(k, S0, K, H, s, T, r)) + 1e-12
                   for k in TYPES for H, Ks in [BARRIERS[parse(k)[0]]] for K in Ks)
print(f"  {check(shift_better, 'shift helps')}: with the shift the N=400 error is no larger for every option")

print()
print("=" * 72)
print("5) Independent check: closed forms vs QuantLib AnalyticBarrierEngine")
print("=" * 72)
try:
    import QuantLib as ql
except ImportError:
    print("  QuantLib not installed (pip install QuantLib)")
    failures.append("QuantLib missing")
else:
    rng = np.random.default_rng(20261020)
    today = ql.Date(20, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    qtype = {("down", "in"): ql.Barrier.DownIn, ("down", "out"): ql.Barrier.DownOut,
             ("up", "in"): ql.Barrier.UpIn, ("up", "out"): ql.Barrier.UpOut}
    n_cases, worst, worst_rel, worst_case = 400, 0.0, 0.0, None
    per_type = {k: 0 for k in TYPES}
    for i in range(n_cases):
        kind = TYPES[i % 8]
        ud, io, cp = parse(kind)
        S = float(rng.uniform(60, 140))
        H = float(S * (rng.uniform(0.70, 0.99) if ud == "down" else rng.uniform(1.01, 1.40)))
        K = float(S * rng.uniform(0.6, 1.5))
        sigma = float(rng.uniform(0.05, 0.6))
        r = float(rng.uniform(-0.01, 0.12))
        q = float(rng.uniform(0.0, 0.06))
        days = int(rng.integers(7, 3 * 365))
        T = days / 365.0
        spot = ql.QuoteHandle(ql.SimpleQuote(S))
        rts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
        qts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
        vts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc))
        process = ql.BlackScholesMertonProcess(spot, qts, rts, vts)
        opt = ql.BarrierOption(qtype[(ud, io)], H, 0.0,
                               ql.PlainVanillaPayoff(ql.Option.Call if cp == "call" else ql.Option.Put, K),
                               ql.EuropeanExercise(today + days))
        opt.setPricingEngine(ql.AnalyticBarrierEngine(process))
        ref = opt.NPV()
        ours = barrier_closed_form(kind, S, K, H, sigma, T, r, q)
        d = abs(ours - ref)
        per_type[kind] += 1
        if d > worst:
            worst, worst_case = d, (kind, round(S, 2), round(K, 2), round(H, 2), round(sigma, 3),
                                    round(T, 3), round(r, 4), round(q, 4), ref)
        worst_rel = max(worst_rel, d / max(abs(ref), 1e-8) if abs(ref) > 1e-6 else 0.0)
    print(f"  QuantLib {ql.__version__}; {n_cases} random cases ({n_cases // 8} per type), "
          f"S in [60,140], K/S in [0.6,1.5], H/S in [0.7,0.99] or [1.01,1.4],")
    print(f"  sigma in [5%,60%], r in [-1%,12%], dividend yield q in [0,6%], T from 1 week to 3 years")
    print(f"  max |ours - QuantLib| = {worst:.1e}   max relative (prices > 1e-6) = {worst_rel:.1e}"
          f"   {check(worst < 1e-10, 'QuantLib')}")
    print(f"  worst case: {worst_case}")

print()
if failures:
    print("FAILED: " + ", ".join(failures))
    sys.exit(1)
print("all barrier checks passed")
