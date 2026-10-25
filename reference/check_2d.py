"""Stage 1 checks for two-stock options as one 2D blur. Exits non-zero if a check fails.

  0) the bivariate normal CDF (Owen's T) against direct integration
  1) the kernel: mass, both forwards, covariance, and the blur ellipse
  2) exchange option vs Margrabe, 3) best-of call vs Stulz          (1e-6)
  4) rho = 0: product payoff vs the product of two 1D blur prices    (1e-10)
  5) QuantLib's Margrabe and Stulz engines on random cases            (1e-6)
  6) grid convergence on a hard case
"""
import sys
import time
import warnings
import numpy as np
import QuantLib as ql
from scipy.integrate import IntegrationWarning, quad
from scipy.special import ndtr
from scipy.stats import norm

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from blur2d import (
    bvn_cdf, margrabe, stulz_call_max, stulz_call_min, bs, drifts, ellipse,
    grid_steps, kernel_2d, exchange_payoff, best_of_call_payoff,
    worst_of_call_payoff, product_payoff, vanilla_1d, price_2d, price_1d,
)

warnings.filterwarnings("ignore", category=IntegrationWarning)  # quad at 1e-15 abs
failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


def head(t):
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


t_start = time.time()
print("Two-stock options by one 2D blur in log-price (float64). Grid step: c = 0.03 of")
print("each stock's conditional sd s_i sqrt(T (1 - rho^2)); kernel window +-8 sd.")

# ---------------------------------------------------------------------------
head("0) Bivariate normal CDF (Owen's T) vs direct integration")
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(200):
    h, k = rng.uniform(-4, 4, 2)
    rho = rng.uniform(-0.99, 0.99)
    s = np.sqrt(1 - rho * rho)
    ref = quad(lambda x: norm.pdf(x) * ndtr((k - rho * x) / s), -40, h,
               epsabs=1e-15, epsrel=1e-14, limit=200)[0]
    worst = max(worst, abs(bvn_cdf(h, k, rho) - ref))
print(f"  200 random (h, k, rho), rho in [-0.99, 0.99]: max |Owen - quad| = {worst:.1e}"
      f"   < 1e-13 {check(worst < 1e-13, 'bvn cdf')}")

# ---------------------------------------------------------------------------
head("1) The kernel: mass, forwards, covariance, ellipse")
print(f"  {'s1':>5} {'s2':>5} {'rho':>5} {'T':>4} {'r':>5} | {'sum-1':>8} {'fwd1 err':>9} {'fwd2 err':>9}"
      f" {'cov err':>8} | {'major':>6} {'minor':>6} {'angle':>6}")
worst = 0.0
for s1, s2, rho, T, r, q1, q2 in [
    (0.30, 0.20, -0.9, 1.0, 0.05, 0.00, 0.00), (0.30, 0.20, 0.0, 1.0, 0.05, 0.00, 0.00),
    (0.30, 0.20, 0.5, 1.0, 0.05, 0.02, 0.01), (0.30, 0.20, 0.9, 1.0, 0.05, 0.00, 0.00),
    (0.25, 0.25, 0.9, 0.5, 0.03, 0.00, 0.00), (0.50, 0.15, -0.5, 2.0, 0.08, 0.03, 0.00),
]:
    dx, dy = grid_steps(s1, s2, rho, T)
    xo, yo, W = kernel_2d(dx, dy, s1, s2, rho, T, r, q1, q2)
    X, Y = xo[:, None], yo[None, :]
    f1 = np.sum(W * np.exp(X)) / np.exp((r - q1) * T) - 1
    f2 = np.sum(W * np.exp(Y)) / np.exp((r - q2) * T) - 1
    m1, m2 = drifts(T, r, s1, s2, q1, q2)
    cov = np.sum(W * (X - m1) * (Y - m2))
    ce = abs(cov - rho * s1 * s2 * T)
    a, b, ang = ellipse(s1, s2, rho, T)
    worst = max(worst, abs(W.sum() - 1), abs(f1), abs(f2), ce)
    print(f"  {s1:>5} {s2:>5} {rho:>5} {T:>4} {r:>5} | {W.sum() - 1:>8.1e} {f1:>9.1e} {f2:>9.1e}"
          f" {ce:>8.1e} | {a:>6.3f} {b:>6.3f} {np.degrees(ang):>5.1f}d")
print(f"  ellipse = 1-sd level set of the kernel in (ln S1, ln S2); angle of the major axis")
print(f"  worst of the four error columns: {worst:.1e}   < 1e-12 {check(worst < 1e-12, 'kernel moments')}")

# ---------------------------------------------------------------------------
MARKETS = [  # S1, S2, s1, s2, T, r, q1, q2
    (100.0, 100.0, 0.20, 0.20, 1.00, 0.05, 0.00, 0.00),
    (100.0, 90.0, 0.30, 0.20, 0.50, 0.03, 0.02, 0.00),
    (80.0, 110.0, 0.45, 0.25, 2.00, 0.08, 0.00, 0.03),
    (120.0, 100.0, 0.15, 0.35, 0.25, 0.01, 0.01, 0.01),
]
RHOS = [-0.9, -0.5, 0.0, 0.5, 0.9, 0.95]

head("2) Exchange option max(S1 - S2, 0) vs Margrabe (1978)")
print(f"  {'S1':>5} {'S2':>5} {'s1':>5} {'s2':>5} {'T':>4} " + " ".join(f"{'rho=' + str(r):>9}" for r in RHOS))
worst = 0.0
t0 = time.time()
n = 0
for S1, S2, s1, s2, T, r, q1, q2 in MARKETS:
    errs = []
    for rho in RHOS:
        v = price_2d(exchange_payoff(), S1, S2, s1, s2, rho, T, r, q1, q2)
        errs.append(abs(v - margrabe(S1, S2, s1, s2, rho, T, q1, q2)))
        n += 1
    worst = max(worst, max(errs))
    print(f"  {S1:>5.0f} {S2:>5.0f} {s1:>5} {s2:>5} {T:>4} " + " ".join(f"{e:>9.1e}" for e in errs))
print(f"  |blur - Margrabe|, {n} cases (q1, q2 as in the markets list); "
      f"{(time.time() - t0) / n * 1e3:.0f} ms per price")
print(f"  max error {worst:.1e}   < 1e-6 {check(worst < 1e-6, 'exchange vs Margrabe')}")

head("3) Call on the better of two stocks max(max(S1, S2) - K, 0) vs Stulz (1982)")
print(f"  {'S1':>5} {'S2':>5} {'K':>5} {'T':>4} " + " ".join(f"{'rho=' + str(r):>9}" for r in RHOS))
worst = 0.0
n = 0
for S1, S2, s1, s2, T, r, q1, q2 in MARKETS:
    for K in (0.9 * min(S1, S2), max(S1, S2), 1.15 * max(S1, S2)):
        errs = []
        for rho in RHOS:
            v = price_2d(best_of_call_payoff(K), S1, S2, s1, s2, rho, T, r, q1, q2)
            errs.append(abs(v - stulz_call_max(S1, S2, K, s1, s2, rho, T, r, q1, q2)))
            n += 1
        worst = max(worst, max(errs))
        print(f"  {S1:>5.0f} {S2:>5.0f} {K:>5.1f} {T:>4} " + " ".join(f"{e:>9.1e}" for e in errs))
print(f"  |blur - Stulz|, {n} cases")
print(f"  max error {worst:.1e}   < 1e-6 {check(worst < 1e-6, 'best-of vs Stulz')}")

# ---------------------------------------------------------------------------
head("4) rho = 0: a product payoff f1(S1) f2(S2) vs the product of two 1D prices")
print("  Independent stocks: E[f1 f2] = E[f1] E[f2], so V_2d = e^{rT} V1 V2, with V1, V2")
print("  the 1D blur prices on the same grid steps (and, for scale, the Black-Scholes ones).")
print(f"  {'payoff':<16} {'market':>6} {'V_2d':>12} {'|V_2d - e^rT V1 V2|':>20} {'vs BS product':>14}")
worst = 0.0
for mi, (S1, S2, s1, s2, T, r, q1, q2) in enumerate(MARKETS):
    for cp1, cp2 in [("call", "call"), ("call", "put"), ("put", "call"), ("put", "put")]:
        K1, K2 = 1.03 * S1, 0.97 * S2
        dx, dy = grid_steps(s1, s2, 0.0, T)
        v2 = price_2d(product_payoff(K1, cp1, K2, cp2), S1, S2, s1, s2, 0.0, T, r, q1, q2)
        f1, k1 = vanilla_1d(K1, cp1)
        f2, k2 = vanilla_1d(K2, cp2)
        e1 = price_1d(f1, k1, S1, s1, T, r, q1, d=dx)
        e2 = price_1d(f2, k2, S2, s2, T, r, q2, d=dy)
        prod = np.exp(-r * T) * e1 * e2  # = e^{rT} V1 V2 with V_i = e^{-rT} e_i
        bsp = np.exp(r * T) * bs(S1, K1, s1, T, r, q1, cp1) * bs(S2, K2, s2, T, r, q2, cp2)
        e = abs(v2 - prod)
        worst = max(worst, e)
        print(f"  {cp1 + ' x ' + cp2:<16} {mi + 1:>6} {v2:>12.6f} {e:>20.1e} {abs(v2 - bsp):>14.1e}")
print(f"  max |V_2d - e^rT V1 V2| = {worst:.1e}   < 1e-10 {check(worst < 1e-10, 'product = product of 1D')}")

# ---------------------------------------------------------------------------
head("5) Independent check: QuantLib 1.44 Margrabe and Stulz engines, random cases")
today = ql.Date(23, 10, 2026)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()


def process(S, sigma, r, q):
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)))


N = 60
rng = np.random.default_rng(2026)
rows = {"exchange (Margrabe engine)": [], "best-of call (Stulz engine)": [], "worst-of call (Stulz engine)": []}
cf_vs_ql = 0.0
t0 = time.time()
for _ in range(N):
    S1, S2 = rng.uniform(50, 150, 2)
    s1, s2 = rng.uniform(0.10, 0.60, 2)
    rho = rng.uniform(-0.95, 0.95)
    days = int(rng.integers(30, 731))
    T = days / 365
    r = rng.uniform(0.0, 0.08)
    q1, q2 = rng.uniform(0.0, 0.04, 2)
    K = rng.uniform(60, 160)
    p1, p2 = process(S1, s1, r, q1), process(S2, s2, r, q2)
    ex = ql.EuropeanExercise(today + days)
    o = ql.MargrabeOption(1, 1, ex)
    o.setPricingEngine(ql.AnalyticEuropeanMargrabeEngine(p1, p2, rho))
    best = ql.BasketOption(ql.MaxBasketPayoff(ql.PlainVanillaPayoff(ql.Option.Call, K)), ex)
    best.setPricingEngine(ql.StulzEngine(p1, p2, rho))
    worst_ = ql.BasketOption(ql.MinBasketPayoff(ql.PlainVanillaPayoff(ql.Option.Call, K)), ex)
    worst_.setPricingEngine(ql.StulzEngine(p1, p2, rho))
    case = (S1, S2, s1, s2, rho, T, r, q1, q2, K)
    for name, pay, qlv, cf in [
        ("exchange (Margrabe engine)", exchange_payoff(), o.NPV(), margrabe(S1, S2, s1, s2, rho, T, q1, q2)),
        ("best-of call (Stulz engine)", best_of_call_payoff(K), best.NPV(),
         stulz_call_max(S1, S2, K, s1, s2, rho, T, r, q1, q2)),
        ("worst-of call (Stulz engine)", worst_of_call_payoff(K), worst_.NPV(),
         stulz_call_min(S1, S2, K, s1, s2, rho, T, r, q1, q2)),
    ]:
        v = price_2d(pay, S1, S2, s1, s2, rho, T, r, q1, q2)
        rows[name].append((abs(v - qlv), qlv, case))
        cf_vs_ql = max(cf_vs_ql, abs(cf - qlv))
print(f"  {N} random cases: S1, S2 in [50, 150], vols in [0.10, 0.60], rho in [-0.95, 0.95],")
print(f"  T in [30, 730] days, r in [0, 8%], dividend yields in [0, 4%], K in [60, 160]")
print(f"  {'contract':<30} {'max |blur - QL|':>16} {'median':>9} {'price range':>18}   < 1e-6")
for name, rs in rows.items():
    e = np.array([x[0] for x in rs])
    p = np.array([x[1] for x in rs])
    ok = e.max() < 1e-6
    print(f"  {name:<30} {e.max():>16.1e} {np.median(e):>9.1e} {p.min():>8.3f}..{p.max():<8.3f}  {check(ok, 'QL ' + name)}")
    i = int(np.argmax(e))
    c = rs[i][2]
    print(f"    worst: S1={c[0]:.1f} S2={c[1]:.1f} s1={c[2]:.3f} s2={c[3]:.3f} rho={c[4]:+.3f} "
          f"T={c[5]:.3f} r={c[6]:.4f} q={c[7]:.4f},{c[8]:.4f} K={c[9]:.1f}")
print(f"  our closed forms vs QuantLib, all {3 * N}: max {cf_vs_ql:.1e}   < 1e-12 "
      f"{check(cf_vs_ql < 1e-12, 'closed forms vs QL')}")
print(f"  ({time.time() - t0:.1f} s for {3 * N} blur prices)")

# ---------------------------------------------------------------------------
head("6) Grid convergence (hard case: rho = 0.95, best-of at the money)")
S1, S2, s1, s2, T, r, K = 100.0, 95.0, 0.30, 0.20, 1.0, 0.05, 100.0
rho = 0.95
print(f"  S1={S1} S2={S2} s1={s1} s2={s2} T={T} r={r} K={K}")
print(f"  {'c':>6} {'dx':>8} {'grid':>11} {'exchange err':>13} {'best-of err':>12}")
for c in (0.12, 0.06, 0.03, 0.015):
    dx, dy = grid_steps(s1, s2, rho, T, c)
    _, (_, _, xo, yo, _, _) = price_2d(exchange_payoff(), S1, S2, s1, s2, rho, T, r, c=c, surface=True)
    e1 = price_2d(exchange_payoff(), S1, S2, s1, s2, rho, T, r, c=c) - margrabe(S1, S2, s1, s2, rho, T)
    e2 = price_2d(best_of_call_payoff(K), S1, S2, s1, s2, rho, T, r, c=c) - stulz_call_max(S1, S2, K, s1, s2, rho, T, r)
    print(f"  {c:>6} {dx:>8.5f} {len(xo):>5}x{len(yo):<5} {e1:>13.1e} {e2:>12.1e}")
print("  error falls ~16x per halving of the step (fourth order): the kink corrections")
print("  remove the dx^2 and dx^3 terms; the Gaussian part is summed to machine precision.")

print()
print(f"total {time.time() - t_start:.0f} s")
if failures:
    print("FAILED:", ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASS")
