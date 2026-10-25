"""Stage 3 intuition checks for two-stock options. Exits non-zero if a check fails.

  1) The exchange option max(S1 - S2, 0) loses value as correlation rises: the
     blur, Margrabe and QuantLib on a sweep rho = -0.9 .. 0.9. Only the spread
     S1/S2 matters, and its vol sqrt(s1^2 + s2^2 - 2 rho s1 s2) falls with rho.
  2) Spread call max(S1 - S2 - K, 0) vs Kirk's approximation (1995), which
     prices it as an exchange option on S1 and (S2 + K), treating S2 + K as
     lognormal with vol s2 F2 / (F2 + K). Exact at K = 0, approximate
     elsewhere. Referee: the 2D blur, itself checked against an exact 1D
     integral (condition on stock 2, then stock 1 is a Black-Scholes call).
     Reports where Kirk's error passes 1 cent and 1% of the price.
Writes docs/exchange_vs_rho.png and docs/kirk_error.png.
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import QuantLib as ql
from scipy.integrate import quad
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from blur2d import bs, margrabe, price_2d, exchange_payoff, spread_payoff

DOCS = Path(__file__).parent.parent / "docs"
failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def kirk(S1, S2, K, s1, s2, rho, T, r, q1=0.0, q2=0.0):
    """Kirk (1995): spread call as an exchange of S1 for (S2 + K) with S2 + K lognormal."""
    F1, F2 = S1 * np.exp((r - q1) * T), S2 * np.exp((r - q2) * T)
    w = F2 / (F2 + K)
    s = np.sqrt(s1 * s1 - 2 * rho * s1 * s2 * w + (s2 * w) ** 2)
    d1 = (np.log(F1 / (F2 + K)) + 0.5 * s * s * T) / (s * np.sqrt(T))
    return np.exp(-r * T) * (F1 * ndtr(d1) - (F2 + K) * ndtr(d1 - s * np.sqrt(T)))


def spread_exact(S1, S2, K, s1, s2, rho, T, r, q1=0.0, q2=0.0):
    """Spread call by one exact 1D integral: given Z2 = z, S2_T is known and
    S1_T is lognormal with vol s1 sqrt(1 - rho^2), so the payoff's expectation
    is a Black-Scholes call with strike S2_T(z) + K."""
    sc = s1 * np.sqrt(1 - rho * rho)

    def f(z):
        S2T = S2 * np.exp((r - q2 - 0.5 * s2 * s2) * T + s2 * np.sqrt(T) * z)
        # S1 shifted so that its conditional forward is right
        S1z = S1 * np.exp(-0.5 * (s1 * rho) ** 2 * T + s1 * rho * np.sqrt(T) * z)
        Kz = S2T + K
        if Kz <= 0:  # always exercised
            v = S1z * np.exp(-q1 * T) - Kz * np.exp(-r * T)
        else:
            v = bs(S1z, Kz, sc, T, r, q1)
        return v * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)

    return quad(f, -12, 12, epsabs=1e-13, epsrel=1e-13, limit=400)[0]


today = ql.Date(23, 10, 2026)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()


def process(S, sigma, r, q):
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)))


def ql_margrabe(S1, S2, s1, s2, rho, days, r):
    o = ql.MargrabeOption(1, 1, ql.EuropeanExercise(today + days))
    o.setPricingEngine(ql.AnalyticEuropeanMargrabeEngine(process(S1, s1, r, 0), process(S2, s2, r, 0), rho))
    return o.NPV()


def ql_kirk(S1, S2, K, s1, s2, rho, days, r):
    o = ql.BasketOption(ql.SpreadBasketPayoff(ql.PlainVanillaPayoff(ql.Option.Call, K)), ql.EuropeanExercise(today + days))
    o.setPricingEngine(ql.KirkEngine(process(S1, s1, r, 0), process(S2, s2, r, 0), rho))
    return o.NPV()


t_start = time.time()

# ---------------------------------------------------------------------------
head("1) Exchange option max(S1 - S2, 0) as correlation rises")
S1, S2, s1, s2, T, r = 100.0, 100.0, 0.30, 0.25, 1.0, 0.05
days = 365
print(f"  S1 = S2 = {S1:.0f}, vols {s1:.0%} / {s2:.0%}, T = {T:.0f}y, r = {r:.0%}")
print(f"  {'rho':>5} {'spread vol':>10} {'blur':>11} {'Margrabe':>11} {'QuantLib':>11} {'|blur - M|':>10}  change")
rhos = np.round(np.arange(-0.9, 0.901, 0.1), 2)
blur_v, marg_v = [], []
worst = 0.0
for rho in rhos:
    v = price_2d(exchange_payoff(), S1, S2, s1, s2, rho, T, r)
    m = margrabe(S1, S2, s1, s2, rho, T)
    qv = ql_margrabe(S1, S2, s1, s2, rho, days, r)
    worst = max(worst, abs(v - m), abs(qv - m))
    ch = "" if not blur_v else f"{v - blur_v[-1]:+.4f}"
    blur_v.append(v)
    marg_v.append(m)
    print(f"  {rho:>5.1f} {np.sqrt(s1*s1 + s2*s2 - 2*rho*s1*s2):>10.4f} {v:>11.6f} {m:>11.6f} {qv:>11.6f} {abs(v - m):>10.1e}  {ch}")
dec = all(b < a for a, b in zip(blur_v, blur_v[1:]))
print(f"  the blur's value falls at every step of the sweep: {check(dec, 'exchange decreasing in rho')}")
print(f"  |blur - Margrabe| and |QuantLib - Margrabe| at most {worst:.1e}: {check(worst < 1e-6, 'exchange sweep vs Margrabe')}")
print(f"  value at rho = -0.9: {blur_v[0]:.4f};  at +0.9: {blur_v[-1]:.4f}  ({blur_v[0] / blur_v[-1]:.1f}x)")
print("  why: the option only cares about the ratio S1/S2. Correlated stocks move")
print("  together, so the ratio moves less: the blur ellipse tilts onto the diagonal")
print("  S1 = S2, the line the payoff's kink lies on, and less of it spills across.")

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
fine = np.linspace(-0.95, 0.95, 191)
ax.plot(fine, [margrabe(S1, S2, s1, s2, p, T) for p in fine], color="#888", lw=1, label="MARGRABE")
ax.plot(rhos, blur_v, "o", color="white", ms=4, label="2D BLUR")
ax.set_xlabel("CORRELATION", color="white", family="monospace")
ax.set_ylabel("EXCHANGE OPTION VALUE", color="white", family="monospace")
ax.set_title(f"S1 = S2 = 100, VOLS {s1:.0%}/{s2:.0%}, T = 1Y", color="white", family="monospace", fontsize=10)
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_color("#666")
ax.legend(facecolor="black", edgecolor="#666", labelcolor="white", prop={"family": "monospace"})
fig.tight_layout()
fig.savefig(DOCS / "exchange_vs_rho.png", facecolor="black")
plt.close(fig)

# ---------------------------------------------------------------------------
head("2) Spread call max(S1 - S2 - K, 0): the blur vs Kirk's approximation")
print("  referee check first: blur vs the exact 1D integral (condition on stock 2)")
ref_cases = [(K, rho) for K in (-20, 0, 10, 30) for rho in (-0.5, 0.5, 0.9)]
worst = 0.0
for K, rho in ref_cases:
    b = price_2d(spread_payoff(K), S1, S2, s1, s2, rho, T, r)
    e = spread_exact(S1, S2, K, s1, s2, rho, T, r)
    worst = max(worst, abs(b - e))
print(f"  {len(ref_cases)} cases, K in -20..30, rho in -0.5..0.9: max |blur - exact| = {worst:.1e}"
      f"   < 1e-6 {check(worst < 1e-6, 'spread blur vs exact')}")
w2 = 0.0
for K, rho in ref_cases:
    w2 = max(w2, abs(kirk(S1, S2, K, s1, s2, rho, T, r) - ql_kirk(S1, S2, K, s1, s2, rho, days, r)))
print(f"  our Kirk vs QuantLib's KirkEngine, same cases: max {w2:.1e}   < 1e-10 {check(w2 < 1e-10, 'Kirk vs QL')}")

print()
print(f"  Kirk minus blur, S1 = S2 = 100, vols {s1:.0%} / {s2:.0%}, T = 1y (dollars):")
Ks = [-30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50]
rho_grid = [-0.9, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95]
print("  " + f"{'K':>5} " + " ".join(f"{'rho=' + str(p):>9}" for p in rho_grid))
err = np.zeros((len(Ks), len(rho_grid)))
rel = np.zeros_like(err)
for i, K in enumerate(Ks):
    row = []
    for j, rho in enumerate(rho_grid):
        b = price_2d(spread_payoff(K), S1, S2, s1, s2, rho, T, r)
        k = kirk(S1, S2, K, s1, s2, rho, T, r)
        err[i, j] = k - b
        rel[i, j] = (k - b) / b
        row.append(f"{k - b:>+9.4f}")
    print(f"  {K:>5} " + " ".join(row))
print("  relative error (Kirk - blur) / blur:")
for i, K in enumerate(Ks):
    print(f"  {K:>5} " + " ".join(f"{rel[i, j]:>+8.2%} " for j in range(len(rho_grid))))
print(f"  K = 0 is exactly Margrabe: max |Kirk - blur| there {np.abs(err[Ks.index(0)]).max():.1e}"
      f"   {check(np.abs(err[Ks.index(0)]).max() < 1e-6, 'Kirk exact at K=0')}")

# Where it breaks down: over a finer grid, find the smallest |K| at which the
# error passes 1 cent and 1% of the price, for each correlation, and the vol mix.
print()
print("  where Kirk breaks down (first strike, moving away from K = 0, where |Kirk - blur|")
print("  passes 1 cent, and where it passes 1% of the price):")
Kf = np.arange(0, 61, 2.5)
for (a, b_) in [(0.30, 0.25), (0.25, 0.50), (0.50, 0.25)]:
    print(f"  vols {a:.0%} / {b_:.0%}:")
    for rho in (-0.5, 0.0, 0.5, 0.8, 0.9, 0.95):
        cent = pct = None
        for K in Kf[1:]:
            bb = price_2d(spread_payoff(K), S1, S2, a, b_, rho, T, r)
            d = kirk(S1, S2, K, a, b_, rho, T, r) - bb
            if cent is None and abs(d) > 0.01:
                cent = (K, d)
            if pct is None and abs(d) > 0.01 * bb:
                pct = (K, d / bb)
            if cent and pct:
                break
        fc = f"K = {cent[0]:>4.1f} ({cent[1]:+.3f})" if cent else "not by K = 60"
        fp = f"K = {pct[0]:>4.1f} ({pct[1]:+.1%})" if pct else "not by K = 60"
        print(f"    rho = {rho:>5.2f}:  1 cent at {fc:<22}  1% at {fp}")

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
for j, rho in enumerate(rho_grid):
    shade = 0.35 + 0.65 * j / (len(rho_grid) - 1)
    ax.plot(Ks, 100 * rel[:, j], "-o", ms=3, lw=1, color=(shade, shade, shade), label=f"RHO {rho:+.2f}")
ax.axhline(0, color="#444", lw=0.8)
ax.set_xlabel("STRIKE K  (S1 = S2 = 100)", color="white", family="monospace")
ax.set_ylabel("KIRK ERROR, % OF PRICE", color="white", family="monospace")
ax.set_title(f"SPREAD CALL, VOLS {s1:.0%}/{s2:.0%}, T = 1Y: KIRK VS 2D BLUR", color="white", family="monospace", fontsize=10)
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_color("#666")
ax.legend(facecolor="black", edgecolor="#666", labelcolor="white", prop={"family": "monospace", "size": 7}, ncol=2)
fig.tight_layout()
fig.savefig(DOCS / "kirk_error.png", facecolor="black")
plt.close(fig)

print()
print(f"wrote docs/exchange_vs_rho.png, docs/kirk_error.png; total {time.time() - t_start:.0f} s")
if failures:
    print("FAILED:", ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASS")
