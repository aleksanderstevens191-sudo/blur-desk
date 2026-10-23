"""Stage 1 checks for the fat-tailed kernels. Exits non-zero if a check fails."""
import sys
import time
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kernels import (
    MODELS, grid_for, kernel, blur_price, density, drift, bs, merton_series,
    vg_mixture, carr_madan, kink_correction,
)

failures = []
N_CASES = 80
rng = np.random.default_rng(20261022)


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


def header(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


def random_case(model):
    S0 = float(rng.uniform(60, 140))
    c = dict(S0=S0, K=float(S0 * rng.uniform(0.7, 1.4)), days=int(rng.integers(7, 3 * 365)),
             r=float(rng.uniform(-0.01, 0.10)), sigma=float(rng.uniform(0.08, 0.5)), p={})
    if model == "merton":
        c["p"] = dict(lam=float(rng.uniform(0.05, 3.0)), m=float(rng.uniform(-0.30, 0.15)),
                      delta=float(rng.uniform(0.02, 0.40)))
    if model == "vg":
        c["sigma"] = float(rng.uniform(0.08, 0.4))
        c["p"] = dict(theta=float(rng.uniform(-0.40, 0.10)), nu=float(rng.uniform(0.05, 0.5)))
    c["T"] = c["days"] / 365.0
    return c


RANGES = {
    "gaussian": "sigma in [8%,50%]",
    "merton": "sigma in [8%,50%], jumps/yr lam in [0.05,3], mean jump m in [-30%,15%], "
              "jump vol delta in [2%,40%]",
    "vg": "sigma in [8%,40%], theta in [-0.40,0.10], nu in [0.05,0.5] (T/nu down to 0.04)",
}

t_start = time.time()
cases = {}
for model in MODELS:
    cases[model] = []
    for _ in range(N_CASES):
        c = random_case(model)
        c["y"] = grid_for(model, c["T"], c["r"], c["sigma"], **c["p"])
        c["w"] = kernel(model, c["y"], c["T"], c["r"], c["sigma"], **c["p"])
        for cp in ("call", "put"):
            c[cp] = blur_price(model, c["S0"], c["K"], c["T"], c["r"], c["sigma"], cp,
                               y=c["y"], w=c["w"], **c["p"])
        cases[model].append(c)

print(f"{N_CASES} random cases per model. All: S0 in [60,140], K/S0 in [0.7,1.4], "
      f"r in [-1%,10%], T from 1 week to 3 years (whole days / 365).")
for model in MODELS:
    sizes = [len(c["y"]) for c in cases[model]]
    print(f"  {model:9s} {RANGES[model]}")
    print(f"  {'':9s} kernel length {min(sizes)} to {max(sizes)} nodes")
print(f"  kernels and blurs built in {time.time() - t_start:.1f} s")
print()

# ----------------------------------------------------------------------------
header("1) Every kernel is a risk-neutral density")
print("  On the grid: sum_j w_j = 1 and sum_j w_j e^{y_j} = e^{rT} (the forward / S0)")
print("  model        max |sum w - 1|   max |sum w e^y - e^rT| / e^rT   check < 1e-10")
for model in MODELS:
    m0 = max(abs(c["w"].sum() - 1) for c in cases[model])
    m1 = max(abs(c["w"] @ np.exp(c["y"]) / np.exp(c["r"] * c["T"]) - 1) for c in cases[model])
    print(f"  {model:9s}    {m0:12.1e}      {m1:12.1e}                    "
          f"{check(m0 < 1e-10 and m1 < 1e-10, f'{model} kernel mass/forward')}")
print()
print("  Off the grid: the closed-form densities integrated with scipy quad")
print("  model      T      int p - 1     int p e^y / e^rT - 1   check < 1e-10")
for model in MODELS:
    for c in cases[model][:3]:
        T, r = c["T"], c["r"]
        mu = drift(model, T, r, c["sigma"], **c["p"])
        f0 = lambda y: density(model, y, T, r, c["sigma"], **c["p"])
        f1 = lambda y: f0(y) * np.exp(y)
        lo, hi = c["y"][0], c["y"][-1]
        i0 = sum(quad(f0, a, b, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
                 for a, b in [(lo, mu), (mu, hi)])
        i1 = sum(quad(f1, a, b, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
                 for a, b in [(lo, mu), (mu, hi)])
        e0, e1 = i0 - 1, i1 / np.exp(r * T) - 1
        print(f"  {model:9s} {T:5.3f}  {e0:12.1e}   {e1:12.1e}             "
              f"{check(abs(e0) < 1e-10 and abs(e1) < 1e-10, f'{model} density integral')}")
print()

# ----------------------------------------------------------------------------
header("2) Put-call parity: C - P = S0 - K e^{-rT}")
for model in MODELS:
    gap = max(abs(c["call"] - c["put"] - (c["S0"] - c["K"] * np.exp(-c["r"] * c["T"])))
              for c in cases[model])
    print(f"  {model:9s} max gap over {N_CASES} cases = {gap:.1e}   "
          f"{check(gap < 1e-10, f'{model} parity')}")
print()

# ----------------------------------------------------------------------------
header("3) Gaussian kernel vs the Black-Scholes formula")
err = max(abs(c[cp] - bs(c["S0"], c["K"], c["sigma"], c["T"], c["r"], cp))
          for c in cases["gaussian"] for cp in ("call", "put"))
print(f"  max |blur - BS| over {N_CASES} cases x call/put = {err:.1e}   {check(err < 1e-8, 'gaussian vs BS')}")
print()

# ----------------------------------------------------------------------------
header("4) Merton blur vs Merton's 1976 series")
print("  series: sum over n jumps of Poisson(lam (1+kappa) T) weights x Black-Scholes")
print("          with sigma_n^2 = sigma^2 + n delta^2/T, r_n = r - lam kappa + n ln(1+kappa)/T")
worst, wc = 0.0, None
for c in cases["merton"]:
    for cp in ("call", "put"):
        ref = merton_series(c["S0"], c["K"], c["T"], c["r"], c["sigma"], cp=cp, **c["p"])
        d = abs(c[cp] - ref)
        if d > worst:
            worst, wc = d, (cp, round(c["K"] / c["S0"], 3), c["days"], round(c["sigma"], 3),
                            {k: round(v, 3) for k, v in c["p"].items()}, round(float(ref), 6))
print(f"  max |blur - series| over {N_CASES} cases x call/put = {worst:.1e}   "
      f"{check(worst < 1e-6, 'merton vs series')}")
print(f"  worst: {wc}")
print()

# ----------------------------------------------------------------------------
header("5) Merton blur vs QuantLib")
import QuantLib as ql

today = ql.Date(15, 1, 2026)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()


def ql_option(cp, K, days):
    return ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if cp == "call" else ql.Option.Put, K),
                            ql.EuropeanExercise(today + days))


def flat(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))


print(f"  QuantLib {ql.__version__}. Its Merton76Process engine (JumpDiffusionEngine) is not in")
print("  the Python bindings, so Merton is priced with QuantLib's Bates engine (Heston + Merton")
print("  jumps, semi-analytic Fourier integral) with the variance pinned: v0 = theta = sigma^2,")
print("  vol of vol 1e-5. QuantLib's jump convention ln(1+J) ~ N(nu, delta^2) is Merton's.")
worst, wc = 0.0, None
for c in cases["merton"]:
    p = c["p"]
    proc = ql.BatesProcess(flat(c["r"]), flat(0.0), ql.QuoteHandle(ql.SimpleQuote(c["S0"])),
                           c["sigma"]**2, 1.0, c["sigma"]**2, 1e-5, 0.0, p["lam"], p["m"], p["delta"])
    eng = ql.BatesEngine(ql.BatesModel(proc))
    for cp in ("call", "put"):
        opt = ql_option(cp, c["K"], c["days"])
        opt.setPricingEngine(eng)
        ref = opt.NPV()
        d = abs(c[cp] - ref)
        if d > worst:
            worst, wc = d, (cp, round(c["K"] / c["S0"], 3), c["days"], round(float(ref), 6))
print(f"  max |blur - QuantLib| over {N_CASES} cases x call/put = {worst:.1e}   "
      f"{check(worst < 1e-7, 'merton vs QuantLib')}")
print(f"  worst: {wc}")
print()

# ----------------------------------------------------------------------------
header("6) Variance gamma blur vs QuantLib")
print("  QuantLib VarianceGammaEngine (Madan-Carr-Chang closed form, integrated numerically).")
print("  QuantLib's engine is itself unreliable when T/nu is small (the density is singular at")
print("  its centre): its own call and put break put-call parity there. A QuantLib price only")
print("  counts as a reference when QuantLib's call and put satisfy parity to 1e-8.")
worst, wc, ql_bad = 0.0, None, []
for c in cases["vg"]:
    p = c["p"]
    proc = ql.VarianceGammaProcess(ql.QuoteHandle(ql.SimpleQuote(c["S0"])), flat(0.0), flat(c["r"]),
                                   c["sigma"], p["nu"], p["theta"])
    eng = ql.VarianceGammaEngine(proc)
    ref = {}
    for cp in ("call", "put"):
        opt = ql_option(cp, c["K"], c["days"])
        opt.setPricingEngine(eng)
        ref[cp] = opt.NPV()
    gap = ref["call"] - ref["put"] - (c["S0"] - c["K"] * np.exp(-c["r"] * c["T"]))
    tn = round(c["T"] / p["nu"], 3)
    if abs(gap) > 1e-8:
        leg = max(("call", "put"), key=lambda cp: abs(c[cp] - ref[cp]))
        ql_bad.append((c["days"], tn, float(gap), leg, float(c[leg] - ref[leg])))
        continue
    for cp in ("call", "put"):
        d = abs(c[cp] - ref[cp])
        if d > worst:
            worst, wc = d, (cp, round(c["K"] / c["S0"], 3), c["days"], tn, round(float(ref[cp]), 6))
n_ok = N_CASES - len(ql_bad)
print(f"  {n_ok} of {N_CASES} cases where QuantLib holds parity (calls and puts):")
print(f"    max |blur - QuantLib| = {worst:.1e}   {check(worst < 1e-6 and n_ok >= 50, 'vg vs QuantLib (and at least 50 usable cases)')}")
print(f"    worst (cp, K/S0, days, T/nu, QuantLib price): {wc}")
big = [b for b in ql_bad if abs(b[2]) > 1e-6]
print(f"  {len(ql_bad)} cases where QuantLib breaks parity by more than 1e-8; the {len(big)} by more than 1e-6")
print(f"  have T/nu from {min(b[1] for b in big):.2f} to {max(b[1] for b in big):.2f}. The largest:")
print("    days   T/nu   QuantLib C-P gap   leg off   blur - QuantLib on that leg")
for b in sorted(ql_bad, key=lambda b: -abs(b[2]))[:6]:
    print(f"    {b[0]:4d}  {b[1]:5.3f}   {b[2]:14.1e}     {b[3]:5s}    {b[4]:12.1e}")
print("  On every one of these the blur still holds parity (section 2) and matches the")
print("  separate adaptive integration in section 7, so the gap is QuantLib's.")
print()

# ----------------------------------------------------------------------------
header("7) Variance gamma blur vs the gamma-mixture formula (adaptive quad, no grid)")
worst, wc = 0.0, None
for c in cases["vg"]:
    for cp in ("call", "put"):
        ref = vg_mixture(c["S0"], c["K"], c["T"], c["r"], c["sigma"], cp=cp, **c["p"])
        d = abs(c[cp] - ref)
        if d > worst:
            worst, wc = d, (cp, round(c["K"] / c["S0"], 3), c["days"], round(c["T"] / c["p"]["nu"], 3), round(float(ref), 6))
print(f"  max |blur - mixture| over {N_CASES} cases x call/put = {worst:.1e}   "
      f"{check(worst < 1e-6, 'vg vs mixture')}")
print(f"  worst (cp, K/S0, days, T/nu, price): {wc}")
print()

# ----------------------------------------------------------------------------
header("8) Every model vs a Carr-Madan FFT pricer (characteristic function only)")
print("  2^18 points, frequency step 0.02, damping alpha = +1.5 (calls), -1.5 (puts),")
print("  log-strike grid centred on ln K. The FFT needs the characteristic function to")
print("  decay; for variance gamma it decays only like |u|^(-2T/nu), so short-dated VG")
print("  is reported separately and not checked at 1e-6.")
print("  model      maturities      cases   max |blur - FFT|   check < 1e-6")
for model in MODELS:
    groups = {"T/nu >= 1" if model == "vg" else "all": [], "T/nu < 1": []}
    for c in cases[model]:
        short = model == "vg" and c["T"] / c["p"]["nu"] < 1
        for cp in ("call", "put"):
            ref = carr_madan(model, c["S0"], c["K"], c["T"], c["r"], c["sigma"], cp,
                             N=2**18, eta=0.02, **c["p"])
            groups["T/nu < 1" if short else list(groups)[0]].append(abs(c[cp] - ref))
    for name, errs in groups.items():
        if not errs:
            continue
        e = max(errs)
        tag = check(e < 1e-6, f"{model} vs Carr-Madan") if name != "T/nu < 1" else "(reported)"
        print(f"  {model:9s}  {name:12s}  {len(errs) // 2:5d}   {e:12.1e}      {tag}")
print()

print(f"total time {time.time() - t_start:.0f} s")
if failures:
    print("FAILED: " + ", ".join(failures))
    sys.exit(1)
print("ALL PASS")
