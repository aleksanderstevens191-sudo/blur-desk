"""Write src/kernel_fixtures.json: fat-tailed kernel prices from the Python reference
(kernels.py, smile.py) that the JS port (src/kernels.js) and the WebGPU blur
(/check.html) must reproduce.

Each case carries its grid exactly (y_j = j dx, j = -J..J) so the browser builds
the kernel on the same nodes instead of re-deriving the width. Per case:
  strikes  one option per strike, priced at S0 = 100 by blur_price
  smile    out-of-the-money options at strikes F e^{o dx} for integer node offsets o
           (F the forward), with their implied vols: the demo's smile panel reads
           all of these off ONE blurred payoff row (a strike-F call, a strike-F put)
  checks   kernel weights at a few nodes, sum w, sum w e^y
  ref      the independent price (Black-Scholes, Merton's series, the VG mixture)

python reference/make_kernel_fixtures.py
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from scipy.special import gammaln, gammainc, gammaincc, gammainccinv
from kernels import grid_for, kernel, blur_price, bs, merton_series, vg_mixture, mean_sd
from smile import implied_vol

S0 = 100.0
spy = json.loads((HERE.parent / "src" / "spy_fit.json").read_text())
fit = {k: spy["merton"][k] for k in ("sigma", "lam", "m", "delta")}

CASES = [
    ("Gaussian, the demo's default", "gaussian", dict(T=1.0, r=0.05, sigma=0.30)),
    ("Merton, the demo's default jumps", "merton", dict(T=1.0, r=0.05, sigma=0.30, lam=1.0, m=-0.10, delta=0.10)),
    *[(f"Merton, SPY fit, {s['expiry']}", "merton", dict(T=s["T"], r=s["r"], **fit), s) for s in spy["slices"]],
    ("Merton, crash-heavy: 2 jumps/yr of -20%, T = 2", "merton", dict(T=2.0, r=0.05, sigma=0.20, lam=2.0, m=-0.20, delta=0.20)),
    ("Variance gamma, theta -0.2, T = 0.5", "vg", dict(T=0.5, r=0.05, sigma=0.18, theta=-0.2, nu=0.2)),
    ("Variance gamma, 4 weeks, T/nu = 0.25", "vg", dict(T=0.075, r=0.05, sigma=0.20, theta=-0.10, nu=0.3)),
]


def reference(model, K, cp, p):
    T, r, sigma = p["T"], p["r"], p["sigma"]
    if model == "gaussian":
        return bs(S0, K, sigma, T, r, cp)
    if model == "merton":
        return merton_series(S0, K, T, r, sigma, p["lam"], p["m"], p["delta"], cp=cp)
    return vg_mixture(S0, K, T, r, sigma, p["theta"], p["nu"], cp=cp)


cases = []
for name, model, p, *market in CASES:
    T, r, sigma = p["T"], p["r"], p["sigma"]
    extra = {k: v for k, v in p.items() if k not in ("T", "r", "sigma")}
    y = grid_for(model, T, r, sigma, **extra)
    J = (len(y) - 1) // 2
    dx = float(y[J + 1])  # y_j = j dx
    w = kernel(model, y, T, r, sigma, **extra)
    price = lambda K, cp: float(blur_price(model, S0, K, T, r, sigma, cp, y=y, w=w, **extra))

    if market:  # the SPY quotes, in units where the spot (net of dividends) is 100
        s = market[0]
        strikes = [(K * S0 / s["S"], cp) for K, cp in zip(s["K"], s["cp"])]
    else:
        strikes = [(K, cp) for K in (70.0, 85.0, 95.0, 100.0, 105.0, 115.0, 140.0) for cp in ("call", "put")]
    strike_rows = [{"K": K, "cp": cp, "py": price(K, cp), "ref": float(reference(model, K, cp, p))}
                   for K, cp in strikes]

    # smile: offsets o, strikes F e^{o dx}, out to 2.5 sd of the kernel each side
    F = S0 * np.exp(r * T)
    sd = mean_sd(model, T, r, sigma, **extra)[1]
    omax = int(np.ceil(2.5 * sd / dx))
    offsets = sorted({int(round(v)) for v in np.linspace(-omax, omax, 21)})
    smile = []
    for o in offsets:
        K = float(F * np.exp(o * dx))
        cp = "call" if o >= 0 else "put"
        v = price(K, cp)
        smile.append({"o": o, "K": K, "cp": cp, "py": v, "iv": float(implied_vol(v, S0, K, T, r, cp))})

    idx = sorted({0, J // 4, J // 2, J - 37, J, J + 11, J + J // 3, 2 * J - 5})
    cases.append({
        "name": name, "model": model, "p": p, "J": J, "dx": dx, "S0": S0, "F": float(F), "omax": omax,
        "checks": {"idx": idx, "w": [float(w[i]) for i in idx], "sum": float(w.sum()), "fwd": float(w @ np.exp(y)),
                   "wmax": float(w.max())},
        "strikes": strike_rows, "smile": smile,
    })
    print(f"{name}: {len(y)} nodes, dx {dx:.3g}, {len(strike_rows)} strikes, {len(smile)} smile points")

special = {
    "gammaln": [[x, float(gammaln(x))] for x in (0.04, 0.3, 0.5, 1.5, 2.5, 7.3, 33.0, 150.0)],
    "gammainc": [[a, x, float(gammainc(a, x)), float(gammaincc(a, x))]
                 for a, x in ((0.04, 1e-9), (0.3, 0.01), (0.5, 2.0), (5.9, 3.0), (5.9, 30.0), (33.0, 20.0), (0.25, 40.0))],
    "gammainccinv": [[a, q, float(gammainccinv(a, q))] for a, q in ((0.04, 1e-18), (0.25, 1e-18), (5.9, 1e-18), (33.0, 1e-20))],
}
out = HERE.parent / "src" / "kernel_fixtures.json"
out.write_text(json.dumps({"cases": cases, "special": special}, indent=1))
print(f"wrote {out}")
