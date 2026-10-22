"""Stage 2 calibration machinery, tested on a synthetic chain with known answers.

A Merton chain is generated (SPY-like spot, 7 expiries from 9 days to 14 months,
$5 strikes from 70% to 125% of spot, a dividend yield the fit is not told about),
mids are jittered by 0.2% and given bid/ask spreads, and the calibrator has to
recover the parameters it was generated with. The real SPY fit runs the same
code on a snapshot. Exits non-zero if a check fails.
"""
import sys
import time
import datetime as dt
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kernels import merton_series
from calibrate import prepare, fit_flat, fit_merton, merton_ivs, errors

failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


t0 = time.time()
rng = np.random.default_rng(1)
S0, asof, r, q = 580.0, dt.date(2026, 10, 22), 0.043, 0.013
TRUE = dict(sigma=0.11, lam=0.9, m=-0.09, delta=0.12)
rows = []
for days in (9, 23, 44, 72, 135, 240, 422):
    T = days / 365
    for K in np.arange(round(S0 * 0.7 / 5) * 5, S0 * 1.25, 5):
        for cp in ("call", "put"):
            p = merton_series(S0 * np.exp(-q * T), K, T, r, *TRUE.values(), cp=cp)
            half = max(0.01, 0.004 * p + 0.02) * rng.uniform(0.5, 1.5)
            mid = p * (1 + rng.normal(0, 0.002))
            rows.append(dict(expiry=asof + dt.timedelta(days=days), K=float(K), cp=cp,
                             bid=max(mid - half, 0.0), ask=mid + half))

slices = prepare(S0, asof, rows)
n = sum(len(s["K"]) for s in slices)
fwd = max(abs(s["F"] / (S0 * np.exp((r - q) * s["T"])) - 1) for s in slices)
print(f"synthetic chain: {len(rows)} quotes, {len(slices)} expiries, {n} out-of-the-money mids kept")
print(f"forward from put-call parity vs true forward: max relative error {fwd:.1e}   "
      f"{check(fwd < 5e-4, 'forward')}")
s_flat, flat = fit_flat(slices)
x = fit_merton(slices)
rms_f, max_f = errors(flat, slices)
rms_m, max_m = errors(merton_ivs(slices, *x, pricer="blur"), slices)
print()
print("               sigma     lam      m        delta")
print("  true      " + "".join(f"{v:9.4f}" for v in TRUE.values()))
print("  fitted    " + "".join(f"{v:9.4f}" for v in x))
err = max(abs(a - b) for a, b in zip(x, TRUE.values()))
print(f"  max parameter error {err:.1e}   {check(err < 0.01, 'parameter recovery')}")
print()
print("  fit error, implied vol points       RMS      max")
print(f"  flat Black-Scholes ({s_flat:.1%})    {rms_f:8.2f} {max_f:8.2f}")
print(f"  Merton (priced by the blur)      {rms_m:8.3f} {max_m:8.3f}   "
      f"{check(rms_m < 0.05, 'merton fit')}  (the 0.2% price noise alone)")
print(f"\ntotal time {time.time() - t0:.0f} s")
if failures:
    print("FAILED: " + ", ".join(failures))
    sys.exit(1)
print("ALL PASS")
