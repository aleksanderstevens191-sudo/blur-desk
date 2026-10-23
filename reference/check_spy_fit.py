"""The real SPY fit, re-run from the committed snapshot (data/spy_chain_2026-10-22.csv).

Refits flat Black-Scholes and Merton to the 20 out-of-the-money mids exactly as
calibrate_spy.py does and checks that
  - the fit reproduces the parameters committed in src/spy_fit.json (the demo's overlay),
  - Merton's jumps cut the flat model's RMS error at least 4 times,
  - Merton stays within 1 vol point of every quote,
  - the market skew is downward in both expiries, and the fitted mean jump is negative.
Exits non-zero if a check fails.
"""
import sys
import json
import time
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from calibrate import fit_flat, fit_merton, merton_ivs, errors
from calibrate_spy import HERE, SPY_CSV, load_spy

failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


t0 = time.time()
committed = json.loads((HERE.parent / "src" / "spy_fit.json").read_text())
spot, asof, rows, slices, dropped = load_spy()
n = sum(len(s["K"]) for s in slices)
print(f"{SPY_CSV.name}: {len(rows)} quotes as of {asof}, {len(slices)} expiries, {n} out-of-the-money mids")

s_flat, flat = fit_flat(slices)
x = fit_merton(slices)
mert = merton_ivs(slices, *x, pricer="blur")
rms_f, max_f = errors(flat, slices)
rms_m, max_m = errors(mert, slices)

print()
print("  refit vs src/spy_fit.json     refit       committed")
names = ("sigma", "lam", "m", "delta")
rel = 0.0
for k, v in zip(names, x):
    c = committed["merton"][k]
    rel = max(rel, abs(v - c) / abs(c))
    print(f"  Merton {k:8s}              {v:9.5f}   {c:9.5f}")
c = committed["flat"]["sigma"]
rel = max(rel, abs(s_flat - c) / c)
print(f"  flat sigma                   {s_flat:9.5f}   {c:9.5f}")
print(f"  largest relative difference {rel:.1e}   {check(rel < 1e-3, 'refit matches spy_fit.json')}")

print()
print("  fit error, implied vol points   RMS     max")
print(f"  flat Black-Scholes           {rms_f:6.2f}  {max_f:6.2f}")
print(f"  Merton (priced by the blur)  {rms_m:6.2f}  {max_m:6.2f}")
print(f"  Merton RMS is {rms_f / rms_m:.1f} times smaller   {check(rms_f / rms_m > 4, 'Merton beats flat 4x')}")
print(f"  Merton within 1 vol point of every quote   {check(max_m < 1, 'Merton max < 1 vol pt')}")

print()
for s, me in zip(slices, mert):
    mkt = s["iv"][0] - s["iv"][-1]
    mod = me[0] - me[-1]
    ok = check(mkt > 0 and mod > 0, f"downward skew {s['expiry']}")
    print(f"  {s['expiry']}: vol at {s['K'][0]:.0f} minus vol at {s['K'][-1]:.0f}: market {mkt * 100:+.2f}, "
          f"Merton {mod * 100:+.2f} vol pts   {ok}")
print(f"  fitted mean log jump m = {x[2]:+.2%}   {check(x[2] < 0, 'negative mean jump')}")

print(f"\ntotal time {time.time() - t0:.0f} s")
if failures:
    print("FAILED: " + ", ".join(failures))
    sys.exit(1)
print("ALL PASS")
