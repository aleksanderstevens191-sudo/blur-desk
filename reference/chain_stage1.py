"""Stage 1 of the full-chain work: load, filter, parity forwards, and the American gap.

python reference/chain_stage1.py data/spy_full_<date>/chain.csv  -> report on stdout
(the committed copy is reference/chain_stage1_results.txt)
"""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chain import load, apply_filters, parity_forwards, european_ivs, early_exercise, FILTERS

path = Path(sys.argv[1])
t_start = time.time()
meta, rows = load(path)
exps = sorted({r["expiry"] for r in rows})
print(f"SPY option chain snapshot {path}")
print(f"  source      {meta.get('label', meta.get('source', '?')).replace('_', ' ')}")
print(f"  timestamp   {meta['t0']:%Y-%m-%d %H:%M %Z} (source stamp {meta.get('source_stamp', 'n/a')})")
print(f"  spot        {meta['spot']:.2f}")
print(f"  contracts   {len(rows)}: {sum(r['cp'] == 'call' for r in rows)} calls, "
      f"{sum(r['cp'] == 'put' for r in rows)} puts, {len(exps)} expiries ({exps[0]} to {exps[-1]}), "
      f"{len({r['K'] for r in rows})} distinct strikes")
print()

kept, report = apply_filters(meta, rows)
print("Filters, applied in this order (a contract is counted under the first filter that removes it)")
print(f"  {'filter':40s} {'removed':>8s} {'on its own':>11s}")
for name, n, alone in report:
    print(f"  {name:40s} {n:8d} {alone:11d}")
if not meta["has_volume"]:
    print("  (this file has no volume column, so the volume filter removes nothing)")
print(f"  {'kept':40s} {len(kept):8d}   ({len(kept) / len(rows):.0%} of the chain)")
print()

# parity may fall back on two-sided quotes the volume filter dropped (a far expiry can trade nothing all day)
two_sided = [r for r in rows if not any(bad(r, meta) for name, bad in FILTERS[:3])]
fwd, curve_day = parity_forwards(meta, kept, fallback_rows=two_sided)
print(f"Spot implied by the options (D F of expiries with no ex-dividend date before them): "
      f"{meta['spot_implied']:.2f} vs quoted {meta['spot']:.2f}")
print(f"Forwards from put-call parity, discounting on the US Treasury curve of {curve_day} (no dividend yield assumed)")
print("  F: median of K + (C - P)/D over the 6 strikes nearest the money. spread: max - min of those 6.")
print("  noise: the quotes' own half-spread on C - P, in dollars of F. PV divs: implied spot - D F; each: per")
print("  quarterly ex-date (3rd Friday of Mar/Jun/Sep/Dec, SPY's schedule). q: the same carry as a flat yield.")
print(f"  {'expiry':10s} {'days':>6s} {'kept':>5s} {'r':>6s} {'F':>9s} {'spread':>7s} {'noise':>6s} "
      f"{'PV divs':>8s} {'ex':>3s} {'each':>6s} {'q':>7s} {'box r':>7s}")
for exp in exps:
    s = fwd.get(exp)
    n = sum(r["expiry"] == exp for r in kept)
    if s is None:
        continue
    if s["F"] is None:
        print(f"  {exp}  {s['T'] * 365:6.2f} {n:5d}  skipped: {s['why']}")
        continue
    each = f"{s['div_each']:6.2f}" if s["div_each"] is not None else "     -"
    box = f"{s['box_rate']:7.2%}" if s["box_rate"] is not None else "      -"
    note = "" if s["source"] == "filtered" else "  *"
    print(f"  {exp} {s['T'] * 365:6.2f} {n:5d} {s['r']:6.2%} {s['F']:9.2f} {s['spread']:7.2f} {s['noise']:6.2f} "
          f"{s['pv_div']:8.2f} {len(s['ex_dates']):3d} {each} {s['q']:7.2%} {box}{note}")
if any(s.get("source", "filtered") != "filtered" for s in fwd.values()):
    print("  * not enough traded pairs; forward from two-sided quotes that had no volume")
print()

european_ivs(kept, fwd)
bad_iv = [r for r in kept if not np.isfinite(r["iv"])]
print(f"European implied vols on the parity forwards: {len(kept) - len(bad_iv)} of {len(kept)} contracts invert")
if bad_iv:
    itm = sum((r["cp"] == "call") == (r["K"] < fwd[r["expiry"]]["F"]) for r in bad_iv if fwd[r["expiry"]]["F"])
    print(f"  {len(bad_iv)} mids lie outside the European no-arbitrage bounds ({itm} of them in the money): "
          "dropped from pricing")
print()

t0 = time.time()
early_exercise(meta, kept, fwd)
priced = [r for r in kept if "eep" in r]
print(f"American vs European: early-exercise premium of every contract ({len(priced)} contracts, "
      f"QuantLib FD 200 x 400, {time.time() - t0:.0f} s)")
print("  The market mid is an American price. Pricing it as European on the parity forward misses the")
print("  early-exercise premium EEP = American - European (same vol, same dividends). Columns: EEP in dollars,")
print("  EEP over the quote's half-spread, and EEP in vol points (how far the European implied vol of the")
print("  mid sits above the American one). no vol: mid - EEP is below the European lower bound, i.e. the")
print("  option trades at or near exercise value and carries no usable European vol.")


def bucket(r):
    F = fwd[r["expiry"]]["F"]
    k = np.log(r["K"] / F) * (1 if r["cp"] == "call" else -1)    # > 0 out of the money
    if k > 0.02:
        return "OTM (> 2%)"
    if k > -0.02:
        return "ATM (within 2%)"
    if k > -0.10:
        return "ITM 2-10%"
    return "ITM > 10%"


def stats(group, label):
    if not group:
        return
    e = np.array([r["eep"] for r in group])
    h = np.array([r["half"] for r in group])
    v = np.array([r["eep_volpts"] for r in group])
    v = v[np.isfinite(v)]
    vp = f"{np.median(v):7.3f} {v.max():7.2f}" if len(v) else f"{'-':>7s} {'-':>7s}"
    print(f"  {label:26s} {len(group):6d} {np.median(e):8.3f} {e.max():8.3f} {np.median(e / h):7.2f} "
          f"{np.mean(e > h):7.0%} {vp} {len(group) - len(v):6d}")


print(f"  {'':26s} {'n':>6s} {'med $':>8s} {'max $':>8s} {'med/hs':>7s} {'>hs':>7s} {'med vp':>7s} {'max vp':>7s} {'no vol':>6s}")
for cp in ("call", "put"):
    for b in ("OTM (> 2%)", "ATM (within 2%)", "ITM 2-10%", "ITM > 10%"):
        stats([r for r in priced if r["cp"] == cp and bucket(r) == b], f"{cp}s {b}")
stats(priced, "all")
print()
print("  Out-of-the-money and at-the-money contracts only, by time to expiry")
for lo, hi, lab in ((0, 30, "up to 30 days"), (30, 180, "30 to 180 days"), (180, 1e9, "over 180 days")):
    stats([r for r in priced if not bucket(r).startswith("ITM") and lo <= fwd[r["expiry"]]["T"] * 365 < hi], lab)
    stats([r for r in priced if bucket(r).startswith("ITM") and lo <= fwd[r["expiry"]]["T"] * 365 < hi],
          "  (in the money)")
print()
otm = [r for r in priced if not bucket(r).startswith("ITM")]
print(f"Summary: {sum(r['eep'] > r['half'] for r in priced)} of {len(priced)} contracts have an early-exercise "
      f"premium wider than half their bid-ask spread;")
print(f"  out of the money and at the money: {sum(r['eep'] > r['half'] for r in otm)} of {len(otm)}. "
      "Pricing from here on is European on parity forwards; the European model is")
print("  reported on every contract, and the comparisons that matter (smile, fits) use the OTM side.")
print(f"\ntotal time {time.time() - t_start:.0f} s")
