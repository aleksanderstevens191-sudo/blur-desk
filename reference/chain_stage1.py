"""Stage 1 of the full-chain work: load, filter, parity forwards, and the American gap.

python reference/chain_stage1.py data/spy_full_<date>/chain.csv  -> report on stdout
(the committed copy is reference/chain_stage1_results.txt)
"""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chain import load, apply_filters, american_forwards, european_ivs, early_exercise, FILTERS

path = Path(sys.argv[1])
t_start = time.time()
meta, rows = load(path)
exps = sorted({r["expiry"] for r in rows})
print(f"SPY option chain snapshot {path}")
print(f"  source      {meta['label']}")
print(f"  quotes at   {meta['t0']:%Y-%m-%d %H:%M:%S %Z}" + (f"  (fetched {meta['fetched']})" if "fetched" in meta else
                                                        "  (no time recorded; taken as the close)"))
print(f"  spot        {meta['spot']:.2f}")
print(f"  contracts   {len(rows)}: {sum(r['cp'] == 'call' for r in rows)} calls, "
      f"{sum(r['cp'] == 'put' for r in rows)} puts, {len(exps)} expiries ({exps[0]} to {exps[-1]}), "
      f"{len({r['K'] for r in rows})} distinct strikes")

yahoo = path.parent / "yahoo_raw.csv"
if yahoo.exists():                                    # second source, same moment: do the quotes agree?
    import csv
    ours = {(str(r["expiry"]), r["K"], r["cp"]): r for r in rows}
    same = diff = 0
    gaps = []
    for q in csv.DictReader(open(yahoo)):
        r = ours.get((q["expiry"], float(q["strike"]), q["type"]))
        if r is None or r["bid"] is None or not float(q["ask"] or 0) > 0 or not (r["ask"] or 0) > 0:
            continue
        m_y = 0.5 * (float(q["bid"]) + float(q["ask"]))
        m_c = 0.5 * (r["bid"] + r["ask"])
        gaps.append(abs(m_y - m_c) / max(0.5 * (r["ask"] - r["bid"]), 0.005))
        same += abs(m_y - m_c) < 0.005
    g = np.array(gaps)
    print(f"  cross-check yahoo_raw.csv (fetched the same minute): {len(g)} two-sided contracts in both; mids equal "
          f"on {same}, median gap {np.median(g):.2f} half-spreads, {np.mean(g > 1):.1%} differ by more than a half-spread")
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
t0 = time.time()
# Forwards use every two-sided quote: parity needs a clean call and put at the same strike, not
# volume, and the far expiries trade only a handful of strikes a day.
fwd, curve_day, hist = american_forwards(meta, two_sided, two_sided)
plain = hist[0]
print(f"Spot implied by the options (D F of expiries with no ex-dividend date before them): "
      f"{meta['spot_implied']:.2f} vs quoted {meta['spot']:.2f}")
print(f"Forwards from put-call parity, discounting on the US Treasury curve of {curve_day} (no dividend yield assumed)")
print("  F plain: median of K + (C - P)/D over the 6 strikes nearest the money, raw mids (American prices), using")
print("  every two-sided quote (parity needs a call and a put at one strike, not volume).")
print(f"  F adj: the same after subtracting each option's early-exercise premium, iterated to a fixed point "
      f"({len(hist) - 1} passes, {time.time() - t0:.0f} s).")
print("  spread: max - min of the 6 strike estimates, plain then adj. noise: the quotes' half-spread on C - P, in dollars of F.")
print("  PV divs: implied spot - D F (adj); each: per quarterly ex-date (3rd Friday of Mar/Jun/Sep/Dec, SPY's")
print("  schedule). q: the same carry as a flat yield. box r: rate implied by regressing C - P on K within 5% of")
print("  F, plain mids then premium-adjusted (Treasury r for comparison).")
print(f"  {'expiry':10s} {'days':>6s} {'kept':>5s} {'r':>6s} {'F plain':>9s} {'F adj':>9s} {'spread plain':>12s} {'adj':>5s} {'noise':>6s} "
      f"{'PV divs':>8s} {'ex':>3s} {'each':>6s} {'q':>7s} {'box r plain':>11s} {'adj':>7s}")
pct = lambda x: f"{x:7.2%}" if x is not None else "      -"
for exp in exps:
    s = fwd.get(exp)
    n = sum(r["expiry"] == exp for r in kept)
    if s is None:
        continue
    if s["F"] is None:
        print(f"  {exp}  {s['T'] * 365:6.2f} {n:5d}  skipped: {s['why']}")
        continue
    each = f"{s['div_each']:6.2f}" if s["div_each"] is not None else "     -"
    note = ""
    print(f"  {exp} {s['T'] * 365:6.2f} {n:5d} {s['r']:6.2%} {plain[exp][0]:9.2f} {s['F']:9.2f} {plain[exp][2]:12.2f} {s['spread']:5.2f} "
          f"{s['noise']:6.2f} {s['pv_div']:8.2f} {len(s['ex_dates']):3d} {each} {s['q']:7.2%}     "
          f"{pct(plain[exp][1])} {pct(s['box_rate'])}{note}")
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
    vp = f"{np.median(v):7.3f} {np.percentile(v, 95):7.2f}" if len(v) else f"{'-':>7s} {'-':>7s}"
    print(f"  {label:26s} {len(group):6d} {np.median(e):8.3f} {e.max():8.3f} {np.median(e / h):7.2f} "
          f"{np.mean(e > h):7.0%} {vp} {len(group) - len(v):6d}")


print(f"  {'':26s} {'n':>6s} {'med $':>8s} {'max $':>8s} {'med/hs':>7s} {'>hs':>7s} {'med vp':>7s} {'p95 vp':>7s} {'no vol':>6s}")
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

# Hand-off to the GPU stages: forwards and every contract that survived the filters.
import json
out = path.parent / "stage1.json"
json.dump({
    "source": meta["label"], "quote_time": meta["t0"].isoformat(), "spot": meta["spot"],
    "spot_implied": meta["spot_implied"], "curve": curve_day,
    "expiries": [{"expiry": str(e), "T": s["T"], "r": s["r"], "D": float(s["D"]), "F": s["F"],
                  "F_plain": plain[e][0], "S": float(s["S"]), "pv_div": float(s["pv_div"]),
                  "div_each": s["div_each"], "spread": s["spread"]}
                 for e, s in sorted(fwd.items()) if s["F"] is not None],
    "contracts": [{"expiry": str(r["expiry"]), "K": r["K"], "cp": r["cp"], "bid": r["bid"], "ask": r["ask"],
                   "volume": r["volume"], "iv": r["iv"] if np.isfinite(r["iv"]) else None,
                   "eep": r.get("eep"), "iv_true": (r.get("iv_true") if np.isfinite(r.get("iv_true", np.nan))
                                                    else None)}
                  for r in kept if fwd.get(r["expiry"]) and fwd[r["expiry"]]["F"] is not None],
}, open(out, "w"), separators=(",", ":"))
print(f"\nwrote {out}: {len(fwd)} expiries, forwards, and the {len(kept)} filtered contracts with vols and premiums")
print(f"total time {time.time() - t_start:.0f} s")
