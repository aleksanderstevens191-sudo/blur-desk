"""Stage 2, Python side: fit Merton per expiry and write the GPU's inputs and reference prices.

python reference/chain_fit.py [data/spy_full_<date>] [out.json]  -> report on stdout, src/chain_data.json

Input is stage1.json (chain_stage1.py): forwards and the filtered contracts with
their European implied vols. Every contract with a European vol is priced twice:

  Black-Scholes at the contract's own implied vol: the European vol of its mid on
    the parity forward, so it should give back the mid. Reference: QuantLib's
    BlackCalculator on (forward, strike, sigma sqrt T, discount).
  Merton, fitted per expiry: sigma, lam, m, delta by least squares on the
    out-of-the-money European-equivalent vols (mid minus the early-exercise
    premium, inverted on the parity forward). Reference: the Merton series.

The fit minimises price errors scaled by vega (= vol errors to first order); the
report gives the exact vol errors after inverting the fitted prices.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from scipy.stats import poisson, norm
import QuantLib as ql

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from kernels import merton_series
from smile import implied_vol

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "src" / "chain_data.json"


def black(F, K, v, D, call):
    """Black's formula, vectorised. v = total vol sigma sqrt T."""
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    c = D * (F * ndtr(d1) - K * ndtr(d2))
    return np.where(call, c, c - D * (F - K))


def merton_vec(F, K, T, D, call, sigma, lam, m, delta, tail=1e-14):
    """Merton as a Poisson mixture of Black prices on the forward (r drops out): with n
    jumps (Poisson(lam T)) the forward is F e^(n (m + delta^2/2) - lam kappa T)."""
    kappa = np.expm1(m + 0.5 * delta**2)
    L = lam * T
    out = np.zeros_like(K)
    n, total = 0, 0.0
    while True:
        pn = poisson.pmf(n, L)
        Fn = F * np.exp(n * (m + 0.5 * delta**2) - lam * kappa * T)
        out += pn * black(Fn, K, np.sqrt(sigma**2 * T + n * delta**2), D, call)
        total += pn
        if n > L and (1 - total < tail or pn < tail * 1e-3):
            return out
        n += 1


def fit_expiry(e, cs):
    """Merton and flat Black-Scholes on one expiry's OTM European-equivalent vols."""
    F, T, D = e["F"], e["T"], e["D"]
    otm = [c for c in cs if c["iv_true"] is not None and (c["cp"] == "call") == (c["K"] >= F)]
    K = np.array([c["K"] for c in otm])
    call = np.array([c["cp"] == "call" for c in otm])
    iv = np.array([c["iv_true"] for c in otm])
    eep = np.array([c["eep"] or 0.0 for c in otm])
    bid = np.array([c["bid"] for c in otm]) - eep
    ask = np.array([c["ask"] for c in otm]) - eep
    tgt = black(F, K, iv * np.sqrt(T), D, call)
    vega = D * F * norm.pdf((np.log(F / K) + 0.5 * iv * iv * T) / (iv * np.sqrt(T))) * np.sqrt(T)
    vega = np.maximum(vega, 1e-3 * D * F * np.sqrt(T))
    res = lambda x: (merton_vec(F, K, T, D, call, *x) - tgt) / vega
    lo, hi = [0.01, 0.0, -0.6, 0.005], [1.0, 30.0, 0.3, 0.8]
    best = None
    for x0 in [(0.096, 1.67, -0.095, 0.05), (0.12, 0.5, -0.15, 0.10), (0.08, 5.0, -0.04, 0.03)]:
        r = least_squares(res, x0, bounds=(lo, hi), x_scale=[0.05, 1.0, 0.05, 0.05], xtol=1e-12, ftol=1e-12)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    model = merton_vec(F, K, T, D, call, *x)
    S, r_ = D * F, e["r"]
    mv = np.array([implied_vol(p, S, k, T, r_, "call" if c else "put") for p, k, c in zip(model, K, call)])
    flat = float(np.sqrt(np.sum(iv * iv * vega**2) / np.sum(vega**2)))     # vega-weighted flat vol
    err = lambda a: (np.sqrt(np.nanmean((a - iv) ** 2)) * 100, np.nanmax(np.abs(a - iv)) * 100)
    inside = lambda pr: float(np.mean((pr >= bid - 1e-9) & (pr <= ask + 1e-9)))
    core = 0.5 * (bid + ask) >= 0.50                     # quotes worth 50 cents or more
    err_core = lambda a: np.sqrt(np.nanmean((a[core] - iv[core]) ** 2)) * 100 if core.any() else np.nan
    fp = black(F, K, flat * np.sqrt(T), D, call)
    return (x, len(otm), err(mv), err(np.full_like(iv, flat)), flat, int(np.sum(~np.isfinite(mv))),
            inside(model), inside(fp), err_core(mv), err_core(np.full_like(iv, flat)), int(core.sum()))


if __name__ == "__main__":
    t0 = time.time()
    s1 = json.load(open(folder / "stage1.json"))
    exps = s1["expiries"]
    priced = [c for c in s1["contracts"] if c["iv"] is not None]
    print(f"SPY chain {folder.name}: quotes at {s1['quote_time']}, {len(exps)} expiries, "
          f"{len(priced)} contracts with a European vol (of {len(s1['contracts'])} after the filters)")
    print()
    print("Merton fitted per expiry on out-of-the-money European-equivalent vols (mid minus early-exercise premium)")
    print("  vs one flat Black-Scholes vol per expiry. Errors in vol points (RMS / max).")
    print(f"  {'expiry':10s} {'days':>6s} {'n':>4s} {'sigma':>7s} {'lam':>6s} {'m':>7s} {'delta':>6s}"
          f" {'Merton rms':>10s} {'max':>6s} {'in b/a':>6s} {'>50c':>6s} {'flat':>6s} {'rms':>6s} {'max':>6s} {'in b/a':>6s} {'>50c':>6s}")
    all_m, all_f, core_m, core_f = [], [], [], []
    for e in exps:
        cs = [c for c in priced if c["expiry"] == e["expiry"]]
        x, n, (mr, mm), (fr, fm), flat, nan, im, iflat, mc, fc, nc = fit_expiry(e, cs)
        e["merton"] = dict(sigma=x[0], lam=x[1], m=x[2], delta=x[3], rms=mr, max=mm, n=n, flat=flat, flat_rms=fr,
                           inside=im, flat_inside=iflat)
        all_m.append((mr, n, im))
        all_f.append((fr, n, iflat))
        core_m.append((mc, nc, 0))
        core_f.append((fc, nc, 0))
        print(f"  {e['expiry']} {e['T'] * 365:6.2f} {n:4d} {x[0]:7.2%} {x[1]:6.2f} {x[2]:+7.2%} {x[3]:6.2%}"
              f" {mr:10.2f} {mm:6.2f} {im:6.0%} {mc:6.2f} {flat:6.2%} {fr:6.2f} {fm:6.2f} {iflat:6.0%} {fc:6.2f}" + (f"  ({nan} unreachable)" if nan else ""))
    agg = lambda a: np.sqrt(sum(r * r * n for r, n, _ in a) / sum(n for _, n, _ in a))
    ins = lambda a: sum(i * n for _, n, i in a) / sum(n for _, n, _ in a)
    print(f"  all expiries: Merton RMS {agg(all_m):.2f} vol pts, {ins(all_m):.0%} inside the bid-ask; flat Black-Scholes "
          f"per expiry {agg(all_f):.2f}, {ins(all_f):.0%} inside")
    print("  in b/a: share of quotes where the model price is inside the bid-ask (less the early-exercise premium).")
    print(f"  >50c: RMS over quotes worth 50 cents or more only: Merton {agg(core_m):.2f}, flat {agg(core_f):.2f} vol pts. The")
    print("  fit weights errors by vega, so far out-of-the-money quotes of a few cents count little in it.")
    print()

    # reference prices for every contract
    t1 = time.time()
    by = {e["expiry"]: e for e in exps}
    bs_ql, mert = [], []
    for c in priced:
        e = by[c["expiry"]]
        typ = ql.Option.Call if c["cp"] == "call" else ql.Option.Put
        bc = ql.BlackCalculator(ql.PlainVanillaPayoff(typ, c["K"]), e["F"], c["iv"] * np.sqrt(e["T"]), e["D"])
        bs_ql.append(bc.value())
        p = e["merton"]
        mert.append(float(merton_series(e["D"] * e["F"], c["K"], e["T"], e["r"], p["sigma"], p["lam"], p["m"],
                                        p["delta"], cp=c["cp"])))
    mid = np.array([0.5 * (c["bid"] + c["ask"]) for c in priced])
    gap = np.abs(np.array(bs_ql) - mid)
    print(f"Reference prices ({time.time() - t1:.0f} s): QuantLib BlackCalculator at each contract's own vol, and the")
    print(f"  Merton series at its expiry's fit. Sanity check, QuantLib Black-Scholes vs the market mid: max "
          f"|diff| {gap.max():.1e} dollars over {len(priced)} contracts")
    # the fit's fast Merton (merton_vec) must be the same model as the series the GPU is checked against
    fast = np.empty(len(priced))
    for e in exps:
        sel = [j for j, c in enumerate(priced) if c["expiry"] == e["expiry"]]
        if sel:
            p = e["merton"]
            fast[sel] = merton_vec(e["F"], np.array([priced[j]["K"] for j in sel]), e["T"], e["D"],
                                   np.array([priced[j]["cp"] == "call" for j in sel]), p["sigma"], p["lam"], p["m"], p["delta"])
    gap = np.max(np.abs(fast - np.array(mert)))
    print(f"  The fit's Merton (Poisson mixture of Black on the forward) vs the series: max |diff| {gap:.1e} dollars")
    assert gap < 1e-8, "the fitted Merton and the reference series disagree"

    idx = {e["expiry"]: i for i, e in enumerate(exps)}
    json.dump({
        "source": s1["source"], "quote_time": s1["quote_time"], "spot": s1["spot"],
        "expiries": [{**{k: e[k] for k in ("expiry", "T", "r", "D", "F")}, "merton": e["merton"]} for e in exps],
        "contracts": {
            "e": [idx[c["expiry"]] for c in priced],
            "K": [c["K"] for c in priced],
            "call": [1 if c["cp"] == "call" else 0 for c in priced],
            "bid": [c["bid"] for c in priced],
            "ask": [c["ask"] for c in priced],
            "iv": [c["iv"] for c in priced],
            "eep": [round(c["eep"], 6) if c["eep"] is not None else None for c in priced],
            "iv_true": [c["iv_true"] for c in priced],
            "bs_ql": bs_ql,
            "merton": mert,
        },
    }, open(OUT, "w"), separators=(",", ":"))
    print(f"\nwrote {OUT.name} ({OUT.stat().st_size / 1e3:.0f} kB): {len(exps)} expiries with "
          f"forwards and Merton fits, {len(priced)} contracts with reference prices")
    print(f"total time {time.time() - t0:.0f} s")
