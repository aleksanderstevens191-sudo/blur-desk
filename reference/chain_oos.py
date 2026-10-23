"""Stage 4: out-of-sample test of Merton against flat Black-Scholes on real SPY quotes.

python reference/chain_oos.py   -> report on stdout (reference/chain_oos_results.txt)

Each test fits on one set of quotes and measures errors on another, in vol points
(1 point = 1% of vol), RMS and max, against the European-equivalent market vols
(mid minus the early-exercise premium, inverted on the parity forward; stage 1).
Merton: sigma, lam, m, delta, one set for every quote in the training set.
Flat Black-Scholes: one vol for the training set (vega-weighted), used as is on the test.
Both are fitted by the same rule (least squares on vega-scaled price errors) and
judged by the same errors, out-of-the-money quotes only (puts below the forward,
calls above).

  1. Across expiries, same day: fit on Nov 20, test on Dec 18; then the reverse.
     Oct 23 (the full Cboe snapshot) and Oct 22 (the earlier 2-expiry Yahoo file).
  2. Across days: fit on Oct 22 (both expiries), test on Oct 23's Nov 20 and Dec 18.

Errors are reported on two domains: every OTM quote the filters kept, and the core,
|ln K/F| <= 4%, which is the strike range the Oct 22 file covers.
"""
import sys
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from chain import load, apply_filters, american_forwards, european_ivs, early_exercise, FILTERS
from chain_fit import black, merton_vec
from smile import implied_vol

ROOT = HERE.parent
PAIR = ("2026-11-20", "2026-12-18")
CORE = 0.04


def slices_from_stage1(path):
    s1 = json.load(open(path))
    out = {}
    for e in s1["expiries"]:
        if e["expiry"] not in PAIR:
            continue
        cs = [c for c in s1["contracts"] if c["expiry"] == e["expiry"] and c["iv_true"] is not None
              and (c["cp"] == "call") == (c["K"] >= e["F"])]
        out[e["expiry"]] = _slice(e, cs)
    return out


def slices_from_csv(path):
    """The Oct 22 file through the same stage 1 pipeline (no volume column, so no volume filter)."""
    meta, rows = load(path)
    kept, _ = apply_filters(meta, rows)
    two = [r for r in rows if not any(bad(r, meta) for _, bad in FILTERS[:3])]
    fwd, _, _ = american_forwards(meta, two, two)
    european_ivs(kept, fwd)
    early_exercise(meta, kept, fwd)
    out = {}
    for exp, s in fwd.items():
        if str(exp) not in PAIR or s["F"] is None:
            continue
        cs = [dict(K=r["K"], cp=r["cp"], bid=r["bid"], ask=r["ask"], eep=r.get("eep"),
                   iv_true=r.get("iv_true") if np.isfinite(r.get("iv_true", np.nan)) else None)
              for r in kept if r["expiry"] == exp]
        cs = [c for c in cs if c["iv_true"] is not None and (c["cp"] == "call") == (c["K"] >= s["F"])]
        out[str(exp)] = _slice(dict(expiry=str(exp), T=s["T"], D=float(s["D"]), F=s["F"], r=s["r"]), cs)
    return out


def _slice(e, cs):
    K = np.array([c["K"] for c in cs])
    iv = np.array([c["iv_true"] for c in cs])
    call = np.array([c["cp"] == "call" for c in cs])
    T, F, D = e["T"], e["F"], e["D"]
    vega = D * F * norm.pdf((np.log(F / K) + 0.5 * iv * iv * T) / (iv * np.sqrt(T))) * np.sqrt(T)
    return dict(expiry=e["expiry"], T=T, F=F, D=D, r=e["r"], K=K, call=call, iv=iv,
                vega=np.maximum(vega, 1e-3 * D * F * np.sqrt(T)), tgt=black(F, K, iv * np.sqrt(T), D, call))


def fit(sl):
    """Merton and flat Black-Scholes fitted jointly to the slices in sl."""
    def res(x):
        return np.concatenate([(merton_vec(s["F"], s["K"], s["T"], s["D"], s["call"], *x) - s["tgt"]) / s["vega"]
                               for s in sl])
    lo, hi = [0.01, 0.0, -0.6, 0.005], [1.0, 30.0, 0.3, 0.8]
    best = None
    for x0 in [(0.096, 1.67, -0.095, 0.05), (0.12, 0.5, -0.15, 0.10), (0.08, 5.0, -0.04, 0.03)]:
        r = least_squares(res, x0, bounds=(lo, hi), x_scale=[0.05, 1.0, 0.05, 0.05], xtol=1e-12, ftol=1e-12)
        if best is None or r.cost < best.cost:
            best = r
    def res_flat(v):
        return np.concatenate([(black(s["F"], s["K"], v[0] * np.sqrt(s["T"]), s["D"], s["call"]) - s["tgt"]) / s["vega"]
                               for s in sl])
    flat = least_squares(res_flat, [0.15], bounds=([0.01], [2.0]), xtol=1e-14, ftol=1e-14).x[0]
    return best.x, flat


def errors(x, flat, s, core_only=False):
    """Vol-point errors of Merton (params x) and flat vol on slice s: (n, rms_m, max_m, rms_f, max_f)."""
    m = np.abs(np.log(s["K"] / s["F"])) <= CORE if core_only else np.ones(len(s["K"]), bool)
    if not m.any():
        return None
    K, call = s["K"][m], s["call"][m]
    pm = merton_vec(s["F"], K, s["T"], s["D"], call, *x)
    S = s["D"] * s["F"]
    ivm = np.array([implied_vol(p, S, k, s["T"], s["r"], "call" if c else "put") for p, k, c in zip(pm, K, call)])
    d_m = (ivm - s["iv"][m]) * 100
    d_f = (flat - s["iv"][m]) * 100
    nan = int(np.sum(~np.isfinite(d_m)))
    d_m = np.where(np.isfinite(d_m), d_m, np.nan)
    rms = lambda d: float(np.sqrt(np.nanmean(d * d)))
    return int(m.sum()), rms(d_m), float(np.nanmax(np.abs(d_m))), rms(d_f), float(np.max(np.abs(d_f))), nan


def report(title, train, tests):
    x, flat = fit(train)
    print(f"{title}")
    print(f"  Merton fit: sigma {x[0]:.2%}, lam {x[1]:.2f}/yr, m {x[2]:+.2%}, delta {x[3]:.2%};  flat vol {flat:.2%}")
    print(f"  {'':34s} {'domain':6s} {'n':>4s}  {'Merton RMS':>10s} {'max':>6s}   {'flat RMS':>8s} {'max':>6s}   "
          f"{'Merton/flat':>11s}")
    rows = []
    for label, s in [("in sample: " + "+".join(t["expiry"] for t in train), None)] + tests:
        for core in (False, True):
            if s is None:
                parts = [errors(x, flat, t, core) for t in train]
                parts = [p for p in parts if p]
                n = sum(p[0] for p in parts)
                rm = np.sqrt(sum(p[0] * p[1] ** 2 for p in parts) / n)
                rf = np.sqrt(sum(p[0] * p[3] ** 2 for p in parts) / n)
                e = (n, rm, max(p[2] for p in parts), rf, max(p[4] for p in parts), sum(p[5] for p in parts))
            else:
                e = errors(x, flat, s, core)
            if e is None:
                continue
            n, rm, mm, rf, mf, nan = e
            print(f"  {label:34s} {'core' if core else 'all':6s} {n:4d}  {rm:10.2f} {mm:6.2f}   {rf:8.2f} {mf:6.2f}   "
                  f"{rm / rf:11.2f}" + (f"  ({nan} quotes Merton can't reach)" if nan else ""))
            rows.append(dict(test=label, domain="core" if core else "all", n=n, merton_rms=rm, merton_max=mm,
                             flat_rms=rf, flat_max=mf))
    print()
    return dict(title=title, merton=list(map(float, x)), flat=float(flat), rows=rows)


if __name__ == "__main__":
    d23 = slices_from_stage1(ROOT / "data" / "spy_full_2026-10-23" / "stage1.json")
    d22 = slices_from_csv(ROOT / "data" / "spy_chain_2026-10-22.csv")
    print("Out-of-sample: Merton vs flat Black-Scholes, errors in vol points against European-equivalent market vols")
    print(f"  all: every out-of-the-money quote kept by the stage 1 filters; core: |ln K/F| <= {CORE:.0%}")
    for day, d in (("Oct 23", d23), ("Oct 22", d22)):
        print(f"  {day}: " + ", ".join(f"{e} {len(s['K'])} OTM quotes, strikes {s['K'].min():g}-{s['K'].max():g}"
                                       for e, s in d.items()))
    print()
    out = []
    a, b = PAIR
    for day, d in (("Oct 23", d23), ("Oct 22", d22)):
        out.append(report(f"1. {day}: fit on {a}, test on {b}", [d[a]], [(f"out of sample: {b}", d[b])]))
        out.append(report(f"1. {day}: fit on {b}, test on {a}", [d[b]], [(f"out of sample: {a}", d[a])]))
    out.append(report("2. Fit on Oct 22 (both expiries), test on Oct 23", [d22[a], d22[b]],
                      [(f"out of sample: Oct 23 {a}", d23[a]), (f"out of sample: Oct 23 {b}", d23[b])]))
    json.dump(out, open(HERE / "chain_oos_results.json", "w"), indent=1)
