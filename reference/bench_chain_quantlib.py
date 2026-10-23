"""QuantLib on the CPU for the full-chain benchmark: the same 4,738 SPY contracts the GPU prices
(src/chain_data.json), on one thread and on every core.

    python3 reference/bench_chain_quantlib.py [runs]     (default 20 runs)

Black-Scholes at each contract's own vol: QuantLib's BlackCalculator, the closed form with
no instrument or engine around it, the fastest way to get a Black price out of QuantLib.
Merton per expiry: QuantLib has no Merton engine in its Python bindings, so Merton is its
Bates engine (Heston plus Merton jumps, semi-analytic Fourier integral) with the variance
pinned (v0 = theta = sigma^2, vol of vol 1e-5); one process and engine per expiry, one
instrument per contract. QuantLib counts time in whole days, so each expiry's rates and
variances are rescaled by T / T_days to match the snapshot's intraday times exactly.

Each run prices every contract from scratch (engines and instruments included). Timed:
median of the runs, with min and max. "All cores" splits the contracts across a pool of
os.cpu_count() processes (started and warmed before timing) and times each pass end to end.
Prints a JSON line at the end for the README table.
"""
import os
import sys
import json
import time
import platform
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import QuantLib as ql

DATA = Path(__file__).parent.parent / "src" / "chain_data.json"
TODAY = (2026, 10, 23)
_data = None


def load():
    global _data
    if _data is None:
        _data = json.load(open(DATA))
    return _data


def price_bs(idx):
    d = load()
    C, E = d["contracts"], d["expiries"]
    out = []
    for i in idx:
        e = E[C["e"][i]]
        typ = ql.Option.Call if C["call"][i] else ql.Option.Put
        bc = ql.BlackCalculator(ql.PlainVanillaPayoff(typ, C["K"][i]), e["F"], C["iv"][i] * np.sqrt(e["T"]), e["D"])
        out.append(bc.value())
    return out


def price_merton(idx):
    d = load()
    C, E = d["contracts"], d["expiries"]
    today = ql.Date(TODAY[2], TODAY[1], TODAY[0])
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    engines = {}
    out = []
    for i in idx:
        ei = C["e"][i]
        e = E[ei]
        y, mth, dd = map(int, e["expiry"].split("-"))
        exp = ql.Date(dd, mth, y)
        if ei not in engines:
            rho = e["T"] / ((exp - today) / 365.0)          # QuantLib's whole-day clock -> intraday T
            p = e["merton"]
            r = e["r"] * rho
            flat = lambda x: ql.YieldTermStructureHandle(ql.FlatForward(today, x, dc, ql.Continuous))
            v = p["sigma"] ** 2 * rho
            proc = ql.BatesProcess(flat(r), flat(0.0), ql.QuoteHandle(ql.SimpleQuote(e["D"] * e["F"])),
                                   v, 1.0, v, 1e-5, 0.0, p["lam"] * rho, p["m"], p["delta"])
            engines[ei] = ql.BatesEngine(ql.BatesModel(proc))
        typ = ql.Option.Call if C["call"][i] else ql.Option.Put
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(typ, C["K"][i]), ql.EuropeanExercise(exp))
        opt.setPricingEngine(engines[ei])
        out.append(opt.NPV())
    return out


def timed(fn, runs):
    ts = []
    res = None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return res, dict(median=float(np.median(ts)), min=float(ts.min()), max=float(ts.max()),
                     p25=float(np.percentile(ts, 25)), p75=float(np.percentile(ts, 75)))


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    d = load()
    n = len(d["contracts"]["K"])
    cores = os.cpu_count()
    allidx = list(range(n))
    chunks = [allidx[k::cores] for k in range(cores)]      # interleaved: every worker gets every expiry
    print(f"QuantLib {ql.__version__}, Python {platform.python_version()}, {platform.machine()} "
          f"{platform.processor() or ''}, {cores} cores")
    print(f"{n} SPY contracts, {len(d['expiries'])} expiries ({d['quote_time']}); {runs} runs each, "
          f"times in ms: median [min, max]")
    res = {"quantlib": ql.__version__, "cores": cores, "contracts": n, "runs": runs, "machine": platform.platform()}
    with Pool(cores, initializer=load) as pool:
        pool.map(price_bs, [[0]] * cores)                    # warm the workers
        for name, fn, ref in (("bs", price_bs, "bs_ql"), ("merton", price_merton, "merton")):
            one, t1 = timed(lambda: fn(allidx), runs)
            def par():
                parts = pool.map(fn, chunks)
                out = [0.0] * n
                for idx, p in zip(chunks, parts):
                    for i, v in zip(idx, p):
                        out[i] = v
                return out
            many, tn = timed(par, runs)
            err = max(abs(a - b) for a, b in zip(one, d["contracts"][ref]))
            agree = max(abs(a - b) for a, b in zip(one, many))
            label = {"bs": "Black-Scholes (BlackCalculator)", "merton": "Merton (Bates engine, vol of vol 1e-5)"}[name]
            print(f"\n{label}: max |QuantLib - reference| {err:.1e} dollars")
            for tag, t in (("1 thread", t1), (f"{cores} cores", tn)):
                print(f"  {tag:9s} {t['median'] * 1e3:9.1f} [{t['min'] * 1e3:.1f}, {t['max'] * 1e3:.1f}] ms"
                      f"  ->  {n / t['median']:12,.0f} contracts/sec")
            res[name] = {"one": t1, "all": tn, "max_err_vs_ref": err}
    print("\nJSON " + json.dumps(res))
