"""The full SPY chain: load a snapshot, filter it, and read each expiry's forward from put-call parity.

Snapshot format (reference/fetch_spy_chain_full.py writes it): a header comment
"# spot=... asof=YYYY-MM-DD fetched=<ISO time> source=... contracts=..." and the
columns expiry,strike,type,bid,ask[,bid_size,ask_size,last,volume,open_interest,
last_trade_utc,vendor_iv]. The older two-expiry file (bid/ask only, no time)
loads too; its quotes are taken as of the 16:00 ET close.

Time to expiry is measured in calendar time from the snapshot to 16:00 ET on the
expiry date (SPY options are PM-settled), in years of 365 days.

Rates: the US Treasury constant-maturity curve from the last close at or before
the snapshot (FMP treasury-rates), bond-equivalent yields converted to continuous
compounding and interpolated linearly in days. The discount factor D = e^{-rT}
comes from that curve, never from the options. The forward comes from the options:
put-call parity C - P = D (F - K) at the strikes nearest the money. No dividend
yield is assumed anywhere; the dividends to each expiry fall out as S0 - D F.

SPY options are American. Everything priced from here on is European on these
parity forwards; early_exercise() measures what that leaves out, contract by
contract, with QuantLib's finite-difference American engine.
"""
import csv
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np

from smile import implied_vol

ET = ZoneInfo("America/New_York")
YEAR = 365.0 * 86400.0

# FMP treasury-rates, percent, bond-equivalent: tenor days -> yield
TENORS = np.array([30, 61, 91, 182, 365, 730, 1095, 1826])
TREASURY = {
    "2026-10-20": [4.10, 4.17, 4.26, 4.36, 4.51, 4.79, 4.86, 4.94],
    "2026-10-21": [4.09, 4.16, 4.25, 4.35, 4.52, 4.81, 4.89, 4.95],
    "2026-10-22": [4.11, 4.16, 4.22, 4.29, 4.41, 4.65, 4.72, 4.78],
}


def curve_for(asof):
    """Continuous-compounding rate function days -> r, from the last close at or before asof."""
    day = max(d for d in TREASURY if d <= str(asof))
    y = np.array(TREASURY[day]) / 100
    rc = 2 * np.log1p(y / 2)                       # semiannual bond-equivalent -> continuous
    return day, lambda days: float(np.interp(days, TENORS, rc))


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load(path):
    """Returns (meta dict, list of contract dicts). meta has spot, asof, t0 (aware datetime), source."""
    meta = {}
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    for ln in lines:
        if ln.startswith("#"):
            for tok in ln[1:].split():
                k, _, v = tok.partition("=")
                if v:
                    meta[k] = v
    meta["spot"] = float(meta["spot"])
    meta["asof"] = dt.date.fromisoformat(meta["asof"])
    if "fetched" in meta:
        meta["t0"] = dt.datetime.fromisoformat(meta["fetched"]).astimezone(ET)
    else:
        meta["t0"] = dt.datetime.combine(meta["asof"], dt.time(16, 0), ET)
    rows = []
    for r in csv.DictReader(ln for ln in lines if not ln.startswith("#")):
        rows.append(dict(expiry=dt.date.fromisoformat(r["expiry"]), K=float(r["strike"]),
                         cp=r["type"].strip().lower(), bid=_num(r["bid"]), ask=_num(r["ask"]),
                         volume=_num(r.get("volume")), oi=_num(r.get("open_interest")),
                         vendor_iv=_num(r.get("vendor_iv"))))
    meta["has_volume"] = any(r["volume"] is not None for r in rows)
    return meta, rows


def years_to(expiry, t0):
    return (dt.datetime.combine(expiry, dt.time(16, 0), ET) - t0).total_seconds() / YEAR


# Each filter: (name, test that returns True when the contract is REMOVED). Applied in this order.
FILTERS = [
    ("expires on the snapshot day", lambda r, m: r["expiry"] <= m["asof"]),
    ("zero or missing bid", lambda r, m: r["bid"] is None or r["bid"] <= 0),
    ("crossed or locked quote (ask <= bid)", lambda r, m: r["ask"] is None or r["bid"] is None
     or r["ask"] <= r["bid"]),
    ("no volume today", lambda r, m: m["has_volume"] and not (r["volume"] or 0) > 0),
]


def apply_filters(meta, rows):
    """Sequential filters. Returns (kept rows, report) where report rows are
    (name, removed in sequence, would remove on its own)."""
    report, keep = [], rows
    for name, bad in FILTERS:
        alone = sum(bad(r, meta) for r in rows)
        nxt = [r for r in keep if not bad(r, meta)]
        report.append((name, len(keep) - len(nxt), alone))
        keep = nxt
    return keep, report


def third_fridays(start, end):
    """SPY goes ex-dividend on the third Friday of March, June, September and December."""
    out = []
    for y in range(start.year, end.year + 1):
        for mth in (3, 6, 9, 12):
            d = dt.date(y, mth, 15)
            d += dt.timedelta(days=(4 - d.weekday()) % 7)
            if start < d <= end:
                out.append(d)
    return out


def parity_forwards(meta, rows, fallback_rows=None, n_near=6):
    """Per expiry: T, r, D, forward F from parity, its dispersion, dividends implied.

    F = median over the n_near strikes with the smallest |C - P| of K + (C - P)/D, using
    mids. Near the money both options carry almost no early-exercise premium, so
    American parity is as close to European as it gets there. If fewer than 2 strikes
    have both a call and a put in rows, fallback_rows (two-sided quotes before the
    volume filter) are used. Returns a dict expiry -> slice info.
    """
    day, rate = curve_for(meta["asof"])
    S0, t0 = meta["spot"], meta["t0"]
    out = {}
    for exp in sorted({r["expiry"] for r in rows}):
        T = years_to(exp, t0)
        r = rate(T * 365)
        D = np.exp(-r * T)
        used = "filtered"
        pairs = _pairs(rows, exp)
        if len(pairs) < 2 and fallback_rows is not None:
            pairs, used = _pairs(fallback_rows, exp), "two-sided quotes incl. no-volume"
        if len(pairs) < 2:
            out[exp] = dict(T=T, r=r, D=D, F=None, why="fewer than 2 strikes with a call and a put")
            continue
        pairs.sort(key=lambda p: abs(p[1] - p[2]))
        near = pairs[:n_near]
        fK = np.array([K + (c - p) / D for K, c, p, _ in near])
        F = float(np.median(fK))
        noise = float(np.median([hs for *_, hs in near])) / D     # parity's own quote noise
        exd = third_fridays(meta["asof"], exp)
        dfs = [np.exp(-rate((d - meta["asof"]).days) * (d - meta["asof"]).days / 365) for d in exd]
        box = None
        if T > 30 / 365:                               # implied rate from a parity regression (box spread)
            wide = [(K, c - p) for K, c, p, _ in pairs if abs(K / F - 1) < 0.05]
            if len(wide) >= 5:
                slope, _ = np.polyfit([w[0] for w in wide], [w[1] for w in wide], 1)
                box = -np.log(-slope) / T if slope < 0 else None
        out[exp] = dict(T=T, r=r, D=D, F=F, S=D * F, n_pairs=len(near), source=used,
                        spread=float(fK.max() - fK.min()), noise=noise, ex_dates=exd, ex_dfs=dfs, box_rate=box)
    # The options' own spot: D F of the expiries with no ex-date before them (no dividends to strip),
    # weighted by how tightly parity pins each one. The quoted spot can be minutes away from the
    # option quotes; the dividends below are measured against this one.
    clean = [s for s in out.values() if s["F"] is not None and not s["ex_dates"]]
    if clean:
        w = np.array([1 / max(s["spread"], s["noise"], 0.01) for s in clean])
        x = np.array([s["S"] for s in clean])
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        S0 = float(x[o][np.searchsorted(cw, cw[-1] / 2)])
    meta["spot_implied"] = S0
    for s in out.values():
        if s["F"] is None:
            continue
        s["pv_div"] = S0 - s["S"]
        s["q"] = s["r"] - np.log(s["F"] / S0) / s["T"]
        s["div_each"] = s["pv_div"] / sum(s["ex_dfs"]) if s["ex_dates"] else None
    return out, day


def _pairs(rows, exp):
    c = {r["K"]: r for r in rows if r["expiry"] == exp and r["cp"] == "call"}
    p = {r["K"]: r for r in rows if r["expiry"] == exp and r["cp"] == "put"}
    mid = lambda q: 0.5 * (q["bid"] + q["ask"])
    return [(K, mid(c[K]), mid(p[K]), 0.25 * (c[K]["ask"] - c[K]["bid"] + p[K]["ask"] - p[K]["bid"]))
            for K in sorted(set(c) & set(p))]


def european_ivs(rows, fwd):
    """Black implied vol of each mid on its expiry's parity forward (NaN if outside European bounds)."""
    for r in rows:
        s = fwd.get(r["expiry"])
        r["mid"] = 0.5 * (r["bid"] + r["ask"])
        r["half"] = 0.5 * (r["ask"] - r["bid"])
        if s is None or s["F"] is None:
            r["iv"] = np.nan
            continue
        r["iv"] = implied_vol(r["mid"], s["S"], r["K"], s["T"], s["r"], r["cp"])
    return rows


def early_exercise(meta, rows, fwd, t_grid=200, x_grid=400):
    """Early-exercise premium of each contract: QuantLib finite differences, American minus
    European, same vol (the contract's European implied vol), same rate, and SPY's
    quarterly cash dividends on the third Fridays, sized so the model forward equals the
    parity forward. Expiries with no ex-date before them get the residual as a flat yield.
    Sets r['eep'] (dollars), r['iv_true'] (European vol of mid - eep: the vol the option
    would have if it were European) and r['eep_volpts'] = iv - iv_true in vol points, NaN
    when mid - eep falls outside the European bounds."""
    import QuantLib as ql
    asof = meta["asof"]
    today = ql.Date(asof.day, asof.month, asof.year)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    spot = ql.SimpleQuote(meta.get("spot_implied", meta["spot"]))
    vol = ql.SimpleQuote(0.2)
    vh = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(vol), dc))
    qd = lambda d: ql.Date(d.day, d.month, d.year)
    for exp, s in fwd.items():
        if s["F"] is None:
            continue
        days = (exp - asof).days
        Tq = days / 365.0
        rh = ql.YieldTermStructureHandle(ql.FlatForward(today, s["r"], dc, ql.Continuous))
        divs = s["ex_dates"] if s["div_each"] is not None and s["div_each"] > 0 else []
        q = 0.0 if divs else s["q"]           # no ex-date: the small residual carry as a flat yield
        qh = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc, ql.Continuous))
        proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot), qh, rh, vh)
        sched = ql.DividendVector([qd(d) for d in divs], [s["div_each"]] * len(divs))
        eng = ql.FdBlackScholesVanillaEngine(proc, sched, t_grid, x_grid)
        typ = {"call": ql.Option.Call, "put": ql.Option.Put}
        for r in rows:
            if r["expiry"] != exp or not np.isfinite(r["iv"]):
                continue
            pay = ql.PlainVanillaPayoff(typ[r["cp"]], r["K"])
            am = ql.VanillaOption(pay, ql.AmericanExercise(today, qd(exp)))
            eu = ql.VanillaOption(pay, ql.EuropeanExercise(qd(exp)))
            am.setPricingEngine(eng)
            eu.setPricingEngine(eng)
            sig = r["iv"]
            for _ in range(2):          # premium at the European vol of the mid, then once more at the corrected vol
                vol.setValue(sig * np.sqrt(s["T"] / Tq))     # same total variance on QuantLib's whole-day clock
                r["eep"] = max(am.NPV() - eu.NPV(), 0.0)
                iv_eu = implied_vol(r["mid"] - r["eep"], s["S"], r["K"], s["T"], s["r"], r["cp"])
                if not np.isfinite(iv_eu):
                    break
                sig = iv_eu
            r["iv_true"] = iv_eu
            r["eep_volpts"] = 100 * (r["iv"] - iv_eu)
    return rows
