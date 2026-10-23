"""
Calibrate Merton's jump-diffusion to an option chain by least squares on implied vol.

Input: a CSV snapshot of one underlying's chain, one row per contract, columns
    expiry (YYYY-MM-DD), strike, type (call/put), bid, ask
plus a header comment line "# spot=<S0> asof=<YYYY-MM-DD>" (see load_chain).

Per expiry, the forward F comes from the market itself: put-call parity
C - P = D (F - K) is a straight line in K. With no rate given, D and F are both
fitted by least squares over strikes near the money; with a Treasury rate
given (the right choice for American options like SPY's, see prepare), D is
e^{-rT} and only F is read from parity. Either way dividends never need to be
guessed. The kernels then price with spot
S0' = D F (the spot net of dividends to expiry) and rate r = -ln(D) / T, which
reproduces the market forward: S0' e^{rT} = F.

Each strike is represented by its out-of-the-money option (puts below F, calls
above), mid price, inverted to a Black-Scholes vol. Fit error is reported in
vol points (1 point = 1% of vol) as RMS and max over all quotes.
"""
import csv
import datetime as dt
import numpy as np
from scipy.optimize import least_squares

from kernels import merton_series, blur_price, grid_for, kernel
from smile import implied_vol


def load_chain(path):
    """Read the CSV; returns spot, as-of date, and a list of dict rows."""
    spot = asof = None
    rows = []
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    for ln in lines:
        if ln.startswith("#"):
            for tok in ln[1:].split():
                k, _, v = tok.partition("=")
                if k == "spot":
                    spot = float(v)
                if k == "asof":
                    asof = dt.date.fromisoformat(v)
    for r in csv.DictReader(ln for ln in lines if not ln.startswith("#")):
        rows.append(dict(expiry=dt.date.fromisoformat(r["expiry"]), K=float(r["strike"]),
                         cp=r["type"].strip().lower(), bid=float(r["bid"]), ask=float(r["ask"])))
    return spot, asof, rows


def arbitrage_violations(calls, puts):
    """Static-arbitrage checks at executable prices (buy at the ask, sell at the bid),
    valid for American as well as European options. Returns a list of strings.

      vertical: a call (put) spread must cost between 0 and the strike gap
      butterfly: price must be convex in strike (adjacent triples, uneven gaps)
    """
    out = []
    for name, book, sign in (("call", calls, -1), ("put", puts, +1)):
        Ks = sorted(book)
        for k1, k2 in zip(Ks, Ks[1:]):
            lo, hi = book[k1], book[k2]
            # sign -1: calls fall with strike, so C(k1) >= C(k2); puts rise
            cheap, dear = (hi, lo) if sign < 0 else (lo, hi)
            if cheap["bid"] > dear["ask"] + 1e-9:
                out.append(f"{name} {k1:g}/{k2:g} spread below zero")
            if dear["bid"] - cheap["ask"] > (k2 - k1) + 1e-9:
                out.append(f"{name} {k1:g}/{k2:g} spread {dear['bid'] - cheap['ask']:.2f} > gap {k2 - k1:g}")
        for k1, k2, k3 in zip(Ks, Ks[1:], Ks[2:]):
            w = (k3 - k2) / (k3 - k1)
            wings = w * book[k1]["ask"] + (1 - w) * book[k3]["ask"]
            if wings < book[k2]["bid"] - 1e-9:
                out.append(f"{name} butterfly {k1:g}/{k2:g}/{k3:g} at {wings - book[k2]['bid']:+.2f}")
    return out


def prepare(spot, asof, rows, min_days=5, max_abs_k=0.35, min_bid=0.05, rates=None,
            screen=True, dropped=None):
    """Group by expiry, find forward and discount, keep OTM mids, invert.

    rates: None to fit the discount factor D from put-call parity along with F
      (European options), or a function expiry -> continuously compounded rate.
      SPY options are American: deep in-the-money puts carry an early-exercise
      premium that tilts C - P and biases a fitted D (it can come out above 1).
      With a rate given, D = e^{-rT} and F comes from parity at the 4 strikes
      nearest the money only, where that premium is negligible.
    screen: drop an expiry whose quotes contain a static arbitrage.
    dropped: optional list that receives (expiry, reason) for skipped expiries.
    Returns a list of slices: dict(T, D, F, S, r, K[], cp[], mid[], iv[], ...).
    """
    def skip(exp, why):
        if dropped is not None:
            dropped.append((exp, why))

    out = []
    for exp in sorted({r["expiry"] for r in rows}):
        T = (exp - asof).days / 365.0
        if T * 365 < min_days:
            skip(exp, f"{T * 365:.0f} days to expiry (< {min_days})")
            continue
        q = [r for r in rows if r["expiry"] == exp and r["bid"] >= min_bid and r["ask"] > r["bid"]]
        calls = {r["K"]: r for r in q if r["cp"] == "call"}
        puts = {r["K"]: r for r in q if r["cp"] == "put"}
        if screen:
            bad = arbitrage_violations(calls, puts)
            if bad:
                skip(exp, f"{len(bad)} static arbitrages, e.g. " + "; ".join(bad[:3]))
                continue
        mid = lambda row: 0.5 * (row["bid"] + row["ask"])
        both = sorted(set(calls) & set(puts))
        if len(both) < 4:
            skip(exp, "fewer than 4 strikes with both a call and a put")
            continue
        if rates is None:
            near = sorted(both, key=lambda K: abs(K - spot))[:10]
            Ks = np.array(near)
            cmp = np.array([mid(calls[K]) - mid(puts[K]) for K in near])
            slope, icpt = np.polyfit(Ks, cmp, 1)       # C - P = D F - D K
            D = -slope
            F = icpt / D
            r = -np.log(D) / T
        else:
            r = rates(exp)
            D = np.exp(-r * T)
            near = sorted(both, key=lambda K: abs(mid(calls[K]) - mid(puts[K])))[:4]
            F = float(np.median([K + (mid(calls[K]) - mid(puts[K])) / D for K in near]))
        S = D * F                                      # spot net of dividends to expiry
        sl = dict(T=T, expiry=exp, D=D, F=F, S=S, r=r, K=[], cp=[], mid=[], iv=[], iv_bid=[], iv_ask=[])
        for K in sorted(set(calls) | set(puts)):
            cp = "call" if K >= F else "put"
            row = (calls if cp == "call" else puts).get(K)
            if row is None or abs(np.log(K / F)) > max_abs_k:
                continue
            iv = implied_vol(mid(row), S, K, T, r, cp)
            if not np.isfinite(iv):
                continue
            sl["K"].append(K)
            sl["cp"].append(cp)
            sl["mid"].append(mid(row))
            sl["iv"].append(iv)
            sl["iv_bid"].append(implied_vol(row["bid"], S, K, T, r, cp))
            sl["iv_ask"].append(implied_vol(row["ask"], S, K, T, r, cp))
        for k in ("K", "mid", "iv", "iv_bid", "iv_ask"):
            sl[k] = np.array(sl[k])
        if len(sl["K"]):
            out.append(sl)
    return out


def merton_ivs(slices, sigma, lam, m, delta, pricer="series"):
    """Model implied vols at every quote. 'series' for speed while fitting;
    'blur' prices the final fit with the Merton kernel itself."""
    out = []
    for sl in slices:
        S, T, r = sl["S"], sl["T"], sl["r"]
        if pricer == "blur":
            p = dict(lam=lam, m=m, delta=delta)
            y = grid_for("merton", T, r, sigma, **p)
            w = kernel("merton", y, T, r, sigma, **p)
        ivs = []
        for K, cp in zip(sl["K"], sl["cp"]):
            if pricer == "blur":
                price = blur_price("merton", S, K, T, r, sigma, cp, y=y, w=w, **p)
            else:
                price = merton_series(S, K, T, r, sigma, lam, m, delta, cp=cp)
            ivs.append(implied_vol(price, S, K, T, r, cp))
        out.append(np.array(ivs))
    return out


def _resid(model_ivs, slices):
    r = np.concatenate([mi - sl["iv"] for mi, sl in zip(model_ivs, slices)])
    return np.where(np.isfinite(r), r, 0.05)  # a quote the model cannot reach costs 5 vol points


def fit_flat(slices):
    """Flat Black-Scholes: one volatility for every strike and expiry."""
    allv = np.concatenate([sl["iv"] for sl in slices])
    return float(np.mean(allv)), [np.full(len(sl["K"]), np.mean(allv)) for sl in slices]


def fit_merton(slices, x0=(0.12, 1.0, -0.08, 0.10)):
    """Least squares over (sigma, lam, m, delta) on implied-vol residuals."""
    lo, hi = [0.01, 0.0, -0.6, 0.005], [1.0, 20.0, 0.3, 0.8]
    best = None
    for start in [x0, (0.10, 0.3, -0.15, 0.10), (0.15, 3.0, -0.03, 0.05)]:
        res = least_squares(lambda x: _resid(merton_ivs(slices, *x), slices), start,
                            bounds=(lo, hi), x_scale=[0.05, 1.0, 0.05, 0.05], xtol=1e-10, ftol=1e-12)
        if best is None or res.cost < best.cost:
            best = res
    return best.x


def errors(model_ivs, slices):
    r = np.concatenate([mi - sl["iv"] for mi, sl in zip(model_ivs, slices)])
    r = r[np.isfinite(r)]
    return np.sqrt(np.mean(r * r)) * 100, np.max(np.abs(r)) * 100  # vol points
