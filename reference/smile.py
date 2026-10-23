"""
The smile: price with a fat-tailed kernel, read the prices back as Black-Scholes
implied volatilities.

A Black-Scholes trader quotes every option by the one volatility that makes the
Gaussian blur reproduce its price. If the true kernel has fatter tails than a
Gaussian, out-of-the-money options (which only pay in the tails) cost more than
the Gaussian with the at-the-money volatility says, so their implied volatility
is higher: the smile. Moneyness is k = ln(K / F), F = S0 e^{rT} the forward.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from kernels import grid_for, kernel, blur_price, bs, mean_sd


def implied_vol(price, S0, K, T, r, cp):
    """Black-Scholes implied volatility by bracketing (Brent). NaN if the price is
    outside the no-arbitrage bounds or too small to carry a volatility."""
    df = np.exp(-r * T)
    intrinsic = max(S0 - K * df, 0.0) if cp == "call" else max(K * df - S0, 0.0)
    if not (intrinsic + 1e-14 < price < (S0 if cp == "call" else K * df)):
        return np.nan
    f = lambda s: bs(S0, K, s, T, r, cp) - price
    lo, hi = 1e-4, 4.0
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)


def vega(S0, K, sigma, T, r):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)


def smile(model, ks, T, r, sigma, S0=100.0, **p):
    """Implied vols at log-moneyness ks for one model and maturity.

    One kernel per maturity, reused for every strike. Each strike is priced with
    its out-of-the-money option (put below the forward, call above): that is the
    option whose price is all time value, so the inversion is best conditioned.
    """
    y = grid_for(model, T, r, sigma, **p)
    w = kernel(model, y, T, r, sigma, **p)
    F = S0 * np.exp(r * T)
    out = []
    for k in ks:
        K = F * np.exp(k)
        cp = "call" if k >= 0 else "put"
        out.append(implied_vol(blur_price(model, S0, K, T, r, sigma, cp, y=y, w=w, **p),
                               S0, K, T, r, cp))
    return np.array(out)


def atm_shape(model, T, r, sigma, h=None, S0=100.0, **p):
    """ATM level, skew dIV/dk and curvature d2IV/dk2 at k = 0, by central
    differences of the implied vol at k = -2h..2h (5-point stencils).
    h defaults to a tenth of the kernel's standard deviation."""
    if h is None:
        h = 0.1 * mean_sd(model, T, r, sigma, **p)[1]
    ks = np.array([-2, -1, 0, 1, 2]) * h
    v = smile(model, ks, T, r, sigma, S0, **p)
    skew = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / (12 * h)
    curv = (-v[0] + 16 * v[1] - 30 * v[2] + 16 * v[3] - v[4]) / (12 * h * h)
    return v[2], skew, curv
