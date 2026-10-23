"""
Fat-tailed blur kernels (float64 reference).

Black-Scholes prices an option by blurring its payoff in log-price with a
Gaussian. Nothing in the blur cares that the kernel is Gaussian: any density
for the log-return X = ln(S_T / S_0) works, and the price is

    V(x) = e^{-rT} * sum_j w_j * f(x + y_j)

with f the payoff in log-price and w_j the kernel sampled on offsets y_j = j dx.
This file builds three kernels:

  gaussian   X ~ N(mu, s^2): Black-Scholes.
  merton     Merton (1976) jump-diffusion: a Gaussian diffusion plus a Poisson
             number of Gaussian jumps in log-price. Given n jumps X is Gaussian,
             so the kernel is a Poisson-weighted mix of Gaussians.
  vg         Variance gamma (Madan, Carr, Chang 1998): Brownian motion with
             drift theta, run on a gamma-distributed clock G (mean T, variance
             nu T). Given G = g, X is Gaussian, so the kernel is a gamma-weighted
             mix of Gaussians.

Risk neutrality. Each model gets a drift correction so E[e^X] = e^{rT} exactly:
  gaussian  mu = (r - sigma^2/2) T
  merton    mu = (r - sigma^2/2 - lam*kappa) T,  kappa = E[e^J] - 1 = e^{m + delta^2/2} - 1
  vg        mu = (r + omega) T,  omega = ln(1 - theta*nu - sigma^2 nu / 2) / nu

Sampling. Instead of sampling the density at nodes, each kernel weight is an
exact integral of the density against a "tent" built from the functions 1 and
e^y, which are exactly the shapes a call or put payoff takes in log-price away
from the strike. Two consequences, both by construction:
  - sum_j w_j = P(X on the grid)                       (integrates to 1)
  - sum_j w_j e^{y_j} = E[e^X; X on the grid] = e^{rT}  (the forward is exact)
and the blur of a call or put is exact everywhere except in the one grid cell
that holds the strike, which gets a small kernel-independent correction on the
payoff samples (kink_correction). The tents only need two integrals of the
density per cell, M0 = int p and M1 = int p e^y, which are differences of the
two "cumulative" functions F0(e) = P(X <= e) and F1(e) = E[e^X; X <= e].
For a Gaussian both are closed forms; mixtures are sums or integrals of those.
"""
import numpy as np
from scipy.special import gammainc, gammainccinv, gammaln, kv, ndtr
from scipy.stats import norm, poisson

MODELS = ("gaussian", "merton", "vg")


# ---------- model parameters, drift corrections, moments ----------

def drift(model, T, r, sigma, lam=0.0, m=0.0, delta=0.0, theta=0.0, nu=0.0):
    """Mean of the diffusion part of X (the risk-neutral drift correction)."""
    if model == "gaussian":
        return (r - 0.5 * sigma**2) * T
    if model == "merton":
        kappa = np.expm1(m + 0.5 * delta**2)
        return (r - 0.5 * sigma**2 - lam * kappa) * T
    if model == "vg":
        omega = np.log1p(-theta * nu - 0.5 * sigma**2 * nu) / nu
        return (r + omega) * T
    raise ValueError(model)


def mean_sd(model, T, r, sigma, lam=0.0, m=0.0, delta=0.0, theta=0.0, nu=0.0):
    """Mean and standard deviation of X (for sizing the grid)."""
    mu = drift(model, T, r, sigma, lam, m, delta, theta, nu)
    if model == "gaussian":
        return mu, sigma * np.sqrt(T)
    if model == "merton":
        return mu + lam * T * m, np.sqrt(sigma**2 * T + lam * T * (m * m + delta * delta))
    return mu + theta * T, np.sqrt((sigma**2 + theta**2 * nu) * T)


def char_fn(model, u, T, r, sigma, lam=0.0, m=0.0, delta=0.0, theta=0.0, nu=0.0):
    """E[exp(i u X)] for complex u (used by the Carr-Madan FFT check)."""
    u = np.asarray(u, dtype=complex)
    mu = drift(model, T, r, sigma, lam, m, delta, theta, nu)
    if model == "gaussian":
        return np.exp(1j * u * mu - 0.5 * sigma**2 * T * u * u)
    if model == "merton":
        jump = np.exp(1j * u * m - 0.5 * delta**2 * u * u) - 1
        return np.exp(1j * u * mu - 0.5 * sigma**2 * T * u * u + lam * T * jump)
    return np.exp(1j * u * mu) * (1 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u * u) ** (-T / nu)


def density(model, y, T, r, sigma, lam=0.0, m=0.0, delta=0.0, theta=0.0, nu=0.0):
    """Pointwise density of X (closed forms; for plots and the integral checks)."""
    y = np.asarray(y, dtype=float)
    mu = drift(model, T, r, sigma, lam, m, delta, theta, nu)
    if model == "gaussian":
        return norm.pdf(y, mu, sigma * np.sqrt(T))
    if model == "merton":
        out = np.zeros_like(y)
        for n, pn in _poisson_terms(lam * T):
            out += pn * norm.pdf(y, mu + n * m, np.sqrt(sigma**2 * T + n * delta**2))
        return out
    # variance gamma: Madan-Carr-Chang (1998) eq. 23, written in logs
    a = T / nu
    z = y - mu
    c = np.sqrt(2 * sigma**2 / nu + theta**2)
    az = np.maximum(np.abs(z), 1e-300)
    logp = (np.log(2) + theta * z / sigma**2 - a * np.log(nu) - 0.5 * np.log(2 * np.pi)
            - np.log(sigma) - gammaln(a) + (a - 0.5) * (np.log(az) - np.log(c)))
    return np.exp(logp) * kv(a - 0.5, az * c / sigma**2)


def _poisson_terms(L, tail=1e-18):
    """(n, P(N = n)) for N ~ Poisson(L), until the remaining mass is below tail."""
    if L == 0:
        return [(0, 1.0)]
    out, n, total = [], 0, 0.0
    while True:
        pn = float(poisson.pmf(n, L))
        out.append((n, pn))
        total += pn
        if n > L and (1.0 - total < tail or pn < tail * 1e-3):
            return out
        n += 1


# ---------- the cumulative functions F0, F1 at cell edges ----------

def _gauss_F(e, mean, sd):
    """F0 = P(X <= e), F1 = E[e^X; X <= e] for X ~ N(mean, sd^2)."""
    return ndtr((e - mean) / sd), np.exp(mean + 0.5 * sd * sd) * ndtr((e - mean - sd * sd) / sd)


def cumulative(model, e, T, r, sigma, lam=0.0, m=0.0, delta=0.0, theta=0.0, nu=0.0,
               g_tiny=None):
    e = np.asarray(e, dtype=float)
    mu = drift(model, T, r, sigma, lam, m, delta, theta, nu)
    if model == "gaussian":
        return _gauss_F(e, mu, sigma * np.sqrt(T))
    if model == "merton":
        F0, F1 = np.zeros_like(e), np.zeros_like(e)
        for n, pn in _poisson_terms(lam * T):
            a0, a1 = _gauss_F(e, mu + n * m, np.sqrt(sigma**2 * T + n * delta**2))
            F0 += pn * a0
            F1 += pn * a1
        return F0, F1
    # variance gamma: given G = g, X ~ N(mu + theta g, sigma^2 g), G ~ Gamma(a = T/nu, scale nu).
    # F0 mixes Phi over G. For F1, tilting by e^X keeps G gamma with the same shape and
    # scale nu' = nu / (1 - (theta + sigma^2/2) nu), and moves the Gaussian mean by
    # sigma^2 g, so F1 = E[e^X] * E'[Phi((e - mu - (theta + sigma^2) g) / (sigma sqrt g))].
    a = T / nu
    if g_tiny is None:  # Gaussian width a thousandth of a grid cell
        g_tiny = (1e-3 * (e[1] - e[0]) / sigma) ** 2
    tilt = 1 - (theta + 0.5 * sigma**2) * nu
    total = np.exp(mu) * tilt ** (-a)  # E[e^X], = e^{rT} when the drift is right
    F0 = _vg_mix(e - mu, a, nu, theta, sigma, g_tiny)
    F1 = total * _vg_mix(e - mu, a, nu / tilt, theta + sigma**2, sigma, g_tiny)
    return F0, F1


_GL16 = np.polynomial.legendre.leggauss(16)


def _vg_mix(d, a, scale, b, sigma, g_tiny, panel=0.25, tail=1e-18):
    """E[Phi((d - b G) / (sigma sqrt G))] for G ~ Gamma(a, scale), for each d.

    Integrated in log g: as a function of log g each Phi is a sigmoid of width
    about 1 wherever it sits, so fixed Gauss-Legendre panels resolve every edge
    alike. Below g_tiny the Gaussian is far narrower than a grid cell and Phi is a
    step at d = 0; that mass, P(G < g_tiny), is added as a step.
    """
    lo = np.log(g_tiny)
    hi = np.log(scale * gammainccinv(a, tail))
    n = max(1, int(np.ceil((hi - lo) / panel)))
    edges = np.linspace(lo, hi, n + 1)
    x, wt = _GL16
    h = 0.5 * (edges[1] - edges[0])
    L = (edges[:-1, None] + h * (x[None, :] + 1)).ravel()
    W = np.tile(h * wt, n)
    g = np.exp(L)
    W = W * np.exp(a * L - g / scale - a * np.log(scale) - gammaln(a))  # gamma pdf * g
    sg = sigma * np.sqrt(g)
    out = np.empty_like(d)
    for i in range(0, len(d), 256):
        di = d[i:i + 256, None]
        out[i:i + 256] = ndtr((di - b * g[None, :]) / sg[None, :]) @ W
    return out + gammainc(a, g_tiny / scale) * (d > 0)


# ---------- the kernel ----------

def half_width(model, T, r, sigma, lam=0.0, m=0.0, delta=0.0, theta=0.0, nu=0.0,
               n_sd=13.0, tail=1e-18):
    """How far from 0 the grid must reach so the mass left off it (also weighted
    by e^y, for the forward) is below ~tail."""
    mu = drift(model, T, r, sigma, lam, m, delta, theta, nu)
    if model == "gaussian":
        return abs(mu) + sigma**2 * T + n_sd * sigma * np.sqrt(T)
    if model == "merton":
        # every Poisson term that matters, at n_sd of its own width
        return max(abs(mu + n * m) + sigma**2 * T + n * delta**2
                   + n_sd * np.sqrt(sigma**2 * T + n * delta**2)
                   for n, pn in _poisson_terms(lam * T) if pn > tail)
    # variance gamma: the tails are exponential with rates (c -/+ theta)/sigma^2,
    # independent of T, so short maturities need room for the tails, not just the sd.
    # The tilted measure behind the forward has the right-hand rate lowered by 1.
    c = np.sqrt(2 * sigma**2 / nu + theta**2)
    rate = min((c + theta) / sigma**2, (c - theta) / sigma**2 - 1)
    _, sd = mean_sd(model, T, r, sigma, lam, m, delta, theta, nu)
    return abs(mu) + max(n_sd * sd, -np.log(tail) / rate)


def grid_for(model, T, r, sigma, lam=0.0, m=0.0, delta=0.0, theta=0.0, nu=0.0,
             dx=None, pts_per_sd=200, dx_max=0.002):
    """Offsets y_j = j dx, symmetric about 0, wide enough to hold the density.

    dx resolves the narrowest piece of the kernel: the diffusion alone (sigma sqrt T)
    for Merton, which has no jumps with probability e^{-lam T}.
    """
    p = dict(lam=lam, m=m, delta=delta, theta=theta, nu=nu)
    if dx is None:
        _, sd = mean_sd(model, T, r, sigma, **p)
        if model == "merton":
            sd = sigma * np.sqrt(T)
        dx = min(sd / pts_per_sd, dx_max)
    J = int(np.ceil(half_width(model, T, r, sigma, **p) / dx))
    return np.arange(-J, J + 1) * dx


def kernel(model, y, T, r, sigma, **p):
    """Tent weights w_j on the offsets y (uniform spacing).

    Between nodes y_j and y_{j+1}, the tents are the two combinations of 1 and e^y
    that equal 1 at one node and 0 at the other:
        L_left(y)  = (e^{y_{j+1}} - e^y) / (e^{y_{j+1}} - e^{y_j})
        L_right(y) = (e^y - e^{y_j})     / (e^{y_{j+1}} - e^{y_j})
    and w_j = int p(y) [tent of node j](y) dy, from the cell integrals M0, M1.
    """
    F0, F1 = cumulative(model, y, T, r, sigma, **p)
    M0, M1 = np.diff(F0), np.diff(F1)       # per cell [y_j, y_{j+1}]
    eL, eR = np.exp(y[:-1]), np.exp(y[1:])
    den = eR - eL
    left = (eR * M0 - M1) / den               # weight going to node j
    right = (M1 - eL * M0) / den              # weight going to node j + 1
    w = np.zeros_like(y)
    w[:-1] += left
    w[1:] += right
    return w


# ---------- payoffs and the blur ----------

def payoff(z, K, cp):
    S = np.exp(z)
    return np.maximum(S - K, 0.0) if cp == "call" else np.maximum(K - S, 0.0)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)


def _gl(fn, a, b):
    if b <= a:
        return 0.0
    x = 0.5 * (b - a) * _GL_X + 0.5 * (a + b)
    return 0.5 * (b - a) * float(_GL_W @ fn(x))


def kink_correction(z, K, cp):
    """Payoff samples, plus a correction on the two nodes around the strike.

    Off the strike cell the tents reproduce the payoff exactly. In the strike cell
    the tent interpolant If misses the kink by err(z) = f(z) - If(z). Adding d_j and
    d_{j+1} to the samples with
        d_j + d_{j+1} = A / dx,  d_{j+1} - d_j = 2 B / dx^2,
        A = int err,  B = int (z - z_mid) err   (over the cell)
    matches the missing integral against any kernel that is smooth across the cell,
    to O(dx^4). The correction does not depend on the kernel, so one corrected
    payoff row serves every model (and the GPU).
    """
    dx = z[1] - z[0]
    f = payoff(z, K, cp)
    k = np.log(K)
    j = int(np.floor((k - z[0]) / dx))
    if j < 0 or j + 1 >= len(z):
        return f
    zl, zr = z[j], z[j + 1]
    el, er = np.exp(zl), np.exp(zr)

    def err(t):
        tent = f[j] * (er - np.exp(t)) / (er - el) + f[j + 1] * (np.exp(t) - el) / (er - el)
        return payoff(t, K, cp) - tent

    zm = 0.5 * (zl + zr)
    A = _gl(err, zl, k) + _gl(err, k, zr)
    B = _gl(lambda t: (t - zm) * err(t), zl, k) + _gl(lambda t: (t - zm) * err(t), k, zr)
    s, d = A / dx, 2 * B / dx**2
    out = f.copy()
    out[j] += 0.5 * (s - d)
    out[j + 1] += 0.5 * (s + d)
    return out


def blur_price(model, S0, K, T, r, sigma, cp="call", y=None, w=None, **p):
    """European option by one blur of the payoff with the model's kernel, at S0."""
    if y is None:
        y = grid_for(model, T, r, sigma, **p)
    if w is None:
        w = kernel(model, y, T, r, sigma, **p)
    z = np.log(S0) + y
    return np.exp(-r * T) * float(w @ kink_correction(z, K, cp))


# ---------- independent prices ----------

def bs(S, K, sigma, T, r, cp="call"):
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    if cp == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def merton_series(S, K, T, r, sigma, lam, m, delta, cp="call", tail=1e-18):
    """Merton (1976): a Poisson-weighted sum of Black-Scholes prices.

    With n jumps the log-return is Gaussian with variance sigma^2 T + n delta^2,
    priced by BS with sigma_n^2 = sigma^2 + n delta^2 / T and rate
    r_n = r - lam kappa + n (m + delta^2/2) / T, under Poisson(lam (1 + kappa) T).
    """
    kappa = np.expm1(m + 0.5 * delta**2)
    lp = lam * (1 + kappa)
    total = 0.0
    for n, pn in _poisson_terms(lp * T, tail):
        sn = np.sqrt(sigma**2 + n * delta**2 / T)
        rn = r - lam * kappa + n * (m + 0.5 * delta**2) / T
        total += pn * bs(S, K, sn, T, rn, cp)
    return total


def vg_mixture(S, K, T, r, sigma, theta, nu, cp="call"):
    """Variance gamma as Madan-Carr-Chang price it: given the gamma clock G = g the
    log-return is Gaussian, so the price is a Black-Scholes-type formula averaged
    over G. Integrated adaptively in log g with scipy's quad: no grid, no kernel."""
    from scipy.integrate import quad
    a = T / nu
    mu = drift("vg", T, r, sigma, theta=theta, nu=nu)

    def f(L):
        g = np.exp(L)
        s = sigma * np.sqrt(g)
        F = S * np.exp(mu + theta * g + 0.5 * s * s)
        d1 = (np.log(F / K) + 0.5 * s * s) / s
        d2 = d1 - s
        v = F * ndtr(d1) - K * ndtr(d2) if cp == "call" else K * ndtr(-d2) - F * ndtr(-d1)
        return v * np.exp(a * L - g / nu - a * np.log(nu) - gammaln(a))

    hi = np.log(nu * gammainccinv(a, 1e-20))
    lo = min(-60.0, hi - 60.0)
    cuts = np.unique(np.clip([lo, -30, -15, -8, -4, -2, 0, 2, hi], lo, hi))
    v = sum(quad(f, u, w, epsabs=1e-15, epsrel=1e-13, limit=400)[0] for u, w in zip(cuts[:-1], cuts[1:]))
    # G below e^lo: the Gaussian is a point, the payoff is taken at S e^mu
    v += gammainc(a, np.exp(lo) / nu) * float(payoff(np.log(S) + mu, K, cp))
    return np.exp(-r * T) * v


def carr_madan(model, S0, K, T, r, sigma, cp="call", N=2**15, eta=0.04, **p):
    """Carr-Madan (1999) FFT price at strike K, from the characteristic function alone.

    Calls are damped with e^{alpha k}, alpha > 0; puts with alpha < 0, so the put is
    priced on its own, not through parity. The log-strike grid is centred on ln K so
    no interpolation is needed. Simpson weights on the frequency grid.
    """
    sign = 1.0 if cp == "call" else -1.0
    alpha = sign * _safe_alpha(model, sign, T, r, sigma, **p)
    lam_k = 2 * np.pi / (N * eta)
    v = np.arange(N) * eta
    k0 = np.log(K) - (N // 2) * lam_k
    phi = char_fn(model, v - (alpha + 1) * 1j, T, r, sigma, **p) * S0 ** (1j * (v - (alpha + 1) * 1j))
    psi = np.exp(-r * T) * phi / (alpha * alpha + alpha - v * v + 1j * (2 * alpha + 1) * v)
    simpson = (3 + (-1.0) ** (np.arange(N) + 1)) / 3
    simpson[0] = 1 / 3
    x = np.exp(-1j * v * k0) * psi * eta * simpson
    y = np.fft.fft(x)
    k = k0 + (N // 2) * lam_k
    return float(np.real(np.exp(-alpha * k) / np.pi * y[N // 2]))


def _safe_alpha(model, sign, T, r, sigma, **p):
    """Damping |alpha| = 1.5, reduced if E[S_T^(1+alpha)] would not exist (VG only)."""
    a = 1.5
    if model == "vg":
        th, nu = p["theta"], p["nu"]
        # need 1 - th nu c - sigma^2 nu c^2 / 2 > 0 at c = 1 + alpha
        while 1 - th * nu * (1 + sign * a) - 0.5 * sigma**2 * nu * (1 + sign * a) ** 2 <= 0.05:
            a *= 0.8
    return a
