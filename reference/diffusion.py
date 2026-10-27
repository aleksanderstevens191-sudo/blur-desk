"""
Option pricing and diffusion models: one heat equation, two directions (float64 reference).

Coordinates. Everything lives in log-return y = ln(S_T / S), with S the slice's
spot net of dividends (spy_fit.json), the coordinate the pricing blur already uses.

Pricing blurs a payoff. Black-Scholes in log-price is
    dX = mu dt + sigma dW,   mu = r - sigma^2/2,
and the price of a payoff f(y) expiring in tau is
    V(x) = e^{-r tau} E[f(x + Y)],   Y ~ N(mu tau, sigma^2 tau),
which on a grid is V_i = e^{-r tau} sum_j w_j f_{i+j}: the payoff, blurred (a
correlation with the kernel). V solves the backward Kolmogorov equation, run from
expiry back to today:
    -dV/dt = mu V_x + (sigma^2/2) V_xx - r V.

A diffusion model blurs a density. The variance-exploding (VE) SDE of Song et al.
(2021, eq. 9) has no drift:
    dx = sqrt(d[sigma^2(t)]/dt) dw,   sigma(t) = sigma_min (sigma_max/sigma_min)^t,
so x(t) = x(0) + N(0, sigma^2(t) - sigma^2(0)) and the density at t is the data
density convolved with that Gaussian. On a grid, q_k = sum_j w_j p_{k-j}: the
density, blurred (a convolution). The density solves the forward (Fokker-Planck)
equation, run from the data toward noise:
    dp/dt = (1/2) g^2(t) p_yy.

Same heat equation, opposite directions. Remove the drift (move the frame by mu t)
and the discount (multiply by e^{r tau}) and both are w_s = (1/2) w_zz in their own
clock s (sigma^2 tau for the payoff, sigma^2(t) - sigma^2(0) for the density). The
payoff runs it backward in calendar time (expiry -> today), the density forward
(data -> noise). Both run the heat equation forward in its own clock: both blur.
Neither is the generative step of a diffusion model, which un-blurs (Stage 2).

The two operators are adjoint: the payoff blur is the transpose of the density
blur, so <p, blurred payoff> = <blurred p, payoff> (the duality). With no drift the
Gaussian kernel is symmetric and the two operators are the same matrix.

DDPM (Ho et al. 2020) adds a shrink at each step:
    x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) eps,
so x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, abar_t = prod (1 - beta_s); in
continuous time the VP SDE dx = -beta(t)/2 x dt + sqrt(beta(t)) dw (Song et al.
2021, eq. 11), an Ornstein-Uhlenbeck process. Divided by sqrt(abar_t) it is the VE
process with sigma^2 = (1 - abar_t)/abar_t (Song et al. 2021, App. B; Karras et al.
2022, Table 1). diffusion_stage1.py, section 5, compares it to drift and discounting.

The market's density is the market odds panel's (panel 13, src/odds.js): the SPY
Merton fit (spy_fit.json), read off its blurred call prices by Breeden-Litzenberger.
Panel 13 recovers Merton's closed form to 1e-5; here the closed form is used.
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import ndtr
from scipy.stats import poisson

HERE = Path(__file__).parent
SPY_FIT = json.loads((HERE.parent / "src" / "spy_fit.json").read_text())


# ---------- the market's density (panel 13) ----------

def market(e=1, fit=SPY_FIT):
    """The SPY slice e (0 Nov 20, 1 Dec 18) with its Merton parameters."""
    sl, mp = fit["slices"][e], fit["merton"]
    return dict(expiry=sl["expiry"], T=sl["T"], r=sl["r"], S=sl["S"], F=sl["F"], spot=fit["spot"],
                sigma=mp["sigma"], lam=mp["lam"], m=mp["m"], delta=mp["delta"], flat=fit["flat"]["sigma"])


def mixture(P, a=1.0, extra=0.0, tail=1e-18):
    """Merton's log-return as a Poisson mix of Gaussians: (weight, mean, var) per
    jump count. a scales the variable (DDPM's shrink), extra adds Gaussian variance
    (the blur). With a = 1, extra = 0 it is the market's density."""
    kappa = np.expm1(P["m"] + 0.5 * P["delta"] ** 2)
    mu = (P["r"] - 0.5 * P["sigma"] ** 2 - P["lam"] * kappa) * P["T"]
    L = P["lam"] * P["T"]
    out, n, tot = [], 0, 0.0
    while True:
        pn = float(poisson.pmf(n, L))
        out.append((pn, a * (mu + n * P["m"]), a * a * (P["sigma"] ** 2 * P["T"] + n * P["delta"] ** 2) + extra))
        tot += pn
        if n > L and (1 - tot < tail or pn < tail * 1e-3):
            return out
        n += 1


def mix_pdf(y, comps):
    y = np.asarray(y, float)
    return sum(w * np.exp(-0.5 * (y - m) ** 2 / v) / np.sqrt(2 * np.pi * v) for w, m, v in comps)


def mix_cdf(y, comps):
    y = np.asarray(y, float)
    return sum(w * ndtr((y - m) / np.sqrt(v)) for w, m, v in comps)


def mix_score(y, comps):
    """d/dy log p for a Gaussian mixture, in closed form (Stage 2's reference)."""
    y = np.asarray(y, float)
    num = sum(w * np.exp(-0.5 * (y - m) ** 2 / v) / np.sqrt(2 * np.pi * v) * (m - y) / v for w, m, v in comps)
    return num / mix_pdf(y, comps)


def mix_sample(comps, n, rng):
    w = np.array([c[0] for c in comps])
    k = rng.choice(len(comps), size=n, p=w / w.sum())
    m = np.array([c[1] for c in comps])[k]
    s = np.sqrt(np.array([c[2] for c in comps]))[k]
    return m + s * rng.standard_normal(n)


# ---------- the two blurs on a grid ----------

def gauss_taps(dx, s, center=0.0, n_sd=9.0):
    """Gaussian N(center, s^2) sampled at offsets j dx (trapezoid weights dx * pdf).
    Returns the weights on offsets -h..h. Their sum is 1 to exp(-2 pi^2 s^2/dx^2)."""
    h = int(np.ceil((n_sd * s + abs(center)) / dx))
    o = np.arange(-h, h + 1) * dx
    return dx * np.exp(-0.5 * ((o - center) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def blur_density(p, w):
    """q_k = sum_j w_j p_{k-j}: the density moves by +offset (convolution)."""
    h = len(w) // 2
    full = fftconvolve(p, w) if len(w) > 64 else np.convolve(p, w)
    return full[h:h + len(p)]


def blur_payoff(f, w):
    """V_i = sum_j w_j f_{i+j}: the payoff is read at +offset (correlation)."""
    h = len(w) // 2
    full = fftconvolve(f, w[::-1]) if len(w) > 64 else np.convolve(f, w[::-1])
    return full[h:h + len(f)]


# ---------- the VE forward process (Song et al. 2021) ----------

def ve_sigma(t, smin, smax):
    return smin * (smax / smin) ** np.asarray(t, float)


def ve_chain(p0, dx, smin, smax, N, keep=()):
    """Run the VE forward process on a grid density: N steps of the SMLD/VE chain
    x_i = x_{i-1} + sqrt(sigma_i^2 - sigma_{i-1}^2) z (Song et al. 2021, eq. 8),
    each one a Gaussian blur of the density. keep: step numbers to return."""
    sig = ve_sigma(np.arange(N + 1) / N, smin, smax)
    p, out = p0.copy(), {}
    for i in range(1, N + 1):
        p = blur_density(p, gauss_taps(dx, np.sqrt(sig[i] ** 2 - sig[i - 1] ** 2)))
        if i in keep:
            out[i] = p.copy()
    return p, out, sig


# ---------- DDPM (Ho et al. 2020) ----------

def ddpm_betas(N=1000, b1=1e-4, bN=0.02):
    return np.linspace(b1, bN, N)


def ddpm_particles(x0, betas, rng, keep=()):
    x, out = x0.copy(), {}
    for i, b in enumerate(betas, 1):
        x = np.sqrt(1 - b) * x + np.sqrt(b) * rng.standard_normal(len(x))
        if i in keep:
            out[i] = x.copy()
    return x, out


# ---------- Stage 2: the reverse run (probability-flow ODE, exact score on the grid) ----------

class GridScore:
    """The score d/dy log p_sigma(y) of the market's density blurred to noise level
    sigma, from the grid alone: blur the grid density (the VE forward process,
    variance sigma^2 - sigma_min^2), take the log, central differences. No network.
    score(x, sigma) interpolates it at any points x."""

    def __init__(self, y, p0, smin):
        self.y, self.p0, self.smin = y, p0, smin
        self.dx = y[1] - y[0]
        self.cache = {}

    def table(self, sigma):
        key = float(sigma)
        if key not in self.cache:
            v = sigma * sigma - self.smin * self.smin
            p = self.p0 if v <= 0 else blur_density(self.p0, gauss_taps(self.dx, np.sqrt(v)))
            self.cache[key] = np.gradient(np.log(np.maximum(p, 1e-300)), self.dx)
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def __call__(self, x, sigma):
        return np.interp(x, self.y, self.table(sigma))


def pf_ode(x1, score, smin, smax, N):
    """Probability-flow ODE of the VE SDE (Song et al. 2021, eq. 13),
        dx/dt = -g(t)^2 / 2 * score(x, t),   g^2 = d sigma^2 / dt,
    run backward from sigma_max to sigma_min. In s = ln sigma it reads
    dx/ds = -sigma^2 score(x, sigma); classical RK4, N steps uniform in s."""
    s = np.linspace(np.log(smax), np.log(smin), N + 1)
    x = np.array(x1, float)
    f = lambda x_, s_: -np.exp(2 * s_) * score(x_, np.exp(s_))
    for i in range(N):
        h = s[i + 1] - s[i]
        k1 = f(x, s[i])
        k2 = f(x + 0.5 * h * k1, s[i] + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, s[i] + 0.5 * h)
        k4 = f(x + h * k3, s[i + 1])
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def mix_ppf(u, comps, lo=-8.0, hi=8.0, iters=80):
    """Inverse CDF of a Gaussian mixture by bisection (vectorized)."""
    u = np.asarray(u, float)
    a, b = np.full_like(u, lo), np.full_like(u, hi)
    for _ in range(iters):
        c = 0.5 * (a + b)
        below = mix_cdf(c, comps) < u
        a, b = np.where(below, c, a), np.where(below, b, c)
    return 0.5 * (a + b)


def tweedie_exact(x, comps, s2):
    """E[x_0 | x_t = x] when x_t = x_0 + N(0, s2) and x_0 is the Gaussian mixture
    comps: each component's posterior mean, weighted by its posterior weight."""
    x = np.asarray(x, float)
    num, den = 0.0, 0.0
    for w, m, v in comps:
        pk = w * np.exp(-0.5 * (x - m) ** 2 / (v + s2)) / np.sqrt(2 * np.pi * (v + s2))
        num = num + pk * (m + v / (v + s2) * (x - m))
        den = den + pk
    return num / den


# ---------- Stage 4: panel 16's algorithm, as the GPU runs it (src/diffusion.js, src/gpu-diffusion.js) ----------
# The reverse run with fewer levels than Stage 2, so a browser can precompute every score table:
#   grid      y_i = i dx, |i| <= half (dx 0.002, |y| <= 6), the market's density p0 and its central
#             difference Dp0 (0 at the two end nodes)
#   levels    sigma_j = sigma_min (sigma_max/sigma_min)^(j/N), j = 0..N; blur sd s_j = sqrt(sigma_j^2 - sigma_min^2)
#   blur      q = sum_k w_k p0_(i-k) / sum_k w_k, w_k = exp(-(k dx)^2 / 2 s^2), |k| <= ceil(9 s / dx), zero
#             outside the grid; q' the same blur of Dp0 (a derivative commutes with a blur)
#   score     q'/q (0 where q <= 1e-30): every term of q is positive, so 32-bit q keeps its relative
#             precision out in the tails, which log-then-difference would not
#   sampler   Heun's method in ln sigma, level N down to 0, score interpolated linearly (clamped at the ends)
#   payoff    the same Gaussian (sd s, centre -s^2/2: Black-Scholes at r = 0, on the forward) on its own taps
#             k s/256, |k| <= 9 * 256, against the call payoff max(100 e^(x+y) - K, 0), weights normalised
PANEL = dict(dx=0.002, half=3000, N=128, smin=0.01, smax=1.0, nsd=9.0, per=256, e=1)


def panel_grid(P, C=PANEL):
    y = np.arange(-C["half"], C["half"] + 1) * C["dx"]
    p0 = mix_pdf(y, mixture(P))
    dp = np.zeros_like(p0)
    dp[1:-1] = (p0[2:] - p0[:-2]) / (2 * C["dx"])
    return y, p0, dp


def panel_levels(C=PANEL):
    sig = C["smin"] * (C["smax"] / C["smin"]) ** (np.arange(C["N"] + 1) / C["N"])
    s = np.sqrt(np.maximum(sig**2 - C["smin"] ** 2, 0.0))
    s[0] = 0.0
    return sig, s


def panel_blur(p, s, C=PANEL):
    if s == 0:
        return p.copy()
    H = int(np.ceil(C["nsd"] * s / C["dx"]))
    k = np.arange(-H, H + 1) * C["dx"]
    w = np.exp(-k * k / (2 * s * s))
    return fftconvolve(p, w / w.sum())[H:H + len(p)]


def panel_tables(p0, dp, s_list, C=PANEL):
    """q and the score at every level (rows)."""
    Q, SC = [], []
    for s in s_list:
        q, qd = panel_blur(p0, s, C), panel_blur(dp, s, C)
        Q.append(q)
        SC.append(np.where(q > 1e-30, qd / np.maximum(q, 1e-300), 0.0))
    return np.array(Q), np.array(SC)


def panel_sampler(x, y, SC, sig, keep=()):
    """Heun's method on dx/d(ln sigma) = -sigma^2 score, from level N down to 0."""
    N = len(sig) - 1
    h = (np.log(sig[0]) - np.log(sig[-1])) / N
    x = np.array(x, float)
    out = {N: x.copy()}
    for j in range(N, 0, -1):
        k1 = -sig[j] ** 2 * np.interp(x, y, SC[j])
        k2 = -sig[j - 1] ** 2 * np.interp(x + h * k1, y, SC[j - 1])
        x = x + 0.5 * h * (k1 + k2)
        if j - 1 in keep:
            out[j - 1] = x.copy()
    out[0] = x
    return out


def panel_payoff(x, s, K, C=PANEL):
    """Undiscounted call (r = 0, per $100 of forward) at log-forward offsets x, blurred by sd s."""
    x = np.asarray(x, float)
    if s == 0:
        return np.maximum(100 * np.exp(x) - K, 0.0)
    H = int(C["nsd"] * C["per"])
    k = np.arange(-H, H + 1) * (s / C["per"])
    w = np.exp(-k * k / (2 * s * s))
    w /= w.sum()
    return np.maximum(100 * np.exp(x[:, None] - 0.5 * s * s + k[None, :]) - K, 0.0) @ w
