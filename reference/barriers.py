"""
Single-barrier options as blurs (float64 reference).

A knock-out option is a vanilla payoff that dies if the price ever touches the
barrier H. In log-price x = ln S the stock is a Brownian motion with drift
nu = r - q - sigma^2/2, and the classic trick for "killed" Brownian motion is
the method of images: put a negative copy of the payoff in the mirror on the
far side of the barrier, so the two cancel exactly on the barrier line.

  (a) plain mirror:   g(z) = F(z) - F(2h - z)
  (b) weighted mirror: g(z) = F(z) - (H/e^z)^(2 nu / sigma^2) F(2h - z)
  (c) step by step:    blur one monitoring interval, zero past the barrier, repeat

where F is the payoff on the live side of the barrier (zero on the other side)
and h = ln H. One blur of g gives the knock-out price. The mirror copy in (a)
only cancels when nu = 0, i.e. r - q = sigma^2/2: with drift, paths drift away
from the mirror at a different rate than toward it, and the weight in (b) is
exactly the likelihood ratio (Girsanov) that corrects for that. With
2 nu / sigma^2 = 2(r - q)/sigma^2 - 1 it is the textbook (H/S)^(2r/sigma^2 - 1).

Sampling. The blur is a sum over grid nodes. Smooth parts of a payoff are
summed to near machine precision, so all the error comes from kinks (the
strike) and jumps (the barrier). Each breakpoint gets an Euler-Maclaurin
correction to its neighbouring samples, which makes those O(dx^2) errors
O(dx^4); see sample_payoff. Barriers sit exactly on grid nodes (the grid is
anchored at h); strikes can fall anywhere.
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm

from blur_pricer import gaussian_kernel

# Broadie-Glasserman-Kou constant: -zeta(1/2) / sqrt(2 pi)
BGK_BETA = 0.5825971579390106

TYPES = [
    "down-and-out call", "down-and-in call", "up-and-out call", "up-and-in call",
    "down-and-out put", "down-and-in put", "up-and-out put", "up-and-in put",
]


def parse(kind):
    words = kind.replace("-", " ").split()
    return words[0], words[2], words[3]  # "down"/"up", "in"/"out", "call"/"put"


# ---------- closed forms ----------

def vanilla(S, K, sigma, T, r, q=0.0, cp="call"):
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    if cp == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def barrier_closed_form(kind, S, K, H, sigma, T, r, q=0.0):
    """Reiner-Rubinstein (1991) continuous-barrier prices, no rebate.

    Knock-ins are computed from their own formulas, not as vanilla minus
    knock-out, so in + out = vanilla is a real check of the algebra.
    """
    ud, io, cp = parse(kind)
    if (ud == "down" and S <= H) or (ud == "up" and S >= H):
        return vanilla(S, K, sigma, T, r, q, cp) if io == "in" else 0.0
    phi = 1.0 if cp == "call" else -1.0
    eta = 1.0 if ud == "down" else -1.0
    b = r - q
    sT = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    fwd = S * np.exp((b - r) * T)
    dK = K * np.exp(-r * T)
    x1 = np.log(S / K) / sT + (1 + mu) * sT
    x2 = np.log(S / H) / sT + (1 + mu) * sT
    y1 = np.log(H * H / (S * K)) / sT + (1 + mu) * sT
    y2 = np.log(H / S) / sT + (1 + mu) * sT
    A = phi * fwd * norm.cdf(phi * x1) - phi * dK * norm.cdf(phi * x1 - phi * sT)
    B = phi * fwd * norm.cdf(phi * x2) - phi * dK * norm.cdf(phi * x2 - phi * sT)
    p1, p0 = (H / S) ** (2 * (mu + 1)), (H / S) ** (2 * mu)
    C = phi * fwd * p1 * norm.cdf(eta * y1) - phi * dK * p0 * norm.cdf(eta * y1 - eta * sT)
    D = phi * fwd * p1 * norm.cdf(eta * y2) - phi * dK * p0 * norm.cdf(eta * y2 - eta * sT)
    above = K > H
    table = {
        ("down", "in", "call"): C if above else A - B + D,
        ("up", "in", "call"): A if above else B - C + D,
        ("down", "in", "put"): B - C + D if above else A,
        ("up", "in", "put"): A - B + D if above else C,
        ("down", "out", "call"): A - C if above else B - D,
        ("up", "out", "call"): 0.0 if above else A - B + C - D,
        ("down", "out", "put"): A - B + C - D if above else 0.0,
        ("up", "out", "put"): B - D if above else A - C,
    }
    return table[(ud, io, cp)]


# ---------- grids and sampling ----------

def anchored_grid(h, x0, half_width, dx):
    """Log grid with the barrier h exactly on a node, covering x0 +/- half_width."""
    lo = int(np.floor((x0 - half_width - h) / dx))
    hi = int(np.ceil((x0 + half_width - h) / dx))
    return h + np.arange(lo, hi + 1) * dx


def _one_sided(g, b, side, eps):
    """Limit value and slope of g at b from one side (quadratic extrapolation)."""
    g1, g2, g3 = (float(g(np.array([b + side * eps * t]))[0]) for t in (1, 2, 3))
    value = 3 * g1 - 3 * g2 + g3
    slope = side * (-2.5 * g1 + 4 * g2 - 1.5 * g3) / eps
    return value, slope


def sample_payoff(g, z, kinks=(), jumps=()):
    """Sample g on the uniform grid z, with Euler-Maclaurin breakpoint corrections.

    jumps: breakpoints where g may jump; each must be a grid node. The node gets
      the midpoint of the two sides plus dx/12 * (slope jump), and its two
      neighbours get +/- (value jump)/24. Together these cancel the dx^2 term
      of the sum's error against the integral, for any blur kernel.
    kinks: breakpoints where g is continuous but its slope jumps; anywhere.
      A kink a fraction theta past node j costs dx^2 B2(theta)/2 * (slope jump)
      (B2 the 2nd Bernoulli polynomial), which is put back on nodes j and j+1.
    """
    dx = z[1] - z[0]
    v = g(z).astype(float)
    pts = sorted(set(kinks) | set(jumps))
    gap = min(np.diff(pts)) if len(pts) > 1 else 1.0
    eps = min(dx / 10, gap / 4)
    for b in jumps:
        i = int(round((b - z[0]) / dx))
        assert abs(z[i] - b) < 1e-9 * max(1.0, abs(b)), "jump must sit on a grid node"
        gl, sl = _one_sided(g, b, -1, eps)
        gr, sr = _one_sided(g, b, +1, eps)
        v[i] = 0.5 * (gl + gr) + dx / 12 * (sr - sl)
        v[i + 1] += (gr - gl) / 24
        v[i - 1] -= (gr - gl) / 24
    for b in kinks:
        if b in jumps:
            continue
        j = int(np.floor((b - z[0]) / dx))
        theta = (b - z[j]) / dx
        _, sl = _one_sided(g, b, -1, eps)
        _, sr = _one_sided(g, b, +1, eps)
        d = dx * (theta * theta - theta + 1 / 6) / 2 * (sr - sl)
        v[j] += (1 - theta) * d
        v[j + 1] += theta * d
    return v


def blur_rows(V, kernel, direct=True):
    """Blur each row of V (same padding as blur_pricer.blur: odd reflection)."""
    V = np.atleast_2d(V)
    pad = len(kernel) // 2
    if pad >= V.shape[1]:
        raise ValueError("kernel wider than grid; widen the grid")
    P = np.pad(V, ((0, 0), (pad, pad)), mode="reflect", reflect_type="odd")
    if direct:
        return np.array([np.convolve(p, kernel[::-1], mode="valid") for p in P])
    return fftconvolve(P, kernel[::-1][None, :], mode="valid", axes=1)


def interp_cubic(z, V, x):
    """4-point Lagrange interpolation of V(z) at x (V smooth near x)."""
    dx = z[1] - z[0]
    j = int(np.floor((x - z[0]) / dx)) - 1
    t = (x - z[j]) / dx  # in [1, 2)
    w = np.array([-(t - 1) * (t - 2) * (t - 3) / 6, t * (t - 2) * (t - 3) / 2,
                  -t * (t - 1) * (t - 3) / 2, t * (t - 1) * (t - 2) / 6])
    return float(w @ V[..., j:j + 4].T) if V.ndim == 1 else V[:, j:j + 4] @ w


def payoff_fn(K, cp):
    if cp == "call":
        return lambda z: np.maximum(np.exp(z) - K, 0.0)
    return lambda z: np.maximum(K - np.exp(z), 0.0)


def grid_width(sigma, T):
    return 10 * sigma * np.sqrt(T) + 0.5


# ---------- (a) and (b): one blur with a mirror image ----------

def mirror_payoff(kind, K, H, sigma, r, q=0.0, weighted=True):
    """The function g(z) whose single blur is the barrier price, plus its breakpoints."""
    ud, io, cp = parse(kind)
    h, k = np.log(H), np.log(K)
    f = payoff_fn(K, cp)
    a = 2 * (r - q - 0.5 * sigma**2) / sigma**2 if weighted else 0.0
    live = (lambda z: z > h) if ud == "down" else (lambda z: z < h)

    def g(z):
        z = np.asarray(z, dtype=float)
        w = np.exp(-a * (z - h))  # (H / e^z)^a
        image = w * f(2 * h - z)
        # knock-out: payoff on the live side, minus its weighted image on the dead side
        # knock-in:  payoff on the dead side, plus the same image (vanilla - knock-out)
        if io == "out":
            return np.where(live(z), f(z), -image)
        return np.where(live(z), 0.0, f(z) + image)

    return g, [k, 2 * h - k], [h]


def mirror_price(kind, S0, K, H, sigma, T, r, q=0.0, dx=0.0005, weighted=True):
    """Price by one blur of the mirrored payoff; returns the value at S0."""
    h, x0 = np.log(H), np.log(S0)
    z = anchored_grid(h, x0, grid_width(sigma, T), dx)
    g, kinks, jumps = mirror_payoff(kind, K, H, sigma, r, q, weighted)
    v = sample_payoff(g, z, kinks, jumps)
    ker = gaussian_kernel(dx, sigma * np.sqrt(T), center=(r - q - 0.5 * sigma**2) * T)
    V = np.exp(-r * T) * blur_rows(v, ker)[0]
    return interp_cubic(z, V, x0)


def vanilla_blur(S0, K, sigma, T, r, q=0.0, cp="call", dx=0.0005, anchor=None):
    """The same sampler on a plain payoff: the baseline for the barrier blurs.
    anchor: put this price on a node (pass H to use the barrier blur's grid)."""
    x0 = np.log(S0)
    z = anchored_grid(np.log(anchor) if anchor else x0, x0, grid_width(sigma, T), dx)
    v = sample_payoff(payoff_fn(K, cp), z, kinks=[np.log(K)])
    ker = gaussian_kernel(dx, sigma * np.sqrt(T), center=(r - q - 0.5 * sigma**2) * T)
    V = np.exp(-r * T) * blur_rows(v, ker)[0]
    return interp_cubic(z, V, x0)


# ---------- (c): step by step, zeroing past the barrier ----------

def _monitor(A, B, z, ib, ud):
    """Where the barrier is breached take B, elsewhere A. A and B are smooth
    rows across the barrier node ib; the jump between them gets the same
    midpoint + Euler-Maclaurin treatment as sample_payoff."""
    dx = z[1] - z[0]
    dead = z < z[ib] if ud == "down" else z > z[ib]
    out = np.where(dead[None, :], B, A)
    # right side minus left side, value and slope
    D = (A - B) if ud == "down" else (B - A)
    dv = D[:, ib]
    ds = (D[:, ib + 1] - D[:, ib - 1]) / (2 * dx)
    out[:, ib] = 0.5 * (A[:, ib] + B[:, ib]) + dx / 12 * ds
    out[:, ib + 1] += dv / 24
    out[:, ib - 1] -= dv / 24
    return out


def stepping_prices(kinds, S0, K, H, sigma, T, r, steps, q=0.0, shift=False,
                    dx_max=0.0005, info=None):
    """Discretely monitored barriers (monitoring at T/steps, 2T/steps, ..., T).

    Blur one interval, discount, take the knocked value past the barrier, repeat.
    With shift=True the barrier is moved toward the spot by BGK_BETA sigma sqrt(dt)
    (Broadie-Glasserman-Kou), so the discrete grid approximates the continuous price
    at the original H. The grid spacing is chosen so the shifted barrier is a node.
    Returns {kind: price at S0} for every kind given (all must share the same
    up/down direction and call/put). If info is a dict it receives the grid.
    """
    ud = {parse(k)[0] for k in kinds}
    cp = {parse(k)[2] for k in kinds}
    assert len(ud) == len(cp) == 1
    ud, cp = ud.pop(), cp.pop()
    dt = T / steps
    beta = BGK_BETA * sigma * np.sqrt(dt)
    m = int(np.ceil(beta / dx_max))
    dx = beta / m
    h = np.log(H)
    if shift:
        h = h + beta if ud == "down" else h - beta
    x0 = np.log(S0)
    z = anchored_grid(h, x0, grid_width(sigma, T), dx)
    ib = int(round((h - z[0]) / dx))
    if info is not None:
        info.update(anchor=float(h), lo=int(round((z[0] - h) / dx)), n=len(z), dx=float(dx), ib=ib)
    f = payoff_fn(K, cp)
    van = sample_payoff(f, z, kinks=[np.log(K)])[None, :]
    # rows: vanilla, knock-out, knock-in
    zero = np.zeros_like(van)
    ko = _monitor(van, zero, z, ib, ud)
    ki = _monitor(zero, van, z, ib, ud)
    V = np.vstack([van, ko, ki])
    ker = gaussian_kernel(dx, sigma * np.sqrt(dt), center=(r - q - 0.5 * sigma**2) * dt)
    disc = np.exp(-r * dt)
    for n in range(steps):
        V = disc * blur_rows(V, ker, direct=False)
        if n < steps - 1:  # no monitoring at t = 0
            van, ko, ki = V[0:1], V[1:2], V[2:3]
            V = np.vstack([van, _monitor(ko, zero, z, ib, ud), _monitor(ki, van, z, ib, ud)])
    p = interp_cubic(z, V, x0)
    out = {}
    for kind in kinds:
        out[kind] = p[1] if parse(kind)[1] == "out" else p[2]
    out["vanilla"] = p[0]
    return out
