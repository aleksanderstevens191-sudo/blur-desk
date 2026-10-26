"""Part 4 Stage 3: pricing under local volatility with a blur whose width changes from spot to spot.

The state is X = ln(S_t / F_t), the log of spot over its forward. With deterministic rates and
dividends X is a martingale in price (E e^X = 1) and

    dX = -1/2 sigma(X, t)^2 dt + sigma(X, t) dW,

with sigma the local vol of Stage 2 read at k = X. Over one small step dt, X moves by a Gaussian
with mean -1/2 sigma^2 dt and variance sigma^2 dt, where sigma is taken at the spot the step
starts from (Euler; exact when sigma is constant). So a step back in time is a blur:

    V(x, t) = sum_j w_j(x) V(x + j dx, t + dt),   w_j(x) ~ exp(-(j dx - mu(x))^2 / (2 s(x)^2)),
    s(x) = sigma(x, t_mid) sqrt(dt),  mu(x) = -1/2 s(x)^2,

each output point with its own kernel width ("gather"), weights normalised to sum to 1. The
point-sampled Gaussian is the trapezoid rule for the conditional expectation; it converges
spectrally once the kernel spans a grid step or more (aliasing error ~ exp(-2 pi^2 s^2 / dx^2)),
so the grid is chosen with dx <= s everywhere. The payoff's kink is never sampled: the last step
before expiry is done in closed form (a Black formula over dt at the local vol of each node).

Two directions, the same matrices:
  backward  price one payoff: start from the payoff at T, gather back to t = 0, read x = 0.
  forward   price every strike of every expiry in one pass: start from a unit mass at x = 0 and
            scatter it forward (each source node spreads with its own width: the transpose of the
            gather), and at each expiry integrate the closed-form last step against the mass.
            Puts are integrated and calls come from parity (c = p + 1 - e^k): the pass keeps E e^X = 1
            exactly (see scatter), so this costs nothing and spares the calls the e^x weight far out.
The forward pass is the transpose of the backward one, so both give the same numbers up to
rounding (checked).

Edges: past the grid, a value is continued as A e^x + B through the last two nodes. Calls, puts,
and anything linear in S far out are exactly that, and the true expectation keeps e^x and 1 fixed.
The forward pass folds mass that leaves the grid back onto the edge nodes with the transpose of
that rule, so it keeps total mass and E e^X = 1 exactly.

Prices here are in forward units: a call struck at K = F_T e^k is worth D_T F_T c(k), c from
the pricer; vols come from the Black formula on the same forward.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import ndtr

TAPS_SD = 8.0          # kernel half-width in its own standard deviations


class LVTable:
    """Local vol on a (t, k) table: rows at times t inside the expiry intervals (edges), linear in k and,
    inside an interval, linear in t (flat out to the interval's ends). Never blended across an expiry:
    Dupire's dw/dT jumps there, and a row from a 3-day interval must not leak into a 78-day one.
    Flat in k outside the table. This is the model every pricer here uses."""

    def __init__(self, t, k, sig, edges=None):
        self.t, self.k, self.sig = np.asarray(t, float), np.asarray(k, float), np.asarray(sig, float)
        self.edges = np.asarray(edges if edges is not None else [0.0, np.inf], float)
        self.cell = np.searchsorted(self.edges, self.t, side="left")      # interval of each row

    def row(self, t):
        j = int(np.searchsorted(self.edges, t, side="left"))
        rows = np.nonzero(self.cell == j)[0]
        if rows.size == 0:                       # past the last edge (or before any row): nearest row
            rows = np.array([len(self.t) - 1 if t > self.t[-1] else 0])
        tt = self.t[rows]
        if t <= tt[0]:
            return self.sig[rows[0]]
        if t >= tt[-1]:
            return self.sig[rows[-1]]
        m = int(np.searchsorted(tt, t)) - 1
        a = (t - tt[m]) / (tt[m + 1] - tt[m])
        return (1 - a) * self.sig[rows[m]] + a * self.sig[rows[m + 1]]

    def __call__(self, x, t):
        return np.interp(x, self.k, self.row(t))

    @classmethod
    def from_surface(cls, S, dk=0.005, per=8, k_lo=-5.0, k_hi=3.5, fine=(-0.4, 0.3, 0.001)):
        """Tabulate Stage 2's regularized local vol: `per` times per expiry interval (cell centres); k every
        dk, and every fine[2] on [fine[0], fine[1]] where the shortest expiries live (a 3-day expiry's
        standard deviation is about 0.01 in k)."""
        from local_vol import cell_times
        k = np.union1d(np.round(np.arange(k_lo, k_hi + dk / 2, dk), 10),
                       np.round(np.arange(fine[0], fine[1] + fine[2] / 2, fine[2]), 10))
        t = cell_times(S, per=per)
        return cls(t, k, np.array([S.local_vol(k, ti)[0] for ti in t]), np.concatenate([[0.0], S.T]))


class Flat:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x, t):
        return np.full_like(np.asarray(x, float), self.sigma)


def grid(dx, x_lo, x_hi):
    """Nodes j dx (x = 0 is a node) covering [x_lo, x_hi]."""
    return dx * np.arange(int(np.floor(x_lo / dx)), int(np.ceil(x_hi / dx)) + 1)


def time_grid(T_points, dt, m=1):
    """Step boundaries from 0 through every time in T_points (each one lands on a boundary): steps <= dt,
    and at least m steps between consecutive points."""
    out = [0.0]
    for T in sorted(T_points):
        a = out[-1]
        n = max(m, int(np.ceil((T - a) / dt - 1e-9)))
        out += list(a + (T - a) * np.arange(1, n + 1) / n)
    return np.array(out)


def weights(x, sig, h, dx):
    """Gather weights: for node i a Gaussian of variance sig_i^2 h about -sig_i^2 h / 2, sampled on the grid
    and normalised. Rows are grouped by kernel half-width (powers of two, out to TAPS_SD standard deviations),
    so a 300% wing node does not make every 15% node pay for its taps. Returns (buckets, H_max), each bucket
    (row indices, weights (rows, 2 H + 1), H)."""
    s = sig * np.sqrt(h)
    mu = -0.5 * s * s
    need = np.ceil((TAPS_SD * s + np.abs(mu)) / dx).astype(int) + 1
    Hb = 2 ** np.ceil(np.log2(np.maximum(need, 2))).astype(int)
    buckets = []
    for H in np.unique(Hb):
        rows = np.nonzero(Hb == H)[0]
        off = np.arange(-H, H + 1) * dx
        W = np.exp(-((off[None, :] - mu[rows, None]) ** 2) / (2 * (s[rows] ** 2)[:, None]))
        W /= W.sum(axis=1, keepdims=True)
        buckets.append((rows, W, int(H)))
    return buckets, int(Hb.max())


def pad(V, x, H, dx):
    """V on the grid continued H nodes each side as A e^x + B through the two edge nodes."""
    def ext(x0, x1, v0, v1, xs):
        A = (v1 - v0) / (np.exp(x1) - np.exp(x0))
        return v0 + A * (np.exp(xs) - np.exp(x0))
    left = x[0] - dx * np.arange(H, 0, -1)
    right = x[-1] + dx * np.arange(1, H + 1)
    return np.concatenate([ext(x[0], x[1], V[0], V[1], left), V, ext(x[-2], x[-1], V[-2], V[-1], right)])


def gather(V, x, B, Hmax, dx):
    Vp = pad(V, x, Hmax, dx)
    out = np.empty_like(V)
    for rows, W, H in B:
        win = sliding_window_view(Vp, 2 * H + 1)[Hmax - H + rows]
        out[rows] = np.einsum("ij,ij->i", W, win)
    return out


def scatter(p, B, Hmax, x, dx):
    """Mass p at each node spread by its own row of weights: the exact transpose of gather(). Mass landing
    past an edge is folded back onto the two edge nodes with the transpose of pad()'s weights, which
    keeps both the total mass and the mean of e^x."""
    n = len(p)
    out = np.zeros(n + 2 * Hmax)
    for rows, W, H in B:
        base = Hmax - H + rows
        pr = p[rows]
        for j in range(2 * H + 1):
            out[base + j] += pr * W[:, j]
    core = out[Hmax:Hmax + n].copy()
    left = x[0] - dx * np.arange(Hmax, 0, -1)
    right = x[-1] + dx * np.arange(1, Hmax + 1)
    lam = (np.exp(left) - np.exp(x[0])) / (np.exp(x[1]) - np.exp(x[0]))
    core[0] += np.sum(out[:Hmax] * (1 - lam))
    core[1] += np.sum(out[:Hmax] * lam)
    lam = (np.exp(right) - np.exp(x[-2])) / (np.exp(x[-1]) - np.exp(x[-2]))
    core[-2] += np.sum(out[Hmax + n:] * (1 - lam))
    core[-1] += np.sum(out[Hmax + n:] * lam)
    return core


def last_step(x, sig, h, k, call):
    """Closed-form expectation over the final step of (e^X - e^k)^+ (call) or (e^k - e^X)^+ (put),
    X ~ x + N(-v/2, v), v = sig^2 h: Black's formula with forward e^x. Shape (len(x), len(k))."""
    v = np.sqrt(sig * sig * h)[:, None]
    ex, ek = np.exp(x)[:, None], np.exp(np.asarray(k, float))[None, :]
    d1 = (x[:, None] - np.asarray(k, float)[None, :] + 0.5 * v * v) / v
    c = ex * ndtr(d1) - ek * ndtr(d1 - v)
    return np.where(call, c, c - ex + ek)


def price_backward(sigma, k, T, call, dt, dx, x_lo, x_hi, payoff_step=None, barrier=None, m=1, knots=()):
    """One vanilla (or, with payoff_step/barrier, another payoff) by gathering back from T to 0.
    Returns the value at x = 0 in forward units."""
    x = grid(dx, x_lo, x_hi)
    tg = time_grid(sorted(set([t for t in knots if t < T] + [T])), dt, m)
    h = tg[-1] - tg[-2]
    sig = sigma(x, 0.5 * (tg[-1] + tg[-2]))
    V = last_step(x, sig, h, [k], call)[:, 0] if payoff_step is None else payoff_step(x, sig, h)
    if barrier is not None:
        V = barrier(x, V, tg[-2])
    for a, b in zip(tg[-3::-1], tg[-2:0:-1]):
        B, Hm = weights(x, sigma(x, 0.5 * (a + b)), b - a, dx)
        V = gather(V, x, B, Hm, dx)
        if barrier is not None:
            V = barrier(x, V, a)
    return float(V[np.argmin(np.abs(x))])


def prices_forward(sigma, expiries, strikes, calls, dt, dx, x_lo, x_hi, density_at=(), m=1, knots=()):
    """Every strike of every expiry in one forward pass. expiries: increasing times; strikes[i], calls[i]:
    arrays of k = ln(K/F) and call flags for expiry i. Returns a list of arrays (forward units).
    density_at: times (on step boundaries) at which to also return the mass on the grid. knots: extra step
    boundaries (the expiries where the local vol jumps, when pricing at other dates)."""
    x = grid(dx, x_lo, x_hi)
    p = np.zeros_like(x)
    p[np.argmin(np.abs(x))] = 1.0
    tg = time_grid(sorted(set(expiries) | {t for t in knots if t < max(expiries)}), dt, m)
    exp_idx = {int(np.argmin(np.abs(tg - T))): i for i, T in enumerate(expiries)}
    out = [None] * len(expiries)
    dens = {}
    for n in range(1, len(tg)):
        a, b = tg[n - 1], tg[n]
        sig = sigma(x, 0.5 * (a + b))
        if n in exp_idx:
            i = exp_idx[n]
            k = np.asarray(strikes[i], float)
            put = p @ last_step(x, sig, b - a, k, False)
            out[i] = np.where(calls[i], put + 1 - np.exp(k), put)      # calls by parity (E e^X = 1)
        if n == len(tg) - 1:
            break
        B, Hm = weights(x, sig, b - a, dx)
        p = scatter(p, B, Hm, x, dx)
        for td in density_at:
            if abs(b - td) < 1e-12:
                dens[td] = (x.copy(), p.copy())
    return (out, dens) if density_at else out


def black_c(k, w, call):
    """Black price in forward units of a call/put struck at e^k, total variance w."""
    v = np.sqrt(w)
    d1 = (-k + 0.5 * w) / v
    c = ndtr(d1) - np.exp(k) * ndtr(d1 - v)
    return np.where(call, c, c - 1 + np.exp(k))


def implied_w(c, k, call, w0=None):
    """Total implied variance from forward-unit prices (vectorised Newton on v = sqrt w, then bisection
    fallback), NaN where the price is outside the no-arbitrage bounds."""
    c, k, call = np.asarray(c, float), np.asarray(k, float), np.asarray(call, bool)
    intr = np.where(call, np.maximum(1 - np.exp(k), 0), np.maximum(np.exp(k) - 1, 0))
    hi = np.where(call, 1.0, np.exp(k))
    ok = (c > intr) & (c < hi)
    lo_v, hi_v = np.full(c.shape, 1e-8), np.full(c.shape, 10.0)
    for _ in range(200):                                        # bisection, robust
        mid = 0.5 * (lo_v + hi_v)
        f = black_c(k, mid * mid, call) - c
        lo_v = np.where(f < 0, mid, lo_v)
        hi_v = np.where(f < 0, hi_v, mid)
    v = 0.5 * (lo_v + hi_v)
    for _ in range(3):                                          # polish with Newton
        d1 = (-k + 0.5 * v * v) / v
        vega = np.exp(k) * np.exp(-0.5 * (d1 - v) ** 2) / np.sqrt(2 * np.pi)
        v = v - np.where(vega > 1e-300, (black_c(k, v * v, call) - c) / np.maximum(vega, 1e-300), 0)
    return np.where(ok, v * v, np.nan)
