"""Part 4 Stage 4: one barrier pricer, three models that all claim to fit SPY's vanillas.

The state is X = ln(S_t / F_t) as in lv_blur.py, priced backward from the payoff by blurring step
by step. A continuously monitored barrier H on the spot is the moving line b(t) = ln(H / F_t) in X.
Each step's blur weights carry the Brownian-bridge survival factor

    w_ij -> w_ij (1 - exp(-2 d_i d_j / s_i^2)),   d = distance to the barrier on the live side,

the probability that a path from node i to node j does not touch the (straight) barrier line in
between, given the step's Gaussian with standard deviation s_i. For constant vol over the step this
is exact, so the barrier needs no grid alignment and no Broadie-Glasserman-Kou shift. Far from the
barrier the factor is 1 and the step is the plain blur of lv_blur.py. The last step before expiry
is the closed-form Black value minus the bridge-crossing part (only near the barrier), so the
strike's kink is never sampled. Knock-ins are vanilla minus knock-out on the same grid.

Models (all on the same grid and time steps):
  bs      Black-Scholes at one vol: Flat(sigma).
  lv      local vol: the LVTable of Stage 3.
  merton  diffusion sigma plus Poisson(lam) jumps N(m, delta^2), split per step: the diffusion
          blur (drift -lam kappa dt keeps E e^X = 1), then the jump blur (a fixed kernel, the
          Poisson mixture of jump densities). A jump that lands past the barrier knocks out.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve

from lv_blur import weights, pad, time_grid, grid, Flat, last_step

BRIDGE_SD = 12.0       # rows nearer the barrier than this many step standard deviations get the bridge factor


class Forward:
    """F(t): ln F linear in t through (0, S0) and the expiries' forwards; flat in growth past the last."""

    def __init__(self, S0, T, F):
        self.t = np.concatenate([[0.0], np.asarray(T, float)])
        self.lf = np.log(np.concatenate([[S0], np.asarray(F, float)]))

    def __call__(self, t):
        if t > self.t[-1]:
            g = (self.lf[-1] - self.lf[-2]) / (self.t[-1] - self.t[-2])
            return float(np.exp(self.lf[-1] + g * (t - self.t[-1])))
        return float(np.exp(np.interp(t, self.t, self.lf)))


class Merton:
    """Diffusion vol sigma plus jumps: the step's diffusion is Flat(sigma); jumps are applied separately."""

    def __init__(self, sigma, lam, m, delta):
        self.sigma, self.lam, self.m, self.delta = sigma, lam, m, delta
        self.kappa = np.expm1(m + 0.5 * delta * delta)

    def __call__(self, x, t):
        return np.full_like(np.asarray(x, float), self.sigma)

    def jump_kernel(self, h, dx, tail=1e-15):
        """Offsets (nodes) and weights of the jump part over a step h: Poisson(lam h) mixture of N(n m, n delta^2),
        each point-sampled and normalised to its Poisson weight; n = 0 is the identity."""
        L = self.lam * h
        p, n, ps = [np.exp(-L)], 0, []
        while True:
            n += 1
            p.append(p[-1] * L / n)
            if p[-1] < tail and n > L:
                break
        nmax = n
        J = int(np.ceil((abs(self.m) * nmax + 8 * self.delta * np.sqrt(nmax)) / dx)) + 1
        off = np.arange(-J, J + 1) * dx
        K = np.zeros(2 * J + 1)
        K[J] = p[0]
        for n in range(1, nmax + 1):
            g = np.exp(-((off - n * self.m) ** 2) / (2 * n * self.delta ** 2))
            K += p[n] * g / g.sum()
        return K, J

    def vanilla(self, k, T, call):
        """Forward-unit vanilla: given n jumps X = g_n + Y, Y ~ N(-v_n/2, v_n), g_n = n (m + delta^2/2) - lam kappa T."""
        from lv_blur import black_c
        k, call = np.asarray(k, float), np.asarray(call, bool)
        L = self.lam * T
        out, p, n = np.zeros(np.broadcast(k, call).shape), np.exp(-L), 0
        while True:
            g = n * (self.m + 0.5 * self.delta ** 2) - self.lam * self.kappa * T
            out = out + p * np.exp(g) * black_c(k - g, self.sigma ** 2 * T + n * self.delta ** 2, call)
            n += 1
            p *= L / n
            if p < 1e-18 and n > L:
                return out


def _bridge_rows(x, rows, H, dx, s, d_start, side, b_end, Vp_col, Hmax, W, f_end=None):
    """For rows near the barrier: sum_j W_ij surv_ij V_j (or, with f_end, sum_j W_ij (1 - surv_ij) f(x_j))."""
    off = np.arange(-H, H + 1) * dx
    xj = x[rows][:, None] + off[None, :]
    dj = (xj - b_end) if side == "down" else (b_end - xj)
    di = d_start[rows][:, None]
    surv = np.where((dj > 0) & (di > 0), -np.expm1(-2 * di * np.maximum(dj, 0) / (s[rows] ** 2)[:, None]), 0.0)
    if f_end is not None:
        return np.einsum("ij,ij->i", W * (1 - surv), f_end(xj))
    win = sliding_window_view(Vp_col, 2 * H + 1)[Hmax - H + rows]
    return np.einsum("ij,ij->i", W * surv, win)


def price(model, T, payoffs, fwd, dt, dx, x_lo, x_hi, barrier=None, m=1, knots=()):
    """Backward blur from T to 0 for a list of payoffs [(k, call)] in forward units (k = ln(K / F_T)).
    barrier: None, or (H, "down"/"up") on the spot, monitored continuously. Returns forward-unit prices at
    x = 0 (multiply by D_T F_T for dollars)."""
    x = grid(dx, x_lo, x_hi)
    tg = time_grid(sorted(set([t for t in knots if t < T] + [T])), dt, m)
    nc = len(payoffs)
    jump = isinstance(model, Merton)
    drift_of = (lambda h: -model.lam * model.kappa * h) if jump else (lambda h: 0.0)

    def dist(t):
        if barrier is None:
            return np.full_like(x, np.inf)
        b = np.log(barrier[0] / fwd(t))
        return (x - b) if barrier[1] == "down" else (b - x), b

    def d_of(t):
        return dist(t)[0] if barrier is not None else np.full_like(x, np.inf)

    def b_of(t):
        return dist(t)[1] if barrier is not None else None

    # last step: closed form, minus the bridge-crossing part near the barrier
    a, b = tg[-2], tg[-1]
    h = b - a
    sig = model(x, 0.5 * (a + b))
    mu = drift_of(h)
    ks = np.array([p[0] for p in payoffs])
    calls = np.array([p[1] for p in payoffs])
    V = last_step(x + mu, sig, h, ks, calls)               # (n, nc): Black with forward e^(x + mu)
    if barrier is not None:
        da = d_of(a)
        s = sig * np.sqrt(h)
        alive = da > 0
        V[~alive] = 0.0
        B, Hm = weights(x, sig, h, dx, drift=mu)
        for rows, W, Hb in B:
            near = rows[alive[rows] & (da[rows] < BRIDGE_SD * s[rows])]
            if near.size == 0:
                continue
            Wn = W[np.searchsorted(rows, near)]
            for c in range(nc):
                f = (lambda xs, kk=ks[c], cc=calls[c]: np.maximum(np.exp(xs) - np.exp(kk), 0) if cc
                     else np.maximum(np.exp(kk) - np.exp(xs), 0))
                V[near, c] -= _bridge_rows(x, near, Hb, dx, s, da, barrier[1], b_of(b), None, Hm, Wn, f_end=f)
    if jump:
        V = _jump(V, x, model, h, dx, d_of(a) if barrier is not None else None)

    for a, b in zip(tg[-3::-1], tg[-2:0:-1]):
        h = b - a
        sig = model(x, 0.5 * (a + b))
        mu = drift_of(h)
        B, Hm = weights(x, sig, h, dx, drift=mu)
        da = d_of(a)
        alive = da > 0
        s = sig * np.sqrt(h)
        out = np.zeros_like(V)
        for c in range(nc):
            Vp = pad(V[:, c], x, Hm, dx)
            for rows, W, Hb in B:
                keep = alive[rows]
                if barrier is not None and not keep.all():
                    rows, W = rows[keep], W[keep]
                if rows.size == 0:
                    continue
                win = sliding_window_view(Vp, 2 * Hb + 1)[Hm - Hb + rows]
                out[rows, c] = np.einsum("ij,ij->i", W, win)
                if barrier is not None:
                    nr = da[rows] < BRIDGE_SD * s[rows]
                    if nr.any():
                        out[rows[nr], c] = _bridge_rows(x, rows[nr], Hb, dx, s, da, barrier[1], b_of(b), Vp, Hm, W[nr])
        V = out
        if jump:
            V = _jump(V, x, model, h, dx, da if barrier is not None else None)
    return V[np.argmin(np.abs(x))]


def _jump(V, x, model, h, dx, d_alive):
    """Backward jump step: V(x) <- E V(x + J); nodes already past the barrier stay dead."""
    K, J = model.jump_kernel(h, dx)
    out = np.empty_like(V)
    for c in range(V.shape[1]):
        Vp = pad(V[:, c], x, J, dx)
        out[:, c] = fftconvolve(Vp, K[::-1], mode="valid")
    if d_alive is not None:
        out[d_alive <= 0] = 0.0
    return out
