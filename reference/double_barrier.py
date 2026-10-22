"""
Double knock-out: two barriers, a hall of mirrors (float64 reference).

The option pays f(S_T) only if the price stays strictly between L and U the
whole time. With one barrier, one weighted mirror image cancels the payoff on
the barrier line. With two barriers, the image in the lower mirror spoils the
upper barrier, so it needs its own image in the upper mirror, which spoils the
lower one, and so on: an infinite series of images, alternating in sign, one
per cell of width w = ln(U/L), tiling the whole line.

In log price, with l = ln L, nu = r - q - sigma^2/2 and a = 2 nu / sigma^2:

  cell j = (l + j w, l + (j+1) w)
  even j = -2n: translated copy   +e^{a n w}           F(z + 2 n w)
  odd  j = 2n-1: reflected copy   -e^{a (l + n w - z)} F(2 l + 2 n w - z)

where F is the payoff on the corridor (l, l + w). Cell 0 is the payoff itself,
cell -1 is the single-barrier mirror image below L, cell +1 the one above U.
One blur of the tiled function gives the price. The weights are the same
Girsanov likelihood ratio as in the single-barrier case, applied image by image.
"""
import numpy as np
from scipy.stats import norm

from blur_pricer import gaussian_kernel
from barriers import (anchored_grid, sample_payoff, blur_rows, interp_cubic,
                      payoff_fn, grid_width, vanilla)


# ---------- closed form: the same image series, integrated exactly ----------

def _gauss_interval(cp, K, a, b, m, s):
    """E[f(e^Y) 1{a < Y < b}] for Y ~ N(m, s^2), f the call or put payoff."""
    k = np.log(K)
    if cp == "call":
        a = max(a, k)
    else:
        b = min(b, k)
    if b <= a:
        return 0.0
    ef = np.exp(m + 0.5 * s * s)
    pS = norm.cdf((b - m - s * s) / s) - norm.cdf((a - m - s * s) / s)
    pK = norm.cdf((b - m) / s) - norm.cdf((a - m) / s)
    return ef * pS - K * pK if cp == "call" else K * pK - ef * pS


def dko_closed_form(S, K, L, U, sigma, T, r, q=0.0, cp="call", tol=1e-17, n_max=200):
    """Double knock-out by the method of images, each image's Gaussian integral done exactly.

    The killed transition density from x is
      sum_n  e^{a n w} N(x + 2nw + nu T, s^2)  -  e^{a(l - x + n w)} N(2l - x + 2nw + nu T, s^2)
    restricted to the corridor. Terms are added in pairs n = +/-k until they stop mattering.
    """
    if not (L < S < U):
        return 0.0
    x, l, u = np.log(S), np.log(L), np.log(U)
    w = u - l
    nu = r - q - 0.5 * sigma**2
    a = 2 * nu / sigma**2
    s = sigma * np.sqrt(T)

    def term(n):
        direct = np.exp(a * n * w) * _gauss_interval(cp, K, l, u, x + 2 * n * w + nu * T, s)
        image = np.exp(a * (l - x + n * w)) * _gauss_interval(cp, K, l, u, 2 * l - x + 2 * n * w + nu * T, s)
        return direct - image

    total = term(0)
    for n in range(1, n_max):
        t = term(n) + term(-n)
        total += t
        if abs(t) < tol and n > 2:
            break
    return np.exp(-r * T) * total


# ---------- the blur ----------

def image_payoff(K, L, U, sigma, r, q=0.0, cp="call", images=None, z_range=None):
    """The tiled function g(z), its breakpoints, and the cell index function.

    images: keep cells |j| <= images (None: every cell inside z_range).
    """
    l, u = np.log(L), np.log(U)
    w = u - l
    a = 2 * (r - q - 0.5 * sigma**2) / sigma**2
    f = payoff_fn(K, cp)
    jmax = images if images is not None else 10**6

    def g(z):
        z = np.asarray(z, dtype=float)
        j = np.floor((z - l) / w).astype(np.int64)
        even = (j % 2) == 0
        n_even = -j // 2
        n_odd = (j + 1) // 2
        trans = np.exp(a * n_even * w) * f(z + 2 * n_even * w)
        refl = -np.exp(a * (l + n_odd * w - z)) * f(2 * l + 2 * n_odd * w - z)
        out = np.where(even, trans, refl)
        return np.where(np.abs(j) <= jmax, out, 0.0)

    lo, hi = z_range
    k = np.log(K)
    jumps, kinks = [], []
    for j in range(int(np.floor((lo - l) / w)) - 1, int(np.ceil((hi - l) / w)) + 2):
        edge = l + j * w
        if lo < edge < hi and abs(j) <= jmax + 1:
            jumps.append(edge)
        if abs(j) > jmax:
            continue
        n = -j // 2 if j % 2 == 0 else (j + 1) // 2
        kz = k - 2 * n * w if j % 2 == 0 else 2 * l + 2 * n * w - k
        if l + j * w < kz < l + (j + 1) * w and lo < kz < hi:
            kinks.append(kz)
    return g, kinks, jumps


def dko_grid(S0, L, U, sigma, T, dx_max=0.0005):
    """Grid with both barriers (and so every mirror line) on nodes."""
    l, u = np.log(L), np.log(U)
    m = int(np.ceil((u - l) / dx_max))
    dx = (u - l) / m
    return anchored_grid(l, np.log(S0), grid_width(sigma, T), dx)


def dko_blur(S0, K, L, U, sigma, T, r, q=0.0, cp="call", images=None, dx_max=0.0005, full=False):
    """Double knock-out price at S0 by ONE blur of the image series."""
    z = dko_grid(S0, L, U, sigma, T, dx_max)
    dx = z[1] - z[0]
    g, kinks, jumps = image_payoff(K, L, U, sigma, r, q, cp, images, (z[1], z[-2]))
    v = sample_payoff(g, z, kinks, jumps)
    ker = gaussian_kernel(dx, sigma * np.sqrt(T), center=(r - q - 0.5 * sigma**2) * T)
    V = np.exp(-r * T) * blur_rows(v, ker)[0]
    if full:
        return z, V, v
    return interp_cubic(z, V, np.log(S0))
