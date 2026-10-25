"""
Two-stock options as one 2D blur (float64 reference).

With two stocks in Black-Scholes, the log-prices x = ln S1 and y = ln S2 move
together as a correlated 2D Brownian motion. Over a time T the pair of
log-returns (X, Y) is a 2D Gaussian with

    mean        (mu1, mu2),  mu_i = (r - q_i - sigma_i^2 / 2) T
    covariance  T * [[s1^2,        rho s1 s2],
                     [rho s1 s2,   s2^2     ]]

so the price of any payoff f(S1, S2) paid at T is the payoff, blurred:

    V(x, y) = e^{-rT} * sum_{i,j} w_ij * f(e^{x + i dx}, e^{y + j dy})

with w_ij the 2D Gaussian sampled on the grid offsets (i dx, j dy). The level
sets of that Gaussian are ellipses: the two volatilities set the ellipse's
width along each axis and the correlation tilts it, toward the diagonal for
rho > 0 (the stocks tend to move the same way) and toward the anti-diagonal
for rho < 0. The drift correction -sigma_i^2/2 is what makes each forward come
out right: E[e^X] = e^{(r - q1) T} and E[e^Y] = e^{(r - q2) T}.

Sampling. A Gaussian times a smooth function is summed on a grid to near
machine precision (the error is aliasing, ~exp(-2 pi^2 k' C k) over the
dual-lattice vectors k), so all the error comes from where the payoff has
kinks. Each kink gets the same Euler-Maclaurin correction as the 1D blur
(barriers.sample_payoff), applied along one direction that crosses it:

  row kinks   kinks a row of the grid crosses (vertical lines x = c, the
              diagonal x = y + c, curves like S1 = S2 + K). On each row the
              kink sits at some x; the row's two nodes around it get the
              Euler-Maclaurin correction (see _kink_corr). After that each
              row sum is a smooth function of y, so summing rows is exact.
  col kinks   horizontal lines y = c, which no row crosses. Corrected the same
              way down each column, on top of the row-corrected values (so a
              product payoff gets exactly the product of the 1D corrections).

None of the corrections depend on the kernel, so one corrected payoff image
serves every volatility, correlation and maturity (and the GPU).
"""
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import ndtr, owens_t

SQRT2PI = np.sqrt(2 * np.pi)


# ---------- closed forms ----------

def bvn_cdf(h, k, rho):
    """P(Z1 <= h, Z2 <= k) for standard normals with correlation rho (Owen's T)."""
    h, k = float(h), float(k)
    if abs(rho) >= 1:
        return float(ndtr(min(h, k))) if rho > 0 else max(0.0, float(ndtr(h) - ndtr(-k)))
    s = np.sqrt(1 - rho * rho)
    if h == 0 and k == 0:
        return 0.25 + np.arcsin(rho) / (2 * np.pi)
    # Owen (1956): Phi2 = (Phi(h) + Phi(k))/2 - T(h, a_h) - T(k, a_k) - beta
    def a(u, v):
        return np.inf if u == 0 else (v - rho * u) / (u * s)
    def t(u, v):
        if u == 0:
            return 0.25 * np.sign(v - rho * u)  # T(0, +-inf) = +-1/4
        return owens_t(u, a(u, v))
    beta = 0.5 if (h * k < 0 or (h * k == 0 and h + k < 0)) else 0.0
    return 0.5 * (ndtr(h) + ndtr(k)) - t(h, k) - t(k, h) - beta


def margrabe(S1, S2, s1, s2, rho, T, q1=0.0, q2=0.0):
    """Option to exchange stock 2 for stock 1 at T: payoff max(S1 - S2, 0).

    No r: the price is a ratio of two stocks, so stock 2 acts as the numeraire.
    """
    s = np.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2)
    d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * s * s) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    return S1 * np.exp(-q1 * T) * ndtr(d1) - S2 * np.exp(-q2 * T) * ndtr(d2)


def stulz_call_max(S1, S2, K, s1, s2, rho, T, r, q1=0.0, q2=0.0):
    """Call on the better of two stocks, max(max(S1, S2) - K, 0) (Stulz 1982)."""
    b1, b2 = r - q1, r - q2
    s = np.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2)
    sT, s1T, s2T = s * np.sqrt(T), s1 * np.sqrt(T), s2 * np.sqrt(T)
    d = (np.log(S1 / S2) + (b1 - b2 + 0.5 * s * s) * T) / sT
    y1 = (np.log(S1 / K) + (b1 + 0.5 * s1 * s1) * T) / s1T
    y2 = (np.log(S2 / K) + (b2 + 0.5 * s2 * s2) * T) / s2T
    rho1, rho2 = (s1 - rho * s2) / s, (s2 - rho * s1) / s
    return (S1 * np.exp(-q1 * T) * bvn_cdf(y1, d, rho1)
            + S2 * np.exp(-q2 * T) * bvn_cdf(y2, -d + sT, rho2)
            - K * np.exp(-r * T) * (1 - bvn_cdf(-y1 + s1T, -y2 + s2T, rho)))


def stulz_call_min(S1, S2, K, s1, s2, rho, T, r, q1=0.0, q2=0.0):
    """Call on the worse of two stocks, via min = S1 + S2 - max and parity:
    C_min(K) = C1(K) + C2(K) - C_max(K)."""
    return (bs(S1, K, s1, T, r, q1) + bs(S2, K, s2, T, r, q2)
            - stulz_call_max(S1, S2, K, s1, s2, rho, T, r, q1, q2))


def bs(S, K, sigma, T, r, q=0.0, cp="call"):
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sT
    d2 = d1 - sT
    c = S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    return c if cp == "call" else c - S * np.exp(-q * T) + K * np.exp(-r * T)


# ---------- the kernel ----------

def drifts(T, r, s1, s2, q1=0.0, q2=0.0):
    """Mean log-returns: the drift correction that makes each forward grow at r - q."""
    return (r - q1 - 0.5 * s1 * s1) * T, (r - q2 - 0.5 * s2 * s2) * T


def ellipse(s1, s2, rho, T):
    """The blur ellipse (1-sd level set of the kernel) in log-price.

    Returns the two semi-axes (major first) and the major axis's angle from the
    x axis in radians. Correlation is what rotates and stretches it.
    """
    C = T * np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])
    ev, vec = np.linalg.eigh(C)
    major = vec[:, 1] * np.sign(vec[0, 1] if vec[0, 1] != 0 else 1)
    return np.sqrt(ev[1]), np.sqrt(ev[0]), float(np.arctan2(major[1], major[0]))


def grid_steps(s1, s2, rho, T, c=0.03):
    """Grid spacings dx, dy: a fraction c of each stock's sd given the other one,
    s_i sqrt(T (1 - rho^2)). That is the kernel's width along a row (or column),
    which is what the kink corrections need resolved; at high |rho| it also keeps
    the ellipse's narrow axis well sampled."""
    f = c * np.sqrt(T * (1 - rho * rho))
    return f * s1, f * s2


def kernel_1d(d, std, center, width=8.0):
    """Offsets and normalized weights of a 1D Gaussian sampled at j*d."""
    half = int(np.ceil((width * std + abs(center)) / d))
    off = np.arange(-half, half + 1) * d
    w = np.exp(-0.5 * ((off - center) / std) ** 2)
    return off, w / w.sum()


def kernel_2d(dx, dy, s1, s2, rho, T, r, q1=0.0, q2=0.0, width=8.0):
    """2D Gaussian kernel on offsets (i dx, j dy): (xo, yo, W) with W[i, j], sum 1.

    The window is +-width marginal sds around each mean. With rho = 0 the result
    is exactly the outer product of the two kernel_1d's.
    """
    m1, m2 = drifts(T, r, s1, s2, q1, q2)
    a, b = s1 * np.sqrt(T), s2 * np.sqrt(T)
    xo, wx = kernel_1d(dx, a, m1, width)
    yo, wy = kernel_1d(dy, b, m2, width)
    if rho == 0:
        return xo, yo, np.outer(wx, wy)
    u = ((xo - m1) / a)[:, None]
    v = ((yo - m2) / b)[None, :]
    W = np.exp(-(u * u - 2 * rho * u * v + v * v) / (2 * (1 - rho * rho)))
    return xo, yo, W / W.sum()


# ---------- payoffs and their kinks ----------
# Each payoff is (g, row_kinks, col_kinks):
#   g(X, Y)        payoff on log-prices (arrays, broadcast)
#   row_kinks(y)   kinks along row y, as (x, j1, j2): position and the jumps of
#                  df/dx and d2f/dx2 across it (may be [])
#   col_kinks      y values of horizontal kink lines (jumps found numerically)

def exchange_payoff():
    """max(S1 - S2, 0): receive stock 1, give up stock 2. Kinks on x = y."""
    return (lambda X, Y: np.maximum(np.exp(X) - np.exp(Y), 0.0),
            lambda y: [(y, np.exp(y), np.exp(y))], [])


def best_of_call_payoff(K):
    """max(max(S1, S2) - K, 0). Kinks: x = ln K below the diagonal, y = ln K
    left of it, and the diagonal itself above (ln K, ln K)."""
    k = np.log(K)
    return (lambda X, Y: np.maximum(np.exp(np.maximum(X, Y)) - K, 0.0),
            lambda y: [(k, K, K)] if y < k else [(y, np.exp(y), np.exp(y))], [k])


def worst_of_call_payoff(K):
    """max(min(S1, S2) - K, 0). Kinks: x = ln K above the diagonal, y = ln K
    right of it, and the diagonal above (ln K, ln K)."""
    k = np.log(K)
    return (lambda X, Y: np.maximum(np.exp(np.minimum(X, Y)) - K, 0.0),
            lambda y: [(k, K, K), (y, -np.exp(y), -np.exp(y))] if y > k else [], [k])


def spread_payoff(K):
    """max(S1 - S2 - K, 0). For K > 0 the kink is the curve S1 = S2 + K."""
    if K == 0:
        return exchange_payoff()
    return (lambda X, Y: np.maximum(np.exp(X) - np.exp(Y) - K, 0.0),
            lambda y: [(np.log(np.exp(y) + K), np.exp(y) + K, np.exp(y) + K)]
            if np.exp(y) + K > 0 else [], [])


def vanilla_1d(K, cp):
    k = np.log(K)
    f = (lambda z: np.maximum(np.exp(z) - K, 0.0)) if cp == "call" else (lambda z: np.maximum(K - np.exp(z), 0.0))
    return f, k


def product_payoff(K1, cp1, K2, cp2):
    """f1(S1) * f2(S2), a product of two vanilla payoffs. Kinks on x = ln K1, y = ln K2."""
    f1, k1 = vanilla_1d(K1, cp1)
    f2, k2 = vanilla_1d(K2, cp2)
    return (lambda X, Y: f1(X) * f2(Y), lambda y: [(k1, K1 * f2(y), K1 * f2(y))], [k2])


# ---------- sampling with kink corrections ----------

def _one_sided(f, b, side, eps):
    """First and second derivatives at b from one side (cubic through four
    points at distances eps .. 4 eps); f maps a 1D array of positions to an
    array (..., n) of values."""
    g1, g2, g3, g4 = (f(np.array([b + side * eps * t]))[..., 0] for t in (1, 2, 3, 4))
    return (side * (-52 * g1 + 114 * g2 - 84 * g3 + 22 * g4) / (12 * eps),
            (3 * g1 - 8 * g2 + 7 * g3 - 2 * g4) / (eps * eps))


def _jumps(f, b, eps):
    (d1r, d2r), (d1l, d2l) = _one_sided(f, b, +1, eps), _one_sided(f, b, -1, eps)
    return d1r - d1l, d2r - d2l


def _kink_corr(theta, d, j1, j2):
    """Corrections for nodes i and i+1 around a kink a fraction theta past node i.

    Euler-Maclaurin: on a grid of spacing d the sum of F = f * kernel misses the
    integral by -d^2 B2(theta)/2 [F'] + d^3 B3(theta)/6 [F''] + O(d^4), where
    [F'] = j1 k(a) and [F''] = j2 k(a) + 2 j1 k'(a) (j1, j2 the jumps of f' and
    f''). Two node corrections with the right sum (the k(a) terms) and the right
    first moment about the kink (the k'(a) term) put both back for any smooth
    kernel k.
    """
    b2 = theta * theta - theta + 1 / 6
    b3 = theta * (theta - 0.5) * (theta - 1)
    D0 = d * b2 / 2 * j1 - d * d * b3 / 6 * j2
    D1 = -d * d * b3 / 3 * j1
    right = D1 / d + theta * D0
    return D0 - right, right


def sample_payoff_2d(payoff, xs, ys):
    """Payoff image P[i, j] = g(xs[i], ys[j]) with every kink corrected.

    Row kinks are corrected along x on each row, from their exact jumps; then
    horizontal (col) kinks are corrected along y on each column, from jumps of
    the row-corrected payoff measured by one-sided cubic fits.
    """
    g, row_kinks, col_kinks = payoff
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    eps_y = dy / 20

    def row_corrected(yv):
        """Row-corrected payoff on columns xs at the rows yv (any y, on grid or not)."""
        out = g(xs[:, None], yv[None, :]).astype(float)
        for jy, y in enumerate(yv):
            for b, j1, j2 in row_kinks(y):
                i = int(np.floor((b - xs[0]) / dx))
                if i < 0 or i + 1 >= len(xs):
                    continue
                theta = (b - xs[i]) / dx
                cl, cr = _kink_corr(theta, dx, j1, j2)
                out[i, jy] += cl
                out[i + 1, jy] += cr
        return out

    P = row_corrected(ys)
    for c in col_kinks:
        j = int(np.floor((c - ys[0]) / dy))
        if j < 0 or j + 1 >= len(ys):
            continue
        theta = (c - ys[j]) / dy
        j1, j2 = _jumps(row_corrected, c, eps_y)
        cl, cr = _kink_corr(theta, dy, j1, j2)
        P[:, j] += cl
        P[:, j + 1] += cr
    return P


# ---------- pricing ----------

def blur2d(P, W):
    """Valid 2D blur: out[a, b] = sum_ij W[i, j] P[a + i, b + j] (float64 FFT)."""
    return fftconvolve(P, W[::-1, ::-1], mode="valid")


def price_2d(payoff, S1, S2, s1, s2, rho, T, r, q1=0.0, q2=0.0, c=0.03, width=8.0,
             grid=None, surface=False):
    """Price a two-stock payoff by one 2D blur, at (S1, S2).

    The grid is anchored so (ln S1, ln S2) is a node, and the payoff image is
    padded by the kernel's reach so the blur never needs values off the image.
    surface=True also returns the price surface on a small patch around the spot.
    """
    dx, dy = grid or grid_steps(s1, s2, rho, T, c)
    xo, yo, W = kernel_2d(dx, dy, s1, s2, rho, T, r, q1, q2, width)
    x0, y0 = np.log(S1), np.log(S2)
    xs, ys = x0 + xo, y0 + yo
    P = sample_payoff_2d(payoff, xs, ys)
    v = np.exp(-r * T) * float(np.sum(W * P))
    if not surface:
        return v
    return v, (dx, dy, xo, yo, W, P)


def price_1d(f, k, S, sigma, T, r, q=0.0, d=None, width=8.0):
    """1D blur with the same kernel_1d and kink correction (for the product check)."""
    std = sigma * np.sqrt(T)
    off, w = kernel_1d(d, std, (r - q - 0.5 * sigma * sigma) * T, width)
    z = np.log(S) + off
    v = f(z).astype(float)
    i = int(np.floor((k - z[0]) / d))
    if 0 <= i < len(z) - 1:
        theta = (k - z[i]) / d
        cl, cr = _kink_corr(theta, d, np.exp(k), np.exp(k))  # call or put: f', f'' jump by K
        v[i] += cl
        v[i + 1] += cr
    return float(np.dot(w, v))  # undiscounted: the caller discounts once
