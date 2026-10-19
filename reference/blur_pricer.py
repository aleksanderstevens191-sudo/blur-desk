"""
Blur Desk reference implementation.

Core idea: an option price is its payoff, blurred.
  - Bachelier model (r = 0): a plain Gaussian blur of the payoff in price space.
  - Black-Scholes: a Gaussian blur in log-price space with the kernel nudged
    off-center by the drift, then discounted.
  - American options: blur a small time step, discount, clip to the payoff, repeat.

This file is the ground truth the browser/GPU version must match.
"""
import numpy as np
from scipy.stats import norm


# ---------- closed forms ----------

def bachelier_call(S, K, sigma_n, T):
    s = sigma_n * np.sqrt(T)
    d = (S - K) / s
    return (S - K) * norm.cdf(d) + s * norm.pdf(d)


def bs_call(S, K, sigma, T, r):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, sigma, T, r):
    return bs_call(S, K, sigma, T, r) - S + K * np.exp(-r * T)


def lognormal_density(K, S, sigma, T, r):
    """Risk-neutral density of S_T under Black-Scholes."""
    m = np.log(S) + (r - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    return np.exp(-((np.log(K) - m) ** 2) / (2 * s * s)) / (K * s * np.sqrt(2 * np.pi))


# ---------- the blur ----------

def gaussian_kernel(dx, std, center=0.0, width=8.0):
    """Sampled Gaussian kernel on a grid of spacing dx.

    center shifts the kernel off zero; this is how Black-Scholes drift enters.
    The kernel is renormalized so it sums to 1 (it is a probability blur).
    """
    half = int(np.ceil((width * std + abs(center)) / dx))
    offs = np.arange(-half, half + 1) * dx
    k = np.exp(-((offs - center) ** 2) / (2 * std * std))
    return k / k.sum()


def blur(values, kernel):
    """1D convolution, 'same' length.

    Edges are extended by odd reflection, which continues the payoff as a
    straight line. Option payoffs are linear far from the strike, so this is
    exact there; padding with a constant would bend the tails and leak error.
    """
    pad = len(kernel) // 2
    if pad >= len(values):
        raise ValueError("kernel wider than grid; widen the grid")
    padded = np.pad(values, pad, mode="reflect", reflect_type="odd")
    # kernel[j] is weight for offset j; output(x) = sum_j k_j * f(x + off_j)
    return np.convolve(padded, kernel[::-1], mode="valid")


# ---------- pricers built from the blur ----------

def bachelier_blur_prices(K, sigma_n, T, S_lo, S_hi, dS):
    S = np.arange(S_lo, S_hi + dS / 2, dS)
    payoff = np.maximum(S - K, 0.0)
    k = gaussian_kernel(dS, sigma_n * np.sqrt(T))
    return S, blur(payoff, k)


def bs_blur_prices(payoff_fn, sigma, T, r, x_lo, x_hi, dx):
    """European option under Black-Scholes by one off-center blur in log space.

    V(x) = e^{-rT} * E[ payoff(exp(x + mu + s Z)) ],  mu = (r - sigma^2/2) T,  s = sigma sqrt(T)
    Sampling the kernel at offsets j*dx with center +mu implements the shift exactly,
    with no interpolation.
    """
    x = np.arange(x_lo, x_hi + dx / 2, dx)
    g = payoff_fn(np.exp(x))
    mu = (r - 0.5 * sigma**2) * T
    k = gaussian_kernel(dx, sigma * np.sqrt(T), center=mu)
    return np.exp(x), np.exp(-r * T) * blur(g, k)


def american_put_blur(K, sigma, T, r, steps, x_lo, x_hi, dx):
    """American put: blur one time step, discount, clip to payoff. Repeat."""
    x = np.arange(x_lo, x_hi + dx / 2, dx)
    S = np.exp(x)
    payoff = np.maximum(K - S, 0.0)
    dt = T / steps
    k = gaussian_kernel(dx, sigma * np.sqrt(dt), center=(r - 0.5 * sigma**2) * dt)
    disc = np.exp(-r * dt)
    V = payoff.copy()
    for _ in range(steps):
        V = np.maximum(disc * blur(V, k), payoff)
    return S, V


def american_put_tree(S0, K, sigma, T, r, steps):
    """Cox-Ross-Rubinstein binomial tree, the textbook reference."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    S = S0 * u ** (steps - j) * d**j
    V = np.maximum(K - S, 0.0)
    for n in range(steps - 1, -1, -1):
        S = S0 * u ** (n - np.arange(n + 1)) * d ** np.arange(n + 1)
        V = np.maximum(disc * (p * V[:-1] + (1 - p) * V[1:]), K - S)
    return V[0]


def interp_at(S_grid, V_grid, S0):
    # cubic-ish accuracy via local quadratic would be better; linear is O(dx^2) and fine here
    return np.interp(S0, S_grid, V_grid)
