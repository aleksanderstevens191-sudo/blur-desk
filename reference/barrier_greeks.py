"""
Greeks of barrier options read straight off the blurred price surface.

One blur prices every spot on the grid at once, so delta and gamma are just
finite differences between neighbouring nodes. The grid is uniform in
x = ln S, so with V_x, V_xx the central differences in x:

  delta = V_x / S,   gamma = (V_xx - V_x) / S^2
"""
import numpy as np

from blur_pricer import gaussian_kernel
from barriers import (parse, anchored_grid, sample_payoff, blur_rows, payoff_fn,
                      grid_width, mirror_payoff)


def blurred_surface(kind, K, H, sigma, T, r, q=0.0, S_center=100.0, dx=0.0005):
    """Weighted-mirror price at every grid node. Returns S, V (0 past the barrier)."""
    h = np.log(H)
    z = anchored_grid(h, np.log(S_center), grid_width(sigma, T), dx)
    g, kinks, jumps = mirror_payoff(kind, K, H, sigma, r, q)
    v = sample_payoff(g, z, kinks, jumps)
    ker = gaussian_kernel(dx, sigma * np.sqrt(T), center=(r - q - 0.5 * sigma**2) * T)
    V = np.exp(-r * T) * blur_rows(v, ker)[0]
    dead = z < h if parse(kind)[0] == "down" else z > h
    if parse(kind)[1] == "out":
        V[dead] = 0.0
    return np.exp(z), V


def vanilla_surface(K, sigma, T, r, q=0.0, cp="call", S_center=100.0, anchor=None, dx=0.0005):
    x0 = np.log(S_center)
    z = anchored_grid(np.log(anchor) if anchor else x0, x0, grid_width(sigma, T), dx)
    v = sample_payoff(payoff_fn(K, cp), z, kinks=[np.log(K)])
    ker = gaussian_kernel(dx, sigma * np.sqrt(T), center=(r - q - 0.5 * sigma**2) * T)
    return np.exp(z), np.exp(-r * T) * blur_rows(v, ker)[0]


def fd_greeks(S, V):
    """Delta and gamma on interior nodes by central differences in ln S."""
    dx = np.log(S[1]) - np.log(S[0])
    Vx = (V[2:] - V[:-2]) / (2 * dx)
    Vxx = (V[2:] - 2 * V[1:-1] + V[:-2]) / dx**2
    s = S[1:-1]
    return s, Vx / s, (Vxx - Vx) / s**2
