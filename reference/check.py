"""Run every check and print the tables that go in the README and the posts."""
import sys
import time
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from blur_pricer import (
    bachelier_call, bs_call, bs_put, lognormal_density,
    bachelier_blur_prices, bs_blur_prices,
    american_put_blur, american_put_tree, interp_at,
)

S0, r = 100.0, 0.05


def grid_around(S0, half_width, dx):
    """Log grid with ln(S0) exactly on a node."""
    n = int(np.ceil(half_width / dx))
    x0 = np.log(S0)
    return x0 - n * dx, x0 + n * dx


print("=" * 64)
print("1) Bachelier: plain blur in price space vs closed form")
print("=" * 64)
sigma_n, T = 20.0, 1.0
worst = 0
for dS in [0.1, 0.05, 0.01]:
    S, V = bachelier_blur_prices(100.0, sigma_n, T, 0.0, 200.0, dS)
    errs = []
    for Sq in [80, 90, 100, 110, 120]:
        errs.append(abs(interp_at(S, V, Sq) - bachelier_call(Sq, 100.0, sigma_n, T)))
    print(f"  grid step {dS:>5}: max error ${max(errs):.2e}")
    worst = max(errs)

print()
print("=" * 64)
print("2) Black-Scholes: off-center blur in log space vs closed form")
print("=" * 64)
sigma = 0.20
print(f"  S0={S0}, r={r}, sigma={sigma}")
print(f"  {'T':>5} {'K':>6} {'blur':>11} {'Black-Scholes':>14} {'error':>10}")
max_err = 0
for dx in [0.002, 0.0005]:
    rows = []
    for T in [0.1, 0.5, 1.0, 2.0]:
        lo, hi = grid_around(S0, 10 * sigma * np.sqrt(T) + 0.5, dx)
        for K in [80, 90, 100, 110, 120]:
            S, V = bs_blur_prices(lambda s, K=K: np.maximum(s - K, 0.0), sigma, T, r, lo, hi, dx)
            i = np.argmin(abs(S - S0))
            b, c = V[i], bs_call(S0, K, sigma, T, r)
            rows.append((T, K, b, c, abs(b - c)))
    e = max(x[4] for x in rows)
    print(f"  -- log grid step {dx}: max error over 20 options = ${e:.2e}")
    if dx == 0.0005:
        for T, K, b, c, err in rows:
            print(f"  {T:>5} {K:>6} {b:>11.6f} {c:>14.6f} {err:>10.2e}")
        max_err = e

print()
print("=" * 64)
print("3) American put: blur, discount, clip, repeat vs binomial tree")
print("=" * 64)
K, T, sigma = 100.0, 1.0, 0.20
t0 = time.perf_counter()
tree_ref = american_put_tree(S0, K, sigma, T, r, 20000)
t_tree = time.perf_counter() - t0
euro = bs_put(S0, K, sigma, T, r)
print(f"  European put (closed form):           {euro:.6f}")
print(f"  American put, 20,000-step tree:       {tree_ref:.6f}   ({t_tree:.2f}s)")
amer_results = {}
for steps, dx in [(100, 0.002), (500, 0.001), (2000, 0.0005)]:
    lo, hi = grid_around(S0, 10 * sigma * np.sqrt(T) + 0.5, dx)
    t0 = time.perf_counter()
    S, V = american_put_blur(K, sigma, T, r, steps, lo, hi, dx)
    dt_ = time.perf_counter() - t0
    i = np.argmin(abs(S - S0))
    amer_results[steps] = V[i]
    print(f"  blur, {steps:>5} steps, dx={dx}:       {V[i]:.6f}   err {abs(V[i]-tree_ref):.1e}   ({dt_:.2f}s)")
# Richardson on time steps (error ~ 1/N for Bermudan->American)
rich = 2 * amer_results[2000] - amer_results[1000] if 1000 in amer_results else None
lo, hi = grid_around(S0, 10 * sigma * np.sqrt(T) + 0.5, 0.0005)
S, V1000 = american_put_blur(K, sigma, T, r, 1000, lo, hi, 0.0005)
v1000 = V1000[np.argmin(abs(S - S0))]
rich = 2 * amer_results[2000] - v1000
print(f"  blur + Richardson (2*N2000 - N1000):  {rich:.6f}   err {abs(rich-tree_ref):.1e}")
print(f"  early-exercise premium: ${tree_ref - euro:.4f}")

print()
print("=" * 64)
print("4) Edge detect: [1,-2,1] across strikes on call prices = market odds")
print("=" * 64)
T, sigma = 0.25, 0.20
dK = 0.5
Ks = np.arange(60, 150 + dK / 2, dK)
C = bs_call(S0, Ks, sigma, T, r)
dens = np.exp(r * T) * (C[:-2] - 2 * C[1:-1] + C[2:]) / dK**2
true = lognormal_density(Ks[1:-1], S0, sigma, T, r)
print(f"  strike spacing ${dK}: max density error {np.max(abs(dens-true)):.2e} "
      f"(peak density {true.max():.4f})")
print(f"  density sums to {np.sum(dens)*dK:.5f} (should be ~1)")
p_down10 = np.sum(dens[Ks[1:-1] < 90]) * dK
print(f"  risk-neutral P(S_T < 90) from edge detect: {p_down10:.4f}")

print()
print("=" * 64)
print("5) Speed of the CPU reference (numpy), European call, 1 blur")
print("=" * 64)
lo, hi = grid_around(S0, 10 * 0.2 + 0.5, 0.0005)
t0 = time.perf_counter()
reps = 50
for _ in range(reps):
    S, V = bs_blur_prices(lambda s: np.maximum(s - 100, 0.0), 0.2, 1.0, r, lo, hi, 0.0005)
dt_ = (time.perf_counter() - t0) / reps
print(f"  grid points: {len(S):,}; one blur prices every spot on the grid in {dt_*1e3:.2f} ms")
print(f"  = {len(S)/dt_:,.0f} option prices/sec on CPU (baseline for the GPU version)")
