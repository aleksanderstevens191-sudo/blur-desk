"""Write src/fixtures_2d.json: two-stock blur prices (float64) for the GPU checks.

Each case is one 2D blur laid out as src/blur2d.js job2d() builds it: grid
steps c = 0.06 of the conditional sds, kernel window +-6 sd (the GPU runs in
32-bit, so 8 sd buys nothing), and a patch of spots +-0.2 in log-price around
(S1, S2). The fixture keeps the spot price, a 5 x 5 sample of the patch, the
closed form where there is one, and the image and kernel sizes so the JS side
can prove it rebuilt the same grid.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from blur2d import (
    grid_steps, kernel_2d, sample_payoff_2d, blur2d, margrabe, stulz_call_max,
    stulz_call_min, bs, exchange_payoff, best_of_call_payoff, worst_of_call_payoff,
    spread_payoff, product_payoff,
)

C, WIDTH, PATCH = 0.06, 6.0, 0.2

CASES = [
    # payoff, K, S1, S2, s1, s2, rho, T, r, q1, q2
    ("exchange", None, 100, 100, 0.30, 0.30, 0.5, 1.0, 0.05, 0.0, 0.0),
    ("exchange", None, 100, 90, 0.30, 0.20, -0.5, 0.5, 0.03, 0.02, 0.0),
    ("exchange", None, 100, 100, 0.20, 0.20, 0.9, 0.5, 0.05, 0.0, 0.0),
    ("exchange", None, 100, 100, 0.60, 0.40, -0.5, 2.0, 0.05, 0.0, 0.0),
    ("best_of_call", 100, 100, 100, 0.30, 0.30, 0.5, 1.0, 0.05, 0.0, 0.0),
    ("best_of_call", 110, 100, 90, 0.25, 0.35, -0.3, 1.0, 0.04, 0.01, 0.02),
    ("best_of_call", 100, 100, 100, 0.60, 0.40, -0.5, 2.0, 0.05, 0.0, 0.0),
    ("worst_of_call", 95, 100, 100, 0.30, 0.30, 0.5, 1.0, 0.05, 0.0, 0.0),
    ("spread", 5, 100, 95, 0.30, 0.25, 0.7, 1.0, 0.05, 0.0, 0.0),
    ("call_x_put", None, 100, 100, 0.25, 0.20, 0.0, 1.0, 0.05, 0.0, 0.0),
]


def payoff_of(name, K):
    if name == "exchange":
        return exchange_payoff()
    if name == "best_of_call":
        return best_of_call_payoff(K)
    if name == "worst_of_call":
        return worst_of_call_payoff(K)
    if name == "spread":
        return spread_payoff(K)
    # (S1 - 103)+ (97 - S2)+ / 100, so prices stay O(10) like the others
    g, rk, ck = product_payoff(103.0, "call", 97.0, "put")
    return (lambda X, Y: g(X, Y) / 100, lambda y: [(b, j1 / 100, j2 / 100) for b, j1, j2 in rk(y)], ck)


def closed_form(name, K, S1, S2, s1, s2, rho, T, r, q1, q2):
    if name == "exchange":
        return margrabe(S1, S2, s1, s2, rho, T, q1, q2)
    if name == "best_of_call":
        return stulz_call_max(S1, S2, K, s1, s2, rho, T, r, q1, q2)
    if name == "worst_of_call":
        return stulz_call_min(S1, S2, K, s1, s2, rho, T, r, q1, q2)
    if name == "call_x_put":  # independent: e^{rT} V1 V2 / 100
        return np.exp(r * T) * bs(S1, 103.0, s1, T, r, q1) * bs(S2, 97.0, s2, T, r, q2, "put") / 100
    return None  # spread: no closed form (Kirk is an approximation; Stage 3)


def build(name, K, S1, S2, s1, s2, rho, T, r, q1, q2):
    dx, dy = grid_steps(s1, s2, rho, T, C)
    xo, yo, W = kernel_2d(dx, dy, s1, s2, rho, T, r, q1, q2, WIDTH)
    qx, qy = int(round(PATCH / dx)), int(round(PATCH / dy))
    px, py = (len(xo) - 1) // 2, (len(yo) - 1) // 2
    xs = np.log(S1) + (np.arange(len(xo) + 2 * qx) - px - qx) * dx
    ys = np.log(S2) + (np.arange(len(yo) + 2 * qy) - py - qy) * dy
    P = sample_payoff_2d(payoff_of(name, K), xs, ys)
    V = np.exp(-r * T) * blur2d(P, W)
    return dict(dx=dx, dy=dy, mx=len(xo), my=len(yo), nx=len(xs), ny=len(ys), qx=qx, qy=qy), V


def main():
    out = {"c": C, "width": WIDTH, "patch": PATCH, "cases": []}
    for case in CASES:
        name, K, S1, S2, s1, s2, rho, T, r, q1, q2 = case
        dims, V = build(*case)
        qx, qy = dims["qx"], dims["qy"]
        ia = np.linspace(0, 2 * qx, 5).round().astype(int)
        ib = np.linspace(0, 2 * qy, 5).round().astype(int)
        cf = closed_form(*case)
        out["cases"].append(dict(
            payoff=name, K=K, S1=S1, S2=S2, s1=s1, s2=s2, rho=rho, T=T, r=r, q1=q1, q2=q2, **dims,
            spot=float(V[qx, qy]), closed=None if cf is None else float(cf),
            samples=[[int(a), int(b), float(V[a, b])] for a in ia for b in ib],
        ))
        print(f"{name:<14} rho={rho:+.1f} image {dims['nx']}x{dims['ny']} kernel {dims['mx']}x{dims['my']}"
              f"  spot {V[qx, qy]:.8f}" + ("" if cf is None else f"  closed {cf:.8f}  diff {abs(V[qx, qy] - cf):.1e}"))
    path = Path(__file__).parent.parent / "src" / "fixtures_2d.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
