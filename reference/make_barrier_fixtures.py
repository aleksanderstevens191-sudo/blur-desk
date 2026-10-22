"""Write src/barrier_fixtures.json: barrier prices from the Python reference that the
JS core (src/barrier.js) and the WebGPU shader (src/gpu.js) must reproduce.

Each case carries its grid exactly (z_j = anchor + (lo + j) dx, j = 0..n-1) so the
browser rebuilds the same nodes instead of re-deriving them.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from barriers import (TYPES, parse, barrier_closed_form, mirror_price, stepping_prices,
                      grid_width)
from double_barrier import dko_blur, dko_closed_form, dko_grid

S0, DX = 100.0, 0.0005


def grid_info(anchor, x0, half_width, dx):
    lo = int(np.floor((x0 - half_width - anchor) / dx))
    hi = int(np.ceil((x0 + half_width - anchor) / dx))
    return {"anchor": float(anchor), "lo": lo, "n": hi - lo + 1, "dx": float(dx)}


cases = []

# (b) weighted mirror, every type, two markets; (a) plain mirror, two cases
BARRIERS = {"down": (90.0, {"call": 100.0, "put": 100.0}), "up": (115.0, {"call": 100.0, "put": 105.0})}
for sigma, T, r in [(0.20, 1.0, 0.05), (0.30, 0.5, 0.08)]:
    for kind in TYPES:
        ud, io, cp = parse(kind)
        H, Ks = BARRIERS[ud]
        K = Ks[cp]
        g = grid_info(np.log(H), np.log(S0), grid_width(sigma, T), DX)
        cases.append({"method": "weighted", "kind": kind, "K": K, "H": H, "sigma": sigma, "T": T, "r": r,
                      "grid": g, "py": float(mirror_price(kind, S0, K, H, sigma, T, r, dx=DX)),
                      "closedForm": float(barrier_closed_form(kind, S0, K, H, sigma, T, r))})
for kind in ["down-and-out call", "up-and-out put"]:
    sigma, T, r = 0.2, 1.0, 0.05
    H, Ks = BARRIERS[parse(kind)[0]]
    K = Ks[parse(kind)[2]]
    g = grid_info(np.log(H), np.log(S0), grid_width(sigma, T), DX)
    cases.append({"method": "plain", "kind": kind, "K": K, "H": H, "sigma": sigma, "T": T, "r": r,
                  "grid": g, "py": float(mirror_price(kind, S0, K, H, sigma, T, r, dx=DX, weighted=False)),
                  "closedForm": float(barrier_closed_form(kind, S0, K, H, sigma, T, r))})

# double knock-out: the image series, including a truncated one and a strike outside
for K, L, U, sigma, T, r, cp, images in [
    (100.0, 90.0, 110.0, 0.20, 1.0, 0.05, "call", None),
    (100.0, 80.0, 125.0, 0.25, 0.5, 0.05, "put", None),
    (85.0, 90.0, 110.0, 0.20, 1.0, 0.05, "call", None),
    (100.0, 90.0, 110.0, 0.20, 1.0, 0.05, "call", 2),
]:
    z = dko_grid(S0, L, U, sigma, T, DX)
    dx = z[1] - z[0]
    l = np.log(L)
    lo = int(round((z[0] - l) / dx))
    m = int(round(np.log(U / L) / dx))
    cases.append({"method": "double", "kind": f"double-knock-out {cp}", "K": K, "L": L, "U": U,
                  "sigma": sigma, "T": T, "r": r, "images": images, "m": m,
                  "grid": {"anchor": float(l), "lo": lo, "n": len(z), "dx": float(dx)},
                  "py": float(dko_blur(S0, K, L, U, sigma, T, r, cp=cp, images=images)),
                  "closedForm": float(dko_closed_form(S0, K, L, U, sigma, T, r, cp=cp))})

# (c) step by step, with and without the BGK shift
for ud, cp, K, H, steps, shift in [("up", "call", 100.0, 120.0, 50, False), ("up", "call", 100.0, 120.0, 50, True),
                                   ("down", "put", 100.0, 90.0, 24, True), ("down", "call", 100.0, 90.0, 24, False)]:
    sigma, T, r = 0.2, 1.0, 0.05
    kinds = [f"{ud}-and-out {cp}", f"{ud}-and-in {cp}"]
    info = {}
    p = stepping_prices(kinds, S0, K, H, sigma, T, r, steps, shift=shift, dx_max=0.001, info=info)
    cases.append({"method": "step", "kind": f"{ud} {cp}", "K": K, "H": H, "sigma": sigma, "T": T, "r": r,
                  "steps": steps, "shift": shift, "ib": info["ib"],
                  "grid": {k: info[k] for k in ["anchor", "lo", "n", "dx"]},
                  "py": {"out": float(p[kinds[0]]), "in": float(p[kinds[1]]), "vanilla": float(p["vanilla"])},
                  "closedForm": {"out": float(barrier_closed_form(kinds[0], S0, K, H, sigma, T, r)),
                                 "in": float(barrier_closed_form(kinds[1], S0, K, H, sigma, T, r))}})

out = {"S0": S0, "cases": cases}
path = Path(__file__).parent.parent / "src" / "barrier_fixtures.json"
path.write_text(json.dumps(out, indent=1))
print(f"wrote {len(cases)} cases to {path.name}")
