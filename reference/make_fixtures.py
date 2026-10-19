"""Write fixtures.json: Python reference outputs the JS core must reproduce."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from blur_pricer import bs_blur_prices, american_put_blur, bs_call, bs_put

S0, r, dx = 100.0, 0.05, 0.001


def grid(T_for_width, sigma):
    hw = 10 * sigma * np.sqrt(T_for_width) + 0.5
    n = int(np.ceil(hw / dx))
    x0 = np.log(S0)
    return x0 - n * dx, x0 + n * dx, hw


cases = []
for sigma, T, K in [(0.2, 0.25, 95), (0.2, 1.0, 100), (0.35, 0.5, 110)]:
    lo, hi, hw = grid(T, sigma)
    S, V = bs_blur_prices(lambda s, K=K: np.maximum(s - K, 0), sigma, T, r, lo, hi, dx)
    i = int(np.argmin(abs(S - S0)))
    cases.append({"type": "euro_call", "sigma": sigma, "T": T, "K": K, "halfWidth": hw,
                  "blur": float(V[i]), "closedForm": float(bs_call(S0, K, sigma, T, r))})

sigma, T, K, steps = 0.2, 1.0, 100, 200
lo, hi, hw = grid(T, sigma)
S, V = american_put_blur(K, sigma, T, r, steps, lo, hi, dx)
i = int(np.argmin(abs(S - S0)))
cases.append({"type": "american_put", "sigma": sigma, "T": T, "K": K, "steps": steps,
              "halfWidth": hw, "blur": float(V[i]), "europeanClosedForm": float(bs_put(S0, K, sigma, T, r))})

out = {"S0": S0, "r": r, "dx": dx, "cases": cases}
Path(__file__).parent.parent.joinpath("src", "fixtures.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
