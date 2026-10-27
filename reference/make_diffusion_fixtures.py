"""Part 5 Stage 4: fixtures for panel 16 on the GPU (src/gpu-diffusion.js) and its 64-bit JS twin (src/diffusion.js).

python reference/make_diffusion_fixtures.py           -> src/diffusion_fixtures.json
python reference/make_diffusion_fixtures.py --check   (CI) recompute and compare with the committed file, 1e-10

Everything is panel 16's algorithm (diffusion.PANEL, the end of diffusion.py) in float64, on the market's Dec 18
density from src/spy_fit.json:
  density   the blurred density and its score at 6 of the 129 levels, every 20th node
  payoff    the blurred call (r = 0, K $770, per $100 of forward) at those levels, 12 spots
  sampler   2,048 quantiles of the density blurred to sigma_max, run back to sigma_min: where they land at
            level 64 and level 0
  quality   the same run on 20,000 quantiles, against the market's CDF (the run's own accuracy, not the GPU's)
"""
import sys
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import diffusion as D

C = D.PANEL
P = D.market(C["e"])
LEVELS = [0, 16, 32, 64, 96, 128]
STRIDE = 20
K_USD = 770.0


def compare(a, b):
    """Largest absolute difference between two JSON trees of the same shape (inf if the shapes differ)."""
    if isinstance(a, dict):
        return max((compare(a[k], b.get(k)) for k in a), default=0.0) if isinstance(b, dict) and a.keys() == b.keys() else np.inf
    if isinstance(a, list):
        return max((compare(x, y) for x, y in zip(a, b)), default=0.0) if isinstance(b, list) and len(a) == len(b) else np.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    return 0.0 if a == b else np.inf


def main():
    y, p0, dp = D.panel_grid(P)
    sig, s = D.panel_levels()
    Q, SC = D.panel_tables(p0, dp, s)
    c0 = D.mixture(P)
    K = 100 * K_USD / P["F"]
    xs = np.round(np.linspace(-0.3, 0.25, 12), 4)
    idx = np.arange(0, len(y), STRIDE)
    dens = [dict(j=j, sigma=sig[j], s=s[j], q=Q[j, idx].tolist(), score=SC[j, idx].tolist()) for j in LEVELS]
    pay = [dict(j=j, s=s[j], price=D.panel_payoff(xs, s[j], K).tolist()) for j in LEVELS]
    u = (np.arange(2048) + 0.5) / 2048
    c1 = D.mixture(P, extra=C["smax"] ** 2 - C["smin"] ** 2)
    x1 = D.mix_ppf(u, c1)
    run = D.panel_sampler(x1, y, SC, sig, keep=(64,))
    M = 20000
    uq = (np.arange(M) + 0.5) / M
    xq = D.panel_sampler(D.mix_ppf(uq, c1), y, SC, sig)[0]
    gap = float(np.max(np.abs(D.mix_cdf(xq, c0) - uq)))
    l1 = float(np.sum(np.abs(np.diff(uq) - D.mix_pdf(0.5 * (xq[1:] + xq[:-1]), c0) * np.diff(xq))))
    out = dict(
        about="reference/make_diffusion_fixtures.py: panel 16's algorithm in float64 (diffusion.PANEL)",
        settings=C, expiry=P["expiry"], K=K, K_usd=K_USD, F=P["F"], S=P["S"], stride=STRIDE,
        density=dens, payoff=dict(x=xs.tolist(), rows=pay),
        sampler=dict(x1=x1.tolist(), level64=run[64].tolist(), x0=run[0].tolist()),
        quality=dict(M=M, cdf_gap=gap, l1=l1),
    )
    path = HERE.parent / "src" / "diffusion_fixtures.json"
    if "--check" in sys.argv:
        worst = compare(json.loads(path.read_text()), json.loads(json.dumps(out)))
        ok = worst < 1e-10
        print(f"{'PASS' if ok else 'FAIL'} {path.name} recomputed: worst difference {worst:.1e} (< 1e-10); "
              f"the run's own accuracy: CDF gap {gap:.1e}, L1 {l1:.1e}")
        sys.exit(0 if ok else 1)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{path.name}: {path.stat().st_size / 1e3:.0f} kB; levels {LEVELS}, {len(idx)} nodes each, {len(xs)} spots, "
          f"2,048 sampler points; the run's own accuracy on {M:,} quantiles: CDF gap {gap:.1e}, L1 {l1:.1e}")


if __name__ == "__main__":
    main()
