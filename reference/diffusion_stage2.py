"""Part 5, Stage 2: a diffusion sampler run on the market's odds, with the exact score and no network.

python reference/diffusion_stage2.py [plot.png]      (exits 1 if any check fails)

The market's Dec 18 density (panel 13's, Stage 1) on a grid is blurred forward by the VE process to
sigma_max, then the probability-flow ODE (Song et al. 2021, eq. 13) runs in reverse,
    dx/dt = -g(t)^2 / 2 * d/dx log p_t(x),
with the score taken from the grid: the grid density blurred to each noise level, its log, central
differences (diffusion.GridScore). Checks:
  1. the score on the grid vs the closed form
  2. the reverse run of a deterministic quantile ladder: it lands on the market's quantiles; L1 of the
     recovered density
  3. 100,000 samples of the blurred density, run in reverse: KS test against the market's CDF (and
     against flat Black-Scholes, to show the test can tell them apart)
  4. Tweedie's formula, E[x_0 | x_t] = x_t + s^2 d/dx log p_t(x_t), vs a Monte Carlo conditional mean
"""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import kstest, norm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import diffusion as D

SMIN, SMAX, NODE = 0.01, 1.0, 400        # the Stage 1 schedule; RK4 steps in ln sigma
DX, HALF = 5e-4, 16000                   # grid: |y| <= 8 = 8 sigma_max
fails = 0


def line(ok, s):
    global fails
    fails += not ok
    print(f"   {'PASS' if ok else 'FAIL'} {s}")


def note(s):
    print(f"        {s}")


def main():
    t00 = time.time()
    P = D.market(1)
    c0 = D.mixture(P)
    y = np.arange(-HALF, HALF + 1) * DX
    p0 = D.mix_pdf(y, c0)                # the grid density; everything below sees only this array
    score = D.GridScore(y, p0, SMIN)
    exact_score = lambda x, s: D.mix_score(x, D.mixture(P, extra=s * s - SMIN * SMIN))
    F = lambda x: D.mix_cdf(x, c0)
    print(f"SPY Dec 18 (T {P['T']:.4f}), the market's density from panel 13's Merton fit, on {len(y)} nodes, dx {DX}.")
    print(f"Forward: VE, sigma {SMIN} -> {SMAX} (Stage 1). Reverse: probability-flow ODE, RK4, {NODE} steps in ln sigma,")
    print("score = central differences of log(grid density blurred to sigma), interpolated linearly.\n")

    # ---------------------------------------------------------------- 1
    print("1. The score from the grid vs the closed form")
    for s in (SMIN, 0.03, 0.1, 0.3, SMAX):
        c = D.mixture(P, extra=s * s - SMIN * SMIN)
        lo, hi = D.mix_ppf(np.array([1e-6, 1 - 1e-6]), c)
        x = np.linspace(lo, hi, 4001)
        g, e = score(x, s), exact_score(x, s)
        rel = np.max(np.abs(g - e)) / np.max(np.abs(e))
        line(rel < 1e-4, f"sigma {s:<5}: max |grid - exact| {np.max(np.abs(g - e)):.1e} over the middle 1 - 2e-6 of the "
                         f"mass (largest score {np.max(np.abs(e)):.0f}; relative {rel:.1e} < 1e-4)")

    # ---------------------------------------------------------------- 2
    print("\n2. The reverse run on a quantile ladder (deterministic)")
    M = 200_000
    u = (np.arange(M) + 0.5) / M
    c1 = D.mixture(P, extra=SMAX**2 - SMIN**2)
    x1 = D.mix_ppf(u, c1)                # the blurred density's quantiles at sigma_max
    t0 = time.time()
    x0 = D.pf_ode(x1, score, SMIN, SMAX, NODE)
    ms = time.time() - t0
    x0e = D.pf_ode(x1, exact_score, SMIN, SMAX, NODE)
    note(f"{M:,} points from the blurred density's quantiles u_i = (i + 1/2)/M, run back to sigma_min ({ms:.1f} s)")
    order = np.all(np.diff(x0) > 0)
    line(order, "the flow keeps the points in order (in 1D the flow carries quantiles to quantiles)")
    kd = np.max(np.abs(F(x0) - u))
    line(kd < 1e-5, f"the market's CDF at the landed points vs u_i: max gap {kd:.1e} (< 1e-5; the Kolmogorov distance of the "
                    f"recovered odds)")
    line(np.max(np.abs(x0 - x0e)) < 1e-5, f"grid score vs closed-form score, same ODE: points land within {np.max(np.abs(x0 - x0e)):.1e}"
                                          f" of each other (< 1e-5 in log-return)")
    dxs = np.diff(x0)
    mid = 0.5 * (x0[1:] + x0[:-1])
    du = np.diff(u)
    l1 = np.sum(np.abs(du - D.mix_pdf(mid, c0) * dxs)) + abs(F(x0[0]) - 0.5 / M) + abs(1 - F(x0[-1]) - 0.5 / M)
    line(l1 < 1e-3, f"L1 between the recovered density (1/M of mass between landed neighbours) and the market's: {l1:.1e} "
                    f"(< 1e-3; 2 = disjoint)")
    l1_start = np.sum(np.abs(np.diff(u) - D.mix_pdf(0.5 * (x1[1:] + x1[:-1]), c0) * np.diff(x1)))
    note(f"(the same L1 at the start, the blurred density vs the market: {l1_start:.3f})")
    for lab, k in (("fall > 10%", np.log(0.9 * P["spot"] / P["S"])), ("rise > 10%", np.log(1.1 * P["spot"] / P["S"]))):
        rec = np.mean(x0 < k) if "fall" in lab else np.mean(x0 > k)
        ex = F(k) if "fall" in lab else 1 - F(k)
        line(abs(rec - ex) < 1e-4, f"P(SPY {lab} by Dec 18): recovered {100 * rec:.3f}%, market {100 * ex:.3f}% (panel 13)")

    # ---------------------------------------------------------------- 3
    print("\n3. 100,000 random samples, run in reverse")
    rng = np.random.default_rng(2026)
    xs1 = D.mix_sample(c1, 100_000, rng)          # the blurred market at sigma_max: (very nearly) noise
    xs0 = D.pf_ode(xs1, score, SMIN, SMAX, NODE)
    ks = kstest(xs0, F)
    line(ks.pvalue > 0.01, f"KS vs the market's CDF: D {ks.statistic:.4f}, p {ks.pvalue:.2f} (p > 0.01)")
    flat = P["flat"]
    ksf = kstest(xs0, lambda v: norm.cdf(v, (P["r"] - 0.5 * flat**2) * P["T"], flat * np.sqrt(P["T"])))
    line(ksf.pvalue < 1e-6, f"KS vs flat Black-Scholes ({flat:.2%}): D {ksf.statistic:.4f}, p {ksf.pvalue:.0e} "
                            f"(the test tells the market from flat vol)")
    ks1 = kstest(xs1, F)
    note(f"the starting samples vs the market: D {ks1.statistic:.3f} (they are noise until the reverse run)")
    edges = np.linspace(-0.6, 0.35, 191)
    w = edges[1] - edges[0]
    mass = np.diff(F(edges))
    h_rev = np.histogram(xs0, edges)[0] / len(xs0)
    h_exa = np.histogram(D.mix_sample(c0, 100_000, rng), edges)[0] / 100_000
    note(f"histogram L1 ({len(edges) - 1} bins of {w:.3f}): reverse-run samples {np.sum(np.abs(h_rev - mass)):.4f}, "
         f"100,000 exact draws from the market {np.sum(np.abs(h_exa - mass)):.4f} (sampling noise, the same size)")
    xn = SMAX * rng.standard_normal(100_000)        # pure noise N(0, sigma_max^2), what an image model starts from
    xn0 = D.pf_ode(xn, score, SMIN, SMAX, NODE)
    ksn = kstest(xn0, F)
    note(f"starting from pure noise N(0, {SMAX}^2) instead (Stage 1: {SMAX}-blurred market is 0.0039 from it in L1):"
         f" KS D {ksn.statistic:.4f}, p {ksn.pvalue:.2f}")

    # ---------------------------------------------------------------- 4
    print("\n4. Tweedie's formula: E[x_0 | x_t] = x_t + s^2 score(x_t), s^2 = sigma^2 - sigma_min^2")
    rng = np.random.default_rng(11)
    n = 2_000_000
    a = D.mix_sample(c0, n, rng)
    for sg in (0.03, 0.1, 0.3):
        s2 = sg * sg - SMIN * SMIN
        xt = a + np.sqrt(s2) * rng.standard_normal(n)
        tw = xt + s2 * score(xt, sg)
        twe = D.tweedie_exact(xt, c0, s2)
        b = np.quantile(xt, np.linspace(0, 1, 41))
        idx = np.clip(np.searchsorted(b, xt) - 1, 0, 39)
        cnt = np.bincount(idx, minlength=40)
        mc = np.bincount(idx, a, 40) / cnt              # Monte Carlo: average x_0 in each bin of x_t
        tb = np.bincount(idx, tw, 40) / cnt             # Tweedie, averaged over the same samples
        res = a - tw
        se = np.sqrt(np.bincount(idx, res * res, 40) / cnt / cnt)
        z = (mc - tb) / se
        chi2 = float(np.sum(z * z))
        line(np.max(np.abs(z)) < 4.5 and chi2 < 80,
             f"sigma {sg}: 40 bins of x_t (50,000 pairs each): max |MC - Tweedie| {np.max(np.abs(mc - tb)):.1e}, worst "
             f"{np.max(np.abs(z)):.1f} standard errors, chi^2 {chi2:.0f} on 40")
        line(np.max(np.abs(tw - twe)) < 1e-5, f"sigma {sg}: Tweedie from the grid score vs the closed-form posterior mean: "
                                               f"max |diff| {np.max(np.abs(tw - twe)):.1e} over {n:,} points")
    note("Tweedie's denoised mean is an average over every x_0 that could have produced x_t: at sigma 0.3 it is a")
    note("smooth curve, not a sample. The sampler (sections 2-3) reaches the market's odds by many small steps instead.")

    print(f"\n{'all Stage 2 checks pass' if not fails else f'{fails} failed'} ({time.time() - t00:.0f} s)")
    if len(sys.argv) > 1:
        plot(sys.argv[1], P, c0, x0, xs0, score)
    return fails


def plot(path, P, c0, x0, xs0, score):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    mono = dict(fontsize=9, family="monospace")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    yy = np.linspace(-0.6, 0.35, 1201)
    ax[0].plot(yy, D.mix_pdf(yy, c0), color="white", lw=1.5, label="market (panel 13)")
    M = len(x0)
    k = 50
    xm = 0.5 * (x0[k:] + x0[:-k])
    ax[0].plot(xm, k / M / (x0[k:] - x0[:-k]), color="#5fd7ff", lw=1, ls="--", label="reverse run, quantile ladder")
    ax[0].hist(xs0, bins=190, range=(-0.6, 0.35), density=True, histtype="step", color="#ffaf00", lw=0.8,
               label="reverse run, 100,000 samples")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-3, 20)
    ax[0].set_title("noise -> the market's Dec 18 odds", **mono)
    ax[0].set_xlabel("log-return ln(S_T/S)", **mono)
    ax[0].legend(prop={"family": "monospace", "size": 8})
    rng = np.random.default_rng(3)
    c1 = D.mixture(P, extra=SMAX**2 - SMIN**2)
    starts = np.sort(D.mix_sample(c1, 60, rng))
    s = np.linspace(np.log(SMAX), np.log(SMIN), 121)
    paths = [starts]
    for i in range(120):
        paths.append(D.pf_ode(paths[-1], score, np.exp(s[i + 1]), np.exp(s[i]), 3))
    paths = np.array(paths)
    for j in range(paths.shape[1]):
        ax[1].plot(np.exp(s), paths[:, j], color="white", lw=0.6, alpha=0.7)
    ax[1].set_xscale("log")
    ax[1].invert_xaxis()
    ax[1].set_ylim(-2.5, 2.5)
    ax[1].set_title("probability-flow paths, sigma 1 -> 0.01 (grid score)", **mono)
    ax[1].set_xlabel("noise level sigma", **mono)
    xt = np.linspace(-0.6, 0.4, 801)
    for sg, col in ((0.03, "#ffffff"), (0.1, "#5fd7ff"), (0.3, "#ffaf00")):
        s2 = sg * sg - SMIN * SMIN
        ax[2].plot(xt, xt + s2 * score(xt, sg), color=col, lw=1.2, label=f"Tweedie, sigma {sg}")
    ax[2].plot(xt, xt, color="gray", lw=0.6, ls=":")
    ax[2].set_title("denoised mean E[x_0 | x_t]", **mono)
    ax[2].set_xlabel("noisy x_t", **mono)
    ax[2].legend(prop={"family": "monospace", "size": 8})
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"plot: {path}")


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
