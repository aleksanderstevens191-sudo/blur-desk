"""Part 5, Stage 1: pricing blurs a payoff, a diffusion model's forward process blurs a density.

python reference/diffusion_stage1.py [plot.png]      (exits 1 if any check fails)

  1. the market's density: the closed form used here is panel 13's (its tail odds and Breeden-Litzenberger)
  2. duality: the price from blurring the payoff = the payoff averaged against the blurred density (1e-12)
  3. the same heat equation in opposite directions: with no drift the two blurs are the same matrix
  4. the VE forward process (Song et al. 2021) run on the market's density = one Gaussian blur of it (1e-10)
  5. DDPM's shrink (Ho et al. 2020) next to Black-Scholes' drift and discounting
"""
import sys
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import diffusion as D
from kernels import bs, merton_series

warnings.filterwarnings("ignore", message="The occurrence of roundoff")  # quad at 1e-14; the gaps are printed
fails = 0


def line(ok, s):
    global fails
    fails += not ok
    print(f"   {'PASS' if ok else 'FAIL'} {s}")


def note(s):
    print(f"        {s}")


def main():
    t00 = time.time()
    M = [D.market(0), D.market(1)]
    P = M[1]
    print(f"SPY {D.SPY_FIT['asof']}, last ${P['spot']}; Merton fit sigma {P['sigma']:.4f}, {P['lam']:.3f} jumps/yr, "
          f"mean jump {P['m']:.4f}, jump sd {P['delta']:.4f}; flat vol {P['flat']:.4f} (src/spy_fit.json)")
    print("Coordinate: log-return y = ln(S_T / S), S the slice's spot net of dividends (as in the pricing blur).\n")

    # ---------------------------------------------------------------- 1
    print("1. The market's density is panel 13's")
    panel13 = {"2026-11-20": (0.04287, 0.00175), "2026-12-18": (0.07615, 0.02988)}  # test-odds.mjs, 3 decimals of %
    for Q in M:
        c = D.mixture(Q)
        down = D.mix_cdf(np.log(0.9 * Q["spot"] / Q["S"]), c)
        up = 1 - D.mix_cdf(np.log(1.1 * Q["spot"] / Q["S"]), c)
        pd, pu = panel13[Q["expiry"]]
        line(abs(down - pd) < 1e-5 and abs(up - pu) < 1e-5,
             f"{Q['expiry']}: P(fall > 10%) {100 * down:.3f}%, P(rise > 10%) {100 * up:.3f}%; panel 13 (node src/test-odds.mjs) "
             f"{100 * pd:.3f}% / {100 * pu:.3f}%")
        # Breeden-Litzenberger on Merton's prices (the panel's method, here on the series prices)
        h, worst = 0.05, 0.0
        for K in (650.0, 700.0, 740.0, 770.0, 800.0, 830.0):
            C = [merton_series(Q["S"], k, Q["T"], Q["r"], Q["sigma"], Q["lam"], Q["m"], Q["delta"]) for k in (K - h, K, K + h)]
            bl = np.exp(Q["r"] * Q["T"]) * (C[0] - 2 * C[1] + C[2]) / h**2
            exact = D.mix_pdf(np.log(K / Q["S"]), c) / K
            worst = max(worst, abs(bl / exact - 1))
        line(worst < 1e-5, f"{Q['expiry']}: e^rT d2C/dK2 of Merton's series prices ($0.05 steps) vs the density, 6 strikes "
                           f"$650-$830: worst relative gap {worst:.1e} (< 1e-5)")

    # ---------------------------------------------------------------- 2
    print("\n2. Duality: blur the payoff, or blur the density and average the payoff against it")
    print("   continuous (quadrature of payoff x density vs closed-form prices, $ per option on SPY):")
    for Q in M:
        S, T, r = Q["S"], Q["T"], Q["r"]
        c = D.mixture(Q)
        for K, cp in ((770.0, "call"), (700.0, "put")):
            pay = (lambda y: np.maximum(S * np.exp(y) - K, 0)) if cp == "call" else (lambda y: np.maximum(K - S * np.exp(y), 0))
            k = np.log(K / S)
            lo, hi = (k, 3.0) if cp == "call" else (-3.0, k)
            avg = np.exp(-r * T) * quad(lambda y: pay(y) * D.mix_pdf(y, c), lo, hi, epsabs=1e-14, epsrel=1e-14, limit=400)[0]
            ser = merton_series(S, K, T, r, Q["sigma"], Q["lam"], Q["m"], Q["delta"], cp)
            line(abs(avg - ser) < 1e-12, f"{Q['expiry']} {cp} K {K:.0f} Merton: e^-rT E[f] ${avg:.12f} vs series "
                                         f"${ser:.12f}: |diff| {abs(avg - ser):.1e} (< 1e-12)")
        s = Q["flat"]
        mu = (r - 0.5 * s * s) * T
        avg = np.exp(-r * T) * quad(lambda y: np.maximum(S * np.exp(y) - 770, 0) * np.exp(-0.5 * ((y - mu) / s) ** 2 / T)
                                    / (s * np.sqrt(2 * np.pi * T)), np.log(770 / S), 3.0, epsabs=1e-14, epsrel=1e-14)[0]
        ref = bs(S, 770.0, s, T, r)
        line(abs(avg - ref) < 1e-12, f"{Q['expiry']} call K 770 Black-Scholes flat {s:.4f}: ${avg:.12f} vs closed form "
                                     f"${ref:.12f}: |diff| {abs(avg - ref):.1e}")

    print("   on the grid (the blur itself): a Dec 18 option valued on Nov 20 under flat vol, averaged over the")
    print("   market's Nov 20 odds; one side blurs the payoff over the 28 days, the other blurs the density")
    A, B = M
    tau, r, s = B["T"] - A["T"], B["r"], B["flat"]
    dx = 1e-4
    y = np.arange(-15000, 15001) * dx                     # ln(S_Nov / S), |y| <= 1.5
    p0 = D.mix_pdf(y, D.mixture(A)) * dx                  # the market's Nov 20 probabilities per node
    for drift_on in (True, False):
        mu = (r - 0.5 * s * s) * tau if drift_on else 0.0
        w = D.gauss_taps(dx, s * np.sqrt(tau), center=mu)
        disc = np.exp(-r * tau) if drift_on else 1.0
        for K, cp in ((770.0, "call"), (700.0, "put"), (800.0, "call")):
            St = A["S"] * np.exp(y)
            f = np.maximum(St - K, 0) if cp == "call" else np.maximum(K - St, 0)
            V = disc * D.blur_payoff(f, w)                 # payoff -> price at every node
            q = D.blur_density(p0, w)                      # Nov 20 odds -> Dec 18 odds
            left, right = float(p0 @ V), disc * float(q @ f)
            line(abs(left - right) < 1e-12, f"{'r, drift ' if drift_on else 'no drift, '}{cp} K {K:.0f}: sum p0 * blur(f) "
                                            f"${left:.13f}, e^-rt sum blur(p0) * f ${right:.13f}: |diff| {abs(left - right):.1e}")
    note(f"{len(y)} nodes, dx {dx}, {len(w)} taps; the payoff blur is the transpose of the density blur, so the two sums")
    note("are the same numbers added in a different order: the gap is roundoff.")

    # ---------------------------------------------------------------- 3
    print("\n3. The same heat equation, opposite directions")
    mu = (r - 0.5 * s * s) * tau
    w = D.gauss_taps(dx, s * np.sqrt(tau), center=mu)
    w0 = D.gauss_taps(dx, s * np.sqrt(tau), center=0.0)
    line(np.max(np.abs(w0 - w0[::-1])) == 0.0, f"no drift: the kernel is symmetric (max |w_j - w_-j| = "
                                              f"{np.max(np.abs(w0 - w0[::-1])):.0e}), so payoff blur = density blur, the same matrix")
    asym = np.max(np.abs(w - w[::-1]))
    q = D.blur_density(p0, w)
    shift = float(q @ y) / q.sum() - float(p0 @ y) / p0.sum()
    line(abs(shift - mu) < 1e-12, f"with drift mu tau = {mu:+.3e}: the density's mean moves by {shift:+.3e} (forward in time)")
    lin = D.blur_payoff(y, w)                              # undiscounted price of the payoff f(y) = y
    mid = len(y) // 2
    line(abs(lin[mid] - mu) < 1e-12, f"... and the payoff is read at x + mu tau: blur of f(y) = y at x = 0 is {lin[mid]:+.3e} "
                                     f"(the payoff's picture moves the other way, by -mu tau)")
    note(f"(kernel asymmetry with drift {asym:.1e}; in the frame moving with mu, both are w_s = w_zz / 2)")
    note("Payoff: backward Kolmogorov, expiry -> today, plus discounting. Density: Fokker-Planck, data -> noise.")
    note("Each runs the heat equation forward in its own clock (each blurs); neither un-blurs.")

    # ---------------------------------------------------------------- 4
    print("\n4. The VE forward process on the market's Dec 18 density")
    smin, smax, N = 0.01, 1.0, 1000
    dx = 5e-4
    y = np.arange(-16000, 16001) * dx                    # |y| <= 8 = 8 sigma_max
    c0 = D.mixture(P)
    p0 = D.mix_pdf(y, c0)
    t0 = time.time()
    keep = (N // 4, N // 2, 3 * N // 4, N)
    pN, snaps, sig = D.ve_chain(p0, dx, smin, smax, N, keep=keep)
    ms = time.time() - t0
    print(f"   sigma(t) = {smin} * ({smax}/{smin})^t (Song et al. 2021 VE, in log-return units; the data's sd is "
          f"{np.sqrt(sum(w_ * (v + m * m) for w_, m, v in c0) - sum(w_ * m for w_, m, _ in c0) ** 2):.4f}),")
    print(f"   {N} chain steps x_i = x_(i-1) + sqrt(sigma_i^2 - sigma_(i-1)^2) z, each a sampled-Gaussian blur of the grid")
    print(f"   density ({len(y)} nodes, dx {dx}, {ms:.1f} s); reference: the market's Merton mixture with sigma^2(t) - sigma^2(0)")
    print("   added to every component's variance (the exact Gaussian blur of the density, closed form)")
    for i in keep:
        ex = D.mix_pdf(y, D.mixture(P, extra=sig[i] ** 2 - sig[0] ** 2))
        err = np.max(np.abs(snaps[i] - ex))
        line(err < 1e-10, f"t = {i / N:.2f}, sigma {sig[i]:.4f}: max |chain - blur| {err:.1e} per unit of y "
                          f"(peak {ex.max():.3f}; < 1e-10), mass {snaps[i].sum() * dx:.12f}")
    one = D.blur_density(p0, D.gauss_taps(dx, np.sqrt(sig[N] ** 2 - sig[0] ** 2)))
    line(np.max(np.abs(one - pN)) < 1e-10, f"one blur by sqrt(sigma_max^2 - sigma_min^2) vs the {N}-step chain: "
                                           f"max |diff| {np.max(np.abs(one - pN)):.1e}")
    rng = np.random.default_rng(5)
    x = D.mix_sample(c0, 100_000, rng)
    for i in range(1, N + 1):
        x = x + np.sqrt(sig[i] ** 2 - sig[i - 1] ** 2) * rng.standard_normal(len(x))
        if i == N // 2:
            ks = kstest(x, lambda v: D.mix_cdf(v, D.mixture(P, extra=sig[i] ** 2 - sig[0] ** 2)))
            line(ks.pvalue > 0.01, f"100,000 particles through the same chain, t = 0.50: KS vs the blur D {ks.statistic:.4f}, "
                                   f"p {ks.pvalue:.2f}")
    ks = kstest(x, lambda v: D.mix_cdf(v, D.mixture(P, extra=sig[N] ** 2 - sig[0] ** 2)))
    line(ks.pvalue > 0.01, f"t = 1.00: KS D {ks.statistic:.4f}, p {ks.pvalue:.2f}")
    prior = np.exp(-0.5 * (y / smax) ** 2) / (smax * np.sqrt(2 * np.pi))
    l1 = float(np.sum(np.abs(pN - prior)) * dx)
    note(f"at t = 1 the blurred market is {l1:.4f} in L1 from the pure-noise prior N(0, {smax}^2) (2 = disjoint);")
    note("the market's skew and jumps are still in it, just small. Stage 2 starts the reverse run there.")

    # ---------------------------------------------------------------- 5
    print("\n5. DDPM's shrink vs Black-Scholes' drift and discounting")
    betas = D.ddpm_betas()
    abar = np.cumprod(1 - betas)
    print(f"   DDPM (Ho et al. 2020): x_t = sqrt(1 - beta_t) x_(t-1) + sqrt(beta_t) eps, beta linear {betas[0]:g}..{betas[-1]:g},"
          f" {len(betas)} steps")
    rng = np.random.default_rng(7)
    x0 = D.mix_sample(c0, 100_000, rng)
    _, snaps = D.ddpm_particles(x0, betas, rng, keep=(100, 500, 1000))
    for i, xi in snaps.items():
        a = np.sqrt(abar[i - 1])
        ks = kstest(xi, lambda v: D.mix_cdf(v, D.mixture(P, a=a, extra=1 - a * a)))
        line(ks.pvalue > 0.01, f"step {i}: particles vs closed form (every component's mean x sqrt(abar), var -> abar var + 1 - abar):"
                               f" KS D {ks.statistic:.4f}, p {ks.pvalue:.2f}; sqrt(abar) {a:.4f}")
    worst = 0.0
    yy = np.linspace(-3, 3, 2001)
    for i in (10, 100, 500):
        a2 = abar[i - 1]
        ddpm = D.mix_pdf(yy, D.mixture(P, a=np.sqrt(a2), extra=1 - a2))
        ve = D.mix_pdf(yy / np.sqrt(a2), D.mixture(P, extra=(1 - a2) / a2)) / np.sqrt(a2)
        worst = max(worst, np.max(np.abs(ddpm - ve)))
    line(worst < 1e-12, f"DDPM divided by sqrt(abar_t) is the VE process with sigma^2 = (1 - abar)/abar: densities agree "
                        f"to {worst:.0e} (steps 10, 100, 500)")
    T, r = P["T"], P["r"]
    print("   Side by side (Dec 18 slice: r {:.2%}, T {:.4f}):".format(r, T))
    rows = [
        ("", "Black-Scholes (pricing)", "DDPM / VP SDE (image models)"),
        ("noise", "sigma dW on ln S (times S on S)", "sqrt(beta) dw on x, additive"),
        ("first-order term", f"ln S: + (r - s^2/2) dt, a shift", "- beta/2 x dt, a shrink toward 0"),
        ("", f"S: + r S dt, growth e^rT = {np.exp(r * T):.5f}", f"mean x sqrt(abar_T) = {np.sqrt(abar[-1]):.4f}"),
        ("remove it by", "discounting S: e^-rt S_t is a martingale", "rescaling: x_t / sqrt(abar_t) is VE"),
        ("discount e^-rT", f"multiplies the value ({np.exp(-r * T):.5f}),", "no counterpart: density mass stays 1"),
        ("", "not the coordinate (killing at rate r)", ""),
        ("equation run", "backward Kolmogorov on the payoff", "Fokker-Planck on the density"),
    ]
    for a_, b_, c_ in rows:
        print(f"   {a_:<17}{b_:<43}{c_}")
    note("The shrink is DDPM's drift, not its discount: in price space Black-Scholes' drift r S is the same linear form")
    note(f"with the opposite sign (growth at r; DDPM shrinks at beta/2, from {betas[0] / 2:.0e} to {betas[-1] / 2:.0e} per step,")
    note(f"a total factor {np.sqrt(abar[-1]):.4f}, like a rate of -{-np.log(np.sqrt(abar[-1])):.2f} over the run).")
    note("In log-price Black-Scholes' drift is a constant shift, not a shrink; the noise is multiplicative in S.")
    note("Discounting has no diffusion-model analog: it scales the payoff's value; a density keeps mass 1.")

    print(f"\n{'all Stage 1 checks pass' if not fails else f'{fails} failed'} ({time.time() - t00:.0f} s)")
    if len(sys.argv) > 1:
        plot(sys.argv[1], P)
    return fails


def plot(path, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    mono = dict(fontsize=9, family="monospace")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    S, T, r, s = P["S"], P["T"], P["r"], P["flat"]
    yy = np.linspace(-0.25, 0.2, 901)
    K = 770.0
    for f_, a_ in ((0.0, 1.0), (0.1, 0.8), (0.3, 0.6), (1.0, 0.45), (3.0, 0.3)):
        tau = f_ * T
        Sx = S * np.exp(yy)
        if tau == 0:
            v = np.maximum(Sx - K, 0)
        else:
            v = bs(Sx, K, s, tau, r)
        ax[0].plot(Sx, v, color="white", alpha=a_, lw=1.2, label=f"tau {tau * 365:.0f} d" if tau else "payoff")
    ax[0].set_title("pricing: the payoff melts into a price (call K 770, flat vol)", **mono)
    ax[0].set_xlabel("SPY $", **mono)
    ax[0].legend(prop={"family": "monospace", "size": 8})
    yy = np.linspace(-1.2, 1.2, 2401)
    for t, a_ in ((0.0, 1.0), (0.25, 0.8), (0.5, 0.6), (0.75, 0.45), (1.0, 0.3)):
        sg = 0.01 * 100 ** t
        ax[1].plot(yy, D.mix_pdf(yy, D.mixture(P, extra=sg * sg - 1e-4)), color="white", alpha=a_, lw=1.2,
                   label=f"t {t:.2f}, sigma {sg:.3f}")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-4, 30)
    ax[1].set_title("diffusion (VE): the market's Dec 18 density melts into noise", **mono)
    ax[1].set_xlabel("log-return ln(S_T/S)", **mono)
    ax[1].legend(prop={"family": "monospace", "size": 8})
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot: {path}")


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
