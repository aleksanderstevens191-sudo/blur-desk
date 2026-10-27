# Option pricing and diffusion models: what can be said

This is a wording audit for Part 5. Each sentence below is one you might post, and each gets one of three verdicts:

- **Exact**: a theorem. Where it can be checked, it was checked here in float64 and holds to roundoff.
- **Approximately true**: right in spirit, but only with a condition attached, or only to a stated accuracy. The condition has to go into the sentence.
- **False**: don't post it, in any wording.

Every number comes from `reference/diffusion_stage1_results.txt` (Stage 1) or `reference/diffusion_stage2_results.txt` (Stage 2). The market's density is the one in the market odds panel (panel 13): the SPY Merton fit to the Oct 22, 2026 quotes (`src/spy_fit.json`), read off its prices by Breeden-Litzenberger. Everything is in log-return, ln(S_T / S).

Short names for the sources (full references at the end):
**Song21** = Song et al. 2021 (score-based SDEs).
**Ho20** = Ho, Jain, Abbeel 2020 (DDPM).
**Efron11** = Efron 2011 (Tweedie's formula).
**Robbins56** = Robbins 1956.
**BS73** = Black & Scholes 1973.
**BL78** = Breeden & Litzenberger 1978.
**Karras22** = Karras et al. 2022.
**Anderson82** = Anderson 1982.

## A. The core analogy

| # | Sentence | Verdict | Why, and the check | Source |
|---|---|---|---|---|
| A1 | "An option's price is its payoff, blurred." | **Exact** under Black-Scholes | With constant volatility, the price is a Gaussian blur of the payoff in log-price, shifted by the drift (r − σ²/2)τ and multiplied by e^(−rτ). With the fitted Merton model the blur kernel is Merton's, not a Gaussian. Checked in Parts 1 and 2. | BS73; `reference/blur_pricer.py` |
| A2 | "The forward process of a diffusion model blurs a probability density." | **Exact** for the variance-exploding (VE) form | The density at noise level σ(t) is the data density convolved with N(0, σ²(t) − σ²(0)). A 1,000-step VE chain run on the SPY density matches one Gaussian blur of it (in closed form) to 6e-14. For DDPM, see B1. | Song21 §3.4, eq. 8–9 |
| A3 | "Pricing blurs a payoff; a diffusion model blurs a probability density." | **Exact** (Black-Scholes and VE) | This is A1 and A2 together. It's the safest one-line version. | BS73; Song21 |
| A4 | "Price = the payoff averaged against the blurred density." | **Exact** | This is the duality: the payoff blur is the transpose of the density blur. On the grid the two agree to $3e-13. By quadrature against the SPY Merton density, they agree with Merton's series prices to $7e-14. | Risk-neutral pricing; BS73 |
| A5 | "With constant volatility, both are Gaussian convolutions of the same heat equation." | **Exact** | Take out the drift (move with it) and the discount (multiply by e^(rτ)), and both become w_s = ½ w_zz. With zero drift, the two blur kernels are the same symmetric matrix (\|w_j − w_−j\| = 0). | BS73; Song21 eq. 9 |
| A6 | "…run in opposite directions." | **Approximately true**: say "opposite in calendar time" | The payoff goes from expiry back to today (the backward Kolmogorov equation). The density goes from the data toward noise (Fokker-Planck). The two equations are adjoints. In its own clock (σ²τ for the payoff, σ²(t) for the density), each one runs the heat equation **forward**: both blur, and neither un-blurs. | Song21 §3; BS73 |
| A7 | "Black-Scholes is the heat equation." | **Exact** after a change of variables | Log-price, a frame moving with the drift, and undiscounted value turn the Black-Scholes PDE into the heat equation. This is how BS73 solve it. | BS73 |
| A8 | "Volatility is the diffusion model's noise level." | **Approximately true** | It holds exactly with variances, not volatilities alone: the payoff's blur variance σ²τ plays the role of σ²(t) − σ²(0). With local volatility or jumps the blur is not one Gaussian. | Song21 eq. 9 |
| A9 | "Time to expiry is diffusion time." | **Approximately true** | The matching clock is the accumulated variance σ²τ, not calendar time. Calendar time matches only up to the factor σ². | |
| A10 | "Pricing an option is running a diffusion model in reverse." | **False** | A diffusion model's reverse run (generation) *removes* blur. Pricing *adds* blur. Pricing runs backward in calendar time, but it is a forward heat flow. | Song21 §3.2, eq. 6; Anderson82 |
| A11 | "Option pricing is denoising." | **False** | Same reason as A10. A price is smoother than its payoff. | |
| A12 | "This works for any volatility model." | **False** | It is the heat equation only with constant volatility (one Gaussian). Local volatility gives a variable-coefficient diffusion (Part 4). Jumps add a non-local term (Merton, Part 2). Duality (A4) does survive in every one of these models. | |

## B. DDPM's shrink step, drift and discounting

| # | Sentence | Verdict | Why, and the check | Source |
|---|---|---|---|---|
| B1 | "DDPM shrinks the data a little at every step before adding noise." | **Exact** | x_t = √(1 − β_t) x_(t−1) + √β_t ε, so x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε. In continuous time this is the VP SDE dx = −½β x dt + √β dw, an Ornstein-Uhlenbeck process. With 100,000 particles of the SPY density, KS p is 0.74, 0.45 and 0.17 at steps 100, 500 and 1000. | Ho20 eq. 2, 4; Song21 eq. 10–11 |
| B2 | "Divide DDPM by √ᾱ_t and you get the variance-exploding process." | **Exact** | The noise level becomes σ² = (1 − ᾱ_t)/ᾱ_t. The densities agree to 2e-15. | Song21 App. B; Karras22 Table 1 |
| B3 | "DDPM's shrink plays the role of Black-Scholes' drift." | **Approximately true** | Both are the first-order term of the SDE, and both disappear under a rescaling: divide by √ᾱ_t for DDPM, discount the stock for Black-Scholes (e^(−rt) S_t is a martingale). There are three differences. In price space, Black-Scholes *grows* at rate r while DDPM *shrinks*. The noise is multiplicative in S, not additive. In log-price, the Black-Scholes drift is a constant shift, not a shrink. | Ho20; Song21; BS73 |
| B4 | "DDPM's shrink is discounting." | **False** | Discounting multiplies the *value* of the option, not the coordinate. In density terms it is mass lost at rate r (Feynman-Kac killing). A diffusion model's density always keeps mass 1, so discounting has no counterpart. | |
| B5 | "DDPM is like a negative interest rate." | **Approximately true, loose** | DDPM's mean decays like e^(−½∫β) and a stock forward grows like e^(rT), so the linear form is the same with the opposite sign. Over Ho20's schedule the total shrink is √ᾱ_1000 = 0.0064, which would be a "rate" of −5.06 over the run. The noise is still additive rather than multiplicative, so treat this as a figure of speech. | Ho20 §4 |

## C. The sampler on the market's odds (Stage 2)

| # | Sentence | Verdict | Why, and the check | Source |
|---|---|---|---|---|
| C1 | "We ran a diffusion sampler on the market's odds, with no neural network." | **Exact** | The probability-flow ODE runs in reverse. Its score is taken from the grid alone: blur the grid density, take the log, take the slope. That score matches the closed form to 2e-5 relative or better. | Song21 §4.3, eq. 13 |
| C2 | "We trained a diffusion model on SPY." | **False** | Nothing was trained or learned. The density is known, so its score is computed directly. | |
| C3 | "The score is the slope of the log of the density." | **Exact** (definition) | | Hyvärinen 2005; Song21 §2 |
| C4 | "The probability-flow ODE has the same marginals as the diffusion." | **Exact**, given the exact score | | Song21 §4.3, App. D |
| C5 | "Starting from noise, the reverse run rebuilds the market's odds." | **Approximately true**: say what "noise" means | Starting from the market blurred to σ = 1: 100,000 samples give KS D 0.0024, p 0.62 against the market. Flat Black-Scholes is rejected (D 0.097). Starting from pure N(0, 1) noise, KS gives p 0.04, a small real mismatch, because the blurred market is 0.0039 from pure noise in L1. | Song21 §4.3 |
| C6 | "The reverse run recovers the market's density exactly." | **False** as worded | It recovers the density to 2.2e-5 in L1, with a CDF gap of at most 3e-6. The 10% fall/rise odds come back as 7.615% and 2.988%, the same as panel 13. Say "to 2e-5", not "exactly". | |
| C7 | "In one dimension the flow carries quantiles to quantiles." | **Exact** | The flow is monotone and keeps the CDF constant along paths. 200,000 quantile points land in order, within 3e-6 in CDF. | Song21 eq. 13 (1D case) |
| C8 | "Tweedie's formula: the denoised mean is the noisy point plus noise variance times the score." | **Exact** for Gaussian noise | E[x_0 \| x_t] = x_t + s² ∂ log p_t(x_t). Checked against a Monte Carlo conditional mean with 2,000,000 pairs at σ = 0.03, 0.1 and 0.3: within 2.5 standard errors in every bin. For DDPM's scaling it reads (x_t + (1 − ᾱ) ∂ log p_t)/√ᾱ. | Robbins56; Efron11 §2 |
| C9 | "Tweedie's formula denoises a sample." | **False** | It gives a posterior *mean*, the average over every x_0 that could have produced x_t. At large noise it is a smooth curve close to the overall mean, not a draw from the market. | Efron11 |
| C10 | "The diffusion sampler tells us something new about SPY." | **False** | It returns what went in: the density implied by the Merton fit. | |

## D. The market's odds

| # | Sentence | Verdict | Why, and the check | Source |
|---|---|---|---|---|
| D1 | "Options priced SPY's chance of falling 10% by Dec 18 at 7.6%." | **Approximately true**: say "market-implied" or "risk-neutral" | It is a risk-neutral probability, from a Merton fit to 10 strikes per expiry on Oct 22, 2026. It is not a real-world probability or a forecast. | BL78; Merton 1976 |
| D2 | "The density came straight from option prices." | **Approximately true** | It goes quotes → Merton fit → second difference in strike (Breeden-Litzenberger). Raw quotes are too noisy to difference twice. | BL78 |
| D3 | "The diffusion model predicts SPY." | **False** | Nothing here forecasts anything (C10, D1). | |

## Claims you must not make

1. "Option pricing is reverse diffusion / denoising / a generative model." Pricing blurs; generation un-blurs (A10, A11).
2. "The same heat equation run backwards in time" without "in calendar time". Running the heat equation backward is ill-posed, and that is not what pricing does (A6).
3. "DDPM's shrink is discounting." The shrink is the counterpart of drift; discounting has no counterpart (B3, B4).
4. "We trained a diffusion model on SPY" or "AI learned the market's odds." Nothing was trained (C2).
5. "The diffusion model predicts / forecasts SPY" or "real-world odds." The odds are risk-neutral and come from one day's option prices (C10, D1, D3).
6. "Recovers the market's odds exactly." Say to 2e-5 in L1 (C6).
7. "Pure noise in, the market's odds out, exactly." From pure N(0, 1) noise the KS p-value is 0.04 (C5).
8. "Diffusion models price options," or "faster option pricing with diffusion models." The sampler never prices anything; the blur does.
9. "This holds for any volatility model." Only the duality does (A12).
10. "Song et al. / Ho et al. showed a connection to option pricing." They didn't. The papers are cited for the diffusion side only.
11. "Tweedie's formula generates samples / denoises the market." It gives a posterior mean (C9).
12. "Everything matches to 1e-12." 1e-12 is the duality target ($3e-13 measured). The forward process matched to 6e-14 against a 1e-10 target. The sampler matched to 2e-5 in L1. Quote each number with what it measures.

## A paragraph that passes the audit

> Pricing an option blurs its payoff. The forward process of a diffusion model blurs a probability density. With constant volatility both are the same Gaussian heat flow, run in opposite directions in calendar time, and they are dual to each other: the price from the blurred payoff equals the payoff averaged against the blurred density (checked to $3e-13). Taking SPY's market-implied odds from Oct 22 option prices, I blurred them into noise and ran a diffusion sampler (the probability-flow ODE, with the score computed exactly, no network) back to the odds, to within 2e-5 in L1.

## References

- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. arXiv:2011.13456.
- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239.
- Efron, B. (2011). *Tweedie's Formula and Selection Bias.* Journal of the American Statistical Association 106(496), 1602–1614.
- Robbins, H. (1956). *An Empirical Bayes Approach to Statistics.* Proc. Third Berkeley Symposium 1, 157–163. (The formula is credited there to M. C. K. Tweedie.)
- Karras, T., Aittala, M., Aila, T., Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. arXiv:2206.00364.
- Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications 12(3), 313–326.
- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR 6, 695–709.
- Black, F., Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy 81(3), 637–654.
- Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous.* Journal of Financial Economics 3, 125–144.
- Breeden, D. T., Litzenberger, R. H. (1978). *Prices of State-Contingent Claims Implicit in Option Prices.* Journal of Business 51(4), 621–651.
