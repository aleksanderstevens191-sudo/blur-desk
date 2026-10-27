# Launch facts

Every number for the launch thread, with what it is, where it was measured and where it comes from. Commit hashes are on `main`.

- **Mac**: Alek's Mac. Apple GPU (Metal-3) through WebGPU in Chrome 155, 14 CPU cores, QuantLib 1.44.
- **Cloud**: the build container, a 4-core 2.8 GHz Xeon with no GPU. WebGPU there runs on SwiftShader, a software GPU, in headless Chromium. GitHub Actions runs the same checks on every push.
- ⚠ **not on the Mac**: measured only in the cloud. Accuracy numbers that don't depend on the machine (64-bit math, the fits, the odds) carry the flag too. Rerun them on the Mac before quoting a Mac figure.

## Accuracy against Black-Scholes

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| A1 | Gaussian blur on the GPU vs the Black-Scholes formula, 14 strikes from 70 to 140 | **1.30e-5** max | Mac | `reference/kernel_gpu_results.txt` (3b98d65) |
| A2 | The demo's live error counter, call at 30% vol and 1 year (the page's defaults), vs Black-Scholes | **1.4e-5** | Mac, 2026-10-24 (blur-desk.vercel.app) | `docs/demo_mac_2026-10-24.png`, from Alek's screen recording |
| A3 | The demo's calls and puts vs Black-Scholes from T = 0.01 to 2 years and σ from 5% to 80% | within **1e-5** | ⚠ cloud (SwiftShader) | README "Notes on accuracy" |
| A4 | CPU reference, 20 Black-Scholes calls (T from 0.1 to 2, K from 80 to 120) | **$0.000013** max | ⚠ cloud (numpy, 64-bit) | `reference/results.txt` (75f9808) |
| A5 | GPU vs the Python reference, every fixture on `/check.html` | all under **1e-4** | Mac and cloud | `reference/kernel_gpu_results.txt`, `reference/chain_check_mac.txt`, `docs/check_mac_2026-10-24.txt` |
| A8 | GPU European calls vs Python, 3 fixtures, current main | **2.1e-7 to 1.4e-6** | Mac, 2026-10-24 | `docs/check_mac_2026-10-24.txt` |
| A9 | GPU barriers vs Python, 26 fixtures (weighted, plain, double, step) | worst **1.7e-6** | Mac, 2026-10-24 | `docs/check_mac_2026-10-24.txt` |
| A10 | The demo's error counter through Alek's screen recording: calls from 5% to 67% vol, expiries of 1 and 0.31 years, and a Merton kernel with −10% jumps | **9.0e-6 to 3.2e-5**, green the whole time (under 1e-4) | Mac, 2026-10-24 | Alek's screen recording (not in the repo) |
| A6 | JavaScript core vs Python | agree to 11 decimal places | ⚠ cloud | README "Results so far" |
| A7 | The 15-second clip's error counter: the zigzag vs its closed form (a sum of calls) while it is drawn and blurred, then vs the Merton series under the SPY fit | **1.9e-6 to 1.8e-5**, then 1.8e-6, green throughout (cloud: 1.9e-6 to 1.4e-5) | Mac, 2026-10-24 | `docs/blur-desk-15s.mp4` |

## The American put against the tree

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| B1 | American put (S = K = 100, σ = 20%, r = 5%, 1 year): blur with Richardson vs a 20,000-step binomial tree | 6.090368 vs 6.090333, **$0.000035** apart | ⚠ cloud (numpy) | `reference/results.txt` (75f9808) |
| B2 | Early-exercise premium in that case | $0.5168 | ⚠ cloud | `reference/results.txt` |
| B3 | GPU American put vs Python, before blocked summation | 4.97e-5 | Mac | commit 4e0f3f3 message, README |
| B4 | GPU American put vs Python, after blocked summation (200 steps) | **6.34e-6** (6.087394 vs 6.087400), 30 ms | Mac, 2026-10-24 | `docs/check_mac_2026-10-24.txt` |

B4 is the current code. It matches the 6.3e-6 the f32 emulation predicted in 4e0f3f3, so blocked summation fixed what fast-math broke (B3 → B4: 8x closer).

## GPU speed

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| C1 | 16,384 barrier options in one dispatch, submit to done (median of 20) | 2.10 ms = **7.8M options/sec**, worst error 9.25e-6 | Mac | `reference/gpu_barrier_bench_mac.txt` (e92a25b) |
| C2 | The same, including reading the prices back (median of 5) | 1.70 ms = 9.6M/sec | Mac | same file |
| C3 | Earlier barrier run, same page | 10.24M/sec | Mac | `reference/gpu_barrier_results.txt` (ce20a8a) |
| C4 | Whole SPY chain, 4,738 contracts, Black-Scholes at each quote's own vol, GPU compute only | 0.40 ms = **11.8M contracts/sec** | Mac | `reference/chain_bench_results.txt` (a836fd1) |
| C5 | Whole chain, Merton per expiry, GPU compute only | 1.20 ms = 3.95M/sec | Mac | same |
| C6 | Whole chain end to end, including the JS kernel build (BS / Merton) | 27.6 ms / 33.3 ms | Mac | same |
| C7 | Demo reprice of the surface (the top bar's ENGINE), call at 30% vol, 1 year | **39.3 ms** (54.7 ms on 2026-10-22) | Mac, 2026-10-24 | `docs/demo_mac_2026-10-24.png` |
| C8 | numpy CPU baseline, one blur over 10,001 points | **0.97M prices/sec** (10.34 ms) | ⚠ cloud | `reference/results.txt` |
| C9 | The same blur in plain JavaScript, 1 thread | 38,800 options/sec | Mac | `reference/gpu_barrier_bench_mac.txt` |
| C10 | The demo's reprice at other settings: 5% vol, 1 year / Merton kernel, 0.31 years | 2.1 ms / 23.8 ms | Mac, 2026-10-24 | Alek's screen recording (not in the repo) |

On the Mac, browser timers tick in 0.1 ms steps, so C1 is good to about 5% and C4 to about 25%.

## QuantLib ratios (both sides on the same Mac, in one session)

| # | What | Value | Source |
| --- | --- | --- | --- |
| Q1 | Barriers: GPU vs QuantLib `AnalyticBarrierEngine` on **1 thread**, repricing (651,000/sec) | **12x** | `reference/bench_quantlib_mac.txt` (269a857), `reference/gpu_barrier_bench_mac.txt` |
| Q2 | Barriers: vs QuantLib building the instruments and pricing them (192,000/sec) | 41x | same |
| Q3 | Barriers: vs QuantLib on **all cores** | ⚠ **never measured** | |
| Q4 | Chain, Black-Scholes, GPU compute vs QuantLib, 1 thread / 14 cores | **22x / 4.2x** | `reference/chain_bench_results.txt` (a836fd1) |
| Q5 | Chain, Merton, GPU compute vs QuantLib (Bates, vol of vol 1e-5), 1 thread / 14 cores | **142x / 38x** | same |
| Q6 | Chain end to end, Black-Scholes, vs 1 thread / 14 cores | 0.3x / 0.06x (the GPU loses) | same |
| Q7 | Chain end to end, Merton, vs 1 thread / 14 cores | 5.1x / 1.4x | same |

Don't quote "~34x". That figure divided a Mac GPU run by QuantLib on the cloud Xeon (305,000/sec, `reference/gpu_barrier_results.txt`).

## FFT speed-up (FFT blur vs direct blur, both on the GPU)

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| F1 | Gaussian, T = 1, 5,341 taps | 133.9 ms → 1.35 ms, **100x**, 778M prices/sec | Mac | `reference/fft_bench_apple.json` (fba44b1), README |
| F2 | Variance gamma, T = 1, 7,087 taps | 168.8 ms → 0.89 ms, **189x**, 1.18B prices/sec | Mac | same |
| F3 | Merton SPY fit, Dec 18, 38,289 taps | 921.1 ms → 2.32 ms, **398x**, 463M prices/sec | Mac | same |
| F4 | Width at which the FFT starts to win | 29 to 90 taps | Mac | same |
| F5 | FFT vs direct near the money | within 4.6e-5 | Mac | `reference/fft_results.txt` (36074fa) |

## The SPY fit

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| S1 | Snapshot: SPY 2026-10-22 close, Nov 20 + Dec 18, 10 OTM mids each | close $763.96 | data | `data/spy_chain_2026-10-22.csv`, `src/spy_fit.json` (8a52f0e) |
| S2 | Merton fit | σ 9.57%, λ 1.67 a year, mean jump −9.46%, jump sd 4.98% | ⚠ cloud (CI reproduces it) | `reference/spy_calibration_results.txt` (8a52f0e) |
| S3 | In sample, RMS: Merton vs flat vol (14.58%) | **0.30 vs 1.72 vol pts**, 5.7x better (max 0.53 vs 3.58) | ⚠ cloud | same |
| S4 | Market's mean bid-ask half-spread | 0.04 vol pts | ⚠ cloud | same |
| S5 | Whole chain, Oct 23, 4,738 contracts / 32 expiries, Merton per expiry: RMS over all OTM quotes | **1.86 vs 16.20** vol pts (flat) | ⚠ cloud | `reference/chain_fit_results.txt` (5786dd0) |
| S6 | Out of sample, near the money (\|ln K/F\| ≤ 4%): Merton vs flat, six tests | 0.70–1.86 vs 1.53–2.61 vol pts. Merton wins every test, at 32–80% of flat vol's error | ⚠ cloud | `reference/chain_oos_results.txt` (5786dd0) |
| S7 | Out of sample, best case: Oct 22 fit → Oct 23 Dec 18 | **0.70 vs 2.19** vol pts | ⚠ cloud | same |
| S8 | Out of sample, all OTM strikes: Oct 22's 10-strike fit → Oct 23 | 14.7 / 13.9 vol pts (it never saw the wings) | ⚠ cloud | same |
| S9 | GPU vs QuantLib / Merton series, whole chain on the Mac | 7.1e-6 / 4.7e-6 dollars | Mac | `reference/chain_check_mac.txt` (a836fd1) |
| S10 | Kernels check on the Mac, worst case | 4.0e-5, vols within 0.004 pts | Mac | `reference/kernel_gpu_results.txt` (3b98d65) |

## The market's odds (panel 13, risk-neutral)

Inputs: the SPY Merton fit (S2) and the last price $763.96. A 10% fall means below $687.56, and a 10% rise means above $840.36.

| # | What | Nov 20 | Dec 18 | Machine | Source |
| --- | --- | ---: | ---: | --- | --- |
| O1 | Market-implied P(SPY falls more than 10%) | **4.3%** (4.287%) | **7.6%** (7.615%) | Mac, 2026-10-24 (cloud agrees) | `docs/check_mac_2026-10-24.txt`, `src/odds.js` (4e80f9e) |
| O2 | The same from one flat vol (14.6%) | 0.4% | 2.9% | Mac | same |
| O3 | P(rises more than 10%) | 0.2% | 3.0% | Mac | same |
| O4 | Distribution sums to | 1.000000 | 1.000000 | Mac | same |
| O5 | Mean vs the parity forward | $768.18 = $768.18 | $768.11 = $768.11 | Mac | same |
| O6 | Tail odds vs Merton's closed-form CDF (GPU) | 2.5e-8 | 1.2e-6 | Mac | same |
| O7 | GPU CDF at every strike vs the same blur in 64-bit | 5.8e-6 | 7.5e-6 | Mac | same |
| O8 | GPU time for the distribution (2 rows, every strike) | 60 ms (5,237 strikes) | 16 ms (4,071 strikes) | Mac | same |

On the Dec 18 panel the market-implied density peaks near $785. It sits above the forward because a fat left tail pulls the mean down. These are risk-neutral odds, which include the price of bearing the risk, so they are not a forecast.

Panel 13 on blur-desk.vercel.app showed the same Dec 18 read-offs on the Mac (7.6%, 2.9%, 3.0%, sums to 1.0000, mean $768.11 = forward $768.11), in 37 ms, and in 17 ms in the clip.

## Two stocks (Part 3, panel 14)

Commits are on branch `part3-2d` until it merges. Market for the panel and the sweep: S1 = S2 = 100, vols 30% / 25%, 1 year, r = 5%.

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| T1 | Exchange option, 2D blur vs Margrabe, 24 cases (ρ −0.9 to 0.95) | max **2.6e-8** | ⚠ cloud (numpy, 64-bit) | `reference/blur2d_results.txt` (ab28520) |
| T2 | Call on the better of two stocks vs Stulz, 72 cases | max **5.1e-8** | ⚠ cloud | same |
| T3 | ρ = 0: product payoff vs product of two 1D prices, 16 cases | max 2.2e-11 | ⚠ cloud | same |
| T4 | vs QuantLib 1.44 Margrabe and Stulz engines, 60 random markets × exchange, best-of, worst-of | max **1.4e-7** | ⚠ cloud | same |
| T5 | GPU vs Python, 10 fixtures × 25 prices, direct / 2D FFT | worst **1.33e-5 / 1.63e-5** | Mac, 2026-10-24 | `reference/twostock_gpu_mac.txt` (146393d) |
| T6 | The same 2D FFT without the numeraire split | 8.1e-4 (fails 1e-4) | Mac | same |
| T7 | Wraparound test: padded leak / one FFT size short | ≤ 8.5e-8 / 7.9e-3 to 0.1, all 6 caught | Mac | same |
| T8 | Where the 2D FFT starts beating direct | **41 to 54 taps per side** (patches of 63 to 511 spots per side) | Mac | `reference/bench2d_apple.json` (146393d) |
| T9 | Peak 2D FFT, 255 × 255 spots, 257 × 257 taps, GPU compute only | 1.26 ms = **51.6M prices/sec** (direct 2.3M/sec, 22.5x) | Mac | same |
| T10 | Fixture grid (233 × 233 taps, T = 1), 65 × 65 spots: direct / FFT / FFT end to end | 1.02M/sec / **3.45M/sec** / 19.2 ms | Mac | same |
| T11 | Exchange value from ρ = −0.9 to +0.9, falling at every step | **21.14 → 5.27** (4.0x), blur vs Margrabe ≤ 2.7e-8 | ⚠ cloud | `reference/intuition_2d_results.txt` (a1d80b7), `docs/exchange_vs_rho.png` |
| T12 | Kirk vs the exact spread, ρ = 0.95: K = 10 / K = 50 | **+1.3% / +98%** too high | ⚠ cloud | same, `docs/kirk_error.png` |
| T13 | Kirk passes 1% of the price, ρ = 0.9 / when stock 2 is the more volatile (25% / 50%) | K = 17.5 / K = 7.5 to 15 at every ρ | ⚠ cloud | same |
| T14 | Demo panel 14, 11 states: GPU at (100, 100) vs Margrabe / Stulz / exact spread | ≤ **7.9e-6**; GPU vs 64-bit blur ≤ 1.2e-5 | ⚠ cloud (SwiftShader) | `src/test-demo-2d.mjs` (CI) |
| T15 | Demo panel 14's ellipse at ρ = +0.9 / −0.9 | +39° / −39° | any (geometry) | `docs/demo_two_stocks.png` |

## Local volatility (Part 4, panel 15)

Commits are on branch `part4-localvol` until it merges. The Oct 23 2026 SPY chain: 32 expiries, 3 to 819 days, 3,320 out-of-the-money quotes; "liquid" means a mid of $0.50 or more (2,457 quotes). Vol points are implied-vol percentage points.

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| L1 | SVI surface, free fit: slices with butterfly arbitrage / neighbouring pairs with calendar arbitrage | 25 of 32 / 16 of 31, almost all past the last quote | ⚠ cloud (numpy) | `reference/svi_surface_results.txt` (9ed5e37) |
| L2 | After the constrained refit | **0 / 0**; RMS 0.52 vol pts on all quotes (free fit 0.45), 0.25 on quotes worth 50c+ | ⚠ cloud (CI rechecks the arbitrage) | same |
| L3 | Gatheral's total-variance Dupire vs Dupire on prices (finite differences), 1,874 reliable points | max relative difference 3.7e-4 | ⚠ cloud | `reference/local_vol_results.txt` (9ba2263) |
| L4 | Regularization rejected: filling flagged cells with the nearest reliable vol | moved quoted options up to **1.66 vol pts**, so raw Dupire held in [2%, 300%] is used instead | ⚠ cloud | `reference/check_lv_blur_results.txt` (9ba2263) |
| L5 | Variable-width blur at a constant vol vs Black-Scholes | **$6.5e-11** | ⚠ cloud (numpy, 64-bit) | same |
| L6 | Round trip: 2,457 liquid quotes priced under local vol, back to implied vol, vs the surface | max **0.073**, RMS 0.017 vol pts | ⚠ cloud | same, `docs/lv_roundtrip.png` |
| L7 | vs QuantLib 1.44's local-vol finite-difference engine, 39 contracts | max **0.039 vol pts** ($0.022) | ⚠ cloud | same |
| L8 | Convergence in the time step | observed order 1.2 to 1.8; 1 day is within 0.034 vol pts of half a day | ⚠ cloud | same |
| L9 | Model risk, 98-day up-and-out call (H 115%, K 100%): Black-Scholes / Merton / local vol | $17.42 / $21.16 / **$25.11** | ⚠ cloud (CI reproduces it) | `reference/model_risk_results.txt` (a849e5f) |
| L10 | 98-day down-and-out put (H 90%, K 100%) | $8.35 / $7.90 / **$3.57** | ⚠ cloud | same |
| L11 | Widest gap between the three models, down-and-out call H 90% K 80%: 98 / 342 days | **$16.51 / $31.24** | ⚠ cloud | same, `docs/model_risk.png` |
| L12 | Merton and local vol on the plain vanillas behind L9–L10 | within 1.5 cents at 98 days | ⚠ cloud | same |
| L13 | Barrier checks: vs Reiner-Rubinstein / Monte Carlo (400k paths) / QuantLib's local-vol barrier engine | $2.6e-4 / within about 1 se / $0.022 | ⚠ cloud | same |
| L14 | The 64-bit JS step vs Python, 32 fixtures | worst 2.3e-10 | ⚠ cloud | `src/test-lv.mjs` (CI) |
| L15 | The WGSL step vs Python, per $100 of forward: 12 flat-vol / 20 local-vol prices, 3 to 342 days | worst **8.1e-6 / 7.5e-6** | ⚠ cloud (SwiftShader). Mac before the fix: 1.44e-4 / 5.7e-5, 31 of 32 PASS | `/check.html?only=lv`, `src/test-gpu.mjs lv` (CI) |
| L16 | Demo panel 15, live GPU round trip at 3, 7, 28 and 98 days (16 liquid strikes each) | worst **0.071 vol pts**; GPU vs Python on the same grid ≤ 0.0001 | ⚠ cloud (SwiftShader) | `src/test-demo-lv.mjs` (CI) |
| L17 | The same at 819 days (1,434 steps) | GPU vs Python 0.0048 vol pts | ⚠ cloud (SwiftShader) | `src/test-demo-lv.mjs` run by hand |
| L18 | Reprice time for panel 15 on a real GPU | ⚠ **never measured** | | |

## Still to measure on the Mac

1. QuantLib barriers on all cores, if you want a barrier ratio against all cores (Q3).
2. Panel 14 on the Mac (T14): the live errors and the reprice time. Open the demo, scroll to 14 / TWO STOCKS and read the numbers off the panel.
3. Local vol on the Mac (L15, L16, L18): `/check.html?only=lv` (all PASS, and the worst |gpu − py|), then panel 15 / LOCAL VOL at a few expiries, reading the round trip, the GPU vs Python gap and the milliseconds off the panel.

Done on 2026-10-24: `/check.html?only=fixtures` (A8, A9, B4) and `/check.html?only=odds` (O1–O8), both all PASS, the demo's top bar and error counter (A2, A10, C7, C10) from Alek's screen recording, and the 15-second clip from `/?record=1` (A7, `docs/blur-desk-15s.mp4`).
