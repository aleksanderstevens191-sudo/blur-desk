# Launch facts

Every number for the launch thread, with what it is, where it was measured and where it comes from. Commit hashes are on `main`.

- **Mac**: Alek's Mac. Apple GPU (Metal-3) through WebGPU in Chrome 155, 14 CPU cores, QuantLib 1.44.
- **Cloud**: the build container, a 4-core 2.8 GHz Xeon with no GPU. WebGPU there runs on SwiftShader, a software GPU, in headless Chromium. GitHub Actions runs the same checks on every push.
- ⚠ **not on the Mac**: measured only in the cloud. Accuracy numbers that don't depend on the machine (64-bit math, the fits, the odds) carry the flag too. Rerun them on the Mac before quoting a Mac figure.

## Accuracy against Black-Scholes

| # | What | Value | Machine | Source |
| --- | --- | --- | --- | --- |
| A1 | Gaussian blur on the GPU vs the Black-Scholes formula, 14 strikes from 70 to 140 | **1.30e-5** max | Mac | `reference/kernel_gpu_results.txt` (3b98d65) |
| A2 | The demo's live error counter, call, vs Black-Scholes | within **1.4e-5** at 54.7 ms per reprice | Mac | Alek's report in chat, 2026-10-22. It is not in the repo, so rerun it before quoting |
| A3 | The demo's calls and puts vs Black-Scholes from T = 0.01 to 2 years and σ from 5% to 80% | within **1e-5** | ⚠ cloud (SwiftShader) | README "Notes on accuracy" |
| A4 | CPU reference, 20 Black-Scholes calls (T from 0.1 to 2, K from 80 to 120) | **$0.000013** max | ⚠ cloud (numpy, 64-bit) | `reference/results.txt` (75f9808) |
| A5 | GPU vs the Python reference, every fixture on `/check.html` | all under **1e-4** | Mac and cloud | `reference/kernel_gpu_results.txt`, `reference/chain_check_mac.txt`, `docs/check_mac_2026-10-24.txt` |
| A8 | GPU European calls vs Python, 3 fixtures, current main | **2.1e-7 to 1.4e-6** | Mac, 2026-10-24 | `docs/check_mac_2026-10-24.txt` |
| A9 | GPU barriers vs Python, 26 fixtures (weighted, plain, double, step) | worst **1.7e-6** | Mac, 2026-10-24 | `docs/check_mac_2026-10-24.txt` |
| A6 | JavaScript core vs Python | agree to 11 decimal places | ⚠ cloud | README "Results so far" |
| A7 | The recording's error counter: zigzag vs its closed form (a sum of calls), through the whole clip | **1.9e-6 to 1.4e-5** | ⚠ cloud (SwiftShader) | PR #9 (2ae40fe) |

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
| C7 | Demo reprice of the surface | 54.7 ms | Mac | Alek's report in chat, 2026-10-22 (see A2) |
| C8 | numpy CPU baseline, one blur over 10,001 points | **0.97M prices/sec** (10.34 ms) | ⚠ cloud | `reference/results.txt` |
| C9 | The same blur in plain JavaScript, 1 thread | 38,800 options/sec | Mac | `reference/gpu_barrier_bench_mac.txt` |

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

## Still to measure on the Mac

1. The demo's error counter and reprice time on current main (A2, C7), plus a run of `/?record=1`.
2. QuantLib barriers on all cores, if you want a barrier ratio against all cores (Q3).

Done on 2026-10-24: `/check.html?only=fixtures` (A8, A9, B4) and `/check.html?only=odds` (O1–O8), both all PASS.
