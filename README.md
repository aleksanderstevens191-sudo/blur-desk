# Blur Desk

[![checks](https://github.com/aleksanderstevens191-sudo/blur-desk/actions/workflows/checks.yml/badge.svg)](https://github.com/aleksanderstevens191-sudo/blur-desk/actions/workflows/checks.yml)

The blur filter in a photo app can price stock options.

An option's price is its payoff, blurred. Blurring an image and pricing an option solve the same equation (the heat equation), so the same Gaussian blur that softens a photo turns a sharp payoff into a smooth price curve.

- **Bachelier model:** a plain Gaussian blur of the payoff.
- **Black-Scholes:** the same blur in log-price, with the blur nudged slightly off-center by the interest-rate drift, then discounted.
- **American options:** blur one small time step, discount, clip to the payoff, repeat.
- **The market's odds:** run the edge-detect kernel `[1, -2, 1]` across a row of call prices and out comes the probability distribution the market is pricing (Breeden–Litzenberger, 1978).

## Results so far (CPU reference)

All numbers come from `reference/check.py`; full output is in `reference/results.txt`.

| Check | Result |
| --- | --- |
| Black-Scholes calls, 20 options (T = 0.1 to 2 years, strikes 80 to 120) | max error **$0.000013** vs the closed form |
| Bachelier calls | error shrinks with the grid, down to $0.0000002 |
| American put (S = K = 100, σ = 20%, r = 5%, 1 year) | 6.090368 vs 6.090333 from a 20,000-step binomial tree (**$0.000035** apart) |
| Edge-detect density (3-month, strikes $0.50 apart) | max error 0.0000085 vs the true density; sums to 0.99996 |
| CPU speed, one blur over a 10,001-point grid | ~0.9M option prices/sec (numpy); this is the GPU baseline |
| JavaScript core vs Python | identical to 11 decimal places |

## Barrier options

A barrier option is a call or put that dies (knock-out) or comes alive (knock-in) the first time the stock touches a level H. There are eight single-barrier types (down or up, in or out, call or put) plus the double knock-out, which dies at either of two levels L < U. A knock-in is always the vanilla minus the knock-out, so the hard part is the knock-out.

Every number below is against the closed form for a barrier watched continuously: Reiner-Rubinstein for one barrier, which agrees with QuantLib to 5.5e-12, and the exact image series for two, which agrees with QuantLib to 1.4e-8 (a 40-digit check puts that gap on QuantLib's side).

| Method | Blurs | Error, down-and-out call K=100, H=90, σ=20%, r=5%, T=1 |
| --- | --- | --- |
| (a) plain mirror: payoff minus its reflection across H | 1 | **+$0.63** (exact only when r = σ²/2) |
| (b) weighted mirror: the reflection times (H/S)^(2ν/σ²) | 1 | **1.2e-9** |
| (c) step by step, monitored N = 25 / 100 / 400 times | N | +$0.69 / +$0.38 / +$0.20 |
| (c) plus the BGK shift, N = 25 / 100 / 400 | N | −$0.0032 / −$0.00020 / −$0.000025 |
| double knock-out, one blur of the image series (72 options) | 1 | max 5.9e-9 |

### Why the plain mirror fails

The reflection principle: for a random walk with no drift, every path that touches the barrier and ends at x has a twin, mirrored after the first touch, that ends at 2h − x (h = ln H). Subtract the payoff's mirror image and the paths that touched cancel out, which is the knock-out price.

In log price the stock drifts at ν = r − q − σ²/2, and drift breaks the symmetry: a path is more likely to go with the drift than against it, so the twin is not equally likely. Weighting the image by e^(2ν(h−x)/σ²) = (H/S)^(2ν/σ²) restores the balance. Without the weight the mirror is right only when ν = 0, that is r − q = σ²/2. At σ = 20% that means r = 2%, where the plain mirror is exact (1.2e-9); at r = 5% it is off by up to $1.59 across the eight types. With the weight, one blur prices all eight types to within 1.3e-9 of the closed form.

Two barriers make two mirrors facing each other: a hall of mirrors. The payoff is tiled across cells of width ln(U/L), alternately reflected and shifted, each with its weight, and one blur of the tiled payoff is the double knock-out price. The number of images needed for 1e-8 is about 11 × σ√T / ln(U/L).

### Why step by step overprices

The other way to price a knock-out: blur one small time step, zero everything past the barrier, repeat. That checks the barrier only on the N monitoring dates. A path can cross the barrier and come back between two dates and the method never sees it, so too few paths die. Every knock-out comes out too high and every knock-in too low (table above), and the gap shrinks only like 1/√N, because a path can wander about σ√dt past the barrier between checks.

### What the shift fixes

Broadie, Glasserman and Kou (1997) showed that a barrier checked on N dates behaves like a continuous barrier moved away from the spot by β σ√dt, with β = −ζ(1/2)/√(2π) ≈ 0.5826. Run that backwards: to get the continuous price out of the step-by-step blur, check a barrier moved toward the spot by 0.5826 σ√dt. The shift cancels the 1/√N term. What is left falls like 1/N when the payoff is still worth something at the barrier (up-and-out call, K < H) and like N^(−3/2) when it is zero there (down-and-out call, K > H):

| Error vs continuous, σ=20%, r=5%, T=1 | slope, no shift | slope, BGK shift | N = 1280, no shift | N = 1280, BGK shift |
| --- | --- | --- | --- | --- |
| up-and-out call K=100, H=120 | −0.50 | −0.99 | $0.067 | $0.0010 |
| down-and-out call K=100, H=90 | −0.46 | −1.48 | $0.11 | $0.0000045 |

![Discrete-monitoring error against the number of steps, with and without the BGK shift](docs/monitoring_convergence.png)

### On the GPU

- All three methods run as WebGPU compute shaders (`src/gpu.js`). `src/barrier.js` is the same code in 64-bit JavaScript and matches Python to about 1e-12 (`node src/test-barriers.mjs`).
- `/check.html` runs 26 barrier fixtures from `reference/make_barrier_fixtures.py` on the GPU in 32-bit floats: worst |gpu − Python| is 1.8e-6 (tolerance 1e-4).
- `/bench.html` prices all 8 types × 32 barriers × 64 strikes = 16,384 options in one dispatch, worst error 8.1e-6 against the closed form. The CPU baseline is QuantLib's analytic engine on the same options, about 305,000 options/sec on one thread (`python3 reference/bench_quantlib.py`). GPU throughput depends on the GPU; run `/bench.html` to measure yours.
- The demo has a barrier panel (one or two barriers, knock-out or knock-in), a method switch (plain, weighted, step by step with or without the shift), the knocked-out region shaded on the surface, an animated hall of mirrors, and a live error against the closed form.
- Near the barrier the Greeks get violent: half a trading day out, an up-and-out call's gamma is 19 times a vanilla's ([plot](docs/barrier_greeks.png)).

Full outputs: `reference/barrier_results.txt`, `double_barrier_results.txt`, `greeks_monitoring_results.txt`, `gpu_barrier_results.txt`.

## Layout

```
reference/blur_pricer.py   ground-truth pricers + closed forms (Python)
reference/check.py         every accuracy check and the CPU speed baseline
reference/make_fixtures.py writes src/fixtures.json for the JS test
src/blur.js                browser core: kernel, blur, European, American, edge detect
src/test.mjs               JS vs Python check
src/gpu.js                 blurGPU / americanGPU: the blur as a WebGPU compute shader (WGSL)
src/check.js, check.html   fixture page: runs the fixtures on the GPU, prints |gpu - py|
src/pricer.js              demo pricing: samples a drawn payoff on the log grid, blurs it on the GPU
src/bs.js                  Black-Scholes closed forms for the demo's live error counter
src/demo.js, index.html    the demo: draw a payoff, move expiry and volatility, watch the price surface
src/scene.js, src/panels.js, src/wire.js  the 3D wireframe views (plain canvas 2D, no libraries)

reference/barriers.py              single barriers: closed forms, plain and weighted mirror, step by step
reference/double_barrier.py        double knock-out: image series, one blur, closed form
reference/barrier_greeks.py        delta and gamma off the blurred surface
reference/check_barriers.py, check_double_barrier.py, check_greeks_monitoring.py  the barrier checks
reference/make_barrier_fixtures.py writes src/barrier_fixtures.json
reference/bench_quantlib.py        QuantLib CPU baseline for the GPU benchmark
src/barrier.js                     barriers in 64-bit JS: closed forms, the CPU twin of the shaders
src/test-barriers.mjs              JS barrier core vs Python
src/bench.js, bench.html           one-dispatch GPU benchmark
.github/workflows/checks.yml       CI: every Python check, both JS tests and the build, on every push
```

## Run it

```bash
pip install -r requirements.txt
python3 reference/check.py
python3 reference/check_barriers.py
python3 reference/check_double_barrier.py
python3 reference/check_greeks_monitoring.py
node src/test.mjs
node src/test-barriers.mjs

npm install
npm run dev        # open the printed URL in a WebGPU browser (Chrome, Edge, Safari 26)
                   # /            the interactive demo
                   # /check.html  the GPU vs Python fixture check
                   # /bench.html  the barrier benchmark
```

## The rest of the week

- [x] Day 1: blur pricer on CPU, Bachelier error table
- [x] Day 2: Black-Scholes as an off-center blur; American puts vs a binomial tree; JS core
- [x] Day 3: port `blur` to a WebGPU compute shader (WGSL), check it against `src/fixtures.json`
- [ ] Day 4: draw-any-payoff canvas + time-to-expiry slider (blur radius = σ√T)
- [ ] Day 5: real SPY option chain; smooth the volatility smile, then edge-detect for the market's odds
- [ ] Day 6: benchmark GPU vs JS vs numpy; record the 15-second demo
- [ ] Day 7: deploy, publish the repo, post the launch thread

## Notes on accuracy

- Payoffs are extended past the grid edges as straight lines (odd reflection). Padding with a constant bent the tails and left a fixed $0.00014 error that never shrank; this fix made the error converge.
- The drift enters by centering the kernel at μ = (r − σ²/2)T, so no interpolation is needed.
- The American price uses a fixed number of exercise dates, so it converges like 1/steps; two runs combined (Richardson) close most of the remaining gap.
- On an Apple Mac (Metal-3) the fixture page passes, but the American put lands 4.97e-5 from Python, against 8.2e-7 on SwiftShader. European calls stay near 1e-5. The likely cause is the Metal driver compiling with fast-math, which reorders the Kahan summation and cancels its correction.
- The demo averages the payoff against a cubic (Keys) kernel at each grid point instead of reading it at the point. Reading it at the point leaves the call/put kink costing up to 4e-3 at short expiries; averaging keeps the demo's calls and puts within 1e-5 of Black-Scholes from T = 0.01 to 2 years and σ = 5% to 80% (measured on SwiftShader).
- Real option prices are noisy, so edge-detecting raw quotes amplifies noise. Fit a smooth volatility curve first, convert it back to prices, then edge-detect.
