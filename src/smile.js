// The smile: price out-of-the-money options across strikes with a fat-tailed
// kernel on the GPU and read the prices back as Black-Scholes implied vols.
//
// One blurred row gives every strike. Prices are homogeneous,
// C(S e^u, K) = e^u C(S, K e^-u), so the blur of ONE call payoff with strike F
// (the forward) on a log grid, read at the spot S0 e^u, is e^u times the call
// at S0 struck at F e^-u. Node c - o of the row is the option struck at
// F e^{o dx}: log-moneyness k = o dx. A put row does the same below the money.
// Two rows, one blur, the whole smile.
import { gridFor, kernel, kinkCorrection, impliedVol, meanSd } from "./kernels.js";
import { fftBlurGPU, directBlurGPU } from "./gpu-fft.js";
import { fftBlurCPU } from "./fft.js";
import { blur } from "./blur.js";

// The demo's kernels: 40 nodes per sd instead of kernels.py's 200, tails cut at
// 10 sd and 1e-15 instead of 13 sd and 1e-18, never wider than +/-12 in ln S.
// test-kernels.mjs holds these prices to 1e-5 of Python's.
export const DEMO_GRID = { ptsPerSd: 40, nSd: 10, tail: 1e-15, maxHalf: 12 };

// Kernel width (taps) from which the FFT blur is used. Stage 3 put the
// crossover at 30-90 taps on an Apple GPU; 256 leaves room for slower GPUs.
export const FFT_FROM = 256;

// Blur `rows` rows of n values (row-major) with one kernel w, grid spacing dx.
export async function blurRows(F, rows, n, w, dx, gpu = true) {
  const wide = w.length > FFT_FROM;
  if (gpu) return wide ? fftBlurGPU(F, rows, n, w, { dx }) : directBlurGPU(F, rows, n, w);
  if (wide) return fftBlurCPU(F, rows, n, w, { dx });
  const out = new Float64Array(rows * n);
  for (let r = 0; r < rows; r++) out.set(blur(F.subarray(r * n, (r + 1) * n), w), r * n);
  return out;
}

const PHI = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

// The smile of `model` with parameters p ({ T, r, sigma, ... }) at strikes
// F e^{o dx}, o = -omax..omax. grid: { y, dx, J } and optionally w, else the
// demo grid; omax: from kMax (largest |log-moneyness|), else 2.5 sd of the kernel.
// Returns { k, K, cp, price, iv } arrays (k ascending) plus the grid used.
// Implied vols are NaN where the price is outside the no-arbitrage bounds or the
// option's vega is under minVega (a 32-bit price there can't pin a volatility).
export async function smile(model, p, { grid = null, omax = null, kMax = null, S0 = 100, gpu = true, minVega = 0.05 } = {}) {
  grid ??= gridFor(model, p, DEMO_GRID);
  const { y, dx, J } = grid;
  const w = grid.w ?? kernel(model, y, p);
  const { T, r } = p;
  omax ??= Math.ceil((kMax ?? 2.5 * meanSd(model, p)[1]) / dx);
  const n = 2 * (J + omax) + 1;
  const cen = J + omax;
  const F = S0 * Math.exp(r * T);
  const z = Float64Array.from({ length: n }, (_, i) => Math.log(S0) + (i - cen) * dx);
  const rowsF = new Float64Array(2 * n);
  rowsF.set(kinkCorrection(z, F, "call"), 0);
  rowsF.set(kinkCorrection(z, F, "put"), n);
  const t0 = performance.now();
  const out = await blurRows(rowsF, 2, n, w, dx, gpu);
  const ms = performance.now() - t0;
  const len = 2 * omax + 1;
  const res = {
    k: new Float64Array(len), K: new Float64Array(len), cp: [], price: new Float64Array(len), iv: new Float64Array(len),
    dx, J, omax, F, w, y, ms, taps: w.length, n,
  };
  const disc = Math.exp(-r * T);
  const sT = Math.sqrt(T);
  for (let o = -omax; o <= omax; o++) {
    const idx = o + omax;
    const cp = o >= 0 ? "call" : "put";
    const K = F * Math.exp(o * dx);
    const v = disc * Math.exp(o * dx) * out[(cp === "call" ? 0 : n) + cen - o];
    let iv = impliedVol(v, S0, K, T, r, cp);
    if (iv > 0) {
      const d1 = (Math.log(S0 / K) + (r + 0.5 * iv * iv) * T) / (iv * sT);
      if (S0 * PHI(d1) * sT < minVega) iv = NaN;
    }
    res.k[idx] = o * dx;
    res.K[idx] = K;
    res.cp.push(cp);
    res.price[idx] = v;
    res.iv[idx] = iv;
  }
  return res;
}

// Implied vol at log-moneyness k, linear between the smile's nodes (NaN outside)
export function ivAt(s, k) {
  const u = k / s.dx + s.omax;
  const i = Math.floor(u);
  if (i < 0 || i + 1 >= s.iv.length) return NaN;
  return s.iv[i] + (u - i) * (s.iv[i + 1] - s.iv[i]);
}
