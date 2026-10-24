// The market's odds: the risk-neutral distribution of SPY at an expiry, read off
// the SPY-calibrated Merton prices by edge detection.
//
// The blur turns a payoff into a price; its second derivative in the strike turns
// the price back into the distribution (Breeden-Litzenberger):
//   P(S_T <= K) = 1 + e^{rT} dC/dK,   density = e^{rT} d2C/dK2.
// The prices are the smile panel's two GPU rows (smile.js): one call row and one
// put row struck at the forward give every strike F e^{o dx}. Puts become calls
// by parity, slopes between neighbouring strikes give the CDF at the midpoints,
// and the CDF's steps give each strike's probability. The steps telescope, so the
// total is the CDF's rise across the strikes (1 minus whatever lies outside them)
// and the mean is exactly what parity leaves in the prices.
//
// Prices come from the Merton fit to real quotes (spy_fit.json), not the raw
// quotes: second differences of noisy quotes amplify the noise, so smooth first.
import { smile } from "./smile.js";
import { drift, poissonTerms, halfWidth } from "./kernels.js";
import { normCdf } from "./bs.js";

// Strikes reach as far as the kernel: past them lies under 1e-12 of the probability
export const ODDS_REACH = { nSd: 7, tail: 1e-12 };
export const DROP = 0.1; // the read-off: P(SPY falls more than 10% by expiry)

// The Merton parameters for one SPY slice (T, r from the slice)
export const fitParams = (fit, sl) => ({ sigma: fit.sigma, lam: fit.lam, m: fit.m, delta: fit.delta, T: sl.T, r: sl.r });

// The distribution from one smile (S0 = 100 units). Returns strikes K, the
// probability at each strike (mass), the CDF at the midpoints between strikes
// (Kmid, cdf), and the density per unit of K at each strike.
export function oddsFromSmile(sm, T, r) {
  const { K, price, cp, F } = sm;
  const n = K.length;
  const disc = Math.exp(-r * T);
  const C = Float64Array.from(price, (v, i) => (cp[i] === "call" ? v : v + disc * (F - K[i])));
  const Kmid = new Float64Array(n - 1);
  const cdf = new Float64Array(n - 1);
  for (let i = 0; i + 1 < n; i++) {
    Kmid[i] = 0.5 * (K[i] + K[i + 1]);
    cdf[i] = 1 + ((C[i + 1] - C[i]) / (K[i + 1] - K[i])) / disc;
  }
  // strike i takes the CDF's rise between the midpoints either side of it; the
  // end strikes take what lies beyond the first and last midpoints
  const mass = new Float64Array(n);
  const density = new Float64Array(n);
  mass[0] = cdf[0];
  mass[n - 1] = 1 - cdf[n - 2];
  for (let i = 1; i + 1 < n; i++) {
    mass[i] = cdf[i] - cdf[i - 1];
    density[i] = mass[i] / (Kmid[i] - Kmid[i - 1]);
  }
  // inside the strikes only: the end strikes carry the tails beyond them
  let sum = 0;
  let mean = 0;
  for (let i = 1; i + 1 < n; i++) {
    sum += mass[i];
    mean += mass[i] * K[i];
  }
  return { K, Kmid, cdf, mass, density, sum, mean: mean / sum, F };
}

// P(S_T <= x), linear in the CDF between midpoints
export function cdfAt(o, x) {
  const { Kmid, cdf } = o;
  if (x <= Kmid[0]) return cdf[0];
  if (x >= Kmid.at(-1)) return cdf.at(-1);
  let a = 0;
  let b = Kmid.length - 1;
  while (b - a > 1) {
    const c = (a + b) >> 1;
    Kmid[c] <= x ? (a = c) : (b = c);
  }
  return cdf[a] + ((x - Kmid[a]) / (Kmid[b] - Kmid[a])) * (cdf[b] - cdf[a]);
}

// Merton's CDF in closed form, P(S_T <= x) for S_0 = S0: given n jumps, ln S_T is
// Gaussian. Independent of the blur and the edge detection; the check uses it.
export function mertonCdf(x, S0, p) {
  const mu = drift("merton", p);
  let P = 0;
  for (const [n, pn] of poissonTerms(p.lam * p.T)) {
    P += pn * normCdf((Math.log(x / S0) - mu - n * p.m) / Math.sqrt(p.sigma * p.sigma * p.T + n * p.delta * p.delta));
  }
  return P;
}

// Black-Scholes at one flat vol: ln S_T ~ N(ln S0 + (r - s^2/2) T, s^2 T)
export const flatCdf = (x, S0, sigma, T, r) => normCdf((Math.log(x / S0) - (r - 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T)));

// The market's odds for one SPY slice. spyFit: spy_fit.json; spot: SPY's last
// price (the read-offs are moves from it). Model units have the slice's spot (net
// of dividends, sl.S) at 100, so dollars = units * sl.S / 100.
export async function marketOdds(spyFit, e, { gpu = true } = {}) {
  const sl = spyFit.slices[e];
  const p = fitParams(spyFit.merton, sl);
  const kMax = halfWidth("merton", p, ODDS_REACH);
  const t0 = performance.now();
  // blurred directly: the second difference needs each price to its own 32-bit
  // precision, and the FFT's error scales with the largest call in the row
  const sm = await smile("merton", p, { kMax, gpu, minVega: Infinity, direct: true });
  const ms = performance.now() - t0;
  const o = oddsFromSmile(sm, p.T, p.r);
  const scale = sl.S / 100;
  const spot = spyFit.spot;
  const down = (1 - DROP) * spot;
  const up = (1 + DROP) * spot;
  const pDown = cdfAt(o, down / scale);
  const pUp = 1 - cdfAt(o, up / scale);
  return {
    ...o, sl, p, scale, spot, ms, strikes: sm.K.length, taps: sm.taps, gpu,
    meanUsd: o.mean * scale, forward: sl.F, down, up, pDown, pUp,
    exact: { pDown: mertonCdf(down / scale, 100, p), pUp: 1 - mertonCdf(up / scale, 100, p) },
    flat: {
      sigma: spyFit.flat.sigma,
      pDown: flatCdf(down / scale, 100, spyFit.flat.sigma, p.T, p.r),
      pUp: 1 - flatCdf(up / scale, 100, spyFit.flat.sigma, p.T, p.r),
    },
  };
}
