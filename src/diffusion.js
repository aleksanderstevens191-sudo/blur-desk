// Panel 16's math in 64-bit: pricing blurs a payoff, a diffusion model blurs a density
// (Part 5). The GPU runs the same steps in 32-bit (gpu-diffusion.js); the Python twin
// is the end of reference/diffusion.py (PANEL), which writes diffusion_fixtures.json.
//   density  the market's Dec 18 odds (panel 13's Merton fit) in log-return y = ln(S_T/S),
//            on a grid, blurred by the variance-exploding forward process: at noise level
//            sigma the blur's sd is sqrt(sigma^2 - sigma_min^2) (Song et al. 2021)
//   score    the slope of the log of the blurred density, as (blur of Dp0) / (blur of p0)
//   sampler  the probability-flow ODE dx/d(ln sigma) = -sigma^2 score, Heun's method, from
//            sigma_max back to sigma_min: noise back into the market's odds
//   payoff   a call blurred by the same Gaussian: Black-Scholes at r = 0, on the forward
import { poissonTerms } from "./kernels.js";
import { normCdf } from "./bs.js";

export const PANEL = { dx: 0.002, half: 3000, N: 128, smin: 0.01, smax: 1.0, nsd: 9, per: 256, e: 1 };

// The SPY slice e of spy_fit.json with its Merton parameters
export function market(fit, e = PANEL.e) {
  const sl = fit.slices[e], mp = fit.merton;
  return { expiry: sl.expiry, T: sl.T, r: sl.r, S: sl.S, F: sl.F, spot: fit.spot, flat: fit.flat.sigma, ...mp };
}

// Merton's log-return as a Poisson mix of Gaussians [{ w, m, v }]; extra adds variance
export function mixture(P, extra = 0) {
  const kappa = Math.expm1(P.m + 0.5 * P.delta * P.delta);
  const mu = (P.r - 0.5 * P.sigma * P.sigma - P.lam * kappa) * P.T;
  return poissonTerms(P.lam * P.T).map(([n, w]) => ({ w, m: mu + n * P.m, v: P.sigma * P.sigma * P.T + n * P.delta * P.delta + extra }));
}

export function mixPdf(y, comps) {
  let p = 0;
  for (const { w, m, v } of comps) p += (w * Math.exp((-0.5 * (y - m) * (y - m)) / v)) / Math.sqrt(2 * Math.PI * v);
  return p;
}

export function mixCdf(y, comps) {
  let F = 0;
  for (const { w, m, v } of comps) F += w * normCdf((y - m) / Math.sqrt(v));
  return F;
}

export function grid(P, C = PANEL) {
  const n = 2 * C.half + 1;
  const y = Float64Array.from({ length: n }, (_, i) => (i - C.half) * C.dx);
  const comps = mixture(P);
  const p0 = Float64Array.from(y, (v) => mixPdf(v, comps));
  const dp = new Float64Array(n);
  for (let i = 1; i + 1 < n; i++) dp[i] = (p0[i + 1] - p0[i - 1]) / (2 * C.dx);
  return { y, p0, dp, n };
}

export function levels(C = PANEL) {
  const sig = Float64Array.from({ length: C.N + 1 }, (_, j) => C.smin * Math.pow(C.smax / C.smin, j / C.N));
  const s = Float64Array.from(sig, (v, j) => (j ? Math.sqrt(Math.max(v * v - C.smin * C.smin, 0)) : 0));
  return { sig, s };
}

export const halfTaps = (s, C = PANEL) => Math.ceil((C.nsd * s) / C.dx);

// The density blur (zero outside the grid, weights normalised over every tap)
export function blurCPU(p, s, C = PANEL) {
  if (s === 0) return Float64Array.from(p);
  const H = halfTaps(s, C), n = p.length;
  const w = Float64Array.from({ length: 2 * H + 1 }, (_, k) => Math.exp(-(((k - H) * C.dx) ** 2) / (2 * s * s)));
  const den = w.reduce((a, b) => a + b, 0);
  const q = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    const k0 = Math.max(-H, i - n + 1), k1 = Math.min(H, i);
    for (let k = k0; k <= k1; k++) acc += w[k + H] * p[i - k];
    q[i] = acc / den;
  }
  return q;
}

export const scoreOf = (q, qd) => Float64Array.from(q, (v, i) => (v > 1e-30 ? qd[i] / v : 0));

// linear interpolation on the grid, clamped at the ends (as np.interp and the shader)
export function interpGrid(row, x, C = PANEL) {
  const n = row.length;
  const u = (x + C.half * C.dx) / C.dx;
  const i0 = Math.min(Math.max(Math.floor(u), 0), n - 2);
  const f = Math.min(Math.max(u - i0, 0), 1);
  return row[i0] + f * (row[i0 + 1] - row[i0]);
}

// Heun's method from level N to 0; tables[j] the score at level j. Returns positions at every level.
export function samplerCPU(x1, tables, sig, C = PANEL) {
  const N = sig.length - 1;
  const h = (Math.log(sig[0]) - Math.log(sig[N])) / N;
  const out = new Array(N + 1);
  let x = Float64Array.from(x1);
  out[N] = x;
  for (let j = N; j > 0; j--) {
    const nx = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const k1 = -sig[j] * sig[j] * interpGrid(tables[j], x[i], C);
      const k2 = -sig[j - 1] * sig[j - 1] * interpGrid(tables[j - 1], x[i] + h * k1, C);
      nx[i] = x[i] + 0.5 * h * (k1 + k2);
    }
    out[j - 1] = x = nx;
  }
  return out;
}

// The undiscounted call (r = 0, per $100 of forward) at log-forward offset x, blurred by sd s
export function payoffCPU(x, s, K, C = PANEL) {
  if (s === 0) return Math.max(100 * Math.exp(x) - K, 0);
  const H = C.nsd * C.per;
  let acc = 0, den = 0;
  for (let k = -H; k <= H; k++) {
    const yk = (k * s) / C.per;
    const w = Math.exp(-(yk * yk) / (2 * s * s));
    acc += w * Math.max(100 * Math.exp(x - 0.5 * s * s + yk) - K, 0);
    den += w;
  }
  return acc / den;
}

// Black (r = 0) call per $100 of forward: the payoff blur's closed form
export function blackCall(x, s, K) {
  const F = 100 * Math.exp(x);
  if (s === 0) return Math.max(F - K, 0);
  const d1 = (Math.log(F / K) + 0.5 * s * s) / s;
  return F * normCdf(d1) - K * normCdf(d1 - s);
}

// Seeded uniform numbers (mulberry32) and standard normals (Box-Muller)
export function rng(seed) {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const z = () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
  return { u, z };
}

// Where the reverse run starts: the market's odds blurred to sigma_max, or (pure) N(0, sigma_max^2)
export function startPoints(n, P, { pure = false, seed = 16, C = PANEL } = {}) {
  const R = rng(seed);
  const comps = mixture(P, C.smax * C.smax - C.smin * C.smin);
  const cum = [];
  comps.reduce((a, c) => (cum.push(a + c.w), a + c.w), 0);
  return Float32Array.from({ length: n }, () => {
    if (pure) return C.smax * R.z();
    const u = R.u() * cum.at(-1);
    const c = comps[Math.max(0, cum.findIndex((v) => v >= u))];
    return c.m + Math.sqrt(c.v) * R.z();
  });
}

// One-sample Kolmogorov-Smirnov test against a CDF; p from the asymptotic distribution
// with Stephens' small-sample correction
export function ksTest(xs, cdf) {
  const a = Float64Array.from(xs).sort();
  const n = a.length;
  let D = 0;
  for (let i = 0; i < n; i++) {
    const F = cdf(a[i]);
    D = Math.max(D, (i + 1) / n - F, F - i / n);
  }
  const lam = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * D;
  let p = 0;
  for (let k = 1; k <= 100; k++) p += 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * lam * lam);
  return { D, p: Math.min(1, Math.max(0, p)) };
}
