// Prices a drawn payoff for the demo page: sample it on a log grid, blur it
// (GPU or CPU), and read the price curve back over the plotted range.
import { gaussianKernel, logGrid, blur } from "./blur.js";
import { blurGPU, americanGPU } from "./gpu.js";

export const S0 = 100;
export const R = 0.05;
export const S_MIN = 40;
export const S_MAX = 160;
export const SIGMA_MAX = 0.8;
export const T_MAX = 2;

// A payoff drawn over [S_MIN, S_MAX], one sample every 0.2. Past the ends it
// continues as a straight line with the slope of its last 1.0 of price,
// so a drawn call keeps rising and a drawn put keeps falling.
export class Payoff {
  static N = 601;
  static STEP = (S_MAX - S_MIN) / (Payoff.N - 1);

  constructor(fn) {
    this.y = Float64Array.from({ length: Payoff.N }, (_, i) => fn(S_MIN + i * Payoff.STEP));
    this.version = 0;
  }

  // Overwrite the samples between two drawn points.
  stroke(S1, y1, S2, y2) {
    const i1 = Math.round((S1 - S_MIN) / Payoff.STEP);
    const i2 = Math.round((S2 - S_MIN) / Payoff.STEP);
    const lo = Math.max(0, Math.min(i1, i2));
    const hi = Math.min(Payoff.N - 1, Math.max(i1, i2));
    for (let i = lo; i <= hi; i++) {
      const t = i1 === i2 ? 1 : (i - i1) / (i2 - i1);
      this.y[i] = Math.max(0, y1 + t * (y2 - y1));
    }
    this.version++;
  }

  value(S) {
    const n = Payoff.N;
    const h = Payoff.STEP;
    const u = (S - S_MIN) / h;
    if (u <= 0) return Math.max(0, this.y[0] + (this.y[0] - this.y[5]) * (-u / 5));
    if (u >= n - 1) return Math.max(0, this.y[n - 1] + (this.y[n - 1] - this.y[n - 6]) * ((u - (n - 1)) / 5));
    const i = Math.floor(u);
    return this.y[i] + (u - i) * (this.y[i + 1] - this.y[i]);
  }
}

// Keys cubic (a = -0.5). Averaging the payoff against it, instead of reading
// it at the grid points, takes the kink out of the sampling error: a call at
// T = 0.01 goes from 4e-3 off Black-Scholes to under 1e-5.
function keys(t) {
  t = Math.abs(t);
  if (t < 1) return 1.5 * t * t * t - 2.5 * t * t + 1;
  if (t < 2) return -0.5 * t * t * t + 2.5 * t * t - 4 * t + 2;
  return 0;
}
const SUB = 16; // quadrature points per grid cell
const SUB_T = Float64Array.from({ length: 4 * SUB }, (_, m) => -2 + (m + 0.5) / SUB);
const SUB_W = Float64Array.from(SUB_T, (t) => keys(t) / SUB);

// Grid spacing halves until the blur is at least 6 cells wide. Each level has
// one fixed grid, wide enough for the largest blur that level can see, so the
// sampled payoff only has to be rebuilt when the drawing changes.
const DX0 = 0.0025;
const DISPLAY_HALF = Math.log(S_MAX / S0) > Math.log(S0 / S_MIN) ? Math.log(S_MAX / S0) : Math.log(S0 / S_MIN);
const grids = new Map();

function level(std) {
  let k = 0;
  while (DX0 / 2 ** k > std / 6 && k < 6) k++;
  return k;
}

function grid(k) {
  if (!grids.has(k)) {
    const dx = DX0 / 2 ** k;
    const maxStd = k === 0 ? SIGMA_MAX * Math.sqrt(T_MAX) : 12 * dx;
    const maxDrift = 0.5 * SIGMA_MAX * SIGMA_MAX * T_MAX;
    const { x, center } = logGrid(S0, DISPLAY_HALF + 8 * maxStd + maxDrift + 20 * dx, dx);
    const S = Float64Array.from(x, Math.exp);
    let lo = 0;
    while (S[lo] < S_MIN - 1e-9) lo++;
    let hi = S.length - 1;
    while (S[hi] > S_MAX + 1e-9) hi--;
    grids.set(k, { k, dx, x, S, center, lo, hi, sampled: null, version: -1 });
  }
  return grids.get(k);
}

function sampled(g, payoff) {
  if (g.sampled && g.version === payoff.version && g.payoff === payoff) return g.sampled;
  const e = Float64Array.from(SUB_T, (t) => Math.exp(t * g.dx));
  const out = new Float64Array(g.S.length);
  for (let j = 0; j < out.length; j++) {
    let s = 0;
    for (let m = 0; m < e.length; m++) s += SUB_W[m] * payoff.value(g.S[j] * e[m]);
    out[j] = s;
  }
  Object.assign(g, { sampled: out, version: payoff.version, payoff });
  return out;
}

// CPU fallback for browsers without WebGPU: same loop, 64-bit floats.
async function blurCPU(values, kernel) {
  return blur(values, kernel);
}
async function americanCPU(P, kernel, steps, disc) {
  const kd = Float64Array.from(kernel, (w) => w * disc);
  let V = P;
  for (let s = 0; s < steps; s++) {
    V = blur(V, kd);
    for (let i = 0; i < V.length; i++) if (P[i] > V[i]) V[i] = P[i];
  }
  return V;
}

// Price `payoff` at every grid point in [S_MIN, S_MAX].
// Returns { S, V, payoff, exercise, steps } for that range; exercise[i] is 1
// where the American price equals the payoff (exercising now is optimal).
export async function priceCurve(payoff, { T, sigma, american, gpu = true }) {
  const std = sigma * Math.sqrt(T);
  const g = grid(level(Math.max(std, 1e-9)));
  const P = sampled(g, payoff);
  const view = (V) => V.subarray(g.lo, g.hi + 1);
  const res = { S: view(g.S), payoff: view(P), dx: g.dx, steps: 0 };

  if (T <= 0) return { ...res, V: view(P), exercise: american ? view(P).map((p) => (p > 0 ? 1 : 0)) : null };

  if (!american) {
    const k = gaussianKernel(g.dx, std, (R - 0.5 * sigma * sigma) * T);
    const V = await (gpu ? blurGPU : blurCPU)(P, k);
    const disc = Math.exp(-R * T);
    const out = Float64Array.from(view(V), (v) => v * disc);
    return { ...res, V: out, exercise: null };
  }

  // Enough exercise dates to converge, but keep each step's blur at least
  // three cells wide so the drift isn't rounded away.
  const steps = Math.max(1, Math.min(200, Math.floor((std / (3 * g.dx)) ** 2)));
  const dt = T / steps;
  const k = gaussianKernel(g.dx, sigma * Math.sqrt(dt), (R - 0.5 * sigma * sigma) * dt);
  const V = await (gpu ? americanGPU : americanCPU)(P, k, steps, Math.exp(-R * dt));
  const Pv = view(P);
  const Vv = view(V);
  // The last clip writes the payoff value itself, so exercise shows up as equality.
  const P32 = gpu ? Float32Array.from(Pv) : Pv;
  const exercise = Uint8Array.from(Vv, (v, i) => (Pv[i] > 0 && v <= P32[i] ? 1 : 0));
  return { ...res, V: Float64Array.from(Vv), exercise, steps };
}
