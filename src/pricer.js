// Prices a drawn payoff for the demo page: sample it on a log grid, blur it
// (GPU or CPU), and read the price curve back over the plotted range.
import { gaussianKernel, logGrid, blur } from "./blur.js";
import { blurGPU, americanGPU, barrierGPU, stepBarrierGPU } from "./gpu.js";
import {
  MODE, ALL_IMAGES, BGK_BETA, discountedKernel, mirrorExponent, barrierBlurCPU, stepBarrierCPU, interpCubic,
} from "./barrier.js";
import { gridFor, kernel as tentKernel, meanSd, kinkPair } from "./kernels.js";
import { DEMO_GRID, blurRows } from "./smile.js";

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

// ---------------------------------------------------------------------------
// Fat-tailed kernels (Merton, variance gamma), European only: one blur of the
// payoff with the model's tent kernel (kernels.js, the code checked against
// Python), the FFT blur once the kernel is wide. The grid spacing
// follows the narrowest part of the kernel (the diffusion, for Merton) as
// above; the grid reaches the kernel's half-width past the plotted range.
// Kernels are cached, so redrawing the payoff reuses them.

const fatGrids = new Map();
function fatGrid(k, H) {
  const key = `${k}:${H}`;
  if (!fatGrids.has(key)) {
    const dx = DX0 / 2 ** k;
    const { x, center } = logGrid(S0, H, dx);
    const S = Float64Array.from(x, Math.exp);
    let lo = 0;
    while (S[lo] < S_MIN - 1e-9) lo++;
    let hi = S.length - 1;
    while (S[hi] > S_MAX + 1e-9) hi--;
    fatGrids.set(key, { k, dx, x, S, center, lo, hi, sampled: null, version: -1 });
  }
  return fatGrids.get(key);
}

// The payoff sampled for the tent kernels: point values, plus the kink
// correction (kernels.js) at every drawn sample where the slope changes.
// Between samples a drawn payoff is a straight line in S, a + b e^z, which the
// tents blur exactly; so only the kinks need anything, and this makes the blur
// as accurate as the Python reference. (The Keys average above suits the
// sampled Gaussian; with the tents it would cost ~1e-4.)
function tentSampled(g, payoff) {
  if (g.tent && g.tentVersion === payoff.version && g.tentPayoff === payoff) return g.tent;
  const out = Float64Array.from(g.S, (S) => payoff.value(S));
  const y = payoff.y;
  const n = y.length;
  const h = Payoff.STEP;
  // slope in S of the segment right of sample i (-1: the extension left of S_MIN)
  const slope = (i) => (i < 0 ? (y[5] - y[0]) / (5 * h) : i >= n - 1 ? (y[n - 1] - y[n - 6]) / (5 * h) : (y[i + 1] - y[i]) / h);
  for (let i = 0; i < n; i++) {
    const jump = slope(i) - slope(i - 1);
    if (jump === 0) continue;
    const [j, a, b] = kinkPair(g.x[0], g.dx, S_MIN + i * h);
    if (j < 0 || j + 1 >= out.length) continue;
    out[j] += jump * a;
    out[j + 1] += jump * b;
  }
  Object.assign(g, { tent: out, tentVersion: payoff.version, tentPayoff: payoff });
  return out;
}

const tentKernels = new Map();
export function demoKernel(model, p, dx) {
  const key = JSON.stringify([model, p, dx]);
  let kern = tentKernels.get(key);
  if (!kern) {
    const { y, J } = gridFor(model, p, { ...DEMO_GRID, dx });
    kern = { y, w: tentKernel(model, y, p), J, dx };
    tentKernels.set(key, kern);
    if (tentKernels.size > 48) tentKernels.delete(tentKernels.keys().next().value);
  }
  return kern;
}

// p: { sigma, lam, m, delta } (Merton) or { sigma, theta, nu } (variance gamma)
export async function priceCurveFat(payoff, { T, model, p, gpu = true }) {
  if (T <= 0) return priceCurve(payoff, { T, sigma: p.sigma, american: false, gpu });
  const pp = { ...p, T, r: R };
  const narrow = model === "merton" ? p.sigma * Math.sqrt(T) : meanSd(model, pp)[1];
  const k = Math.min(level(Math.max(narrow, 1e-9)), 4);
  const dx = DX0 / 2 ** k;
  const kern = demoKernel(model, pp, dx);
  const g = fatGrid(k, Math.ceil(DISPLAY_HALF + (kern.J + 20) * dx));
  const P = tentSampled(g, payoff);
  const V = await blurRows(P, 1, P.length, kern.w, dx, gpu);
  const disc = Math.exp(-R * T);
  const view = (A) => A.subarray(g.lo, g.hi + 1);
  return { S: view(g.S), V: Float64Array.from(view(V), (v) => v * disc), payoff: view(P), dx, steps: 0, exercise: null, taps: kern.w.length };
}

// ---------------------------------------------------------------------------
// Barriers. Barrier levels snap to the coarsest grid (spacing DX0 in ln S,
// anchored at S0), so they are nodes on every finer level too.

export const SLICES = 12; // surface slices: tau = T/12, 2T/12, ..., T

export function snapBarrier(S) {
  const m = Math.round(Math.log(S / S0) / DX0);
  return { m, S: S0 * Math.exp(m * DX0) };
}

// Dead (knocked) nodes of a curve given its S values and the barrier levels.
function deadMask(S, b) {
  return Uint8Array.from(S, (s) => ((b.L && s < b.L.S * (1 - 1e-12)) || (b.U && s > b.U.S * (1 + 1e-12)) ? 1 : 0));
}

function barrierMode(b) {
  const base = b.ud === "down" ? MODE.down : b.ud === "up" ? MODE.up : MODE.double;
  return base + (b.io === "in" ? MODE.IN : 0);
}

// Weighted (a = 2 nu / sigma^2) or plain (a = 0) mirror: every slice of the
// surface is one blur of the image-tiled payoff, and all of them, plus the
// image-count study at S0, go to the GPU as ONE dispatch.
async function mirrorSurface(payoff, { T, sigma, barrier: b, method, gpu }) {
  const taus = Array.from({ length: SLICES }, (_, j) => (T * (j + 1)) / SLICES);
  const levels = taus.map((tau) => level(Math.max(sigma * Math.sqrt(tau), 1e-9)));
  const pOff = new Map();
  const Ps = [];
  let nPtot = 0;
  for (const k of new Set(levels)) {
    const P = sampled(grid(k), payoff);
    pOff.set(k, nPtot);
    Ps.push(P);
    nPtot += P.length;
  }
  const Pall = new Float64Array(nPtot);
  let o = 0;
  for (const P of Ps) {
    Pall.set(P, o);
    o += P.length;
  }
  const kers = [];
  let kTot = 0;
  const kernelFor = (k, tau) => {
    const ker = discountedKernel(grid(k).dx, sigma, tau, R);
    const off = kTot;
    kers.push(ker);
    kTot += ker.length;
    return { kOff: off, half: (ker.length - 1) / 2 };
  };
  const mode = barrierMode(b);
  const aW = mirrorExponent(sigma, R);
  const a = method === "plain" ? 0 : aW;
  const opts = [];
  let outTot = 0;
  const add = (o) => {
    opts.push({ ...o, outOff: outTot });
    outTot += o.nOut;
    return opts.length - 1;
  };
  const slices = taus.map((tau, j) => {
    const k = levels[j];
    const g = grid(k);
    const node = (m) => g.center + m * 2 ** k;
    const base = {
      pOff: pOff.get(k), nP: g.S.length, dx: g.dx, ...kernelFor(k, tau), nOut: g.hi - g.lo + 1, out0: g.lo,
      lo: b.ud === "up" ? node(b.U.m) : node(b.L.m), hi: b.ud === "double" ? node(b.U.m) : -1, images: ALL_IMAGES,
    };
    return {
      tau, g, base,
      main: add({ ...base, mode, a }),
      vanilla: b.io === "in" ? add({ ...base, mode: MODE.none, a: 0 }) : -1,
    };
  });
  // at S0 on the front: price with 0, 1, 2, ... images, and both mirror weights
  const front = slices[SLICES - 1];
  const at0 = { ...front.base, nOut: 1, out0: front.g.center };
  const maxImages = b.ud === "double" ? 16 : 1;
  const imageIdx = Array.from({ length: maxImages + 1 }, (_, M) => add({ ...at0, mode, a, images: M }));
  const plainIdx = add({ ...at0, mode, a: 0 });
  const weightedIdx = add({ ...at0, mode, a: aW });

  const K = new Float64Array(kTot);
  let ko = 0;
  for (const ker of kers) {
    K.set(ker, ko);
    ko += ker.length;
  }
  const spec = { P: Pall, K, opts };
  const out = gpu ? await barrierGPU(spec) : barrierBlurCPU(spec);
  const read = (idx) => Float64Array.from(out.subarray(opts[idx].outOff, opts[idx].outOff + opts[idx].nOut));
  const curves = slices.map((s) => {
    const S = s.g.S.subarray(s.g.lo, s.g.hi + 1);
    const V = read(s.main);
    const dead = deadMask(S, b);
    const van = s.vanilla >= 0 ? read(s.vanilla) : null;
    for (let i = 0; i < V.length; i++) if (dead[i]) V[i] = van ? van[i] : 0;
    return { tau: s.tau, S, V, dead, dx: s.g.dx, vanilla: van };
  });
  const last = curves[SLICES - 1];
  return {
    curves,
    vanillaAtS0: last.vanilla ? last.vanilla[front.g.center - front.g.lo] : null,
    images: imageIdx.map((idx, M) => ({ M, V: out[opts[idx].outOff] })),
    atS0: { plain: out[opts[plainIdx].outOff], weighted: out[opts[weightedIdx].outOff] },
    threads: outTot,
  };
}

// Step by step: blur T/N, discount, knock out past the barrier, N times. The
// grid is anchored at the monitored barrier (moved toward the spot by
// 0.5826 sigma sqrt(dt) with the BGK shift) so it sits on a node.
async function stepSurface(payoff, { T, sigma, barrier: b, steps, shift, gpu, snapshots = true }) {
  const dt = T / steps;
  const beta = shift ? BGK_BETA * sigma * Math.sqrt(dt) : 0;
  const dxMax = Math.min(DX0, (sigma * Math.sqrt(dt)) / 4);
  const l = b.L ? Math.log(b.L.S) + beta : null;
  const u = b.U ? Math.log(b.U.S) - beta : null;
  let anchor = l ?? u;
  let dx = dxMax;
  let m = 0;
  if (b.ud === "double") {
    if (u - l < 4 * dxMax) return null; // the shifted corridor has closed: worth ~0
    m = Math.ceil((u - l) / dxMax);
    dx = (u - l) / m;
    anchor = l;
  }
  const reach = 8 * sigma * Math.sqrt(T) + Math.abs(R - 0.5 * sigma * sigma) * T + 20 * dx;
  const lo = Math.floor((Math.log(S_MIN) - reach - anchor) / dx);
  const hi = Math.ceil((Math.log(S_MAX) + reach - anchor) / dx);
  const z = Float64Array.from({ length: hi - lo + 1 }, (_, j) => anchor + (lo + j) * dx);
  const g = { dx, S: Float64Array.from(z, Math.exp) };
  const P = sampled(g, payoff);
  const ib = -lo;
  let iLo = 0;
  while (g.S[iLo] < S_MIN - 1e-9) iLo++;
  let iHi = g.S.length - 1;
  while (g.S[iHi] > S_MAX + 1e-9) iHi--;
  const nOut = iHi - iLo + 1;
  const snaps = snapshots ? Array.from({ length: SLICES }, (_, j) => ({ step: ((j + 1) * steps) / SLICES, out0: iLo, nOut })) : [];
  const spec = {
    P, kernel: discountedKernel(dx, sigma, dt, R), steps, dx, snaps,
    lo: b.ud === "up" ? -1 : ib, hi: b.ud === "up" ? ib : b.ud === "double" ? ib + m : -1,
  };
  const res = gpu ? await stepBarrierGPU(spec) : stepBarrierCPU(spec);
  const row = b.io === "in" ? 2 : 1;
  const S = g.S.subarray(iLo, iHi + 1);
  const dead = deadMask(S, b);
  const curves = res.snaps.map((shot, j) => {
    const V = Float64Array.from(shot.subarray(row * nOut, (row + 1) * nOut));
    for (let i = 0; i < nOut; i++) if (dead[i]) V[i] = b.io === "in" ? shot[i] : 0;
    return { tau: (T * (j + 1)) / SLICES, S, V, dead, dx };
  });
  const n = P.length;
  const at0 = interpCubic(z[0], dx, Float64Array.from(res.rows.subarray(row * n, (row + 1) * n)), Math.log(S0));
  return { curves, atS0: at0, N: steps, threads: 3 * n };
}

// Price a barrier option on `payoff` over the plotted range, at SLICES times
// to expiry. method: "plain" | "weighted" | "step". Returns
// { curves: [{ tau, S, V, dead }], images: [{ M, V }], atS0: { plain, weighted, step } }.
export async function priceBarrier(payoff, { T, sigma, barrier, method, shift = true, steps = 48, gpu = true }) {
  const mirror = await mirrorSurface(payoff, { T, sigma, barrier, method: method === "step" ? "weighted" : method, gpu });
  const step = await stepSurface(payoff, { T, sigma, barrier, steps, shift, gpu, snapshots: method === "step" });
  // the shifted corridor closed: knocked out at once, so out = 0 and in = vanilla
  const closed = (c) => ({ ...c, V: barrier.io === "in" ? c.vanilla : new Float64Array(c.V.length) });
  const curves = method === "step" ? (step ? step.curves : mirror.curves.map(closed)) : mirror.curves;
  return {
    curves,
    images: mirror.images,
    atS0: { ...mirror.atS0, step: step ? step.atS0 : barrier.io === "in" ? mirror.vanillaAtS0 : 0 },
    steps,
  };
}
