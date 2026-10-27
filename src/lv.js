// Local volatility (Part 4): pricing by a blur whose width changes from spot to spot.
// The 64-bit twin of reference/lv_blur.py's backward pass, and the job the GPU runs
// (gpu-lv.js). The state is x = ln(S / F_t); each step back in time replaces every
// node by a Gaussian average of its neighbours whose width is the local vol at that
// node and time, times sqrt(dt):
//   V(x_i) <- sum_j w_ij V(x_i + j dx),  w_ij ~ exp(-(j dx - mu_i)^2 / (2 s_i^2)),
//   s_i = sigma(x_i, t_mid) sqrt(h),  mu_i = -s_i^2 / 2,  weights normalised to 1,
// out to TAPS_SD standard deviations (half-widths rounded up to powers of two, as in
// Python). Past the grid a value is continued as A e^x + B through the edge nodes.
// The last step before expiry is Black's formula at each node's local vol, so the
// strike's kink is never sampled. Prices come back per $100 of forward, discounted.
import { normCdf } from "./bs.js";

export const TAPS_SD = 8;

// ---- the local-vol table (lv_data.json): rows at times t inside the expiry intervals
// (edges); linear in k; linear in t inside an interval, flat to its ends; never blended
// across an expiry, where Dupire's dw/dT jumps.
export function lvTable(data) {
  const { t, edges, k, sig } = data.table;
  const rows = sig.map((r) => Float64Array.from(r, (v) => v / 1e4));
  const cell = t.map((ti) => searchLeft(edges, ti));
  return { t, edges, k: Float64Array.from(k), rows, cell };
}

// first index i with a[i] >= v (numpy searchsorted, side="left")
function searchLeft(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// The row blend at time t: { ra, rb, a } with sigma = (1 - a) row ra + a row rb.
export function rowBlend(tab, t) {
  const j = searchLeft(tab.edges, t);
  let rows = [];
  for (let r = 0; r < tab.t.length; r++) if (tab.cell[r] === j) rows.push(r);
  if (!rows.length) rows = [t > tab.t[tab.t.length - 1] ? tab.t.length - 1 : 0];
  const tt = rows.map((r) => tab.t[r]);
  if (t <= tt[0]) return { ra: rows[0], rb: rows[0], a: 0 };
  if (t >= tt[tt.length - 1]) return { ra: rows[rows.length - 1], rb: rows[rows.length - 1], a: 0 };
  const m = searchLeft(tt, t) - 1;
  return { ra: rows[m], rb: rows[m + 1], a: (t - tt[m]) / (tt[m + 1] - tt[m]) };
}

// np.interp(x, k, row): linear, flat outside
export function interp(x, k, row) {
  const n = k.length;
  if (x <= k[0]) return row[0];
  if (x >= k[n - 1]) return row[n - 1];
  const i = searchLeft(k, x) - 1;
  const w = (x - k[i]) / (k[i + 1] - k[i]);
  return row[i] + w * (row[i + 1] - row[i]);
}

// Each table row read at the grid's nodes: rows x n. The GPU blends two of these per step.
export function sigmaAtNodes(tab, x) {
  return tab.rows.map((row) => Float64Array.from(x, (xi) => interp(xi, tab.k, row)));
}

export function grid(dx, lo, hi) {
  const a = Math.floor(lo / dx), b = Math.ceil(hi / dx);
  return Float64Array.from({ length: b - a + 1 }, (_, i) => dx * (a + i));
}

// Step boundaries from 0 through every point: steps <= dt, at least m between points.
export function timeGrid(points, dt, m) {
  const out = [0];
  for (const T of [...points].sort((p, q) => p - q)) {
    const a = out[out.length - 1];
    const n = Math.max(m, Math.ceil((T - a) / dt - 1e-9));
    for (let i = 1; i <= n; i++) out.push(a + ((T - a) * i) / n);
  }
  return out;
}

// half-width in nodes of a node's kernel: a power of two >= the taps out to TAPS_SD sds
export function halfWidth(s, dx) {
  const need = Math.ceil((TAPS_SD * s + 0.5 * s * s) / dx) + 1;
  return 2 ** Math.ceil(Math.log2(Math.max(need, 2)));
}

// value at node index j of column V, continued past the grid as A e^x + B
export function padded(V, x, j) {
  const n = x.length;
  if (j >= 0 && j < n) return V[j];
  const [i0, i1] = j < 0 ? [0, 1] : [n - 2, n - 1];
  const x0 = x[i0], x1 = x[i1];
  const xj = x[0] + j * (x[1] - x[0]);
  const A = (V[i1] - V[i0]) / (Math.exp(x1) - Math.exp(x0));
  return V[i0] + A * (Math.exp(xj) - Math.exp(x0));
}

// Black's formula over the last step at each node: E (e^X - e^k)^+ (or the put),
// X ~ N(x - v/2, v), v = sig^2 h.
export function lastStep(x, sig, h, k, call) {
  const V = new Float64Array(x.length);
  const ek = Math.exp(k);
  for (let i = 0; i < x.length; i++) {
    const v = Math.sqrt(sig[i] * sig[i] * h);
    const ex = Math.exp(x[i]);
    const d1 = (x[i] - k + 0.5 * v * v) / v;
    const c = ex * normCdf(d1) - ek * normCdf(d1 - v);
    V[i] = call ? c : c - ex + ek;
  }
  return V;
}

// The job: everything the GPU needs to price a set of vanillas at one expiry.
//   data: lv_data.json; T: expiry (years); strikes: [{ k, call }]; D: discount to T;
//   sigma: optional flat vol instead of the local-vol table.
export function lvJob(data, { T, strikes, D = 1, sigma = null, settings = data.settings, tab = null }) {
  const { dt, m, dx, x_lo, x_hi } = settings;
  const x = grid(dx, x_lo, x_hi);
  const knots = data.expiries.map((e) => e.T).filter((t) => t < T);
  const tg = timeGrid([...new Set([...knots, T])], dt, m);
  const steps = [];
  for (let s = 1; s < tg.length; s++) steps.push({ a: tg[s - 1], b: tg[s] });
  let SN, blends;
  if (sigma != null) {
    SN = [new Float64Array(x.length).fill(sigma)];
    blends = steps.map(() => ({ ra: 0, rb: 0, a: 0 }));
  } else {
    tab ??= lvTable(data);
    SN = sigmaAtNodes(tab, x);
    blends = steps.map(({ a, b }) => rowBlend(tab, 0.5 * (a + b)));
  }
  const sigAt = (s) => {
    const { ra, rb, a } = blends[s];
    return Float64Array.from(x, (_, i) => (1 - a) * SN[ra][i] + a * SN[rb][i]);
  };
  const last = steps.length - 1;
  const sl = sigAt(last);
  const V0 = strikes.map(({ k, call }) => lastStep(x, sl, steps[last].b - steps[last].a, k, call));
  let center = 0;
  for (let i = 1; i < x.length; i++) if (Math.abs(x[i]) < Math.abs(x[center])) center = i;
  return { x, dx, steps, blends, SN, sigAt, V0, center, scale: 100 * D, T, strikes };
}

// One step back over h on the CPU, every column: the gather. Each node's weights are
// worked out once and shared by the columns.
export function stepCPU(Vs, x, sig, h, dx) {
  const n = x.length, C = Vs.length;
  const outs = Vs.map(() => new Float64Array(n));
  const num = new Float64Array(C);
  for (let i = 0; i < n; i++) {
    const s = sig[i] * Math.sqrt(h);
    const mu = -0.5 * s * s;
    const H = halfWidth(s, dx);
    const inside = i - H >= 0 && i + H < n;
    let den = 0;
    num.fill(0);
    for (let j = -H; j <= H; j++) {
      const z = j * dx - mu;
      const w = Math.exp(-(z * z) / (2 * s * s));
      den += w;
      for (let c = 0; c < C; c++) num[c] += w * (inside ? Vs[c][i + j] : padded(Vs[c], x, i + j));
    }
    for (let c = 0; c < C; c++) outs[c][i] = num[c] / den;
  }
  return outs;
}

// The whole backward pass in 64-bit: prices per $100 of forward, discounted.
export function priceLvCPU(job) {
  let Vs = job.V0;
  for (let s = job.steps.length - 2; s >= 0; s--) {
    const { a, b } = job.steps[s];
    Vs = stepCPU(Vs, job.x, job.sigAt(s), b - a, job.dx);
  }
  return Vs.map((V) => job.scale * V[job.center]);
}

// Black in forward units and its inverse (vol), for the round trip.
export function blackFwd(k, w, call) {
  const v = Math.sqrt(w);
  const d1 = (-k + 0.5 * w) / v;
  const c = normCdf(d1) - Math.exp(k) * normCdf(d1 - v);
  return call ? c : c - 1 + Math.exp(k);
}

export function impliedVol(c, k, T, call) {
  const intr = call ? Math.max(1 - Math.exp(k), 0) : Math.max(Math.exp(k) - 1, 0);
  if (!(c > intr) || c >= (call ? 1 : Math.exp(k))) return NaN;
  let lo = 1e-6, hi = 10;
  for (let it = 0; it < 100; it++) {
    const mid = 0.5 * (lo + hi);
    if (blackFwd(k, mid * mid, call) < c) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi) / Math.sqrt(T);
}
