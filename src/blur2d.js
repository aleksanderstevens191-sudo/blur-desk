// Two-stock options as one 2D blur: the 64-bit JS twin of reference/blur2d.py.
// Keep the two in step; src/test-2d.mjs checks this file against the Python
// fixtures, and the GPU (gpu2d.js) is checked against both.
//
// Layout: an image A over (x, y) = (ln S1, ln S2) is a Float64Array with
// A[i * ny + j] at x index i, y index j (a "row" runs along y).
import { normCdf } from "./bs.js";

// ---------- closed forms ----------

// Gauss-Legendre nodes and weights on [-1, 1], by Newton on P_n.
function gaussLegendre(n) {
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) [p0, p1] = [p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k];
      const dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) {
        x[i] = z;
        w[i] = 2 / ((1 - z * z) * dp * dp);
        break;
      }
    }
  }
  return { x, w };
}
const GL = { 6: gaussLegendre(6), 12: gaussLegendre(12), 20: gaussLegendre(20) };

// P(Z1 > h, Z2 > k), correlation r: Genz's BVNU (Drezner-Wesolowsky with
// Gauss-Legendre), about 1e-15.
function bvnUpper(h, k, r) {
  const { x, w } = Math.abs(r) < 0.3 ? GL[6] : Math.abs(r) < 0.75 ? GL[12] : GL[20];
  let hk = h * k;
  let bvn = 0;
  if (Math.abs(r) < 0.925) {
    const hs = (h * h + k * k) / 2;
    const asr = Math.asin(r);
    for (let i = 0; i < x.length; i++) {
      const sn = Math.sin((asr * (x[i] + 1)) / 2);
      bvn += w[i] * Math.exp((sn * hk - hs) / (1 - sn * sn));
    }
    return (bvn * asr) / (4 * Math.PI) + normCdf(-h) * normCdf(-k);
  }
  if (r < 0) {
    k = -k;
    hk = -hk;
  }
  if (Math.abs(r) < 1) {
    const as = (1 - r) * (1 + r);
    let a = Math.sqrt(as);
    const bs = (h - k) ** 2;
    const c = (4 - hk) / 8;
    const d = (12 - hk) / 16;
    bvn = a * Math.exp(-(bs / as + hk) / 2) * (1 - (c * (bs - as) * (1 - (d * bs) / 5)) / 3 + (c * d * as * as) / 5);
    if (hk > -160) {
      const b = Math.sqrt(bs);
      bvn -= Math.exp(-hk / 2) * Math.sqrt(2 * Math.PI) * normCdf(-b / a) * b * (1 - (c * bs * (1 - (d * bs) / 5)) / 3);
    }
    a /= 2;
    for (let i = 0; i < x.length; i++) {
      const xs = (a * (x[i] + 1)) ** 2;
      const rs = Math.sqrt(1 - xs);
      const e = -(bs / xs + hk) / 2;
      if (e > -100) bvn += a * w[i] * Math.exp(e) * (Math.exp((-hk * xs) / (2 * (1 + rs) ** 2)) / rs - (1 + c * xs * (1 + d * xs)));
    }
    bvn = -bvn / (2 * Math.PI);
  }
  if (r > 0) return bvn + normCdf(-Math.max(h, k));
  if (h >= k) return -bvn;
  const L = h < 0 ? normCdf(k) - normCdf(h) : normCdf(-h) - normCdf(-k);
  return L - bvn;
}

// P(Z1 <= h, Z2 <= k) with correlation rho.
export const bvnCdf = (h, k, rho) => bvnUpper(-h, -k, rho);

export function bs(S, K, sigma, T, r, q = 0, cp = "call") {
  const v = sigma * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v;
  const c = S * Math.exp(-q * T) * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d1 - v);
  return cp === "call" ? c : c - S * Math.exp(-q * T) + K * Math.exp(-r * T);
}

// Exchange stock 2 for stock 1, max(S1 - S2, 0) (Margrabe 1978).
export function margrabe(S1, S2, s1, s2, rho, T, q1 = 0, q2 = 0) {
  const s = Math.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2);
  const d1 = (Math.log(S1 / S2) + (q2 - q1 + 0.5 * s * s) * T) / (s * Math.sqrt(T));
  return S1 * Math.exp(-q1 * T) * normCdf(d1) - S2 * Math.exp(-q2 * T) * normCdf(d1 - s * Math.sqrt(T));
}

// Call on the better of two, max(max(S1, S2) - K, 0) (Stulz 1982).
export function stulzCallMax(S1, S2, K, s1, s2, rho, T, r, q1 = 0, q2 = 0) {
  const s = Math.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2);
  const [sT, s1T, s2T] = [s, s1, s2].map((v) => v * Math.sqrt(T));
  const d = (Math.log(S1 / S2) + (q2 - q1 + 0.5 * s * s) * T) / sT;
  const y1 = (Math.log(S1 / K) + (r - q1 + 0.5 * s1 * s1) * T) / s1T;
  const y2 = (Math.log(S2 / K) + (r - q2 + 0.5 * s2 * s2) * T) / s2T;
  return S1 * Math.exp(-q1 * T) * bvnCdf(y1, d, (s1 - rho * s2) / s) +
    S2 * Math.exp(-q2 * T) * bvnCdf(y2, -d + sT, (s2 - rho * s1) / s) -
    K * Math.exp(-r * T) * (1 - bvnCdf(-y1 + s1T, -y2 + s2T, rho));
}

// Call on the worse of two: C1 + C2 - C_max.
export const stulzCallMin = (S1, S2, K, s1, s2, rho, T, r, q1 = 0, q2 = 0) =>
  bs(S1, K, s1, T, r, q1) + bs(S2, K, s2, T, r, q2) - stulzCallMax(S1, S2, K, s1, s2, rho, T, r, q1, q2);

// ---------- the kernel ----------

export const drifts = (T, r, s1, s2, q1 = 0, q2 = 0) => [(r - q1 - 0.5 * s1 * s1) * T, (r - q2 - 0.5 * s2 * s2) * T];

// The 1-sd ellipse of the kernel in (ln S1, ln S2): semi-axes (major first)
// and the major axis's angle from the x axis, in radians.
export function ellipse(s1, s2, rho, T) {
  const a = s1 * s1 * T, b = rho * s1 * s2 * T, c = s2 * s2 * T;
  const m = (a + c) / 2;
  const d = Math.hypot((a - c) / 2, b);
  return { major: Math.sqrt(m + d), minor: Math.sqrt(Math.max(m - d, 0)), angle: 0.5 * Math.atan2(2 * b, a - c) };
}

// Grid steps: a fraction c of each stock's sd given the other, s_i sqrt(T (1 - rho^2)).
export function gridSteps(s1, s2, rho, T, c = 0.03) {
  const f = c * Math.sqrt(T * (1 - rho * rho));
  return [f * s1, f * s2];
}

export function kernel1d(d, std, center, width = 8) {
  const half = Math.ceil((width * std + Math.abs(center)) / d);
  const off = Float64Array.from({ length: 2 * half + 1 }, (_, j) => (j - half) * d);
  const w = off.map((o) => Math.exp(-0.5 * ((o - center) / std) ** 2));
  const s = w.reduce((a, b) => a + b, 0);
  return { off, w: w.map((v) => v / s), half };
}

// The 2D kernel W[i * my + j] on offsets (xo[i], yo[j]), sum 1. With rho = 0
// it is the outer product of the two kernel1d's, as in Python.
export function kernel2d(dx, dy, s1, s2, rho, T, r, q1 = 0, q2 = 0, width = 8) {
  const [m1, m2] = drifts(T, r, s1, s2, q1, q2);
  const a = s1 * Math.sqrt(T), b = s2 * Math.sqrt(T);
  const kx = kernel1d(dx, a, m1, width);
  const ky = kernel1d(dy, b, m2, width);
  const mx = kx.off.length, my = ky.off.length;
  const W = new Float64Array(mx * my);
  if (rho === 0) {
    for (let i = 0; i < mx; i++) for (let j = 0; j < my; j++) W[i * my + j] = kx.w[i] * ky.w[j];
  } else {
    let s = 0;
    const den = 2 * (1 - rho * rho);
    for (let i = 0; i < mx; i++) {
      const u = (kx.off[i] - m1) / a;
      for (let j = 0; j < my; j++) {
        const v = (ky.off[j] - m2) / b;
        const e = Math.exp(-(u * u - 2 * rho * u * v + v * v) / den);
        W[i * my + j] = e;
        s += e;
      }
    }
    for (let t = 0; t < W.length; t++) W[t] /= s;
  }
  return { xo: kx.off, yo: ky.off, W, mx, my };
}

// ---------- payoffs and their kinks ----------
// { g(x, y), rowKinks(y) -> [[x, j1, j2], ...], colKinks: [y, ...] }, as in Python.

export function exchangePayoff() {
  return { g: (x, y) => Math.max(Math.exp(x) - Math.exp(y), 0), rowKinks: (y) => [[y, Math.exp(y), Math.exp(y)]], colKinks: [] };
}

export function bestOfCallPayoff(K) {
  const k = Math.log(K);
  return {
    g: (x, y) => Math.max(Math.exp(Math.max(x, y)) - K, 0),
    rowKinks: (y) => (y < k ? [[k, K, K]] : [[y, Math.exp(y), Math.exp(y)]]),
    colKinks: [k],
  };
}

export function worstOfCallPayoff(K) {
  const k = Math.log(K);
  return {
    g: (x, y) => Math.max(Math.exp(Math.min(x, y)) - K, 0),
    rowKinks: (y) => (y > k ? [[k, K, K], [y, -Math.exp(y), -Math.exp(y)]] : []),
    colKinks: [k],
  };
}

export function spreadPayoff(K) {
  if (K === 0) return exchangePayoff();
  return {
    g: (x, y) => Math.max(Math.exp(x) - Math.exp(y) - K, 0),
    rowKinks: (y) => (Math.exp(y) + K > 0 ? [[Math.log(Math.exp(y) + K), Math.exp(y) + K, Math.exp(y) + K]] : []),
    colKinks: [],
  };
}

const vanilla = (K, cp) => (cp === "call" ? (z) => Math.max(Math.exp(z) - K, 0) : (z) => Math.max(K - Math.exp(z), 0));

// scale * f1(S1) * f2(S2)
export function productPayoff(K1, cp1, K2, cp2, scale = 1) {
  const f1 = vanilla(K1, cp1), f2 = vanilla(K2, cp2);
  return {
    g: (x, y) => scale * f1(x) * f2(y),
    rowKinks: (y) => [[Math.log(K1), scale * K1 * f2(y), scale * K1 * f2(y)]],
    colKinks: [Math.log(K2)],
  };
}

// The payoffs of src/fixtures_2d.json by name (call_x_put: (S1 - 103)+ (97 - S2)+ / 100).
export function fixturePayoff({ payoff, K }) {
  if (payoff === "exchange") return exchangePayoff();
  if (payoff === "best_of_call") return bestOfCallPayoff(K);
  if (payoff === "worst_of_call") return worstOfCallPayoff(K);
  if (payoff === "spread") return spreadPayoff(K);
  if (payoff === "call_x_put") return productPayoff(103, "call", 97, "put", 1 / 100);
  throw new Error(`unknown payoff ${payoff}`);
}

// ---------- sampling with kink corrections ----------

// One-sided first and second derivatives at b (cubic through b + side eps t, t = 1..4).
function oneSided(f, b, side, eps) {
  const [g1, g2, g3, g4] = [1, 2, 3, 4].map((t) => f(b + side * eps * t));
  const d1 = g1.map((_, i) => (side * (-52 * g1[i] + 114 * g2[i] - 84 * g3[i] + 22 * g4[i])) / (12 * eps));
  const d2 = g1.map((_, i) => (3 * g1[i] - 8 * g2[i] + 7 * g3[i] - 2 * g4[i]) / (eps * eps));
  return [d1, d2];
}

// Node corrections around a kink a fraction theta past node i (see Python _kink_corr).
export function kinkCorr(theta, d, j1, j2) {
  const b2 = theta * theta - theta + 1 / 6;
  const b3 = theta * (theta - 0.5) * (theta - 1);
  const D0 = (d * b2 * j1) / 2 - (d * d * b3 * j2) / 6;
  const D1 = (-d * d * b3 * j1) / 3;
  const right = D1 / d + theta * D0;
  return [D0 - right, right];
}

// Payoff image on xs x ys with every kink corrected (rows along x from the
// exact jumps, then horizontal kinks down the columns, as in Python).
export function samplePayoff2d(payoff, xs, ys) {
  const { g, rowKinks, colKinks } = payoff;
  const nx = xs.length, ny = ys.length;
  const dx = xs[1] - xs[0], dy = ys[1] - ys[0];
  // the row-corrected payoff on all columns at one y (any y)
  const column = (y) => {
    const out = Float64Array.from(xs, (x) => g(x, y));
    for (const [b, j1, j2] of rowKinks(y)) {
      const i = Math.floor((b - xs[0]) / dx);
      if (i < 0 || i + 1 >= nx) continue;
      const [cl, cr] = kinkCorr((b - xs[i]) / dx, dx, j1, j2);
      out[i] += cl;
      out[i + 1] += cr;
    }
    return out;
  };
  const P = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const c = column(ys[j]);
    for (let i = 0; i < nx; i++) P[i * ny + j] = c[i];
  }
  const eps = dy / 20;
  for (const c of colKinks) {
    const j = Math.floor((c - ys[0]) / dy);
    if (j < 0 || j + 1 >= ny) continue;
    const theta = (c - ys[j]) / dy;
    const [r1, r2] = oneSided(column, c, +1, eps);
    const [l1, l2] = oneSided(column, c, -1, eps);
    for (let i = 0; i < nx; i++) {
      const [cl, cr] = kinkCorr(theta, dy, r1[i] - l1[i], r2[i] - l2[i]);
      P[i * ny + j] += cl;
      P[i * ny + j + 1] += cr;
    }
  }
  return P;
}

// ---------- the job: image, kernel, output patch ----------

// Everything one 2D blur needs, laid out as in reference/make_2d_fixtures.py:
// the kernel, and the payoff image covering a patch of (2 qx + 1) x (2 qy + 1)
// spots around (S1, S2) plus the kernel's reach. Output (a, b) is the price at
// ln S1 + (a - qx) dx, ln S2 + (b - qy) dy; the spot is (qx, qy).
// payoff: a preset (with kinks) or a function g(x, y) with no corrections.
export function job2d({ payoff, S1, S2, s1, s2, rho, T, r, q1 = 0, q2 = 0, c = 0.06, width = 6, qx, qy, patch = 0.2, steps }) {
  const [dx, dy] = steps ?? gridSteps(s1, s2, rho, T, c);
  const K = kernel2d(dx, dy, s1, s2, rho, T, r, q1, q2, width);
  qx ??= Math.round(patch / dx);
  qy ??= Math.round(patch / dy);
  const px = (K.mx - 1) / 2, py = (K.my - 1) / 2;
  const nx = K.mx + 2 * qx, ny = K.my + 2 * qy;
  const x0 = Math.log(S1), y0 = Math.log(S2);
  const xs = Float64Array.from({ length: nx }, (_, i) => x0 + (i - px - qx) * dx);
  const ys = Float64Array.from({ length: ny }, (_, j) => y0 + (j - py - qy) * dy);
  const P = typeof payoff === "function"
    ? Float64Array.from({ length: nx * ny }, (_, t) => payoff(xs[Math.floor(t / ny)], ys[t % ny]))
    : samplePayoff2d(payoff, xs, ys);
  return { ...K, P, nx, ny, qx, qy, ox: 2 * qx + 1, oy: 2 * qy + 1, dx, dy, xs, ys, T, disc: Math.exp(-r * T) };
}

// The valid blur in 64-bit at output (a, b), discounted: the reference the GPU is checked against.
export function priceAt(job, a, b) {
  const { P, W, mx, my, ny, disc } = job;
  let s = 0;
  for (let i = 0; i < mx; i++) {
    let row = 0;
    const base = (a + i) * ny + b;
    for (let j = 0; j < my; j++) row += W[i * my + j] * P[base + j];
    s += row;
  }
  return disc * s;
}
