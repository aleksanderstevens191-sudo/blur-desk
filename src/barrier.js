// Barrier options in plain JavaScript (64-bit floats): closed forms, payoff
// sampling, and the CPU twin of the barrier shaders in gpu.js. Mirrors
// reference/barriers.py and reference/double_barrier.py; the GPU must match this.
import { normCdf } from "./bs.js";
import { gaussianKernel, blur } from "./blur.js";

export const BGK_BETA = 0.5825971579390106; // -zeta(1/2) / sqrt(2 pi)

// ---------- closed forms ----------

export function parseKind(kind) {
  const w = kind.replace(/-/g, " ").split(/\s+/);
  return { ud: w[0], io: w[2], cp: w[3] };
}

export function vanillaBS(S, K, sigma, T, r, q = 0, cp = "call") {
  if (T <= 0) return cp === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
  const v = sigma * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v;
  const d2 = d1 - v;
  if (cp === "call") return S * Math.exp(-q * T) * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
  return K * Math.exp(-r * T) * normCdf(-d2) - S * Math.exp(-q * T) * normCdf(-d1);
}

// Reiner-Rubinstein continuous single barriers, no rebate (reference/barriers.py).
export function barrierClosedForm(kind, S, K, H, sigma, T, r, q = 0) {
  const { ud, io, cp } = parseKind(kind);
  if ((ud === "down" && S <= H) || (ud === "up" && S >= H)) return io === "in" ? vanillaBS(S, K, sigma, T, r, q, cp) : 0;
  if (T <= 0) return io === "in" ? 0 : vanillaBS(S, K, sigma, 0, r, q, cp);
  const phi = cp === "call" ? 1 : -1;
  const eta = ud === "down" ? 1 : -1;
  const b = r - q;
  const sT = sigma * Math.sqrt(T);
  const mu = (b - 0.5 * sigma * sigma) / (sigma * sigma);
  const fwd = S * Math.exp((b - r) * T);
  const dK = K * Math.exp(-r * T);
  const x1 = Math.log(S / K) / sT + (1 + mu) * sT;
  const x2 = Math.log(S / H) / sT + (1 + mu) * sT;
  const y1 = Math.log((H * H) / (S * K)) / sT + (1 + mu) * sT;
  const y2 = Math.log(H / S) / sT + (1 + mu) * sT;
  const N = normCdf;
  const A = phi * fwd * N(phi * x1) - phi * dK * N(phi * x1 - phi * sT);
  const B = phi * fwd * N(phi * x2) - phi * dK * N(phi * x2 - phi * sT);
  const p1 = (H / S) ** (2 * (mu + 1));
  const p0 = (H / S) ** (2 * mu);
  const C = phi * fwd * p1 * N(eta * y1) - phi * dK * p0 * N(eta * y1 - eta * sT);
  const D = phi * fwd * p1 * N(eta * y2) - phi * dK * p0 * N(eta * y2 - eta * sT);
  const above = K > H;
  const key = `${ud} ${io} ${cp}`;
  return {
    "down in call": above ? C : A - B + D,
    "up in call": above ? A : B - C + D,
    "down in put": above ? B - C + D : A,
    "up in put": above ? A - B + D : C,
    "down out call": above ? A - C : B - D,
    "up out call": above ? 0 : A - B + C - D,
    "down out put": above ? A - B + C - D : 0,
    "up out put": above ? B - D : A - C,
  }[key];
}

// E[f(e^Y) 1{a < Y < b}], Y ~ N(m, s^2)
function gaussInterval(cp, K, a, b, m, s) {
  const k = Math.log(K);
  if (cp === "call") a = Math.max(a, k);
  else b = Math.min(b, k);
  if (b <= a) return 0;
  const ef = Math.exp(m + 0.5 * s * s);
  const pS = normCdf((b - m - s * s) / s) - normCdf((a - m - s * s) / s);
  const pK = normCdf((b - m) / s) - normCdf((a - m) / s);
  return cp === "call" ? ef * pS - K * pK : K * pK - ef * pS;
}

// Double knock-out by the image series, each Gaussian integral exact (reference/double_barrier.py).
export function dkoClosedForm(S, K, L, U, sigma, T, r, q = 0, cp = "call") {
  if (!(L < S && S < U)) return 0;
  if (T <= 0) return vanillaBS(S, K, sigma, 0, r, q, cp);
  const x = Math.log(S);
  const l = Math.log(L);
  const w = Math.log(U) - l;
  const nu = r - q - 0.5 * sigma * sigma;
  const a = (2 * nu) / (sigma * sigma);
  const s = sigma * Math.sqrt(T);
  const term = (n) =>
    Math.exp(a * n * w) * gaussInterval(cp, K, l, l + w, x + 2 * n * w + nu * T, s) -
    Math.exp(a * (l - x + n * w)) * gaussInterval(cp, K, l, l + w, 2 * l - x + 2 * n * w + nu * T, s);
  let total = term(0);
  for (let n = 1; n < 200; n++) {
    const t = term(n) + term(-n);
    total += t;
    if (Math.abs(t) < 1e-17 && n > 2) break;
  }
  return Math.exp(-r * T) * total;
}

// ---------- grids and sampling ----------

// z_j = anchor + (lo + j) dx, the same nodes as numpy's anchor + arange(lo, hi + 1) * dx.
export function gridZ({ anchor, lo, n, dx }) {
  return Float64Array.from({ length: n }, (_, j) => anchor + (lo + j) * dx);
}

// Call or put payoff on the grid. The kink at ln K costs the blur
// dx^2 B2(theta)/2 * K, which goes back onto the two nodes around it
// (Euler-Maclaurin; reference/barriers.py sample_payoff).
export function sampleVanilla(z, K, cp) {
  const n = z.length;
  const dx = z[1] - z[0];
  const v = Float64Array.from(z, (zi) => (cp === "call" ? Math.max(Math.exp(zi) - K, 0) : Math.max(K - Math.exp(zi), 0)));
  const k = Math.log(K);
  const j = Math.floor((k - z[0]) / dx);
  if (j >= 0 && j + 1 < n) {
    const th = (k - z[j]) / dx;
    const d = ((dx * (th * th - th + 1 / 6)) / 2) * K;
    v[j] += (1 - th) * d;
    v[j + 1] += th * d;
  }
  return v;
}

export function interpCubic(z0, dx, V, x, off = 0) {
  const j = Math.floor((x - z0) / dx) - 1;
  const t = (x - (z0 + j * dx)) / dx;
  const w = [-(t - 1) * (t - 2) * (t - 3) / 6, (t * (t - 2) * (t - 3)) / 2, (-t * (t - 1) * (t - 3)) / 2, (t * (t - 1) * (t - 2)) / 6];
  let s = 0;
  for (let k = 0; k < 4; k++) s += w[k] * V[off + j + k];
  return s;
}

// ---------- the hall of mirrors, as one blur ----------
//
// One "option" is one blur of an image-tiled payoff, written to a run of outputs:
//   { pOff, nP, dx }    payoff samples P[pOff .. pOff + nP), on a uniform log grid
//   { kOff, half }      kernel taps K[kOff .. kOff + 2 half], discount folded in
//   { outOff, nOut, out0 } outputs out[outOff + q] = price at node out0 + q
//   mode                MODE.down / up: single barrier at node lo (images 0 or 1)
//                       MODE.double: barriers at nodes lo < hi, images = cells kept each side
//                       MODE.none: no barrier; + MODE.IN for the knock-in (vanilla - knock-out)
//   a                   mirror weight exponent 2 nu / sigma^2 (0 = plain mirror)
// The shader in gpu.js evaluates exactly this g(j); keep the two in step.

export const MODE = { down: 0, up: 1, double: 2, none: 3, IN: 4 };
export const ALL_IMAGES = 1 << 20;

export function makeImage(o, P) {
  const n = o.nP;
  const b0 = o.pOff;
  const dx = o.dx;
  const a = o.a;
  const images = o.images ?? ALL_IMAGES;
  const pv = (j) => (j < 0 ? 2 * P[b0] - P[b0 - j] : j >= n ? 2 * P[b0 + n - 1] - P[b0 + 2 * (n - 1) - j] : P[b0 + j]);
  const slope = (j, dir) => (-3 * pv(j) + 4 * pv(j + dir) - pv(j + 2 * dir)) / (2 * dir * dx);
  // Euler-Maclaurin treatment of a jump at a node: midpoint + dx/12 * slope jump
  // on the node, +/- (value jump)/24 on its neighbours.
  const edge = (L, R) => ({ mid: 0.5 * (L[0] + R[0]) + (dx / 12) * (R[1] - L[1]), dip: (R[0] - L[0]) / 24 });
  const mode = o.mode & 3;
  const knockIn = (o.mode & 4) !== 0;
  let out;

  if (mode === 0 || mode === 1) {
    const b = o.lo;
    const dir = mode === 0 ? 1 : -1; // live side is j > b (down) or j < b (up)
    const fb = pv(b);
    const live = [fb, slope(b, dir)];
    const image = images >= 1 ? [-fb, a * fb + live[1]] : [0, 0];
    const e = dir > 0 ? edge(image, live) : edge(live, image);
    out = (j) => {
      if (j === b) return e.mid;
      let v;
      if ((j - b) * dir > 0) v = pv(j);
      else v = images >= 1 ? -Math.exp(-a * dx * (j - b)) * pv(2 * b - j) : 0;
      if (j === b + 1) v += e.dip;
      if (j === b - 1) v -= e.dip;
      return v;
    };
  } else if (mode === 2) {
    const l = o.lo;
    const m = o.hi - o.lo;
    const fl = pv(l);
    const sl = slope(l, 1);
    const fu = pv(o.hi);
    const su = slope(o.hi, -1);
    // value and slope where cell c ends (left or right)
    const cellEnd = (c, right) => {
      if (Math.abs(c) > images) return [0, 0];
      if (c % 2 === 0) {
        const E = Math.exp(a * dx * (-c / 2) * m);
        return right ? [E * fu, E * su] : [E * fl, E * sl];
      }
      const nn = (c + 1) / 2;
      if (!right) {
        const E = Math.exp(a * dx * (1 - nn) * m);
        return [-E * fu, E * (a * fu + su)];
      }
      const E = Math.exp(-a * dx * nn * m);
      return [-E * fl, E * (a * fl + sl)];
    };
    const edgeAt = (c) => edge(cellEnd(c - 1, true), cellEnd(c, false));
    out = (j) => {
      const d = j - l;
      const c = Math.floor(d / m);
      const r = d - c * m;
      if (r === 0) return edgeAt(c).mid;
      let v = 0;
      if (Math.abs(c) <= images) {
        if (c % 2 === 0) {
          const nn = -c / 2;
          v = Math.exp(a * dx * nn * m) * pv(j + 2 * nn * m);
        } else {
          const nn = (c + 1) / 2;
          v = -Math.exp(a * dx * (l + nn * m - j)) * pv(2 * l + 2 * nn * m - j);
        }
      }
      if (r === 1) v += edgeAt(c).dip;
      if (r === m - 1) v -= edgeAt(c + 1).dip;
      return v;
    };
  } else {
    return pv;
  }
  return knockIn ? (j) => pv(j) - out(j) : out;
}

export function barrierBlurCPU({ P, K, opts }) {
  const total = opts.reduce((s, o) => Math.max(s, o.outOff + o.nOut), 0);
  const res = new Float64Array(total);
  for (const o of opts) {
    const g = makeImage(o, P);
    const m = 2 * o.half + 1;
    for (let q = 0; q < o.nOut; q++) {
      const base = o.out0 + q - o.half;
      let s = 0;
      for (let t = 0; t < m; t++) s += K[o.kOff + t] * g(base + t);
      res[o.outOff + q] = s;
    }
  }
  return res;
}

// The image-tiled payoff itself, for drawing the hall of mirrors.
export function imagePayoff(o, P, j0, j1) {
  const g = makeImage(o, P);
  return Float64Array.from({ length: j1 - j0 }, (_, k) => g(j0 + k));
}

// ---------- step by step: blur, discount, knock out, repeat ----------
//
// Rows [vanilla, knock-out, knock-in] on one grid. lo / hi are the barrier
// nodes (-1 if absent); past them the knock-out is 0 and the knock-in takes the
// vanilla value. The jump this makes at the barrier node gets the same
// midpoint + Euler-Maclaurin treatment as a sampled payoff.

export function monitorRows(Y, n, lo, hi, dx) {
  const out = new Float64Array(3 * n);
  out.set(Y.subarray(0, n), 0);
  for (const r of [1, 2]) {
    const A = (i) => Y[r * n + i];
    const B = (i) => (r === 1 ? 0 : Y[i]);
    for (let i = 0; i < n; i++) {
      const dead = (lo >= 0 && i < lo) || (hi >= 0 && i > hi);
      out[r * n + i] = dead ? B(i) : A(i);
    }
    for (const [b, sgn] of [[lo, 1], [hi, -1]]) {
      if (b < 0) continue;
      const D = (i) => sgn * (A(i) - B(i));
      const dv = D(b);
      const ds = (D(b + 1) - D(b - 1)) / (2 * dx);
      out[r * n + b] = 0.5 * (A(b) + B(b)) + (dx / 12) * ds;
      out[r * n + b + 1] += dv / 24;
      out[r * n + b - 1] -= dv / 24;
    }
  }
  return out;
}

// P: vanilla payoff samples (n); kernel: one step's taps with the discount folded in.
// snaps: [{ step, out0, nOut }] copies all three rows' [out0, out0 + nOut) after that
// many steps (before that date's knock-out). Returns { rows, snaps }.
export function stepBarrierCPU({ P, kernel, steps, lo, hi, dx, snaps = [] }) {
  const n = P.length;
  let Y = new Float64Array(3 * n);
  Y.set(P, 0);
  Y.set(P, n);
  Y = monitorRows(Y, n, lo, hi, dx);
  const shots = [];
  for (let s = 1; s <= steps; s++) {
    const X = new Float64Array(3 * n);
    for (let r = 0; r < 3; r++) X.set(blur(Y.subarray(r * n, (r + 1) * n), kernel), r * n);
    for (const sn of snaps) {
      if (sn.step !== s) continue;
      const shot = new Float64Array(3 * sn.nOut);
      for (let r = 0; r < 3; r++) shot.set(X.subarray(r * n + sn.out0, r * n + sn.out0 + sn.nOut), r * sn.nOut);
      shots.push(shot);
    }
    Y = s < steps ? monitorRows(X, n, lo, hi, dx) : X;
  }
  return { rows: Y, snaps: shots };
}

// ---------- building blocks for callers ----------

// Taps for a blur of length T with the discount folded in.
export function discountedKernel(dx, sigma, T, r, q = 0) {
  const k = gaussianKernel(dx, sigma * Math.sqrt(T), (r - q - 0.5 * sigma * sigma) * T);
  const disc = Math.exp(-r * T);
  for (let i = 0; i < k.length; i++) k[i] *= disc;
  return k;
}

export function mirrorExponent(sigma, r, q = 0) {
  return (2 * (r - q - 0.5 * sigma * sigma)) / (sigma * sigma);
}
