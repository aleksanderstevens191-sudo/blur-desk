// Fat-tailed blur kernels in JavaScript: a line-by-line port of
// reference/kernels.py (Gaussian, Merton jump-diffusion, variance gamma).
//
// The price is one blur of the payoff in log-price with the model's density of
// X = ln(S_T / S_0). Each kernel weight is the exact integral of the density
// against a "tent" made of 1 and e^y, so sum w = 1 and sum w e^y = e^{rT} by
// construction, and a call or put is blurred exactly except in the strike cell
// (kinkCorrection). The tents need only F0(e) = P(X <= e) and
// F1(e) = E[e^X; X <= e] at the nodes. Parameters, as in Python:
//   { T, r, sigma, lam, m, delta }  Merton: jumps a year, mean and sd of a jump in ln S
//   { T, r, sigma, theta, nu }      variance gamma: drift and variance rate of the gamma clock
import { normCdf } from "./bs.js";

export const MODELS = ["gaussian", "merton", "vg"];

// ---------- special functions ----------

// ln Gamma(x), Lanczos (g = 7, 9 terms): ~1e-15 relative for x > 0
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];
export function gammaln(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - gammaln(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// Regularized incomplete gamma: P(a, x) by its series below x = a + 1, Q(a, x)
// by its continued fraction (modified Lentz) above, each the other's complement.
function gser(a, x) {
  let ap = a;
  let del = 1 / a;
  let sum = del;
  for (let n = 0; n < 1000; n++) {
    ap += 1;
    del *= x / ap;
    sum += del;
    if (Math.abs(del) < Math.abs(sum) * 1e-17) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - gammaln(a));
}
function gcf(a, x) {
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-17) break;
  }
  return Math.exp(-x + a * Math.log(x) - gammaln(a)) * h;
}
export function gammainc(a, x) {
  if (x <= 0) return 0;
  return x < a + 1 ? gser(a, x) : 1 - gcf(a, x);
}
export function gammaincc(a, x) {
  if (x <= 0) return 1;
  return x < a + 1 ? 1 - gser(a, x) : gcf(a, x);
}
// x with Q(a, x) = q: bracket, then bisect in ln x (Q falls monotonically)
export function gammainccinv(a, q) {
  let lo = Math.log(Math.max(1e-300, a * 1e-3));
  let hi = Math.log(a + 10);
  while (gammaincc(a, Math.exp(lo)) < q) lo -= 2;
  while (gammaincc(a, Math.exp(hi)) > q) hi += 1;
  for (let i = 0; i < 200 && hi - lo > 1e-15 * Math.max(1, Math.abs(hi)); i++) {
    const mid = 0.5 * (lo + hi);
    if (gammaincc(a, Math.exp(mid)) > q) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

// Gauss-Legendre nodes and weights on [-1, 1] (Newton on P_n)
export function gaussLegendre(n) {
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  for (let i = 0; i < Math.ceil(n / 2); i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let dp = 0;
    for (let it = 0; it < 100; it++) {
      let p1 = 1;
      let p2 = 0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      dp = (n * (z * p1 - p2)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
  }
  return { x, w };
}
const GL16 = gaussLegendre(16);
const GL12 = gaussLegendre(12);

// ---------- model parameters, drift corrections, moments ----------

const P0 = { lam: 0, m: 0, delta: 0, theta: 0, nu: 0 };

// Mean of the diffusion part of X (the risk-neutral drift correction).
export function drift(model, p) {
  const { T, r, sigma, lam, m, delta, theta, nu } = { ...P0, ...p };
  if (model === "gaussian") return (r - 0.5 * sigma * sigma) * T;
  if (model === "merton") return (r - 0.5 * sigma * sigma - lam * Math.expm1(m + 0.5 * delta * delta)) * T;
  if (model === "vg") return (r + Math.log1p(-theta * nu - 0.5 * sigma * sigma * nu) / nu) * T;
  throw new Error(`unknown model ${model}`);
}

// [mean, sd] of X
export function meanSd(model, p) {
  const { T, sigma, lam, m, delta, theta, nu } = { ...P0, ...p };
  const mu = drift(model, p);
  if (model === "gaussian") return [mu, sigma * Math.sqrt(T)];
  if (model === "merton") return [mu + lam * T * m, Math.sqrt(sigma * sigma * T + lam * T * (m * m + delta * delta))];
  return [mu + theta * T, Math.sqrt((sigma * sigma + theta * theta * nu) * T)];
}

// [n, P(N = n)] for N ~ Poisson(L), past the mean until P(N = n) < tail / 1000.
// (kernels.py also stops once its running total rounds to 1, which hangs on the
// last bit; here only P(N = n) decides, so the list is never the shorter one.)
export function poissonTerms(L, tail = 1e-18) {
  if (L === 0) return [[0, 1]];
  const out = [];
  for (let n = 0; ; n++) {
    const pn = Math.exp(n * Math.log(L) - L - gammaln(n + 1));
    out.push([n, pn]);
    if (n > L && pn < tail * 1e-3) return out;
  }
}

// ---------- the cumulative functions F0, F1 at the nodes ----------

// adds wt * (P(X <= e), E[e^X; X <= e]) for X ~ N(mean, sd^2)
function addGaussF(e, mean, sd, wt, F0, F1) {
  const c = wt * Math.exp(mean + 0.5 * sd * sd);
  for (let i = 0; i < e.length; i++) {
    const z = (e[i] - mean) / sd;
    F0[i] += wt * normCdf(z);
    F1[i] += c * normCdf(z - sd);
  }
}

export function cumulative(model, e, p, gTiny = null) {
  const { T, sigma, lam, m, delta, theta, nu } = { ...P0, ...p };
  const mu = drift(model, p);
  const F0 = new Float64Array(e.length);
  const F1 = new Float64Array(e.length);
  if (model === "gaussian") {
    addGaussF(e, mu, sigma * Math.sqrt(T), 1, F0, F1);
  } else if (model === "merton") {
    for (const [n, pn] of poissonTerms(lam * T)) addGaussF(e, mu + n * m, Math.sqrt(sigma * sigma * T + n * delta * delta), pn, F0, F1);
  } else {
    // variance gamma: given G = g, X ~ N(mu + theta g, sigma^2 g), G ~ Gamma(T/nu, scale nu);
    // tilting by e^X keeps G gamma with scale nu / tilt and moves the mean by sigma^2 g
    const a = T / nu;
    gTiny ??= ((1e-3 * (e[1] - e[0])) / sigma) ** 2; // Gaussian width a thousandth of a cell
    const tilt = 1 - (theta + 0.5 * sigma * sigma) * nu;
    const total = Math.exp(mu) * tilt ** -a; // E[e^X] = e^{rT}
    const d = Float64Array.from(e, (v) => v - mu);
    vgMix(d, a, nu, theta, sigma, gTiny, F0);
    vgMix(d, a, nu / tilt, theta + sigma * sigma, sigma, gTiny, F1);
    for (let i = 0; i < e.length; i++) F1[i] *= total;
  }
  return { F0, F1 };
}

// out[i] = E[Phi((d_i - b G) / (sigma sqrt G))], G ~ Gamma(a, scale), on a uniform
// d: Gauss-Legendre panels in ln g, plus the mass below gTiny as a step at d = 0
// (as kernels.py). Quadrature nodes carrying under 1e-30 of weight are skipped,
// and Phi is taken as exactly 0 or 1 past 9 sd (both below 1e-18 of a price), so
// each node only evaluates Phi on the nodes d within 9 sd of its centre b g.
function vgMix(d, a, scale, b, sigma, gTiny, out, panel = 0.25, tail = 1e-18) {
  const lo = Math.log(gTiny);
  const hi = Math.log(scale * gammainccinv(a, tail));
  const n = Math.max(1, Math.ceil((hi - lo) / panel));
  const h = (0.5 * (hi - lo)) / n;
  const norm = a * Math.log(scale) + gammaln(a);
  const nd = d.length;
  const d0 = d[0];
  const dd = (d[nd - 1] - d[0]) / (nd - 1);
  const above = new Float64Array(nd + 1); // weight of the nodes with Phi = 1 from index i on
  out.fill(0);
  for (let k = 0; k < n; k++) {
    const e0 = lo + (k * (hi - lo)) / n;
    for (let q = 0; q < 16; q++) {
      const L = e0 + h * (GL16.x[q] + 1);
      const g = Math.exp(L);
      const wt = h * GL16.w[q] * Math.exp(a * L - g / scale - norm);
      if (wt < 1e-30) continue;
      const c = b * g;
      const s = sigma * Math.sqrt(g);
      const i0 = Math.max(0, Math.floor((c - 9 * s - d0) / dd) - 1);
      const i1 = Math.min(nd - 1, Math.ceil((c + 9 * s - d0) / dd) + 1);
      for (let i = i0; i <= i1; i++) {
        const z = (d[i] - c) / s;
        if (z > 9) out[i] += wt;
        else if (z > -9) out[i] += wt * normCdf(z);
      }
      above[Math.max(0, i1 + 1)] += wt;
    }
  }
  const step = gammainc(a, gTiny / scale);
  let run = 0;
  for (let i = 0; i < nd; i++) {
    run += above[i];
    out[i] += run + (d[i] > 0 ? step : 0);
  }
}

// ---------- the kernel ----------

// How far from 0 the grid must reach so the mass left off it (also weighted by
// e^y, for the forward) is below ~tail.
export function halfWidth(model, p, { nSd = 13, tail = 1e-18 } = {}) {
  const { T, sigma, lam, m, delta, theta, nu } = { ...P0, ...p };
  const mu = drift(model, p);
  if (model === "gaussian") return Math.abs(mu) + sigma * sigma * T + nSd * sigma * Math.sqrt(T);
  if (model === "merton") {
    let w = 0;
    for (const [n, pn] of poissonTerms(lam * T)) {
      const v = sigma * sigma * T + n * delta * delta;
      if (pn > tail) w = Math.max(w, Math.abs(mu + n * m) + v + nSd * Math.sqrt(v));
    }
    return w;
  }
  // variance gamma tails are exponential, rates (c -/+ theta) / sigma^2, independent of T
  const c = Math.sqrt((2 * sigma * sigma) / nu + theta * theta);
  const rate = Math.min((c + theta) / (sigma * sigma), (c - theta) / (sigma * sigma) - 1);
  const sd = meanSd(model, p)[1];
  return Math.abs(mu) + Math.max(nSd * sd, -Math.log(tail) / rate);
}

// Offsets y_j = j dx, j = -J..J. dx resolves the narrowest piece of the kernel:
// the diffusion alone for Merton (no jumps with probability e^{-lam T}).
// maxHalf caps the half-width (the demo's extreme sliders; kernels.py has no cap).
export function gridFor(model, p, { dx = null, ptsPerSd = 200, dxMax = 0.002, nSd = 13, tail = 1e-18, maxHalf = Infinity } = {}) {
  if (dx === null) {
    const sd = model === "merton" ? p.sigma * Math.sqrt(p.T) : meanSd(model, p)[1];
    dx = Math.min(sd / ptsPerSd, dxMax);
  }
  const J = Math.ceil(Math.min(halfWidth(model, p, { nSd, tail }), maxHalf) / dx);
  const y = Float64Array.from({ length: 2 * J + 1 }, (_, j) => (j - J) * dx);
  return { y, dx, J };
}

// Tent weights on the offsets y: w_j = int p(y) [tent of node j](y) dy, where on
// each cell the two tents are the combinations of 1 and e^y that are 1 at one end
// and 0 at the other.
export function kernel(model, y, p) {
  const { F0, F1 } = cumulative(model, y, p);
  const w = new Float64Array(y.length);
  let eL = Math.exp(y[0]);
  for (let j = 0; j + 1 < y.length; j++) {
    const eR = Math.exp(y[j + 1]);
    const M0 = F0[j + 1] - F0[j];
    const M1 = F1[j + 1] - F1[j];
    const den = eR - eL;
    w[j] += (eR * M0 - M1) / den;
    w[j + 1] += (M1 - eL * M0) / den;
    eL = eR;
  }
  return w;
}

// ---------- payoffs and the blur ----------

export const payoff = (z, K, cp) => (cp === "call" ? Math.max(Math.exp(z) - K, 0) : Math.max(K - Math.exp(z), 0));

function gl(fn, a, b) {
  if (b <= a) return 0;
  let s = 0;
  for (let q = 0; q < 12; q++) s += GL12.w[q] * fn(0.5 * (b - a) * GL12.x[q] + 0.5 * (a + b));
  return 0.5 * (b - a) * s;
}

// Payoff samples on the nodes z, plus the correction on the two nodes around the
// strike that makes the tents' blur exact there too, to O(dx^4), for any kernel
// smooth across the cell. Kernel-independent: one row serves every model.
export function kinkCorrection(z, K, cp) {
  const dx = z[1] - z[0];
  const f = Float64Array.from(z, (v) => payoff(v, K, cp));
  const k = Math.log(K);
  const j = Math.floor((k - z[0]) / dx);
  if (j < 0 || j + 1 >= z.length) return f;
  const zl = z[j];
  const zr = z[j + 1];
  const el = Math.exp(zl);
  const er = Math.exp(zr);
  const err = (t) => payoff(t, K, cp) - (f[j] * (er - Math.exp(t)) + f[j + 1] * (Math.exp(t) - el)) / (er - el);
  const zm = 0.5 * (zl + zr);
  const mom = (t) => (t - zm) * err(t);
  const A = gl(err, zl, k) + gl(err, k, zr);
  const B = gl(mom, zl, k) + gl(mom, k, zr);
  const s = A / dx;
  const d = (2 * B) / (dx * dx);
  f[j] += 0.5 * (s - d);
  f[j + 1] += 0.5 * (s + d);
  return f;
}

// The same correction for a unit kink at K on the uniform grid z_p = z0 + p dx:
// [j, d_j, d_{j+1}], what kinkCorrection adds for a call (or put) struck at K.
// The corrections are linear in the payoff, so a payoff that is straight in S
// between kinks (a drawn one) gets its slope change times this at each kink.
export function kinkPair(z0, dx, K) {
  const k = Math.log(K);
  const j = Math.floor((k - z0) / dx);
  const zl = z0 + j * dx;
  const zr = z0 + (j + 1) * dx;
  const el = Math.exp(zl);
  const er = Math.exp(zr);
  const fr = Math.max(er - K, 0);
  const err = (t) => Math.max(Math.exp(t) - K, 0) - (Math.max(el - K, 0) * (er - Math.exp(t)) + fr * (Math.exp(t) - el)) / (er - el);
  const zm = 0.5 * (zl + zr);
  const mom = (t) => (t - zm) * err(t);
  const s = (gl(err, zl, k) + gl(err, k, zr)) / dx;
  const d = (2 * (gl(mom, zl, k) + gl(mom, k, zr))) / (dx * dx);
  return [j, 0.5 * (s - d), 0.5 * (s + d)];
}

// European option by one blur of the payoff with the model's kernel, at S0 (CPU, 64-bit)
export function blurPrice(model, S0, K, cp, p, { y = null, w = null } = {}) {
  y ??= gridFor(model, p).y;
  w ??= kernel(model, y, p);
  const f = kinkCorrection(Float64Array.from(y, (v) => Math.log(S0) + v), K, cp);
  let s = 0;
  for (let j = 0; j < w.length; j++) s += w[j] * f[j];
  return Math.exp(-p.r * p.T) * s;
}

// ---------- independent prices ----------

export function bs(S, K, sigma, T, r, cp) {
  const v = sigma * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v;
  const d2 = d1 - v;
  return cp === "call" ? S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2) : K * Math.exp(-r * T) * normCdf(-d2) - S * normCdf(-d1);
}

// Merton (1976): Poisson-weighted Black-Scholes prices
export function mertonSeries(S, K, cp, p, tail = 1e-18) {
  const { T, r, sigma, lam, m, delta } = { ...P0, ...p };
  const kappa = Math.expm1(m + 0.5 * delta * delta);
  let total = 0;
  for (const [n, pn] of poissonTerms(lam * (1 + kappa) * T, tail)) {
    const sn = Math.sqrt(sigma * sigma + (n * delta * delta) / T);
    const rn = r - lam * kappa + (n * (m + 0.5 * delta * delta)) / T;
    total += pn * bs(S, K, sn, T, rn, cp);
  }
  return total;
}

// Variance gamma as Madan-Carr-Chang price it: a Black-Scholes-type price given
// the gamma clock G = g, averaged over G. Gauss-Legendre panels in ln g (Python
// integrates the same thing adaptively); no grid, no kernel.
export function vgPrice(S, K, cp, p) {
  const { T, r, sigma, theta, nu } = { ...P0, ...p };
  const a = T / nu;
  const mu = drift("vg", p);
  const hi = Math.log(nu * gammainccinv(a, 1e-20));
  const lo = Math.min(-60, hi - 60);
  const n = Math.ceil((hi - lo) / 0.25);
  const h = (0.5 * (hi - lo)) / n;
  const norm = a * Math.log(nu) + gammaln(a);
  let v = 0;
  for (let k = 0; k < n; k++) {
    const e0 = lo + (k * (hi - lo)) / n;
    for (let q = 0; q < 16; q++) {
      const L = e0 + h * (GL16.x[q] + 1);
      const g = Math.exp(L);
      const wt = h * GL16.w[q] * Math.exp(a * L - g / nu - norm);
      if (wt < 1e-30) continue;
      const s = sigma * Math.sqrt(g);
      const F = S * Math.exp(mu + theta * g + 0.5 * s * s);
      const d1 = (Math.log(F / K) + 0.5 * s * s) / s;
      const d2 = d1 - s;
      v += wt * (cp === "call" ? F * normCdf(d1) - K * normCdf(d2) : K * normCdf(-d2) - F * normCdf(-d1));
    }
  }
  // G below e^lo: the Gaussian is a point, the payoff is taken at S e^mu
  v += gammainc(a, Math.exp(lo) / nu) * payoff(Math.log(S) + mu, K, cp);
  return Math.exp(-r * T) * v;
}

// ---------- implied volatility ----------

// Brent's method on [a, b] with f(a), f(b) of opposite signs
function brent(f, a, b, tol = 1e-15) {
  let fa = f(a);
  let fb = f(b);
  let c = a;
  let fc = fa;
  let d = b - a;
  let e = d;
  for (let it = 0; it < 200; it++) {
    if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
      c = a;
      fc = fa;
      e = d = b - a;
    }
    if (Math.abs(fc) < Math.abs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    const tol1 = 2 * Number.EPSILON * Math.abs(b) + 0.5 * tol;
    const xm = 0.5 * (c - b);
    if (Math.abs(xm) <= tol1 || fb === 0) return b;
    if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
      const s = fb / fa;
      let pp;
      let q;
      if (a === c) {
        pp = 2 * xm * s;
        q = 1 - s;
      } else {
        const qq = fa / fc;
        const rr = fb / fc;
        pp = s * (2 * xm * qq * (qq - rr) - (b - a) * (rr - 1));
        q = (qq - 1) * (rr - 1) * (s - 1);
      }
      if (pp > 0) q = -q;
      pp = Math.abs(pp);
      if (2 * pp < Math.min(3 * xm * q - Math.abs(tol1 * q), Math.abs(e * q))) {
        e = d;
        d = pp / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    a = b;
    fa = fb;
    b += Math.abs(d) > tol1 ? d : xm > 0 ? tol1 : -tol1;
    fb = f(b);
  }
  return b;
}

// Black-Scholes implied volatility; NaN outside the no-arbitrage bounds
export function impliedVol(price, S0, K, T, r, cp) {
  const df = Math.exp(-r * T);
  const intrinsic = cp === "call" ? Math.max(S0 - K * df, 0) : Math.max(K * df - S0, 0);
  if (!(intrinsic + 1e-14 < price && price < (cp === "call" ? S0 : K * df))) return NaN;
  const f = (s) => bs(S0, K, s, T, r, cp) - price;
  if (f(1e-4) > 0 || f(4) < 0) return NaN;
  return brent(f, 1e-4, 4);
}
