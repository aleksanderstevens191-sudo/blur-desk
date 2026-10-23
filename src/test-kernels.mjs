// node src/test-kernels.mjs — the JS fat-tailed kernels (kernels.js) against the
// Python reference (kernel_fixtures.json, from reference/make_kernel_fixtures.py),
// all in 64-bit on the CPU. The GPU half of the comparison is on /check.html.
import { readFileSync } from "node:fs";
import {
  gammaln, gammainc, gammaincc, gammainccinv, gridFor, kernel, kinkCorrection, blurPrice, bs, mertonSeries, vgPrice,
  impliedVol,
} from "./kernels.js";
import { DEMO_GRID } from "./smile.js";
import { Payoff, priceCurveFat, R } from "./pricer.js";

const { cases, special } = JSON.parse(readFileSync(new URL("./kernel_fixtures.json", import.meta.url)));
let fails = 0;
const line = (ok, s) => {
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s}`);
};
const e = (x) => x.toExponential(1);
const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));

console.log("special functions vs scipy (relative, or absolute below 1)");
let sf = 0;
for (const [x, v] of special.gammaln) sf = Math.max(sf, rel(gammaln(x), v));
for (const [a, x, P, Q] of special.gammainc) sf = Math.max(sf, Math.abs(gammainc(a, x) - P) / P, Math.abs(gammaincc(a, x) - Q) / Q);
for (const [a, q, v] of special.gammainccinv) sf = Math.max(sf, Math.abs(gammainccinv(a, q) - v) / v);
line(sf < 1e-13, `gammaln, gammainc, gammaincc, gammainccinv: max ${e(sf)}`);

// the blur at one node: sum_j w_j f[i + j - J]
const blurAt = (f, w, i, J) => {
  let s = 0;
  for (let j = 0; j < w.length; j++) s += w[j] * f[i + j - J];
  return s;
};
const grid = (J, dx) => Float64Array.from({ length: 2 * J + 1 }, (_, j) => (j - J) * dx);
const REF = { gaussian: (K, cp, p) => bs(100, K, p.sigma, p.T, p.r, cp), merton: (K, cp, p) => mertonSeries(100, K, cp, p), vg: (K, cp, p) => vgPrice(100, K, cp, p) };

for (const c of cases) {
  const { model, p, J, dx, S0, F, omax } = c;
  console.log(`\n${c.name}: ${2 * J + 1} nodes, dx ${dx.toPrecision(3)}`);
  const g = gridFor(model, p);
  line(Math.abs(g.dx - dx) <= 1e-15 * dx && g.J >= J && g.J <= 1.1 * J,
    `grid: JS half-width ${g.J} nodes, Python ${J} (JS keeps every Poisson term Python keeps, may keep more)`);

  const y = grid(J, dx);
  const t0 = performance.now();
  const w = kernel(model, y, p);
  const ms = performance.now() - t0;
  let sum = 0;
  let fwd = 0;
  for (let j = 0; j < w.length; j++) {
    sum += w[j];
    fwd += w[j] * Math.exp(y[j]);
  }
  const wErr = c.checks.idx.reduce((s, i, k) => Math.max(s, Math.abs(w[i] - c.checks.w[k])), 0) / c.checks.wmax;
  line(Math.abs(sum - c.checks.sum) < 1e-12 && Math.abs(fwd - c.checks.fwd) < 1e-12 && wErr < 2e-9,
    `kernel (${ms.toFixed(0)} ms): sum w ${e(sum - c.checks.sum)}, sum w e^y ${e(fwd - c.checks.fwd)} off Python;` +
    ` weights at ${c.checks.idx.length} nodes within ${e(wErr)} of the largest`);

  let dp = 0;
  let dref = 0;
  let gap = 0;
  for (const s of c.strikes) {
    dp = Math.max(dp, Math.abs(blurPrice(model, S0, s.K, s.cp, p, { y, w }) - s.py));
    dref = Math.max(dref, Math.abs(REF[model](s.K, s.cp, p) - s.ref));
    gap = Math.max(gap, Math.abs(s.py - s.ref));
  }
  line(dp < 1e-9 && dref < 1e-8, `${c.strikes.length} strikes: |JS blur - Python blur| ${e(dp)}, JS vs Python ${model === "gaussian" ? "Black-Scholes" : model === "merton" ? "Merton series" : "VG mixture"} ${e(dref)} (blur vs that formula: ${e(gap)})`);

  // the smile off one call row and one put row with strike F, on a grid omax wider
  // each side: node c - o is the option struck at F e^{o dx}, scaled by e^{-o dx}
  const n = 2 * (J + omax) + 1;
  const cen = J + omax;
  const z = Float64Array.from({ length: n }, (_, i) => Math.log(S0) + (i - cen) * dx);
  const rows = { call: kinkCorrection(z, F, "call"), put: kinkCorrection(z, F, "put") };
  let ds = 0;
  let dv = 0;
  for (const s of c.smile) {
    const v = Math.exp(-p.r * p.T) * Math.exp(s.o * dx) * blurAt(rows[s.cp], w, cen - s.o, J);
    ds = Math.max(ds, Math.abs(v - s.py));
    dv = Math.max(dv, Math.abs(impliedVol(v, S0, s.K, p.T, p.r, s.cp) - s.iv));
  }
  line(ds < 1e-9 && dv < 1e-8, `smile from one row: ${c.smile.length} strikes, |price - Python| ${e(ds)}, |implied vol - Python| ${e(dv)}`);

  // the demo's lighter grid (fewer nodes per sd, shorter tails)
  const d = gridFor(model, p, DEMO_GRID);
  const wd = kernel(model, d.y, p);
  let dd = 0;
  for (const s of c.strikes) dd = Math.max(dd, Math.abs(blurPrice(model, S0, s.K, s.cp, p, { y: d.y, w: wd }) - s.py));
  line(dd < 1e-5, `demo grid: ${d.y.length} nodes, dx ${d.dx.toPrecision(3)}, |price - Python| ${e(dd)}`);
}

// The demo's price curves (pricer.js, CPU path): a call and a put sampled on
// the demo grid (point values plus kink corrections), blurred with the demo's
// kernel, against Merton's series and the VG mixture at every 4th node, S 40-160.
console.log("\nthe demo's curves, 64-bit: priceCurveFat vs the formulas, S from 40 to 160");
const CURVES = [
  ["merton", { sigma: 0.3, lam: 1, m: -0.1, delta: 0.1 }, 1],
  ["merton", { sigma: 0.05, lam: 3, m: -0.3, delta: 0.3 }, 0.01],
  ["vg", { sigma: 0.3, theta: -0.2, nu: 0.2 }, 0.5],
  ["vg", { sigma: 0.2, theta: -0.3, nu: 0.5 }, 0.02],
];
for (const [model, p, T] of CURVES) {
  let worst = 0;
  for (const cp of ["call", "put"]) {
    const c = await priceCurveFat(new Payoff(cp === "call" ? (S) => Math.max(S - 100, 0) : (S) => Math.max(100 - S, 0)), { T, model, p, gpu: false });
    const pp = { ...p, T, r: R };
    const ref = model === "merton" ? mertonSeries : vgPrice;
    for (let i = 0; i < c.S.length; i += 4) worst = Math.max(worst, Math.abs(c.V[i] - ref(c.S[i], 100, cp, pp)));
  }
  line(worst < 1e-9, `${model.padEnd(7)} T=${String(T).padEnd(4)} ${JSON.stringify(p)}: max |curve - formula| ${e(worst)}`);
}

console.log(fails ? `\n${fails} FAILED` : "\nall kernel checks pass");
process.exit(fails ? 1 : 0);
