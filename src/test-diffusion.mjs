// node src/test-diffusion.mjs — panel 16's math (diffusion.js) in 64-bit against the
// Python fixtures (reference/make_diffusion_fixtures.py): the blurred market density and
// its score, the blurred call, and the reverse sampler's points. The GPU (32-bit) half
// is /check.html?only=diffusion.
import { readFileSync } from "node:fs";
import { PANEL, market, grid, levels, blurCPU, scoreOf, samplerCPU, payoffCPU, blackCall } from "./diffusion.js";

const fx = JSON.parse(readFileSync(new URL("./diffusion_fixtures.json", import.meta.url)));
const SPY = JSON.parse(readFileSync(new URL("./spy_fit.json", import.meta.url)));
const TOL = 1e-10;
const e = (x) => x.toExponential(1);
let fails = 0;
const line = (ok, s) => {
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s}`);
};

const P = market(SPY);
const G = grid(P);
const { sig, s } = levels();
console.log(`${fx.expiry}: grid ${G.n} nodes, dx ${PANEL.dx}; ${PANEL.N + 1} noise levels ${PANEL.smin} -> ${PANEL.smax}\n`);

const t0 = performance.now();
const Q = [], SC = [];
for (let j = 0; j <= PANEL.N; j++) {
  Q[j] = blurCPU(G.p0, s[j]);
  SC[j] = scoreOf(Q[j], blurCPU(G.dp, s[j]));
}
console.log(`129 levels of blurred density and score in ${((performance.now() - t0) / 1e3).toFixed(1)} s`);

for (const row of fx.density) {
  let dq = 0, ds = 0;
  row.q.forEach((v, n) => {
    const i = n * fx.stride;
    dq = Math.max(dq, Math.abs(Q[row.j][i] - v));
    if (v > 1e-4) ds = Math.max(ds, Math.abs(SC[row.j][i] - row.score[n]) / Math.max(1, Math.abs(row.score[n])));
  });
  line(dq < TOL && ds < 1e-8, `density, level ${String(row.j).padStart(3)} (sigma ${row.sigma.toFixed(4)}): |js - py| ${e(dq)} per unit y, score ${e(ds)} (relative, where q > 1e-4)`);
}
for (const row of fx.payoff.rows) {
  let d = 0, bk = 0;
  fx.payoff.x.forEach((x, n) => {
    const v = payoffCPU(x, row.s, fx.K);
    d = Math.max(d, Math.abs(v - row.price[n]));
    bk = Math.max(bk, Math.abs(v - blackCall(x, row.s, fx.K)));
  });
  line(d < TOL, `payoff, level ${String(row.j).padStart(3)} (sd ${row.s.toFixed(4)}): |js - py| ${e(d)} per $100; vs Black closed form ${e(bk)}`);
}
const t1 = performance.now();
const run = samplerCPU(fx.sampler.x1, SC, sig);
let d64 = 0, d0 = 0;
fx.sampler.x0.forEach((v, i) => {
  d0 = Math.max(d0, Math.abs(run[0][i] - v));
  d64 = Math.max(d64, Math.abs(run[64][i] - fx.sampler.level64[i]));
});
line(d0 < 1e-9 && d64 < 1e-9, `sampler, 2,048 points: |js - py| ${e(d64)} at level 64, ${e(d0)} at level 0 (log-return; ${((performance.now() - t1) / 1e3).toFixed(1)} s)`);
console.log(`\nthe run's own accuracy (Python, 20,000 quantiles): CDF gap ${e(fx.quality.cdf_gap)}, L1 ${e(fx.quality.l1)}`);
console.log(fails ? `\n${fails} failed` : "\nall panel 16 checks pass (64-bit)");
process.exit(fails ? 1 : 0);
