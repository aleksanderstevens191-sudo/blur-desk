// /check.html, diffusion section (?only=diffusion): panel 16 on the GPU (gpu-diffusion.js)
// against the Python fixtures (reference/make_diffusion_fixtures.py): the market's
// density blurred toward noise and its score, the call blurred by the same Gaussian,
// and the reverse sampler's points. Then 65,536 random points run from noise back to
// the market's odds, tested against them (KS).
import fx from "./diffusion_fixtures.json";
import SPY from "./spy_fit.json";
import { PANEL, market, grid, levels, mixture, mixCdf, startPoints, ksTest } from "./diffusion.js";
import { diffusionGPU, payoffGPU, KEEP } from "./gpu-diffusion.js";

const e = (x) => x.toExponential(2);

export async function diffusionChecks(log, TOL) {
  let fails = 0;
  const line = (ok, s) => {
    if (!ok) fails++;
    log(`${ok ? "PASS" : "FAIL"} ${s}`);
  };
  const P = market(SPY);
  const G = grid(P);
  const lev = levels();
  log(`${fx.expiry}: ${G.n} nodes, dx ${PANEL.dx}, ${PANEL.N + 1} noise levels ${PANEL.smin} -> ${PANEL.smax} (log-return units)`);
  const run = await diffusionGPU(G, lev);
  log(`every level's blurred density and score in one dispatch: ${run.ms.toFixed(0)} ms`);
  for (const row of fx.density) {
    let dq = 0;
    row.q.forEach((v, k) => {
      const i = k * fx.stride;
      dq = Math.max(dq, Math.abs(run.Q[row.j * G.n + i] - v));
    });
    line(dq < TOL, `density, level ${String(row.j).padStart(3)} (sigma ${row.sigma.toFixed(4)}): |gpu - py| ${e(dq)} per unit of y (peak ${Math.max(...row.q).toFixed(3)})`);
  }
  for (const row of fx.payoff.rows) {
    const { prices, ms } = await payoffGPU(fx.payoff.x, row.s, fx.K);
    const d = Math.max(...row.price.map((v, k) => Math.abs(prices[k] - v)));
    line(d < TOL, `payoff, level ${String(row.j).padStart(3)} (sd ${row.s.toFixed(4)}): call K $${fx.K_usd} at 12 spots, |gpu - py| ${e(d)} per $100 of forward (${ms.toFixed(1)} ms)`);
  }
  const x1 = Float32Array.from(fx.sampler.x1);
  const sr = await run.sampler(x1);
  const d64 = Math.max(...fx.sampler.level64.map((v, i) => Math.abs(sr.levels[64 / KEEP][i] - v)));
  const d0 = Math.max(...fx.sampler.x0.map((v, i) => Math.abs(sr.levels[0][i] - v)));
  line(d64 < TOL && d0 < TOL, `sampler, 2,048 quantiles of the noise run back: |gpu - py| ${e(d64)} at level 64, ${e(d0)} at level 0 (log-return; ${sr.ms.toFixed(0)} ms)`);
  const comps = mixture(P);
  const xr = startPoints(65536, P);
  const rr = await run.sampler(xr);
  const ks = ksTest(rr.levels[0], (v) => mixCdf(v, comps));
  const down = Math.log((0.9 * P.spot) / P.S);
  const pd = rr.levels[0].reduce((a, v) => a + (v < down), 0) / 65536;
  line(ks.p > 0.01, `65,536 random points of noise run back to the market's odds: KS D ${ks.D.toFixed(4)}, p ${ks.p.toFixed(2)}; ` +
    `P(fall > 10%) ${(100 * pd).toFixed(2)}% vs the market's ${(100 * mixCdf(down, comps)).toFixed(3)}% (${rr.ms.toFixed(0)} ms)`);
  run.destroy();
  return fails;
}
