// /check.html?only=chain: every contract of the SPY snapshot priced in one GPU
// submit (chain.js) against the Python reference prices (chain_data.json, from
// reference/chain_fit.py). Black-Scholes at each contract's own vol vs QuantLib,
// Merton per expiry vs the series; also Black-Scholes vs the market mid.
import data from "./chain_data.json";
import { chainJobs, chainJobGPU } from "./chain.js";

export async function chainChecks(log, TOL) {
  let fails = 0;
  const C = data.contracts;
  const n = C.K.length;
  const t0 = performance.now();
  const J = chainJobs(data);
  const t1 = performance.now();
  const job = await chainJobGPU(J);
  const t2 = performance.now();
  const out = await job.read();
  const t3 = performance.now();
  const { bs, merton } = J.finish(out);
  const t4 = performance.now();
  job.destroy();
  log(`${n} contracts, ${data.expiries.length} expiries, SPY ${data.quote_time}: ${J.nJobs} jobs in one dispatch ` +
    `(${J.info.mertonNodes} Merton strike-row nodes, ${(J.info.mertonTaps / 1e6).toFixed(1)}M taps; ` +
    `${n} Black-Scholes kernels, ${(J.info.bsTaps / 1e6).toFixed(2)}M taps)`);
  log(`build ${(t1 - t0).toFixed(0)} ms (CPU, 64-bit), upload + pipeline ${(t2 - t1).toFixed(0)} ms, ` +
    `submit + read back ${(t3 - t2).toFixed(0)} ms, finish ${(t4 - t3).toFixed(0)} ms. Pass: |gpu - ref| < ${TOL} dollars on every contract.`);
  const mid = C.bid.map((b, i) => 0.5 * (b + C.ask[i]));
  const check = (name, got, ref, tol) => {
    let d = 0;
    let at = 0;
    let rel = 0;
    for (let i = 0; i < n; i++) {
      const g = Math.abs(got[i] - ref[i]);
      if (!(g <= d)) { d = g; at = i; }
      rel = Math.max(rel, g / Math.max(Math.abs(ref[i]), 1e-2));
    }
    const ok = d < tol;
    if (!ok) fails++;
    const ex = data.expiries[C.e[at]].expiry;
    log(`${ok ? "PASS" : "FAIL"} ${name}: max |gpu - ref| ${d.toExponential(2)} dollars ` +
      `(worst ${ex} ${C.K[at]} ${C.call[at] ? "call" : "put"} at ${ref[at].toFixed(4)})`);
  };
  check("Black-Scholes, own implied vol, vs QuantLib", bs, C.bs_ql, TOL);
  check("Black-Scholes, own implied vol, vs the market mid", bs, mid, TOL);
  check("Merton fitted per expiry vs the Merton series", merton, C.merton, TOL);
  return fails;
}
