// node src/test-chain.mjs — the full-chain pricer (chain.js) run in 64-bit on the
// CPU against the Python reference prices in chain_data.json (reference/chain_fit.py):
// Black-Scholes at each contract's own vol vs QuantLib, Merton per expiry vs the series.
// The GPU (32-bit) half of the comparison is /check.html?only=chain.
import { readFileSync } from "node:fs";
import { chainJobs, runJobsCPU } from "./chain.js";

const data = JSON.parse(readFileSync(new URL("./chain_data.json", import.meta.url)));
const C = data.contracts;
const t0 = performance.now();
const J = chainJobs(data);
const t1 = performance.now();
const { bs, merton } = J.finish(runJobsCPU(J));
const t2 = performance.now();
let fails = 0;
const worst = (got, ref) => {
  let d = 0;
  let at = -1;
  for (let i = 0; i < ref.length; i++) {
    const g = Math.abs(got[i] - ref[i]);
    if (!(g <= d)) { d = g; at = i; }
  }
  return [d, at];
};
const show = (name, got, ref, tol) => {
  const [d, i] = worst(got, ref);
  const ok = d < tol;
  if (!ok) fails++;
  const ex = data.expiries[C.e[i]].expiry;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: max |js - ref| ${d.toExponential(2)} dollars over ${ref.length} contracts` +
    ` (worst ${ex} ${C.K[i]} ${C.call[i] ? "call" : "put"})`);
};
console.log(`${C.K.length} contracts, ${data.expiries.length} expiries: ${J.nJobs} jobs ` +
  `(${J.info.mertonNodes} Merton row nodes, ${(J.info.mertonTaps / 1e6).toFixed(1)}M + ` +
  `${(J.info.bsTaps / 1e6).toFixed(1)}M taps), built in ${(t1 - t0).toFixed(0)} ms, run in ${(t2 - t1).toFixed(0)} ms`);
show("Black-Scholes, own vol, vs QuantLib", bs, C.bs_ql, 1e-8);
show("Merton per expiry vs the Merton series", merton, C.merton, 1e-8);
const mid = C.bid.map((b, i) => 0.5 * (b + C.ask[i]));
show("Black-Scholes, own vol, vs the market mid", bs, mid, 1e-8);
process.exit(fails ? 1 : 0);
