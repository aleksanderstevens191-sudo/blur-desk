// /check.html, local-vol section (?only=lv): the variable-width blur on the GPU
// (gpu-lv.js) against the Python fixtures (reference/make_lv_data.py), on the same
// table, grid and time steps. Prices per $100 of forward, discounted.
import fx from "./lv_fixtures.json";
import { lvJob, impliedVol } from "./lv.js";
import { lvGPU } from "./gpu-lv.js";

const e = (x) => x.toExponential(2);

export async function lvChecks(log, TOL) {
  const data = (await import("./lv_data.json")).default;
  let fails = 0;
  const { dt, m, dx, x_lo, x_hi } = data.settings;
  log(`grid dx ${dx} on [${x_lo}, ${x_hi}], steps <= ${dt * 365} day and >= ${m} per expiry interval; per $100 of forward`);
  const groups = new Map();
  for (const c of fx.cases) {
    const key = `${c.model}|${c.sigma ?? ""}|${c.i}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(c);
  }
  for (const cs of groups.values()) {
    const c0 = cs[0];
    const ex = data.expiries[c0.i];
    const job = lvJob(data, { T: ex.T, D: ex.D, strikes: cs.map((c) => ({ k: c.k, call: c.call })), sigma: c0.model === "flat" ? c0.sigma : null });
    const { prices, ms } = await lvGPU(job);
    cs.forEach((c, j) => {
      const d = Math.abs(prices[j] - c.py);
      const ok = d < TOL;
      if (!ok) fails++;
      const extra = c.model === "flat" ? `  |gpu - Black| ${e(Math.abs(prices[j] - c.bs))}`
        : `  vol ${(100 * impliedVol(prices[j] / job.scale, c.k, ex.T, c.call)).toFixed(3)} (surface ${(100 * c.iv_surface).toFixed(3)})`;
      log(`${ok ? "PASS" : "FAIL"} ${c.model.padEnd(4)} ${ex.name} k ${c.k.toFixed(4).padStart(7)} ${c.call ? "call" : "put "}` +
        ` gpu=${prices[j].toFixed(6)} py=${c.py.toFixed(6)} |gpu-py|=${e(d)}${extra}` +
        (j === 0 ? `  (${job.steps.length} steps, ${cs.length} strikes, ${ms.toFixed(0)} ms)` : ""));
    });
  }
  return fails;
}
