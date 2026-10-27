// node src/test-lv.mjs — local vol (Part 4): the 64-bit JS twin of the variable-width
// blur (lv.js) against the Python fixtures (reference/make_lv_data.py), on the same
// table, grid and time steps; flat vol against Black's formula; and the job's table
// reads against Python's (the round trip stored in lv_data.json).
import { readFileSync } from "node:fs";
import { lvJob, priceLvCPU, impliedVol } from "./lv.js";

const data = JSON.parse(readFileSync(new URL("./lv_data.json", import.meta.url)));
const fx = JSON.parse(readFileSync(new URL("./lv_fixtures.json", import.meta.url)));
let fails = 0;
const line = (ok, s) => {
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s}`);
};
const e = (x) => x.toExponential(1);
const { dt, m, dx, x_lo, x_hi } = data.settings;
console.log(`JS 64-bit local-vol blur vs Python, same table and grid (dx ${dx} on [${x_lo}, ${x_hi}], ` +
  `steps <= ${dt * 365} day, >= ${m} per interval); prices per $100 of forward, discounted`);

// group the cases by (model, sigma, expiry): one job each
const groups = new Map();
for (const c of fx.cases) {
  const key = `${c.model}|${c.sigma ?? ""}|${c.i}`;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(c);
}
let worst = 0;
for (const cs of groups.values()) {
  const c0 = cs[0];
  const ex = data.expiries[c0.i];
  const t0 = performance.now();
  const job = lvJob(data, { T: ex.T, D: ex.D, strikes: cs.map((c) => ({ k: c.k, call: c.call })), sigma: c0.model === "flat" ? c0.sigma : null });
  const p = priceLvCPU(job);
  const ms = performance.now() - t0;
  cs.forEach((c, j) => {
    const d = Math.abs(p[j] - c.py);
    worst = Math.max(worst, d);
    let extra = "";
    if (c.model === "flat") extra = `  |js - Black| ${e(Math.abs(p[j] - c.bs))}`;
    else extra = `  vol ${(100 * impliedVol(p[j] / job.scale, c.k, ex.T, c.call)).toFixed(3)} (surface ${(100 * c.iv_surface).toFixed(3)})`;
    line(d < 1e-9 && (c.model !== "flat" || Math.abs(p[j] - c.bs) < 1e-6),
      `${c.model.padEnd(4)} ${ex.name} k ${c.k.toFixed(4).padStart(7)} ${c.call ? "call" : "put "} js ${p[j].toFixed(8)} py ${c.py.toFixed(8)} ` +
      `|js - py| ${e(d)}${extra}  (${job.steps.length} steps, ${(ms / cs.length).toFixed(0)} ms each)`);
  });
}
console.log(`worst |js - py| ${e(worst)}`);
console.log(fails ? `\n${fails} failed` : "\nall PASS");
process.exit(fails ? 1 : 0);
