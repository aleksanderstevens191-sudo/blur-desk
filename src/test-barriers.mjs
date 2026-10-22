// node src/test-barriers.mjs — checks the JS barrier core against the Python reference.
import { readFileSync } from "node:fs";
import { barrierBlurCPU, stepBarrierCPU, barrierClosedForm, dkoClosedForm } from "./barrier.js";
import { blurSpec, blurValue, stepSpec, stepValues } from "./barrier-cases.js";

const fx = JSON.parse(readFileSync(new URL("./barrier_fixtures.json", import.meta.url)));
const S0 = fx.S0;
const TOL = 1e-9; // JS vs Python, both 64-bit
const CF_TOL = 1e-11; // closed forms, JS vs Python

let fails = 0;
const line = (ok, s) => {
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s}`);
};

for (const c of fx.cases) {
  const t0 = performance.now();
  if (c.method === "step") {
    const spec = stepSpec(c);
    const v = stepValues(spec, stepBarrierCPU(spec).rows, S0);
    const ms = performance.now() - t0;
    const d = Math.max(...["out", "in", "vanilla"].map((k) => Math.abs(v[k] - c.py[k])));
    line(d < TOL, `step     ${c.kind.padEnd(22)} N=${c.steps} shift=${c.shift ? "yes" : "no "}` +
      `  out=${v.out.toFixed(8)} in=${v.in.toFixed(8)}  |js-py|=${d.toExponential(1)}  (${ms.toFixed(0)} ms)`);
    const [ud, cp] = c.kind.split(" ");
    const cf = barrierClosedForm(`${ud}-and-out ${cp}`, S0, c.K, c.H, c.sigma, c.T, c.r);
    if (Math.abs(cf - c.closedForm.out) >= CF_TOL) line(false, `closed form ${ud}-and-out ${cp}: js=${cf} py=${c.closedForm.out}`);
    continue;
  }
  const v = blurValue(c, barrierBlurCPU(blurSpec(c, S0)), S0);
  const ms = performance.now() - t0;
  const d = Math.abs(v - c.py);
  line(d < TOL, `${c.method.padEnd(8)} ${c.kind.padEnd(22)} K=${c.K} sigma=${c.sigma} T=${c.T}` +
    (c.images != null ? ` images=${c.images}` : "") +
    `  js=${v.toFixed(8)}  py=${c.py.toFixed(8)}  |js-py|=${d.toExponential(1)}  (${ms.toFixed(0)} ms)`);
  const cf = c.method === "double"
    ? dkoClosedForm(S0, c.K, c.L, c.U, c.sigma, c.T, c.r, 0, c.kind.split(" ")[1])
    : barrierClosedForm(c.kind, S0, c.K, c.H, c.sigma, c.T, c.r);
  if (Math.abs(cf - c.closedForm) >= CF_TOL) line(false, `closed form ${c.kind}: js=${cf} py=${c.closedForm}`);
}
console.log(fails ? `${fails} failed` : "all JS barrier results (and closed forms) match the Python reference");
process.exit(fails ? 1 : 0);
