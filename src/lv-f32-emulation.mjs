// node src/lv-f32-emulation.mjs [value|change] — the WGSL local-vol step (gpu-lv.js) emulated
// in JavaScript with every operation rounded to 32 bits (Math.fround) and an exact exp, on
// the 1-year flat 20% at-the-money call and put of src/lv_fixtures.json. "value" divides
// the weighted sum of the values by the sum of the weights, as the shader first did: the
// rounding of that sum is the same factor in every step, and the prices come out 2.5e-4
// to 2.9e-4 low per $100 (the Mac measured 1.44e-4). "change" (the shader now) averages
// the neighbours' differences from each node instead: about 8e-6.
import { readFileSync } from "node:fs";
import { lvJob } from "./lv.js";

const f = Math.fround;
const MODE = process.argv[2] || "change";
const data = JSON.parse(readFileSync(new URL("./lv_data.json", import.meta.url)));
const fx = JSON.parse(readFileSync(new URL("./lv_fixtures.json", import.meta.url)));
const cs = fx.cases.filter((c) => c.model === "flat" && c.i === 26 && c.k === 0);
const ex = data.expiries[26];
const job = lvJob(data, { T: ex.T, D: ex.D, strikes: cs.map((c) => ({ k: c.k, call: c.call })), sigma: cs[0].sigma });
const n = job.x.length, dx = f(job.dx), x0 = f(job.x[0]);
const ex32 = (x) => f(Math.exp(x));
const e0 = ex32(x0), e1 = ex32(f(x0 + dx)), e2 = ex32(f(x0 + f(f(n - 2) * dx))), e3 = ex32(f(x0 + f(f(n - 1) * dx)));
const val = (V, j) => {
  if (j >= 0 && j < n) return V[j];
  const e = ex32(f(x0 + f(f(j) * dx)));
  if (j < 0) {
    const A = f(f(V[1] - V[0]) / f(e1 - e0));
    return f(f(V[0] - f(A * e0)) + f(A * e));
  }
  const A = f(f(V[n - 1] - V[n - 2]) / f(e3 - e2));
  return f(f(V[n - 2] - f(A * e2)) + f(A * e));
};
let Vs = job.V0.map((V) => Float32Array.from(V));
for (let st = job.steps.length - 2; st >= 0; st--) {
  const h = f(job.steps[st].b - job.steps[st].a);
  const s = f(f(cs[0].sigma) * f(Math.sqrt(h)));
  const inv = f(1 / f(f(2 * s) * s));
  const need = Math.ceil(f(f(f(8 * s) + f(f(0.5 * s) * s)) / dx)) + 1;
  let H = 2;
  while (H < need) H *= 2;
  const w = Float32Array.from({ length: 2 * H + 1 }, (_, m) => {
    const y = f(f(m - H) * dx);
    return f(ex32(f(f(-y * y) * inv)) * ex32(f(-0.5 * y)));
  });
  Vs = Vs.map((V) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const vi = MODE === "change" ? V[i] : 0;
      let acc = 0, den = 0;
      for (let j0 = -H; j0 <= H; j0 += 32) {
        let blk = 0, bden = 0;
        for (let j = j0; j < Math.min(j0 + 32, H + 1); j++) {
          bden = f(bden + w[j + H]);
          blk = f(blk + f(w[j + H] * f(val(V, i + j) - vi)));
        }
        den = f(den + bden);
        acc = f(acc + blk);
      }
      out[i] = f(vi + f(acc / den));
    }
    return out;
  });
}
cs.forEach((c, j) => {
  const v = job.scale * Vs[j][job.center];
  console.log(`${MODE}: ${c.call ? "call" : "put "} 32-bit ${v.toFixed(7)}  python ${c.py.toFixed(7)}  diff ${(v - c.py).toExponential(2)}`);
});
