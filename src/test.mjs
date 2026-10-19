// node src/test.mjs — checks the JS core against the Python reference.
import { readFileSync } from "node:fs";
import { logGrid, bsBlurPrices, americanBlur } from "./blur.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url)));
const { S0, r, dx } = fx;
let fails = 0;

for (const c of fx.cases) {
  const { x, center } = logGrid(S0, c.halfWidth, dx);
  let v;
  const t0 = performance.now();
  if (c.type === "euro_call") {
    v = bsBlurPrices((s) => Math.max(s - c.K, 0), c.sigma, c.T, r, x, dx)[center];
  } else {
    v = americanBlur((s) => Math.max(c.K - s, 0), c.sigma, c.T, r, c.steps, x, dx)[center];
  }
  const ms = performance.now() - t0;
  const diff = Math.abs(v - c.blur);
  const ok = diff < 1e-9;
  if (!ok) fails++;
  console.log(
    `${ok ? "PASS" : "FAIL"} ${c.type.padEnd(12)} K=${c.K} T=${c.T} sigma=${c.sigma}` +
      `  js=${v.toFixed(8)}  py=${c.blur.toFixed(8)}  |js-py|=${diff.toExponential(1)}` +
      (c.closedForm !== undefined ? `  |js-BS|=${Math.abs(v - c.closedForm).toExponential(1)}` : "") +
      `  (${ms.toFixed(1)} ms)`
  );
}
console.log(fails ? `${fails} failed` : "all JS results match the Python reference");
process.exit(fails ? 1 : 0);
