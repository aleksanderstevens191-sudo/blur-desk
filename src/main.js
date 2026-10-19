// Runs the fixtures on the GPU and compares with the Python reference values.
import fx from "./fixtures.json";
import { logGrid, gaussianKernel } from "./blur.js";
import { blurGPU, americanGPU, gpuInfo } from "./gpu.js";

const TOL = 1e-4;
const out = document.getElementById("out");
const log = (line = "") => (out.textContent += line + "\n");

async function price(c) {
  const { S0, r, dx } = fx;
  const { x, center } = logGrid(S0, c.halfWidth, dx);
  if (c.type === "euro_call") {
    const g = Float64Array.from(x, (xi) => Math.max(Math.exp(xi) - c.K, 0));
    const k = gaussianKernel(dx, c.sigma * Math.sqrt(c.T), (r - 0.5 * c.sigma * c.sigma) * c.T);
    const V = await blurGPU(g, k);
    return V[center] * Math.exp(-r * c.T);
  }
  const P = Float64Array.from(x, (xi) => Math.max(c.K - Math.exp(xi), 0));
  const dt = c.T / c.steps;
  const k = gaussianKernel(dx, c.sigma * Math.sqrt(dt), (r - 0.5 * c.sigma * c.sigma) * dt);
  const V = await americanGPU(P, k, c.steps, Math.exp(-r * dt));
  return V[center];
}

async function main() {
  log(`GPU: ${await gpuInfo()}`);
  log(`tolerance: |gpu - py| < ${TOL}\n`);
  let fails = 0;
  for (const c of fx.cases) {
    const t0 = performance.now();
    const v = await price(c);
    const ms = performance.now() - t0;
    const diff = Math.abs(v - c.blur);
    const ok = diff < TOL;
    if (!ok) fails++;
    log(
      `${ok ? "PASS" : "FAIL"} ${c.type.padEnd(12)} K=${c.K} T=${c.T} sigma=${c.sigma}` +
        (c.steps ? ` steps=${c.steps}` : "") +
        `  gpu=${v.toFixed(8)}  py=${c.blur.toFixed(8)}  |gpu-py|=${diff.toExponential(2)}  (${ms.toFixed(1)} ms)`
    );
  }
  log(fails ? `\n${fails} failed` : "\nall GPU results within 1e-4 of the Python reference");
  document.title = fails ? "FAIL" : "PASS";
}

main().catch((e) => {
  log(`error: ${e.message}`);
  document.title = "ERROR";
});
