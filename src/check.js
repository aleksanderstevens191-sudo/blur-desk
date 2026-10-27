// Runs the fixtures on the GPU and compares with the Python reference values.
import fx from "./fixtures.json";
import bfx from "./barrier_fixtures.json";
import { logGrid, gaussianKernel } from "./blur.js";
import { blurGPU, americanGPU, gpuInfo, barrierGPU, stepBarrierGPU } from "./gpu.js";
import { blurSpec, blurValue, stepSpec, stepValues } from "./barrier-cases.js";
import { fftChecks } from "./fft-check.js";
import { kernelChecks } from "./kernel-check.js";
import { chainChecks } from "./chain-check.js";
import { oddsChecks } from "./odds-check.js";
import { twoStockChecks } from "./check-2d.js";
import { lvChecks } from "./check-lv.js";
import SPY from "./spy_fit.json";

const TOL = 1e-4;
const out = document.getElementById("out");
const log = (line = "") => (out.textContent += line + "\n");
// /check.html?only=fft runs just the FFT section, ?only=kernels the fat-tailed kernels,
// ?only=chain the full SPY chain, ?only=odds the market's odds, ?only=2d the two-stock blur,
// ?only=lv the local-vol blur
const only = new URLSearchParams(location.search).get("only");
const run = (section) => !only || only === section;

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
  for (const c of run("fixtures") ? fx.cases : []) {
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
  if (run("fixtures")) log("\nbarriers (reference/make_barrier_fixtures.py):");
  for (const c of run("fixtures") ? bfx.cases : []) {
    const t0 = performance.now();
    let diffs;
    let shown;
    if (c.method === "step") {
      const spec = stepSpec(c);
      const v = stepValues(spec, (await stepBarrierGPU(spec)).rows, bfx.S0);
      diffs = ["out", "in"].map((k) => Math.abs(v[k] - c.py[k]));
      shown = `out=${v.out.toFixed(6)} in=${v.in.toFixed(6)}`;
    } else {
      const v = blurValue(c, await barrierGPU(blurSpec(c, bfx.S0)), bfx.S0);
      diffs = [Math.abs(v - c.py)];
      shown = `gpu=${v.toFixed(8)} py=${c.py.toFixed(8)}`;
    }
    const ms = performance.now() - t0;
    const diff = Math.max(...diffs);
    const ok = diff < TOL;
    if (!ok) fails++;
    const what = c.method === "step" ? `${c.kind} N=${c.steps}${c.shift ? " shifted" : ""}` : c.kind;
    log(`${ok ? "PASS" : "FAIL"} ${c.method.padEnd(8)} ${what.padEnd(24)}` +
      (c.images != null ? ` images=${c.images}` : "") +
      ` ${shown}  |gpu-py|=${diff.toExponential(2)}  (${ms.toFixed(1)} ms)`);
  }
  if (run("fft")) {
    log("\nFFT convolution vs direct convolution, both on the GPU (src/fft-check.js):");
    fails += await fftChecks(log, TOL);
  }
  if (run("kernels")) {
    log("\nfat-tailed kernels (Gaussian, Merton, variance gamma) vs Python (reference/make_kernel_fixtures.py):");
    fails += await kernelChecks(log, TOL);
  }
  if (run("chain")) {
    log("\nthe whole SPY chain in one GPU submit vs QuantLib and the Merton series (src/chain-check.js):");
    fails += await chainChecks(log, TOL);
  }
  if (run("odds")) {
    log("\nthe market's odds: the SPY fit's distribution, edge-detected from GPU prices (src/odds-check.js):");
    fails += await oddsChecks(log, SPY);
  }
  if (run("2d")) {
    log("\ntwo-stock options, 2D blur on the GPU (direct and FFT) vs Python (reference/make_2d_fixtures.py):");
    fails += await twoStockChecks(log, TOL);
  }
  if (run("lv")) {
    log("\nlocal vol: the variable-width blur on the GPU vs Python (reference/make_lv_data.py):");
    fails += await lvChecks(log, TOL);
  }
  log(fails ? `\n${fails} failed` : `\nall GPU results within 1e-4 of the Python reference${run("fft") ? ", FFT within 1e-4 of direct" : ""}`);
  document.title = fails ? "FAIL" : "PASS";
}

main().catch((e) => {
  log(`error: ${e.message}`);
  document.title = "ERROR";
});
