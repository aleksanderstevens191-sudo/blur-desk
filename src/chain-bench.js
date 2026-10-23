// /chain-bench.html: the 4,738-contract SPY chain on the GPU, timed two ways, for the
// comparison with QuantLib (reference/bench_chain_quantlib.py prices the same contracts).
//
//   compute only   jobs already on the GPU: encode, submit, wait for the GPU to finish
//                  (device.queue.onSubmittedWorkDone). Wall clock, so it includes the
//                  submit latency.
//   end to end     from the chain data to dollar prices: build the kernels and rows
//                  (CPU, 64-bit), compile the shader (a fresh, never-seen copy every run,
//                  so the browser's pipeline cache can't help), upload, run, read back,
//                  finish (interpolation and parity, CPU).
// Each for Black-Scholes alone, Merton alone and both, REPS runs (default 20, ?reps=):
// median with min and max. The last line is JSON for the README table.
import data from "./chain_data.json";
import { chainJobs, chainJobGPU, CHAIN_WGSL } from "./chain.js";
import { context, gpuInfo } from "./gpu.js";

const out = document.getElementById("out");
const log = (line = "") => (out.textContent += line + "\n");
const REPS = +(new URLSearchParams(location.search).get("reps") ?? 20);
const C = data.contracts;
const n = C.K.length;
const stats = (ts) => {
  const s = [...ts].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { median: s.length % 2 ? s[(s.length - 1) / 2] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]), min: s[0], max: s.at(-1), p25: q(0.25), p75: q(0.75) };
};
const fmt = (t) => `${t.median.toFixed(2).padStart(9)} [${t.min.toFixed(2)}, ${t.max.toFixed(2)}] ms`;
const maxErr = (got, ref) => got.reduce((m, v, i) => Math.max(m, Math.abs(v - ref[i])), 0);

async function main() {
  const t0 = performance.now();
  const { device } = await context();
  const tInit = performance.now() - t0;
  log(`GPU: ${await gpuInfo()}   (device setup ${tInit.toFixed(0)} ms, once per page, not counted below)`);
  log(`${navigator.userAgent}`);
  log(`${n} SPY contracts, ${data.expiries.length} expiries, quotes ${data.quote_time}; ${REPS} runs each: median [min, max]\n`);
  const res = { gpu: await gpuInfo(), contracts: n, runs: REPS, ua: navigator.userAgent };
  for (const [name, models] of [["bs", ["bs"]], ["merton", ["merton"]], ["both", ["bs", "merton"]]]) {
    // end to end, with a fresh shader compile every run
    const e2e = [];
    const parts = { build: [], compile: [], run: [], finish: [] };
    let prices;
    let J;
    for (let k = 0; k < REPS; k++) {
      const a = performance.now();
      J = chainJobs(data, { models });
      const b = performance.now();
      const module = device.createShaderModule({ code: `${CHAIN_WGSL}\n// run ${name} ${k} ${Math.random()}\n` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const c = performance.now();
      const job = await chainJobGPU(J, { pipeline });
      const o = await job.read();
      const d = performance.now();
      prices = J.finish(o);
      const e = performance.now();
      job.destroy();
      e2e.push(e - a);
      parts.build.push(b - a); parts.compile.push(c - b); parts.run.push(d - c); parts.finish.push(e - d);
    }
    // compute only: upload once, then submit and wait
    const job = await chainJobGPU(J);
    await job.read(); // warm up
    const comp = [];
    for (let k = 0; k < REPS; k++) {
      const a = performance.now();
      const enc = device.createCommandEncoder();
      job.encode(enc);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      comp.push(performance.now() - a);
    }
    const warm = [];
    for (let k = 0; k < REPS; k++) {
      const a = performance.now();
      await job.read();
      warm.push(performance.now() - a);
    }
    job.destroy();
    const priced = models.length * n;
    const label = { bs: "Black-Scholes, own vols", merton: "Merton per expiry", both: "both models" }[name];
    const errs = [];
    if (models.includes("bs")) errs.push(`Black-Scholes vs QuantLib ${maxErr(prices.bs, C.bs_ql).toExponential(1)}`);
    if (models.includes("merton")) errs.push(`Merton vs series ${maxErr(prices.merton, C.merton).toExponential(1)}`);
    const S = { compute: stats(comp), computeRead: stats(warm), e2e: stats(e2e),
      parts: Object.fromEntries(Object.entries(parts).map(([k, v]) => [k, stats(v).median])) };
    log(`${label}: ${priced} prices, ${J.nJobs} jobs, ${((J.info.bsTaps + J.info.mertonTaps) / 1e6).toFixed(1)}M taps; ` +
      `max error (dollars): ${errs.join(", ")}`);
    log(`  compute only          ${fmt(S.compute)}  ->  ${Math.round(n / (S.compute.median / 1e3)).toLocaleString().padStart(12)} contracts/sec`);
    log(`  compute + read back   ${fmt(S.computeRead)}  ->  ${Math.round(n / (S.computeRead.median / 1e3)).toLocaleString().padStart(12)} contracts/sec`);
    log(`  end to end            ${fmt(S.e2e)}  ->  ${Math.round(n / (S.e2e.median / 1e3)).toLocaleString().padStart(12)} contracts/sec`);
    log(`    of which (medians): build ${S.parts.build.toFixed(1)} ms, shader compile ${S.parts.compile.toFixed(1)} ms, ` +
      `upload + run + read back ${S.parts.run.toFixed(1)} ms, finish ${S.parts.finish.toFixed(1)} ms\n`);
    res[name] = S;
  }
  log("JSON " + JSON.stringify(res));
  document.title = "DONE";
}

main().catch((e) => {
  log(`error: ${e.message}`);
  document.title = "ERROR";
});
