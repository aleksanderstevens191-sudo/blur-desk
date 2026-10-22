// Benchmark: every single-barrier type across a grid of barriers and strikes,
// priced in ONE GPU dispatch. Compare with reference/bench_quantlib.py (same options).
// /bench.html?scale=2 doubles barriers and strikes (4x the options).
import { MODE, gridZ, sampleVanilla, discountedKernel, mirrorExponent, barrierClosedForm, barrierBlurCPU, parseKind } from "./barrier.js";
import { barrierJob, gpuInfo } from "./gpu.js";

const out = document.getElementById("out");
const log = (line = "") => (out.textContent += line + "\n");
const params = new URLSearchParams(location.search);
const scale = +(params.get("scale") ?? 1);
const REPS = +(params.get("reps") ?? 20);

// The option set (reference/bench_quantlib.py builds the same one).
const MARKET = { S0: 100, sigma: 0.2, T: 1, r: 0.05, dx: 0.002 };
const TYPES = ["down-and-out call", "down-and-in call", "up-and-out call", "up-and-in call",
  "down-and-out put", "down-and-in put", "up-and-out put", "up-and-in put"];

function optionSet(nH, nK) {
  const { S0, dx } = MARKET;
  // barriers sit on grid nodes: H = S0 e^{-/+ m dx}, from about 1% to 30% (down) / 40% (up) away
  const steps = (mMax) => Array.from({ length: nH }, (_, i) => Math.round(5 + (i * (mMax - 5)) / (nH - 1)));
  const down = steps(Math.round(Math.log(S0 / 70) / dx));
  const up = steps(Math.round(Math.log(1.4) / dx));
  const strikes = Array.from({ length: nK }, (_, i) => 70 + (i * 65) / (nK - 1));
  const list = [];
  for (const kind of TYPES) {
    const isDown = kind.startsWith("down");
    for (const m of isDown ? down : up) {
      for (const K of strikes) list.push({ kind, K, m: isDown ? -m : m, H: S0 * Math.exp((isDown ? -m : m) * dx) });
    }
  }
  return { list, strikes };
}

async function main() {
  const { S0, sigma, T, r, dx } = MARKET;
  const nH = 32 * scale;
  const nK = 64 * scale;
  const { list, strikes } = optionSet(nH, nK);
  log(`GPU: ${await gpuInfo()}`);
  log(`market: S0=${S0} sigma=${sigma} T=${T} r=${r}; weighted mirror, one blur per option, dx=${dx} in ln S`);
  log(`options: 8 types x ${nH} barriers x ${nK} strikes = ${list.length.toLocaleString()}\n`);

  // grid anchored at S0 so spot and every barrier are nodes; strikes fall anywhere
  const hw = 10 * sigma * Math.sqrt(T) + 0.5;
  const nHalf = Math.ceil(hw / dx);
  const grid = { anchor: Math.log(S0), lo: -nHalf, n: 2 * nHalf + 1, dx };
  const z = gridZ(grid);
  const center = nHalf;
  const t0 = performance.now();
  const rows = new Map();
  const P = new Float64Array(2 * nK * grid.n);
  strikes.forEach((K, i) => {
    for (const [c, cp] of [[0, "call"], [1, "put"]]) {
      P.set(sampleVanilla(z, K, cp), (2 * i + c) * grid.n);
      rows.set(`${K} ${cp}`, 2 * i + c);
    }
  });
  const K = discountedKernel(dx, sigma, T, r);
  const half = (K.length - 1) / 2;
  const a = mirrorExponent(sigma, r);
  const opts = list.map((o, idx) => {
    const { ud, io, cp } = parseKind(o.kind);
    return {
      pOff: rows.get(`${o.K} ${cp}`) * grid.n, nP: grid.n, dx, kOff: 0, half,
      outOff: idx, nOut: 1, out0: center, lo: center + o.m, hi: -1, images: 1, a,
      mode: (ud === "down" ? MODE.down : MODE.up) + (io === "in" ? MODE.IN : 0),
    };
  });
  const prepMs = performance.now() - t0;
  log(`setup on the CPU (payoff rows, kernel, option table): ${prepMs.toFixed(1)} ms, once`);
  log(`kernel: ${K.length} taps, so ${(list.length * K.length / 1e6).toFixed(1)}M tap evaluations per dispatch\n`);

  const job = await barrierJob({ P, K, opts });
  const prices = await job.read(); // warm-up, and the answers
  const { device } = job;
  const reps = REPS;
  const gpuMs = [];
  for (let i = 0; i < reps; i++) {
    const s = performance.now();
    const enc = device.createCommandEncoder();
    job.encode(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    gpuMs.push(performance.now() - s);
  }
  const rbMs = [];
  for (let i = 0; i < Math.min(5, reps); i++) {
    const s = performance.now();
    await job.read();
    rbMs.push(performance.now() - s);
  }
  job.destroy();
  const med = (xs) => xs.slice().sort((p, q) => p - q)[Math.floor(xs.length / 2)];
  const g = med(gpuMs);
  const rb = med(rbMs);
  log(`one dispatch, submit to done (median of ${reps}): ${g.toFixed(2)} ms  ->  ${Math.round(list.length / (g / 1000)).toLocaleString()} options/sec`);
  log(`including reading the prices back (median of ${rbMs.length}): ${rb.toFixed(2)} ms  ->  ${Math.round(list.length / (rb / 1000)).toLocaleString()} options/sec`);

  // accuracy: every price against the closed form at the same (grid-node) barrier
  let worst = 0;
  let worstAt = null;
  list.forEach((o, i) => {
    const cf = barrierClosedForm(o.kind, S0, o.K, o.H, sigma, T, r);
    const e = Math.abs(prices[i] - cf);
    if (e > worst) [worst, worstAt] = [e, { ...o, gpu: prices[i], cf }];
  });
  const ok = worst < 1e-4;
  log(`\nmax |gpu - closed form| over all ${list.length.toLocaleString()} options: ${worst.toExponential(2)}  ${ok ? "PASS" : "FAIL"} (< 1e-4)`);
  log(`  worst: ${worstAt.kind} K=${worstAt.K.toFixed(2)} H=${worstAt.H.toFixed(3)} gpu=${worstAt.gpu.toFixed(6)} closed form=${worstAt.cf.toFixed(6)}`);

  // the same blurs on the CPU in JavaScript (64-bit), on a sample, for scale
  const sample = opts.filter((_, i) => i % 16 === 0).map((o, i) => ({ ...o, outOff: i }));
  const s = performance.now();
  barrierBlurCPU({ P, K, opts: sample });
  const cpuRate = sample.length / ((performance.now() - s) / 1000);
  log(`\nsame blur on the CPU in plain JavaScript (1 thread, 64-bit), ${sample.length} options: ${Math.round(cpuRate).toLocaleString()} options/sec`);
  log(`QuantLib on the CPU: python3 reference/bench_quantlib.py (same option set; it prints these four too):`);
  for (const i of [0, Math.floor(list.length / 3), Math.floor((2 * list.length) / 3), list.length - 1]) {
    const o = list[i];
    log(`  #${i}: ${o.kind} K=${o.K.toFixed(4)} H=${o.H.toFixed(4)}  ${prices[i].toFixed(8)}`);
  }
  window.benchResult = { options: list.length, gpuMs: g, readbackMs: rb, worst, cpuRate };
  document.title = ok ? "PASS" : "FAIL";
}

main().catch((e) => {
  log(`error: ${e.message}`);
  document.title = "ERROR";
});
