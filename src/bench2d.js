// /bench2d.html: the two-stock 2D blur on the GPU, direct vs FFT, across patch
// sizes (O x O spots priced at once) and kernel sizes (m x m taps). For each O
// the m sweep runs until the FFT is clearly ahead; the crossover m* is where
// the two times cross (log-log interpolation). Then prices/sec at the sizes the
// fixtures use, compute-only and end to end.
//   ?quick=1   small sweep (for a software adapter)
// A price is one exchange option (Margrabe's payoff) at one (S1, S2) spot pair.
import { gpuInfo } from "./gpu.js";
import { job2d, exchangePayoff, bestOfCallPayoff } from "./blur2d.js";
import { direct2dJob, fft2dJob } from "./gpu2d.js";

const out = document.getElementById("out");
const log = (line = "") => (out.textContent += line + "\n");
const q = new URLSearchParams(location.search);
const QUICK = q.has("quick");
const SIZES = QUICK ? [31, 63] : [63, 127, 255, 511]; // odd: the spot sits in the middle
const MS = QUICK ? [9, 17, 33, 65] : [9, 17, 33, 65, 129, 257, 513, 1025];
const TARGET_MS = QUICK ? 1 : 40; // GPU time per timed submit
const AHEAD = QUICK ? 3 : 8; // stop the m sweep once the FFT is this many times faster
const TOL = 1e-4;
const MARKET = { S1: 100, S2: 100, s1: 0.3, s2: 0.3, rho: 0.5, T: 1, r: 0.05 };

const median = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];

// ms per run: `reps` runs in one submit, median of 5 submits (GPU work only)
async function time(job) {
  const { device } = job;
  const submit = async (reps) => {
    const s = performance.now();
    const enc = device.createCommandEncoder();
    for (let i = 0; i < reps; i++) job.encode(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - s;
  };
  await submit(1);
  const one = await submit(1);
  const reps = Math.max(1, Math.min(64, Math.ceil(TARGET_MS / one)));
  const trials = [];
  for (let t = 0; t < 5; t++) trials.push((await submit(reps)) / reps);
  return median(trials);
}

// An O x O patch spanning +-0.5 in ln S1 and ln S2 (grid step 1/(O-1)), and the
// maturity T that makes the kernel m x m taps (+-6 sd, as in the fixtures):
// small m is a short-dated option, large m a long-dated one.
function sizedJob(O, m, payoff = exchangePayoff()) {
  const { s1, r } = MARKET;
  const d = 1 / (O - 1);
  const half = (m - 1) / 2;
  // kernel half = ceil((6 sd + |drift|) / d); aim a hair inside so ceil lands on `half`
  const reach = (T) => 6 * s1 * Math.sqrt(T) + Math.abs((r - s1 * s1 / 2) * T);
  let [lo, hi] = [0, 100];
  for (let it = 0; it < 200; it++) {
    const T = (lo + hi) / 2;
    if (reach(T) < (half - 0.01) * d) lo = T;
    else hi = T;
  }
  return job2d({ ...MARKET, T: lo, payoff, width: 6, steps: [d, d], qx: (O - 1) / 2, qy: (O - 1) / 2 });
}

// worst |fft - direct| over the patch, scaled like the 1D bench: 1e-4 absolute
// on prices up to 100, 1e-6 relative above (32-bit floats can't do better)
function scaledDiff(a, b, disc) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, (disc * Math.abs(a[i] - b[i])) / Math.max(1, (disc * Math.abs(b[i])) / 100));
  return m;
}

async function measure(job) {
  const direct = await direct2dJob(job);
  const fft = await fft2dJob(job);
  const near = scaledDiff(await fft.read(), await direct.read(), job.disc);
  const res = { O: job.ox, m: job.mx, T: job.T, N: fft.N, parts: fft.parts.length, directMs: await time(direct), fftMs: await time(fft), near };
  direct.destroy();
  fft.destroy();
  return res;
}

const fmt = (ms) => (ms < 1 ? ms.toFixed(3) : ms < 100 ? ms.toFixed(2) : ms.toFixed(0)).padStart(8);
const rate = (x) => (x >= 1e9 ? `${(x / 1e9).toFixed(2)}G` : x >= 1e6 ? `${(x / 1e6).toFixed(1)}M` : `${(x / 1e3).toFixed(1)}k`);

async function main() {
  const gpu = await gpuInfo();
  log(`GPU: ${gpu}`);
  log(`exchange option, S1 = S2 = 100, vols 30%/30%, rho = 0.5, r = 5%; kernel +-6 sd`);
  log(`a run prices an O x O patch of spot pairs spanning +-0.5 in ln S1 and ln S2 (S/1.65 to 1.65 S);`);
  log(`the kernel size m is set by the maturity (m = 1025 at O = 511 is about 4 months);`);
  log(`times are ms per run, GPU work only`);
  log(`(payoff and kernel uploaded once; the FFT transforms its kernels every run).`);
  let fails = 0;
  const sweep = [];
  const crossover = [];
  for (const O of SIZES) {
    log(`\nO = ${O} (${(O * O).toLocaleString()} prices)   m x m      N   parts   direct ms     fft ms   direct/fft   |fft-direct|`);
    let prev = null;
    let cross = null;
    for (const m of MS) {
      const r = await measure(sizedJob(O, m));
      sweep.push(r);
      const ok = r.near < TOL;
      if (!ok) fails++;
      log(`${" ".repeat(25)}${String(m).padStart(5)}  ${String(r.N[0]).padStart(5)}   ${r.parts}   ${fmt(r.directMs)}   ${fmt(r.fftMs)}   ` +
        `${(r.directMs / r.fftMs).toFixed(2).padStart(9)}   ${r.near.toExponential(1)}${ok ? "" : " FAIL"}`);
      if (prev && cross === null && prev.directMs < prev.fftMs && r.directMs >= r.fftMs) {
        const g0 = Math.log(prev.directMs / prev.fftMs);
        const g1 = Math.log(r.directMs / r.fftMs);
        cross = Math.exp(Math.log(prev.m) + (Math.log(r.m) - Math.log(prev.m)) * (g0 / (g0 - g1)));
      }
      if (prev === null && r.directMs >= r.fftMs) cross = m;
      prev = r;
      if (r.directMs > AHEAD * r.fftMs) break;
    }
    crossover.push({ O, mStar: cross });
    log(cross === null ? "  FFT never caught up in this sweep" : `  crossover: FFT faster from m ~ ${Math.round(cross)} taps per side`);
  }
  log("\ncrossover m* (kernel taps per side) by patch size:");
  for (const c of crossover) log(`  O = ${String(c.O).padStart(4)}   m* = ${c.mStar === null ? "none" : Math.round(c.mStar)}`);

  // the fixture grid (c = 0.06 of the conditional sd): 233 x 233 taps at these vols
  log("\nprices/sec on the fixture grid (c = 0.06, kernel 233 x 233), whole patch at once:");
  const realistic = [];
  for (const [what, payoff] of [["exchange", exchangePayoff()], ["best-of call K=100", bestOfCallPayoff(100)]]) {
    for (const O of QUICK ? [33] : [65, 257]) {
      const job = job2d({ ...MARKET, T: 1, payoff, c: 0.06, width: 6, qx: (O - 1) / 2, qy: (O - 1) / 2 });
      const r = await measure(job);
      if (r.near >= TOL) fails++;
      // end to end: build the job (payoff image + kernel), plan, upload, run, read back
      const e2e = [];
      for (let t = 0; t < (QUICK ? 1 : 5); t++) {
        const s = performance.now();
        const j = job2d({ ...MARKET, T: 1, payoff, c: 0.06, width: 6, qx: (O - 1) / 2, qy: (O - 1) / 2 });
        const f = await fft2dJob(j);
        await f.read();
        f.destroy();
        e2e.push(performance.now() - s);
      }
      const n = O * O;
      realistic.push({ what, ...r, e2eMs: median(e2e) });
      log(`  ${what.padEnd(18)} O=${String(O).padStart(3)} (${n.toLocaleString().padStart(6)} prices) direct ${fmt(r.directMs)} ms ${rate((n * 1000) / r.directMs).padStart(7)}/s` +
        `   fft ${fmt(r.fftMs)} ms ${rate((n * 1000) / r.fftMs).padStart(7)}/s   fft end to end ${fmt(median(e2e))} ms` +
        `   |fft-direct| ${r.near.toExponential(1)}${r.near < TOL ? "" : " FAIL"}`);
    }
  }

  const result = { gpu, date: new Date().toISOString().slice(0, 10), quick: QUICK, sweep, crossover, realistic };
  window.benchResult = result;
  const json = JSON.stringify(result);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  a.download = "bench2d.json";
  a.textContent = "download bench2d.json";
  document.body.append(a);
  log(`\n${fails ? `${fails} accuracy checks failed` : "all FFT results within 1e-4 of direct"}`);
  log(`JSON:\n${json}`);
  document.title = fails ? "FAIL" : "PASS";
}

main().catch((e) => {
  log(`error: ${e.message}`);
  document.title = "ERROR";
});
