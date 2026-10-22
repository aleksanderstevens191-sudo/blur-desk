// /fft-bench.html: FFT blur vs direct blur on the GPU, across grid sizes n and
// kernel widths m. For each n the m sweep runs until the FFT is clearly ahead;
// the crossover m*(n) is where the two times cross (log-log interpolation).
// Then three batches at the sizes the Stage 1 kernels really use.
//   ?quick=1        small sweep (for a software adapter)
//   ?nmin=8&nmax=16 log2 of the smallest and largest grid
// The JSON at the end feeds reference/plot_fft_crossover.py.
import { gpuInfo } from "./gpu.js";
import { fftBlurJob, directBlurJob } from "./gpu-fft.js";
import { fftSize } from "./fft.js";
import { payoffRows, fatKernel } from "./fft-cases.js";

const out = document.getElementById("out");
const log = (line = "") => (out.textContent += line + "\n");
const q = new URLSearchParams(location.search);
const QUICK = q.has("quick");
const NMIN = +(q.get("nmin") ?? 8);
const NMAX = +(q.get("nmax") ?? (QUICK ? 11 : 16));
const SAMPLES = QUICK ? 2 ** 15 : 2 ** 20; // rows x n per batch
const TARGET_MS = QUICK ? 1 : 40; // GPU time per timed submit
const AHEAD = QUICK ? 3 : 8; // stop the m sweep once the FFT is this many times faster
const TOL = 1e-4;

const median = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];

// ms per batch: `reps` batches in one submit, median of 3 submits
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
  const reps = Math.max(1, Math.min(256, Math.ceil(TARGET_MS / one)));
  const trials = [];
  for (let t = 0; t < 3; t++) trials.push((await submit(reps)) / reps);
  return median(trials);
}

// payoff rows spanning +/-3 in ln S whatever n is, the kernel in grid steps
async function measure(n, m, rows) {
  const dx = 6 / n;
  const { F, center } = payoffRows({ n, dx, rows });
  const kernel = fatKernel(m);
  const direct = await directBlurJob({ F, rows, n, kernel });
  const fft = await fftBlurJob({ F, rows, n, kernel, dx });
  const [a, b] = [await direct.read(), await fft.read()];
  const w = Math.min(center, Math.round(Math.LN2 / dx));
  let near = 0;
  for (let r = 0; r < rows; r++) {
    for (let i = center - w; i <= center + w; i++) near = Math.max(near, Math.abs(a[r * n + i] - b[r * n + i]));
  }
  const res = { n, m, N: fft.N, rows, directMs: await time(direct), fftMs: await time(fft), near };
  direct.destroy();
  fft.destroy();
  return res;
}

const fmt = (ms) => (ms < 1 ? ms.toFixed(3) : ms < 100 ? ms.toFixed(2) : ms.toFixed(0)).padStart(8);
const rate = (x) => (x >= 1e9 ? `${(x / 1e9).toFixed(2)}G` : x >= 1e6 ? `${(x / 1e6).toFixed(1)}M` : `${(x / 1e3).toFixed(1)}k`);

async function main() {
  const gpu = await gpuInfo();
  log(`GPU: ${gpu}`);
  log(`each batch: rows x n = ${SAMPLES.toLocaleString()} payoff values (calls and puts), one kernel of m taps;`);
  log(`FFT length N = nextPow2(n + m - 1) (padded, no wraparound); times are ms per batch, GPU work only`);
  log(`(payoffs and kernel uploaded once; the FFT includes transforming the kernel every batch).`);
  let fails = 0;
  const sweep = [];
  const crossover = [];
  for (let p = NMIN; p <= NMAX; p++) {
    const n = 2 ** p;
    const rows = Math.max(2, SAMPLES / n);
    log(`\nn = ${n} (${rows} rows)        m   direct ms      fft ms   direct/fft   |fft-direct| near the money`);
    let prev = null;
    let cross = null;
    for (let h = 1; h < n; h = 2 * h + 1) {
      const m = 2 * h + 1;
      const r = await measure(n, m, rows);
      sweep.push(r);
      const ok = r.near < TOL;
      if (!ok) fails++;
      log(`${" ".repeat(18)}${String(m).padStart(7)} ${fmt(r.directMs)}    ${fmt(r.fftMs)}   ${(r.directMs / r.fftMs).toFixed(2).padStart(9)}   ${r.near.toExponential(1)}${ok ? "" : " FAIL"}`);
      if (prev && cross === null && prev.directMs < prev.fftMs && r.directMs >= r.fftMs) {
        // where log(direct/fft) crosses 0, linear in log m
        const g0 = Math.log(prev.directMs / prev.fftMs);
        const g1 = Math.log(r.directMs / r.fftMs);
        cross = Math.exp(Math.log(prev.m) + (Math.log(r.m) - Math.log(prev.m)) * (g0 / (g0 - g1)));
      }
      if (prev === null && r.directMs >= r.fftMs) cross = m; // FFT ahead from the start
      prev = r;
      if (r.directMs > AHEAD * r.fftMs) break;
    }
    crossover.push({ n, mStar: cross });
    log(cross === null ? `  FFT never caught up for m < 2n` : `  crossover: FFT faster from m ~ ${Math.round(cross)} taps`);
  }
  log("\ncrossover m* by grid size:");
  for (const c of crossover) log(`  n = ${String(c.n).padStart(6)}   m* = ${c.mStar === null ? "none" : Math.round(c.mStar)}`);

  // The kernels of reference/kernels.py on their own grids (n = m): Gaussian
  // T=1, variance gamma T=1, Merton fitted to SPY at the Dec 18 expiry.
  const real = [
    { what: "Gaussian, T=1", m: 5341 },
    { what: "variance gamma, T=1", m: 7087 },
    { what: "Merton (SPY fit), Dec 18", m: 38289 },
  ];
  log("\noptions/sec at the sizes the fat-tailed kernels use (grid = kernel width, n = m):");
  log("one row = one strike priced at every spot on the grid; a price = one strike at one spot");
  const realistic = [];
  for (const c of QUICK ? real.slice(0, 1) : real) {
    const rows = QUICK ? 2 : 2 * Math.max(1, Math.round(SAMPLES / c.m / 2));
    const r = await measure(c.m, c.m, rows);
    const ok = r.near < TOL;
    if (!ok) fails++;
    realistic.push({ ...c, ...r });
    for (const [name, ms] of [["direct", r.directMs], ["FFT", r.fftMs]]) {
      log(`  ${c.what.padEnd(26)} ${name.padEnd(6)} ${fmt(ms)} ms per ${rows} rows -> ` +
        `${rate((rows * 1000) / ms).padStart(7)} strikes/sec, ${rate((rows * c.m * 1000) / ms).padStart(7)} prices/sec`);
    }
    log(`  ${" ".repeat(26)} FFT ${(r.directMs / r.fftMs).toFixed(1)}x faster, N=${fftSize(c.m, c.m)}, |fft-direct| near the money ${r.near.toExponential(1)}${ok ? "" : " FAIL"}`);
  }

  const result = { gpu, date: new Date().toISOString().slice(0, 10), quick: QUICK, samples: SAMPLES, sweep, crossover, realistic };
  window.benchResult = result;
  const json = JSON.stringify(result);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  a.download = "fft_bench.json";
  a.textContent = "download fft_bench.json";
  document.body.append(a);
  log(`\n${fails ? `${fails} accuracy checks failed` : "all FFT results within 1e-4 of direct near the money"}`);
  log(`JSON for reference/plot_fft_crossover.py (or use the download link below):\n${json}`);
  document.title = fails ? "FAIL" : "PASS";
}

main().catch((e) => {
  log(`error: ${e.message}`);
  document.title = "ERROR";
});
