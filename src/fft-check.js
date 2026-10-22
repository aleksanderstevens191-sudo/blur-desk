// /check.html, FFT section: the FFT blur against the direct blur, both on the
// GPU in 32-bit, each also against the exact (64-bit) blur, and the wraparound
// test (with the proof that it bites).
import { blur } from "./blur.js";
import { fftBlurGPU, directBlurGPU } from "./gpu-fft.js";
import { fftSize } from "./fft.js";
import { payoffRows, fatKernel, sentinelRow } from "./fft-cases.js";

// grid nodes n, kernel taps m, spacing dx in ln S (S0 = 100 at the center)
const CASES = [
  { n: 1001, m: 3, dx: 0.001 },
  { n: 1001, m: 401, dx: 0.001 },
  { n: 2000, m: 3999, dx: 0.001 }, // kernel almost twice as wide as the grid
  { n: 4097, m: 4097, dx: 0.001 },
  { n: 5341, m: 5341, dx: 0.001 }, // kernels.py's Gaussian grid at T=1: +/-2.67 in ln S
  { n: 7087, m: 7087, dx: 0.001 }, // its variance gamma grid at T=1: +/-3.56
];
const ROWS = 16;

const maxAbs = (a) => a.reduce((s, v) => Math.max(s, Math.abs(v)), 0);
const maxDiff = (a, b) => a.reduce((s, v, i) => Math.max(s, Math.abs(v - b[i])), 0);
const e = (x) => x.toExponential(2);

function exact(F, rows, n, k) {
  const out = new Float64Array(rows * n);
  for (let r = 0; r < rows; r++) out.set(blur(F.subarray(r * n, (r + 1) * n), k), r * n);
  return out;
}

export async function fftChecks(log, TOL) {
  let fails = 0;
  const line = (ok, s) => {
    if (!ok) fails++;
    log(`${ok ? "PASS" : "FAIL"} ${s}`);
  };
  log(`${ROWS} rows per case: calls and puts, strikes 74 to 135, S0 = 100 at the middle node.`);
  log("The FFT blurs each row minus its a + b S part (fft.js). Pass: |fft - direct| < 1e-4 near the");
  log("money (spots S0/2 to 2 S0, where prices are read) and, over the whole grid, |fft - exact| < 1e-4");
  log("on nodes worth up to S0 and < 1e-6 of the value on deep in-the-money nodes worth more ('scaled').");
  log("32-bit floats can't hold 1e-4 on those: one f32 step at 3383 is 2.4e-4. exact = blur() in 64-bit.");
  let plain = null;
  const scaledErr = (a, ex) => a.reduce((s, v, i) => Math.max(s, Math.abs(v - ex[i]) / Math.max(1, Math.abs(ex[i]) / 100)), 0);
  for (const { n, m, dx } of CASES) {
    const { F, center } = payoffRows({ n, dx, rows: ROWS });
    const k = fatKernel(m);
    const t0 = performance.now();
    const fft = await fftBlurGPU(F, ROWS, n, k, { dx });
    const dir = await directBlurGPU(F, ROWS, n, k);
    const ms = performance.now() - t0;
    const ex = exact(F, ROWS, n, k);
    const w = Math.round(Math.LN2 / dx);
    let near = 0;
    for (let r = 0; r < ROWS; r++) {
      for (let i = Math.max(0, center - w); i <= Math.min(n - 1, center + w); i++) near = Math.max(near, Math.abs(fft[r * n + i] - dir[r * n + i]));
    }
    const sf = scaledErr(fft, ex);
    line(near < TOL && sf < TOL, `n=${String(n).padEnd(5)} m=${String(m).padEnd(5)} N=${String(fftSize(n, m)).padEnd(6)}` +
      `payoffs to ${maxAbs(F).toFixed(0).padStart(4)}  near: |fft-direct| ${e(near)}` +
      `  whole grid, scaled: |fft-exact| ${e(sf)}, |direct-exact| ${e(scaledErr(dir, ex))}  (${ms.toFixed(0)} ms)`);
    if (n === CASES.at(-1).n) plain = scaledErr(await fftBlurGPU(F, ROWS, n, k), ex);
  }
  log(`without the split, the same FFT on the last case: whole grid, scaled |fft-exact| ${e(plain)}` +
    " (the 32-bit error of the biggest payoffs spreads to every node)");

  // Wraparound. A row that is zero except at its right end: its true blur is
  // exactly zero on every node more than `half` from that end. The padded FFT
  // must keep them at roundoff; an FFT one size too small (N < n + m - 1)
  // folds the end of the row onto the start, and the test has to catch that.
  log("\nwraparound test: row zero except f[n-1] = 100, so the blur must be 0 on nodes i < n-1-half");
  const n = 1000;
  const m = 401;
  const half = (m - 1) / 2;
  const k = fatKernel(m);
  const N = fftSize(n, m);
  const leak = (out) => maxAbs(out.subarray(0, n - 1 - half));
  const safe = leak(await fftBlurGPU(sentinelRow(n), 1, n, k));
  const short = leak(await fftBlurGPU(sentinelRow(n), 1, n, k, { N: N / 2 }));
  line(safe < TOL, `padded,         N=${N} >= n+m-1=${n + m - 1}: max |blur| there = ${e(safe)}`);
  const caught = short >= TOL;
  line(caught, `one size short, N=${N / 2} <  n+m-1=${n + m - 1}: max |blur| there = ${e(short)}` +
    (caught ? "  (leak caught, as it should be)" : "  (leak NOT caught: the test is blind)"));
  // the same short FFT on payoff rows is caught by the fft-vs-direct check
  const c = CASES[1];
  const { F } = payoffRows({ n: c.n, dx: c.dx, rows: ROWS });
  const kc = fatKernel(c.m);
  const Nc = fftSize(c.n, c.m);
  const d = maxDiff(await fftBlurGPU(F, ROWS, c.n, kc, { N: Nc / 2, dx: c.dx }), await directBlurGPU(F, ROWS, c.n, kc));
  line(d >= TOL, `one size short on the n=${c.n} m=${c.m} payoff rows (N=${Nc / 2}): |fft-direct| ${e(d)}` +
    (d >= TOL ? "  (caught)" : "  (NOT caught)"));
  return fails;
}
