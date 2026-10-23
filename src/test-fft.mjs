// node src/test-fft.mjs — the FFT blur (fft.js, 64-bit twin of the WGSL) against
// blur() from blur.js, and the wraparound test, with its proof that it bites.
import { blur } from "./blur.js";
import { fftBlurCPU, fftSize, expBlur } from "./fft.js";
import { payoffRows, fatKernel, sentinelRow } from "./fft-cases.js";

let fails = 0;
const line = (ok, s) => {
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s}`);
};
const e = (x) => x.toExponential(1);

function direct(F, rows, n, k) {
  const out = new Float64Array(rows * n);
  for (let r = 0; r < rows; r++) out.set(blur(F.subarray(r * n, (r + 1) * n), k), r * n);
  return out;
}
const maxDiff = (a, b) => a.reduce((s, v, i) => Math.max(s, Math.abs(v - b[i])), 0);
const maxAbs = (a) => a.reduce((s, v) => Math.max(s, Math.abs(v)), 0);

// n, m, rows, dx: odd and even row counts, radix-2 and radix-4 lengths,
// m = 1, m close to 2n (the widest kernel blur() accepts)
const CASES = [
  [10, 1, 1, 0.1], [10, 3, 3, 0.1], [64, 17, 4, 0.01], [100, 99, 5, 0.01],
  [256, 511, 2, 0.004], [1000, 257, 7, 0.001], [1001, 401, 16, 0.001], [3000, 2001, 6, 0.001],
];
console.log("FFT blur vs blur(), 64-bit, payoff rows (calls and puts); error relative to the largest value");
for (const [n, m, rows, dx] of CASES) {
  const { F } = payoffRows({ n, dx, rows });
  const k = fatKernel(m);
  const ref = direct(F, rows, n, k);
  const scale = maxAbs(ref);
  const plain = maxDiff(fftBlurCPU(F, rows, n, k), ref) / scale;
  const split = maxDiff(fftBlurCPU(F, rows, n, k, { dx }), ref) / scale;
  line(plain < 1e-13 && split < 1e-13, `n=${String(n).padEnd(5)} m=${String(m).padEnd(5)} rows=${String(rows).padEnd(3)}` +
    `N=${String(fftSize(n, m)).padEnd(5)} plain ${e(plain)}  split ${e(split)}`);
}

console.log("\nexpBlur (the exact blur of e^z, reflection included) vs blur() of e^z");
for (const [n, m, dx] of [[50, 99, 0.05], [1001, 401, 0.001], [2000, 3999, 0.001], [7087, 7087, 0.001]]) {
  const k = fatKernel(m);
  const c = (n - 1) / 2;
  const ref = blur(Float64Array.from({ length: n }, (_, i) => Math.exp((i - c) * dx)), k);
  const { E } = expBlur(k, n, dx);
  const d = maxDiff(E, ref) / maxAbs(ref);
  line(d < 1e-13, `n=${String(n).padEnd(5)} m=${String(m).padEnd(5)} relative ${e(d)}`);
}

console.log("\nwraparound: row zero except f[n-1] = 100; its blur is exactly 0 on nodes i < n-1-half");
for (const [n, m] of [[1000, 401], [512, 3], [300, 397]]) {
  const k = fatKernel(m);
  const half = (m - 1) / 2;
  const N = fftSize(n, m);
  const leak = (out) => maxAbs(out.subarray(0, n - 1 - half));
  const safe = leak(fftBlurCPU(sentinelRow(n), 1, n, k));
  const short = leak(fftBlurCPU(sentinelRow(n), 1, n, k, { N: N / 2 }));
  line(safe < 1e-12, `n=${n} m=${m} padded N=${N}: leak ${e(safe)}`);
  line(short > 1e-3, `n=${n} m=${m} one size short N=${N / 2} < n+m-1=${n + m - 1}: leak ${e(short)}` +
    (short > 1e-3 ? " (caught)" : " (NOT caught: the test is blind)"));
}

// a kernel as wide as the grid leaves no node out of the spike's reach; there
// the comparison with blur() is what catches an undersized FFT
for (const [n, m, dx] of [[300, 599, 0.01], [1001, 401, 0.001]]) {
  const { F } = payoffRows({ n, dx, rows: 4 });
  const k = fatKernel(m);
  const N = fftSize(n, m);
  const d = maxDiff(fftBlurCPU(F, 4, n, k, { N: N / 2, dx }), direct(F, 4, n, k));
  line(d > 1e-3, `n=${n} m=${m} payoff rows, one size short N=${N / 2}: |fft - blur()| ${e(d)}` +
    (d > 1e-3 ? " (caught)" : " (NOT caught)"));
}

console.log(fails ? `${fails} failed` : "all FFT checks pass");
process.exit(fails ? 1 : 0);
