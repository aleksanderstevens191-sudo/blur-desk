// Inputs shared by the FFT checks (check.js, test-fft.mjs) and the benchmark
// (fft-bench.js): payoff rows on a log grid and a fat-tailed kernel.

// Rows of calls and puts (alternating, same strike in each pair) on the grid
// z_i = ln S0 + (i - center) dx, strikes spread over +/-30% of S0.
export function payoffRows({ n, dx, rows, S0 = 100 }) {
  const center = Math.floor(n / 2);
  const F = new Float64Array(rows * n);
  const pairs = Math.ceil(rows / 2);
  for (let r = 0; r < rows; r++) {
    const pair = r >> 1;
    const K = S0 * Math.exp(pairs > 1 ? -0.3 + (0.6 * pair) / (pairs - 1) : 0);
    const call = r % 2 === 0;
    for (let i = 0; i < n; i++) {
      const S = S0 * Math.exp((i - center) * dx);
      F[r * n + i] = Math.max(call ? S - K : K - S, 0);
    }
  }
  return { F, center };
}

// m taps spanning the whole width: a diffusion plus a skewed jump component
// (85% N(0, s), 15% N(-2s, 2.5s)), normalized to sum 1. The shape barely
// matters for speed; it gives the FFT a kernel with real tails to get right.
export function fatKernel(m) {
  const half = (m - 1) / 2;
  const s = Math.max(half / 14, 0.5);
  const k = new Float64Array(m);
  let sum = 0;
  for (let j = 0; j < m; j++) {
    const y = j - half;
    const a = Math.exp(-0.5 * (y / s) ** 2) / s;
    const b = Math.exp(-0.5 * ((y + 2 * s) / (2.5 * s)) ** 2) / (2.5 * s);
    k[j] = 0.85 * a + 0.15 * b;
    sum += k[j];
  }
  for (let j = 0; j < m; j++) k[j] /= sum;
  return k;
}

// A row that is zero except at its right end. The true blur is exactly zero on
// nodes i < n - 1 - half; any value there came round the circle.
export function sentinelRow(n, value = 100) {
  const f = new Float64Array(n);
  f[n - 1] = value;
  return f;
}
