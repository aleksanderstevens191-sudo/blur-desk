// The blur through the FFT. CPU twin (64-bit) of the WGSL in gpu-fft.js; keep
// the two in step. blur(f, k) from blur.js is
//   out[i] = sum_j k[j] f(i + j - half),  odd-reflection edges, m = 2 half + 1 taps,
// which costs n m per row. Through the FFT it costs about N log N, N ~ n + m:
//   1. pad:    x[p] = f(p - half) for p = 0 .. L-1, L = n + 2 half = n + m - 1
//              (the reflected edges written out), zeros up to N
//   2. kernel: g[q] = k[m-1-q] (reversed), zeros up to N
//   3. y = IFFT(FFT(x) FFT(g)), the circular convolution of length N:
//              y[t] = sum_q x[(t - q) mod N] g[q]
//   4. out[i] = y[i + 2 half]
// Output t = i + 2 half reads x[t - q] for q < m, i.e. x[i .. i + 2 half], all
// inside 0 .. L-1, so nothing wraps around as long as N >= L. The zeros after L
// only reach outputs t < 2 half, which are thrown away. With N < L the far end
// of the row folds back onto its start and leaks into prices: pack() folds
// mod N exactly as an undersized FFT would, and test-fft.mjs checks that the
// leak is caught.
//
// Two real rows ride in one complex row (re, im): the kernel is real, so the
// real and imaginary parts of the result are the two blurs.
//
// Rounding. A 32-bit FFT makes an error of about 1e-7 log N times the row's
// typical size at EVERY node, where the direct blur's error stays local. A
// call row on a wide grid reaches thousands of times S0 at its right end, and
// that error lands on the deep out-of-the-money nodes too. So each row is split
// first: f = r + a + b u, u_p = e^{(p - c) dx}, with a and b fitted to the row
// by least squares. For a call, b u takes out the e^z growth and r is about
// as big as the put at the same strike (bounded by K).
// The FFT blurs r; the blur of a + b u, odd reflection included, is known in
// closed form (expBlur), so it is added back exactly.

export function nextPow2(x) {
  let N = 1;
  while (N < x) N *= 2;
  return N;
}

// The smallest FFT length that keeps the wraparound out of every output.
export function fftSize(n, m) {
  return nextPow2(n + m - 1);
}

// Radix-4 Stockham passes, with one radix-2 pass first when log2 N is odd.
export function radices(N) {
  const out = [];
  let p = 1;
  if (Math.log2(N) % 2) {
    out.push(2);
    p = 2;
  }
  while (p < N) {
    out.push(4);
    p *= 4;
  }
  return out;
}

// tw[t] = exp(-2 pi i t / N), t < N, as interleaved (re, im). Computed in 64-bit;
// the GPU gets them rounded once to 32-bit.
export function twiddles(N) {
  const tw = new Float64Array(2 * N);
  for (let t = 0; t < N; t++) {
    const a = (-2 * Math.PI * t) / N;
    tw[2 * t] = Math.cos(a);
    tw[2 * t + 1] = Math.sin(a);
  }
  return tw;
}

// Rows f (rows x n, row-major) and kernel k into C + 1 complex rows of N:
// rows 2c and 2c+1 in complex row c, the reversed kernel in row C.
// Slot s holds the sum over p = s, s + N, ... < L, so an N below L folds.
export function pack(f, rows, n, k, N) {
  const m = k.length;
  const half = (m - 1) / 2;
  const L = n + m - 1;
  const C = Math.ceil(rows / 2);
  const x = new Float64Array(2 * (C + 1) * N);
  const at = (r, i) => {
    const b = r * n;
    if (i < 0) return 2 * f[b] - f[b - i];
    if (i >= n) return 2 * f[b + n - 1] - f[b + 2 * (n - 1) - i];
    return f[b + i];
  };
  for (let c = 0; c < C; c++) {
    for (let p = 0; p < L; p++) {
      const s = 2 * (c * N + (p % N));
      x[s] += at(2 * c, p - half);
      if (2 * c + 1 < rows) x[s + 1] += at(2 * c + 1, p - half);
    }
  }
  for (let q = 0; q < m; q++) x[2 * (C * N + (q % N))] += k[m - 1 - q];
  return x;
}

// One Stockham pass of radix R over `count` complex rows of length N, src -> dst.
// Ns is the length of the sub-transforms already done; sign -1 forward, +1 inverse.
export function stockhamPass(src, dst, N, count, Ns, R, sign, tw) {
  const stride = N / R;
  const step = N / (R * Ns);
  const v = new Float64Array(2 * R);
  for (let row = 0; row < count; row++) {
    const base = row * N;
    for (let j = 0; j < stride; j++) {
      const k = j % Ns;
      for (let r = 0; r < R; r++) {
        const a = 2 * (base + j + r * stride);
        const t = 2 * ((r * k * step) % N);
        const wr = tw[t];
        const wi = -sign * tw[t + 1];
        v[2 * r] = src[a] * wr - src[a + 1] * wi;
        v[2 * r + 1] = src[a] * wi + src[a + 1] * wr;
      }
      const d = base + Math.floor(j / Ns) * Ns * R + k;
      if (R === 2) {
        dst[2 * d] = v[0] + v[2];
        dst[2 * d + 1] = v[1] + v[3];
        dst[2 * (d + Ns)] = v[0] - v[2];
        dst[2 * (d + Ns) + 1] = v[1] - v[3];
        continue;
      }
      // 4-point DFT; (sign i) b1 = sign * (-b1.im, b1.re)
      const a0r = v[0] + v[4], a0i = v[1] + v[5];
      const a1r = v[0] - v[4], a1i = v[1] - v[5];
      const b0r = v[2] + v[6], b0i = v[3] + v[7];
      const b1r = v[2] - v[6], b1i = v[3] - v[7];
      const rr = -sign * b1i, ri = sign * b1r;
      dst[2 * d] = a0r + b0r;
      dst[2 * d + 1] = a0i + b0i;
      dst[2 * (d + Ns)] = a1r + rr;
      dst[2 * (d + Ns) + 1] = a1i + ri;
      dst[2 * (d + 2 * Ns)] = a0r - b0r;
      dst[2 * (d + 2 * Ns) + 1] = a0i - b0i;
      dst[2 * (d + 3 * Ns)] = a1r - rr;
      dst[2 * (d + 3 * Ns) + 1] = a1i - ri;
    }
  }
}

// All passes of one transform; returns the buffer holding the result.
export function transform(a, b, N, count, sign, tw) {
  let Ns = 1;
  for (const R of radices(N)) {
    stockhamPass(a, b, N, count, Ns, R, sign, tw);
    [a, b] = [b, a];
    Ns *= R;
  }
  return [a, b];
}

// The split f = r + a + b u of each row (u_p = e^{(p - c) dx}, c the middle
// node), a and b by least squares, which keeps r small at both ends.
// Returns the rows r and the (a, b) pairs.
export function expSplit(f, rows, n, dx) {
  const c = (n - 1) / 2;
  const u = Float64Array.from({ length: n }, (_, i) => Math.exp((i - c) * dx));
  let su = 0;
  let suu = 0;
  for (let i = 0; i < n; i++) {
    su += u[i];
    suu += u[i] * u[i];
  }
  const det = n * suu - su * su;
  const r = new Float64Array(rows * n);
  const ab = new Float64Array(2 * rows);
  for (let row = 0; row < rows; row++) {
    const o = row * n;
    let sf = 0;
    let sfu = 0;
    for (let i = 0; i < n; i++) {
      sf += f[o + i];
      sfu += f[o + i] * u[i];
    }
    const a = (sf * suu - su * sfu) / det;
    const b = (n * sfu - su * sf) / det;
    ab[2 * row] = a;
    ab[2 * row + 1] = b;
    for (let i = 0; i < n; i++) r[o + i] = f[o + i] - a - b * u[i];
  }
  return { r, ab };
}

// The blur of u (odd reflection included) at every node, E[i], and S = sum k,
// so that blur(a + b u)[i] = a S + b E[i]. O(n + m) with prefix sums:
//   inside the grid u blurs to u_i M, M = sum_j k_j e^{(j - half) dx};
//   past the left end u_p is reflected to 2 u_0 - u_{-p}, past the right end
//   to 2 u_{n-1} - u_{2(n-1)-p}, so those taps add k_j (2 u_end - u_mirror - u_p).
export function expBlur(k, n, dx) {
  const m = k.length;
  const half = (m - 1) / 2;
  const c = (n - 1) / 2;
  const u = (p) => Math.exp((p - c) * dx);
  // prefix sums (j < J) and suffix sums (j >= J) of k_j, k_j e^{-j dx}, k_j e^{j dx}
  const pre = [new Float64Array(m + 1), new Float64Array(m + 1), new Float64Array(m + 1)];
  const suf = [new Float64Array(m + 1), new Float64Array(m + 1), new Float64Array(m + 1)];
  for (let j = 0; j < m; j++) {
    pre[0][j + 1] = pre[0][j] + k[j];
    pre[1][j + 1] = pre[1][j] + k[j] * Math.exp(-j * dx);
    pre[2][j + 1] = pre[2][j] + k[j] * Math.exp(j * dx);
  }
  for (let j = m - 1; j >= 0; j--) {
    suf[0][j] = suf[0][j + 1] + k[j];
    suf[1][j] = suf[1][j + 1] + k[j] * Math.exp(-j * dx);
    suf[2][j] = suf[2][j + 1] + k[j] * Math.exp(j * dx);
  }
  let M = 0;
  for (let j = 0; j < m; j++) M += k[j] * Math.exp((j - half) * dx);
  const E = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let e = u(i) * M;
    const J = half - i; // taps j < J land at p = i + j - half < 0
    if (J > 0) {
      // sum_{j<J} k_j (2 u_0 - u_{half-i-j} - u_{i+j-half})
      e += 2 * u(0) * pre[0][J] - u(half - i) * pre[1][J] - u(i - half) * pre[2][J];
    }
    const J1 = n - i + half; // taps j >= J1 land at p > n - 1
    if (J1 < m) {
      // sum_{j>=J1} k_j (2 u_{n-1} - u_{2(n-1)-i+half-j} - u_{i+j-half})
      e += 2 * u(n - 1) * suf[0][J1] - u(2 * (n - 1) - i + half) * suf[1][J1] - u(i - half) * suf[2][J1];
    }
    E[i] = e;
  }
  return { E, S: pre[0][m] };
}

// blur() of every row through the FFT. N defaults to the safe size; pass a
// smaller N only to watch the wraparound happen. With dx, rows are split as
// above and only r goes through the FFT.
export function fftBlurCPU(f, rows, n, k, { N = fftSize(n, k.length), dx } = {}) {
  const m = k.length;
  const half = (m - 1) / 2;
  if (half >= n) throw new Error("kernel wider than grid; widen the grid");
  const split = dx !== undefined ? expSplit(f, rows, n, dx) : null;
  const C = Math.ceil(rows / 2);
  const tw = twiddles(N);
  let [x, y] = transform(pack(split ? split.r : f, rows, n, k, N), new Float64Array(2 * (C + 1) * N), N, C + 1, -1, tw);
  const g = 2 * C * N;
  for (let c = 0; c < C; c++) {
    for (let s = 0; s < N; s++) {
      const a = 2 * (c * N + s);
      const gr = x[g + 2 * s] / N, gi = x[g + 2 * s + 1] / N;
      const re = x[a] * gr - x[a + 1] * gi;
      x[a + 1] = x[a] * gi + x[a + 1] * gr;
      x[a] = re;
    }
  }
  [x, y] = transform(x, y, N, C, +1, tw);
  const out = new Float64Array(rows * n);
  const { E, S } = split ? expBlur(k, n, dx) : { E: new Float64Array(n), S: 0 };
  for (let r = 0; r < rows; r++) {
    const c = r >> 1;
    const [a, b] = split ? split.ab.subarray(2 * r, 2 * r + 2) : [0, 0];
    for (let i = 0; i < n; i++) out[r * n + i] = x[2 * (c * N + ((i + 2 * half) % N)) + (r & 1)] + a * S + b * E[i];
  }
  return out;
}
