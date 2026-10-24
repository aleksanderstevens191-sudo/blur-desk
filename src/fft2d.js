// The 2D blur through the FFT: host-side plan shared with the GPU (gpu2d.js),
// and its 64-bit CPU twin. Keep the two in step.
//
// A job from blur2d.js job2d() is a payoff image P (nx x ny) and a kernel W
// (mx x my); the blur wanted is the "valid" one,
//   out[a][b] = sum_ij W[i][j] P[a + i][b + j],  a < nx - mx + 1, b < ny - my + 1.
// Through the FFT it is a circular convolution of size Nx x Ny:
//   x = P padded with zeros, g[i][j] = W[mx-1-i][my-1-j] (reversed),
//   y = IFFT2(FFT2(x) FFT2(g)),  out[a][b] = y[a + mx - 1][b + my - 1].
// Output (a, b) reads P[a .. a+mx-1][b .. b+my-1], all inside the image, so
// nothing wraps as long as Nx >= nx and Ny >= ny. pack() folds the image mod
// (Nx, Ny), exactly what an undersized FFT does, so the wraparound tests can
// prove they would catch it.
//
// Rounding. A 32-bit FFT makes an error of ~1e-7 log N times the image's
// typical size at EVERY output, and a two-stock payoff grows like S1 or S2 (or
// their product) towards the image's corners: 28,000 on a 100 x 100 spot pair
// at 60% vol over two years. So the image is cut into parts, each measured in
// the numeraire that keeps it small (a change of numeraire, done per pixel):
//   P = sum_t e^{a_t u + b_t v} Q_t,  u = x - ln S1, v = y - ln S2,
// with tilts (a, b) in TILTS and each pixel in the part whose Q is smallest.
// A tilt moves into the kernel exactly, because the grid is uniform:
//   sum_ij W[i][j] e^{a u_{a+i} + b v_{b+j}} Q[a+i][b+j]
//     = e^{a u_a + b v_b} sum_ij (W[i][j] e^{a xo_i + b yo_j}) Q[a+i][b+j],
// so each part is blurred with its tilted kernel and multiplied back by
// e^{a u + b v} at the output. Near the spot that factor is ~1, and every Q_t
// is bounded by about S, so the FFT's error is that of a bounded image.
// Parts that are empty are skipped. Two parts ride in one complex image (re, im),
// and so do their two kernels; the spectra are separated by symmetry:
//   Z = FFT(q0 + i q1):  F0 = (Z[k] + conj Z[-k]) / 2,  F1 = (Z[k] - conj Z[-k]) / (2i).

export const TILTS = [[0, 0], [1, 0], [0, 1], [1, 1]];

export function nextPow2(x) {
  let N = 1;
  while (N < x) N *= 2;
  return N;
}

// The smallest FFT size that keeps the wraparound out of every output.
export const fftSize2d = (job) => [nextPow2(job.nx), nextPow2(job.ny)];

// Cut the payoff into tilted parts. Returns { tilts, Q: [Float64Array], G: [Float64Array] }
// with only the tilts some pixel uses, padded to an even count with an empty part.
export function tiltParts(job, tilts = TILTS) {
  const { P, nx, ny, mx, my, W, xs, ys, xo, yo } = job;
  const u0 = xs[(mx - 1) / 2 + job.qx], v0 = ys[(my - 1) / 2 + job.qy];
  const pick = new Uint8Array(nx * ny);
  const used = new Set();
  for (let i = 0; i < nx; i++) {
    const u = xs[i] - u0;
    for (let j = 0; j < ny; j++) {
      const v = ys[j] - v0;
      const p = Math.abs(P[i * ny + j]);
      if (p === 0) continue; // empty pixels belong to no part
      let best = 0;
      let bm = Infinity;
      tilts.forEach(([a, b], t) => {
        const m = p * Math.exp(-a * u - b * v);
        if (m < bm) [bm, best] = [m, t];
      });
      pick[i * ny + j] = best + 1;
      used.add(best);
    }
  }
  const ids = [...used].sort();
  if (ids.length === 0) ids.push(0);
  const out = { tilts: [], Q: [], G: [] };
  for (const t of ids) {
    const [a, b] = tilts[t];
    const Q = new Float64Array(nx * ny);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        if (pick[i * ny + j] === t + 1) Q[i * ny + j] = P[i * ny + j] * Math.exp(-a * (xs[i] - u0) - b * (ys[j] - v0));
      }
    }
    const G = new Float64Array(mx * my);
    for (let i = 0; i < mx; i++) for (let j = 0; j < my; j++) G[i * my + j] = W[i * my + j] * Math.exp(a * xo[i] + b * yo[j]);
    out.tilts.push([a, b]);
    out.Q.push(Q);
    out.G.push(G);
  }
  if (out.tilts.length % 2) {
    out.tilts.push([0, 0]);
    out.Q.push(new Float64Array(nx * ny));
    out.G.push(new Float64Array(mx * my));
  }
  return out;
}

// Two real images (rows x cols) into one complex Nx x Ny image (interleaved
// re, im), folded mod (Nx, Ny). reverse: store A[rows-1-i][cols-1-j] (the kernel).
export function pack2(A, B, rows, cols, Nx, Ny, reverse = false, dst = new Float64Array(2 * Nx * Ny)) {
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const s = reverse ? (rows - 1 - i) * cols + (cols - 1 - j) : i * cols + j;
      const d = 2 * ((i % Nx) * Ny + (j % Ny));
      dst[d] += A[s];
      dst[d + 1] += B[s];
    }
  }
  return dst;
}

// The untilt factors at the outputs: e^{a u_a + b v_b}, u_a = (a - qx) dx.
export const untilt = (job, [a, b], ia, ib) => Math.exp(a * (ia - job.qx) * job.dx + b * (ib - job.qy) * job.dy);

// ---------- CPU twin (64-bit) ----------

// In-place radix-2 FFT of n complex points at re/im index 2 (off + t * stride).
function fft1(z, off, stride, n, sign) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const a = 2 * (off + i * stride), b = 2 * (off + j * stride);
      [z[a], z[b]] = [z[b], z[a]];
      [z[a + 1], z[b + 1]] = [z[b + 1], z[a + 1]];
    }
  }
  for (let len = 2; len <= n; len *= 2) {
    const ang = (sign * 2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        const a = 2 * (off + (i + k) * stride), b = 2 * (off + (i + k + len / 2) * stride);
        const xr = z[b] * wr - z[b + 1] * wi, xi = z[b] * wi + z[b + 1] * wr;
        z[b] = z[a] - xr;
        z[b + 1] = z[a + 1] - xi;
        z[a] += xr;
        z[a + 1] += xi;
      }
    }
  }
}

function fft2(z, Nx, Ny, sign) {
  for (let i = 0; i < Nx; i++) fft1(z, i * Ny, 1, Ny, sign);
  for (let j = 0; j < Ny; j++) fft1(z, j, Ny, Nx, sign);
}

// Spectrum of the pair's two blurs, H = F0 G0 + i F1 G1 (see the header).
export function mulPair(ZP, ZG, Nx, Ny) {
  const H = new Float64Array(2 * Nx * Ny);
  const s = 1 / (Nx * Ny);
  for (let kx = 0; kx < Nx; kx++) {
    for (let ky = 0; ky < Ny; ky++) {
      const k = 2 * (kx * Ny + ky);
      const m = 2 * (((Nx - kx) % Nx) * Ny + ((Ny - ky) % Ny));
      // X = Z[k], Y = conj Z[-k]; F0 = (X + Y)/2, F1 = (X - Y)/(2i) = (-i)(X - Y)/2
      const sep = (Z) => {
        const xr = Z[k], xi = Z[k + 1], yr = Z[m], yi = -Z[m + 1];
        return [(xr + yr) / 2, (xi + yi) / 2, (xi - yi) / 2, -(xr - yr) / 2];
      };
      const [f0r, f0i, f1r, f1i] = sep(ZP);
      const [g0r, g0i, g1r, g1i] = sep(ZG);
      const h0r = f0r * g0r - f0i * g0i, h0i = f0r * g0i + f0i * g0r;
      const h1r = f1r * g1r - f1i * g1i, h1i = f1r * g1i + f1i * g1r;
      H[k] = s * (h0r - h1i);
      H[k + 1] = s * (h0i + h1r);
    }
  }
  return H;
}

// The whole FFT blur in 64-bit: undiscounted outputs (ox x oy), like the GPU's.
// N = [Nx, Ny] defaults to the safe size; tilts = [[0, 0]] turns the split off.
export function fftBlur2dCPU(job, { N, tilts } = {}) {
  const [Nx, Ny] = N ?? fftSize2d(job);
  const { nx, ny, mx, my, ox, oy } = job;
  const parts = tiltParts(job, tilts);
  const out = new Float64Array(ox * oy);
  for (let p = 0; p < parts.tilts.length; p += 2) {
    const ZP = pack2(parts.Q[p], parts.Q[p + 1], nx, ny, Nx, Ny);
    const ZG = pack2(parts.G[p], parts.G[p + 1], mx, my, Nx, Ny, true);
    fft2(ZP, Nx, Ny, -1);
    fft2(ZG, Nx, Ny, -1);
    const Y = mulPair(ZP, ZG, Nx, Ny);
    fft2(Y, Nx, Ny, +1);
    for (let a = 0; a < ox; a++) {
      for (let b = 0; b < oy; b++) {
        const t = 2 * (((a + mx - 1) % Nx) * Ny + ((b + my - 1) % Ny));
        out[a * oy + b] += Y[t] * untilt(job, parts.tilts[p], a, b) + Y[t + 1] * untilt(job, parts.tilts[p + 1], a, b);
      }
    }
  }
  return out;
}

// The direct valid blur in 64-bit, undiscounted, for the CPU tests.
export function directBlur2dCPU(job) {
  const { P, W, mx, my, ny, ox, oy } = job;
  const out = new Float64Array(ox * oy);
  for (let a = 0; a < ox; a++) {
    for (let b = 0; b < oy; b++) {
      let s = 0;
      for (let i = 0; i < mx; i++) {
        const base = (a + i) * ny + b;
        for (let j = 0; j < my; j++) s += W[i * my + j] * P[base + j];
      }
      out[a * oy + b] = s;
    }
  }
  return out;
}
