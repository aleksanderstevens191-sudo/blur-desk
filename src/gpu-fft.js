// The blur through the FFT on the GPU (WebGPU / WGSL, 32-bit). The CPU twin,
// with the padding argument, is in fft.js; keep the two in step.
// One submit does: pack the rows (reflected edges written out, two real rows
// per complex row, the reversed kernel as one more row), forward FFT of all
// of them, multiply by the kernel's transform, inverse FFT, unpack.
// The direct blur it is compared with is DIRECT_WGSL below: ROWS_WGSL from
// gpu.js (the barrier step-pricer's shader) with a second level of blocking.
import { context, storage, uniform, readback, groups } from "./gpu.js";
import { fftSize, radices, twiddles, expSplit, expBlur } from "./fft.js";

const WG = 64;

// x[c][s]: rows 2c, 2c+1 padded, folded mod N (a short N folds, as it would
// in any undersized FFT); row C holds the reversed kernel.
export const PACK_WGSL = /* wgsl */ `
struct Params { n: u32, rows: u32, half: u32, N: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<vec2<f32>>;

// payoff row continued as a straight line past the grid (odd reflection)
fn at(row: u32, i: i32) -> f32 {
  let n = i32(p.n);
  let b = row * p.n;
  if (i < 0) { return 2.0 * f[b] - f[b + u32(-i)]; }
  if (i >= n) { return 2.0 * f[b + u32(n - 1)] - f[b + u32(2 * (n - 1) - i)]; }
  return f[b + u32(i)];
}

fn padded(row: u32, s: u32) -> f32 {
  if (row >= p.rows) { return 0.0; }
  let L = p.n + 2u * p.half;
  var v = 0.0;
  for (var q = s; q < L; q += p.N) { v += at(row, i32(q) - i32(p.half)); }
  return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  let c = gid.y;
  let C = (p.rows + 1u) / 2u;
  if (s >= p.N || c > C) { return; }
  if (c < C) {
    x[c * p.N + s] = vec2(padded(2u * c, s), padded(2u * c + 1u, s));
    return;
  }
  let m = 2u * p.half + 1u;
  var g = 0.0;
  for (var q = s; q < m; q += p.N) { g += k[m - 1u - q]; }
  x[C * p.N + s] = vec2(g, 0.0);
}
`;

// One radix-R Stockham pass (autosorting: no bit reversal, the output of the
// last pass is in natural order). Thread j of row y reads R points N/R apart,
// twists them by the twiddles of its sub-transform, does an R-point DFT and
// writes them Ns apart.
function stockham(R) {
  const dft = R === 2
    ? `dst[d] = v0 + v1;
  dst[d + p.Ns] = v0 - v1;`
    : `let a0 = v0 + v2;
  let a1 = v0 - v2;
  let b0 = v1 + v3;
  let b1 = v1 - v3;
  let rb = sg * vec2(-b1.y, b1.x);   // (sign i) b1
  dst[d] = a0 + b0;
  dst[d + p.Ns] = a1 + rb;
  dst[d + 2u * p.Ns] = a0 - b0;
  dst[d + 3u * p.Ns] = a1 - rb;`;
  const loads = Array.from({ length: R }, (_, r) =>
    r === 0
      ? `let v0 = src[base + j];`
      : `let v${r} = cmul(src[base + j + ${r}u * stride], tw[${r}u * k * step] * cj);`
  ).join("\n  ");
  return /* wgsl */ `
struct Params { N: u32, Ns: u32, inv: u32, p0: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = p.N / ${R}u;
  let j = gid.x;
  if (j >= stride) { return; }
  let base = gid.y * p.N;
  let k = j % p.Ns;
  let step = stride / p.Ns;
  let sg = select(-1.0, 1.0, p.inv == 1u);   // -1 forward, +1 inverse
  let cj = vec2(1.0, -sg);                    // inverse: conjugate twiddles
  ${loads}
  let d = base + (j / p.Ns) * p.Ns * ${R}u + k;
  ${dft}
}
`;
}
export const STOCKHAM2_WGSL = stockham(2);
export const STOCKHAM4_WGSL = stockham(4);

// Spectrum of each payoff pair times the kernel's (row C), and the 1/N of the
// inverse (a power of two, so exact).
export const MUL_WGSL = /* wgsl */ `
struct Params { N: u32, C: u32, p0: u32, p1: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  let c = gid.y;
  if (s >= p.N || c >= p.C) { return; }
  let g = x[p.C * p.N + s] / f32(p.N);
  let a = x[c * p.N + s];
  x[c * p.N + s] = vec2(a.x * g.x - a.y * g.y, a.x * g.y + a.y * g.x);
}
`;

// out[row][i] = y[i + 2 half]: real part for row 2c, imaginary for 2c+1,
// plus the exactly known blur a S + b E[i] of the part split off (fft.js).
export const UNPACK_WGSL = /* wgsl */ `
struct Params { n: u32, rows: u32, half: u32, N: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> ab: array<vec2<f32>>;  // (a, b) per row
@group(0) @binding(4) var<storage, read> E: array<f32>;         // E[0..n-1], then S

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let c = gid.y;
  if (i >= p.n || 2u * c >= p.rows) { return; }
  let v = x[c * p.N + (i + 2u * p.half) % p.N];
  let S = E[p.n];
  let e = E[i];
  let lo = ab[2u * c];
  out[2u * c * p.n + i] = v.x + lo.x * S + lo.y * e;
  if (2u * c + 1u < p.rows) {
    let hi = ab[2u * c + 1u];
    out[(2u * c + 1u) * p.n + i] = v.y + hi.x * S + hi.y * e;
  }
}
`;

// ROWS_WGSL with sums in blocks of 32 taps, blocks summed in super-blocks of
// 1024, super-blocks summed last. One level of blocks is enough for the
// barrier kernels (~1600 taps), but on the 38k-tap Merton kernel the running
// sum of 1200 block sums drifts by ~2e-4 (measured on the Mac). Two levels
// keep it near 3e-5 at the same speed.
export const DIRECT_WGSL = /* wgsl */ `
struct Params { n: u32, rows: u32, half: u32, p0: u32 }
const BLOCK = 32u;
const SUPER = 1024u;
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn at(row: u32, i: i32) -> f32 {
  let n = i32(p.n);
  let b = row * p.n;
  if (i < 0) { return 2.0 * f[b] - f[b + u32(-i)]; }
  if (i >= n) { return 2.0 * f[b + u32(n - 1)] - f[b + u32(2 * (n - 1) - i)]; }
  return f[b + u32(i)];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  let row = t / p.n;
  let i = t % p.n;
  if (row >= p.rows) { return; }
  let m = 2u * p.half + 1u;
  let base = i32(i) - i32(p.half);
  let interior = base >= 0 && base + i32(m) <= i32(p.n);
  var s = 0.0;
  for (var J0 = 0u; J0 < m; J0 += SUPER) {
    let J1 = min(J0 + SUPER, m);
    var sb = 0.0;
    for (var j0 = J0; j0 < J1; j0 += BLOCK) {
      let j1 = min(j0 + BLOCK, J1);
      var b = 0.0;
      for (var j = j0; j < j1; j++) {
        var v: f32;
        if (interior) { v = f[row * p.n + u32(base) + j]; } else { v = at(row, base + i32(j)); }
        b += k[j] * v;
      }
      sb += b;
    }
    s += sb;
  }
  out[row * p.n + i] = s;
}
`;

function checkShape(rows, n, m) {
  if (m % 2 !== 1) throw new Error("kernel length must be odd");
  if ((m - 1) / 2 >= n) throw new Error("kernel wider than grid; widen the grid");
  if (rows < 1) throw new Error("no rows");
}

const bind = (device, pipe, buffers) =>
  device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

const dispatch = (encoder, pipe, group, gx, gy = 1) => {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(gx, gy);
  pass.end();
};

// Set up the FFT blur of `rows` payoff rows F (rows x n) with `kernel`.
// Returns { device, N, encode(encoder), read(), setKernel(k), destroy() } so a
// benchmark can re-run it without re-uploading. N defaults to the safe size.
// With dx (the grid spacing in ln S) the rows are split as in fft.js on upload
// and only the bounded part goes through the FFT; leave it out for a plain FFT.
export async function fftBlurJob({ F, rows, n, kernel, N, dx }) {
  const m = kernel.length;
  checkShape(rows, n, m);
  N ??= fftSize(n, m);
  const half = (m - 1) / 2;
  const C = Math.ceil(rows / 2);
  if (C + 1 > 65535) throw new Error("too many rows for one dispatch");
  const { device, pipelineFor } = await context();
  const pipes = {
    pack: pipelineFor(PACK_WGSL),
    2: pipelineFor(STOCKHAM2_WGSL),
    4: pipelineFor(STOCKHAM4_WGSL),
    mul: pipelineFor(MUL_WGSL),
    unpack: pipelineFor(UNPACK_WGSL),
  };
  const split = dx !== undefined ? expSplit(F, rows, n, dx) : null;
  const expPart = (k) => {
    const { E, S } = split ? expBlur(k, n, dx) : { E: new Float64Array(n), S: 0 };
    const w = new Float32Array(n + 1);
    w.set(E);
    w[n] = S;
    return w;
  };
  const cbytes = (C + 1) * N * 8;
  const big = () => device.createBuffer({ size: cbytes, usage: GPUBufferUsage.STORAGE });
  const bufs = {
    shape: uniform(device, new Uint32Array([n, rows, half, N])),
    mulU: uniform(device, new Uint32Array([N, C, 0, 0])),
    f: storage(device, Float32Array.from(split ? split.r : F)),
    k: storage(device, Float32Array.from(kernel)),
    tw: storage(device, Float32Array.from(twiddles(N))),
    ab: storage(device, split ? Float32Array.from(split.ab) : new Float32Array(2 * rows)),
    E: storage(device, expPart(kernel)),
    a: big(),
    b: big(),
    out: device.createBuffer({ size: Math.max(16, rows * n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
  };
  // the pass list: forward over C + 1 rows (payoffs and kernel), inverse over C
  const steps = [];
  let [src, dst] = [bufs.a, bufs.b];
  const passUniforms = [];
  for (const inv of [0, 1]) {
    let Ns = 1;
    for (const R of radices(N)) {
      const u = uniform(device, new Uint32Array([N, Ns, inv, 0]));
      passUniforms.push(u);
      steps.push({ pipe: pipes[R], group: bind(device, pipes[R], [u, src, dst, bufs.tw]), gx: Math.ceil(N / R / WG), gy: inv ? C : C + 1 });
      [src, dst] = [dst, src];
      Ns *= R;
    }
    if (!inv) steps.push({ pipe: pipes.mul, group: bind(device, pipes.mul, [bufs.mulU, src]), gx: Math.ceil(N / WG), gy: C });
  }
  const packG = bind(device, pipes.pack, [bufs.shape, bufs.f, bufs.k, bufs.a]);
  const unpackG = bind(device, pipes.unpack, [bufs.shape, src, bufs.out, bufs.ab, bufs.E]);
  const encode = (encoder) => {
    dispatch(encoder, pipes.pack, packG, Math.ceil(N / WG), C + 1);
    for (const s of steps) dispatch(encoder, s.pipe, s.group, s.gx, s.gy);
    dispatch(encoder, pipes.unpack, unpackG, Math.ceil(n / WG), C);
  };
  return {
    device,
    N,
    passes: steps.length + 2,
    encode,
    setKernel: (k) => {
      device.queue.writeBuffer(bufs.k, 0, Float32Array.from(k));
      device.queue.writeBuffer(bufs.E, 0, expPart(k));
    },
    read: () => {
      const enc = device.createCommandEncoder();
      encode(enc);
      return readback(device, enc, bufs.out, rows * n);
    },
    destroy: () => [...Object.values(bufs), ...passUniforms].forEach((b) => b.destroy()),
  };
}

// The same rows blurred directly (DIRECT_WGSL: n m taps per row), same interface.
export async function directBlurJob({ F, rows, n, kernel }) {
  const m = kernel.length;
  checkShape(rows, n, m);
  const { device, pipelineFor } = await context();
  const pipe = pipelineFor(DIRECT_WGSL);
  const bufs = {
    u: uniform(device, new Uint32Array([n, rows, (m - 1) / 2, 0])),
    f: storage(device, Float32Array.from(F)),
    k: storage(device, Float32Array.from(kernel)),
    out: device.createBuffer({ size: Math.max(16, rows * n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
  };
  const group = bind(device, pipe, [bufs.u, bufs.f, bufs.k, bufs.out]);
  const [gx, gy] = groups(rows * n);
  const encode = (encoder) => dispatch(encoder, pipe, group, gx, gy);
  return {
    device,
    passes: 1,
    encode,
    setKernel: (k) => device.queue.writeBuffer(bufs.k, 0, Float32Array.from(k)),
    read: () => {
      const enc = device.createCommandEncoder();
      encode(enc);
      return readback(device, enc, bufs.out, rows * n);
    },
    destroy: () => Object.values(bufs).forEach((b) => b.destroy()),
  };
}

async function once(make, spec) {
  const job = await make(spec);
  const out = await job.read();
  job.destroy();
  return out;
}

// fftBlurCPU from fft.js, on the GPU.
export const fftBlurGPU = (F, rows, n, kernel, { N, dx } = {}) => once(fftBlurJob, { F, rows, n, kernel, N, dx });
// blur() of every row, directly, on the GPU.
export const directBlurGPU = (F, rows, n, kernel) => once(directBlurJob, { F, rows, n, kernel });
