// The two-stock blur on the GPU (WebGPU / WGSL, 32-bit): the valid 2D blur of
// a payoff image by the correlated kernel, two ways.
//   direct  DIRECT2D_WGSL: one thread per output, mx my taps, in blocks of 32
//           (a row of the kernel in blocks, rows in blocks): the same blocked
//           summation as the 1D shaders, which Metal's compiler can't undo.
//   fft     the plan in fft2d.js: the payoff cut into tilted parts (a change
//           of numeraire per pixel, so a 32-bit FFT only ever sees a bounded
//           image), two parts per complex image, 2D Stockham FFTs (rows, then
//           columns), the spectra separated and multiplied, inverse, untilt.
// Both read job2d() from blur2d.js and return the undiscounted outputs on the
// (2 qx + 1) x (2 qy + 1) patch of spots.
import { context, storage, readback, groups } from "./gpu.js";
import { radices, twiddles } from "./fft.js";
import { fftSize2d, tiltParts, pack2 } from "./fft2d.js";

const WG = 64;

export const DIRECT2D_WGSL = /* wgsl */ `
struct Params { nx: u32, ny: u32, mx: u32, my: u32, ox: u32, oy: u32, p0: u32, p1: u32 }
const BLOCK = 32u;
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  if (t >= p.ox * p.oy) { return; }
  let a = t / p.oy;
  let b = t % p.oy;
  var s = 0.0;
  for (var I0 = 0u; I0 < p.mx; I0 += BLOCK) {
    let I1 = min(I0 + BLOCK, p.mx);
    var rows = 0.0;
    for (var i = I0; i < I1; i++) {
      let pb = (a + i) * p.ny + b;
      let wb = i * p.my;
      var row = 0.0;
      for (var j0 = 0u; j0 < p.my; j0 += BLOCK) {
        let j1 = min(j0 + BLOCK, p.my);
        var blk = 0.0;
        for (var j = j0; j < j1; j++) { blk += W[wb + j] * P[pb + j]; }
        row += blk;
      }
      rows += row;
    }
    s += rows;
  }
  out[t] = s;
}
`;

// One radix-R Stockham pass on transforms laid out with strides: element t of
// transform g of image z sits at z img + g gs + t es. Rows: es = 1, gs = Ny;
// columns: es = Ny, gs = 1. Otherwise the 1D pass of gpu-fft.js.
function stockham2d(R) {
  const dft = R === 2
    ? `dst[at(d)] = v0 + v1;
  dst[at(d + p.Ns)] = v0 - v1;`
    : `let a0 = v0 + v2;
  let a1 = v0 - v2;
  let b0 = v1 + v3;
  let b1 = v1 - v3;
  let rb = sg * vec2(-b1.y, b1.x);   // (sign i) b1
  dst[at(d)] = a0 + b0;
  dst[at(d + p.Ns)] = a1 + rb;
  dst[at(d + 2u * p.Ns)] = a0 - b0;
  dst[at(d + 3u * p.Ns)] = a1 - rb;`;
  const loads = Array.from({ length: R }, (_, r) =>
    r === 0
      ? `let v0 = src[at(j)];`
      : `let v${r} = cmul(src[at(j + ${r}u * stride)], tw[${r}u * k * step] * cj);`
  ).join("\n  ");
  return /* wgsl */ `
struct Params { N: u32, Ns: u32, inv: u32, es: u32, gs: u32, img: u32, count: u32, p0: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> tw: array<vec2<f32>>;

var<private> base: u32;
fn at(t: u32) -> u32 { return base + t * p.es; }

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = p.N / ${R}u;
  let j = gid.x;
  if (j >= stride || gid.y >= p.count) { return; }
  base = gid.z * p.img + gid.y * p.gs;
  let k = j % p.Ns;
  let step = stride / p.Ns;
  let sg = select(-1.0, 1.0, p.inv == 1u);   // -1 forward, +1 inverse
  let cj = vec2(1.0, -sg);                    // inverse: conjugate twiddles
  ${loads}
  let d = (j / p.Ns) * p.Ns * ${R}u + k;
  ${dft}
}
`;
}
export const STOCKHAM2D_2_WGSL = stockham2d(2);
export const STOCKHAM2D_4_WGSL = stockham2d(4);

// Pair p: payoff parts in image p, their kernels in image P + p. Separate each
// image's two real spectra by symmetry, multiply part by kernel, and write
// H = F0 G0 + i F1 G1 (scaled by 1/(Nx Ny)) to image p of dst.
export const MUL2D_WGSL = /* wgsl */ `
struct Params { Nx: u32, Ny: u32, P: u32, p0: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Z: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> H: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  let img = p.Nx * p.Ny;
  if (t >= img) { return; }
  let kx = t / p.Ny;
  let ky = t % p.Ny;
  let m = ((p.Nx - kx) % p.Nx) * p.Ny + (p.Ny - ky) % p.Ny;
  let pf = gid.z * img;
  let pg = (p.P + gid.z) * img;
  // X = Z[k], Y = conj Z[-k]: F0 = (X + Y)/2, F1 = -i (X - Y)/2
  let xf = Z[pf + t];
  let yf = Z[pf + m] * vec2(1.0, -1.0);
  let xg = Z[pg + t];
  let yg = Z[pg + m] * vec2(1.0, -1.0);
  let f0 = 0.5 * (xf + yf);
  let df = 0.5 * (xf - yf);
  let f1 = vec2(df.y, -df.x);
  let g0 = 0.5 * (xg + yg);
  let dg = 0.5 * (xg - yg);
  let g1 = vec2(dg.y, -dg.x);
  let h0 = cmul(f0, g0);
  let h1 = cmul(f1, g1);
  H[pf + t] = vec2(h0.x - h1.y, h0.y + h1.x) / f32(img);
}
`;

// out[a][b] = sum over pairs of re y[a+mx-1][b+my-1] e^{tilt0 . (u, v)} + im ... e^{tilt1 . (u, v)}.
export const UNPACK2D_WGSL = /* wgsl */ `
struct Params { ox: u32, oy: u32, Nx: u32, Ny: u32, mx: u32, my: u32, P: u32, qx: u32, qy: u32, dx: f32, dy: f32, p0: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Y: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> tilt: array<vec4<f32>>;   // (a0, b0, a1, b1) per pair
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  if (t >= p.ox * p.oy) { return; }
  let a = t / p.oy;
  let b = t % p.oy;
  let u = (f32(a) - f32(p.qx)) * p.dx;
  let v = (f32(b) - f32(p.qy)) * p.dy;
  let s = ((a + p.mx - 1u) % p.Nx) * p.Ny + (b + p.my - 1u) % p.Ny;
  var acc = 0.0;
  for (var q = 0u; q < p.P; q++) {
    let y = Y[q * p.Nx * p.Ny + s];
    let w = tilt[q];
    acc += y.x * exp(w.x * u + w.y * v) + y.y * exp(w.z * u + w.w * v);
  }
  out[t] = acc;
}
`;

function uniformWords(device, words) {
  const buf = device.createBuffer({ size: Math.max(16, words.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, words);
  return buf;
}

const bind = (device, pipe, buffers) =>
  device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

const dispatch = (encoder, pipe, group, gx, gy = 1, gz = 1) => {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(gx, gy, gz);
  pass.end();
};

const outBuffer = (device, n) => device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

// Direct blur of a job: { device, passes, encode(encoder), read(), destroy() }.
export async function direct2dJob(job) {
  const { device, pipelineFor } = await context();
  const pipe = pipelineFor(DIRECT2D_WGSL);
  const n = job.ox * job.oy;
  const bufs = {
    u: uniformWords(device, new Uint32Array([job.nx, job.ny, job.mx, job.my, job.ox, job.oy, 0, 0])),
    P: storage(device, Float32Array.from(job.P)),
    W: storage(device, Float32Array.from(job.W)),
    out: outBuffer(device, n),
  };
  const group = bind(device, pipe, [bufs.u, bufs.P, bufs.W, bufs.out]);
  const [gx, gy] = groups(n);
  const encode = (encoder) => dispatch(encoder, pipe, group, gx, gy);
  return {
    device,
    passes: 1,
    encode,
    read: () => {
      const enc = device.createCommandEncoder();
      encode(enc);
      return readback(device, enc, bufs.out, n);
    },
    destroy: () => Object.values(bufs).forEach((b) => b.destroy()),
  };
}

// FFT blur of a job. N = [Nx, Ny] defaults to the safe size (a smaller one is
// for the wraparound test); tilts as in fft2d.js tiltParts.
export async function fft2dJob(job, { N, tilts } = {}) {
  const [Nx, Ny] = N ?? fftSize2d(job);
  const img = Nx * Ny;
  if (Nx > 65535 * WG || Ny > 65535) throw new Error("image too large");
  const { device, pipelineFor } = await context();
  const pipes = {
    2: pipelineFor(STOCKHAM2D_2_WGSL),
    4: pipelineFor(STOCKHAM2D_4_WGSL),
    mul: pipelineFor(MUL2D_WGSL),
    unpack: pipelineFor(UNPACK2D_WGSL),
  };
  const parts = tiltParts(job, tilts);
  const P = parts.tilts.length / 2;
  // host side, in 64-bit: pack (and fold) parts and reversed kernels, then round once
  const Z = new Float64Array(2 * 2 * P * img);
  for (let q = 0; q < P; q++) {
    pack2(parts.Q[2 * q], parts.Q[2 * q + 1], job.nx, job.ny, Nx, Ny, false, Z.subarray(2 * q * img, 2 * (q + 1) * img));
    pack2(parts.G[2 * q], parts.G[2 * q + 1], job.mx, job.my, Nx, Ny, true, Z.subarray(2 * (P + q) * img, 2 * (P + q + 1) * img));
  }
  const tiltWords = Float32Array.from(parts.tilts.flat());
  const ubuf = new ArrayBuffer(48);
  new Uint32Array(ubuf, 0, 9).set([job.ox, job.oy, Nx, Ny, job.mx, job.my, P, job.qx, job.qy]);
  new Float32Array(ubuf, 36, 2).set([job.dx, job.dy]);
  const bytes = 2 * P * img * 8;
  const big = () => device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const nOut = job.ox * job.oy;
  const bufs = {
    z0: device.createBuffer({ size: 2 * P * img * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
    a: big(),
    b: big(),
    twx: storage(device, Float32Array.from(twiddles(Nx))),
    twy: storage(device, Float32Array.from(twiddles(Ny))),
    mulU: uniformWords(device, new Uint32Array([Nx, Ny, P, 0])),
    unpackU: uniformWords(device, new Uint32Array(ubuf)),
    tilt: storage(device, tiltWords),
    out: outBuffer(device, nOut),
  };
  device.queue.writeBuffer(bufs.z0, 0, Float32Array.from(Z));
  const steps = [];
  const uniforms = [];
  let [src, dst] = [bufs.a, bufs.b];
  // the four transforms: forward rows and columns over 2P images, inverse over P
  const transform = (inv, images) => {
    for (const [len, es, gs, count, tw] of [[Ny, 1, Ny, Nx, bufs.twy], [Nx, Ny, 1, Ny, bufs.twx]]) {
      let Ns = 1;
      for (const R of radices(len)) {
        const u = uniformWords(device, new Uint32Array([len, Ns, inv, es, gs, img, count, 0]));
        uniforms.push(u);
        steps.push({ pipe: pipes[R], group: bind(device, pipes[R], [u, src, dst, tw]), g: [Math.ceil(len / R / WG), count, images] });
        [src, dst] = [dst, src];
        Ns *= R;
      }
    }
  };
  transform(0, 2 * P);
  const [mx, my] = groups(img);
  steps.push({ pipe: pipes.mul, group: bind(device, pipes.mul, [bufs.mulU, src, dst]), g: [mx, my, P] });
  [src, dst] = [dst, src];
  transform(1, P);
  const [ux, uy] = groups(nOut);
  steps.push({ pipe: pipes.unpack, group: bind(device, pipes.unpack, [bufs.unpackU, src, bufs.tilt, bufs.out]), g: [ux, uy, 1] });
  // the forward passes overwrite their input, so each run starts with a copy of it
  const encode = (encoder) => {
    encoder.copyBufferToBuffer(bufs.z0, 0, bufs.a, 0, bytes);
    for (const s of steps) dispatch(encoder, s.pipe, s.group, ...s.g);
  };
  return {
    device,
    N: [Nx, Ny],
    parts: parts.tilts.filter((_, i) => parts.Q[i].some((v) => v !== 0)),
    passes: steps.length,
    encode,
    read: () => {
      const enc = device.createCommandEncoder();
      encode(enc);
      return readback(device, enc, bufs.out, nOut);
    },
    destroy: () => [...Object.values(bufs), ...uniforms].forEach((b) => b.destroy()),
  };
}

async function once(make, ...args) {
  const job = await make(...args);
  const out = await job.read();
  job.destroy();
  return out;
}

export const direct2dGPU = (job) => once(direct2dJob, job);
export const fft2dGPU = (job, opts) => once(fft2dJob, job, opts);
