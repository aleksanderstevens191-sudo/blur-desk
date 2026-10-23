// Blur Desk on the GPU: the blur() from blur.js as a WebGPU compute shader.
// Same math: out[i] = sum_j k[j] * f(i + j - half), odd-reflection edges,
// kernel passed in as a buffer. WebGPU computes in 32-bit floats.

export const BLUR_WGSL = /* wgsl */ `
struct Params {
  n: u32,
  half: u32,
  mode: u32,   // 0 = plain blur, 1 = American step: max(blur, payoff)
}

const BLOCK = 32u;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> payoff: array<f32>;

// Payoffs continue as straight lines past the grid (odd reflection).
fn at(i: i32) -> f32 {
  let n = i32(p.n);
  if (i < 0) { return 2.0 * f[0] - f[-i]; }
  if (i >= n) { return 2.0 * f[n - 1] - f[2 * (n - 1) - i]; }
  return f[i];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let m = 2u * p.half + 1u;
  let base = i32(i) - i32(p.half);
  // Sum in blocks of BLOCK taps, then add the block sums. A single running
  // f32 sum drifts by ~1e-5 on wide kernels and the error compounds across
  // American steps. Kahan summation fixes that too, but Metal's fast-math
  // compiler removes the compensation; blocking survives any compiler.
  var s = 0.0;
  let interior = base >= 0 && base + i32(m) <= i32(p.n);
  for (var j0 = 0u; j0 < m; j0 += BLOCK) {
    let j1 = min(j0 + BLOCK, m);
    var b = 0.0;
    for (var j = j0; j < j1; j++) {
      var v: f32;
      if (interior) { v = f[u32(base) + j]; } else { v = at(base + i32(j)); }
      b += k[j] * v;
    }
    s += b;
  }
  if (p.mode == 1u) { s = max(s, payoff[i]); }
  out[i] = s;
}
`;

const WORKGROUP = 64;
let ctxPromise;

// One device and pipeline, created on first use.
export function context() {
  ctxPromise ??= (async () => {
    if (!("gpu" in navigator)) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter found");
    const device = await adapter.requestDevice();
    const pipelines = new Map();
    // Further shaders (barriers) share the device; each pipeline is built once.
    const pipelineFor = (code) => {
      if (!pipelines.has(code)) {
        const module = device.createShaderModule({ code });
        pipelines.set(code, device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } }));
      }
      return pipelines.get(code);
    };
    return { device, pipeline: pipelineFor(BLUR_WGSL), pipelineFor, adapter };
  })();
  return ctxPromise;
}

export async function gpuInfo() {
  const { adapter } = await context();
  const info = adapter.info ?? {};
  return [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown adapter";
}

export function storage(device, data, extra = 0) {
  const buf = device.createBuffer({
    size: Math.max(16, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

function params(device, n, half, mode) {
  const ab = new Uint32Array([n, half, mode, 0]);
  const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, ab);
  return buf;
}

function checkSizes(n, kLen) {
  if (kLen % 2 !== 1) throw new Error("kernel length must be odd");
  if ((kLen - 1) / 2 >= n) throw new Error("kernel wider than grid; widen the grid");
}

export async function readback(device, encoder, src, n) {
  const read = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  encoder.copyBufferToBuffer(src, 0, read, 0, n * 4);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();
  return result;
}

// Run `steps` blur passes, ping-ponging between two buffers in one submit.
async function run(values, kernel, steps, mode, payoffValues) {
  const n = values.length;
  checkSizes(n, kernel.length);
  const { device, pipeline } = await context();
  const half = (kernel.length - 1) / 2;
  const a = storage(device, Float32Array.from(values), GPUBufferUsage.COPY_SRC);
  const b = storage(device, new Float32Array(n), GPUBufferUsage.COPY_SRC);
  const k = storage(device, Float32Array.from(kernel));
  const pay = payoffValues ? storage(device, Float32Array.from(payoffValues)) : storage(device, new Float32Array(1));
  const u = params(device, n, half, mode);
  const layout = pipeline.getBindGroupLayout(0);
  const group = (src, dst) =>
    device.createBindGroup({
      layout,
      entries: [u, src, k, dst, pay].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  const ab = group(a, b);
  const ba = group(b, a);
  const encoder = device.createCommandEncoder();
  for (let s = 0; s < steps; s++) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, s % 2 === 0 ? ab : ba);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP));
    pass.end();
  }
  const result = await readback(device, encoder, steps % 2 === 1 ? b : a, n);
  for (const buf of [a, b, k, pay, u]) buf.destroy();
  return result;
}

// blur(values, kernel) from blur.js, on the GPU.
export function blurGPU(values, kernel) {
  return run(values, kernel, 1, 0);
}

// americanBlur's loop on the GPU: blur, discount, clip to payoff, `steps` times,
// all in one submit. payoffValues is the payoff sampled on the grid.
// The discount is folded into the kernel in 64-bit first: rounding disc to f32
// on its own biases every step the same way (~1.6e-5 after 200 steps).
export function americanGPU(payoffValues, kernel, steps, disc) {
  const kd = Float64Array.from(kernel, (w) => w * disc);
  return run(payoffValues, kd, steps, 1, payoffValues);
}

// ---------------------------------------------------------------------------
// Barriers on the GPU. The CPU twin of every function below is in barrier.js;
// keep the two in step.

// The hall of mirrors as one blur. Each thread prices one output node of one
// "option": it walks the kernel taps and builds the image-tiled payoff on the
// fly from the plain payoff samples P (see makeImage in barrier.js):
//   single barrier: live side as is, dead side = -(H/S)^a times the mirror image
//   double barrier: cells of width ln(U/L), alternately translated (+) and
//                   reflected (-) copies of the corridor, weighted by e^{a ...}
// plus the Euler-Maclaurin midpoint and +/- 1/24 corrections at every mirror line.
export const BARRIER_WGSL = /* wgsl */ `
struct Opt {
  pOff: u32, nP: u32, kOff: u32, half: u32,
  outOff: u32, nOut: u32, out0: i32, mode: u32,  // mode: 0 down, 1 up, 2 double, 3 none; +4 knock-in
  lo: i32, hi: i32, images: i32, a: f32,
  dx: f32, p0: f32, p1: f32, p2: f32,
}
struct Dispatch { maxOut: u32, nOpt: u32, p0: u32, p1: u32 }

const BLOCK = 32u;

@group(0) @binding(0) var<uniform> d: Dispatch;
@group(0) @binding(1) var<storage, read> opts: array<Opt>;
@group(0) @binding(2) var<storage, read> P: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<private> o: Opt;
var<private> fl: f32;   // payoff value and slope (in ln S) at the lower barrier, from inside
var<private> sl: f32;
var<private> fu: f32;   // ... and at the upper barrier
var<private> su: f32;
var<private> e1: vec2<f32>;  // single barrier: (node value, dipole)

// payoff sample j, continued as a straight line past the grid (odd reflection)
fn pv(j: i32) -> f32 {
  let n = i32(o.nP);
  if (j < 0) { return 2.0 * P[o.pOff] - P[o.pOff + u32(-j)]; }
  if (j >= n) { return 2.0 * P[o.pOff + u32(n - 1)] - P[o.pOff + u32(2 * (n - 1) - j)]; }
  return P[o.pOff + u32(j)];
}

// one-sided slope at node j, looking in direction dir
fn slope(j: i32, dir: i32) -> f32 {
  return (-3.0 * pv(j) + 4.0 * pv(j + dir) - pv(j + 2 * dir)) / (2.0 * f32(dir) * o.dx);
}

// a jump at a node: (midpoint + dx/12 * slope jump, value jump / 24); L, R = (value, slope)
fn edge(L: vec2<f32>, R: vec2<f32>) -> vec2<f32> {
  return vec2(0.5 * (L.x + R.x) + o.dx / 12.0 * (R.y - L.y), (R.x - L.x) / 24.0);
}

// value and slope at the left or right end of image cell c
fn cellEnd(c: i32, right: bool) -> vec2<f32> {
  if (abs(c) > o.images) { return vec2(0.0); }
  let m = o.hi - o.lo;
  let ad = o.a * o.dx;
  if ((c & 1) == 0) {
    let E = exp(ad * f32((-c / 2) * m));
    if (right) { return E * vec2(fu, su); }
    return E * vec2(fl, sl);
  }
  let n = (c + 1) / 2;
  if (!right) {
    let E = exp(ad * f32((1 - n) * m));
    return vec2(-E * fu, E * (o.a * fu + su));
  }
  let E = exp(-ad * f32(n * m));
  return vec2(-E * fl, E * (o.a * fl + sl));
}

fn edgeAt(c: i32) -> vec2<f32> {
  return edge(cellEnd(c - 1, true), cellEnd(c, false));
}

// the knock-out's image-tiled payoff at node j
fn gOut(j: i32) -> f32 {
  let mode = o.mode & 3u;
  let ad = o.a * o.dx;
  if (mode < 2u) {
    let b = o.lo;
    if (j == b) { return e1.x; }
    let dir = select(-1, 1, mode == 0u);
    var v = 0.0;
    if ((j - b) * dir > 0) {
      v = pv(j);
    } else if (o.images >= 1) {
      v = -exp(-ad * f32(j - b)) * pv(2 * b - j);
    }
    if (j == b + 1) { v += e1.y; }
    if (j == b - 1) { v -= e1.y; }
    return v;
  }
  let m = o.hi - o.lo;
  let dd = j - o.lo;
  var c = dd / m;
  if (dd % m != 0 && dd < 0) { c -= 1; }
  let r = dd - c * m;
  if (r == 0) { return edgeAt(c).x; }
  var v = 0.0;
  if (abs(c) <= o.images) {
    if ((c & 1) == 0) {
      let n = -c / 2;
      v = exp(ad * f32(n * m)) * pv(j + 2 * n * m);
    } else {
      let n = (c + 1) / 2;
      v = -exp(ad * f32(o.lo + n * m - j)) * pv(2 * o.lo + 2 * n * m - j);
    }
  }
  if (r == 1) { v += edgeAt(c).y; }
  if (r == m - 1) { v -= edgeAt(c + 1).y; }
  return v;
}

fn value(j: i32) -> f32 {
  if ((o.mode & 3u) == 3u) { return pv(j); }
  if ((o.mode & 4u) != 0u) { return pv(j) - gOut(j); }
  return gOut(j);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  let oi = t / d.maxOut;
  let q = t % d.maxOut;
  if (oi >= d.nOpt) { return; }
  o = opts[oi];
  if (q >= o.nOut) { return; }
  let mode = o.mode & 3u;
  if (mode < 2u) {
    let dir = select(-1, 1, mode == 0u);
    let fb = pv(o.lo);
    let live = vec2(fb, slope(o.lo, dir));
    var img = vec2(0.0);
    if (o.images >= 1) { img = vec2(-fb, o.a * fb + live.y); }
    if (dir > 0) { e1 = edge(img, live); } else { e1 = edge(live, img); }
  } else if (mode == 2u) {
    fl = pv(o.lo);
    sl = slope(o.lo, 1);
    fu = pv(o.hi);
    su = slope(o.hi, -1);
  }
  let base = o.out0 + i32(q) - i32(o.half);
  let m = 2u * o.half + 1u;
  var s = 0.0;
  for (var j0 = 0u; j0 < m; j0 += BLOCK) {
    let j1 = min(j0 + BLOCK, m);
    var b = 0.0;
    for (var j = j0; j < j1; j++) {
      b += K[o.kOff + j] * value(base + i32(j));
    }
    s += b;
  }
  out[o.outOff + q] = s;
}
`;

// Step by step: blur every row of a [rows x n] block (odd reflection per row).
export const ROWS_WGSL = /* wgsl */ `
struct Params { n: u32, rows: u32, half: u32, p0: u32 }
const BLOCK = 32u;
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
  for (var j0 = 0u; j0 < m; j0 += BLOCK) {
    let j1 = min(j0 + BLOCK, m);
    var b = 0.0;
    for (var j = j0; j < j1; j++) {
      var v: f32;
      if (interior) { v = f[row * p.n + u32(base) + j]; } else { v = at(row, base + i32(j)); }
      b += k[j] * v;
    }
    s += b;
  }
  out[row * p.n + i] = s;
}
`;

// Step by step: the monitoring date. Rows [vanilla, knock-out, knock-in]; past
// a barrier the knock-out takes 0 and the knock-in the vanilla value, with the
// jump at the barrier node treated like any other (midpoint + Euler-Maclaurin).
export const MONITOR_WGSL = /* wgsl */ `
struct Params { n: u32, lo: i32, hi: i32, dx: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;

fn A(r: u32, i: i32) -> f32 { return X[r * p.n + u32(i)]; }
fn B(r: u32, i: i32) -> f32 { if (r == 1u) { return 0.0; } return X[u32(i)]; }
fn D(r: u32, i: i32, sgn: f32) -> f32 { return sgn * (A(r, i) - B(r, i)); }

fn fix(r: u32, i: i32, b: i32, sgn: f32, v: f32) -> f32 {
  if (b < 0) { return v; }
  if (i == b) {
    return 0.5 * (A(r, b) + B(r, b)) + p.dx / 12.0 * (D(r, b + 1, sgn) - D(r, b - 1, sgn)) / (2.0 * p.dx);
  }
  if (i == b + 1) { return v + D(r, b, sgn) / 24.0; }
  if (i == b - 1) { return v - D(r, b, sgn) / 24.0; }
  return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 64u;
  let r = t / p.n;
  let i = i32(t % p.n);
  if (r >= 3u) { return; }
  if (r == 0u) { Y[t] = X[t]; return; }
  let dead = (p.lo >= 0 && i < p.lo) || (p.hi >= 0 && i > p.hi);
  var v = select(A(r, i), B(r, i), dead);
  v = fix(r, i, p.lo, 1.0, v);
  v = fix(r, i, p.hi, -1.0, v);
  Y[t] = v;
}
`;

export function groups(threads) {
  const wg = Math.ceil(threads / WORKGROUP);
  const x = Math.min(wg, 65535);
  return [x, Math.ceil(wg / x)];
}

export function uniform(device, words) {
  const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, words);
  return buf;
}

const OPT_WORDS = 16;
function packOpts(opts) {
  const ab = new ArrayBuffer(opts.length * OPT_WORDS * 4);
  const u = new Uint32Array(ab);
  const i = new Int32Array(ab);
  const f = new Float32Array(ab);
  opts.forEach((o, k) => {
    const b = k * OPT_WORDS;
    u[b] = o.pOff; u[b + 1] = o.nP; u[b + 2] = o.kOff; u[b + 3] = o.half;
    u[b + 4] = o.outOff; u[b + 5] = o.nOut; i[b + 6] = o.out0; u[b + 7] = o.mode;
    i[b + 8] = o.lo; i[b + 9] = o.hi ?? -1; i[b + 10] = o.images ?? 1 << 20; f[b + 11] = o.a ?? 0;
    f[b + 12] = o.dx;
  });
  return ab;
}

// Set up one barrier dispatch: { P, K, opts } as in barrierBlurCPU. Returns
// { encode(encoder), read(): Promise<Float32Array>, destroy() } so a benchmark
// can re-run the same dispatch without re-uploading.
export async function barrierJob({ P, K, opts }) {
  const { device, pipelineFor } = await context();
  const pipeline = pipelineFor(BARRIER_WGSL);
  const maxOut = Math.max(...opts.map((o) => o.nOut));
  const total = opts.reduce((s, o) => Math.max(s, o.outOff + o.nOut), 0);
  const bufs = {
    u: uniform(device, new Uint32Array([maxOut, opts.length, 0, 0])),
    opts: storage(device, packOpts(opts)),
    P: storage(device, Float32Array.from(P)),
    K: storage(device, Float32Array.from(K)),
    out: device.createBuffer({ size: Math.max(16, total * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
  };
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [bufs.u, bufs.opts, bufs.P, bufs.K, bufs.out].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const [gx, gy] = groups(opts.length * maxOut);
  const encode = (encoder) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
  };
  return {
    device,
    encode,
    read: () => {
      const enc = device.createCommandEncoder();
      encode(enc);
      return readback(device, enc, bufs.out, total);
    },
    destroy: () => Object.values(bufs).forEach((b) => b.destroy()),
  };
}

// barrierBlurCPU from barrier.js, on the GPU, in one dispatch.
export async function barrierGPU(spec) {
  const job = await barrierJob(spec);
  const out = await job.read();
  job.destroy();
  return out;
}

// stepBarrierCPU from barrier.js, on the GPU: all steps in one submit.
export async function stepBarrierGPU({ P, kernel, steps, lo, hi, dx, snaps = [] }) {
  const n = P.length;
  checkSizes(n, kernel.length);
  const { device, pipelineFor } = await context();
  const rowsPipe = pipelineFor(ROWS_WGSL);
  const monPipe = pipelineFor(MONITOR_WGSL);
  const init = new Float32Array(3 * n);
  init.set(Float32Array.from(P), 0);
  init.set(Float32Array.from(P), n);
  const X = storage(device, init, GPUBufferUsage.COPY_SRC);
  const Y = storage(device, new Float32Array(3 * n), GPUBufferUsage.COPY_SRC);
  const k = storage(device, Float32Array.from(kernel));
  const uRows = uniform(device, new Uint32Array([n, 3, (kernel.length - 1) / 2, 0]));
  const monWords = new ArrayBuffer(16);
  new Uint32Array(monWords)[0] = n;
  new Int32Array(monWords)[1] = lo;
  new Int32Array(monWords)[2] = hi;
  new Float32Array(monWords)[3] = dx;
  const uMon = uniform(device, monWords);
  const snapLen = snaps.reduce((s, sn) => s + 3 * sn.nOut, 0);
  const snapBuf = device.createBuffer({ size: Math.max(16, snapLen * 4), usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const bind = (pipe, entries) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  const blurYX = bind(rowsPipe, [uRows, Y, k, X]);
  const monXY = bind(monPipe, [uMon, X, Y]);
  const [gx, gy] = groups(3 * n);
  const encoder = device.createCommandEncoder();
  const run = (pipe, group) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
  };
  run(monPipe, monXY); // knock out the payoff itself at expiry
  let off = 0;
  for (let s = 1; s <= steps; s++) {
    run(rowsPipe, blurYX);
    for (const sn of snaps) {
      if (sn.step !== s) continue;
      for (let r = 0; r < 3; r++) {
        encoder.copyBufferToBuffer(X, (r * n + sn.out0) * 4, snapBuf, (off + r * sn.nOut) * 4, sn.nOut * 4);
      }
      sn.offset = off;
      off += 3 * sn.nOut;
    }
    if (s < steps) run(monPipe, monXY);
  }
  const rowsOut = await readback(device, encoder, X, 3 * n);
  let shots = [];
  if (snapLen) {
    const enc = device.createCommandEncoder();
    const all = await readback(device, enc, snapBuf, snapLen);
    shots = snaps.filter((sn) => sn.offset !== undefined).map((sn) => all.subarray(sn.offset, sn.offset + 3 * sn.nOut));
  }
  for (const b of [X, Y, k, uRows, uMon, snapBuf]) b.destroy();
  return { rows: rowsOut, snaps: shots };
}
