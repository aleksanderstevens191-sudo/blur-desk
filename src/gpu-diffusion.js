// Panel 16 on the GPU (WebGPU / WGSL, 32-bit). The 64-bit twin is diffusion.js; the
// Python twin is the end of reference/diffusion.py.
//   TABLES   one thread per (grid node, noise level): the market's density blurred to that
//            level and its score, (blur of Dp0) / (blur of p0). Every level in one dispatch.
//            Weights come from a table normalised in 64-bit on the CPU, so the shader only
//            multiplies and adds (blocked summation, blocks of 32, as in the other shaders).
//   PAYOFF   one thread per spot: the call blurred by the same Gaussian, taps s/256 apart,
//            the payoff evaluated at each tap through expf (gpu-lv.js), no grid edge to fall off
//   SAMPLER  one thread per point: the probability-flow ODE from sigma_max back to
//            sigma_min, Heun's method in ln sigma, the score read off the tables; every
//            KEEP-th level is written out for the animation
import { context, storage, readback, groups } from "./gpu.js";
import { EXPF } from "./gpu-lv.js";
import { PANEL, halfTaps } from "./diffusion.js";

export const KEEP = 4;
const BLOCK = 32;

const TABLES_WGSL = /* wgsl */ `
struct Params { n: u32, L: u32, pad0: u32, pad1: u32 }
const BLOCK = ${BLOCK};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> LV: array<vec2<u32>>;   // per level: offset into W, half-width H
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> P0: array<f32>;
@group(0) @binding(4) var<storage, read> DP: array<f32>;
@group(0) @binding(5) var<storage, read_write> Q: array<f32>;
@group(0) @binding(6) var<storage, read_write> SC: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= p.n || j >= p.L) { return; }
  let off = LV[j].x;
  let H = i32(LV[j].y);
  let n = i32(p.n);
  var a = 0.0;
  var b = 0.0;
  var ba = 0.0;
  var bb = 0.0;
  for (var k0 = -H; k0 <= H; k0 += BLOCK) {
    let k1 = min(k0 + BLOCK, H + 1);
    ba = 0.0;
    bb = 0.0;
    for (var k = max(k0, i32(i) - n + 1); k < min(k1, i32(i) + 1); k++) {
      let w = W[off + u32(k + H)];
      let m = u32(i32(i) - k);
      ba += w * P0[m];
      bb += w * DP[m];
    }
    a += ba;
    b += bb;
  }
  Q[j * p.n + i] = a;
  SC[j * p.n + i] = select(0.0, b / a, a > 1e-30);
}
`;

const PAYOFF_WGSL = /* wgsl */ `
${EXPF}
struct Params { nx: u32, H: u32, pad0: u32, pad1: u32, s: f32, K: f32, step: f32, shift: f32 }
const BLOCK = ${BLOCK};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.nx) { return; }
  let H = i32(p.H);
  let x = X[i] + p.shift;
  var acc = 0.0;
  var blk = 0.0;
  for (var k0 = -H; k0 <= H; k0 += BLOCK) {
    let k1 = min(k0 + BLOCK, H + 1);
    blk = 0.0;
    for (var k = k0; k < k1; k++) {
      let f = max(100.0 * expf(x + f32(k) * p.step) - p.K, 0.0);
      blk += W[u32(k + H)] * f;
    }
    acc += blk;
  }
  OUT[i] = acc;
}
`;

const SAMPLER_WGSL = /* wgsl */ `
struct Params { n: u32, N: u32, np: u32, keep: u32, dx: f32, y0: f32, h: f32, pad: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> SIG2: array<f32>;
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> X1: array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<f32>;   // level j / keep, then the point

fn score(j: u32, x: f32) -> f32 {
  let u = (x - p.y0) / p.dx;
  let i0 = u32(clamp(floor(u), 0.0, f32(p.n - 2u)));
  let f = clamp(u - f32(i0), 0.0, 1.0);
  let a = SC[j * p.n + i0];
  return a + f * (SC[j * p.n + i0 + 1u] - a);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 64u;
  if (i >= p.np) { return; }
  var x = X1[i];
  OUT[(p.N / p.keep) * p.np + i] = x;
  for (var j = p.N; j > 0u; j--) {
    let k1 = -SIG2[j] * score(j, x);
    let k2 = -SIG2[j - 1u] * score(j - 1u, x + p.h * k1);
    x = x + 0.5 * p.h * (k1 + k2);
    if ((j - 1u) % p.keep == 0u) { OUT[((j - 1u) / p.keep) * p.np + i] = x; }
  }
}
`;

function uniformBuf(device, ab) {
  const buf = device.createBuffer({ size: ab.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, ab);
  return buf;
}

function bind(device, pipeline, bufs) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

// Normalised Gaussian taps for each level, in 64-bit, concatenated
function levelWeights(s, C) {
  const lv = new Uint32Array(2 * s.length);
  const rows = [];
  let off = 0;
  s.forEach((sj, j) => {
    const H = sj === 0 ? 0 : halfTaps(sj, C);
    const w = sj === 0 ? [1] : Array.from({ length: 2 * H + 1 }, (_, k) => Math.exp(-(((k - H) * C.dx) ** 2) / (2 * sj * sj)));
    const den = w.reduce((a, b) => a + b, 0);
    rows.push(w.map((v) => v / den));
    lv[2 * j] = off;
    lv[2 * j + 1] = H;
    off += w.length;
  });
  return { lv, W: Float32Array.from(rows.flat()) };
}

// The score tables (and blurred densities) at every level. Keeps the score on the GPU for
// the sampler. Returns { Q: Float32Array (L x n), ms, sampler(x1) -> runs, destroy() }.
export async function diffusionGPU(G, lev, C = PANEL) {
  const { device, pipelineFor } = await context();
  const L = lev.s.length, n = G.n;
  const t0 = performance.now();
  const { lv, W } = levelWeights(lev.s, C);
  const lvBuf = storage(device, lv), wBuf = storage(device, W);
  const p0 = storage(device, Float32Array.from(G.p0)), dp = storage(device, Float32Array.from(G.dp));
  const qBuf = storage(device, new Float32Array(L * n), GPUBufferUsage.COPY_SRC);
  const scBuf = storage(device, new Float32Array(L * n), GPUBufferUsage.COPY_SRC);
  const u = uniformBuf(device, new Uint32Array([n, L, 0, 0]));
  const tables = pipelineFor(TABLES_WGSL);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(tables);
  pass.setBindGroup(0, bind(device, tables, [u, lvBuf, wBuf, p0, dp, qBuf, scBuf]));
  pass.dispatchWorkgroups(Math.ceil(n / 64), L);
  pass.end();
  const Q = await readback(device, encoder, qBuf, L * n);
  const ms = performance.now() - t0;
  for (const b of [lvBuf, wBuf, p0, dp, qBuf, u]) b.destroy();
  const sig2 = storage(device, Float32Array.from(lev.sig, (v) => v * v));

  // the reverse run from x1 (Float32Array); returns { levels: [L / KEEP + 1 arrays], ms }
  async function sampler(x1) {
    const t1 = performance.now();
    const np = x1.length, N = L - 1, snaps = N / KEEP + 1;
    const xBuf = storage(device, x1);
    const out = storage(device, new Float32Array(snaps * np), GPUBufferUsage.COPY_SRC);
    const ab = new ArrayBuffer(32);
    new Uint32Array(ab, 0, 4).set([n, N, np, KEEP]);
    new Float32Array(ab, 16, 4).set([C.dx, -C.half * C.dx, (Math.log(lev.sig[0]) - Math.log(lev.sig[N])) / N, 0]);
    const pu = uniformBuf(device, ab);
    const pipe = pipelineFor(SAMPLER_WGSL);
    const enc = device.createCommandEncoder();
    const ps = enc.beginComputePass();
    ps.setPipeline(pipe);
    ps.setBindGroup(0, bind(device, pipe, [pu, sig2, scBuf, xBuf, out]));
    ps.dispatchWorkgroups(...groups(np));
    ps.end();
    const all = await readback(device, enc, out, snaps * np);
    for (const b of [xBuf, out, pu]) b.destroy();
    const levels = Array.from({ length: snaps }, (_, k) => all.subarray(k * np, (k + 1) * np));
    return { levels, ms: performance.now() - t1 };
  }
  return {
    Q, ms, sampler,
    destroy() {
      scBuf.destroy();
      sig2.destroy();
    },
  };
}

// The call (r = 0, per $100 of forward) blurred by sd s at log-forward offsets x
const payW = new Map();
export async function payoffGPU(x, s, K, C = PANEL) {
  if (s === 0) return { prices: Float64Array.from(x, (v) => Math.max(100 * Math.exp(v) - K, 0)), ms: 0 };
  const { device, pipelineFor } = await context();
  const t0 = performance.now();
  const H = C.nsd * C.per;
  if (!payW.has(H)) {
    const w = Array.from({ length: 2 * H + 1 }, (_, k) => Math.exp(-(((k - H) / C.per) ** 2) / 2));
    const den = w.reduce((a, b) => a + b, 0);
    payW.set(H, Float32Array.from(w, (v) => v / den));
  }
  const wBuf = storage(device, payW.get(H));
  const xBuf = storage(device, Float32Array.from(x));
  const out = storage(device, new Float32Array(x.length), GPUBufferUsage.COPY_SRC);
  const ab = new ArrayBuffer(32);
  new Uint32Array(ab, 0, 4).set([x.length, H, 0, 0]);
  new Float32Array(ab, 16, 4).set([s, K, s / C.per, -0.5 * s * s]);
  const u = uniformBuf(device, ab);
  const pipe = pipelineFor(PAYOFF_WGSL);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bind(device, pipe, [u, wBuf, xBuf, out]));
  pass.dispatchWorkgroups(Math.ceil(x.length / 64));
  pass.end();
  const prices = await readback(device, enc, out, x.length);
  for (const b of [wBuf, xBuf, out, u]) b.destroy();
  return { prices, ms: performance.now() - t0 };
}
