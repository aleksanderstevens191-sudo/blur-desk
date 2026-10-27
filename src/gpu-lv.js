// The local-vol blur on the GPU (WebGPU / WGSL, 32-bit): one step back in time per
// dispatch, every step in one submit. Each thread owns one grid node and walks its own
// kernel: the width comes from the local vol at that node and the step's time (two
// table rows read at the grid's nodes, blended in time), so neighbouring threads blur
// by different amounts. Taps out to 8 standard deviations, half-width rounded up to a
// power of two, weights normalised in the shader; blocked summation (blocks of 32) as in
// the other shaders, which Metal's compiler can't undo. The weight
// exp(-(y - mu)^2 / 2s^2) is computed as exp(-y^2 / 2s^2) exp(-y/2) (the same up to a
// constant, as mu = -s^2/2): rounding in the first factor is then the same for +y and
// -y, so it can't shift the kernel's mean, which in the plain form added up over 957
// steps to 1.2e-4 per $100 on a 1-year at-the-money call. Up to MAXC strikes share each
// node's weights. The 64-bit twin is stepCPU() in lv.js; the job comes from lvJob().
// An in-the-money strike is priced as its out-of-the-money twin plus put-call parity
// (call - put = 1 - e^k per unit of forward): its value is mostly intrinsic, and carrying
// that through hundreds of 32-bit steps cost up to 2e-4 per $100 on a 1-year flat vol.
// Each node's new value is its old value plus the weighted average of its neighbours'
// differences from it, not the weighted average itself. With a flat vol every node and
// step has the same weights, so the rounding of their sum is the same factor every step;
// dividing whole prices by it shrank them 956 times over (3e-4 per $100 on a 1-year
// at-the-money put, on the Mac and in a 32-bit emulation). Dividing only the change
// leaves 8e-6.
import { context, storage, readback, groups } from "./gpu.js";
import { lastStep } from "./lv.js";

export const MAXC = 16;
export const STRIKES = MAXC;

// exp with Cody-Waite reduction through fma, which fast math can't reassociate, and a
// degree-7 polynomial: about 1 ulp at every argument. Metal's fast-math exp may be off by
// up to 3 + |2x| ulp (about 8e-6 at the 8-sd tap, x = -32), which bends the kernel.
const EXPF = /* wgsl */ `
fn expf(x: f32) -> f32 {
  let k = round(x * 1.4426950409);
  var r = fma(-k, 0.693145751953125, x);
  r = fma(-k, 1.428606765330187e-6, r);
  let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.041666668 + r * (0.008333334 + r * (0.0013888889 + r * 0.0001984127))))));
  return ldexp(p, i32(k));
}
`;

export const LV_STEP_WGSL = /* wgsl */ `
${EXPF}
struct Params { n: u32, C: u32, ra: u32, rb: u32, a: f32, h: f32, dx: f32, x0: f32 }
const BLOCK = 32;
const MAXC = ${MAXC}u;
const TAPS_SD = 8.0;
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> SN: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<private> A: array<f32, ${MAXC}>;   // A e^x + B past the low edge (B folded in: v0 - A e^x0)
var<private> Bv: array<f32, ${MAXC}>;
var<private> A1: array<f32, ${MAXC}>;  // and past the high edge
var<private> B1: array<f32, ${MAXC}>;

fn val(c: u32, j: i32) -> f32 {
  let n = i32(p.n);
  if (j >= 0 && j < n) { return V[c * p.n + u32(j)]; }
  let ex = exp(p.x0 + f32(j) * p.dx);
  if (j < 0) { return Bv[c] + A[c] * ex; }
  return B1[c] + A1[c] * ex;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 64u;
  if (i >= p.n) { return; }
  let n = p.n;
  let e0 = exp(p.x0);
  let e1 = exp(p.x0 + p.dx);
  let e2 = exp(p.x0 + f32(n - 2u) * p.dx);
  let e3 = exp(p.x0 + f32(n - 1u) * p.dx);
  for (var c = 0u; c < p.C; c++) {
    let v0 = V[c * n];
    let v1 = V[c * n + 1u];
    A[c] = (v1 - v0) / (e1 - e0);
    Bv[c] = v0 - A[c] * e0;
    let w0 = V[c * n + n - 2u];
    let w1 = V[c * n + n - 1u];
    A1[c] = (w1 - w0) / (e3 - e2);
    B1[c] = w0 - A1[c] * e2;
  }
  let sig = (1.0 - p.a) * SN[p.ra * n + i] + p.a * SN[p.rb * n + i];
  let s = sig * sqrt(p.h);
  let need = u32(ceil((TAPS_SD * s + 0.5 * s * s) / p.dx)) + 1u;
  var H = 2u;
  while (H < need) { H = H * 2u; }
  let inv = 1.0 / (2.0 * s * s);
  var acc = array<f32, ${MAXC}>();
  var blk = array<f32, ${MAXC}>();
  var vi = array<f32, ${MAXC}>();
  for (var c = 0u; c < p.C; c++) { vi[c] = V[c * n + i]; }
  var den = 0.0;
  let Hi = i32(H);
  for (var j0 = -Hi; j0 <= Hi; j0 += BLOCK) {
    let j1 = min(j0 + BLOCK, Hi + 1);
    for (var c = 0u; c < p.C; c++) { blk[c] = 0.0; }   // zeroed by hand: declared in the loop, SwiftShader kept it from the last block
    var bden = 0.0;
    for (var j = j0; j < j1; j++) {
      let y = f32(j) * p.dx;
      let w = expf(-y * y * inv) * expf(-0.5 * y);
      bden += w;
      for (var c = 0u; c < p.C; c++) { blk[c] += w * (val(c, i32(i) + j) - vi[c]); }
    }
    den += bden;
    for (var c = 0u; c < p.C; c++) { acc[c] += blk[c]; }
  }
  for (var c = 0u; c < p.C; c++) { out[c * n + i] = vi[c] + acc[c] / den; }
}
`;

function params(device, n, C, b, h, dx, x0) {
  const ab = new ArrayBuffer(32);
  new Uint32Array(ab, 0, 4).set([n, C, b.ra, b.rb]);
  new Float32Array(ab, 16, 4).set([b.a, h, dx, x0]);
  const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, ab);
  return buf;
}

// Price the job's strikes (lvJob in lv.js) on the GPU; returns { prices, ms }:
// prices per $100 of forward, discounted, one per strike.
export async function lvGPU(job) {
  const { device, pipelineFor } = await context();
  const pipeline = pipelineFor(LV_STEP_WGSL);
  const layout = pipeline.getBindGroupLayout(0);
  const n = job.x.length;
  const SN = new Float32Array(job.SN.length * n);
  job.SN.forEach((row, r) => SN.set(row, r * n));
  const snBuf = storage(device, SN);
  const prices = [];
  const t0 = performance.now();
  const last = job.steps.length - 1;
  const itm = job.strikes.map(({ k, call }) => (call ? k < 0 : k > 0));
  const sl = itm.some(Boolean) ? job.sigAt(last) : null;
  const V0s = job.V0.map((V, c) => (itm[c] ? lastStep(job.x, sl, job.steps[last].b - job.steps[last].a, job.strikes[c].k, !job.strikes[c].call) : V));
  for (let c0 = 0; c0 < V0s.length; c0 += STRIKES) {
    const cols = V0s.slice(c0, c0 + STRIKES);
    const C = cols.length;
    const V0 = new Float32Array(C * n);
    cols.forEach((V, c) => V0.set(V, c * n));
    const a = storage(device, V0, GPUBufferUsage.COPY_SRC);
    const b = storage(device, new Float32Array(C * n), GPUBufferUsage.COPY_SRC);
    const encoder = device.createCommandEncoder();
    const uniforms = [];
    let src = a, dst = b;
    const [gx, gy] = groups(n);
    for (let s = job.steps.length - 2; s >= 0; s--) {
      const { a: ta, b: tb } = job.steps[s];
      const u = params(device, n, C, job.blends[s], tb - ta, job.dx, job.x[0]);
      uniforms.push(u);
      const group = device.createBindGroup({
        layout,
        entries: [u, snBuf, src, dst].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(gx, gy);
      pass.end();
      [src, dst] = [dst, src];
    }
    const V = await readback(device, encoder, src, C * n);
    for (let c = 0; c < C; c++) {
      const { k, call } = job.strikes[c0 + c];
      const par = itm[c0 + c] ? (call ? 1 : -1) * (1 - Math.exp(k)) : 0;
      prices.push(job.scale * (V[c * n + job.center] + par));
    }
    for (const buf of [a, b, ...uniforms]) buf.destroy();
  }
  snBuf.destroy();
  return { prices, ms: performance.now() - t0 };
}

// For /check.html: the shader's exp (built in, and expf above) against 64-bit exp on
// [-34, 0.2], the range the weights use. Returns the worst relative errors.
const EXP_TEST_WGSL = /* wgsl */ `
${EXPF}
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&X)) { return; }
  Y[2u * i] = exp(X[i]);
  Y[2u * i + 1u] = expf(X[i]);
}
`;

export async function expErrors(m = 4096) {
  const { device, pipelineFor } = await context();
  const pipeline = pipelineFor(EXP_TEST_WGSL);
  const X = Float32Array.from({ length: m }, (_, i) => -34 + (34.2 * i) / (m - 1));
  const xb = storage(device, X);
  const yb = storage(device, new Float32Array(2 * m), GPUBufferUsage.COPY_SRC);
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [xb, yb].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(m / 64));
  pass.end();
  const Y = await readback(device, encoder, yb, 2 * m);
  xb.destroy();
  yb.destroy();
  let builtin = 0, ours = 0;
  X.forEach((x, i) => {
    const t = Math.exp(x);
    builtin = Math.max(builtin, Math.abs(Y[2 * i] / t - 1));
    ours = Math.max(ours, Math.abs(Y[2 * i + 1] / t - 1));
  });
  return { builtin, ours };
}
