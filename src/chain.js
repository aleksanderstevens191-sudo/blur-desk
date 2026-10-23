// The whole SPY chain in one GPU submit (stage 2 of the full-chain work).
//
// Everything is priced on the forward, normalised to 1: a European option on
// expiry e is D F c(k) with k = ln(K / F) and c the price of an option struck at
// e^k on an underlying whose forward is 1. Only the out-of-the-money side goes
// through the GPU (time value, never more than the at-the-money price); the
// in-the-money side follows by parity in 64-bit, C - P = D (F - K).
//
// Merton, fitted per expiry: the smile panel's strike rows. One kernel per
// expiry, one call row and one put row struck at the forward,
//   out(z) = E[(e^(z + x) - 1)+],   c(k) = e^k out(-k),
// and every strike is read off the rows: out at -k by 8-point Lagrange
// interpolation between nodes (the strikes are $1 or $5 apart, not on the grid).
// The rows are evaluated only at the nodes those stencils need.
//
// Black-Scholes at each contract's own implied vol: the kernel width differs
// from contract to contract, so each gets its own Gaussian kernel and its own
// row, on a grid spaced so that its strike is a node (the payoff is then exactly
// piecewise linear in e^z and the blur is exact up to the tails), read at one node.
//
// Both are the same operation, a dot product of kernel taps with payoff values,
// so the GPU runs one shader over a list of jobs (kernel offset, row offset,
// length): one workgroup per job, 64 threads, sums in blocks of 32 then a tree.
import { gridFor, kernel } from "./kernels.js";
import { context, storage } from "./gpu.js";

export const MERTON_GRID = { ptsPerSd: 40, nSd: 10, tail: 1e-15 };
export const BS_GRID = { ptsPerSd: 10, nSd: 9, tail: 1e-15 };
const STENCIL = 8; // Lagrange nodes around each strike: i0 - 3 .. i0 + 4

const payoffCall = (z, k) => Math.max(Math.exp(z) - Math.exp(k), 0);
const payoffPut = (z, k) => Math.max(Math.exp(k) - Math.exp(z), 0);

// Build every job of the chain in 64-bit. data: src/chain_data.json. Returns
// { W, Fv, jobs, nJobs, finish(out) -> { bs, merton } (dollar prices per contract), info }.
export function chainJobs(data, { models = ["bs", "merton"] } = {}) {
  const { expiries } = data;
  const C = data.contracts;
  const n = C.K.length;
  const Wl = [];
  const Fl = [];
  let wLen = 0;
  let fLen = 0;
  const jobs = [];
  const pushW = (a) => { Wl.push(a); wLen += a.length; return wLen - a.length; };
  const pushF = (a) => { Fl.push(a); fLen += a.length; return fLen - a.length; };
  const k = Float64Array.from(C.K, (K, i) => Math.log(K / expiries[C.e[i]].F));
  const otmCall = Uint8Array.from(C.K, (K, i) => (K >= expiries[C.e[i]].F ? 1 : 0));
  const info = { mertonTaps: 0, mertonNodes: 0, bsTaps: 0, perExpiry: [] };

  // Black-Scholes: one job per contract, the job's output is c(k) itself
  const bsJob = new Int32Array(n).fill(-1);
  if (models.includes("bs")) {
    for (let i = 0; i < n; i++) {
      const { T } = expiries[C.e[i]];
      const p = { T, r: 0, sigma: C.iv[i] };
      const dx0 = (C.iv[i] * Math.sqrt(T)) / BS_GRID.ptsPerSd;
      const dx = Math.abs(k[i]) < 1e-12 ? dx0 : Math.abs(k[i]) / Math.ceil(Math.abs(k[i]) / dx0);
      const { y } = gridFor("gaussian", p, { dx, nSd: BS_GRID.nSd, tail: BS_GRID.tail });
      const w = kernel("gaussian", y, p);
      const pay = otmCall[i] ? payoffCall : payoffPut;
      const f = Float64Array.from(y, (z) => pay(z, k[i]));
      bsJob[i] = jobs.length;
      jobs.push([pushW(w), pushF(f), w.length]);
      info.bsTaps += w.length;
    }
  }

  // Merton: per expiry one kernel, two rows struck at the forward, the stencil nodes as jobs
  const mStencil = []; // per contract: [first job of its 8 nodes is not contiguous, so keep a list]
  if (models.includes("merton")) {
    expiries.forEach((ex, e) => {
      const { T } = ex;
      const p = { T, r: 0, ...ex.merton };
      const { y, dx, J } = gridFor("merton", p, MERTON_GRID);
      const w = kernel("merton", y, p);
      const wOff = pushW(w);
      const mine = [];
      for (let i = 0; i < n; i++) if (C.e[i] === e) mine.push(i);
      for (const side of [1, 0]) {
        const cs = mine.filter((i) => otmCall[i] === side);
        if (!cs.length) continue;
        // node u = -k / dx; stencil i0 - 3 .. i0 + 4 with i0 = floor(u)
        const lo0 = Math.min(...cs.map((i) => Math.floor(-k[i] / dx))) - (STENCIL / 2 - 1);
        const hi0 = Math.max(...cs.map((i) => Math.floor(-k[i] / dx))) + STENCIL / 2;
        const start = lo0 - J; // row covers nodes start .. hi0 + J
        const len = hi0 + J - start + 1;
        const pay = side ? payoffCall : payoffPut;
        const f = Float64Array.from({ length: len }, (_, j) => pay((start + j) * dx, 0));
        const fOff = pushF(f);
        const node = new Map(); // node index -> job
        for (const i of cs) {
          const i0 = Math.floor(-k[i] / dx);
          const st = [];
          for (let s = i0 - (STENCIL / 2 - 1); s <= i0 + STENCIL / 2; s++) {
            if (!node.has(s)) {
              node.set(s, jobs.length);
              jobs.push([wOff, fOff + (s - J - start), w.length]);
            }
            st.push(node.get(s));
          }
          mStencil[i] = { i0, dx, jobs: st };
        }
        info.mertonNodes += node.size;
        info.mertonTaps += node.size * w.length;
      }
      info.perExpiry.push({ expiry: ex.expiry, taps: w.length, dx });
    });
  }

  const W = new Float64Array(wLen);
  let o = 0;
  for (const a of Wl) { W.set(a, o); o += a.length; }
  const Fv = new Float64Array(fLen);
  o = 0;
  for (const a of Fl) { Fv.set(a, o); o += a.length; }
  const J32 = new Uint32Array(jobs.length * 4);
  jobs.forEach(([a, b, c], j) => J32.set([a, b, c, 0], 4 * j));

  // job outputs -> dollar prices of every contract (64-bit)
  const finish = (out) => {
    const bs = new Float64Array(n).fill(NaN);
    const merton = new Float64Array(n).fill(NaN);
    for (let i = 0; i < n; i++) {
      const { D, F } = expiries[C.e[i]];
      const toDollars = (c) => {
        const otm = D * F * c; // price of the out-of-the-money option at this strike
        if (otmCall[i] === C.call[i]) return otm;
        return C.call[i] ? otm + D * (F - C.K[i]) : otm - D * (F - C.K[i]);
      };
      if (bsJob[i] >= 0) bs[i] = toDollars(out[bsJob[i]]);
      const st = mStencil[i];
      if (st) {
        const u = -k[i] / st.dx;
        const base = st.i0 - (STENCIL / 2 - 1);
        let v = 0;
        for (let a = 0; a < STENCIL; a++) {
          let L = 1;
          for (let b = 0; b < STENCIL; b++) if (b !== a) L *= (u - (base + b)) / (a - b);
          v += L * out[st.jobs[a]];
        }
        merton[i] = toDollars(Math.exp(k[i]) * v);
      }
    }
    return { bs, merton };
  };
  return { W, Fv, jobs: J32, nJobs: jobs.length, finish, info };
}

// The same jobs in 64-bit on the CPU: the reference twin of the shader.
export function runJobsCPU({ W, Fv, jobs, nJobs }) {
  const out = new Float64Array(nJobs);
  for (let j = 0; j < nJobs; j++) {
    const a = jobs[4 * j];
    const b = jobs[4 * j + 1];
    const len = jobs[4 * j + 2];
    let s = 0;
    for (let t = 0; t < len; t++) s += W[a + t] * Fv[b + t];
    out[j] = s;
  }
  return out;
}

export const CHAIN_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> Fv: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BLOCK = 32u;
var<workgroup> part: array<f32, 64>;

// One workgroup per job: out = sum_t W[w + t] * Fv[f + t], t < len.
// Thread l takes taps l, l + 64, ...; its running sum is flushed every 32 taps
// (Metal's fast math removes Kahan compensation, blocking survives), then the
// 64 partial sums are added as a tree.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) l: u32) {
  let id = wg.x + wg.y * nwg.x;
  let live = id < arrayLength(&jobs);
  var s = 0.0;
  if (live) {
    let jb = jobs[id];
    var b = 0.0;
    var c = 0u;
    for (var t = l; t < jb.z; t += 64u) {
      b += W[jb.x + t] * Fv[jb.y + t];
      c++;
      if (c == BLOCK) { s += b; b = 0.0; c = 0u; }
    }
    s += b;
  }
  part[l] = s;
  workgroupBarrier();
  for (var h = 32u; h > 0u; h >>= 1u) {
    if (l < h) { part[l] += part[l + h]; }
    workgroupBarrier();
  }
  if (live && l == 0u) { out[id] = part[0]; }
}
`;

// Upload the jobs once; returns { encode(encoder), read(), destroy(), bytes } so a
// benchmark can re-run the same submit. All jobs go out in one dispatch.
export async function chainJobGPU({ W, Fv, jobs, nJobs }) {
  const { device, pipelineFor } = await context();
  const pipe = pipelineFor(CHAIN_WGSL);
  const bufs = {
    W: storage(device, Float32Array.from(W)),
    Fv: storage(device, Float32Array.from(Fv)),
    jobs: storage(device, jobs),
    out: device.createBuffer({ size: Math.max(16, nJobs * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
  };
  const group = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [bufs.W, bufs.Fv, bufs.jobs, bufs.out].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const gx = Math.min(nJobs, 65535);
  const gy = Math.ceil(nJobs / gx);
  const encode = (encoder) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
  };
  const read = async () => {
    const encoder = device.createCommandEncoder();
    encode(encoder);
    const rd = device.createBuffer({ size: Math.max(16, nJobs * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(bufs.out, 0, rd, 0, nJobs * 4);
    device.queue.submit([encoder.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(rd.getMappedRange().slice(0));
    rd.unmap();
    rd.destroy();
    return res;
  };
  const bytes = (W.length + Fv.length + nJobs * 5) * 4;
  const destroy = () => Object.values(bufs).forEach((b) => b.destroy());
  return { device, encode, read, destroy, bytes };
}

// Price the chain on the GPU: build (CPU, 64-bit), upload, one submit, read back, finish.
export async function priceChainGPU(data, opts = {}) {
  const t0 = performance.now();
  const J = chainJobs(data, opts);
  const t1 = performance.now();
  const job = await chainJobGPU(J);
  const out = await job.read();
  const t2 = performance.now();
  const res = J.finish(out);
  const t3 = performance.now();
  job.destroy();
  return { ...res, J, ms: { build: t1 - t0, gpu: t2 - t1, finish: t3 - t2 } };
}
