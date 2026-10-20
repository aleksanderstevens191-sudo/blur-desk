// Blur Desk on the GPU: the blur() from blur.js as a WebGPU compute shader.
// Same math: out[i] = sum_j k[j] * f(i + j - half), odd-reflection edges,
// kernel passed in as a buffer. WebGPU computes in 32-bit floats.

export const BLUR_WGSL = /* wgsl */ `
struct Params {
  n: u32,
  half: u32,
  mode: u32,   // 0 = plain blur, 1 = American step: max(blur, payoff)
}

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
  // Kahan summation: plain f32 sums drift by ~1e-5 on wide kernels and
  // the error compounds across American steps.
  var s = 0.0;
  var c = 0.0;
  let interior = base >= 0 && base + i32(m) <= i32(p.n);
  for (var j = 0u; j < m; j++) {
    var v: f32;
    if (interior) { v = f[u32(base) + j]; } else { v = at(base + i32(j)); }
    let y = k[j] * v - c;
    let t = s + y;
    c = (t - s) - y;
    s = t;
  }
  if (p.mode == 1u) { s = max(s, payoff[i]); }
  out[i] = s;
}
`;

const WORKGROUP = 64;
let ctxPromise;

// One device and pipeline, created on first use.
function context() {
  ctxPromise ??= (async () => {
    if (!("gpu" in navigator)) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter found");
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: BLUR_WGSL });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    return { device, pipeline, adapter };
  })();
  return ctxPromise;
}

export async function gpuInfo() {
  const { adapter } = await context();
  const info = adapter.info ?? {};
  return [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown adapter";
}

function storage(device, data, extra = 0) {
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

async function readback(device, encoder, src, n) {
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
