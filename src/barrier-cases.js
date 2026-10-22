// Turns a case from barrier_fixtures.json into inputs for the barrier blur
// (CPU twin or GPU), so node's test and the browser check run the same thing.
import { MODE, gridZ, sampleVanilla, interpCubic, discountedKernel, mirrorExponent, parseKind } from "./barrier.js";

// Mirror and image-series cases: { P, K, opts } for barrierBlurCPU / barrierGPU,
// with four output nodes around S0 for the cubic read-out.
export function blurSpec(c, S0) {
  const g = c.grid;
  const z = gridZ(g);
  const x0 = Math.log(S0);
  const jx = Math.floor((x0 - z[0]) / g.dx);
  const cp = c.method === "double" ? c.kind.split(" ")[1] : parseKind(c.kind).cp;
  const P = sampleVanilla(z, c.K, cp);
  const K = discountedKernel(g.dx, c.sigma, c.T, c.r);
  const b = -g.lo; // the grid is anchored at the (lower) barrier
  let mode;
  let hi = -1;
  let images = 1;
  if (c.method === "double") {
    mode = MODE.double;
    hi = b + c.m;
    images = c.images ?? undefined;
  } else {
    const { ud, io } = parseKind(c.kind);
    mode = (ud === "down" ? MODE.down : MODE.up) + (io === "in" ? MODE.IN : 0);
  }
  const opt = {
    pOff: 0, nP: g.n, dx: g.dx, kOff: 0, half: (K.length - 1) / 2, outOff: 0, nOut: 4, out0: jx - 1,
    mode, lo: b, hi, images, a: c.method === "plain" ? 0 : mirrorExponent(c.sigma, c.r),
  };
  return { P, K, opts: [opt] };
}

// Price at S0 from the four output nodes of blurSpec.
export function blurValue(c, out, S0) {
  const g = c.grid;
  const x0 = Math.log(S0);
  const z0 = g.anchor + g.lo * g.dx;
  const jx = Math.floor((x0 - z0) / g.dx);
  return interpCubic(z0 + (jx - 1) * g.dx, g.dx, out, x0, 0);
}

// Step-by-step cases: inputs for stepBarrierCPU / stepBarrierGPU.
export function stepSpec(c) {
  const [ud, cp] = c.kind.split(" ");
  const z = gridZ(c.grid);
  return {
    P: sampleVanilla(z, c.K, cp),
    kernel: discountedKernel(c.grid.dx, c.sigma, c.T / c.steps, c.r),
    steps: c.steps,
    lo: ud === "down" ? c.ib : -1,
    hi: ud === "up" ? c.ib : -1,
    dx: c.grid.dx,
    z0: z[0],
  };
}

// { out, in, vanilla } at S0 from the step-by-step rows.
export function stepValues(spec, rows, S0) {
  const n = spec.P.length;
  const v = (r) => interpCubic(spec.z0, spec.dx, rows, Math.log(S0), r * n);
  return { vanilla: v(0), out: v(1), in: v(2) };
}
