// node src/test-2d.mjs — the two-stock blur's JS core (blur2d.js) and the 64-bit
// twin of the GPU's FFT plan (fft2d.js) against the Python fixtures
// (reference/make_2d_fixtures.py), and the 2D wraparound test with its proof
// that it bites.
import { readFileSync } from "node:fs";
import {
  job2d, priceAt, margrabe, stulzCallMax, stulzCallMin, bs, fixturePayoff,
} from "./blur2d.js";
import { fftBlur2dCPU, directBlur2dCPU, fftSize2d, tiltParts } from "./fft2d.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures_2d.json", import.meta.url)));
let fails = 0;
const line = (ok, s) => {
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s}`);
};
const e = (x) => x.toExponential(1);
const maxAbs = (a) => a.reduce((s, v) => Math.max(s, Math.abs(v)), 0);
const maxDiff = (a, b) => a.reduce((s, v, i) => Math.max(s, Math.abs(v - b[i])), 0);

function closedForm(c) {
  const { S1, S2, K, s1, s2, rho, T, r, q1, q2 } = c;
  if (c.payoff === "exchange") return margrabe(S1, S2, s1, s2, rho, T, q1, q2);
  if (c.payoff === "best_of_call") return stulzCallMax(S1, S2, K, s1, s2, rho, T, r, q1, q2);
  if (c.payoff === "worst_of_call") return stulzCallMin(S1, S2, K, s1, s2, rho, T, r, q1, q2);
  if (c.payoff === "call_x_put") return (Math.exp(r * T) * bs(S1, 103, s1, T, r, q1) * bs(S2, 97, s2, T, r, q2, "put")) / 100;
  return null;
}

const jobOf = (c) => job2d({ ...c, payoff: fixturePayoff(c), c: fx.c, width: fx.width, patch: fx.patch });

console.log("JS 64-bit blur vs Python, same grid (c = 0.06, +-6 sd, patch +-0.2 in ln S)");
const jobs = [];
for (const c of fx.cases) {
  const job = jobOf(c);
  jobs.push(job);
  const same = ["nx", "ny", "mx", "my", "qx", "qy"].every((k) => job[k] === c[k]);
  const d = Math.max(...c.samples.map(([a, b, v]) => Math.abs(priceAt(job, a, b) - v)));
  const cf = closedForm(c);
  const dc = cf === null ? 0 : Math.abs(cf - c.closed);
  line(same && d < 1e-9 && dc < 1e-12, `${c.payoff.padEnd(14)} rho=${c.rho.toFixed(1).padStart(4)} image ${job.nx}x${job.ny}` +
    ` kernel ${job.mx}x${job.my}  25 patch prices |js-py| ${e(d)}` + (cf === null ? "" : `  closed form |js-py| ${e(dc)}`));
}

console.log("\nFFT plan (fft2d.js) in 64-bit vs the direct blur, relative to the largest price");
for (const [k, job] of jobs.entries()) {
  if (job.nx > 400) continue; // the 577 x 577 case is the same code, only slower
  const c = fx.cases[k];
  const dir = directBlur2dCPU(job);
  const scale = maxAbs(dir);
  const parts = tiltParts(job);
  const split = maxDiff(fftBlur2dCPU(job), dir) / scale;
  const plain = maxDiff(fftBlur2dCPU(job, { tilts: [[0, 0]] }), dir) / scale;
  const [Nx, Ny] = fftSize2d(job);
  line(split < 1e-12 && plain < 1e-12, `${c.payoff.padEnd(14)} rho=${c.rho.toFixed(1).padStart(4)} N=${Nx}x${Ny}` +
    ` parts ${parts.tilts.filter((t, i) => parts.Q[i].some((v) => v !== 0)).map((t) => `(${t})`).join(" ").padEnd(26)}` +
    ` split ${e(split)}  plain ${e(plain)}`);
}

// Wraparound. An image that is zero except its last row and last column (= 100):
// only outputs in the last row or column of the patch can reach them, so every
// other output is exactly 0. The padded FFT must keep them at roundoff; an FFT
// one size too small in either direction folds that edge back onto the start,
// and the test has to catch that. The kernel is flat (every tap 1/(mx my)), so
// a fold leaks O(1) whatever its depth; a Gaussian's 6-sd tail would hide a
// fold of a few rows below roundoff.
export function sentinel(job) {
  const P = new Float64Array(job.nx * job.ny);
  for (let i = 0; i < job.nx; i++) P[i * job.ny + job.ny - 1] = 100;
  for (let j = 0; j < job.ny; j++) P[(job.nx - 1) * job.ny + j] = 100;
  return P;
}
export const flat = (job) => new Float64Array(job.mx * job.my).fill(1 / (job.mx * job.my));
console.log("\nwraparound: image zero except its far edges = 100; outputs that can't reach them must be 0");
for (const k of [0, 1, 9]) {
  const job = { ...jobs[k], P: sentinel(jobs[k]), W: flat(jobs[k]) };
  const [Nx, Ny] = fftSize2d(job);
  const leak = (out) => {
    let m = 0;
    for (let a = 0; a < job.ox - 1; a++) for (let b = 0; b < job.oy - 1; b++) m = Math.max(m, Math.abs(out[a * job.oy + b]));
    return m;
  };
  const safe = leak(fftBlur2dCPU(job));
  line(safe < 1e-12, `image ${job.nx}x${job.ny} padded N=${Nx}x${Ny}: leak ${e(safe)}`);
  for (const N of [[Nx / 2, Ny], [Nx, Ny / 2]]) {
    const short = leak(fftBlur2dCPU(job, { N }));
    line(short > 1e-3, `image ${job.nx}x${job.ny} one size short N=${N[0]}x${N[1]}: leak ${e(short)}` +
      (short > 1e-3 ? " (caught)" : " (NOT caught: the test is blind)"));
  }
}
// and on a real payoff the comparison with the direct blur catches it
{
  const job = jobs[0];
  const [Nx, Ny] = fftSize2d(job);
  const d = maxDiff(fftBlur2dCPU(job, { N: [Nx / 2, Ny] }), directBlur2dCPU(job));
  // only 3 rows fold, through the kernel's 6-sd tail: small, but far above 64-bit roundoff
  line(d > 1e-10, `exchange payoff, one size short N=${Nx / 2}x${Ny}: |fft - direct| ${e(d)}` + (d > 1e-10 ? " (caught)" : " (NOT caught)"));
}

console.log(fails ? `${fails} failed` : "all 2D checks pass");
process.exit(fails ? 1 : 0);
