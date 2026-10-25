// /check.html, two-stock section (?only=2d): the 2D blur on the GPU, direct and
// through the FFT, against the Python fixtures (reference/make_2d_fixtures.py),
// and the 2D wraparound test with its proof that it bites.
import fx from "./fixtures_2d.json";
import { job2d, fixturePayoff } from "./blur2d.js";
import { fftSize2d } from "./fft2d.js";
import { direct2dGPU, fft2dJob, fft2dGPU } from "./gpu2d.js";

export const fixtureJob = (c) => job2d({ ...c, payoff: fixturePayoff(c), c: fx.c, width: fx.width, patch: fx.patch });

const e = (x) => x.toExponential(2);

// worst |gpu - py| over the fixture's 25 patch prices (the spot among them)
const worst = (c, job, out) => Math.max(...c.samples.map(([a, b, v]) => Math.abs(job.disc * out[a * job.oy + b] - v)));

export async function twoStockChecks(log, TOL) {
  let fails = 0;
  const line = (ok, s) => {
    if (!ok) fails++;
    log(`${ok ? "PASS" : "FAIL"} ${s}`);
  };
  log(`grid c = ${fx.c} of the conditional sds, kernel +-${fx.width} sd, patch +-${fx.patch} in ln S1 and ln S2;`);
  log("|gpu - py| is the worst of 25 prices on the patch (5 x 5, the spot among them).");
  for (const c of fx.cases) {
    const job = fixtureJob(c);
    let t0 = performance.now();
    const dir = await direct2dGPU(job);
    const tDir = performance.now() - t0;
    t0 = performance.now();
    const fj = await fft2dJob(job);
    const fft = await fj.read();
    fj.destroy();
    const tFft = performance.now() - t0;
    const dd = worst(c, job, dir);
    const df = worst(c, job, fft);
    const spot = job.disc * fft[job.qx * job.oy + job.qy];
    line(dd < TOL && df < TOL, `${c.payoff.padEnd(13)} rho=${c.rho.toFixed(1).padStart(4)} s=${c.s1}/${c.s2} T=${c.T}` +
      ` image ${job.nx}x${job.ny} kernel ${job.mx}x${job.my}  spot gpu=${spot.toFixed(6)} py=${c.spot.toFixed(6)}` +
      `  direct ${e(dd)} (${tDir.toFixed(0)} ms)  fft N=${fj.N.join("x")} parts ${fj.parts.length} ${e(df)} (${tFft.toFixed(0)} ms)`);
  }
  // the same FFT with the numeraire split turned off, on the widest image
  const wide = fx.cases.find((c) => c.s1 === 0.6 && c.payoff === "best_of_call");
  const job = fixtureJob(wide);
  const plain = worst(wide, job, await fft2dGPU(job, { tilts: [[0, 0]] }));
  log(`without the numeraire split, the FFT on ${wide.payoff} s=${wide.s1}/${wide.s2} T=${wide.T}` +
    ` (payoff up to ${Math.max(...job.P).toFixed(0)}): ${e(plain)}`);

  // wraparound: zero image except its far row and column; flat kernel
  log("\n2D wraparound: image zero except its last row and column (= 100), flat kernel;");
  log("outputs off the patch's last row and column can't reach them, so must be 0");
  for (const c of [fx.cases[0], fx.cases[1], fx.cases[9]]) {
    const base = fixtureJob(c);
    const P = new Float64Array(base.nx * base.ny);
    for (let i = 0; i < base.nx; i++) P[i * base.ny + base.ny - 1] = 100;
    for (let j = 0; j < base.ny; j++) P[(base.nx - 1) * base.ny + j] = 100;
    const s = { ...base, P, W: new Float64Array(base.mx * base.my).fill(1 / (base.mx * base.my)) };
    const leak = (out) => {
      let m = 0;
      for (let a = 0; a < s.ox - 1; a++) for (let b = 0; b < s.oy - 1; b++) m = Math.max(m, Math.abs(out[a * s.oy + b]));
      return m;
    };
    const [Nx, Ny] = fftSize2d(s);
    const safe = leak(await fft2dGPU(s));
    line(safe < TOL, `image ${s.nx}x${s.ny} padded N=${Nx}x${Ny} >= image: max |blur| there ${e(safe)}`);
    for (const N of [[Nx / 2, Ny], [Nx, Ny / 2]]) {
      const short = leak(await fft2dGPU(s, { N }));
      const caught = short >= 1e-3;
      line(caught, `image ${s.nx}x${s.ny} one size short N=${N.join("x")}: max |blur| there ${e(short)}` +
        (caught ? "  (leak caught, as it should be)" : "  (leak NOT caught: the test is blind)"));
    }
  }
  return fails;
}
