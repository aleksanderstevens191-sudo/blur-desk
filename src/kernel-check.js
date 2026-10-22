// /check.html, fat-tailed kernels section: the JS kernels (kernels.js) blurred
// on the GPU against the Python reference (kernel_fixtures.json). Each case is
// built on Python's grid exactly. Two comparisons per case:
//   strikes  one kink-corrected payoff row per strike, priced at S0 (the centre node)
//   smile    the demo's smile panel code (smile.js): one call row and one put row
//            with strike F, every strike read off them, prices and implied vols
// Wide kernels go through the FFT blur, as in the demo; the smile rows are also
// blurred directly as a cross-check.
import kfx from "./kernel_fixtures.json";
import { kernel, kinkCorrection } from "./kernels.js";
import { smile, blurRows, FFT_FROM } from "./smile.js";
import { directBlurGPU } from "./gpu-fft.js";

const e = (x) => x.toExponential(2);
const grid = (J, dx) => Float64Array.from({ length: 2 * J + 1 }, (_, j) => (j - J) * dx);

export async function kernelChecks(log, TOL) {
  let fails = 0;
  const line = (ok, s) => {
    if (!ok) fails++;
    log(`${ok ? "PASS" : "FAIL"} ${s}`);
  };
  log(`Each case on Python's grid (y_j = j dx, j = -J..J), kernel built in JS (64-bit), blurred on the GPU`);
  log(`(32-bit; the FFT for kernels over ${FFT_FROM} taps). S0 = 100. Pass: |gpu - py| < 1e-4 on every price.`);
  for (const c of kfx.cases) {
    const { model, p, J, dx, S0 } = c;
    const m = 2 * J + 1;
    log(`\n${c.name}: ${m} taps, dx ${dx.toPrecision(3)}`);
    const t0 = performance.now();
    const y = grid(J, dx);
    const w = kernel(model, y, p);
    const tk = performance.now() - t0;

    // one row per strike, read at the centre node
    const rows = c.strikes.length;
    const z = Float64Array.from(y, (v) => Math.log(S0) + v);
    const F = new Float64Array(rows * m);
    c.strikes.forEach((s, i) => F.set(kinkCorrection(z, s.K, s.cp), i * m));
    const t1 = performance.now();
    const out = await blurRows(F, rows, m, w, dx);
    const tb = performance.now() - t1;
    const disc = Math.exp(-p.r * p.T);
    let d = 0;
    let gap = 0;
    c.strikes.forEach((s, i) => {
      d = Math.max(d, Math.abs(disc * out[i * m + J] - s.py));
      gap = Math.max(gap, Math.abs(disc * out[i * m + J] - s.ref));
    });
    const refName = model === "gaussian" ? "Black-Scholes" : model === "merton" ? "Merton series" : "VG mixture";
    line(d < TOL, `${String(rows).padStart(2)} strikes, ${c.strikes.map((s) => s.K.toFixed(0)).filter((v, i, a) => a.indexOf(v) === i).join(" ")}:` +
      ` |gpu - py| ${e(d)}   (|gpu - ${refName}| ${e(gap)}; kernel ${tk.toFixed(0)} ms, blur ${tb.toFixed(0)} ms)`);

    // the smile, as the demo computes it, at the fixture's strikes
    const t2 = performance.now();
    const sm = await smile(model, p, { grid: { y, dx, J, w }, omax: c.omax, S0 });
    const ts = performance.now() - t2;
    let dp = 0;
    let dv = 0;
    let masked = 0;
    for (const s of c.smile) {
      const i = s.o + sm.omax;
      dp = Math.max(dp, Math.abs(sm.price[i] - s.py));
      if (Number.isFinite(sm.iv[i])) dv = Math.max(dv, Math.abs(sm.iv[i] - s.iv));
      else masked++;
    }
    line(dp < TOL, `smile, one call row + one put row, ${sm.k.length} strikes (k = ${sm.k[0].toFixed(3)} to ${sm.k.at(-1).toFixed(3)}):` +
      ` at the ${c.smile.length} fixture strikes |gpu - py| ${e(dp)}, implied vol within ${(dv * 100).toFixed(4)} vol pts` +
      (masked ? ` (${masked} far out of the money, vega < 0.05, not inverted)` : "") + ` (${ts.toFixed(0)} ms)`);

    // the same two rows blurred directly
    const n = sm.n;
    const cen = J + sm.omax;
    const zz = Float64Array.from({ length: n }, (_, i) => Math.log(S0) + (i - cen) * dx);
    const R2 = new Float64Array(2 * n);
    R2.set(kinkCorrection(zz, sm.F, "call"), 0);
    R2.set(kinkCorrection(zz, sm.F, "put"), n);
    const t3 = performance.now();
    const dir = await directBlurGPU(R2, 2, n, w);
    const td = performance.now() - t3;
    let dd = 0;
    for (const s of c.smile) {
      const v = disc * Math.exp(s.o * dx) * dir[(s.cp === "call" ? 0 : n) + cen - s.o];
      dd = Math.max(dd, Math.abs(v - s.py));
    }
    line(dd < TOL, `smile rows blurred directly instead: |gpu - py| ${e(dd)} (${td.toFixed(0)} ms)`);
  }
  return fails;
}
