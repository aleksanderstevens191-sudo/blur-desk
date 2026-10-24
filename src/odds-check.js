// The market-odds checks (odds.js), shared by /check.html?only=odds (GPU, 32-bit)
// and node src/test-odds.mjs (CPU, 64-bit). For each SPY expiry of spy_fit.json:
//   - the distribution edge-detected from the blurred prices sums to 1 within 0.001
//   - its mean is the parity forward of the quotes it was fitted to, within 1 cent
//   - P(fall > 10%) and P(rise > 10%) agree with Merton's closed-form CDF within 1e-4
//   - no strike gets a probability below -1e-4 (no butterfly arbitrage beyond rounding)
//   - on the GPU: the CDF at every strike within 1e-4 of the same blur in 64-bit
import { marketOdds, ODDS_REACH } from "./odds.js";

export async function oddsChecks(log, SPY, { gpu = true } = {}) {
  let fails = 0;
  const line = (ok, s) => {
    if (!ok) fails++;
    log(`${ok ? "PASS" : "FAIL"} ${s}`);
  };
  const pc = (v) => `${(v * 100).toFixed(3)}%`;
  log(`SPY ${SPY.asof}, last $${SPY.spot}; Merton fit sigma ${SPY.merton.sigma.toFixed(4)}, ${SPY.merton.lam.toFixed(3)} jumps/yr, ` +
    `mean jump ${SPY.merton.m.toFixed(4)}, jump sd ${SPY.merton.delta.toFixed(4)} (spy_fit.json).`);
  log(`One call row + one put row blurred ${gpu ? "on the GPU (32-bit)" : "on the CPU (64-bit)"}, every strike read off them, ` +
    `out to ${ODDS_REACH.nSd} sd past the kernel's widest jump term; P(S_T <= K) = 1 + e^rT dC/dK.`);
  for (let e = 0; e < SPY.slices.length; e++) {
    const o = await marketOdds(SPY, e, { gpu });
    const usd = (k) => `$${(k * o.scale).toFixed(0)}`;
    log(`\n${o.sl.expiry} (T ${o.p.T.toFixed(4)}, r ${(o.p.r * 100).toFixed(2)}%): ${o.strikes} strikes ${usd(o.K[0])} to ${usd(o.K.at(-1))}, ` +
      `${o.taps} taps (${o.ms.toFixed(0)} ms)`);
    line(Math.abs(o.sum - 1) < 1e-3, `sums to ${o.sum.toFixed(6)} (|sum - 1| ${Math.abs(o.sum - 1).toExponential(1)} < 1e-3)`);
    const dF = o.meanUsd - o.forward;
    line(Math.abs(dF) < 0.01, `mean $${o.meanUsd.toFixed(4)} vs parity forward $${o.forward.toFixed(4)}: ` +
      `${dF >= 0 ? "+" : "-"}$${Math.abs(dF).toExponential(1)} (< $0.01)`);
    const d1 = Math.abs(o.pDown - o.exact.pDown);
    const d2 = Math.abs(o.pUp - o.exact.pUp);
    line(d1 < 1e-4, `P(SPY < $${o.down.toFixed(2)}, down 10%) ${pc(o.pDown)} vs Merton closed form ${pc(o.exact.pDown)}: |diff| ${d1.toExponential(1)}`);
    line(d2 < 1e-4, `P(SPY > $${o.up.toFixed(2)}, up 10%) ${pc(o.pUp)} vs Merton closed form ${pc(o.exact.pUp)}: |diff| ${d2.toExponential(1)}`);
    let neg = 0;
    for (let i = 1; i + 1 < o.mass.length; i++) neg = Math.min(neg, o.mass[i]);
    line(neg > -1e-4, `smallest probability at a strike ${neg.toExponential(1)} (> -1e-4)`);
    if (gpu) {
      const c = await marketOdds(SPY, e, { gpu: false });
      let dc = 0;
      for (let i = 0; i < c.cdf.length; i++) dc = Math.max(dc, Math.abs(o.cdf[i] - c.cdf[i]));
      line(dc < 1e-4, `CDF at every strike vs the same blur in 64-bit on the CPU: max |diff| ${dc.toExponential(1)}`);
    }
    log(`     flat vol ${(o.flat.sigma * 100).toFixed(2)}% for comparison: down 10% ${pc(o.flat.pDown)}, up 10% ${pc(o.flat.pUp)}`);
  }
  return fails;
}
