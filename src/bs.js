// Black-Scholes closed forms, used by the demo to measure the blur's error.

// Standard normal CDF, Hart (1968) / West (2005); about 1e-14 absolute error.
export function normCdf(x) {
  const a = Math.abs(x);
  let c = 0;
  if (a < 37) {
    const e = Math.exp(-0.5 * a * a);
    if (a < 7.07106781186547) {
      let n = 3.52624965998911e-2 * a + 0.700383064443688;
      n = n * a + 6.37396220353165;
      n = n * a + 33.912866078383;
      n = n * a + 112.079291497871;
      n = n * a + 221.213596169931;
      n = n * a + 220.206867912376;
      let d = 8.83883476483184e-2 * a + 1.75566716318264;
      d = d * a + 16.064177579207;
      d = d * a + 86.7807322029461;
      d = d * a + 296.564248779674;
      d = d * a + 637.333633378831;
      d = d * a + 793.826512519948;
      d = d * a + 440.413735824752;
      c = (e * n) / d;
    } else {
      let d = a + 0.65;
      d = a + 4 / d;
      d = a + 3 / d;
      d = a + 2 / d;
      d = a + 1 / d;
      c = e / d / 2.506628274631;
    }
  }
  return x > 0 ? 1 - c : c;
}

export function bsCall(S, K, sigma, T, r) {
  if (T <= 0) return Math.max(S - K, 0);
  const v = sigma * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v;
  return S * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d1 - v);
}

export function bsPut(S, K, sigma, T, r) {
  if (T <= 0) return Math.max(K - S, 0);
  const v = sigma * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v;
  return K * Math.exp(-r * T) * normCdf(v - d1) - S * normCdf(-d1);
}
