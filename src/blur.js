// Blur Desk core, plain JavaScript (runs in any browser and in Node).
// Mirrors reference/blur_pricer.py; the GPU version in week day 3 must match this.

export function gaussianKernel(dx, std, center = 0, width = 8) {
  const half = Math.ceil((width * std + Math.abs(center)) / dx);
  const k = new Float64Array(2 * half + 1);
  let sum = 0;
  for (let j = 0; j < k.length; j++) {
    const off = (j - half) * dx - center;
    k[j] = Math.exp(-(off * off) / (2 * std * std));
    sum += k[j];
  }
  for (let j = 0; j < k.length; j++) k[j] /= sum;
  return k;
}

// out[i] = sum_j k[j] * f(i + j - half), with odd-reflection edges
// (payoffs continue as straight lines past the grid).
export function blur(f, k) {
  const n = f.length;
  const half = (k.length - 1) / 2;
  if (half >= n) throw new Error("kernel wider than grid; widen the grid");
  const at = (i) => {
    if (i < 0) return 2 * f[0] - f[-i];
    if (i >= n) return 2 * f[n - 1] - f[2 * (n - 1) - i];
    return f[i];
  };
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const base = i - half;
    if (base >= 0 && base + k.length <= n) {
      for (let j = 0; j < k.length; j++) s += k[j] * f[base + j];
    } else {
      for (let j = 0; j < k.length; j++) s += k[j] * at(base + j);
    }
    out[i] = s;
  }
  return out;
}

// Log grid with ln(S0) exactly on a node.
export function logGrid(S0, halfWidth, dx) {
  const n = Math.ceil(halfWidth / dx);
  const x0 = Math.log(S0);
  const x = new Float64Array(2 * n + 1);
  for (let i = 0; i < x.length; i++) x[i] = x0 + (i - n) * dx;
  return { x, center: n };
}

// European option under Black-Scholes: one off-center blur in log space.
export function bsBlurPrices(payoff, sigma, T, r, x, dx) {
  const g = Float64Array.from(x, (xi) => payoff(Math.exp(xi)));
  const mu = (r - 0.5 * sigma * sigma) * T;
  const k = gaussianKernel(dx, sigma * Math.sqrt(T), mu);
  const V = blur(g, k);
  const disc = Math.exp(-r * T);
  for (let i = 0; i < V.length; i++) V[i] *= disc;
  return V;
}

// American option: blur one step, discount, clip to payoff, repeat.
export function americanBlur(payoff, sigma, T, r, steps, x, dx) {
  const P = Float64Array.from(x, (xi) => payoff(Math.exp(xi)));
  const dt = T / steps;
  const k = gaussianKernel(dx, sigma * Math.sqrt(dt), (r - 0.5 * sigma * sigma) * dt);
  const disc = Math.exp(-r * dt);
  let V = P.slice();
  for (let s = 0; s < steps; s++) {
    V = blur(V, k);
    for (let i = 0; i < V.length; i++) V[i] = Math.max(disc * V[i], P[i]);
  }
  return V;
}

// Edge detect across strikes: [1, -2, 1] / dK^2, scaled by e^{rT}.
// Turns a row of call prices into the market's probability density.
export function edgeDetectDensity(callPrices, dK, r, T) {
  const out = new Float64Array(callPrices.length - 2);
  const g = Math.exp(r * T) / (dK * dK);
  for (let i = 1; i < callPrices.length - 1; i++) {
    out[i - 1] = g * (callPrices[i - 1] - 2 * callPrices[i] + callPrices[i + 1]);
  }
  return out;
}
