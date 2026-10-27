// Panel 16, DIFFUSION: pricing blurs a payoff, a diffusion model blurs a density (Part 5).
//   left   a call's payoff melting into its price: Black-Scholes at r = 0 on the forward,
//          a Gaussian blur of the payoff in log-price
//   right  the market's Dec 18 odds (panel 13's) melting into noise: the variance-exploding
//          forward process of a diffusion model, a Gaussian blur of the density
//   slider one blur for both: the sd in log-return. On the left it is 14.6% vol over some
//          time; on the right it is the noise level sigma (sd^2 = sigma^2 - 0.01^2).
//   play   the reverse sampler: 65,536 points of noise run back through the probability-flow
//          ODE with the exact score (no network), the slider walking back with them.
// Both sides are checked live against closed forms (Black's formula; Merton's mixture with
// the blur's variance added); /check.html?only=diffusion checks the GPU against Python.
import SPY from "./spy_fit.json";
import { PANEL, market, grid, levels, mixture, mixPdf, mixCdf, blurCPU, payoffCPU, blackCall, startPoints, ksTest } from "./diffusion.js";
import { diffusionGPU, payoffGPU, KEEP } from "./gpu-diffusion.js";
import { glow, label, fit, WHITE, AMBER, TEXT } from "./wire.js";

const $ = (id) => document.getElementById(id);
const TOL = 1e-4;
const NP = 65536;
const PLAY_MS = 4200;
const X_LO = -0.3, X_HI = 0.25, NX = 276;         // the payoff side, ln(F_T / F)
const Y_LO = -1.5, Y_HI = 1.5, BINS = 200;         // the density side, ln(S_T / S)
const D_LO = 1e-3, D_HI = 20;

let P, G, lev, comps0, K, xs;
let gpu = null, gpuOK = true, ready = false, failure = null;
let st = { j: 48, pure: false };
let pay = new Map();                                // level -> prices per $100 of forward
let qCPU = new Map();
let live = null, busy = false, again = false, asked = 0;
let run = null, playing = null, seed = 16, tablesMs = 0, plays = 0, prefetching = null;

const qRow = (j) => (gpu ? gpu.Q.subarray(j * G.n, (j + 1) * G.n) : qCPU.get(j));
const sdOf = (j) => lev.s[j];

function horizon(s) {
  const yrs = (s * s) / P.flat / P.flat;
  return yrs < 2 ? `${Math.round(yrs * 365)} DAYS` : `${yrs.toFixed(1)} YEARS`;
}

// ---------- the live reprice at one level: both blurs, both closed forms ----------

async function level(j) {
  if (!pay.has(j)) {
    const s = sdOf(j);
    const prices = gpuOK ? (await payoffGPU(xs, s, K)).prices : Float64Array.from(xs, (x) => payoffCPU(x, s, K));
    pay.set(j, prices);
  }
  if (!gpu && !qCPU.has(j)) {
    qCPU.set(j, blurCPU(G.p0, sdOf(j)));
  }
  const s = sdOf(j);
  const comps = mixture(P, s * s);
  const q = qRow(j);
  let dErr = 0;
  for (let i = 0; i < G.n; i += 2) {
    if (G.y[i] < Y_LO || G.y[i] > Y_HI) continue;
    dErr = Math.max(dErr, Math.abs(q[i] - mixPdf(G.y[i], comps)));
  }
  const pr = pay.get(j);
  let pErr = 0;
  xs.forEach((x, i) => (pErr = Math.max(pErr, Math.abs(pr[i] - blackCall(x, s, K)))));
  return { j, dErr, pErr, worst: Math.max(dErr, pErr) };
}

async function reprice() {
  asked++;
  if (busy) {
    again = true;
    return;
  }
  busy = true;
  try {
    while (true) {
      again = false;
      const seq = asked;
      live = { ...(await level(st.j)), seq };
      if (!again) break;
    }
    failure = null;
  } catch (err) {
    failure = err;
  } finally {
    busy = false;
  }
  show();
}

// ---------- the reverse run ----------

// t0: when the walk back starts (performance.now() time); the recording passes its own clock
async function play(t0) {
  if (!gpu || playing) return null;
  const n = ++plays;
  const x1 = startPoints(NP, P, { pure: st.pure, seed: ++seed });
  const r = await gpu.sampler(x1);
  if (n !== plays) return null;                     // reset while the GPU was busy
  const final = r.levels[0];
  const ks = ksTest(final, (v) => mixCdf(v, comps0));
  const down = Math.log((0.9 * P.spot) / P.S);
  const pDown = final.reduce((a, v) => a + (v < down), 0) / NP;
  run = { levels: r.levels, ms: r.ms, ks, pDown, mDown: mixCdf(down, comps0), pure: st.pure, n: NP };
  playing = { t0: t0 ?? performance.now() };
  $("dfPlay").disabled = true;
  return run;
}

function tick(now) {
  if (!playing) return;
  const a = Math.min((now - playing.t0) / PLAY_MS, 1);
  const e = a < 0.5 ? 2 * a * a : 1 - 2 * (1 - a) * (1 - a);
  const j = Math.round(PANEL.N * (1 - e));
  if (j !== st.j) setLevel(j, false);
  if (a >= 1) {
    playing = null;
    $("dfPlay").disabled = false;
    show();
  }
}

// the points at level j, interpolated between the kept levels
function pointsAt(j) {
  const k = j / KEEP, k0 = Math.floor(k), k1 = Math.min(k0 + 1, run.levels.length - 1), f = k - k0;
  const a = run.levels[k0], b = run.levels[k1];
  return f === 0 ? a : Float32Array.from(a, (v, i) => v + f * (b[i] - v));
}

// the nearest level whose price is ready (the play animation can outrun the prefetch)
function payNear(j) {
  for (let d = 0; d <= PANEL.N; d++) {
    if (pay.has(j - d)) return pay.get(j - d);
    if (pay.has(j + d)) return pay.get(j + d);
  }
  return null;
}

// every level's price, one dispatch each, in the background
async function prefetch() {
  for (let j = PANEL.N; j >= 0; j--) {
    if (!pay.has(j)) pay.set(j, (await payoffGPU(xs, sdOf(j), K)).prices);
  }
}

// ---------- drawing ----------

// the margins grow with the labels (the recordings draw them 1.6x)
const box = (W, H) => ({ x0: 34 * TEXT.scale, y0: 18 * TEXT.scale, x1: W - 8, y1: H - 18 * TEXT.scale });

function drawPay() {
  const cv = $("dfPay"), ctx = cv.getContext("2d");
  const [W, H] = fit(cv, ctx);
  const b = box(W, H);
  const usd = P.F / 100, TOP = 480;
  const xOf = (x) => b.x0 + ((x - X_LO) / (X_HI - X_LO)) * (b.x1 - b.x0);
  const yOf = (v) => b.y1 - (Math.min(v, TOP) / TOP) * (b.y1 - b.y0);
  const curve = (f) => {
    const p = new Path2D();
    xs.forEach((x, i) => (i ? p.lineTo(xOf(x), yOf(f(x, i))) : p.moveTo(xOf(x), yOf(f(x, i)))));
    return p;
  };
  ctx.setLineDash([3, 3]);
  glow(ctx, curve((x) => usd * Math.max(100 * Math.exp(x) - K, 0)), WHITE, 0.35, 1, 0);
  ctx.setLineDash([]);
  const pr = payNear(st.j);
  if (pr) glow(ctx, curve((x, i) => usd * pr[i]), WHITE, 0.95, 1.4, 1);
  const ax = new Path2D();
  ax.moveTo(b.x0, b.y1);
  ax.lineTo(b.x1, b.y1);
  glow(ctx, ax, WHITE, 0.3, 1, 0);
  for (const v of [100, 200, 300, 400]) label(ctx, `$${v}`, b.x0 - 4, yOf(v), { size: 8, align: "right", alpha: 0.5 });
  label(ctx, `PAYOFF → PRICE · CALL $770 · r = 0 · ${(100 * P.flat).toFixed(1)}% VOL × ${horizon(sdOf(st.j))}`, b.x0, 8, { size: 8, align: "left", alpha: 0.6, max: W - b.x0 - 2 });
  label(ctx, "SPY FORWARD →", b.x1, b.y1 + 9 * TEXT.scale, { size: 8, align: "right", alpha: 0.45 });
  const fwdAt = b.x1 - ctx.measureText("SPY FORWARD →").width - 6;
  for (const usdX of [650, 700, 770, 850]) {
    const x = Math.log(usdX / P.F);
    if (x <= X_LO || x >= X_HI) continue;
    if (xOf(x) + ctx.measureText(`$${usdX}`).width / 2 > fwdAt) break;   // it would run into the label
    label(ctx, `$${usdX}`, xOf(x), b.y1 + 9 * TEXT.scale, { size: 8, alpha: usdX === 770 ? 0.8 : 0.5 });
  }
}

function drawDen() {
  const cv = $("dfDen"), ctx = cv.getContext("2d");
  const [W, H] = fit(cv, ctx);
  const b = box(W, H);
  const xOf = (y) => b.x0 + ((y - Y_LO) / (Y_HI - Y_LO)) * (b.x1 - b.x0);
  const yOf = (d) => b.y1 - ((Math.log(Math.max(d, D_LO)) - Math.log(D_LO)) / (Math.log(D_HI) - Math.log(D_LO))) * (b.y1 - b.y0);
  const curve = (row) => {
    const p = new Path2D();
    let pen = false;
    for (let i = 0; i < G.n; i += 2) {
      const y = G.y[i];
      if (y < Y_LO || y > Y_HI) continue;
      if (row[i] < D_LO) {
        pen = false;
        continue;
      }
      pen ? p.lineTo(xOf(y), yOf(row[i])) : p.moveTo(xOf(y), yOf(row[i]));
      pen = true;
    }
    return p;
  };
  ctx.setLineDash([3, 3]);
  glow(ctx, curve(G.p0), WHITE, 0.35, 1, 0);
  ctx.setLineDash([]);
  const q = qRow(st.j);
  if (q) glow(ctx, curve(q), WHITE, 0.95, 1.4, 1);
  if (run) {
    const pts = pointsAt(st.j);
    const w = (Y_HI - Y_LO) / BINS;
    const cnt = new Float64Array(BINS);
    for (const v of pts) {
      const k = Math.floor((v - Y_LO) / w);
      if (k >= 0 && k < BINS) cnt[k]++;
    }
    const p = new Path2D();
    let pen = false;
    cnt.forEach((c, k) => {
      const d = c / (pts.length * w);
      if (d < D_LO) {
        pen = false;
        return;
      }
      const xa = xOf(Y_LO + k * w), xb = xOf(Y_LO + (k + 1) * w), yy = yOf(d);
      pen ? p.lineTo(xa, yy) : p.moveTo(xa, yy);
      p.lineTo(xb, yy);
      pen = true;
    });
    glow(ctx, p, AMBER, 0.9, 1, 0.5);
  }
  for (const d of [0.01, 0.1, 1, 10]) label(ctx, String(d), b.x0 - 4, yOf(d), { size: 8, align: "right", alpha: 0.5 });
  for (const y of [-1, -0.5, 0, 0.5]) label(ctx, y.toFixed(1), xOf(y), b.y1 + 9 * TEXT.scale, { size: 8, alpha: 0.5 });
  label(ctx, `MARKET ODDS ${P.expiry} → NOISE · σ ${lev.sig[st.j].toFixed(3)}`, b.x0, 8, { size: 8, align: "left", alpha: 0.6 });
  if (run) label(ctx, playing ? "● REVERSE RUN" : "● RUN BACK", b.x1, 8, { size: 8, align: "right", rgb: AMBER, alpha: 0.95 });
  label(ctx, "LN S_T/S →", b.x1, b.y1 + 9 * TEXT.scale, { size: 8, align: "right", alpha: 0.45 });
}

function show() {
  const s = sdOf(st.j);
  $("dfLvl").textContent = `SD ${s.toFixed(3)}`;
  $("dfSame").textContent = `ONE GAUSSIAN, SD ${s.toFixed(4)} IN LOG-RETURN: ${(100 * P.flat).toFixed(1)}% VOL OVER ${horizon(s)} ON THE LEFT, NOISE LEVEL SIGMA ${lev.sig[st.j].toFixed(3)} ON THE RIGHT`;
  const big = $("dfErr");
  if (failure) {
    big.className = "big na";
    big.textContent = `GPU ERROR: ${failure.message}`;
  } else if (live && live.j === st.j) {
    big.className = `big ${live.worst < TOL ? "good" : "bad"}`;
    big.textContent = live.worst.toExponential(1).toUpperCase();
    $("dfVs").textContent = `${gpuOK ? "GPU" : "CPU"} VS CLOSED FORMS: PRICE ${live.pErr.toExponential(1)} PER $100 (BLACK), DENSITY ${live.dErr.toExponential(1)} (MERTON + BLUR)`;
  }
  if (!gpu) {
    $("dfRun").textContent = gpuOK ? "PREPARING THE SCORE TABLES…" : "WEBGPU NEEDED FOR THE REVERSE RUN";
  } else if (run && !playing) {
    $("dfRun").innerHTML = `REBUILT FROM ${run.pure ? "PURE N(0,1) NOISE" : "THE ODDS BLURRED TO SIGMA 1"}: KS D <b>${run.ks.D.toFixed(4)}</b> (p ${run.ks.p.toFixed(2)}) VS THE MARKET; ` +
      `P(FALL > 10%) <b>${(100 * run.pDown).toFixed(2)}%</b>, MARKET ${(100 * run.mDown).toFixed(2)}%`;
  } else if (!run) {
    $("dfRun").textContent = "PLAY: 65,536 POINTS OF NOISE, RUN BACK BY THE PROBABILITY-FLOW ODE WITH THE EXACT SCORE (NO NETWORK)";
  }
  if (gpu) {
    $("dfGpu").textContent = `GPU: ${PANEL.N + 1} LEVELS × ${G.n} NODES OF BLUR AND SCORE ${tablesMs.toFixed(0)} MS` +
      (run ? ` · ${run.n.toLocaleString("en-US")} POINTS × ${PANEL.N} HEUN STEPS ${run.ms.toFixed(0)} MS` : "");
  }
}

function frame(now) {
  if (ready) {
    tick(now);
    drawPay();
    drawDen();
  }
  requestAnimationFrame(frame);
}

function setLevel(j, user = true) {
  st.j = j;
  $("dfS").value = String(j);
  if (user && playing) return;
  show();
  reprice();
}

export async function initDiffusion({ gpu: useGPU = true } = {}) {
  gpuOK = useGPU;
  P = market(SPY);
  G = grid(P);
  lev = levels();
  comps0 = mixture(P);
  K = (100 * 770) / P.F;
  xs = Float64Array.from({ length: NX }, (_, i) => X_LO + (i * (X_HI - X_LO)) / (NX - 1));
  $("dfS").max = String(PANEL.N);
  $("dfS").value = String(st.j);
  $("dfS").addEventListener("input", () => {
    if (!playing) setLevel(+$("dfS").value);
  });
  for (const b of document.querySelectorAll("[data-dfs]")) {
    b.addEventListener("click", () => {
      st.pure = b.dataset.dfs === "1";
      for (const c of document.querySelectorAll("[data-dfs]")) c.setAttribute("aria-pressed", String(c === b));
    });
  }
  $("dfPlay").addEventListener("click", () => play().catch((err) => ((failure = err), show())));
  $("dfPlay").disabled = true;
  ready = true;
  show();
  requestAnimationFrame(frame);
  if (gpuOK) {
    try {
      const t0 = performance.now();
      gpu = await diffusionGPU(G, lev);
      tablesMs = performance.now() - t0;
      $("dfPlay").disabled = false;
      prefetching = prefetch().catch(() => {});
    } catch (err) {
      failure = err;
      gpuOK = false;
    }
  }
  reprice();
}

// ?record=2 (demo.js) drives the panel through these
export const recorder = {
  NP,
  get run() { return run; },
  get playing() { return !!playing; },
  get expiry() { return P.expiry; },
  // the score tables and every level's price, so the sweep never waits on the GPU
  async warm() {
    if (!gpu) throw failure || new Error("WEBGPU NEEDED FOR THE REVERSE RUN");
    await prefetching;
  },
  // back to the first run's seed (17), so every take rebuilds the same points
  reset() {
    plays++;
    playing = null;
    run = null;
    seed = 16;
    st.pure = false;
    for (const c of document.querySelectorAll("[data-dfs]")) c.setAttribute("aria-pressed", String(c.dataset.dfs === "0"));
    $("dfPlay").disabled = !gpu;
    setLevel(0);
  },
  setLevel(j) {
    if (j !== st.j) setLevel(j);
  },
  play: (t0) => play(t0),
};

// for the headless check (live.seq === asked with busy false: live answers the latest request)
window.diffusion = {
  st,
  get live() { return live; },
  get run() { return run; },
  get asked() { return asked; },
  get busy() { return busy; },
  get playing() { return !!playing; },
  get ready() { return !!gpu; },
  get failure() { return failure; },
  setLevel,
  play: () => play(),
  setPure(pure) {
    document.querySelector(`[data-dfs="${pure ? 1 : 0}"]`).click();
  },
};
