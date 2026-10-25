// Panel 14, TWO STOCKS: the 2D blur live. Paint any payoff on (ln S1, ln S2),
// or pick Exchange, Best-of or Spread; move both vols and the correlation and
// watch the kernel's ellipse turn and stretch while the GPU reprices a patch
// of spot pairs around (100, 100). The spot price is checked live against the
// closed form (Margrabe, Stulz; the spread against an exact 1D integral, with
// Kirk's approximation beside it) and the patch against the 64-bit blur, the
// JS twin of reference/blur2d.py (src/test-2d.mjs holds the two within 1e-9).
import { job2d, priceAt, ellipse, drifts, bs, margrabe, stulzCallMax, exchangePayoff, bestOfCallPayoff, spreadPayoff } from "./blur2d.js";
import { fftBlur2dCPU } from "./fft2d.js";
import { direct2dGPU, fft2dJob } from "./gpu2d.js";
import { Camera, Lines, glow, label, fit, WHITE, AMBER } from "./wire.js";

const $ = (id) => document.getElementById(id);
const S0 = 100, R = 0.05, TOL = 1e-4;
const Q = 12; // the priced patch: (2Q + 1)^2 spot pairs, Q grid steps each side
const L = 1.5; // the paint canvas spans ln S0 +- L in both stocks (S 22 to 448)
const G = 96; // paint grid, G x G cells
const VMAX = 60; // the paint canvas's colour scale, $ (brightest above)
const BRUSH = 40; // painted height, $

const st = { preset: "exchange", s1: 0.3, s2: 0.25, rho: 0.5, T: 1, K: { best: 100, spread: 5 }, method: "fft" };
const paint = new Float32Array(G * G); // dollars at cell (i, j): i along ln S1, j along ln S2
let res = null; // the last reprice
let busy = false, again = false, gpuOK = true;

// ---------- payoffs ----------

const cellX = (i) => Math.log(S0) - L + ((i + 0.5) * 2 * L) / G;

// the painted image as a payoff: bilinear between cell centres, flat past the edges
function paintedPayoff() {
  const img = paint.slice();
  return (x, y) => {
    const u = Math.min(G - 1, Math.max(0, ((x - Math.log(S0) + L) * G) / (2 * L) - 0.5));
    const v = Math.min(G - 1, Math.max(0, ((y - Math.log(S0) + L) * G) / (2 * L) - 0.5));
    const i = Math.min(G - 2, Math.floor(u)), j = Math.min(G - 2, Math.floor(v));
    const fu = u - i, fv = v - j;
    const at = (a, b) => img[a * G + b];
    return (1 - fu) * ((1 - fv) * at(i, j) + fv * at(i, j + 1)) + fu * ((1 - fv) * at(i + 1, j) + fv * at(i + 1, j + 1));
  };
}

function presetPayoff() {
  if (st.preset === "exchange") return exchangePayoff();
  if (st.preset === "best") return bestOfCallPayoff(st.K.best);
  return spreadPayoff(st.K.spread);
}

// the payoff as a plain function (for drawing and for copying into the paint grid)
const payoffFn = () => (st.preset === "paint" ? paintedPayoff() : presetPayoff().g);

// ---------- referees ----------

// Kirk: the spread as a call on S1 struck at F2 + K with a blended vol
function kirk(S1, S2, K, s1, s2, rho, T, r) {
  const F1 = S1 * Math.exp(r * T), F2 = S2 * Math.exp(r * T);
  const w = F2 / (F2 + K);
  const v = Math.sqrt(s1 * s1 - 2 * rho * s1 * s2 * w + s2 * s2 * w * w);
  return Math.exp(-r * T) * bs(F1, F2 + K, v, T, 0);
}

// Exact spread: given z (stock 2's shock) the payoff is a call on S1 struck at
// S2_T + K with vol s1 sqrt(1 - rho^2); trapezoid over z (spectral for a Gaussian).
function spreadExact(S1, S2, K, s1, s2, rho, T, r) {
  const n = 721, zmax = 9, h = (2 * zmax) / (n - 1), sq = Math.sqrt(T);
  const vc = s1 * Math.sqrt(1 - rho * rho);
  let s = 0;
  for (let k = 0; k < n; k++) {
    const z = -zmax + k * h;
    const S2T = S2 * Math.exp((r - 0.5 * s2 * s2) * T + s2 * sq * z);
    const F1 = S1 * Math.exp((r - 0.5 * s1 * s1) * T + rho * s1 * sq * z + 0.5 * vc * vc * T);
    s += Math.exp(-0.5 * z * z) * bs(F1, S2T + K, vc, T, 0);
  }
  return (Math.exp(-r * T) * s * h) / Math.sqrt(2 * Math.PI);
}

function closedForm() {
  const { s1, s2, rho, T } = st;
  if (st.preset === "exchange") return { name: "MARGRABE", v: margrabe(S0, S0, s1, s2, rho, T) };
  if (st.preset === "best") return { name: "STULZ", v: stulzCallMax(S0, S0, st.K.best, s1, s2, rho, T, R) };
  if (st.preset === "spread") return { name: "EXACT 1D", v: spreadExact(S0, S0, st.K.spread, s1, s2, rho, T, R), kirk: kirk(S0, S0, st.K.spread, s1, s2, rho, T, R) };
  return null;
}

// ---------- the reprice ----------

async function reprice() {
  if (busy) {
    again = true;
    return;
  }
  busy = true;
  try {
    do {
      again = false;
      const { s1, s2, rho, T } = st;
      const payoff = st.preset === "paint" ? paintedPayoff() : presetPayoff();
      const t0 = performance.now();
      const job = job2d({ payoff, S1: S0, S2: S0, s1, s2, rho, T, r: R, c: 0.06, width: 6, qx: Q, qy: Q });
      let out, N = null, parts = null;
      if (gpuOK) {
        if (st.method === "fft") {
          const f = await fft2dJob(job);
          out = await f.read();
          [N, parts] = [f.N, f.parts.length];
          f.destroy();
        } else out = await direct2dGPU(job);
      } else out = fftBlur2dCPU(job);
      const ms = performance.now() - t0;
      // the 64-bit blur at the spot, the patch's corners and its edge midpoints
      let worst = 0;
      for (const a of [0, Q, 2 * Q]) for (const b of [0, Q, 2 * Q]) worst = Math.max(worst, Math.abs(job.disc * out[a * job.oy + b] - priceAt(job, a, b)));
      res = { spot: job.disc * out[Q * job.oy + Q], worst, ms, N, parts, taps: [job.mx, job.my], dx: job.dx, dy: job.dy, cf: closedForm(), method: gpuOK ? st.method : "cpu" };
      readout();
    } while (again);
  } catch (e) {
    $("tsTag").textContent = e.message.toUpperCase();
  } finally {
    busy = false;
  }
}

// ---------- readouts ----------

const ex = (x) => x.toExponential(1).toUpperCase();

function readout() {
  const r = res;
  const big = $("tsErr");
  $("tsPrice").textContent = r.spot.toFixed(4);
  if (r.cf) {
    const err = Math.abs(r.spot - r.cf.v);
    big.textContent = ex(err);
    big.className = `big ${err < TOL ? "good" : "bad"}`;
    $("tsVs").textContent = `|GPU − ${r.cf.name}|`;
    $("tsCf").textContent = `${r.cf.name} ${r.cf.v.toFixed(4)}`;
  } else {
    big.textContent = "NO CLOSED FORM";
    big.className = "big na";
    $("tsVs").textContent = "PAINTED PAYOFF";
    $("tsCf").textContent = "CHECKED AGAINST THE 64-BIT BLUR ONLY";
  }
  if (r.cf && r.cf.kirk !== undefined) {
    const gap = r.cf.kirk - r.cf.v;
    $("tsKirk").textContent = `KIRK ${r.cf.kirk.toFixed(4)} · ${gap >= 0 ? "+" : ""}${((100 * gap) / r.cf.v).toFixed(2)}%`;
  } else $("tsKirk").textContent = "";
  const ok = r.worst < TOL;
  $("tsGpu").innerHTML = `GPU VS 64-BIT BLUR (9 POINTS) <b class="${ok ? "good" : "bad"}">${ex(r.worst)}</b>`;
  const how = r.method === "fft" ? `FFT ${r.N.join("×")} · ${r.parts} PARTS` : r.method === "direct" ? "DIRECT" : "CPU FFT (NO WEBGPU)";
  $("tsHow").textContent = `${how} · KERNEL ${r.taps.join("×")} · ${(2 * Q + 1) ** 2} PRICES · ${r.ms.toFixed(0)} MS`;
  const e = ellipse(st.s1, st.s2, st.rho, st.T);
  $("tsTag").textContent = `ELLIPSE ${((e.angle * 180) / Math.PI).toFixed(0)}° · AXES ${e.major.toFixed(2)} × ${e.minor.toFixed(2)}`;
}

// ---------- drawing ----------

const paintCv = $("tsPaint"), surfCv = $("tsSurf");
const ctxP = paintCv.getContext("2d"), ctxS = surfCv.getContext("2d");
const heat = document.createElement("canvas");
heat.width = heat.height = G;
const heatCtx = heat.getContext("2d");
let heatDirty = true;

function drawHeat() {
  const g = payoffFn();
  const img = heatCtx.createImageData(G, G);
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const t = Math.min(1, Math.max(0, g(cellX(i), cellX(j)) / VMAX));
      const k = 4 * ((G - 1 - j) * G + i); // ln S2 up
      // black -> amber -> white
      img.data[k] = 255 * Math.min(1, 1.6 * t);
      img.data[k + 1] = 184 * Math.min(1, 1.6 * t) + 71 * Math.max(0, (t - 0.6) / 0.4);
      img.data[k + 2] = 64 * Math.min(1, 1.6 * t) * 0.4 + 191 * Math.max(0, (t - 0.6) / 0.4);
      img.data[k + 3] = 255 * (0.12 + 0.88 * Math.sqrt(t));
    }
  }
  heatCtx.putImageData(img, 0, 0);
  heatDirty = false;
}

// the paint canvas's square: [x0, y0, side] in CSS px
function square(W, H) {
  const side = Math.min(W - 34, H - 40);
  return [W - side - 6, 14, side];
}
const toPx = ([x0, y0, side], x, y) => [x0 + ((x - Math.log(S0) + L) / (2 * L)) * side, y0 + (1 - (y - Math.log(S0) + L) / (2 * L)) * side];

function drawPaint() {
  const [W, H] = fit(paintCv, ctxP);
  const sq = square(W, H);
  const [x0, y0, side] = sq;
  if (heatDirty) drawHeat();
  ctxP.imageSmoothingEnabled = true;
  ctxP.drawImage(heat, x0, y0, side, side);
  ctxP.strokeStyle = "rgba(236,238,242,0.18)";
  ctxP.strokeRect(x0 + 0.5, y0 + 0.5, side - 1, side - 1);
  // S1 = S2
  const diag = new Path2D();
  diag.moveTo(x0, y0 + side);
  diag.lineTo(x0 + side, y0);
  ctxP.setLineDash([3, 4]);
  glow(ctxP, diag, WHITE, 0.25, 1, 0);
  ctxP.setLineDash([]);
  // the kernel: 1, 2 and 3 sd ellipses around the spot's drifted mean, and the axes
  const e = ellipse(st.s1, st.s2, st.rho, st.T);
  const [m1, m2] = drifts(st.T, R, st.s1, st.s2);
  const cx = Math.log(S0) + m1, cy = Math.log(S0) + m2;
  const [ca, sa] = [Math.cos(e.angle), Math.sin(e.angle)];
  for (const k of [1, 2, 3]) {
    const p = new Path2D();
    for (let t = 0; t <= 96; t++) {
      const th = (t / 96) * 2 * Math.PI;
      const u = k * e.major * Math.cos(th), v = k * e.minor * Math.sin(th);
      const [px, py] = toPx(sq, cx + u * ca - v * sa, cy + u * sa + v * ca);
      t ? p.lineTo(px, py) : p.moveTo(px, py);
    }
    glow(ctxP, p, WHITE, k === 1 ? 0.95 : k === 2 ? 0.6 : 0.35, 1.2, 1);
  }
  const axes = new Path2D();
  for (const [len, dx, dy] of [[3 * e.major, ca, sa], [3 * e.minor, -sa, ca]]) {
    axes.moveTo(...toPx(sq, cx - len * dx, cy - len * dy));
    axes.lineTo(...toPx(sq, cx + len * dx, cy + len * dy));
  }
  glow(ctxP, axes, WHITE, 0.4, 1, 0);
  // the priced patch
  if (res) {
    const [a0, b0] = toPx(sq, Math.log(S0) - Q * res.dx, Math.log(S0) + Q * res.dy);
    const [a1, b1] = toPx(sq, Math.log(S0) + Q * res.dx, Math.log(S0) - Q * res.dy);
    const p = new Path2D();
    p.rect(a0, b0, a1 - a0, b1 - b0);
    glow(ctxP, p, AMBER, 0.9, 1, 1);
  }
  const [sx, sy] = toPx(sq, Math.log(S0), Math.log(S0));
  ctxP.fillStyle = `rgba(${AMBER},1)`;
  ctxP.fillRect(sx - 1.5, sy - 1.5, 3, 3);
  // ticks: S = 50, 100, 200, 400
  for (const S of [25, 50, 100, 200, 400]) {
    const [px] = toPx(sq, Math.log(S), 0);
    const [, py] = toPx(sq, 0, Math.log(S));
    label(ctxP, String(S), px, y0 + side + 9, { size: 8, alpha: 0.5 });
    label(ctxP, String(S), x0 - 4, py, { size: 8, alpha: 0.5, align: "right" });
  }
  label(ctxP, "S1 →", x0 + side, y0 + side + 19, { size: 8, align: "right", alpha: 0.6 });
  label(ctxP, "S2 ↑", x0, y0 - 8, { size: 8, align: "left", alpha: 0.6 });
  if (st.preset === "paint") label(ctxP, "DRAG TO PAINT · SHIFT OR RIGHT-DRAG ERASES", x0 + 6, y0 + 10, { size: 8, align: "left", alpha: 0.7 });
}

const cam = new Camera({ az: -0.7, el: 0.62, dist: 3.6, target: [0, 0.2, 0] });

// The blur itself: the kernel's density over (ln S1, ln S2) as a wireframe,
// on a window of +-3.5 of the larger marginal sd (both axes at one scale), with
// its 1 and 2 sd ellipses on the floor. Correlation turns and stretches it.
function drawKernel3d(t) {
  const [W, H] = fit(surfCv, ctxS);
  cam.az = -0.7 + 0.3 * Math.sin(t * 0.15);
  cam.zoom = 1.25;
  cam.setViewport(W, H, W / 2, H * 0.56);
  const { s1, s2, rho, T } = st;
  const a1 = s1 * Math.sqrt(T), a2 = s2 * Math.sqrt(T);
  const span = 3.5 * Math.max(a1, a2);
  const dens = (x, y) => {
    const u = x / a1, v = y / a2;
    return Math.exp(-(u * u - 2 * rho * u * v + v * v) / (2 * (1 - rho * rho)));
  };
  const n = 36;
  const P = (i, j) => {
    const x = -span + (2 * span * i) / n, y = -span + (2 * span * j) / n;
    return [x / span, 0.85 * dens(x, y), -y / span];
  };
  const floor = new Lines(cam);
  for (const [a, b, c, d] of [[-1, -1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [-1, 1, -1, -1]]) floor.seg([a, 0, b], [c, 0, d]);
  floor.seg([-1, 0, 1], [1, 0, -1]); // S1 = S2 (relative to the spot)
  glow(ctxS, floor.path, WHITE, 0.2, 1, 0);
  const ln = new Lines(cam);
  for (let i = 0; i <= n; i++) {
    ln.lift();
    for (let j = 0; j <= n; j++) ln.line(...P(i, j));
  }
  for (let j = 0; j <= n; j++) {
    ln.lift();
    for (let i = 0; i <= n; i++) ln.line(...P(i, j));
  }
  glow(ctxS, ln.path, WHITE, 0.55, 0.8, 0.6);
  const e = ellipse(s1, s2, rho, T);
  const [ca, sa] = [Math.cos(e.angle), Math.sin(e.angle)];
  for (const k of [1, 2]) {
    const el = new Lines(cam);
    for (let q = 0; q <= 96; q++) {
      const th = (q / 96) * 2 * Math.PI;
      const u = k * e.major * Math.cos(th), v = k * e.minor * Math.sin(th);
      el.line((u * ca - v * sa) / span, 0, -(u * sa + v * ca) / span);
    }
    glow(ctxS, el.path, AMBER, k === 1 ? 0.95 : 0.55, 1.2, 1);
  }
  label(ctxS, `THE BLUR · ±${(3.5 * Math.max(a1, a2)).toFixed(2)} IN LN S`, 8, 12, { size: 8, align: "left", alpha: 0.55 });
  label(ctxS, "— KERNEL DENSITY", W - 8, 12, { size: 8, align: "right", alpha: 0.6 });
  label(ctxS, "— 1 AND 2 SD ELLIPSES", W - 8, 24, { size: 8, align: "right", rgb: AMBER, alpha: 0.9 });
  label(ctxS, "S1 →", ...cam.project(1.12, 0, 0).slice(0, 2), { size: 8, alpha: 0.5 });
  label(ctxS, "S2 →", ...cam.project(0, 0, -1.12).slice(0, 2), { size: 8, alpha: 0.5 });
}

function frame(ms) {
  drawPaint();
  drawKernel3d(ms / 1000);
  requestAnimationFrame(frame);
}

// ---------- controls ----------

function setPreset(name) {
  if (name === "paint" && st.preset !== "paint") {
    // start the painting from what is on screen
    const g = payoffFn();
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) paint[i * G + j] = g(cellX(i), cellX(j));
  }
  st.preset = name;
  for (const b of document.querySelectorAll("[data-ts]")) b.setAttribute("aria-pressed", String(b.dataset.ts === name));
  const k = $("tsK");
  k.disabled = name === "exchange" || name === "paint";
  if (name === "best") [k.min, k.max, k.step, k.value] = [70, 140, 1, st.K.best];
  if (name === "spread") [k.min, k.max, k.step, k.value] = [0, 40, 0.5, st.K.spread];
  $("tsClear").disabled = name !== "paint";
  syncLabels();
  heatDirty = true;
  reprice();
}

function syncLabels() {
  $("tsV1").textContent = `${(100 * st.s1).toFixed(0)}%`;
  $("tsV2").textContent = `${(100 * st.s2).toFixed(0)}%`;
  $("tsRho").textContent = st.rho.toFixed(2);
  $("tsT").textContent = `${st.T.toFixed(2)}Y`;
  $("tsKv").textContent = st.preset === "best" ? `${st.K.best}` : st.preset === "spread" ? `${st.K.spread}` : "—";
}

function readSliders() {
  st.s1 = +$("tsS1").value / 100;
  st.s2 = +$("tsS2").value / 100;
  st.rho = +$("tsR").value;
  st.T = +$("tsTm").value;
  if (st.preset === "best") st.K.best = +$("tsK").value;
  if (st.preset === "spread") st.K.spread = +$("tsK").value;
  syncLabels();
  if (st.preset !== "paint") heatDirty = true;
  reprice();
}

// painting: raise cells toward BRUSH under a soft round brush, or erase
let painting = 0;
function brush(ev) {
  if (st.preset !== "paint") setPreset("paint");
  const rect = paintCv.getBoundingClientRect();
  const [x0, y0, side] = square(rect.width, rect.height);
  const u = ((ev.clientX - rect.left - x0) / side) * G, v = (1 - (ev.clientY - rect.top - y0) / side) * G;
  const rad = 5;
  for (let i = Math.max(0, Math.floor(u - rad)); i < Math.min(G, Math.ceil(u + rad)); i++) {
    for (let j = Math.max(0, Math.floor(v - rad)); j < Math.min(G, Math.ceil(v + rad)); j++) {
      const d2 = ((i + 0.5 - u) ** 2 + (j + 0.5 - v) ** 2) / (rad * rad);
      if (d2 >= 1) continue;
      const w = (1 - d2) ** 2;
      const k = i * G + j;
      paint[k] = painting < 0 ? paint[k] * (1 - w) : Math.max(paint[k], paint[k] + (BRUSH - paint[k]) * w * 0.5);
    }
  }
  heatDirty = true;
  reprice();
}

export function initTwoStock({ gpu = true } = {}) {
  gpuOK = gpu;
  for (const b of document.querySelectorAll("[data-ts]")) b.addEventListener("click", () => setPreset(b.dataset.ts));
  for (const b of document.querySelectorAll("[data-tsm]")) {
    b.addEventListener("click", () => {
      st.method = b.dataset.tsm;
      for (const c of document.querySelectorAll("[data-tsm]")) c.setAttribute("aria-pressed", String(c === b));
      reprice();
    });
  }
  for (const id of ["tsS1", "tsS2", "tsR", "tsTm", "tsK"]) $(id).addEventListener("input", readSliders);
  $("tsClear").addEventListener("click", () => {
    paint.fill(0);
    heatDirty = true;
    reprice();
  });
  paintCv.addEventListener("contextmenu", (e) => e.preventDefault());
  paintCv.addEventListener("pointerdown", (e) => {
    painting = e.button === 2 || e.shiftKey ? -1 : 1;
    paintCv.setPointerCapture(e.pointerId);
    brush(e);
  });
  paintCv.addEventListener("pointermove", (e) => painting && brush(e));
  paintCv.addEventListener("pointerup", () => (painting = 0));
  readSliders();
  setPreset("exchange");
  requestAnimationFrame(frame);
}

// for the headless check: the current state and the last result
window.twoStock = { st, get res() { return res; }, setPreset, reprice };
