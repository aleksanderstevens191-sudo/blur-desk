// The demo page: draw a payoff, move expiry and volatility, and the GPU blurs
// it into a price surface over (stock price, time to expiry).
import { Payoff, priceCurve, S0, R } from "./pricer.js";
import { bsCall, bsPut } from "./bs.js";
import { gpuInfo } from "./gpu.js";
import { Camera, fit } from "./wire.js";
import { drawScene, toMesh, maskToMesh, MESH_S } from "./scene.js";
import { drawPad, padXY, drawKernel, drawCoil, drawExercise, drawError, drawBars } from "./panels.js";

const SLICES = 12; // surface slices between the payoff and today
const PRESETS = {
  call: (S) => Math.max(S - S0, 0),
  put: (S) => Math.max(S0 - S, 0),
  straddle: (S) => Math.abs(S - S0),
  // Teeth 20 wide and 25 tall between 60 and 140, flat outside.
  zigzag: (S) => (S <= 60 || S >= 140 ? 0 : 25 * (1 - Math.abs((((S - 60) % 20) - 10) / 10))),
};
const CLOSED_FORM = { call: bsCall, put: bsPut };

const $ = (id) => document.getElementById(id);
const state = { preset: "call", T: +$("T").value, sigma: +$("sigma").value, american: false };
let payoff = new Payoff(PRESETS.call);
let useGPU = true;

// What the scene draws; filled in as pricing results arrive.
const view = {
  T: state.T,
  yMax: 60,
  yTarget: 60,
  american: false,
  payoff: null,
  payoffEx: null,
  front: null,
  slices: [],
};
let errs = null; // |V - BS| binned onto the mesh
let boundary = []; // [S, tau] points of the early-exercise boundary
const times = [];

// ---- pricing: front curve first (fast feedback), then the surface ----
let busy = false;
let dirty = false;

function payoffMesh() {
  view.payoff = Float64Array.from(MESH_S, (S) => payoff.value(S));
  view.payoffEx = Uint8Array.from(view.payoff, (p) => (p > 0 ? 1 : 0));
}

async function price(T) {
  try {
    return await priceCurve(payoff, { T, sigma: state.sigma, american: state.american, gpu: useGPU });
  } catch (e) {
    if (!useGPU) throw e;
    useGPU = false; // device lost or unsupported: fall back to the CPU blur
    $("cGpu").textContent = "CPU FALLBACK";
    $("fGpu").textContent = `WEBGPU FAILED: ${e.message}`;
    return priceCurve(payoff, { T, sigma: state.sigma, american: state.american, gpu: false });
  }
}

async function reprice() {
  dirty = true;
  if (busy) return;
  busy = true;
  while (dirty) {
    dirty = false;
    const snap = { ...state };
    payoffMesh();
    const t0 = performance.now();
    const c = await price(snap.T);
    const ms = performance.now() - t0;
    const result = { ...c, ...snap, ms };
    const err = maxError(result);
    window.blurDesk = { result, error: err }; // for scripted checks
    view.T = snap.T;
    view.american = snap.american;
    view.front = { tau: snap.T, V: toMesh(c.S, c.V), ex: maskToMesh(c.S, c.exercise) };
    if (snap.T <= 0) view.slices = [];
    times.push(ms);
    if (times.length > 64) times.shift();
    updateStats(result, err);
    if (!snap.american) boundary = [];
    else if (!boundary.length) boundary = exerciseBoundary();
    if (dirty || snap.T <= 0) {
      if (snap.T <= 0) boundary = [];
      continue;
    }

    const taus = Array.from({ length: SLICES - 1 }, (_, j) => (snap.T * (j + 1)) / SLICES);
    const curves = await Promise.all(taus.map(price));
    view.slices = curves.map((s, j) => ({ tau: taus[j], V: toMesh(s.S, s.V), ex: maskToMesh(s.S, s.exercise) }));
    boundary = exerciseBoundary();
  }
  busy = false;
}

function exerciseBoundary() {
  if (!view.american) return [];
  const all = [{ tau: 0, ex: view.payoffEx }, ...view.slices, view.front];
  const pts = [];
  for (const s of all) {
    if (!s || !s.ex) continue;
    for (let i = 0; i + 1 < s.ex.length; i++) {
      if (s.ex[i] !== s.ex[i + 1]) {
        pts.push([MESH_S[s.ex[i] ? i : i + 1], s.tau]);
        break;
      }
    }
  }
  return pts;
}

function maxError(r) {
  const f = CLOSED_FORM[r.preset];
  if (!f || r.american) {
    errs = null;
    return null;
  }
  errs = new Float64Array(MESH_S.length);
  let e = 0;
  for (let i = 0; i < r.S.length; i++) {
    const d = Math.abs(r.V[i] - f(r.S[i], S0, r.sigma, r.T, R));
    e = Math.max(e, d);
    const k = Math.min(MESH_S.length - 1, Math.max(0, Math.round(r.S[i] - MESH_S[0])));
    errs[k] = Math.max(errs[k], d);
  }
  return e;
}

function updateStats(r, e) {
  const fmt = (x) => x.toExponential(1).toUpperCase();
  const cErr = $("cErr");
  const big = $("bigErr");
  if (e !== null) {
    const cls = e < 1e-4 ? "good" : "bad";
    cErr.textContent = fmt(e);
    cErr.className = cls;
    big.textContent = fmt(e);
    big.className = `big ${cls}`;
  } else {
    const why = r.american ? "AMERICAN: NO CLOSED FORM" : "NO CLOSED FORM";
    cErr.textContent = "—";
    cErr.className = "";
    big.textContent = why;
    big.className = "big na";
  }
  $("fN").textContent = `${r.S.length.toLocaleString()} PRICES`;
  $("fMs").innerHTML = `<b>${r.ms.toFixed(1)}</b> MS`;
  $("cGpu").textContent = `${useGPU ? "GPU" : "CPU"} ${r.ms.toFixed(1)} MS`;
  $("fEx2").textContent = r.american ? `${r.steps} DATES` : "";
  const std = r.sigma * Math.sqrt(r.T);
  $("fStd").textContent = std.toFixed(3);
  view.yTarget = Math.max(60, Math.ceil(Math.max(...view.front.V) / 20) * 20);
}

// ---- input ----
function readSliders() {
  state.T = +$("T").value;
  state.sigma = +$("sigma").value;
  $("cT").textContent = $("fT").textContent = `${state.T.toFixed(2)} YR`;
  $("cSigma").textContent = $("fSigma").textContent = `${Math.round(state.sigma * 100)}%`;
  $("fStd").textContent = (state.sigma * Math.sqrt(state.T)).toFixed(3);
  $("fSlices").textContent = state.T > 0 ? String(SLICES) : "0";
}
for (const id of ["T", "sigma"]) {
  $(id).addEventListener("input", () => {
    readSliders();
    reprice();
  });
}

function setAmerican(on) {
  state.american = on;
  const b = $("american");
  b.setAttribute("aria-pressed", String(on));
  b.textContent = on ? "AMERICAN: ON" : "AMERICAN: OFF";
  $("cMode").textContent = on ? "AMERICAN" : "EUROPEAN";
  $("exTag").textContent = on ? "AMERICAN" : "EUROPEAN";
  $("fEx1").textContent = on ? "BLUR → DISCOUNT → CLIP" : "EXPIRY ONLY";
  $("legEx").hidden = !on;
  reprice();
}
$("american").addEventListener("click", () => setAmerican(!state.american));
$("exercise").addEventListener("click", () => setAmerican(!state.american));

function setPreset(name) {
  state.preset = name;
  for (const b of document.querySelectorAll("[data-preset]")) b.setAttribute("aria-pressed", String(b.dataset.preset === name));
  if (name) payoff = new Payoff(PRESETS[name]);
}
for (const b of document.querySelectorAll("[data-preset]")) {
  b.addEventListener("click", () => {
    setPreset(b.dataset.preset);
    reprice();
  });
}

// drawing pad
const pad = $("pad");
let last = null;
function padPoint(e) {
  const rect = pad.getBoundingClientRect();
  const { toS, toY } = padXY(rect.width, rect.height);
  return { S: toS(e.clientX - rect.left), y: Math.min(60, Math.max(0, toY(e.clientY - rect.top))) };
}
pad.addEventListener("pointerdown", (e) => {
  pad.setPointerCapture(e.pointerId);
  last = padPoint(e);
  setPreset(null);
  payoff.stroke(last.S, last.y, last.S, last.y);
  reprice();
});
pad.addEventListener("pointermove", (e) => {
  if (!last) return;
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  for (const ev of evs.length ? evs : [e]) {
    const p = padPoint(ev);
    payoff.stroke(last.S, last.y, p.S, p.y);
    last = p;
  }
  reprice();
});
pad.addEventListener("pointerup", () => (last = null));
pad.addEventListener("pointercancel", () => (last = null));

// camera: auto orbit, preset views, drag to orbit
const scene = $("scene");
const cam = new Camera({ az: -0.55, el: 0.42, dist: 3.6, target: [0, 0.34, 0] });
let mode = "orbit";
const VIEWS = {
  front: { az: 0, el: 0.06 },
  side: { az: -Math.PI / 2 + 0.001, el: 0.06 },
  top: { az: 0, el: 1.5 },
};
function setView(m) {
  mode = m;
  for (const b of document.querySelectorAll("[data-view]")) b.setAttribute("aria-pressed", String(b.dataset.view === m));
}
for (const b of document.querySelectorAll("[data-view]")) b.addEventListener("click", () => setView(b.dataset.view));
let drag = null;
scene.addEventListener("pointerdown", (e) => {
  scene.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY };
  setView("free");
});
scene.addEventListener("pointermove", (e) => {
  if (!drag) return;
  cam.az -= (e.clientX - drag.x) * 0.008;
  cam.el = Math.min(1.5, Math.max(0.02, cam.el + (e.clientY - drag.y) * 0.006));
  drag = { x: e.clientX, y: e.clientY };
});
scene.addEventListener("pointerup", () => (drag = null));
scene.addEventListener("pointercancel", () => (drag = null));

// ---- render loop ----
const ctxs = {};
for (const id of ["scene", "pad", "kernel", "coil", "exercise", "error", "bars"]) ctxs[id] = $(id).getContext("2d");

function frame(ms) {
  const t = ms / 1000;
  if (mode === "orbit") {
    cam.az += (-0.55 + 0.5 * Math.sin(t * 0.13) - cam.az) * 0.05;
    cam.el += (0.42 + 0.08 * Math.sin(t * 0.09) - cam.el) * 0.05;
  } else if (VIEWS[mode]) {
    cam.az += (VIEWS[mode].az - cam.az) * 0.08;
    cam.el += (VIEWS[mode].el - cam.el) * 0.08;
  }
  view.yMax += (view.yTarget - view.yMax) * 0.15;

  if (view.payoff) {
    const [W, H] = fit(scene, ctxs.scene);
    cam.zoom = W / H < 1.2 ? 1.25 : Math.min(1.6, W / H);
    cam.setViewport(W, H, W / 2, H * 0.5);
    drawScene(ctxs.scene, cam, view, W, H, t);
  }
  {
    const [W, H] = fit(pad, ctxs.pad);
    drawPad(ctxs.pad, W, H, payoff, view.front && view.front.V, MESH_S);
  }
  {
    const [W, H] = fit($("kernel"), ctxs.kernel);
    drawKernel(ctxs.kernel, W, H, t, state.sigma * Math.sqrt(state.T));
  }
  {
    const [W, H] = fit($("coil"), ctxs.coil);
    drawCoil(ctxs.coil, W, H, t, state.T);
  }
  {
    const [W, H] = fit($("exercise"), ctxs.exercise);
    drawExercise(ctxs.exercise, W, H, t, view.american, boundary, view.T);
  }
  {
    const [W, H] = fit($("error"), ctxs.error);
    drawError(ctxs.error, W, H, errs, MESH_S);
  }
  {
    const [W, H] = fit($("bars"), ctxs.bars);
    drawBars(ctxs.bars, W, H, t, times);
  }
  requestAnimationFrame(frame);
}

async function start() {
  readSliders();
  setPreset("call");
  try {
    const name = await gpuInfo();
    $("fGpu").textContent = name.toUpperCase();
  } catch (e) {
    useGPU = false;
    $("cGpu").textContent = "CPU FALLBACK";
    $("fGpu").textContent = e.message.toUpperCase();
  }
  requestAnimationFrame(frame);
  reprice();
}

start();
