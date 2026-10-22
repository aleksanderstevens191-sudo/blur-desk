// The demo page: draw a payoff, move expiry and volatility, and the GPU blurs
// it into a price surface over (stock price, time to expiry). With a barrier
// on, the surface comes from the hall of mirrors (or step-by-step monitoring).
import { Payoff, priceCurve, priceBarrier, snapBarrier, S0, R } from "./pricer.js";
import { bsCall, bsPut } from "./bs.js";
import { barrierClosedForm, dkoClosedForm, vanillaBS, mirrorExponent } from "./barrier.js";
import { gpuInfo } from "./gpu.js";
import { Camera, fit } from "./wire.js";
import { drawScene, toMesh, maskToMesh, MESH_S } from "./scene.js";
import {
  drawPad, padXY, drawKernel, drawCoil, drawExercise, drawError, drawBars, drawBarrierView, drawMethod, drawMirrors,
} from "./panels.js";

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
const state = {
  preset: "call", T: +$("T").value, sigma: +$("sigma").value, american: false,
  barrier: "off", io: "out", L: +$("L").value, U: +$("U").value, method: "weighted", shift: true, steps: 48,
};
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
  barrier: null, // { L, U, io }: barrier levels (S) for the scene
  cf: null, // closed form on MESH_S, when there is one
  method: null, // rows for panel 08
  mirrors: null, // what panel 09 animates
};
let errs = null; // |V - BS| binned onto the mesh
let boundary = []; // [S, tau] points of the early-exercise boundary
const times = [];

// ---- pricing: front curve first (fast feedback), then the surface ----
let busy = false;
let dirty = false;

// Barrier levels from the sliders, snapped onto the pricing grid.
function barrierSpec(st) {
  if (st.barrier === "off") return null;
  const ud = st.barrier;
  return { ud, io: st.io, L: ud !== "up" ? snapBarrier(st.L) : null, U: ud !== "down" ? snapBarrier(st.U) : null };
}
const knocked = (b, S) => (b.L && S < b.L.S) || (b.U && S > b.U.S);

// At expiry a knock-out pays where the barrier hasn't been crossed, a knock-in where it has.
function payoffMesh(b) {
  view.payoff = Float64Array.from(MESH_S, (S) => (!b || knocked(b, S) === (b.io === "in") ? payoff.value(S) : 0));
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
  document.body.dataset.busy = "1"; // for scripted checks
  while (dirty) {
    dirty = false;
    const snap = { ...state, bar: barrierSpec(state) };
    payoffMesh(snap.bar);
    if (snap.bar) {
      await barrierPass(snap);
      continue;
    }
    view.barrier = view.cf = view.method = view.mirrors = null;
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
  document.body.dataset.busy = "0";
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
  return curveError(r.S, r.V, (S) => f(S, S0, r.sigma, r.T, R));
}

// max |V - closed form| over the curve; also bins it onto the mesh for panel 05
function curveError(S, V, f) {
  errs = new Float64Array(MESH_S.length);
  let e = 0;
  for (let i = 0; i < S.length; i++) {
    const d = Math.abs(V[i] - f(S[i]));
    e = Math.max(e, d);
    const k = Math.min(MESH_S.length - 1, Math.max(0, Math.round(S[i] - MESH_S[0])));
    errs[k] = Math.max(errs[k], d);
  }
  return e;
}

// ---- barriers ----
const IO_NAME = { out: "KNOCK-OUT", in: "KNOCK-IN" };
const barrierName = (b) => (b.ud === "double" ? `DOUBLE ${IO_NAME[b.io]}` : `${b.ud.toUpperCase()}-AND-${b.io.toUpperCase()}`);
const METHOD_NAME = { plain: "PLAIN MIRROR", weighted: "WEIGHTED MIRROR", step: "STEP BY STEP" };

// Continuous-barrier closed form for the call and put presets (strike 100).
function barrierCF(snap) {
  const cp = snap.preset === "call" || snap.preset === "put" ? snap.preset : null;
  if (!cp) return null;
  const { bar: b, sigma, T } = snap;
  if (b.ud === "double") {
    return (S) => {
      const d = dkoClosedForm(S, S0, b.L.S, b.U.S, sigma, T, R, 0, cp);
      return b.io === "out" ? d : vanillaBS(S, S0, sigma, T, R, 0, cp) - d;
    };
  }
  const H = b.ud === "down" ? b.L.S : b.U.S;
  return (S) => barrierClosedForm(`${b.ud}-and-${b.io} ${cp}`, S, S0, H, sigma, T, R);
}

async function priceB(snap) {
  const opts = { T: snap.T, sigma: snap.sigma, barrier: snap.bar, method: snap.method, shift: snap.shift, steps: snap.steps };
  try {
    return await priceBarrier(payoff, { ...opts, gpu: useGPU });
  } catch (e) {
    if (!useGPU) throw e;
    useGPU = false;
    $("cGpu").textContent = "CPU FALLBACK";
    $("fGpu").textContent = `WEBGPU FAILED: ${e.message}`;
    return priceBarrier(payoff, { ...opts, gpu: false });
  }
}

// One barrier reprice: every slice of the surface (and the image-count study
// for panel 09) comes back from one dispatch.
async function barrierPass(snap) {
  const b = snap.bar;
  const t0 = performance.now();
  const r = snap.T > 0 ? await priceB(snap) : null;
  const ms = performance.now() - t0;
  const front = r ? r.curves[r.curves.length - 1] : { S: MESH_S, V: view.payoff };
  const cf = barrierCF(snap);
  errs = null;
  const err = cf ? curveError(front.S, front.V, cf) : null;
  view.T = snap.T;
  view.american = false;
  view.barrier = { L: b.L && b.L.S, U: b.U && b.U.S, io: b.io };
  view.front = { tau: snap.T, V: toMesh(front.S, front.V), ex: null };
  view.slices = r ? r.curves.slice(0, -1).map((c) => ({ tau: c.tau, V: toMesh(c.S, c.V), ex: null })) : [];
  view.cf = cf ? Float64Array.from(MESH_S, cf) : null;
  boundary = [];

  // 08: each method at S = 100, against the closed form (or the weighted mirror)
  const ref = cf ? cf(S0) : r ? r.atS0.weighted : 0;
  const errOf = (v, isRef) => (isRef ? null : Math.abs(v - ref));
  view.method = r && [
    { name: "PLAIN", err: errOf(r.atS0.plain, false), sel: snap.method === "plain" },
    { name: "WEIGHTED", err: errOf(r.atS0.weighted, !cf), sel: snap.method === "weighted" },
    { name: `STEP ${snap.steps}${snap.shift ? "+BGK" : ""}`, err: errOf(r.atS0.step, false), sel: snap.method === "step" },
  ];
  // 09: the images, and how the price at S = 100 converges as they are added
  const last = r ? r.images[r.images.length - 1].V : 0;
  view.mirrors = r && {
    lo: b.L ? Math.log(b.L.S) : null,
    hi: b.U ? Math.log(b.U.S) : null,
    a: snap.method === "plain" ? 0 : mirrorExponent(snap.sigma, R),
    f: (x) => payoff.value(Math.exp(x)),
    x0: Math.log(S0),
    maxCells: 4,
    errs: r.images.map((im, M) => ({ M, err: cf ? Math.abs(im.V - cf(S0)) : M === r.images.length - 1 ? null : Math.abs(im.V - last) })),
    front: { S: front.S, V: front.V },
    std: snap.sigma * Math.sqrt(snap.T),
  };

  const result = { ...snap, S: front.S, V: front.V, ms, atS0: r && r.atS0, images: r && r.images };
  window.blurDesk = { result, error: err, method: view.method, closedFormAtS0: cf ? cf(S0) : null };
  times.push(ms);
  if (times.length > 64) times.shift();
  updateStats(result, err);
  const nPrices = r ? r.curves.reduce((s, c) => s + c.S.length, 0) : front.S.length;
  $("fN").textContent = `${nPrices.toLocaleString()} PRICES`;
  $("cMode").textContent = barrierName(b);
  $("barTag").textContent = barrierName(b);
  $("fBar").textContent = err === null ? "NO CLOSED FORM FOR THIS PAYOFF" : `MAX |BLUR − CLOSED FORM| ${err.toExponential(1).toUpperCase()}`;
  $("methTag").textContent = METHOD_NAME[snap.method];
  $("fMeth1").textContent = cf ? "AT S=100 vs CLOSED FORM" : "AT S=100 vs WEIGHTED MIRROR";
  $("fMeth2").innerHTML = r ? `PRICE <b>${r.atS0[snap.method].toFixed(6)}</b>` : "";
  const a = mirrorExponent(snap.sigma, R);
  $("mirTag").textContent = snap.method === "plain" ? "PLAIN: WEIGHT 1" : snap.method === "weighted" ? "WEIGHTED: (H/S)^a" : "STEP BY STEP USES NO MIRRORS · SHOWN WEIGHTED";
  $("fMir1").textContent = b.io === "out" ? "KNOCK-OUT = BLUR OF PAYOFF − ITS IMAGES" : "KNOCK-IN = VANILLA − KNOCK-OUT";
  $("fMir2").innerHTML = `a = 2ν/σ² = <b>${a.toFixed(3)}</b>`;
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
  if (!r.bar) $("cMode").textContent = r.american ? "AMERICAN" : "EUROPEAN";
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
  if (on && state.barrier !== "off") return; // barriers here are European
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

// barrier controls
function readBarrier() {
  state.L = +$("L").value;
  state.U = +$("U").value;
  const b = barrierSpec(state);
  $("fL").textContent = b && b.L ? b.L.S.toFixed(2) : "—";
  $("fU").textContent = b && b.U ? b.U.S.toFixed(2) : "—";
}
function setBarrier(name) {
  state.barrier = name;
  for (const el of document.querySelectorAll("[data-barrier]")) el.setAttribute("aria-pressed", String(el.dataset.barrier === name));
  const on = name !== "off";
  if (on && state.american) setAmerican(false);
  $("american").disabled = on;
  $("L").disabled = !(name === "down" || name === "double");
  $("U").disabled = !(name === "up" || name === "double");
  $("cErrLbl").textContent = on ? "MAX ERROR vs CLOSED FORM" : "MAX ERROR vs B-S";
  $("errTitle").textContent = on ? "05 / ERROR vs CLOSED FORM" : "05 / ERROR vs B-S";
  $("legBar").hidden = !on;
  $("legBarText").textContent = state.io === "in" ? "BARRIER · KNOCKED IN" : "BARRIER · KNOCKED OUT";
  if (!on) {
    $("barTag").textContent = "OFF";
    $("fBar").textContent = "PICK A BARRIER";
    $("mirTag").textContent = "—";
    $("fMir1").textContent = $("fMir2").textContent = $("fMeth2").textContent = "";
    $("cMode").textContent = state.american ? "AMERICAN" : "EUROPEAN";
  }
  readBarrier();
}
const pick = (attr, value) => {
  for (const el of document.querySelectorAll(`[data-${attr}]`)) el.setAttribute("aria-pressed", String(el.dataset[attr] === String(value)));
};
for (const el of document.querySelectorAll("[data-barrier]")) {
  el.addEventListener("click", () => {
    setBarrier(el.dataset.barrier);
    reprice();
  });
}
for (const el of document.querySelectorAll("[data-io]")) {
  el.addEventListener("click", () => {
    state.io = el.dataset.io;
    pick("io", state.io);
    $("legBarText").textContent = state.io === "in" ? "BARRIER · KNOCKED IN" : "BARRIER · KNOCKED OUT";
    reprice();
  });
}
for (const el of document.querySelectorAll("[data-method]")) {
  el.addEventListener("click", () => {
    state.method = el.dataset.method;
    pick("method", state.method);
    reprice();
  });
}
for (const el of document.querySelectorAll("[data-steps]")) {
  el.addEventListener("click", () => {
    state.steps = +el.dataset.steps;
    pick("steps", state.steps);
    reprice();
  });
}
$("shift").addEventListener("click", () => {
  state.shift = !state.shift;
  $("shift").setAttribute("aria-pressed", String(state.shift));
  $("shift").textContent = state.shift ? "BGK SHIFT: ON" : "BGK SHIFT: OFF";
  reprice();
});
for (const id of ["L", "U"]) {
  $(id).addEventListener("input", () => {
    readBarrier();
    reprice();
  });
}

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
for (const id of ["scene", "pad", "kernel", "coil", "exercise", "error", "bars", "barrier", "method", "mirrors"]) ctxs[id] = $(id).getContext("2d");

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
    drawPad(ctxs.pad, W, H, payoff, view.front && view.front.V, MESH_S, view.barrier ? [view.barrier.L, view.barrier.U].filter(Boolean) : []);
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
  {
    const [W, H] = fit($("barrier"), ctxs.barrier);
    const d = view.barrier && view.front && { S: MESH_S, V: view.front.V, cf: view.cf, payoff: view.payoff, bar: view.barrier };
    drawBarrierView(ctxs.barrier, W, H, d);
  }
  {
    const [W, H] = fit($("method"), ctxs.method);
    drawMethod(ctxs.method, W, H, view.barrier && (view.method || "T = 0 · NOTHING TO BLUR"));
  }
  {
    const [W, H] = fit($("mirrors"), ctxs.mirrors);
    drawMirrors(ctxs.mirrors, W, H, t, view.barrier && (view.mirrors || "T = 0 · NOTHING TO BLUR"));
  }
  requestAnimationFrame(frame);
}

async function start() {
  readSliders();
  setPreset("call");
  setBarrier("off");
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
