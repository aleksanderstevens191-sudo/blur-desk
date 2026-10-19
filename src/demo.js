// The demo page: draw a payoff, move the sliders, watch the GPU blur it into a price.
import { Payoff, priceCurve, S_MIN, S_MAX, S0, R } from "./pricer.js";
import { bsCall, bsPut } from "./bs.js";
import { gpuInfo } from "./gpu.js";

const Y_MAX = 60;
const PRESETS = {
  call: (S) => Math.max(S - S0, 0),
  put: (S) => Math.max(S0 - S, 0),
  straddle: (S) => Math.abs(S - S0),
  // Teeth 20 wide and 25 tall between 60 and 140, flat outside.
  zigzag: (S) => (S <= 60 || S >= 140 ? 0 : 25 * (1 - Math.abs((((S - 60) % 20) - 10) / 10))),
};
const CLOSED_FORM = { call: bsCall, put: bsPut };

const $ = (id) => document.getElementById(id);
const canvas = $("plot");
const ctx = canvas.getContext("2d");
const css = getComputedStyle(document.documentElement);
const color = (name) => css.getPropertyValue(name).trim();

const state = { preset: "call", T: +$("T").value, sigma: +$("sigma").value, american: false };
let payoff = new Payoff(PRESETS.call);
let result = null; // last priced curve
let useGPU = true;
let drawn = false;

// ---- pricing: always price the latest state, never queue stale frames ----
let busy = false;
let dirty = false;

async function reprice() {
  dirty = true;
  if (busy) return;
  busy = true;
  while (dirty) {
    dirty = false;
    const snap = { ...state, payoff, version: payoff.version };
    const t0 = performance.now();
    let curve;
    try {
      curve = await priceCurve(payoff, { T: snap.T, sigma: snap.sigma, american: snap.american, gpu: useGPU });
    } catch (e) {
      if (!useGPU) throw e;
      useGPU = false; // device lost or unsupported: fall back to the CPU blur
      $("backend").textContent = `CPU (WebGPU failed: ${e.message})`;
      dirty = true;
      continue;
    }
    const ms = performance.now() - t0;
    result = { ...curve, ...snap, ms };
    window.blurDesk = { result, error: maxError(result) }; // for scripted checks
    updateStats();
    scheduleDraw();
  }
  busy = false;
}

function maxError(r) {
  const f = CLOSED_FORM[r.preset];
  if (!f || r.american) return null;
  let e = 0;
  for (let i = 0; i < r.S.length; i++) e = Math.max(e, Math.abs(r.V[i] - f(r.S[i], S0, r.sigma, r.T, R)));
  return e;
}

function updateStats() {
  const r = result;
  const e = maxError(r);
  const err = $("err");
  if (e !== null) {
    const cls = e < 1e-4 ? "good" : "bad";
    err.innerHTML = `max error vs Black-Scholes <b class="${cls}">${e.toExponential(1)}</b>`;
  } else if (r.american) {
    err.innerHTML = `American: blur, discount, clip × <b>${r.steps}</b>`;
  } else {
    err.innerHTML = `${r.preset ? r.preset : "your payoff"}: <b>no closed form</b>`;
  }
  const n = r.S.length;
  $("speed").innerHTML = `<b>${r.ms.toFixed(1)} ms</b> per curve (${n.toLocaleString()} prices)`;
}

// ---- drawing ----
let W = 0;
let H = 0;
let dpr = 1;
const pad = { l: 40, r: 14, t: 40, b: 50 };
const px = (S) => pad.l + ((S - S_MIN) / (S_MAX - S_MIN)) * (W - pad.l - pad.r);
const py = (y) => pad.t + (1 - y / Y_MAX) * (H - pad.t - pad.b);
const toS = (x) => S_MIN + ((x - pad.l) / (W - pad.l - pad.r)) * (S_MAX - S_MIN);
const toY = (y) => (1 - (y - pad.t) / (H - pad.t - pad.b)) * Y_MAX;

function resize() {
  dpr = window.devicePixelRatio || 1;
  W = canvas.clientWidth;
  H = canvas.clientHeight;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  pad.l = W < 500 ? 32 : 40;
  draw();
}

let drawQueued = false;
function scheduleDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => {
    drawQueued = false;
    draw();
  });
}

function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const small = W < 500;
  const fontPx = small ? 11 : 12;
  ctx.font = `${fontPx}px ui-monospace, "SF Mono", Menlo, Consolas, monospace`;

  // grid + axis labels
  ctx.lineWidth = 1;
  ctx.strokeStyle = color("--line");
  ctx.fillStyle = color("--muted");
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let S = S_MIN; S <= S_MAX; S += 20) {
    const x = Math.round(px(S)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, H - pad.b);
    ctx.stroke();
    ctx.fillText(String(S), x, H - pad.b + 7);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let y = 0; y <= Y_MAX; y += 20) {
    const yy = Math.round(py(y)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.l, yy);
    ctx.lineTo(W - pad.r, yy);
    ctx.stroke();
    ctx.fillText(String(y), pad.l - 6, yy);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("stock price at expiry", (pad.l + W - pad.r) / 2, H - 6);

  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
  ctx.clip();

  const r = result;
  // early exercise region
  if (r && r.exercise) {
    ctx.fillStyle = "rgba(255, 107, 107, 0.16)";
    let start = -1;
    for (let i = 0; i <= r.S.length; i++) {
      const on = i < r.S.length && r.exercise[i];
      if (on && start < 0) start = i;
      if (!on && start >= 0) {
        const x0 = px(r.S[start]);
        const x1 = px(r.S[i - 1]);
        ctx.fillRect(x0, pad.t, Math.max(1, x1 - x0), H - pad.t - pad.b);
        start = -1;
      }
    }
  }

  // payoff (what you drew)
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = color("--payoff");
  ctx.globalAlpha = 0.75;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < Payoff.N; i++) {
    const x = px(S_MIN + i * Payoff.STEP);
    const y = py(payoff.y[i]);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // price today (the blur)
  if (r) {
    ctx.strokeStyle = color("--price");
    ctx.lineWidth = 3;
    ctx.shadowColor = "rgba(255, 176, 32, 0.45)";
    ctx.shadowBlur = 10;
    ctx.beginPath();
    const stride = Math.max(1, Math.floor(r.S.length / (W * 1.5)));
    for (let i = 0; i < r.S.length; i += stride) {
      const x = px(r.S[i]);
      const y = py(r.V[i]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
  ctx.restore();

  // legend
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.font = `${small ? 12 : 13}px -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif`;
  let lx = pad.l;
  const ly = pad.t / 2 + 1;
  const item = (label, stroke, fill) => {
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fillRect(lx, ly - 6, 16, 12);
    } else {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(lx + 16, ly);
      ctx.stroke();
    }
    ctx.fillStyle = color("--text");
    ctx.fillText(label, lx + 22, ly);
    lx += 22 + ctx.measureText(label).width + 16;
  };
  item("payoff", color("--payoff"));
  item("price today", color("--price"));
  if (r && r.exercise) item("exercise early", null, "rgba(255, 107, 107, 0.45)");

  const hint = "drag to draw a payoff";
  if (!drawn && lx + ctx.measureText(hint).width < W - pad.r) {
    ctx.fillStyle = color("--muted");
    ctx.textAlign = "right";
    ctx.fillText(hint, W - pad.r, ly);
  }
}

// ---- input ----
let last = null;

function pointerPoint(e) {
  const rect = canvas.getBoundingClientRect();
  const S = toS(e.clientX - rect.left);
  const y = Math.min(Y_MAX, Math.max(0, toY(e.clientY - rect.top)));
  return { S, y };
}

canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  last = pointerPoint(e);
  setPreset(null);
  drawn = true;
  payoff.stroke(last.S, last.y, last.S, last.y);
  scheduleDraw();
  reprice();
});
canvas.addEventListener("pointermove", (e) => {
  if (!last) return;
  const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of events.length ? events : [e]) {
    const p = pointerPoint(ev);
    payoff.stroke(last.S, last.y, p.S, p.y);
    last = p;
  }
  scheduleDraw();
  reprice();
});
const endStroke = () => (last = null);
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

function setPreset(name) {
  state.preset = name;
  for (const b of document.querySelectorAll("[data-preset]")) b.setAttribute("aria-pressed", String(b.dataset.preset === name));
  if (name) payoff = new Payoff(PRESETS[name]);
}

for (const b of document.querySelectorAll("[data-preset]")) {
  b.addEventListener("click", () => {
    setPreset(b.dataset.preset);
    scheduleDraw();
    reprice();
  });
}

function readSliders() {
  state.T = +$("T").value;
  state.sigma = +$("sigma").value;
  $("Tout").textContent = `${state.T.toFixed(2)} yr`;
  $("sigmaout").textContent = `${Math.round(state.sigma * 100)}%`;
}
for (const id of ["T", "sigma"]) {
  $(id).addEventListener("input", () => {
    readSliders();
    reprice();
  });
}
$("american").addEventListener("change", (e) => {
  state.american = e.target.checked;
  reprice();
});

async function start() {
  readSliders();
  setPreset("call");
  state.american = $("american").checked;
  new ResizeObserver(resize).observe(canvas);
  try {
    $("backend").textContent = `GPU: ${await gpuInfo()}`;
  } catch (e) {
    useGPU = false;
    $("backend").textContent = `CPU fallback (${e.message})`;
  }
  reprice();
}

start();
