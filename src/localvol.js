// Panel 15, LOCAL VOL: SPY's Dupire local vol (Part 4) and what it does to a blur.
//   heat   the local vol by strike and time (white = high), the chosen expiry marked
//   blur   one day's blur at spots across the strikes at that time: each bell is as
//          wide as the local vol there, so the blur is narrow near the money and wide
//          in the put wing. This is the variable-width step the GPU runs.
//   trip   round trip: the expiry's quotes priced under local vol and turned back into
//          implied vol, minus the surface they came from. Dots: Python at the GPU's
//          settings, every quote; amber: the GPU, live, on up to 16 liquid strikes.
//   risk   Stage 4: Part 1's barrier options on SPY under Black-Scholes at the
//          at-the-money vol, a fitted Merton and local vol, which all fit the vanillas.
import { lvTable, rowBlend, interp, lvJob, impliedVol } from "./lv.js";
import { lvGPU, STRIKES } from "./gpu-lv.js";
import { glow, label, fit, WHITE, AMBER, RED } from "./wire.js";

const $ = (id) => document.getElementById(id);
const K_LO = -0.6, K_HI = 0.4; // strikes shown, k = ln(K / F)
const D_LO = 1, D_HI = 900; // days shown (log)
const TOL = 0.1; // vol points: the Stage 3 requirement on liquid quotes

let data = null, tab = null, gpuOK = true;
let st = { e: 11, risk: 0 };
let res = null, busy = false, again = false, asked = 0, failure = null;
let heatImg = null;

const sigmaAt = (k, t) => {
  const { ra, rb, a } = rowBlend(tab, t);
  return (1 - a) * interp(k, tab.k, tab.rows[ra]) + a * interp(k, tab.k, tab.rows[rb]);
};
// the middle of the interval that ends at expiry e (where its local vol lives)
const midOf = (e) => 0.5 * ((e ? data.expiries[e - 1].T : 0) + data.expiries[e].T);

// ---------- heat ----------

function buildHeat(w, h) {
  const img = new ImageData(w, h);
  for (let py = 0; py < h; py++) {
    const days = D_HI * Math.pow(D_LO / D_HI, py / (h - 1));
    for (let px = 0; px < w; px++) {
      const k = K_LO + ((px + 0.5) / w) * (K_HI - K_LO);
      const v = Math.min(sigmaAt(k, days / 365) / 0.6, 1);
      const g = Math.round(255 * Math.pow(v, 0.8));
      const o = 4 * (py * w + px);
      img.data[o] = img.data[o + 1] = img.data[o + 2] = g;
      img.data[o + 3] = 255;
    }
  }
  return img;
}

function frameBox(W, H) {
  return { x0: 30, y0: 18, x1: W - 6, y1: H - 18 };
}

function drawHeat() {
  const cv = $("lvHeat"), ctx = cv.getContext("2d");
  const [W, H] = fit(cv, ctx);
  const b = frameBox(W, H);
  const w = Math.max(2, Math.round(b.x1 - b.x0)), h = Math.max(2, Math.round(b.y1 - b.y0));
  if (!heatImg || heatImg.width !== w || heatImg.height !== h) heatImg = buildHeat(w, h);
  const off = new OffscreenCanvas(w, h);
  off.getContext("2d").putImageData(heatImg, 0, 0);
  ctx.globalAlpha = 0.85;
  ctx.drawImage(off, b.x0, b.y0);
  ctx.globalAlpha = 1;
  const yOf = (days) => b.y0 + ((Math.log(D_HI) - Math.log(days)) / (Math.log(D_HI) - Math.log(D_LO))) * (b.y1 - b.y0);
  const xOf = (k) => b.x0 + ((k - K_LO) / (K_HI - K_LO)) * (b.x1 - b.x0);
  // the chosen expiry and the interval before it
  const ex = data.expiries[st.e];
  const p = new Path2D();
  p.moveTo(b.x0, yOf(ex.T * 365));
  p.lineTo(b.x1, yOf(ex.T * 365));
  glow(ctx, p, AMBER, 0.9, 1, 1);
  const q = new Path2D();
  q.moveTo(b.x0, yOf(midOf(st.e) * 365));
  q.lineTo(b.x1, yOf(midOf(st.e) * 365));
  ctx.setLineDash([3, 3]);
  glow(ctx, q, AMBER, 0.5, 1, 0);
  ctx.setLineDash([]);
  // the quoted strikes of each expiry interval: past them Dupire has no data (Stage 2's E cells)
  for (const side of [0, 1]) {
    const r = new Path2D();
    let t0 = D_LO;
    data.expiries.forEach((e) => {
      const kq = side ? Math.max(...e.k) : Math.min(...e.k);
      const x = xOf(Math.min(Math.max(kq, K_LO), K_HI));
      const t1 = Math.min(Math.max(e.T * 365, D_LO), D_HI);
      r.moveTo(x, yOf(t0));
      r.lineTo(x, yOf(t1));
      t0 = t1;
    });
    ctx.setLineDash([3, 2]);
    glow(ctx, r, RED, 1, 1.5, 0.6);
    ctx.setLineDash([]);
  }
  for (const d of [3, 10, 30, 100, 300]) label(ctx, `${d}D`, b.x0 - 4, yOf(d), { size: 8, align: "right", alpha: 0.5 });
  for (const k of [-0.4, -0.2, 0]) label(ctx, k.toFixed(1), xOf(k), b.y1 + 9, { size: 8, alpha: 0.5 });
  label(ctx, "LOCAL VOL, WHITE ≥ 60% · RED: LAST QUOTES", b.x0, 8, { size: 8, align: "left", alpha: 0.6 });
  label(ctx, "LN K/F →", b.x1, b.y1 + 9, { size: 8, align: "right", alpha: 0.45 });
}

// ---------- the variable-width blur ----------

function drawBlur() {
  const cv = $("lvBlur"), ctx = cv.getContext("2d");
  const [W, H] = fit(cv, ctx);
  const b = frameBox(W, H);
  const t = midOf(st.e);
  const h = 1 / 365;
  const xOf = (k) => b.x0 + ((k - K_LO) / (K_HI - K_LO)) * (b.x1 - b.x0);
  // local vol across the strikes (top), scale 0..80%
  const yv = (s) => b.y0 + 4 + (1 - Math.min(s, 0.8) / 0.8) * 0.38 * (b.y1 - b.y0);
  const lv = new Path2D();
  for (let i = 0; i <= 200; i++) {
    const k = K_LO + (i / 200) * (K_HI - K_LO);
    const y = yv(sigmaAt(k, t));
    i ? lv.lineTo(xOf(k), y) : lv.moveTo(xOf(k), y);
  }
  glow(ctx, lv, WHITE, 0.9, 1.2, 1);
  label(ctx, `LOCAL VOL AT ${(t * 365).toFixed(1)} DAYS`, b.x0, 8, { size: 8, align: "left", alpha: 0.6 });
  // bells: one day's blur at 13 spots, width = sigma(k, t) sqrt(1 day), drawn 4x wider
  const base = b.y1 - 4, top = b.y0 + 0.48 * (b.y1 - b.y0);
  const EXAG = 4;
  const spots = 13;
  const ks = Array.from({ length: spots }, (_, n) => K_LO + 0.04 + (n / (spots - 1)) * (K_HI - K_LO - 0.08));
  const ss = ks.map((k0) => sigmaAt(k0, t) * Math.sqrt(h));
  const narrow = Math.min(...ss), wide = Math.max(...ss);
  ks.forEach((k0, n) => {
    const sd = EXAG * ss[n];
    const pk = new Path2D();
    for (let i = 0; i <= 80; i++) {
      const z = -4 + (8 * i) / 80;
      const k = Math.min(Math.max(k0 + z * sd, K_LO), K_HI);
      const y = base - ((narrow / ss[n]) * Math.exp(-0.5 * z * z)) * (base - top);   // equal areas: the narrowest is tallest
      i ? pk.lineTo(xOf(k), y) : pk.moveTo(xOf(k), y);
    }
    glow(ctx, pk, AMBER, 0.85, 1, 0.7);
  });
  const ax = new Path2D();
  ax.moveTo(b.x0, base);
  ax.lineTo(b.x1, base);
  glow(ctx, ax, WHITE, 0.3, 1, 0);
  label(ctx, "1-DAY BLUR, 4X WIDE", b.x0, top - 10, { size: 8, align: "left", rgb: AMBER, alpha: 0.85 });
  label(ctx, `SD ${(100 * narrow).toFixed(2)}% → ${(100 * wide).toFixed(2)}%`, b.x1, top - 10, { size: 8, align: "right", alpha: 0.6 });
  for (const k of [-0.4, -0.2, 0]) label(ctx, k.toFixed(1), xOf(k), b.y1 + 9, { size: 8, alpha: 0.5 });
  label(ctx, "LN K/F →", b.x1, b.y1 + 9, { size: 8, align: "right", alpha: 0.45 });
}

// ---------- round trip ----------

function drawTrip() {
  const cv = $("lvTrip"), ctx = cv.getContext("2d");
  const [W, H] = fit(cv, ctx);
  const b = frameBox(W, H);
  const ex = data.expiries[st.e];
  const ks = ex.k.filter((_, j) => ex.liq[j]);
  const lo = Math.min(...ks, -0.05) - 0.02, hi = Math.max(...ks, 0.05) + 0.02;
  const R = 0.15;
  const xOf = (k) => b.x0 + ((k - lo) / (hi - lo)) * (b.x1 - b.x0);
  const yOf = (e) => b.y0 + ((R - Math.max(-R, Math.min(R, e))) / (2 * R)) * (b.y1 - b.y0);
  for (const e of [-TOL, TOL]) {
    const p = new Path2D();
    p.moveTo(b.x0, yOf(e));
    p.lineTo(b.x1, yOf(e));
    ctx.setLineDash([3, 3]);
    glow(ctx, p, RED, 0.45, 1, 0);
    ctx.setLineDash([]);
  }
  const z = new Path2D();
  z.moveTo(b.x0, yOf(0));
  z.lineTo(b.x1, yOf(0));
  glow(ctx, z, WHITE, 0.25, 1, 0);
  ex.k.forEach((k, j) => {
    if (k < lo || k > hi || ex.err[j] == null) return;
    ctx.fillStyle = ex.liq[j] ? "rgba(236,238,242,0.85)" : "rgba(236,238,242,0.22)";
    ctx.fillRect(xOf(k) - 1.5, yOf(ex.err[j]) - 1.5, 3, 3);
  });
  if (res && res.e === st.e) {
    ctx.fillStyle = `rgba(${AMBER},1)`;
    res.pts.forEach(({ k, err }) => {
      if (!Number.isFinite(err)) return;
      ctx.beginPath();
      ctx.arc(xOf(k), yOf(err), 3.2, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
  for (const e of [-0.1, 0, 0.1]) label(ctx, e.toFixed(1), b.x0 - 4, yOf(e), { size: 8, align: "right", alpha: 0.5 });
  label(ctx, "ROUND TRIP · VOL PTS", b.x0, 8, { size: 8, align: "left", alpha: 0.6 });
  label(ctx, "● GPU LIVE", b.x1, 8, { size: 8, align: "right", rgb: AMBER, alpha: 0.95 });
  label(ctx, `${lo.toFixed(2)}`, b.x0, b.y1 + 9, { size: 8, align: "left", alpha: 0.5 });
  label(ctx, `${hi.toFixed(2)} LN K/F`, b.x1, b.y1 + 9, { size: 8, align: "right", alpha: 0.5 });
}

// ---------- model risk ----------

function drawRisk() {
  const cv = $("lvRisk"), ctx = cv.getContext("2d");
  const [W, H] = fit(cv, ctx);
  const mr = data.modelRisk;
  const name = Object.keys(mr.expiries)[st.risk];
  const t = mr.expiries[name];
  const rows = t.rows.filter((r) => Math.max(r.bs, r.merton, r.lv) > 0.05);
  const narrow = W < 560; // phones: each row's name on its own line, above its bar
  const x0 = narrow ? 14 : 250, x1 = W - (narrow ? 92 : 120), y0 = narrow ? 36 : 22, y1 = H - 16;
  const lmin = Math.log(0.01), lmax = Math.log(300);
  const xOf = (v) => x0 + ((Math.log(Math.max(v, 0.01)) - lmin) / (lmax - lmin)) * (x1 - x0);
  const dy = (y1 - y0) / rows.length;
  for (const v of [0.1, 1, 10, 100]) {
    const p = new Path2D();
    p.moveTo(xOf(v), y0 - 4);
    p.lineTo(xOf(v), y1);
    glow(ctx, p, WHITE, 0.12, 1, 0);
    label(ctx, `$${v}`, xOf(v), y1 + 8, { size: 8, alpha: 0.45 });
  }
  rows.forEach((r, n) => {
    const y = y0 + (n + (narrow ? 0.7 : 0.5)) * dy;
    const name = `${r.type.toUpperCase()} H ${(100 * r.barrier).toFixed(0)}% K ${(100 * r.strike).toFixed(0)}%`;
    if (narrow) label(ctx, name, x0 - 6, y - 0.42 * dy, { size: 8, align: "left", alpha: 0.7 });
    else label(ctx, name, x0 - 10, y, { size: 8, align: "right", alpha: 0.7 });
    const vals = [r.bs, r.merton, r.lv];
    const p = new Path2D();
    p.moveTo(xOf(Math.min(...vals)), y);
    p.lineTo(xOf(Math.max(...vals)), y);
    glow(ctx, p, WHITE, 0.4, 1, 0);
    ctx.strokeStyle = "rgba(236,238,242,0.6)";
    ctx.beginPath();
    ctx.arc(xOf(r.bs), y, 3, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.strokeRect(xOf(r.merton) - 3, y - 3, 6, 6);
    ctx.fillStyle = `rgba(${AMBER},1)`;
    ctx.beginPath();
    ctx.moveTo(xOf(r.lv), y - 4);
    ctx.lineTo(xOf(r.lv) + 4, y);
    ctx.lineTo(xOf(r.lv), y + 4);
    ctx.lineTo(xOf(r.lv) - 4, y);
    ctx.fill();
    label(ctx, `$${(Math.max(...vals) - Math.min(...vals)).toFixed(2)} APART`, x1 + 8, y, { size: 8, align: "left", alpha: 0.75 });
  });
  if (narrow) {
    label(ctx, `MODEL RISK · ${name} (${t.days} DAYS) · $ PER SHARE (LOG)`, 8, 8, { size: 8, align: "left", alpha: 0.6, max: W - 16 });
    label(ctx, "○ BS ATM VOL   □ MERTON   ◆ LOCAL VOL", 8, 21, { size: 8, align: "left", alpha: 0.75, max: W - 16 });
  } else {
    label(ctx, `MODEL RISK · ${name} (${t.days} DAYS) · SAME VANILLAS, DIFFERENT BARRIERS · $ PER SHARE (LOG)`, 8, 8, { size: 8, align: "left", alpha: 0.6 });
    label(ctx, "○ BS AT-THE-MONEY VOL   □ MERTON (FITTED)   ◆ LOCAL VOL", W - 8, 8, { size: 8, align: "right", alpha: 0.75 });
  }
}

// ---------- the live GPU round trip ----------

function strikesFor(e) {
  const ex = data.expiries[e];
  const liq = ex.k.map((k, j) => j).filter((j) => ex.liq[j] && ex.err[j] != null);
  if (liq.length <= STRIKES) return liq;
  return Array.from({ length: STRIKES }, (_, n) => liq[Math.round((n * (liq.length - 1)) / (STRIKES - 1))]);
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
      const seq = asked, e = st.e;
      const ex = data.expiries[e];
      const js = strikesFor(e);
      if (!gpuOK) {
        res = { seq, e, pts: [], worst: NaN, ms: 0 };
      } else {
        const job = lvJob(data, { T: ex.T, D: ex.D, strikes: js.map((j) => ({ k: ex.k[j], call: ex.call[j] })), tab });
        const { prices, ms } = await lvGPU(job);
        const pts = js.map((j, n) => {
          const iv = impliedVol(prices[n] / job.scale, ex.k[j], ex.T, ex.call[j]);
          return { k: ex.k[j], err: 100 * (iv - ex.iv[j]), py: ex.err[j] };
        });
        const worst = Math.max(...pts.map((p) => Math.abs(p.err)));
        const vsPy = Math.max(...pts.map((p) => Math.abs(p.err - p.py)));
        res = { seq, e, pts, worst, vsPy, ms, steps: job.steps.length, n: job.x.length };
      }
      if (!again) break;
    }
    failure = null;
  } catch (err) {
    failure = err;
    res = null;
  } finally {
    busy = false;
  }
  show();
}

function show() {
  const ex = data.expiries[st.e];
  $("lvExp").textContent = `${ex.name} · ${Math.round(ex.T * 365)}D`;
  const big = $("lvErr");
  if (!gpuOK || failure) {
    big.className = "big na";
    big.textContent = failure ? `GPU ERROR: ${failure.message}` : "WEBGPU NEEDED FOR THE LIVE ROUND TRIP";
  } else if (res && res.e === st.e) {
    big.className = `big ${res.worst < TOL ? "good" : "bad"}`;
    big.textContent = `${res.worst.toFixed(3)} VOL PTS`;
    $("lvGpu").textContent = `GPU: ${res.pts.length} STRIKES, ${res.steps} STEPS OF A VARIABLE-WIDTH BLUR OVER ${res.n} SPOTS, ${res.ms.toFixed(0)} MS; ` +
      `VS PYTHON ON THE SAME GRID ${res.vsPy.toFixed(4)} VOL PTS`;
  }
  const liqE = ex.err.filter((v, j) => ex.liq[j] && v != null).map(Math.abs);
  $("lvPy").textContent = `PYTHON, ALL ${liqE.length} LIQUID QUOTES OF THIS EXPIRY: WORST ${Math.max(...liqE).toFixed(3)}; ` +
    `ALL ${data.rt.n} ON THE CHAIN: WORST ${data.rt.max.toFixed(3)}, RMS ${data.rt.rms.toFixed(3)} VOL PTS`;
  const mr = data.modelRisk;
  const name = Object.keys(mr.expiries)[st.risk];
  const big1 = mr.expiries[name].rows.reduce((a, r) => (r.spread > a.spread ? r : a));
  $("lvRiskFoot").textContent = `WIDEST: ${big1.type.toUpperCase()} H ${(100 * big1.barrier).toFixed(0)}% K ${(100 * big1.strike).toFixed(0)}%, ` +
    `$${big1.spread.toFixed(2)} APART. MERTON ${(100 * mr.merton.sigma).toFixed(1)}% + ${mr.merton.lam.toFixed(2)} JUMPS/YR OF ${(100 * mr.merton.m).toFixed(0)}%`;
}

function frame() {
  if (data) {
    drawHeat();
    drawBlur();
    drawTrip();
    drawRisk();
  }
  requestAnimationFrame(frame);
}

export async function initLocalVol({ gpu = true } = {}) {
  gpuOK = gpu;
  data = (await import("./lv_data.json")).default;
  tab = lvTable(data);
  $("lvE").max = String(data.expiries.length - 1);
  $("lvE").addEventListener("input", () => {
    st.e = +$("lvE").value;
    show();
    reprice();
  });
  for (const b of document.querySelectorAll("[data-lvr]")) {
    b.addEventListener("click", () => {
      st.risk = +b.dataset.lvr;
      for (const c of document.querySelectorAll("[data-lvr]")) c.setAttribute("aria-pressed", String(c === b));
      show();
    });
  }
  show();
  requestAnimationFrame(frame);
  reprice();
}

// for the headless check: the state and the last result
// (res.seq === asked with busy false: the result answers the latest request)
window.localVol = {
  st,
  get res() { return res; },
  get asked() { return asked; },
  get busy() { return busy; },
  get failure() { return failure; },
  get data() { return data; },
  setExpiry(e) {
    st.e = e;
    $("lvE").value = String(e);
    show();
    reprice();
  },
};
