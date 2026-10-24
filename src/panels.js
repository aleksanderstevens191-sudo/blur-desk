// The small inspection panels under the main view. Each one draws a little
// wireframe of what its control does.
import { Camera, Lines, glow, label, WHITE, AMBER, RED, TEXT } from "./wire.js";
import { S_MIN, S_MAX } from "./pricer.js";

const TAU = 2 * Math.PI;

// 01: the drawing pad (flat, so the mouse maps straight onto the payoff).
export const PAD = { l: 6, r: 6, t: 8, b: 14, yMax: 60 };
export function padXY(W, H) {
  const px = (S) => PAD.l + ((S - S_MIN) / (S_MAX - S_MIN)) * (W - PAD.l - PAD.r);
  const py = (y) => PAD.t + (1 - y / PAD.yMax) * (H - PAD.t - PAD.b);
  const toS = (x) => S_MIN + ((x - PAD.l) / (W - PAD.l - PAD.r)) * (S_MAX - S_MIN);
  const toY = (y) => (1 - (y - PAD.t) / (H - PAD.t - PAD.b)) * PAD.yMax;
  return { px, py, toS, toY };
}

export function drawPad(ctx, W, H, payoff, frontMesh, meshS, barriers = []) {
  const { px, py } = padXY(W, H);
  const grid = new Path2D();
  for (let S = 40; S <= 160; S += 10) {
    for (let y = 0; y <= 60; y += 10) {
      const x = px(S);
      const yy = py(y);
      grid.moveTo(x - 1.5, yy);
      grid.lineTo(x + 1.5, yy);
      grid.moveTo(x, yy - 1.5);
      grid.lineTo(x, yy + 1.5);
    }
  }
  glow(ctx, grid, WHITE, 0.25, 1, 0);
  const frame = new Path2D();
  frame.rect(px(40), py(60), px(160) - px(40), py(0) - py(60));
  glow(ctx, frame, WHITE, 0.18, 1, 0);
  label(ctx, "40", px(40), H - 6, { align: "left", size: 9, alpha: 0.45 });
  label(ctx, "100", px(100), H - 6, { size: 9, alpha: 0.45 });
  label(ctx, "160", px(160), H - 6, { align: "right", size: 9, alpha: 0.45 });
  if (barriers.length) {
    const bl = new Path2D();
    for (const S of barriers) {
      bl.moveTo(px(S), py(60));
      bl.lineTo(px(S), py(0));
    }
    glow(ctx, bl, RED, 0.7, 1, 0.6);
  }

  if (frontMesh) {
    const fr = new Path2D();
    for (let i = 0; i < meshS.length; i++) {
      const x = px(meshS[i]);
      const y = py(Math.min(frontMesh[i], 64));
      i ? fr.lineTo(x, y) : fr.moveTo(x, y);
    }
    glow(ctx, fr, AMBER, 0.55, 1, 0.6);
  }
  const p = new Path2D();
  for (let i = 0; i < payoff.y.length; i++) {
    const x = px(S_MIN + i * ((S_MAX - S_MIN) / (payoff.y.length - 1)));
    const y = py(payoff.y[i]);
    i ? p.lineTo(x, y) : p.moveTo(x, y);
  }
  glow(ctx, p, WHITE, 1, 1.5, 1.2);
}

// 02: the blur kernel as a bell of revolution, width σ√T.
const bellCam = new Camera({ el: 0.5, dist: 3.2, target: [0, 0.4, 0], zoom: 2.5 });
export function drawKernel(ctx, W, H, t, std) {
  bellCam.az = t * 0.35;
  bellCam.setViewport(W, H);
  const w = Math.min(0.75, Math.max(0.025, (std / 1.13) * 0.75));
  const peak = Math.min(1.05, 0.11 / w + 0.05);
  const R = 1;
  const h = (r) => peak * Math.exp((-r * r) / (2 * w * w));
  const rings = new Lines(bellCam);
  const merid = new Lines(bellCam);
  const NR = 18;
  for (let k = 1; k <= NR; k++) {
    const r = (R * k) / NR;
    rings.lift();
    for (let a = 0; a <= 48; a++) rings.line(r * Math.cos((a / 48) * TAU), h(r), r * Math.sin((a / 48) * TAU));
  }
  for (let a = 0; a < 24; a++) {
    merid.lift();
    const c = Math.cos((a / 24) * TAU);
    const s = Math.sin((a / 24) * TAU);
    for (let k = 0; k <= 60; k++) {
      const r = (R * k) / 60;
      merid.line(r * c, h(r), r * s);
    }
  }
  glow(ctx, rings.path, WHITE, 0.35, 1, 0.4);
  glow(ctx, merid.path, WHITE, 0.5, 1, 0.6);
  const axis = new Lines(bellCam).seg([0, 0, 0], [0, 1.1, 0]);
  glow(ctx, axis.path, AMBER, 0.7, 1, 0.8);
}

// 03: time as a coil that winds up as expiry grows, inside a 2-year cage.
const coilCam = new Camera({ el: 0.35, dist: 3.4, target: [0, 0.5, 0], zoom: 2.4 });
export function drawCoil(ctx, W, H, t, T) {
  coilCam.az = -t * 0.25;
  coilCam.setViewport(W, H);
  const r = 0.55;
  const hMax = 1;
  const cage = new Lines(coilCam);
  for (const y of [0, hMax]) {
    cage.lift();
    for (let a = 0; a <= 40; a++) cage.line(r * Math.cos((a / 40) * TAU), y, r * Math.sin((a / 40) * TAU));
  }
  for (let a = 0; a < 8; a++) cage.seg([r * Math.cos((a / 8) * TAU), 0, r * Math.sin((a / 8) * TAU)], [r * Math.cos((a / 8) * TAU), hMax, r * Math.sin((a / 8) * TAU)]);
  for (let k = 1; k < 8; k++) {
    const y = (k / 8) * hMax;
    cage.seg([r, y, 0], [r + 0.08, y, 0]);
  }
  glow(ctx, cage.path, WHITE, 0.22, 1, 0);
  const coil = new Lines(coilCam);
  const turns = 6 * (T / 2);
  const steps = Math.max(2, Math.round(turns * 40));
  for (let k = 0; k <= steps; k++) {
    const u = k / steps;
    const a = u * turns * TAU;
    coil.line(r * 0.85 * Math.cos(a), u * (T / 2) * hMax, r * 0.85 * Math.sin(a));
  }
  glow(ctx, coil.path, AMBER, 0.9, 1.3, 1.2);
  const lid = new Lines(coilCam);
  for (let a = 0; a <= 40; a++) lid.line(r * Math.cos((a / 40) * TAU), (T / 2) * hMax, r * Math.sin((a / 40) * TAU));
  glow(ctx, lid.path, AMBER, 0.6, 1, 0.8);
}

// 04: the exercise boundary S*(τ) in its own little box, or a flat plate when European.
const exCam = new Camera({ el: 0.55, dist: 3.4, target: [0, 0.15, 0], zoom: 2.3 });
export function drawExercise(ctx, W, H, t, american, boundary, T) {
  exCam.az = -0.7 + 0.3 * Math.sin(t * 0.4);
  exCam.setViewport(W, H);
  const box = new Lines(exCam);
  const corners = [[-0.8, -0.6], [0.8, -0.6], [0.8, 0.6], [-0.8, 0.6], [-0.8, -0.6]];
  for (const [x, z] of corners) box.line(x, 0, z);
  for (let k = 1; k < 8; k++) box.seg([-0.8 + 0.2 * k, 0, -0.6], [-0.8 + 0.2 * k, 0, 0.6]);
  glow(ctx, box.path, WHITE, 0.18, 1, 0);
  if (!american) {
    const edge = new Lines(exCam).seg([-0.8, 0, -0.6], [0.8, 0, -0.6]);
    for (let k = 0; k <= 16; k++) edge.seg([-0.8 + 0.1 * k, 0, -0.6], [-0.8 + 0.1 * k, 0.25, -0.6]);
    glow(ctx, edge.path, WHITE, 0.5, 1, 0.6);
    return;
  }
  // boundary: list of [S, tau]
  const X = (S) => ((S - 100) / 60) * 0.8;
  const Z = (tau) => -0.6 + (tau / Math.max(T, 1e-9)) * 1.2;
  const curve = new Lines(exCam);
  const wall = new Lines(exCam);
  for (const [S, tau] of boundary) {
    curve.line(X(S), 0.35, Z(tau));
    wall.seg([X(S), 0, Z(tau)], [X(S), 0.35, Z(tau)]);
    wall.seg([X(S), 0, Z(tau)], [-0.8, 0, Z(tau)]);
  }
  glow(ctx, wall.path, RED, 0.35, 1, 0.5);
  glow(ctx, curve.path, RED, 1, 1.6, 1.4);
}

// 05: |price - Black-Scholes| across S, log scale.
export function drawError(ctx, W, H, errs, meshS) {
  const l = 4;
  const r = 4;
  const top = 6;
  const bot = 12;
  const lo = -9;
  const hi = -3;
  const px = (i) => l + (i / (meshS.length - 1)) * (W - l - r);
  const py = (e) => top + (1 - (Math.log10(Math.max(e, 1e-12)) - lo) / (hi - lo)) * (H - top - bot);
  const g = new Path2D();
  for (let d = lo; d <= hi; d++) {
    g.moveTo(l, py(10 ** d));
    g.lineTo(W - r, py(10 ** d));
  }
  glow(ctx, g, WHITE, 0.1, 1, 0);
  const lim = new Path2D();
  lim.moveTo(l, py(1e-4));
  lim.lineTo(W - r, py(1e-4));
  ctx.setLineDash([3, 3]);
  glow(ctx, lim, RED, 0.6, 1, 0);
  ctx.setLineDash([]);
  label(ctx, "1E-4", W - r, py(1e-4) - 6, { align: "right", size: 9, rgb: RED, alpha: 0.7 });
  label(ctx, "1E-9", l, H - 4, { align: "left", size: 9, alpha: 0.4 });
  if (!errs) return;
  const p = new Path2D();
  const bars = new Path2D();
  for (let i = 0; i < errs.length; i++) {
    const x = px(i);
    const y = py(errs[i]);
    i ? p.lineTo(x, y) : p.moveTo(x, y);
    if (i % 3 === 0) {
      bars.moveTo(x, py(1e-9));
      bars.lineTo(x, y);
    }
  }
  glow(ctx, bars, WHITE, 0.12, 1, 0);
  glow(ctx, p, WHITE, 0.9, 1.2, 1);
}

// 06: recent GPU frame times as a rotating row of wireframe bars.
const barCam = new Camera({ el: 0.4, dist: 3.4, target: [0, 0.4, 0], zoom: 2.2 });
export function drawBars(ctx, W, H, t, times) {
  barCam.az = 0.5 + 0.25 * Math.sin(t * 0.3);
  barCam.setViewport(W, H);
  const n = 16;
  const max = Math.max(16, ...times);
  const floor = new Lines(barCam);
  floor.move(-0.9, 0, -0.15).line(0.9, 0, -0.15).line(0.9, 0, 0.15).line(-0.9, 0, 0.15).line(-0.9, 0, -0.15);
  // 16 ms line = 60 fps
  const y60 = (16 / max) * 0.9;
  floor.seg([-0.9, y60, -0.15], [0.9, y60, -0.15]);
  glow(ctx, floor.path, WHITE, 0.2, 1, 0);
  const bars = new Lines(barCam);
  const recent = times.slice(-n);
  for (let k = 0; k < recent.length; k++) {
    const x0 = -0.9 + (k / n) * 1.8 + 0.02;
    const x1 = x0 + 1.8 / n - 0.04;
    const h = Math.max(0.01, (recent[k] / max) * 0.9);
    const z0 = -0.08;
    const z1 = 0.08;
    for (const z of [z0, z1]) bars.move(x0, 0, z).line(x0, h, z).line(x1, h, z).line(x1, 0, z);
    bars.seg([x0, h, z0], [x0, h, z1]).seg([x1, h, z0], [x1, h, z1]);
  }
  glow(ctx, bars.path, AMBER, 0.75, 1, 0.8);
}

// a dim note in the middle of an empty panel
function idle(ctx, W, H, text) {
  label(ctx, text, W / 2, H / 2, { size: 9, alpha: 0.4 });
}

// 09: price today against S, flat, with the barriers, the knocked region
// hatched, the closed form dashed under the blur price.
export function drawBarrierView(ctx, W, H, d) {
  const l = 6;
  const r = 6;
  const top = 8;
  const bot = 14;
  const px = (S) => l + ((S - S_MIN) / (S_MAX - S_MIN)) * (W - l - r);
  const frame = new Path2D();
  frame.rect(px(S_MIN), top, px(S_MAX) - px(S_MIN), H - top - bot);
  glow(ctx, frame, WHITE, 0.18, 1, 0);
  label(ctx, "40", px(40), H - 6, { align: "left", size: 9, alpha: 0.45 });
  label(ctx, "100", px(100), H - 6, { size: 9, alpha: 0.45 });
  label(ctx, "160", px(160), H - 6, { align: "right", size: 9, alpha: 0.45 });
  if (!d) return idle(ctx, W, H, "BARRIER OFF · PICK DOWN, UP OR DOUBLE");
  const { S, V, cf, payoff, bar } = d;
  let yMax = 1;
  for (let i = 0; i < S.length; i++) yMax = Math.max(yMax, V[i], cf ? cf[i] : 0, payoff[i]);
  yMax = Math.ceil((yMax * 1.1) / 5) * 5;
  const py = (y) => top + (1 - Math.min(y, yMax * 1.05) / yMax) * (H - top - bot);
  const curve = (Y) => {
    const p = new Path2D();
    for (let i = 0; i < S.length; i++) i ? p.lineTo(px(S[i]), py(Y[i])) : p.moveTo(px(S[i]), py(Y[i]));
    return p;
  };
  // knocked strips
  const hatch = new Path2D();
  const strips = [];
  if (bar.L) strips.push([px(S_MIN), px(bar.L)]);
  if (bar.U) strips.push([px(bar.U), px(S_MAX)]);
  const h = H - top - bot;
  for (const [xa, xb] of strips) {
    for (let x = xa - h; x < xb; x += 7) {
      const x1 = Math.max(xa, x);
      const x2 = Math.min(xb, x + h);
      if (x2 <= x1) continue;
      hatch.moveTo(x1, top + h - (x1 - x));
      hatch.lineTo(x2, top + h - (x2 - x));
    }
  }
  glow(ctx, hatch, RED, 0.2, 1, 0);
  const bl = new Path2D();
  for (const [Sb, name] of [[bar.L, "L"], [bar.U, "U"]]) {
    if (!Sb) continue;
    bl.moveTo(px(Sb), top);
    bl.lineTo(px(Sb), H - bot);
    label(ctx, name, px(Sb) + (name === "L" ? -6 : 6), top + 6, { size: 9, rgb: RED, alpha: 0.9 });
  }
  glow(ctx, bl, RED, 0.9, 1, 0.8);
  glow(ctx, curve(payoff), WHITE, 0.35, 1, 0);
  if (cf) {
    ctx.setLineDash([4, 4]);
    glow(ctx, curve(cf), WHITE, 0.8, 1, 0);
    ctx.setLineDash([]);
  }
  glow(ctx, curve(V), AMBER, 1, 1.6, 1.2);
  const [xs, ys] = [px(100), py(V[60])];
  ctx.strokeStyle = `rgba(${AMBER},0.9)`;
  ctx.beginPath();
  ctx.arc(xs, ys, 4, 0, TAU);
  ctx.stroke();
  label(ctx, cf ? "AMBER BLUR · DASHED CLOSED FORM" : "AMBER BLUR · NO CLOSED FORM", W - r - 4, top + 7, { align: "right", size: 9, alpha: 0.5 });
}

// 10: |error| at S = 100 for the three methods, log scale.
export function drawMethod(ctx, W, H, rows) {
  if (!Array.isArray(rows)) return idle(ctx, W, H, rows || "NEEDS A BARRIER");
  const l = 88;
  const r = 58;
  const lo = -9;
  const hi = 0;
  const px = (e) => l + ((Math.log10(Math.max(e, 1e-12)) - lo) / (hi - lo)) * (W - l - r);
  const rowH = (H - 16) / rows.length;
  const g = new Path2D();
  for (let k = lo; k <= hi; k += 3) {
    g.moveTo(px(10 ** k), 2);
    g.lineTo(px(10 ** k), H - 12);
    label(ctx, `1E${k}`, px(10 ** k), H - 5, { size: 8, alpha: 0.4 });
  }
  glow(ctx, g, WHITE, 0.1, 1, 0);
  const lim = new Path2D();
  lim.moveTo(px(1e-4), 2);
  lim.lineTo(px(1e-4), H - 12);
  ctx.setLineDash([3, 3]);
  glow(ctx, lim, RED, 0.6, 1, 0);
  ctx.setLineDash([]);
  rows.forEach((row, k) => {
    const y = 4 + k * rowH + rowH / 2;
    const rgb = row.sel ? AMBER : WHITE;
    label(ctx, row.name, 4, y, { align: "left", size: 9, rgb, alpha: row.sel ? 1 : 0.6 });
    if (row.err === null) {
      label(ctx, "REFERENCE", l + 4, y, { align: "left", size: 9, alpha: 0.5 });
      return;
    }
    const bar = new Path2D();
    const x1 = Math.max(px(row.err), l + 2);
    bar.rect(l, y - rowH * 0.28, x1 - l, rowH * 0.56);
    glow(ctx, bar, row.err < 1e-4 ? "78,240,160" : RED, row.sel ? 1 : 0.55, 1, row.sel ? 0.8 : 0);
    label(ctx, row.err.toExponential(1).toUpperCase(), W - 4, y, { align: "right", size: 9, rgb, alpha: row.sel ? 1 : 0.6 });
  });
}

// 11: the hall of mirrors. The knock-out payoff is the payoff minus its mirror
// images across the barriers. Each image is the one before it flipped about
// the mirror's foot point, then weighted by e^{a (x_m - x)} (a = 0: plain mirror).
// Images appear one at a time; then the kernel sweeps the corridor and the
// price comes out.
const SWING = 0.8;
const HOLD = 1.4;
const SWEEP = 2.6;
const ease = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * u));

export function mirrorCells(m) {
  const { lo, hi, a, f } = m;
  const XW = 1;
  const x0 = m.x0;
  if (lo !== null && hi !== null) {
    const w = hi - lo;
    const cMin = Math.max(-m.maxCells, Math.floor((x0 - XW - lo) / w));
    const cMax = Math.min(m.maxCells, Math.floor((x0 + XW - lo) / w));
    const g = (c, x) => {
      if (c % 2 === 0) {
        const n = -c / 2;
        return Math.exp(a * n * w) * f(x + 2 * n * w);
      }
      const n = (c + 1) / 2;
      return -Math.exp(a * (lo + n * w - x)) * f(2 * lo + 2 * n * w - x);
    };
    const order = [];
    for (let k = 1; k <= Math.max(cMax, -cMin); k++) {
      if (k <= cMax) order.push(k);
      if (-k >= cMin) order.push(-k);
    }
    const cells = order.map((c) => ({
      c, M: Math.abs(c), x0: lo + c * w, x1: lo + (c + 1) * w, mirror: c > 0 ? lo + c * w : lo + (c + 1) * w, src: c > 0 ? c - 1 : c + 1,
    }));
    return { XW, home: { c: 0, x0: lo, x1: hi }, cells, g, mirrors: Array.from({ length: cMax - cMin + 2 }, (_, k) => lo + (cMin + k) * w) };
  }
  const h = lo ?? hi;
  const down = lo !== null;
  const g = (c, x) => (c === 0 ? f(x) : -Math.exp(a * (h - x)) * f(2 * h - x));
  const far = down ? [h, x0 + XW + 1] : [x0 - XW - 1, h];
  const img = down ? [x0 - XW - 1, h] : [h, x0 + XW + 1];
  return { XW, home: { c: 0, x0: far[0], x1: far[1] }, cells: [{ c: down ? -1 : 1, M: 1, x0: img[0], x1: img[1], mirror: h, src: 0 }], g, mirrors: [h] };
}

export function drawMirrors(ctx, W, H, t, m) {
  if (!m || typeof m === "string") return idle(ctx, W, H, m || "BARRIER OFF · THE MIRRORS APPEAR WITH A BARRIER");
  const { XW, home, cells, g, mirrors } = mirrorCells(m);
  const x0 = m.x0;
  const chart = W >= 640 ? 190 : 0; // error vs number of images, on the right
  const l = 8;
  const r = 8 + (chart ? chart + 20 : 0);
  const top = chart ? 22 : 34;
  const bot = 18;
  const px = (x) => l + ((x - (x0 - XW)) / (2 * XW)) * (W - l - r);
  const mid = top + (H - top - bot) / 2;
  // vertical scale: the payoff over the plotted S range, with room for the images
  let yMax = 1;
  for (let k = 0; k <= 60; k++) {
    const x = Math.log(S_MIN) + ((Math.log(S_MAX) - Math.log(S_MIN)) * k) / 60;
    if (x >= home.x0 && x <= home.x1) yMax = Math.max(yMax, Math.abs(g(0, x)));
  }
  yMax *= 1.6;
  const half = (H - top - bot) / 2;
  const py = (y) => mid - (Math.max(-6 * yMax, Math.min(6 * yMax, y)) / yMax) * half; // clipped below
  const NPT = 90;
  const sample = (c, xa, xb) => {
    const a = Math.max(xa, x0 - XW - 0.02);
    const b = Math.min(xb, x0 + XW + 0.02);
    if (b <= a) return [];
    return Array.from({ length: NPT + 1 }, (_, k) => {
      const x = a + ((b - a) * k) / NPT;
      return [x, g(c, x)];
    });
  };
  const poly = (pts) => {
    const p = new Path2D();
    pts.forEach(([X, Y], k) => (k ? p.lineTo(X, Y) : p.moveTo(X, Y)));
    return p;
  };

  // timeline
  const per = SWING + (m.a !== 0 ? 0.5 : 0.15);
  const cycle = cells.length * per + SWEEP + HOLD;
  const tt = t % cycle;
  const shown = Math.min(cells.length, Math.floor(tt / per));
  const active = tt < cells.length * per ? cells[shown] : null;
  const u = active ? (tt - shown * per) / per : 1;
  const sweep = tt > cells.length * per ? Math.min(1, (tt - cells.length * per) / SWEEP) : 0;

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, top - 6, W - r + 4, H - top - bot + 12);
  ctx.clip();
  // axis and mirrors
  const ax = new Path2D();
  ax.moveTo(l, mid);
  ax.lineTo(W - r, mid);
  glow(ctx, ax, WHITE, 0.25, 1, 0);
  const mir = new Path2D();
  const real = new Path2D();
  for (const x of mirrors) {
    if (x < x0 - XW || x > x0 + XW) continue;
    const isReal = Math.abs(x - (m.lo ?? NaN)) < 1e-9 || Math.abs(x - (m.hi ?? NaN)) < 1e-9;
    (isReal ? real : mir).moveTo(px(x), top - 4);
    (isReal ? real : mir).lineTo(px(x), H - bot + 4);
  }
  ctx.setLineDash([3, 4]);
  glow(ctx, mir, RED, 0.35, 1, 0);
  ctx.setLineDash([]);
  glow(ctx, real, RED, 0.9, 1.2, 0.8);

  const dimNow = sweep > 0 ? 0.45 : 1;
  // the payoff itself (live corridor)
  const homePts = sample(0, home.x0, home.x1).map(([x, y]) => [px(x), py(y)]);
  glow(ctx, poly(homePts), WHITE, 0.95 * dimNow, 1.6, 1.2);
  // images already in place
  for (let k = 0; k < shown; k++) {
    const cl = cells[k];
    glow(ctx, poly(sample(cl.c, cl.x0, cl.x1).map(([x, y]) => [px(x), py(y)])), AMBER, 0.75 * dimNow, 1.2, 0.8);
  }
  // the one being placed: swing about the foot point, then take on its weight
  if (active) {
    const swing = ease(u * (per / SWING));
    const morph = per > SWING ? ease((u * per - SWING) / (per - SWING)) : 1;
    const th = (active.c > 0 ? -1 : 1) * Math.PI * swing;
    const fx = px(active.mirror);
    const pts = sample(active.c, active.x0, active.x1).map(([x]) => {
      const xs = 2 * active.mirror - x; // where this point comes from
      const ys = g(active.src, xs);
      const wgt = 1 + morph * (Math.exp(m.a * (active.mirror - x)) - 1);
      // flip about the foot point (fx, mid), in screen space, y up
      const du = px(xs) - fx;
      const dv = mid - py(ys);
      const uu = du * Math.cos(th) - dv * Math.sin(th);
      const vv = du * Math.sin(th) + dv * Math.cos(th);
      const ySwing = mid - vv; // screen y after the swing
      const yW = py(-ys * wgt); // after the weight
      return [fx + uu, swing < 1 ? ySwing : yW];
    });
    glow(ctx, poly(pts), AMBER, 1, 1.6, 1.4);
    ctx.globalCompositeOperation = "lighter";
    ctx.strokeStyle = `rgba(${AMBER},0.9)`;
    ctx.beginPath();
    ctx.arc(fx, mid, 4 + 2 * Math.sin(t * 8), 0, TAU);
    ctx.stroke();
    ctx.globalCompositeOperation = "source-over";
  }
  // the blur: a kernel sweeping the corridor, and the price behind it
  if (sweep > 0 && m.front) {
    const xa = Math.max(x0 - XW, Math.log(S_MIN / 100) + x0);
    const xb = Math.min(x0 + XW, Math.log(S_MAX / 100) + x0);
    const xc = xa + (xb - xa) * ease(sweep);
    const bell = [];
    const std = Math.max(m.std, 0.01);
    for (let k = 0; k <= 80; k++) {
      const x = xc - 4 * std + (8 * std * k) / 80;
      bell.push([px(x), mid - 0.55 * half * Math.exp((-(x - xc) * (x - xc)) / (2 * std * std))]);
    }
    glow(ctx, poly(bell), WHITE, 0.6, 1, 0.6);
    const pr = [];
    for (let i = 0; i < m.front.S.length; i++) {
      const x = Math.log(m.front.S[i] / 100) + x0;
      if (x > xc) break;
      pr.push([px(x), py(m.front.V[i])]);
    }
    if (pr.length > 1) glow(ctx, poly(pr), AMBER, 1, 2.2, 1.6);
  }
  ctx.restore();

  // labels: barrier levels, and the error with this many images
  for (const x of mirrors) {
    if (x < x0 - XW || x > x0 + XW) continue;
    label(ctx, (100 * Math.exp(x - x0)).toFixed(0), px(x), H - 7, { size: 8, rgb: RED, alpha: 0.6 });
  }
  // images counted as complete rings: every cell with |c| <= M is in place.
  // The price itself always uses every image the grid holds.
  const placed = active ? shown : cells.length;
  let Mnow = 0;
  while (cells.some((cl) => cl.M === Mnow + 1) && cells.every((cl, k) => cl.M > Mnow + 1 || k < placed)) Mnow++;
  if (sweep > 0) Mnow = m.errs.length - 1;
  const e = m.errs[Math.min(Mnow, m.errs.length - 1)];
  const eTxt = e === undefined ? "" : e.err === null ? "(REFERENCE)" : `ERROR AT S=100 ${e.err.toExponential(1).toUpperCase()}`;
  const double = m.lo !== null && m.hi !== null;
  const count = sweep > 0 ? "ALL IMAGES" : `IMAGES ${Mnow}${double ? " EACH SIDE" : ""}`;
  label(ctx, `${count}   ${eTxt}`, l, 8, { align: "left", size: 10, rgb: AMBER, alpha: 0.95 });
  if (chart) drawImageErrors(ctx, W - chart - 8, top, chart, H - top - bot, m.errs, Mnow, double);
  const what = sweep > 0 ? "BLUR THE IMAGE-TILED PAYOFF → PRICE" : active ? `FLIP ABOUT ${(100 * Math.exp(active.mirror - x0)).toFixed(1)}${m.a !== 0 ? ", THEN WEIGHT e^a(xm−x)" : ""}` : "";
  if (chart) label(ctx, what, W - r, 8, { align: "right", size: 9, alpha: 0.6 });
  else label(ctx, what, l, 21, { align: "left", size: 9, alpha: 0.6 });
}

// |price at S=100 - closed form| against the number of images, log scale
function drawImageErrors(ctx, x, y, w, h, errs, Mnow, double) {
  const lo = -9;
  const hi = 1;
  const top = y + 14;
  const bot = y + h - 12;
  const py = (e) => bot - ((Math.log10(Math.max(e, 1e-12)) - lo) / (hi - lo)) * (bot - top);
  label(ctx, double ? "ERROR vs IMAGES EACH SIDE" : "ERROR vs IMAGES", x, y + 2, { align: "left", size: 9, alpha: 0.6 });
  const g = new Path2D();
  for (let k = lo; k <= hi; k += 2) {
    g.moveTo(x, py(10 ** k));
    g.lineTo(x + w, py(10 ** k));
  }
  glow(ctx, g, WHITE, 0.08, 1, 0);
  const lim = new Path2D();
  lim.moveTo(x, py(1e-4));
  lim.lineTo(x + w, py(1e-4));
  ctx.setLineDash([3, 3]);
  glow(ctx, lim, RED, 0.5, 1, 0);
  ctx.setLineDash([]);
  label(ctx, "1E-4", x + w, py(1e-4) - 6, { align: "right", size: 8, rgb: RED, alpha: 0.7 });
  const n = errs.length;
  const bw = w / n;
  errs.forEach(({ M, err }) => {
    const bx = x + M * bw + bw * 0.18;
    const cur = M === Mnow;
    if (err === null) {
      label(ctx, "REF", bx + bw * 0.32, bot - 6, { size: 8, alpha: 0.5 });
    } else {
      const p = new Path2D();
      p.rect(bx, py(err), bw * 0.64, bot - py(err));
      glow(ctx, p, cur ? AMBER : WHITE, cur ? 1 : 0.3, 1, cur ? 0.8 : 0);
    }
    if (n <= 4 || M % 4 === 0 || cur) label(ctx, String(M), bx + bw * 0.32, bot + 7, { size: 8, rgb: cur ? AMBER : WHITE, alpha: cur ? 1 : 0.45 });
  });
}

// round tick values between lo and hi, about n of them
function ticks(lo, hi, n = 4) {
  const raw = (hi - lo) / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? 10 * p;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}

// polyline through (xs[i], ys[i]), broken where y is not finite
function polyline(xs, ys, px, py, i0 = 0, i1 = xs.length - 1) {
  const p = new Path2D();
  let pen = false;
  for (let i = i0; i <= i1; i++) {
    if (!Number.isFinite(ys[i])) {
      pen = false;
      continue;
    }
    pen ? p.lineTo(px(xs[i]), py(ys[i])) : p.moveTo(px(xs[i]), py(ys[i]));
    pen = true;
  }
  return p;
}

// 07: the kernel's density on a log scale (weights / dx) against a Gaussian
// with the same mean and variance, which is a parabola there. Fat tails are
// where the kernel lies above it.
export function drawLogKernel(ctx, W, H, d) {
  if (!d || typeof d === "string") return idle(ctx, W, H, d || "—");
  const { y, w, dx, mean, sd } = d;
  const l = 6;
  const r = 6;
  const top = 16;
  const bot = 14;
  const x0 = mean - 6 * sd;
  const x1 = mean + 6 * sd;
  let peak = 0;
  for (let j = 0; j < w.length; j++) peak = Math.max(peak, w[j] / dx);
  const hi = Math.log10(peak) + 0.3;
  const lo = hi - 8.6;
  const px = (x) => l + ((x - x0) / (x1 - x0)) * (W - l - r);
  const py = (v) => top + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - top - bot);
  const grid = new Path2D();
  for (let k = Math.ceil(lo); k <= hi; k += 2) {
    grid.moveTo(l, py(k));
    grid.lineTo(W - r, py(k));
  }
  const sdTicks = new Path2D();
  for (const s of [-3, 0, 3]) {
    sdTicks.moveTo(px(mean + s * sd), H - bot);
    sdTicks.lineTo(px(mean + s * sd), H - bot + 3);
  }
  glow(ctx, grid, WHITE, 0.1, 1, 0);
  glow(ctx, sdTicks, WHITE, 0.4, 1, 0);
  label(ctx, "−3 SD", px(mean - 3 * sd), H - 5, { size: 8, alpha: 0.45 });
  label(ctx, "MEAN", px(mean), H - 5, { size: 8, alpha: 0.45 });
  label(ctx, "+3 SD", px(mean + 3 * sd), H - 5, { size: 8, alpha: 0.45 });
  // the Gaussian with the same mean and sd
  const g = new Path2D();
  for (let k = 0; k <= 160; k++) {
    const x = x0 + ((x1 - x0) * k) / 160;
    const z = (x - mean) / sd;
    const v = -0.5 * z * z * Math.LOG10E - Math.log10(sd * Math.sqrt(2 * Math.PI));
    k ? g.lineTo(px(x), py(v)) : g.moveTo(px(x), py(v));
  }
  ctx.setLineDash([4, 3]);
  glow(ctx, g, WHITE, 0.5, 1, 0);
  ctx.setLineDash([]);
  // the kernel, at most ~2 points per pixel
  const J = (w.length - 1) / 2;
  const j0 = Math.max(0, Math.floor(x0 / dx) + J);
  const j1 = Math.min(w.length - 1, Math.ceil(x1 / dx) + J);
  const stride = Math.max(1, Math.floor((j1 - j0) / (2 * W)));
  const p = new Path2D();
  let pen = false;
  for (let j = j0; j <= j1; j += stride) {
    const v = w[j] > 0 ? Math.log10(w[j] / dx) : -Infinity;
    if (v < lo) {
      pen = false; // below the plotted range
      continue;
    }
    pen ? p.lineTo(px(y[j]), py(v)) : p.moveTo(px(y[j]), py(v));
    pen = true;
  }
  glow(ctx, p, AMBER, 1, 1.4, 1.1);
  label(ctx, "AMBER KERNEL · DASHED GAUSSIAN, SAME VARIANCE", l, 6, { align: "left", size: 8, alpha: 0.5 });
}

// 08: Black-Scholes implied vol against strike from the GPU prices. Live: the
// selected kernel at the expiry slider's T, against the flat line a Gaussian
// kernel would give. SPY: the market's bid-ask vols at one expiry, the
// calibrated Merton curve dashed, the live kernel solid.
export function drawSmile(ctx, W, H, d) {
  if (!d || typeof d === "string") return idle(ctx, W, H, d || "—");
  const l = 34 * TEXT.scale;
  const r = 8;
  const top = 16;
  const bot = 16;
  const spy = d.mode === "spy";
  const s = d.live;
  // x: log-moneyness k; labelled as strike / forward, or as the SPY strike in dollars
  const kLo = spy ? d.kLo : s.k[0];
  const kHi = spy ? d.kHi : s.k.at(-1);
  const inRange = (sm) => {
    let a = 0;
    while (a < sm.k.length - 1 && sm.k[a] < kLo) a++;
    let b = sm.k.length - 1;
    while (b > 0 && sm.k[b] > kHi) b--;
    return [a, b];
  };
  const [a, b] = inRange(s);
  let vLo = Infinity;
  let vHi = -Infinity;
  const take = (v) => {
    if (Number.isFinite(v)) {
      vLo = Math.min(vLo, v);
      vHi = Math.max(vHi, v);
    }
  };
  for (let i = a; i <= b; i++) take(s.iv[i]);
  if (spy) {
    d.market.ivBid.forEach(take);
    d.market.ivAsk.forEach(take);
    const [fa, fb] = inRange(d.fit);
    for (let i = fa; i <= fb; i++) take(d.fit.iv[i]);
  }
  if (!Number.isFinite(vLo)) return idle(ctx, W, H, "NO PRICE HERE CARRIES A VOLATILITY");
  const pad = Math.max(0.004, 0.12 * (vHi - vLo));
  vLo -= pad;
  vHi += pad;
  const px = (k) => l + ((k - kLo) / (kHi - kLo)) * (W - l - r);
  const py = (v) => top + (1 - (v - vLo) / (vHi - vLo)) * (H - top - bot);
  const frame = new Path2D();
  frame.rect(l, top, W - l - r, H - top - bot);
  glow(ctx, frame, WHITE, 0.15, 1, 0);
  const grid = new Path2D();
  for (const v of ticks(vLo * 100, vHi * 100, 4)) {
    grid.moveTo(l, py(v / 100));
    grid.lineTo(W - r, py(v / 100));
    label(ctx, `${v}%`, l - 4, py(v / 100), { align: "right", size: 8, alpha: 0.45 });
  }
  const xt = spy ? ticks(d.F * Math.exp(kLo), d.F * Math.exp(kHi), 6) : ticks(100 * Math.exp(kLo), 100 * Math.exp(kHi), 6);
  for (const v of xt) {
    const x = px(Math.log(v / (spy ? d.F : 100)));
    grid.moveTo(x, top);
    grid.lineTo(x, H - bot);
    label(ctx, spy ? `$${v}` : `${v}%`, x, H - 6, { size: 8, alpha: 0.45 });
  }
  glow(ctx, grid, WHITE, 0.07, 1, 0);
  label(ctx, spy ? "SPY STRIKE" : "STRIKE / FORWARD", W - r - 2, top + 7, { align: "right", size: 8, alpha: 0.3 });

  if (spy) {
    const m = d.market;
    const bars = new Path2D();
    for (let i = 0; i < m.k.length; i++) {
      const x = px(m.k[i]);
      bars.moveTo(x, py(m.ivBid[i]));
      bars.lineTo(x, py(m.ivAsk[i]));
      for (const v of [m.ivBid[i], m.ivAsk[i]]) {
        bars.moveTo(x - 3, py(v));
        bars.lineTo(x + 3, py(v));
      }
    }
    glow(ctx, bars, AMBER, 0.85, 1, 0.6);
    ctx.fillStyle = `rgba(${AMBER},0.95)`;
    for (let i = 0; i < m.k.length; i++) {
      ctx.beginPath();
      ctx.arc(px(m.k[i]), py(m.iv[i]), 2.4, 0, TAU);
      ctx.fill();
    }
    const [fa, fb] = inRange(d.fit);
    ctx.setLineDash([5, 4]);
    glow(ctx, polyline(d.fit.k, d.fit.iv, px, py, fa, fb), WHITE, 0.6, 1, 0);
    ctx.setLineDash([]);
    label(ctx, "AMBER SPY BID–ASK · DASHED MERTON FIT · WHITE LIVE KERNEL", l, 6, { align: "left", size: 8, alpha: 0.5 });
  } else {
    const flat = new Path2D();
    flat.moveTo(l, py(d.atm));
    flat.lineTo(W - r, py(d.atm));
    ctx.setLineDash([4, 4]);
    glow(ctx, flat, WHITE, 0.4, 1, 0);
    ctx.setLineDash([]);
    label(ctx, "WHITE: GPU PRICES READ AS IMPLIED VOLS · DASHED: FLAT, WHAT A GAUSSIAN GIVES", l, 6, { align: "left", size: 8, alpha: 0.5 });
  }
  glow(ctx, polyline(s.k, s.iv, px, py, a, b), WHITE, 1, 1.5, 1.2);
}

// 12: the whole SPY chain. One row per expiry (nearest at the top), one cell per
// out-of-the-money quote at its strike in standard deviations from the forward
// (ln(K / F) / σ√T, σ the expiry's flat vol), coloured by model minus market vol:
// amber where the model is above the market, blue below, dark where they agree.
const BLUE = "96,168,255";
export const CHAIN_SCALE = 5; // vol points at full colour
export function chainLayout(W, H, d) {
  const l = 34;
  const r = 8;
  const top = 18;
  const bot = 16;
  const rh = (H - top - bot) / d.rows.length;
  const px = (x) => l + ((x - d.xLo) / (d.xHi - d.xLo)) * (W - l - r);
  return { l, r, top, bot, rh, px, toX: (x) => d.xLo + ((x - l) / (W - l - r)) * (d.xHi - d.xLo) };
}

export function drawChain(ctx, W, H, d, hover) {
  if (!d || typeof d === "string") return idle(ctx, W, H, d || "—");
  const { l, r, top, bot, rh, px } = chainLayout(W, H, d);
  const frame = new Path2D();
  frame.rect(l, top, W - l - r, H - top - bot);
  glow(ctx, frame, WHITE, 0.15, 1, 0);
  const grid = new Path2D();
  for (const x of ticks(d.xLo, d.xHi, 8)) {
    grid.moveTo(px(x), top);
    grid.lineTo(px(x), H - bot);
    label(ctx, x === 0 ? "F" : `${x > 0 ? "+" : "−"}${Math.abs(x)}σ`, px(x), H - 6, { size: 8, alpha: 0.45 });
  }
  glow(ctx, grid, WHITE, 0.07, 1, 0);
  // rows: label as many as fit, days to expiry
  let lastY = -Infinity;
  d.rows.forEach((row, i) => {
    const y = top + (i + 0.5) * rh;
    if (y - lastY < 11 && !(hover && hover.row === i)) return;
    lastY = y;
    const days = row.T * 365;
    const text = days < 365 ? `${Math.round(days)}D` : `${(days / 365).toFixed(1)}Y`;
    label(ctx, text, l - 4, y, { align: "right", size: 8, alpha: hover && hover.row === i ? 1 : 0.45 });
  });

  ctx.globalCompositeOperation = "source-over";
  const unreached = new Path2D();
  d.rows.forEach((row, i) => {
    const y = top + i * rh;
    for (const c of row.cells) {
      const x0 = px(c.x0);
      const w = Math.max(px(c.x1) - x0 - 0.5, 1);
      if (!d.priced) {
        ctx.fillStyle = `rgba(${WHITE},0.22)`;
        ctx.fillRect(px(c.x) - 0.5, y + 0.25 * rh, 1, 0.5 * rh);
        continue;
      }
      ctx.fillStyle = `rgba(${WHITE},0.05)`;
      ctx.fillRect(x0, y + 0.5, w, rh - 1);
      if (!Number.isFinite(c.err)) {
        unreached.rect(x0 + 0.5, y + 1, w - 1, rh - 2);
        continue;
      }
      const a = Math.min(Math.abs(c.err) / CHAIN_SCALE, 1) ** 0.8;
      ctx.fillStyle = `rgba(${c.err > 0 ? AMBER : BLUE},${(0.95 * a).toFixed(3)})`;
      ctx.fillRect(x0, y + 0.5, w, rh - 1);
    }
  });
  glow(ctx, unreached, WHITE, 0.5, 1, 0);
  const fwd = new Path2D();
  fwd.moveTo(px(0), top);
  fwd.lineTo(px(0), H - bot);
  ctx.setLineDash([3, 3]);
  glow(ctx, fwd, WHITE, 0.35, 1, 0);
  ctx.setLineDash([]);
  if (hover) {
    const c = d.rows[hover.row].cells[hover.i];
    const box = new Path2D();
    const x0 = px(c.x0);
    box.rect(x0 - 1, top + hover.row * rh - 1, Math.max(px(c.x1) - x0, 2) + 2, rh + 2);
    glow(ctx, box, WHITE, 1, 1, 1);
  }

  // legend: blue, dark, amber across ±CHAIN_SCALE
  const lw = Math.min(120, 0.25 * W);
  const lx = W - r - lw - 64;
  const name = `${d.model === "flat" ? "FLAT VOL" : "MERTON"} − MARKET VOL`;
  let cap = d.priced ? `${name} · PUTS LEFT OF F, CALLS RIGHT` : "EVERY OUT-OF-THE-MONEY QUOTE · PRICE THE CHAIN TO COLOUR IT";
  ctx.font = `500 8px ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace`;
  if (d.priced && l + ctx.measureText(cap).width > lx - 24) cap = name; // narrow: leave room for the legend
  label(ctx, cap, l, 7, { align: "left", size: 8, alpha: 0.5 });
  if (d.priced) {
    ctx.globalCompositeOperation = "source-over";
    for (let j = 0; j < lw; j++) {
      const e = ((j + 0.5) / lw - 0.5) * 2 * CHAIN_SCALE;
      const a = Math.min(Math.abs(e) / CHAIN_SCALE, 1) ** 0.8;
      ctx.fillStyle = `rgba(${WHITE},0.05)`;
      ctx.fillRect(lx + j, 4, 1, 7);
      ctx.fillStyle = `rgba(${e > 0 ? AMBER : BLUE},${(0.95 * a).toFixed(3)})`;
      ctx.fillRect(lx + j, 4, 1, 7);
    }
    label(ctx, `−${CHAIN_SCALE}`, lx - 4, 8, { align: "right", size: 8, alpha: 0.5 });
    label(ctx, `+${CHAIN_SCALE} VOL PTS`, lx + lw + 4, 8, { align: "left", size: 8, alpha: 0.5 });
  }
}

// 13: the market's odds. The risk-neutral density of SPY at one expiry, edge-
// detected from the SPY fit's GPU prices (white), against what one flat vol gives
// (dashed). The tail below a 10% fall is filled amber; marks at today's price,
// the fall and the forward.
export function drawOdds(ctx, W, H, d) {
  if (!d || typeof d === "string") return idle(ctx, W, H, d || "—");
  const { x, dens, flatDens: flat, lo, hi, spot, down, forward } = d;
  const l = 8;
  const r = 8;
  const top = d.top ?? 18;
  const bot = 16;
  let peak = 0;
  for (let i = 0; i < dens.length; i++) peak = Math.max(peak, dens[i], flat[i]);
  const px = (S) => l + ((S - lo) / (hi - lo)) * (W - l - r);
  const py = (v) => top + (1 - v / (1.08 * peak)) * (H - top - bot);
  const frame = new Path2D();
  frame.moveTo(l, H - bot);
  frame.lineTo(W - r, H - bot);
  glow(ctx, frame, WHITE, 0.3, 1, 0);
  const grid = new Path2D();
  for (const v of ticks(lo, hi, Math.max(4, Math.min(10, Math.floor(W / 90))))) {
    grid.moveTo(px(v), top);
    grid.lineTo(px(v), H - bot);
    label(ctx, `$${v}`, px(v), H - 6, { size: d.tick ?? 8, alpha: 0.45 });
  }
  glow(ctx, grid, WHITE, 0.06, 1, 0);

  // the tail past the fall, filled under the market's curve
  ctx.globalCompositeOperation = "lighter";
  const tail = new Path2D();
  tail.moveTo(px(x[0]), py(0));
  for (let i = 0; i < x.length && x[i] <= down; i++) tail.lineTo(px(x[i]), py(dens[i]));
  tail.lineTo(px(down), py(0));
  tail.closePath();
  ctx.fillStyle = `rgba(${AMBER},0.28)`;
  ctx.fill(tail);
  ctx.globalCompositeOperation = "source-over";

  const mark = (S, alpha, dash) => {
    const p = new Path2D();
    p.moveTo(px(S), top);
    p.lineTo(px(S), H - bot);
    ctx.setLineDash(dash);
    glow(ctx, p, WHITE, alpha, 1, 0);
    ctx.setLineDash([]);
  };
  mark(spot, 0.5, [2, 3]);
  mark(forward, 0.25, [1, 3]);
  const dn = new Path2D();
  dn.moveTo(px(down), top);
  dn.lineTo(px(down), H - bot);
  glow(ctx, dn, AMBER, 0.9, 1, 0.6);
  const sz = d.tick ?? 8;
  label(ctx, `TODAY $${spot.toFixed(2)}`, px(spot) + 4, H - bot - 8, { align: "left", size: sz, alpha: 0.6 });
  label(ctx, `−10% $${down.toFixed(2)}`, px(down) - 4, H - bot - 8, { align: "right", size: sz, rgb: AMBER, alpha: 0.95 });

  ctx.setLineDash([4, 4]);
  glow(ctx, polyline(x, flat, px, py), WHITE, 0.45, 1, 0);
  ctx.setLineDash([]);
  glow(ctx, polyline(x, dens, px, py), WHITE, 1, 1.5, 1.2);
  label(ctx, "WHITE: MARKET-IMPLIED (RISK-NEUTRAL) DENSITY · DASHED: ONE FLAT VOL", l, 7, { align: "left", size: sz, alpha: 0.5 });
}
