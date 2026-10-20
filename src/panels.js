// The small inspection panels under the main view. Each one draws a little
// wireframe of what its control does.
import { Camera, Lines, glow, label, WHITE, AMBER, RED } from "./wire.js";
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

export function drawPad(ctx, W, H, payoff, frontMesh, meshS) {
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
