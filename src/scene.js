// The main 3D view: price as a surface over (stock price, time to expiry).
// The payoff sits on the back edge (T = 0); each slice toward the viewer is
// the same payoff blurred for longer; the front edge is today's price.
import { Lines, glow, label, WHITE, AMBER, RED } from "./wire.js";

export const MESH_S = Float64Array.from({ length: 121 }, (_, i) => 40 + i);
const T_AXIS = 2;

const X = (S) => (S - 100) / 60;
const Z = (tau) => (tau / T_AXIS) * 1.6 - 0.8;

export function drawScene(ctx, cam, v, W, H, time) {
  const Y = (p) => (p / v.yMax) * 0.95;
  const P = (i, tau, val) => [X(MESH_S[i]), Y(val), Z(tau)];
  const n = MESH_S.length;
  const small = W < 520;

  // floor grid
  const floor = new Lines(cam);
  for (let S = 40; S <= 160; S += 20) floor.seg([X(S), 0, Z(0)], [X(S), 0, Z(T_AXIS)]);
  for (let tau = 0; tau <= T_AXIS + 1e-9; tau += 0.25) floor.seg([X(40), 0, Z(tau)], [X(160), 0, Z(tau)]);
  glow(ctx, floor.path, WHITE, 0.09, 1, 0);

  // cage
  const top = 0.95;
  const cage = new Lines(cam);
  const xs = [X(40), X(160)];
  const zs = [Z(0), Z(T_AXIS)];
  for (const x of xs) for (const z of zs) cage.seg([x, 0, z], [x, top, z]);
  for (const y of [0, top]) {
    for (const x of xs) cage.seg([x, y, zs[0]], [x, y, zs[1]]);
    for (const z of zs) cage.seg([xs[0], y, z], [xs[1], y, z]);
  }
  // price ticks on the back-left edge
  for (let p = 0; p <= v.yMax + 1e-9; p += v.yMax / 4) cage.seg([xs[0], Y(p), zs[0]], [xs[0] - 0.04, Y(p), zs[0]]);
  glow(ctx, cage.path, WHITE, 0.32, 1, 0.6);

  // axis labels
  const at = (x, y, z) => cam.project(x, y, z);
  const fs = small ? 9 : 10;
  for (let S = 40; S <= 160; S += small ? 40 : 20) {
    const [sx, sy] = at(X(S), 0, Z(T_AXIS) + 0.08);
    label(ctx, String(S), sx, sy, { size: fs, alpha: 0.55 });
  }
  for (let tau = 0; tau <= T_AXIS; tau += 0.5) {
    const [sx, sy] = at(X(160) + 0.16, 0, Z(tau));
    label(ctx, `${tau.toFixed(1)}Y`, sx, sy, { size: fs, alpha: 0.55 });
  }
  for (let p = 0; p <= v.yMax + 1e-9; p += v.yMax / 4) {
    const [sx, sy] = at(xs[0] - 0.1, Y(p), zs[0]);
    label(ctx, p.toFixed(0), sx, sy, { size: fs, alpha: 0.5 });
  }
  {
    const [sx, sy] = at(0, 0, Z(T_AXIS) + 0.2);
    label(ctx, "STOCK PRICE", sx, sy, { size: fs, alpha: 0.4 });
    const [tx, ty] = at(X(160) + 0.5, 0, Z(1.25));
    label(ctx, "TIME TO EXPIRY", tx, ty, { size: fs, alpha: 0.4 });
  }

  // barriers: a wall at each one from expiry to today, the knocked region hatched on the floor
  if (v.barrier) drawBarriers(ctx, cam, v, top, fs);

  // the blur front: a plane at the current expiry
  const zT = Z(v.T);
  const plane = new Lines(cam);
  plane.move(xs[0], 0, zT).line(xs[1], 0, zT).line(xs[1], top, zT).line(xs[0], top, zT).line(xs[0], 0, zT);
  for (let k = 1; k < 6; k++) plane.seg([xs[0], (top * k) / 6, zT], [xs[0] + 0.05, (top * k) / 6, zT]);
  glow(ctx, plane.path, AMBER, 0.22, 1, 0.5);

  const slices = [{ tau: 0, V: v.payoff, ex: v.american ? v.payoffEx : null }, ...v.slices.filter((s) => s.tau < v.T - 1e-9)];
  if (v.front && v.T > 0) slices.push(v.front);

  // curtains: hatching from today's price and from the payoff down to the floor
  if (v.front) {
    const cur = new Lines(cam);
    for (let i = 0; i < n; i += 2) cur.seg(P(i, v.T, v.front.V[i]), [X(MESH_S[i]), 0, zT]);
    glow(ctx, cur.path, AMBER, 0.11, 1, 0);
  }
  const back = new Lines(cam);
  for (let i = 0; i < n; i += 3) back.seg(P(i, 0, v.payoff[i]), [X(MESH_S[i]), 0, Z(0)]);
  glow(ctx, back.path, WHITE, 0.08, 1, 0);

  // interior slices, brighter toward the front
  for (let j = 1; j < slices.length - 1; j++) {
    const s = slices[j];
    const l = new Lines(cam);
    for (let i = 0; i < n; i++) l.line(...P(i, s.tau, s.V[i]));
    glow(ctx, l.path, WHITE, 0.2 + 0.4 * (s.tau / Math.max(v.T, 1e-9)), 1, 0.4);
  }

  // columns across time
  if (slices.length > 1) {
    const col = new Lines(cam);
    for (let i = 0; i < n; i += 5) {
      col.lift();
      for (const s of slices) col.line(...P(i, s.tau, s.V[i]));
    }
    glow(ctx, col.path, WHITE, 0.26, 1, 0.3);
  }

  // early exercise: slice segments where V == payoff, and the boundary between
  if (v.american) {
    const ex = new Lines(cam);
    const boundary = [];
    for (const s of slices) {
      if (!s.ex) continue;
      ex.lift();
      const pts = [];
      for (let i = 0; i < n; i++) {
        if (s.ex[i]) ex.line(...P(i, s.tau, s.V[i]));
        else ex.lift();
        if (i < n - 1 && s.ex[i] !== s.ex[i + 1]) pts.push(s.ex[i] ? i : i + 1);
      }
      boundary.push({ s, pts });
      // fill the exercised floor strip
      for (let i = 0; i < n; i += 2) if (s.ex[i]) ex.seg([X(MESH_S[i]), 0, Z(s.tau)], P(i, s.tau, s.V[i]));
    }
    glow(ctx, ex.path, RED, 0.3, 1, 0.7);
    const b = new Lines(cam);
    const bf = new Lines(cam);
    for (let j = 0; j + 1 < boundary.length; j++) {
      const a = boundary[j];
      const c = boundary[j + 1];
      if (a.pts.length !== c.pts.length) continue;
      for (let k = 0; k < a.pts.length; k++) {
        b.seg(P(a.pts[k], a.s.tau, a.s.V[a.pts[k]]), P(c.pts[k], c.s.tau, c.s.V[c.pts[k]]));
        bf.seg([X(MESH_S[a.pts[k]]), 0, Z(a.s.tau)], [X(MESH_S[c.pts[k]]), 0, Z(c.s.tau)]);
      }
    }
    glow(ctx, b.path, RED, 1, 2, 1.4);
    glow(ctx, bf.path, RED, 0.6, 1, 1);
  }

  // payoff (expiry edge) and today's price (front edge)
  const pay = new Lines(cam);
  for (let i = 0; i < n; i++) pay.line(...P(i, 0, v.payoff[i]));
  glow(ctx, pay.path, WHITE, 0.95, 1.6, 1.4);
  if (v.front) {
    const fr = new Lines(cam);
    for (let i = 0; i < n; i++) fr.line(...P(i, v.T, v.front.V[i]));
    glow(ctx, fr.path, AMBER, 1, 2.4, 1.6);

    // inspection ring at S = 100
    const i0 = 60;
    const [sx, sy] = cam.project(...P(i0, v.T, v.front.V[i0]));
    const pulse = 9 + 2 * Math.sin(time * 3);
    ctx.globalCompositeOperation = "lighter";
    ctx.strokeStyle = `rgba(${AMBER},0.9)`;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(sx, sy, pulse, 0, 2 * Math.PI);
    ctx.moveTo(sx - pulse - 6, sy);
    ctx.lineTo(sx - pulse + 3, sy);
    ctx.moveTo(sx + pulse - 3, sy);
    ctx.lineTo(sx + pulse + 6, sy);
    ctx.stroke();
    ctx.globalCompositeOperation = "source-over";
    label(ctx, `S=100  V=${v.front.V[i0].toFixed(4)}`, sx + pulse + 10, sy - 10, { rgb: AMBER, alpha: 0.95, align: "left", size: fs + 1 });
  }
  {
    let im = 0;
    for (let i = 1; i < n; i++) if (v.payoff[i] > v.payoff[im]) im = i;
    const [sx, sy] = cam.project(...P(im, 0, v.payoff[im]));
    label(ctx, "PAYOFF", sx, sy - 12, { size: fs, alpha: 0.85 });
  }
  if (v.front) {
    const [sx, sy] = cam.project(...P(0, v.T, v.front.V[0]));
    label(ctx, `PRICE · T=${v.T.toFixed(2)}`, Math.max(sx, 60), sy - 14, { size: fs, rgb: AMBER, alpha: 0.9 });
  }
}

function drawBarriers(ctx, cam, v, top, fs) {
  const b = v.barrier;
  const z0 = Z(0);
  const z1 = Z(Math.max(v.T, 0.05));
  const wall = new Lines(cam);
  const hatch = new Lines(cam);
  for (const S of [b.L, b.U]) {
    if (!S) continue;
    const x = X(S);
    wall.move(x, 0, z0).line(x, top, z0).line(x, top, z1).line(x, 0, z1).line(x, 0, z0);
    for (let k = 1; k < 10; k++) {
      const z = z0 + ((z1 - z0) * k) / 10;
      wall.seg([x, 0, z], [x, top, z]);
    }
  }
  // diagonals x = d + (z - z0), clipped to each knocked strip
  const strips = [];
  if (b.L) strips.push([X(40), X(b.L)]);
  if (b.U) strips.push([X(b.U), X(160)]);
  for (const [xa, xb] of strips) {
    for (let d = xa - (z1 - z0); d < xb; d += 0.05) {
      const za = Math.max(z0, z0 + xa - d);
      const zb = Math.min(z1, z0 + xb - d);
      if (zb > za) hatch.seg([d + (za - z0), 0, za], [d + (zb - z0), 0, zb]);
    }
  }
  glow(ctx, hatch.path, RED, 0.22, 1, 0);
  glow(ctx, wall.path, RED, 0.35, 1, 0.6);
  for (const [S, name] of [[b.L, "L"], [b.U, "U"]]) {
    if (!S) continue;
    const [sx, sy] = cam.project(X(S), top, z1);
    label(ctx, `${name} ${S.toFixed(2)}`, sx, sy - 10, { rgb: RED, alpha: 0.9, size: fs });
  }
  const text = b.io === "in" ? "KNOCKED IN" : "KNOCKED OUT";
  for (const [xa, xb] of strips) {
    if (xb - xa < 0.25) continue;
    const [sx, sy] = cam.project((xa + xb) / 2, 0, (z0 + z1) / 2);
    label(ctx, text, sx, sy, { rgb: RED, alpha: 0.75, size: fs });
  }
}

// Linear interpolation of a priced curve (log-spaced S) onto MESH_S.
export function toMesh(S, V) {
  const out = new Float64Array(MESH_S.length);
  let j = 0;
  for (let i = 0; i < MESH_S.length; i++) {
    const s = MESH_S[i];
    while (j < S.length - 2 && S[j + 1] < s) j++;
    const t = Math.min(1, Math.max(0, (s - S[j]) / (S[j + 1] - S[j])));
    out[i] = V[j] + t * (V[j + 1] - V[j]);
  }
  return out;
}

export function maskToMesh(S, ex) {
  if (!ex) return null;
  const out = new Uint8Array(MESH_S.length);
  let j = 0;
  for (let i = 0; i < MESH_S.length; i++) {
    while (j < S.length - 1 && Math.abs(S[j + 1] - MESH_S[i]) <= Math.abs(S[j] - MESH_S[i])) j++;
    out[i] = ex[j];
  }
  return out;
}
