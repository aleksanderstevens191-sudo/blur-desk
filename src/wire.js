// A tiny wireframe renderer on a 2D canvas: perspective camera, glowing lines.

export class Camera {
  constructor({ az = -0.6, el = 0.45, dist = 3.4, target = [0, 0, 0], zoom = 1 } = {}) {
    Object.assign(this, { az, el, dist, target, zoom });
    this.W = 1;
    this.H = 1;
  }

  setViewport(W, H, cx = W / 2, cy = H / 2) {
    Object.assign(this, { W, H, cx, cy });
    this.f = Math.min(W, H) * 0.95 * this.zoom;
    this.ca = Math.cos(this.az);
    this.sa = Math.sin(this.az);
    this.ce = Math.cos(this.el);
    this.se = Math.sin(this.el);
  }

  // [screenX, screenY, depth]; depth grows away from the camera.
  project(x, y, z) {
    x -= this.target[0];
    y -= this.target[1];
    z -= this.target[2];
    const x1 = x * this.ca - z * this.sa;
    const z1 = x * this.sa + z * this.ca;
    const y2 = y * this.ce - z1 * this.se;
    const z2 = y * this.se + z1 * this.ce;
    const d = this.dist - z2;
    const s = this.f / d;
    return [this.cx + x1 * s, this.cy - y2 * s, d];
  }
}

// Collects 3D polylines into one Path2D.
export class Lines {
  constructor(cam) {
    this.cam = cam;
    this.path = new Path2D();
    this.pen = false;
  }
  move(x, y, z) {
    const [sx, sy] = this.cam.project(x, y, z);
    this.path.moveTo(sx, sy);
    this.pen = true;
    return this;
  }
  line(x, y, z) {
    const [sx, sy] = this.cam.project(x, y, z);
    if (this.pen) this.path.lineTo(sx, sy);
    else this.path.moveTo(sx, sy);
    this.pen = true;
    return this;
  }
  lift() {
    this.pen = false;
    return this;
  }
  seg(a, b) {
    return this.move(...a).line(...b);
  }
}

// Stroke a path with a soft halo under a sharp core, additively blended.
export function glow(ctx, path, rgb, alpha = 1, width = 1, halo = 1) {
  ctx.globalCompositeOperation = "lighter";
  if (halo > 0) {
    ctx.strokeStyle = `rgba(${rgb},${0.07 * alpha * halo})`;
    ctx.lineWidth = width + 5;
    ctx.stroke(path);
    ctx.strokeStyle = `rgba(${rgb},${0.16 * alpha * halo})`;
    ctx.lineWidth = width + 2;
    ctx.stroke(path);
  }
  ctx.strokeStyle = `rgba(${rgb},${alpha})`;
  ctx.lineWidth = width;
  ctx.stroke(path);
  ctx.globalCompositeOperation = "source-over";
}

// Text size multiplier for every canvas label (the recording mode enlarges them)
export const TEXT = { scale: 1 };

export function label(ctx, text, x, y, { rgb = "200,204,210", alpha = 0.75, align = "center", base = "middle", size = 10, weight = 500 } = {}) {
  ctx.font = `${weight} ${size * TEXT.scale}px ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace`;
  ctx.fillStyle = `rgba(${rgb},${alpha})`;
  ctx.textAlign = align;
  ctx.textBaseline = base;
  ctx.fillText(text, x, y);
}

// Size a canvas to its CSS box at device resolution; returns [W, H] in CSS px.
export function fit(canvas, ctx) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return [W, H];
}

export const WHITE = "236,238,242";
export const AMBER = "255,184,64";
export const RED = "255,77,94";
