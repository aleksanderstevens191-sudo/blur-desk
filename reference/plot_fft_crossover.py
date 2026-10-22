"""Plot where the FFT blur overtakes the direct blur, from the JSON that
/fft-bench.html prints.

python reference/plot_fft_crossover.py [bench.json] [out.png]

Defaults: reference/fft_bench_apple.json if it exists (Alek's Mac), else
reference/fft_bench_swiftshader.json (this repo's CI container, a software
adapter: the shape of the curves, not GPU speed).
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plain = FuncFormatter(lambda v, _: f"{v:g}")

here = Path(__file__).parent
default = here / "fft_bench_apple.json"
if not default.exists():
    default = here / "fft_bench_swiftshader.json"
src = Path(sys.argv[1]) if len(sys.argv) > 1 else default
out = sys.argv[2] if len(sys.argv) > 2 else str(here.parent / "docs" / "fft_crossover.png")
d = json.loads(src.read_text())

ns = sorted({r["n"] for r in d["sweep"]})
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
mono = dict(fontsize=9, family="monospace")

ax = axes[0]
for i, n in enumerate(ns):
    rs = [r for r in d["sweep"] if r["n"] == n]
    m = np.array([r["m"] for r in rs])
    ratio = np.array([r["directMs"] / r["fftMs"] for r in rs])
    a = 0.35 + 0.65 * i / max(1, len(ns) - 1)
    ax.loglog(m, ratio, "-o", color="white", alpha=a, lw=1.1, ms=3, label=f"n = {n}")
ax.axhline(1, color="gray", lw=0.8)
ax.text(ax.get_xlim()[0] * 1.2, 1.08, "break-even", color="gray", fontsize=7, family="monospace")
ax.set_title("DIRECT TIME / FFT TIME (ABOVE 1: FFT WINS)", **mono)
ax.set_xlabel("kernel width m (taps)", family="monospace")
ax.set_ylabel("speed-up of the FFT", family="monospace")
ax.grid(alpha=0.15, which="both")
ax.legend(fontsize=7, frameon=False, loc="upper left")

ax = axes[1]
cr = [c for c in d["crossover"] if c["mStar"] is not None]
n = np.array([c["n"] for c in cr])
ms = np.array([c["mStar"] for c in cr])
ax.loglog(n, ms, "-o", color="white", lw=1.4, ms=4, label="crossover m*(n)")
for c in d.get("realistic", []):
    ax.plot(c["n"], c["m"], "s", color="white", ms=5, mfc="none")
    ax.annotate(f"{c['what']}: FFT {c['directMs'] / c['fftMs']:.0f}x", (c["n"], c["m"]),
                textcoords="offset points", xytext=(-8, -12), ha="right", color="white",
                fontsize=7, family="monospace")
ax.set_title("FFT WINS ABOVE THE LINE; SQUARES: THE KERNELS WE USE", **mono)
ax.set_xlabel("grid size n (nodes)", family="monospace")
ax.set_ylabel("kernel width m (taps)", family="monospace")
ax.grid(alpha=0.15, which="both")
ax.legend(fontsize=7, frameon=False, loc="upper left")

for ax in axes:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
fig.suptitle(f"FFT VS DIRECT BLUR ON THE GPU: {d['gpu'].upper()}, {d['date']}"
             + (" (QUICK SWEEP, SOFTWARE ADAPTER)" if d.get("quick") else ""), **mono)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"wrote {out} from {src}")
