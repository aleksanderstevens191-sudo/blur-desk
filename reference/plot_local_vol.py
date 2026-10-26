"""Plot Stage 2: raw Dupire local vol, where it breaks, the regularized surface, and two slices.

python reference/plot_local_vol.py [out.png]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from local_vol import Surface, cell_times, E_FLAG, G_FLAG, N_FLAG

S = Surface.load(HERE.parent / "data" / "spy_full_2026-10-23")
K = np.linspace(-1.0, 0.5, 601)
TC = cell_times(S, per=8)
RAW = np.array([S.dupire_raw(K, t)[0] for t in TC])
REG = np.array([S.local_vol(K, t)[0] for t in TC])
F = np.array([S.flags(K, t)[0] for t in TC])
days = TC * 365

plt.style.use("dark_background")
mono = dict(fontsize=9, family="monospace")
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))


def heat(ax, Z, title):
    m = ax.pcolormesh(K, days, np.clip(np.nan_to_num(Z, nan=1.0), 0, 1.0) * 100, cmap="gray", vmin=0, vmax=60,
                      shading="nearest", rasterized=True)
    ax.set_yscale("log")
    ax.set_title(title, **mono)
    ax.set_xlabel("k = ln(K / F_T)", family="monospace")
    ax.set_ylabel("days to expiry (log)", family="monospace")
    return m


m = heat(axes[0, 0], RAW, "01 RAW DUPIRE LOCAL VOL (CLIPPED AT 60% FOR COLOUR)")
fig.colorbar(m, ax=axes[0, 0], label="local vol, %")

ax = axes[0, 1]
code = np.where(F & E_FLAG, 3, np.where(F & G_FLAG, 2, np.where(F & N_FLAG, 1, 0)))
cmap = ListedColormap(["#ffffff", "#9a9a9a", "#4a4a4a", "#141414"])
ax.pcolormesh(K, days, code, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest", rasterized=True)
ax.set_yscale("log")
ax.set_title("02 WHERE DUPIRE IS UNRELIABLE", **mono)
ax.set_xlabel("k = ln(K / F_T)", family="monospace")
ax.set_ylabel("days to expiry (log)", family="monospace")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(
    ["#ffffff", "#9a9a9a", "#4a4a4a", "#141414"],
    ["reliable: Dupire as is", "N: quote noise > 10% (kept, marked)", "G: denominator g < 0.1 (regularized)",
     "E: beyond quoted strikes (regularized)"])], fontsize=7, loc="lower left", framealpha=0.9,
    labelcolor="black", facecolor="#dddddd")

m = heat(axes[1, 0], REG, "03 REGULARIZED LOCAL VOL (WHAT STAGE 3 PRICES WITH)")
fig.colorbar(m, ax=axes[1, 0], label="local vol, %")
for ax in (axes[0, 0], axes[1, 0]):
    ax.contour(K, days, ((F & (E_FLAG | G_FLAG)) == 0).astype(float), levels=[0.5], colors="white",
               linewidths=0.6, linestyles="--")

ax = axes[1, 1]
for t_d, a in ((30, 1.0), (365, 0.55)):
    t = t_d / 365
    lv, raw, f = S.local_vol(K, t)
    iv = S.implied_vol(K, t)
    ax.plot(K, raw * 100, ":", color="white", alpha=a, lw=1.0, label=f"{t_d}d raw Dupire")
    ax.plot(K, lv * 100, "-", color="white", alpha=a, lw=1.4, label=f"{t_d}d regularized")
    ax.plot(K, iv * 100, "--", color="gray", alpha=a, lw=0.9, label=f"{t_d}d implied")
    bad = (f & (E_FLAG | G_FLAG)) > 0
    ax.fill_between(K, 0, 150, where=bad, color="white", alpha=0.06 * a, step="mid")
ax.set_ylim(0, 120)
ax.set_title("04 SLICES: RAW VS REGULARIZED (SHADED = REGULARIZED REGION)", **mono)
ax.set_xlabel("k = ln(K / F_T)", family="monospace")
ax.set_ylabel("vol, %", family="monospace")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.grid(alpha=0.15)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "docs" / "local_vol.png")
fig.savefig(out, dpi=140)
print("wrote", out)
