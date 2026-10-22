"""Plot the smiles, the skew from a negative mean jump, and how both fade with maturity.

python reference/plot_smile.py [out.png]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from smile import smile, atm_shape

r = 0.04
PAIRS = {
    "MERTON": [("m = 0", "merton", 0.15, dict(lam=1.0, m=0.0, delta=0.15)),
               ("m = -10%", "merton", 0.15, dict(lam=1.0, m=-0.10, delta=0.15))],
    "VARIANCE GAMMA": [("theta = 0", "vg", 0.18, dict(theta=0.0, nu=0.2)),
                       ("theta = -0.2", "vg", 0.18, dict(theta=-0.2, nu=0.2))],
}
SHOW_T = [(1 / 12, "1m"), (0.25, "3m"), (1.0, "1y"), (5.0, "5y")]
FADE_T = np.geomspace(7 / 365, 5, 14)
ks = np.linspace(-0.35, 0.25, 61)

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
mono = dict(fontsize=9, family="monospace")
for ax, (title, pair) in zip(axes[:2], PAIRS.items()):
    for (name, model, s, p), ls in zip(pair, ["--", "-"]):
        for (T, tl), a in zip(SHOW_T, [1.0, 0.75, 0.5, 0.3]):
            v = smile(model, ks, T, r, s, **p)
            ax.plot(ks, v * 100, ls, color="white", alpha=a, lw=1.2,
                    label=f"{name}, {tl}")
    ax.set_title(f"{title} SMILE: DASHED SYMMETRIC, SOLID NEGATIVE JUMP", **mono)
    ax.set_xlabel("k = ln(K / F)", family="monospace")
    ax.set_ylabel("Black-Scholes implied vol, %", family="monospace")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper right")

ax = axes[2]
for (title, pair) in PAIRS.items():
    for name, model, s, p in pair:
        sh = np.array([atm_shape(model, T, r, s, **p) for T in FADE_T])
        ls = "-" if model == "merton" else ":"
        ax.loglog(FADE_T, np.abs(sh[:, 1]), ls, color="white", lw=1.4 if "-" in name else 0.8,
                  label=f"{'Merton' if model == 'merton' else 'VG'} {name}")
ax.loglog(FADE_T, 0.09 / FADE_T, color="gray", lw=0.6, alpha=0.6)
ax.text(2.2, 0.09 / 2.2 * 1.3, "slope -1", color="gray", fontsize=7, family="monospace")
ax.set_title("ATM SKEW |dIV/dk| FADES WITH MATURITY", **mono)
ax.set_xlabel("maturity T, years (log)", family="monospace")
ax.grid(alpha=0.15, which="both")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "smile.png"
fig.savefig(out, dpi=150)
print("wrote", out)
