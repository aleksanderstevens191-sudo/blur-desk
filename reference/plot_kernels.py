"""Plot the three kernels on a log scale, same variance, so the tails can be compared.

python reference/plot_kernels.py [out.png]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from kernels import density, mean_sd

T, r = 0.25, 0.04
MODELS = {
    "Gaussian (Black-Scholes)": ("gaussian", 0.0, {}),
    "Merton: 1 jump/yr, mean -10%, vol 15%": ("merton", 0.0, dict(lam=1.0, m=-0.10, delta=0.15)),
    "variance gamma: theta -0.14, nu 0.2": ("vg", 0.0, dict(theta=-0.14, nu=0.2)),
}
# match every model's standard deviation to 20% vol over T
target = 0.20 * np.sqrt(T)
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
y = np.linspace(-0.9, 0.6, 3001)
for (label, (model, _, p)), ls in zip(MODELS.items(), ["-", "--", ":"]):
    lo, hi = 0.01, 1.0  # bisect the diffusion vol so the total sd matches
    for _ in range(60):
        s = 0.5 * (lo + hi)
        lo, hi = (s, hi) if mean_sd(model, T, r, s, **p)[1] < target else (lo, s)
    d = density(model, y, T, r, s, **p)
    for ax in axes:
        ax.plot(y, d, ls, color="white", lw=1.2, label=f"{label}, diffusion vol {s:.1%}")
axes[0].set_title("LOG-RETURN KERNELS, SAME VARIANCE (T = 3 MONTHS)", fontsize=9, family="monospace")
axes[1].set_title("SAME, LOG SCALE: THE TAILS", fontsize=9, family="monospace")
axes[1].set_yscale("log")
axes[1].set_ylim(1e-8, 30)
for ax in axes:
    ax.set_xlabel("ln(S_T / S_0)", family="monospace")
    ax.grid(alpha=0.15)
axes[0].legend(fontsize=7, loc="upper left", frameon=False)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "kernels.png"
fig.savefig(out, dpi=150)
print("wrote", out)
