"""Plot the SPY smile (bid/ask), the flat Black-Scholes fit and the Merton fit.

python reference/plot_spy_fit.py [out.png]   (reads src/spy_fit.json from calibrate_spy.py)
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from smile import smile

fit = json.load(open(HERE.parent / "src" / "spy_fit.json"))
mp = {k: fit["merton"][k] for k in ("lam", "m", "delta")}
plt.style.use("dark_background")
fig, axes = plt.subplots(1, len(fit["slices"]), figsize=(6 * len(fit["slices"]), 4.4), sharey=True)
mono = dict(family="monospace")
for ax, s in zip(np.atleast_1d(axes), fit["slices"]):
    K = np.array(s["K"])
    mid, bid, ask = (np.array(s[k]) * 100 for k in ("iv", "iv_bid", "iv_ask"))
    ax.errorbar(K, mid, yerr=[mid - bid, ask - mid], fmt="o", ms=3.5, color="white", capsize=2,
                label="SPY market (bid / ask)")
    ks = np.linspace(np.log(K.min() / s["F"]) - 0.01, np.log(K.max() / s["F"]) + 0.01, 60)
    curve = smile("merton", ks, s["T"], s["r"], fit["merton"]["sigma"], s["S"], **mp)
    ax.plot(s["F"] * np.exp(ks), curve * 100, "-", color="white", lw=1.2, label="Merton fit (blur)")
    ax.axhline(fit["flat"]["sigma"] * 100, ls="--", color="gray", lw=1, label="flat Black-Scholes")
    ax.axvline(s["F"], color="gray", lw=0.5, alpha=0.5)
    ax.set_title(f"SPY {s['expiry']} ({s['T'] * 365:.0f} DAYS)", fontsize=9, **mono)
    ax.set_xlabel("strike", **mono)
    ax.grid(alpha=0.15)
np.atleast_1d(axes)[0].set_ylabel("implied vol, %", **mono)
np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
m = fit["merton"]
fig.suptitle(f"AS OF {fit['asof']}: FLAT BS RMS {fit['flat']['rms_vol_pts']:.2f} VOL PTS, MERTON RMS "
             f"{m['rms_vol_pts']:.2f} (LAM {m['lam']:.2f}/YR, MEAN JUMP {m['m']:+.1%}, JUMP VOL {m['delta']:.1%}, "
             f"DIFFUSION {m['sigma']:.1%})", fontsize=8.5, **mono)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "spy_fit.png"
fig.savefig(out, dpi=150)
print("wrote", out)
