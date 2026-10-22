"""Stage 3: why traders hate barriers.

1) Delta and gamma of an up-and-out call by finite differences on the blurred surface.
2) Discrete-monitoring error of the step-by-step blur against the number of steps,
   with and without the Broadie-Glasserman-Kou shift; slopes are measured, not assumed.

Writes docs/barrier_greeks.png and docs/monitoring_convergence.png. Exits non-zero on failure.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from barriers import barrier_closed_form, stepping_prices
from barrier_greeks import blurred_surface, vanilla_surface, fd_greeks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = Path(__file__).parent.parent / "docs"
DOCS.mkdir(exist_ok=True)
failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
    return "PASS" if ok else "FAIL"


# ---------- plot style: white wireframe on black, monospace uppercase ----------
INK, DIM, FAINT, BG = "#f2f2f2", "#9a9a9a", "#4a4a4a", "#000000"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": FAINT, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "text.color": INK, "font.family": "DejaVu Sans Mono", "font.size": 9,
    "axes.grid": True, "grid.color": "#1c1c1c", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def panel_title(ax, n, text):
    ax.set_title(f"{n:02d}  {text.upper()}", loc="left", fontsize=9, color=INK, pad=8)


S0, K, H, sigma, r = 100.0, 100.0, 120.0, 0.20, 0.05
KIND = "up-and-out call"

print("=" * 72)
print("1) Up-and-out call near its barrier: Greeks off the blurred surface")
print("=" * 72)
print(f"  K={K:.0f}, H={H:.0f}, sigma={sigma}, r={r}; one blur prices every spot, dx = 0.0005 in ln S")
print(f"  delta = V_x / S, gamma = (V_xx - V_x) / S^2 from neighbouring grid nodes")
print()
print(f"  {'T':>6} {'delta<0 from':>13} {'delta at H-':>12} {'min gamma':>10} {'at S':>7} "
      f"{'vanilla peak':>13} {'|ratio|':>8} {'|delta err|':>12} {'|gamma err|':>12}")
TS = [1.0, 0.25, 0.05, 0.01, 0.002]
curves, peaks, vpeaks = {}, [], []
for T in TS:
    S, V = blurred_surface(KIND, K, H, sigma, T, r)
    s, d, g = fd_greeks(S, V)
    # strictly below the barrier: at H itself delta jumps to 0, so gamma there is a spike
    live = (s > 60) & (s < H * (1 - 1e-9))
    Sv, Vv = vanilla_surface(K, sigma, T, r, anchor=H)
    sv, dv, gv = fd_greeks(Sv, Vv)
    van_peak = gv[(sv > 60) & (sv < 160)].max()
    i = np.argmin(g[live])
    peak, at = g[live][i], s[live][i]
    flip = s[live][np.argmax(d[live] < 0)]

    # against finite differences of the closed form (bump $0.001), every 25th node
    cf = lambda x: barrier_closed_form(KIND, x, K, H, sigma, T, r)
    b = 0.001
    idx = [j for j in np.where(live)[0][::25] if s[j] + b < H]
    ed = max(abs(d[j] - (cf(s[j] + b) - cf(s[j] - b)) / (2 * b)) for j in idx)
    eg = max(abs(g[j] - (cf(s[j] + b) - 2 * cf(s[j]) + cf(s[j] - b)) / b**2) for j in idx)
    ratio = abs(peak) / van_peak
    print(f"  {T:>6} {f'S = {flip:.1f}':>13} {d[live][-1]:>12.3f} {peak:>10.3f} {at:>7.2f} "
          f"{van_peak:>13.4f} {ratio:>7.1f}x {ed:>12.1e} {eg:>12.1e}")
    curves[T] = (s, d, g, S, V, sv, dv, gv, Sv, Vv)
    peaks.append(abs(peak))
    vpeaks.append(van_peak)
    check(d[live].min() < 0, f"delta flips T={T}")
    check(eg < 1e-3 * abs(peak) and ed < 1e-3 * abs(d[live]).max(), f"greeks vs closed form T={T}")
sb = np.polyfit(np.log(TS), np.log(peaks), 1)[0]
sv_ = np.polyfit(np.log(TS), np.log(vpeaks), 1)[0]
print(f"  measured scaling as expiry nears: barrier |gamma| peak ~ T^{sb:.2f}, vanilla gamma peak ~ T^{sv_:.2f}")
print(f"  at half a trading day (T=0.002) the barrier's gamma is {peaks[-1] / vpeaks[-1]:.0f}x the vanilla's "
      f"and its delta just below H is {curves[0.002][1][(curves[0.002][0] < H * (1 - 1e-9))][-1]:.1f}")
check(sb < -0.8 and abs(sv_ + 0.5) < 0.1 and peaks[-1] / vpeaks[-1] > 10, "gamma blow-up")
print("  Delta turns negative below the barrier: near H, a rise in the stock brings the option")
print("  closer to dying. Gamma near H is large and negative and grows like 1/T, because the")
print("  price has to fall from ~$20 of intrinsic value to 0 across a distance ~ sigma sqrt(T);")
print("  a vanilla call's gamma grows only like 1/sqrt(T). A hedger just below H half a day out is")
print("  short ~14 shares per option, and must buy them all back the instant it knocks out.")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
lo, hi = 85, 125
for T, lw, col, lab in [(0.01, 1.8, INK, "UP-AND-OUT, T = 0.01 (2.5 DAYS)"), (0.05, 1.1, DIM, "UP-AND-OUT, T = 0.05")]:
    s, d, g, S, V, sv, dv, gv, Sv, Vv = curves[T]
    m = (S > lo) & (S < hi)
    axes[0].plot(S[m], V[m], color=col, lw=lw, label=lab)
    m = (s > lo) & (s < H * (1 - 1e-9))
    axes[1].plot(s[m], d[m], color=col, lw=lw, label=lab)
    axes[2].plot(s[m], g[m], color=col, lw=lw, label=lab)
s, d, g, S, V, sv, dv, gv, Sv, Vv = curves[0.01]
m = (Sv > lo) & (Sv < hi)
axes[0].plot(Sv[m], Vv[m], color=DIM, lw=1, ls=(0, (4, 3)), label="VANILLA CALL, T = 0.01")
m = (sv > lo) & (sv < hi)
axes[1].plot(sv[m], dv[m], color=DIM, lw=1, ls=(0, (4, 3)), label="VANILLA CALL, T = 0.01")
axes[2].plot(sv[m], gv[m], color=DIM, lw=1, ls=(0, (4, 3)), label="VANILLA CALL, T = 0.01")
for ax in axes:
    ax.axvline(H, color=INK, lw=0.8, ls=":")
    ax.axvspan(H, hi, color="#151515", lw=0)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("SPOT")
axes[0].text(H + 0.6, axes[0].get_ylim()[1] * 0.92, "KNOCKED\nOUT", color=DIM, fontsize=8, va="top")
axes[1].axhline(0, color=FAINT, lw=0.8)
axes[2].axhline(0, color=FAINT, lw=0.8)
panel_title(axes[0], 1, "price")
panel_title(axes[1], 2, "delta")
panel_title(axes[2], 3, "gamma")
pk1, vp1 = peaks[TS.index(0.01)], vpeaks[TS.index(0.01)]
at1 = curves[0.01][0][np.argmin(np.where(curves[0.01][0] < H * (1 - 1e-9), curves[0.01][2], 0))]
axes[2].annotate(f"MIN GAMMA {-pk1:.2f}\n= {pk1 / vp1:.1f}x VANILLA PEAK", xy=(at1, -pk1), xytext=(88, -pk1 * 0.55),
                 color=INK, fontsize=8, arrowprops=dict(arrowstyle="-", color=DIM, lw=0.8))
axes[0].legend(loc="upper left", fontsize=7.5, labelcolor=DIM)
fig.suptitle(f"UP-AND-OUT CALL  K = {K:.0f}  H = {H:.0f}  SIGMA = {sigma:.0%}  R = {r:.0%}   "
             "GREEKS BY FINITE DIFFERENCES ON THE BLURRED SURFACE", x=0.01, ha="left", fontsize=9, color=DIM)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(DOCS / "barrier_greeks.png", dpi=160)
plt.close(fig)
print(f"  plot: docs/barrier_greeks.png")

print()
print("=" * 72)
print("2) Discrete-monitoring error vs number of steps (log-log)")
print("=" * 72)
T = 1.0
Ns = [10, 20, 40, 80, 160, 320, 640, 1280]
FIT_FROM = 40
cases = [("up-and-out call", 100.0, 120.0, "UOC K=100 H=120 (payoff $20 at H)"),
         ("down-and-out call", 100.0, 90.0, "DOC K=100 H=90 (payoff 0 at H)")]
print(f"  step-by-step blur, monitored at T/N, ..., T; error vs the continuous closed form")
print(f"  sigma={sigma}, r={r}, T={T}; slope = least-squares fit of log|error| on log N for N >= {FIT_FROM}")
print()
results = {}
for kind, Kc, Hc, label in cases:
    cf = barrier_closed_form(kind, S0, Kc, Hc, sigma, T, r)
    print(f"  {label}: continuous price {cf:.6f}")
    print(f"  {'N':>6} {'no shift':>11} {'local':>6}  {'BGK shift':>11} {'local':>6}")
    errs = {}
    for sh in [False, True]:
        errs[sh] = np.array([stepping_prices([kind], S0, Kc, Hc, sigma, T, r, N, shift=sh)[kind] - cf for N in Ns])
    for i, N in enumerate(Ns):
        loc = ["" if i == 0 else f"{np.log(abs(errs[sh][i] / errs[sh][i - 1])) / np.log(2):.2f}" for sh in [False, True]]
        print(f"  {N:>6} {errs[False][i]:>11.3e} {loc[0]:>6}  {errs[True][i]:>11.3e} {loc[1]:>6}")
    fit = {}
    for sh in [False, True]:
        m = np.array(Ns) >= FIT_FROM
        fit[sh] = np.polyfit(np.log(np.array(Ns)[m]), np.log(np.abs(errs[sh][m])), 1)[0]
    # is the shifted error at the largest N real, or grid error? halve dx and look again
    fine = stepping_prices([kind], S0, Kc, Hc, sigma, T, r, Ns[-1], shift=True, dx_max=0.00025)[kind] - cf
    print(f"  measured slope: no shift {fit[False]:.3f}, with shift {fit[True]:.3f}")
    print(f"  grid check at N={Ns[-1]} with shift: error {errs[True][-1]:.3e} at dx<=0.0005, {fine:.3e} at dx<=0.00025")
    print()
    results[label] = (errs, fit)
    check(abs(fit[False] + 0.5) < 0.05, f"{kind} no-shift slope")
    check(fit[True] < fit[False] - 0.4, f"{kind} shift slope")
    check(abs(fine - errs[True][-1]) < 0.05 * abs(errs[True][-1]), f"{kind} grid check")
print("  Without the shift the error falls like N^-1/2, as expected: a path can cross the barrier")
print("  and come back between monitoring dates, and the overshoot is ~ sigma sqrt(dt).")
print("  The shift removes that leading term. What is left depends on the payoff at the barrier:")
print("  N^-1 when the payoff is still worth something there (UOC, K < H), N^-3/2 when it is 0 (DOC).")

fig, ax = plt.subplots(figsize=(7.5, 5))
styles = {cases[0][3]: (INK, "o"), cases[1][3]: (DIM, "s")}
for label, (errs, fit) in results.items():
    col, mk = styles[label]
    for sh, ls in [(False, "-"), (True, (0, (4, 3)))]:
        e = np.abs(errs[sh])
        ax.loglog(Ns, e, color=col, lw=1.4, ls=ls, marker=mk, ms=4.5, mfc=BG, mec=col)
        tag = f"{label.split()[0]} {'SHIFT' if sh else 'NO SHIFT'}  SLOPE {fit[sh]:.2f}"
        ax.annotate(tag, xy=(Ns[-1], e[-1]), xytext=(6, 0), textcoords="offset points",
                    color=col, fontsize=7.5, va="center")
ax.set_xlim(8, 1280 * 9)
ax.set_xlabel("MONITORING STEPS N")
ax.set_ylabel("|PRICE ERROR| VS CONTINUOUS ($)")
ax.set_title("04  DISCRETE-MONITORING ERROR, STEP-BY-STEP BLUR", loc="left", fontsize=9, color=INK, pad=30)
ax.text(0.0, 1.0, "SOLID = NO SHIFT   DASHED = BGK SHIFT (BARRIER MOVED 0.5826 SIGMA SQRT(DT) TOWARD SPOT)\n"
        "UOC: K=100 H=120   DOC: K=100 H=90   SIGMA=20% R=5% T=1",
        transform=ax.transAxes, fontsize=7, color=DIM, va="bottom")
fig.tight_layout()
fig.savefig(DOCS / "monitoring_convergence.png", dpi=160)
plt.close(fig)
print(f"  plot: docs/monitoring_convergence.png")

print()
if failures:
    print("FAILED: " + ", ".join(failures))
    sys.exit(1)
print("all stage 3 checks passed")
