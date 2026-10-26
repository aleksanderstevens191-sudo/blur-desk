"""Part 4 Stage 3 checks: pricing under local vol with a blur whose width changes (lv_blur.py).

python reference/check_lv_blur.py [data/spy_full_<date>]
  -> report on stdout (saved as reference/check_lv_blur_results.txt), docs/lv_roundtrip.png

1. Constant local vol against Black-Scholes, in dollars on SPY's forwards and discount factors.
2. Forward pass against backward pass under the real local vol (they are transposes of each other).
3. Round trip: every OTM SPY quote priced under Stage 2's local vol, turned back into implied vol, and
   compared with the Stage 1 surface it came from. Also the rejected wing fill, for its cost.
4. QuantLib's finite-difference engine with local vol (FdBlackScholesVanillaEngine, localVol=True) on the
   same local-vol table, 40 contracts.
5. Convergence: the round trip repeated with finer time steps, then with a finer grid.
The heavy runs (one forward pass each, all 32 expiries) go to a process pool.
"""
import os
import sys
import json
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import local_vol as LVM
from lv_blur import LVTable, Flat, prices_forward, price_backward, implied_w, black_c

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
DAY = 1 / 365
X_LO, X_HI = -4.0, 2.5              # ln(S/F); -3 cost the 2-year deep puts (k = -1.7) up to 0.05 vol points
PROD = dict(dt=1 * DAY, m=32, dx=0.0008)          # the settings the round trip is reported at
LIQ_MID = 0.50                                    # liquid: mid price >= $0.50


def surface():
    return LVM.Surface.load(FOLDER)


def quotes(S):
    """OTM quotes per expiry with an implied vol: k, call flag, mid, liquid flag."""
    s1 = json.load(open(FOLDER / "stage1.json"))
    out = []
    for i, e in enumerate(S.names):
        F = S.F[i]
        cs = [c for c in s1["contracts"] if c["expiry"] == e and c["iv_true"] is not None and (c["cp"] == "call") == (c["K"] >= F)]
        mid = np.array([0.5 * (c["bid"] + c["ask"]) for c in cs])
        out.append(dict(k=np.array([np.log(c["K"] / F) for c in cs]), call=np.array([c["cp"] == "call" for c in cs]),
                        K=np.array([c["K"] for c in cs]), mid=mid, liq=mid >= LIQ_MID))
    return out


def run(job):
    """One forward pass in a worker: job = (variant, dt, m, dx, expiries, strikes, calls, knots[,
    x_lo, x_hi])."""
    variant, dt, m, dx, exps, ks, cl, knots, *lim = job
    lo, hi = lim if lim else (X_LO, X_HI)
    S = surface()
    if variant == "filled":
        S.local_vol = S.local_vol_filled
    tab = LVTable.from_surface(S)
    t0 = time.time()
    out = prices_forward(tab, exps, ks, cl, dt, dx, lo, hi, m=m, knots=knots)
    return out, time.time() - t0


def vols(c, k, call, T):
    return np.sqrt(implied_w(c, k, call) / T)


def ql_prices(tab, S, contracts, grids):
    """QuantLib FD local-vol prices (forward units: spot 1, zero rates) of (days, k, call) contracts on the
    same table: rows at the table's times, plus copies of each interval's end rows just inside its ends
    (QuantLib interpolates linearly in time between rows, and the table must not blend across an expiry)."""
    import QuantLib as ql
    eps = 1e-7
    times, rows = [], []
    for j in range(1, len(tab.edges)):
        r = np.nonzero(tab.cell == j)[0]
        times += [tab.edges[j - 1] + eps] + list(tab.t[r]) + [tab.edges[j] - eps]
        rows += [tab.sig[r[0]]] + [tab.sig[i] for i in r] + [tab.sig[r[-1]]]
    times += [tab.edges[-1] + eps, tab.edges[-1] + 10.0]
    rows += [tab.sig[-1], tab.sig[-1]]
    ref = ql.Date(23, 10, 2026)
    ql.Settings.instance().evaluationDate = ref
    dc = ql.Actual365Fixed()
    strikes = [float(x) for x in np.exp(tab.k)]
    M = ql.Matrix(len(strikes), len(times))
    for c, row in enumerate(rows):
        for r_, v in enumerate(row):
            M[r_][c] = float(v)
    lv = ql.FixedLocalVolSurface(ref, [float(t) for t in times], strikes, M, dc,
                                 ql.FixedLocalVolSurface.ConstantExtrapolation, ql.FixedLocalVolSurface.ConstantExtrapolation)
    lv.enableExtrapolation()
    x0 = ql.QuoteHandle(ql.SimpleQuote(1.0))
    zero = ql.YieldTermStructureHandle(ql.FlatForward(ref, 0.0, dc))
    # The engine sizes its log-spot mesh from the Black vol it is given (the local vol is used only in the
    # operator); at 20% the mesh is too narrow for SPY's 100%+ put-wing local vol and prices come out 5% off.
    # A 100% Black vol widens it to cover the wings.
    bv = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref, ql.NullCalendar(), 1.0, dc))
    proc = ql.GeneralizedBlackScholesProcess(x0, zero, zero, bv, ql.LocalVolTermStructureHandle(lv))
    out = np.zeros((len(contracts), len(grids)))
    for n, (days, k, call) in enumerate(contracts):
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, float(np.exp(k))),
                               ql.EuropeanExercise(ref + int(days)))
        for g, (tg, xg) in enumerate(grids):
            opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, tg, xg, 4, ql.FdmSchemeDesc.CrankNicolson(), True))
            out[n, g] = opt.NPV()
    return out


def main():
    S = surface()
    Q = quotes(S)
    n = len(S.T)
    ks, cl = [q["k"] for q in Q], [q["call"] for q in Q]
    exps = list(S.T)
    tab = LVTable.from_surface(S)
    print(f"Stage 3: pricing under local vol by blurring with a width that changes from spot to spot "
          f"({FOLDER.name}, {n} expiries, {sum(len(k) for k in ks)} OTM quotes). Python float64.")
    print(f"Settings reported: time steps <= {PROD['dt'] * 365:g} day and at least {PROD['m']} per expiry interval, "
          f"grid step dx = {PROD['dx']} in ln(S/F) on [{X_LO}, {X_HI}], kernels to 8 standard deviations.")
    print()

    # the heavy runs, all at once
    D = PROD["dx"]
    jobs = {
        "prod": ("clip", PROD["dt"], PROD["m"], D, exps, ks, cl, ()),
        "filled": ("filled", PROD["dt"], PROD["m"], D, exps, ks, cl, ()),
        "t4": ("clip", 4 * DAY, 8, D, exps, ks, cl, ()),
        "t2": ("clip", 2 * DAY, 16, D, exps, ks, cl, ()),
        "t05": ("clip", 0.5 * DAY, 64, D, exps, ks, cl, ()),
        "x16": ("clip", PROD["dt"], PROD["m"], 0.0016, exps, ks, cl, ()),
        "x11": ("clip", PROD["dt"], PROD["m"], 0.0008 * np.sqrt(2), exps, ks, cl, ()),
        "x057": ("clip", PROD["dt"], PROD["m"], 0.0008 / np.sqrt(2), exps, ks, cl, ()),
        "wide": ("clip", PROD["dt"], PROD["m"], D, exps, ks, cl, (), X_LO - 1.0, X_HI + 1.0),
    }
    # QuantLib contracts: 8 expiries in whole days, 5 liquid OTM strikes each (nearest quotes to
    # -2, -1, 0, +1, +1.5 standard deviations)
    pick = [0, 4, 9, 11, 15, 19, 24, 28]
    qc, qk, qcl, qD = [], [], [], []
    for i in pick:
        days = int(np.floor(S.T[i] * 365))
        sd = np.sqrt(S.implied_vol(np.array([0.0]), S.T[i])[0] ** 2 * S.T[i])
        q = Q[i]
        ch = []
        for z in (-2.0, -1.0, 0.0, 1.0, 1.5):
            cand = np.nonzero(q["liq"])[0]
            j = cand[np.argmin(np.abs(q["k"][cand] - z * sd))]
            if j not in ch:
                ch.append(j)
        ch.sort()
        qc += [(days, float(q["k"][j]), bool(q["call"][j])) for j in ch]
        qD += [S.D[i] * S.F[i]] * len(ch)
    qT = sorted({c[0] / 365 for c in qc})
    qks = [np.array([c[1] for c in qc if c[0] / 365 == T]) for T in qT]
    qcls = [np.array([c[2] for c in qc if c[0] / 365 == T]) for T in qT]
    jobs["ql1"] = ("clip", PROD["dt"], PROD["m"], D, qT, qks, qcls, tuple(exps))
    jobs["ql2"] = ("clip", 0.5 * PROD["dt"], 2 * PROD["m"], D, qT, qks, qcls, tuple(exps))
    t0 = time.time()
    cache = os.environ.get("LV_CACHE")                # keep the heavy runs while developing the report
    R = pickle.load(open(cache, "rb")) if cache and os.path.exists(cache) else {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        jobs_run = {nm: j for nm, j in jobs.items() if nm not in R}
        fut = {name: ex.submit(run, job) for name, job in sorted(jobs_run.items(), key=lambda kv: kv[0] in ("t05", "x057"), reverse=True)}

        # 1. Black-Scholes (runs here while the pool works)
        print("1. Constant local vol against Black-Scholes (dollars: D_T F_T times the forward-unit price; SPY's")
        print("   forwards and discount factors; strikes 0.6 F .. 1.4 F every 0.05 F, calls and puts; grid out to")
        print("   8 standard deviations of ln S at the last expiry each way, and at least the production grid)")
        worst = 0.0
        for sig in (0.15, 0.40):
            idx = [0, 11, 26, n - 1]
            kk = np.log(np.arange(0.6, 1.4001, 0.05))
            kz = [np.concatenate([kk, kk])] * len(idx)
            cz = [np.r_[np.ones(len(kk), bool), np.zeros(len(kk), bool)]] * len(idx)
            Tz = [S.T[i] for i in idx]
            half = max(X_HI, 8 * sig * np.sqrt(max(Tz)))      # the grid spans 8 standard deviations each way
            fw = prices_forward(Flat(sig), Tz, kz, cz, PROD["dt"], D, -half, half, m=PROD["m"])
            for a, i in enumerate(idx):
                bs = black_c(kz[a], sig * sig * S.T[i], cz[a])
                err = np.max(np.abs(fw[a] - bs)) * S.D[i] * S.F[i]
                bk = [price_backward(Flat(sig), float(kz[a][j]), S.T[i], bool(cz[a][j]), PROD["dt"], D, -half, half, m=PROD["m"])
                      for j in (0, 8, 16, 17, 25, 33)]
                errb = np.max(np.abs(np.array(bk) - bs[[0, 8, 16, 17, 25, 33]])) * S.D[i] * S.F[i]
                worst = max(worst, err, errb)
                print(f"   vol {sig:.0%}  {S.T[i] * 365:6.1f}d  F ${S.F[i]:.2f}  max |error| forward pass ${err:.1e} "
                      f"(34 options), backward ${errb:.1e} (6)")
        print(f"   worst ${worst:.1e} against the 1e-6 requirement: {'PASS' if worst < 1e-6 else 'FAIL'}")
        print()

        # 2. forward = backward under local vol
        print("2. Forward pass against backward pass under the local vol (the scatter is the transpose of the gather)")
        tf = [S.T[0], S.T[4]]
        kt = [np.array([-0.03, 0.0, 0.02])] * 2
        ct = [np.array([False, True, True])] * 2
        fw = prices_forward(tab, tf, kt, ct, PROD["dt"], D, X_LO, X_HI, m=PROD["m"], knots=exps)
        diffs = [abs(price_backward(tab, float(kt[a][j]), tf[a], bool(ct[a][j]), PROD["dt"], D, X_LO, X_HI, m=PROD["m"],
                                    knots=exps) - fw[a][j]) * S.D[4 * a] * S.F[4 * a] for a in range(2) for j in range(3)]
        print(f"   6 options at {tf[0] * 365:.1f}d and {tf[1] * 365:.1f}d: max |forward - backward| ${max(diffs):.1e}")
        print()
        R.update({name: f.result() for name, f in fut.items()})
    if cache:
        pickle.dump(R, open(cache, "wb"))
    print(f"   (heavy runs: {len(jobs)} forward passes in {time.time() - t0:.0f} s on 4 processes; production run "
          f"{R['prod'][1]:.0f} s)")
    print()

    # 3. Round trip
    def errs(out):
        return [(vols(out[i], ks[i], cl[i], S.T[i]) - S.implied_vol(ks[i], S.T[i])) * 100 for i in range(n)]
    E = errs(R["prod"][0])
    EF = errs(R["filled"][0])
    Ed = [(R["prod"][0][i] - black_c(ks[i], S.implied_vol(ks[i], S.T[i]) ** 2 * S.T[i], cl[i])) * S.D[i] * S.F[i] for i in range(n)]
    liq = [q["liq"] for q in Q]
    print(f"3. Round trip: OTM quotes priced under the local vol, back to implied vol, minus the Stage 1 surface")
    print(f"   (vol points). Liquid = mid >= ${LIQ_MID:.2f}; requirement 0.1 vol points there.")
    print("   expiry          days  liquid  max    rms     $max   | all quotes  max   (k range)    | rejected fill max")
    for i in range(n):
        e, l = E[i], liq[i]
        el = np.abs(e[l]) if l.any() else np.array([np.nan])
        print(f"   {S.names[i]}  {S.T[i] * 365:6.1f}  {l.sum():5d}  {np.nanmax(el):5.3f}  {np.sqrt(np.nanmean(el ** 2)):5.3f}  "
              f"{np.max(np.abs(Ed[i][l])) if l.any() else np.nan:7.4f}  |  {len(e):4d}  {np.nanmax(np.abs(e)):6.3f}  "
              f"({ks[i].min():+.2f}..{ks[i].max():+.2f})  |  {np.nanmax(np.abs(EF[i][l])) if l.any() else np.nan:6.3f}")
    a = np.concatenate([E[i][liq[i]] for i in range(n)])
    af = np.concatenate([EF[i][liq[i]] for i in range(n)])
    allq = np.concatenate(E)
    print(f"   liquid ({a.size} quotes): max {np.nanmax(np.abs(a)):.3f}, rms {np.sqrt(np.nanmean(a ** 2)):.3f}, "
          f"95th pct {np.nanpercentile(np.abs(a), 95):.3f} vol points; over 0.1: {np.mean(np.abs(a) > 0.1):.1%}  "
          f"-> {'PASS' if np.nanmax(np.abs(a)) < 0.1 else 'FAIL'}")
    print(f"   all quotes ({allq.size}, incl. sub-50c wings): max {np.nanmax(np.abs(allq)):.3f}, "
          f"rms {np.sqrt(np.nanmean(allq ** 2)):.3f}; NaN (price outside bounds) {np.isnan(allq).sum()}")
    print(f"   rejected wing fill (Stage 2 first version): liquid max {np.nanmax(np.abs(af)):.3f}, "
          f"rms {np.sqrt(np.nanmean(af ** 2)):.3f}, over 0.1: {np.mean(np.abs(af) > 0.1):.1%}")
    print("   error map: docs/lv_roundtrip.png")
    print()
    plot_roundtrip(S, Q, E, EF)

    # 4. QuantLib
    import QuantLib as ql
    grids = [(1000, 1000), (2000, 2000)]
    t0 = time.time()
    QL = ql_prices(tab, S, qc, grids)
    tql = time.time() - t0
    b1 = np.concatenate(R["ql1"][0])
    b2 = np.concatenate(R["ql2"][0])
    kq = np.array([c[1] for c in qc]); cq = np.array([c[2] for c in qc]); Tq = np.array([c[0] / 365 for c in qc])
    qD = np.array(qD)
    v = lambda c: vols(c, kq, cq, Tq) * 100
    print(f"4. QuantLib {ql.__version__} FdBlackScholesVanillaEngine(localVol=True, Crank-Nicolson, 4 damping steps) on")
    print(f"   the same local-vol table (FixedLocalVolSurface), spot 1 and zero rates (forward units), "
          f"{len(qc)} contracts at {len(qT)} expiries")
    print("   (whole days), 5 liquid OTM strikes each; dollars = D_T F_T x forward units. QuantLib at "
          f"{grids[-1][0]} time x {grids[-1][1]} space")
    print("   steps; its own change from the coarser grid shows how far it is converged.")
    print("   days    k      type   QuantLib $   blur $      diff $    vol pts  | blur at half the step, vs QL | QL grid change")
    for j, (days, k, call) in enumerate(qc):
        print(f"   {days:4d}  {k:+.3f}  {'call' if call else 'put '}  {QL[j, -1] * qD[j]:10.4f}  {b1[j] * qD[j]:10.4f}  "
              f"{(b1[j] - QL[j, -1]) * qD[j]:+9.5f}  {v(b1)[j] - v(QL[:, -1])[j]:+7.4f}  |  {v(b2)[j] - v(QL[:, -1])[j]:+7.4f}"
              f"                    |  {v(QL[:, -1])[j] - v(QL[:, 0])[j]:+7.4f}")
    dv, dv2, dq = v(b1) - v(QL[:, -1]), v(b2) - v(QL[:, -1]), v(QL[:, -1]) - v(QL[:, 0])
    print(f"   blur (production) vs QuantLib: max {np.max(np.abs(dv)):.4f} vol points, max ${np.max(np.abs(b1 - QL[:, -1]) * qD):.4f}")
    print(f"   blur at half the time step vs QuantLib: max {np.max(np.abs(dv2)):.4f} vol points, "
          f"max ${np.max(np.abs(b2 - QL[:, -1]) * qD):.4f}")
    print(f"   QuantLib's own grid change ({grids[0][0]} -> {grids[1][0]}): max {np.max(np.abs(dq)):.4f} vol points;  "
          f"QuantLib time {tql:.0f} s")
    print()

    # 5. Convergence
    print("5. Convergence: liquid-quote implied vols at each setting minus the finest one, vol points (max | rms),")
    print("   by expiry group; then the observed order from successive differences")
    groups = [("3-7d", S.T <= 7.5 / 365), ("10-42d", (S.T > 7.5 / 365) & (S.T <= 45 / 365)), ("56d+", S.T > 45 / 365)]

    def ivs(out):
        return [vols(out[i], ks[i], cl[i], S.T[i])[liq[i]] * 100 for i in range(n)]

    for title, names, labels, ratio in [
            (f"time (grid dx {D}):", ["t4", "t2", "prod", "t05"], ["4d/8", "2d/16", "1d/32", "0.5d/64"], 2.0),
            (f"space (steps 1d, >= 32 per interval):", ["x16", "x11", "prod", "x057"],
             ["dx .0016", "dx .00113", "dx .0008", "dx .00057"], np.sqrt(2))]:
        print(f"   {title}  (step <= / min steps per interval)" if ratio == 2 else f"   {title}")
        V = {nm: ivs(R[nm][0]) for nm in names}
        fin = V[names[-1]]
        succ = {g: [] for g, _ in groups}
        for nm, lab in zip(names, labels):
            line = f"     {lab:10s}"
            for g, msk in groups:
                d = np.concatenate([V[nm][i] - fin[i] for i in range(n) if msk[i]])
                line += f"   {g} {np.nanmax(np.abs(d)):.4f} | {np.sqrt(np.nanmean(d ** 2)):.4f}"
            print(line + f"   ({R[nm][1]:.0f} s)")
        for g, msk in groups:
            ds = [np.sqrt(np.nanmean(np.concatenate([V[names[j + 1]][i] - V[names[j]][i] for i in range(n) if msk[i]]) ** 2))
                  for j in range(len(names) - 1)]
            orders = [np.log(ds[j] / ds[j + 1]) / np.log(ratio) if ds[j + 1] > 0 else np.inf for j in range(len(ds) - 1)]
            print(f"     {g:7s} rms successive differences " + "  ".join(f"{x:.1e}" for x in ds) +
                  "   observed order " + "  ".join(f"{o:.2f}" for o in orders))
    print(f"   domain: ln(S/F) on [{X_LO - 1:g}, {X_HI + 1:g}] instead of [{X_LO}, {X_HI}] (same steps and dx), change in vol points:")
    Vw, Vp = ivs(R["wide"][0]), ivs(R["prod"][0])
    line = "     "
    for g, msk in groups:
        d = np.concatenate([Vw[i] - Vp[i] for i in range(n) if msk[i]])
        line += f"   {g} {np.nanmax(np.abs(d)):.1e} | {np.sqrt(np.nanmean(d ** 2)):.1e}"
    print(line + f"   ({R['wide'][1]:.0f} s)")
    print()


def plot_roundtrip(S, Q, E, EF):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    mono = dict(fontsize=9, family="monospace")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, EE, title in ((axes[0], E, "01 ROUND TRIP: LOCAL VOL PRICE -> IMPLIED VOL - SURFACE"),
                          (axes[1], EF, "02 SAME WITH THE REJECTED WING FILL")):
        for i in range(len(S.T)):
            q = Q[i]
            d = np.full(len(q["k"]), S.T[i] * 365)
            ax.scatter(q["k"][~q["liq"]], d[~q["liq"]], c="#333333", s=4, marker="s", lw=0)
            sc = ax.scatter(q["k"][q["liq"]], d[q["liq"]], c=EE[i][q["liq"]], cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                            s=9, marker="s", lw=0)
        ax.set_yscale("log")
        ax.set_xlim(-1.0, 0.45)
        ax.set_title(title, **mono)
        ax.set_xlabel("k = ln(K / F_T)", family="monospace")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("days to expiry (log)", family="monospace")
    fig.colorbar(sc, ax=axes, label="error, vol points (colour stops at +-0.1)", shrink=0.9)
    axes[0].text(0.02, 0.02, "grey = quote under $0.50 (not in the requirement)", transform=axes[0].transAxes,
                 fontsize=8, family="monospace", color="#aaaaaa")
    fig.savefig(HERE.parent / "docs" / "lv_roundtrip.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
