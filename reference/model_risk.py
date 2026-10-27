"""Part 4 Stage 4: model risk. Part 1's barrier options on SPY under three models that all fit the vanillas.

python reference/model_risk.py [data/spy_full_<date>]
  -> report on stdout (saved as reference/model_risk_results.txt), docs/model_risk.png,
     <folder>/model_risk.json (the table, for the demo)

The options are Part 1's: all eight single-barrier types, a down barrier at 90% of spot (strikes 80% and
100%) and an up barrier at 115% (strikes 100% and 125%), monitored continuously, scaled to SPY's spot on
Oct 23 2026 and priced at two listed expiries (about 3 months and about 1 year), on SPY's own forwards and
discount factors. Prices are per share, in dollars.

Models:
  bs      Black-Scholes at the at-the-money implied vol of the expiry (the Stage 1 surface at k = 0).
  merton  one Merton jump-diffusion (sigma, lam, m, delta) fitted to the Stage 1 surface at the liquid
          quotes of every expiry up to the longer barrier's, each expiry weighted equally.
  lv      Stage 2's Dupire local vol (Stage 3's table), which reprices every liquid vanilla within 0.1 vol.
All three are priced by the same blur on the same grid and time steps (barrier_models.py).

Checks: Black-Scholes barriers against the Reiner-Rubinstein closed form; Merton vanillas against
the Merton series and Merton barriers against Monte Carlo; local-vol barriers against QuantLib's
finite-difference barrier engine with the same local vol; convergence in time step and grid.
"""
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import local_vol as LVM
from lv_blur import LVTable, Flat, implied_w, black_c
from barrier_models import Forward, Merton, price
from barriers import barrier_closed_form, vanilla as bs_vanilla, TYPES
from kernels import merton_series

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
DAY = 1 / 365
X_LO, X_HI = -2.5, 1.5
PROD = dict(dt=1 * DAY, m=32, dx=0.0008)
PICK = (18, 26)                                    # 2027-01-29 (98 days) and 2027-09-30 (342 days)
BARRIERS = [(0.90, "down", (0.80, 1.00)), (1.15, "up", (1.00, 1.25))]
LIQ_MID = 0.50


def market():
    S = LVM.Surface.load(FOLDER)
    s1 = json.load(open(FOLDER / "stage1.json"))
    return S, s1, Forward(s1["spot_implied"], S.T, S.F)


def payoffs(S0, FT):
    """(k, call) for the four vanillas of each barrier: strikes x call/put."""
    return {side: [(float(np.log(K * S0 / FT)), c) for K in Ks for c in (True, False)] for _, side, Ks in BARRIERS}


def model_of(name, params, S):
    if name == "lv":
        return LVTable.from_surface(S)
    if name == "merton":
        return Merton(*params)
    return Flat(params)


def run(job):
    """One backward pass in a worker: (model name, params, expiry index, barrier side or None, dt, m, dx)."""
    name, params, i, side, dt, m, dx = job
    S, s1, fwd = market()
    S0, T, FT = s1["spot_implied"], S.T[i], S.F[i]
    pays = payoffs(S0, FT)[side or "down"] + ([] if side else payoffs(S0, FT)["up"])
    bar = None
    if side:
        frac = [b for b, sd, _ in BARRIERS if sd == side][0]
        bar = (frac * S0, side)
    t0 = time.time()
    v = price(model_of(name, params, S), T, pays, fwd, dt, dx, X_LO, X_HI, barrier=bar, m=m, knots=list(S.T))
    return v * S.D[i] * FT, time.time() - t0


def fit_merton(S, s1, upto):
    """One Merton for all expiries up to index `upto`, on the surface's vols at the liquid OTM quote strikes."""
    ks, Ts, cl, wt, iv = [], [], [], [], []
    for i in range(upto + 1):
        e, F = S.names[i], S.F[i]
        cs = [c for c in s1["contracts"] if c["expiry"] == e and c["iv_true"] is not None
              and (c["cp"] == "call") == (c["K"] >= F) and 0.5 * (c["bid"] + c["ask"]) >= LIQ_MID]
        k = np.array([np.log(c["K"] / F) for c in cs])
        ks.append(k)
        Ts.append(np.full(len(k), S.T[i]))
        cl.append(np.array([c["cp"] == "call" for c in cs]))
        wt.append(np.full(len(k), 1 / np.sqrt(max(len(k), 1))))
        iv.append(S.implied_vol(k, S.T[i]))
    ks, Ts, cl, wt, iv = map(np.concatenate, (ks, Ts, cl, wt, iv))

    def err(p):
        M = Merton(*p)
        c = np.concatenate([M.vanilla(ks[Ts == T], T, cl[Ts == T]) for T in np.unique(Ts)])
        order = np.concatenate([np.nonzero(Ts == T)[0] for T in np.unique(Ts)])
        v = np.empty_like(c)
        v[order] = c
        return (np.sqrt(implied_w(v, ks, cl) / Ts) - iv) * 100

    r = least_squares(lambda p: np.nan_to_num(err(p), nan=50.0) * wt, [0.10, 1.0, -0.08, 0.06],
                      bounds=([0.01, 0.0, -0.5, 0.005], [0.6, 20.0, 0.2, 0.5]), x_scale=[0.05, 1.0, 0.05, 0.03])
    e = err(r.x)
    by = {float(T): (float(np.sqrt(np.mean(e[Ts == T] ** 2))), float(np.max(np.abs(e[Ts == T])))) for T in np.unique(Ts)}
    return tuple(float(x) for x in r.x), e, by


def merton_mc(M, S0, fwd, T, D, K, call, H, side, n_paths=400_000, h=0.25 * DAY, seed=7):
    """Monte Carlo knock-out with the diffusion's Brownian-bridge crossing probability per step and jumps at
    step ends (a jump past the barrier knocks out). Control variate: the vanilla, known exactly."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(T / h))
    h = T / n
    s = M.sigma * np.sqrt(h)
    mu = -0.5 * s * s - M.lam * M.kappa * h
    out, van = [], []
    for chunk in range(n_paths // 50_000):
        x = np.zeros(50_000)
        alive = np.ones(50_000, bool)
        for j in range(n):
            t0, t1 = j * h, (j + 1) * h
            b0, b1 = np.log(H / fwd(t0)), np.log(H / fwd(t1))
            y = x + mu + s * rng.standard_normal(x.size)
            d0, d1 = (x - b0, y - b1) if side == "down" else (b0 - x, b1 - y)
            p_surv = np.where((d0 > 0) & (d1 > 0), -np.expm1(-2 * d0 * np.maximum(d1, 0) / (s * s)), 0.0)
            alive &= rng.random(x.size) < p_surv
            nj = rng.poisson(M.lam * h, x.size)
            y = y + nj * M.m + np.sqrt(nj) * M.delta * rng.standard_normal(x.size)
            d1 = (y - b1) if side == "down" else (b1 - y)
            alive &= d1 > 0
            x = y
        kk = np.log(K / fwd(T))
        pay = np.maximum(np.exp(x) - np.exp(kk), 0) if call else np.maximum(np.exp(kk) - np.exp(x), 0)
        out.append(pay * alive)
        van.append(pay)
    out, van = np.concatenate(out), np.concatenate(van)
    exact = float(M.vanilla(np.array([np.log(K / fwd(T))]), T, np.array([call]))[0])
    beta = np.cov(out, van)[0, 1] / np.var(van)
    est = out - beta * (van - exact)
    scale = D * fwd(T)
    return float(est.mean() * scale), float(est.std() / np.sqrt(est.size) * scale)


def ql_lv_barriers(tab, T_days, Hs, cases):
    """QuantLib FdBlackScholesBarrierEngine(localVol=True) in forward units (spot 1, zero rates, so the barrier
    is a constant level): same local-vol table as in check_lv_blur.ql_prices."""
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
    strikes = [float(v) for v in np.exp(tab.k)]
    Mx = ql.Matrix(len(strikes), len(times))
    for c, row in enumerate(rows):
        for r_, v in enumerate(row):
            Mx[r_][c] = float(v)
    lv = ql.FixedLocalVolSurface(ref, [float(t) for t in times], strikes, Mx, dc,
                                 ql.FixedLocalVolSurface.ConstantExtrapolation, ql.FixedLocalVolSurface.ConstantExtrapolation)
    lv.enableExtrapolation()
    zero = ql.YieldTermStructureHandle(ql.FlatForward(ref, 0.0, dc))
    bv = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref, ql.NullCalendar(), 1.0, dc))
    proc = ql.GeneralizedBlackScholesProcess(ql.QuoteHandle(ql.SimpleQuote(1.0)), zero, zero, bv,
                                             ql.LocalVolTermStructureHandle(lv))
    out = []
    for (H, side, K, call) in cases:
        typ = ql.Barrier.DownOut if side == "down" else ql.Barrier.UpOut
        opt = ql.BarrierOption(typ, H, 0.0, ql.PlainVanillaPayoff(ql.Option.Call if call else ql.Option.Put, K),
                               ql.EuropeanExercise(ref + int(T_days)))
        row = []
        for tg, xg in ((1000, 1000), (2000, 2000)):
            opt.setPricingEngine(ql.FdBlackScholesBarrierEngine(proc, tg, xg, 4, ql.FdmSchemeDesc.CrankNicolson(), True))
            row.append(opt.NPV())
        out.append(row)
    return np.array(out)


def lv_flat_forward_job(job):
    """Local-vol knock-outs with F constant (spot 1, zero rates) for the QuantLib comparison."""
    T, cases, dt, m, dx = job
    S, s1, _ = market()
    tab = LVTable.from_surface(S)
    one = lambda t: 1.0
    out = []
    for H, side, K, call in cases:
        out.append(float(price(tab, T, [(float(np.log(K)), call)], one, dt, dx, X_LO, X_HI, barrier=(H, side), m=m,
                               knots=list(S.T))[0]))
    return np.array(out)


def name_of(side, inout, call):
    return f"{side}-and-{inout} {'call' if call else 'put'}"


def main():
    S, s1, fwd = market()
    S0 = s1["spot_implied"]
    print(f"Stage 4: Part 1's barrier options on SPY ({FOLDER.name}, spot ${S0:.2f}) under three models. "
          f"Python float64.")
    print(f"Grid: dx = {PROD['dx']} in ln(S/F) on [{X_LO}, {X_HI}]; time steps <= {PROD['dt'] * 365:g} day and at "
          f"least {PROD['m']} per expiry interval; continuous monitoring (Brownian-bridge weights).")
    print()

    # models
    t0 = time.time()
    mp, merr, by = fit_merton(S, s1, max(PICK))
    atm = {i: float(S.implied_vol(np.array([0.0]), S.T[i])[0]) for i in PICK}
    print("1. The three models")
    for i in PICK:
        print(f"   Black-Scholes, {S.names[i]} ({S.T[i] * 365:.0f} days): at-the-money implied vol {atm[i]:.2%}")
    print(f"   Merton, one fit to the surface at the {merr.size} liquid OTM quotes of all expiries to "
          f"{S.names[max(PICK)]} (each expiry weighted equally, {time.time() - t0:.0f} s):")
    print(f"     diffusion vol {mp[0]:.2%}, {mp[1]:.2f} jumps a year, mean jump {mp[2]:+.2%}, jump vol {mp[3]:.2%}")
    print(f"     fit: RMS {np.sqrt(np.mean(merr ** 2)):.2f} vol points, max {np.max(np.abs(merr)):.2f}; by expiry (RMS / max):")
    print("     " + "  ".join(f"{T * 365:.0f}d {r:.2f}/{mx:.2f}" for T, (r, mx) in list(by.items())))
    print("   Local vol: Stage 2's Dupire surface as tabulated in Stage 3 (round trip within 0.073 vol points on every "
          "liquid quote).")
    print()

    # heavy runs
    jobs = {}
    for i in PICK:
        for name, params in (("bs", atm[i]), ("merton", mp), ("lv", None)):
            for side in (None, "down", "up"):
                for tag, (dt, m, dx) in {"prod": (PROD["dt"], PROD["m"], PROD["dx"]),
                                         "dt2": (2 * PROD["dt"], PROD["m"] // 2, PROD["dx"]),
                                         "dt05": (0.5 * PROD["dt"], 2 * PROD["m"], PROD["dx"]),
                                         "dx2": (PROD["dt"], PROD["m"], 2 * PROD["dx"])}.items():
                    if tag != "prod" and side is None:
                        continue
                    jobs[(i, name, side, tag)] = (name, params, i, side, dt, m, dx)
    # QuantLib comparison cases: forward units, the same barriers and strikes, 98 days
    qcases = [(b, side, K, c) for b, side, Ks in BARRIERS for K in Ks for c in (True, False)]
    T98 = int(np.floor(S.T[PICK[0]] * 365))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        order = sorted(jobs, key=lambda j: (j[1] != "lv", -j[0]))
        fut = {j: ex.submit(run, jobs[j]) for j in order}
        fq = ex.submit(lv_flat_forward_job, (T98 / 365, qcases, PROD["dt"], PROD["m"], PROD["dx"]))
        fq2 = ex.submit(lv_flat_forward_job, (T98 / 365, qcases, 0.5 * PROD["dt"], 2 * PROD["m"], PROD["dx"]))

        # 2. checks that run here while the pool works
        print("2. Checks")
        print("   (a) Black-Scholes barriers against Reiner-Rubinstein (constant r = 4.2%, q = 1.3%, vol 15%; spot $760.50;")
        print("       16 options x 2 expiries; dollars)")
        r_, q_, sg = 0.042, 0.013, 0.15
        fc = lambda t: S0 * np.exp((r_ - q_) * t)
        worst = 0.0
        for i in PICK:
            T = S.T[i]
            for frac, side, Ks in BARRIERS:
                pays = [(float(np.log(K * S0 / fc(T))), c) for K in Ks for c in (True, False)]
                v = price(Flat(sg), T, pays, fc, PROD["dt"], PROD["dx"], X_LO, X_HI, barrier=(frac * S0, side), m=1)
                v = v * np.exp(-r_ * T) * fc(T)
                cf = np.array([barrier_closed_form(name_of(side, "out", c), S0, K * S0, frac * S0, sg, T, r_, q_)
                               for K in Ks for c in (True, False)])
                worst = max(worst, float(np.max(np.abs(v - cf))))
        print(f"       max |blur - closed form| ${worst:.1e} (knock-outs; knock-ins are vanilla minus these)")
        print("   (b) Merton vanillas against the Merton series (forward units x D F; the fitted model; 8 strikes x 2 expiries)")
        M = Merton(*mp)
        worst = 0.0
        for i in PICK:
            T, FT, D = S.T[i], S.F[i], S.D[i]
            ks = np.log(np.array([0.8, 0.9, 1.0, 1.1, 1.25]) * S0 / FT)
            for k in ks:
                for c in (True, False):
                    b = price(M, T, [(float(k), c)], fwd, PROD["dt"], PROD["dx"], X_LO, X_HI, m=1)[0]
                    ref = merton_series(1.0, float(np.exp(k)), T, 0.0, *mp, cp="call" if c else "put")
                    worst = max(worst, abs(b - ref) * D * FT)
        print(f"       max |blur - series| ${worst:.1e}")
        print("   (c) Merton knock-outs against Monte Carlo (400,000 paths, quarter-day steps, bridge crossing per step,")
        print("       vanilla control variate), 98 days:")
        i = PICK[0]
        for frac, side, Ks in BARRIERS:
            for K in Ks:
                for c in (True, False):
                    if (side == "down" and not c and K < frac) or (side == "up" and c and K > frac):
                        continue                              # worthless by construction
                    mc, se = merton_mc(M, S0, fwd, S.T[i], S.D[i], K * S0, c, frac * S0, side)
                    b = price(M, S.T[i], [(float(np.log(K * S0 / S.F[i])), c)], fwd, PROD["dt"], PROD["dx"], X_LO, X_HI,
                              barrier=(frac * S0, side), m=1)[0] * S.D[i] * S.F[i]
                    print(f"       {name_of(side, 'out', c):18s} K {K:.0%}  blur ${b:9.4f}   MC ${mc:9.4f} +- {2 * se:.4f} (2 se)  "
                          f"diff {(b - mc) / se:+.1f} se")
        R = {j: f.result() for j, f in fut.items()}
        QB, QB2 = fq.result(), fq2.result()
    print(f"   (heavy runs: {len(jobs) + 2} passes in {time.time() - t0:.0f} s on 4 processes)")

    tab = LVTable.from_surface(S)
    QL = ql_lv_barriers(tab, T98, None, qcases)
    print(f"   (d) Local-vol knock-outs against QuantLib FdBlackScholesBarrierEngine(localVol=True, Crank-Nicolson) on")
    print(f"       the same table, forward units (spot 1, zero rates: the barrier is a fixed level), {T98} days;")
    print("       dollars = x D F of the expiry. QuantLib at 2000 x 2000; its change from 1000 x 1000 shown.")
    DF = S.D[PICK[0]] * S.F[PICK[0]]
    for j, (H, side, K, c) in enumerate(qcases):
        if QL[j, 1] < 1e-9 and QB[j] < 1e-9:
            continue
        print(f"       {name_of(side, 'out', c):18s} H {H:.2f} K {K:.2f}   QuantLib ${QL[j, 1] * DF:9.4f}   blur ${QB[j] * DF:9.4f}"
              f"   diff ${(QB[j] - QL[j, 1]) * DF:+.4f} ({(QB[j] / QL[j, 1] - 1):+.2%})   half step ${(QB2[j] - QL[j, 1]) * DF:+.4f}"
              f"   QL grid ${(QL[j, 1] - QL[j, 0]) * DF:+.4f}")
    print()

    # 3. the table
    table = {}
    print("3. Model risk: the same options, three models. Dollars per share; spread = max - min over the models, and")
    print("   as a percentage of the local-vol price.")
    for i in PICK:
        T, FT, D = S.T[i], S.F[i], S.D[i]
        print(f"   {S.names[i]} ({T * 365:.0f} days), forward ${FT:.2f}")
        van = {nm: dict(zip(["down", "up"], np.split(R[(i, nm, None, "prod")][0], 2))) for nm in ("bs", "merton", "lv")}
        print("   vanillas (the market is the Stage 1 surface):")
        print("     strike  type   market $     BS-ATM $   Merton $   local vol $")
        seen = set()
        for frac, side, Ks in BARRIERS:
            pays = payoffs(S0, FT)[side]
            for j, (k, c) in enumerate(pays):
                Kf = Ks[j // 2]
                if (Kf, c) in seen:
                    continue
                seen.add((Kf, c))
                mk = black_c(k, S.implied_vol(np.array([k]), T)[0] ** 2 * T, c) * D * FT
                print(f"     {Kf:5.0%}   {'call' if c else 'put '}  {mk:9.3f}   {van['bs'][side][j]:9.3f}  "
                      f"{van['merton'][side][j]:9.3f}   {van['lv'][side][j]:9.3f}")
        print("     type                 barrier strike    BS-ATM $   Merton $  local vol $   spread $   spread %")
        rows = []
        for frac, side, Ks in BARRIERS:
            for inout in ("out", "in"):
                for j, (K, c) in enumerate([(K, c) for K in Ks for c in (True, False)]):
                    vals = {}
                    for nm in ("bs", "merton", "lv"):
                        o = R[(i, nm, side, "prod")][0][j]
                        vals[nm] = o if inout == "out" else van[nm][side][j] - o
                    sp = max(vals.values()) - min(vals.values())
                    pct = sp / vals["lv"] if vals["lv"] > 1e-6 else np.nan
                    rows.append(dict(type=name_of(side, inout, c), barrier=frac, strike=K, **{k_: float(v_) for k_, v_ in vals.items()},
                                     spread=float(sp), spread_pct=None if np.isnan(pct) else float(pct)))
                    print(f"     {name_of(side, inout, c):20s} {frac:5.0%}  {K:5.0%}   {vals['bs']:9.3f}  {vals['merton']:9.3f}"
                          f"  {vals['lv']:9.3f}   {sp:8.3f}   " + (f"{pct:7.1%}" if not np.isnan(pct) else "     --"))
        table[S.names[i]] = dict(days=round(T * 365), forward=FT, rows=rows)
        print()

    # 4. convergence
    print("4. Convergence: largest change in any knock-out price (dollars) from the production setting when")
    print("   the time step is doubled or halved, or the grid step doubled")
    for i in PICK:
        for nm in ("bs", "merton", "lv"):
            line = f"   {S.names[i]} {nm:7s}"
            for tag, lab in (("dt2", "step x2"), ("dt05", "step /2"), ("dx2", "dx x2")):
                d = max(float(np.max(np.abs(R[(i, nm, sd, tag)][0] - R[(i, nm, sd, "prod")][0]))) for sd in ("down", "up"))
                line += f"   {lab} ${d:.4f}"
            secs = sum(R[(i, nm, sd, "prod")][1] for sd in (None, "down", "up"))
            print(line + f"   ({secs:.0f} s for the 3 production passes)")
    print()

    json.dump(dict(note="Part 4 Stage 4: Part 1 barrier options on SPY under BS (ATM vol), Merton (fitted), local vol; "
                        "dollars per share; continuous monitoring", spot=S0, merton=dict(zip(["sigma", "lam", "m", "delta"], mp)),
                   atm_vol={S.names[i]: atm[i] for i in PICK}, expiries=table),
              open(FOLDER / "model_risk.json", "w"), indent=1)
    print(f"wrote {(FOLDER / 'model_risk.json').relative_to(HERE.parent)}")
    plot(table)


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    mono = dict(fontsize=9, family="monospace")
    fig, axes = plt.subplots(1, len(table), figsize=(15, 6.5), sharex=False)
    for ax, (name, t) in zip(np.atleast_1d(axes), table.items()):
        rows = [r for r in t["rows"] if max(r["bs"], r["merton"], r["lv"]) > 0.01]
        y = np.arange(len(rows))
        for nm, mk, col in (("bs", "o", "#888888"), ("merton", "s", "#bbbbbb"), ("lv", "D", "#ffffff")):
            ax.scatter([r[nm] for r in rows], y, marker=mk, s=22, facecolors="none" if nm != "lv" else col,
                       edgecolors=col, label={"bs": "BS at-the-money vol", "merton": "Merton (fitted)", "lv": "local vol"}[nm])
        for yy, r in zip(y, rows):
            lo, hi = min(r["bs"], r["merton"], r["lv"]), max(r["bs"], r["merton"], r["lv"])
            ax.plot([lo, hi], [yy, yy], color="white", lw=0.6, alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['type'].upper()}  H {r['barrier']:.0%}  K {r['strike']:.0%}" for r in rows], **mono)
        ax.set_xscale("log")
        ax.set_xlabel("price, dollars per share (log)", family="monospace")
        ax.set_title(f"{name} ({t['days']} DAYS): SAME VANILLAS, DIFFERENT BARRIERS", **mono)
        ax.grid(alpha=0.15)
        ax.invert_yaxis()
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE.parent / "docs" / "model_risk.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
