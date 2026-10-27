"""Part 4: a second, independent arbitrage-free SVI surface for the same SPY quotes.

python reference/svi_surface_alt.py [data/spy_full_<date>]  -> report on stdout, <folder>/svi_surface_alt.json

Built separately from Stage 1's surface (reference/svi_surface.py) with a different repair:
an SSVI backbone, a weak prior on the SVI wing slopes, and slice-by-slice constrained refits.
It fits the quotes about as well (0.29 vs 0.25 vol points RMS on quotes worth 50c+). Stage 2
(local_vol.py) uses it only as evidence: where Dupire on the two surfaces disagrees, the local
vol is not pinned down by the quotes.

Input is stage1.json (chain_stage1.py): parity forwards and, per contract, the
European-equivalent implied vol (mid minus QuantLib's early-exercise premium,
inverted on the parity forward). Only the out-of-the-money side is used.

Coordinates: log-moneyness k = ln(K / F_T) and total implied variance
w(k, T) = sigma_imp(k, T)^2 T. Each expiry gets a raw SVI slice

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2)).

Fit: least squares on implied vol, each quote's error divided by its own
half-spread in vol (half bid-ask over vega, floored at 0.25 vol points), so a
liquid at-the-money quote counts far more than a one-cent wing quote.

Arbitrage checks on a fixed grid k in [-3, 1.5] (step 0.001), far past every
listed strike, plus the wing slopes beyond it:
  butterfly: the risk-neutral density, which is the second strike derivative of
    the call price (the "edge detect" of the price curve), must be >= 0. For SVI
    it equals a positive factor times Gatheral's
        g(k) = (1 - k w'/(2w))^2 - w'^2/4 (1/w + 1/4) + w''/2,
    so we check g >= 0 and, independently, finite second differences of Black
    call prices on the strike grid. Roger Lee's moment bound b (1 + |rho|) <= 2
    keeps it valid past the grid (SVI wings are straight lines).
  calendar: total variance must not decrease with expiry at any fixed k
    (w(k, T_i) <= w(k, T_{i+1})).

Repair: refit each slice from the shortest expiry up, with the butterfly
condition g >= 0 on the grid, Lee's bound, and w_i >= w_{i-1} on the grid as
hard inequality constraints (SLSQP), starting from the unconstrained fit.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import ndtr
from scipy.stats import norm

HERE = Path(__file__).parent
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
OUT = folder / "svi_surface_alt.json"

KGRID = np.round(np.arange(-3.0, 1.5 + 1e-12, 0.001), 6)
KFAR = np.array([-1e4, -1e3, -100.0, -30.0, -10.0, -5.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e3, 1e4])   # past the grid
KALL = np.concatenate([KGRID, KFAR])
VOL_FLOOR = 0.0025                       # half-spread floor in the fit weights: 0.25 vol points
G_EPS = 1e-6                             # constrained fits keep g and calendar gaps at least this far from zero
CAL_EPS = 1e-7


def svi(p, k):
    a, b, rho, m, s = p
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + s * s))


def svi_d(p, k):
    """w, w', w'' of raw SVI."""
    a, b, rho, m, s = p
    x = k - m
    r = np.sqrt(x * x + s * s)
    return a + b * (rho * x + r), b * (rho + x / r), b * s * s / r**3


def g_fun(p, k):
    w, w1, w2 = svi_d(p, k)
    return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2


def black_call(F, K, w, D):
    v = np.sqrt(w)
    d1 = (np.log(F / K) + 0.5 * w) / v
    return D * (F * ndtr(d1) - K * ndtr(d1 - v))


def black_put(F, K, w, D):
    v = np.sqrt(w)
    d1 = (np.log(F / K) + 0.5 * w) / v
    return D * (K * ndtr(-(d1 - v)) - F * ndtr(-d1))


def fd_density(p, F, D, k):
    """Risk-neutral density in K from second differences of SVI Black prices (independent of g). Puts below the
    forward and calls above (same second derivative by parity): out-of-the-money prices keep the rounding error
    far below the density even in the far wings."""
    K = F * np.exp(k)
    h = np.diff(K)
    d2 = lambda C: 2 * ((C[2:] - C[1:-1]) / h[1:] - (C[1:-1] - C[:-2]) / h[:-1]) / (h[1:] + h[:-1])
    w = svi(p, k)
    dens = np.where(K[1:-1] < F, d2(black_put(F, K, w, D)), d2(black_call(F, K, w, D)))
    return dens / D, K[1:-1]


def load_slices(s1):
    out = []
    for e in s1["expiries"]:
        F, T, D = e["F"], e["T"], e["D"]
        cs = [c for c in s1["contracts"] if c["expiry"] == e["expiry"] and c["iv_true"] is not None
              and (c["cp"] == "call") == (c["K"] >= F)]
        K = np.array([c["K"] for c in cs])
        iv = np.array([c["iv_true"] for c in cs])
        k = np.log(K / F)
        sq = np.sqrt(T)
        vega = D * F * norm.pdf((-k + 0.5 * iv * iv * T) / (iv * sq)) * sq
        half = np.array([0.5 * (c["ask"] - c["bid"]) for c in cs])
        hv = np.maximum(half / np.maximum(vega, 1e-12), VOL_FLOOR)
        mid = np.array([0.5 * (c["ask"] + c["bid"]) for c in cs]) - np.array([c["eep"] or 0.0 for c in cs])
        o = np.argsort(k)
        out.append(dict(expiry=e["expiry"], T=T, F=F, D=D, r=e["r"], k=k[o], K=K[o], iv=iv[o], hv=hv[o],
                        mid=mid[o], vega=vega[o]))
    return out


def resid(p, sl):
    w = svi(p, sl["k"])
    return (np.sqrt(np.maximum(w, 1e-12) / sl["T"]) - sl["iv"]) / sl["hv"]


def resid_jac(p, sl):
    a, b, rho, m, s = p
    x = sl["k"] - m
    r = np.sqrt(x * x + s * s)
    w = np.maximum(a + b * (rho * x + r), 1e-12)
    dw = np.stack([np.ones_like(x), rho * x + r, b * x, -b * (rho + x / r), b * s / r], axis=1)
    return dw * (0.5 / np.sqrt(w * sl["T"]) / sl["hv"])[:, None]


BOUNDS = ([-1.0, 1e-6, -0.999, -1.0, 1e-4], [1.0, 5.0, 0.999, 1.0, 2.0])


def fit_free(sl):
    """Unconstrained raw SVI: several starts, best by weighted least squares."""
    T, k, iv = sl["T"], sl["k"], sl["iv"]
    atm = float(np.interp(0.0, k, iv)) ** 2 * T
    best = None
    for m0 in (-0.05, 0.0, 0.05):
        for s0 in (0.03, 0.2):
            for rho0 in (-0.6,):
                b0 = max(0.05 * np.sqrt(T), 1e-3)
                a0 = max(atm - b0 * s0, 1e-6)
                x0 = np.clip([a0, b0, rho0, m0 * np.sqrt(T) * 3, s0 * np.sqrt(T) * 2],
                             np.array(BOUNDS[0]) + 1e-9, np.array(BOUNDS[1]) - 1e-9)
                r = least_squares(resid, x0, jac=resid_jac, args=(sl,), bounds=BOUNDS, x_scale="jac",
                                  xtol=1e-12, ftol=1e-12, max_nfev=2000)
                if best is None or r.cost < best.cost:
                    best = r
    return best.x


# ---- SSVI backbone (Gatheral & Jacquier 2014): arbitrage-free by construction ----
#   w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),
#   phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma)).
# theta increasing in T, eta (1 + |rho|) <= 2 and 0 < gamma <= 1/2 rule out butterfly and calendar
# arbitrage everywhere (their Theorem 4.1, 4.2 and Remark 4.4). Each SSVI slice is a raw SVI slice.

def ssvi_unpack(x, n):
    theta = np.cumsum(np.exp(x[:n]))
    rho = 0.98 * np.tanh(x[n])
    eta = 2 / (1 + abs(rho)) / (1 + np.exp(-x[n + 1]))
    gamma = 0.5 / (1 + np.exp(-x[n + 2]))
    return theta, rho, eta, gamma


def ssvi_raw(theta, rho, eta, gamma):
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
    return np.array([theta / 2 * (1 - rho * rho), theta * phi / 2, rho, -rho / phi, np.sqrt(1 - rho * rho) / phi])


def fit_ssvi(slices):
    n = len(slices)
    atm = np.array([float(np.interp(0.0, s["k"], s["iv"])) ** 2 * s["T"] for s in slices])
    th0 = np.maximum.accumulate(atm) + np.arange(n) * 1e-7
    x0 = np.concatenate([np.log(np.diff(np.concatenate([[0.0], th0])) + 1e-8), [np.arctanh(-0.7 / 0.98), 1.0, 0.0]])

    def res(x):
        theta, rho, eta, gamma = ssvi_unpack(x, n)
        return np.concatenate([resid(ssvi_raw(t, rho, eta, gamma), s) for t, s in zip(theta, slices)])

    r = least_squares(res, x0, xtol=1e-12, ftol=1e-12, max_nfev=4000)
    theta, rho, eta, gamma = ssvi_unpack(r.x, n)
    return [ssvi_raw(t, rho, eta, gamma) for t in theta], dict(rho=float(rho), eta=float(eta), gamma=float(gamma),
                                                                theta=[float(t) for t in theta])


# ---- repair: refit slice by slice, shortest first, with the arbitrage conditions as hard constraints ----
KC = np.concatenate([KGRID[::4], KFAR])          # constraint points every 0.004 in k; the fine grid is the judge
WING_TOL = 0.5                                   # wing-slope prior: within ~50% of the SSVI backbone's slope


def objective_res(p, sl, prior):
    """Vol errors over half-spreads, plus a weak prior on the two wing slopes b (1 +- rho). Past the last quote
    nothing pins a raw SVI wing, and unpinned wings are where the unconstrained fits cross each other; the
    prior pulls them towards the SSVI backbone's slopes, which grow with expiry."""
    sr, sl_ = prior
    return np.concatenate([resid(p, sl), [(p[1] * (1 + p[2]) - sr) / (WING_TOL * sr),
                                          (p[1] * (1 - p[2]) - sl_) / (WING_TOL * sl_)]])


def wing_prior(b):
    return b[1] * (1 + b[2]), b[1] * (1 - b[2])


def feasible(p, lo, hi, k=KALL):
    w = svi(p, k)
    return (g_fun(p, k).min() >= 0 and p[1] * (1 + abs(p[2])) <= 2 and w.min() > 0
            and (lo is None or np.min(w - svi(lo, k)) >= 0) and (hi is None or np.min(svi(hi, k) - w) >= 0))


def refit_between(sl, p_ok, lo, hi, prior, starts=()):
    """Penalised least squares (penalties are zero inside the constraints) from p_ok, a slice known to be
    arbitrage-free here, and from each extra start; constraints first on a coarse grid, then on the fine grid if
    the coarse answer breaks one between its points. A result that still breaks a constraint on the fine grid is
    pulled back along the segment towards p_ok until it does not, so the answer is always arbitrage-free."""
    wsc = float(svi(p_ok, 0.0)) * 1e-3

    def res(q, lam, kc, wlo, whi):
        w = svi(q, kc)
        out = [objective_res(q, sl, prior), lam * np.maximum(2 * G_EPS - g_fun(q, kc), 0),
               [lam * max(q[1] * (1 + abs(q[2])) - 2 + 1e-6, 0)]]
        if wlo is not None:
            out.append(lam * np.maximum(wlo + 2 * CAL_EPS - w, 0) / wsc)
        if whi is not None:
            out.append(lam * np.maximum(w - whi + 2 * CAL_EPS, 0) / wsc)
        return np.concatenate(out)

    cost = lambda q: np.sum(objective_res(q, sl, prior) ** 2)
    best, c0 = p_ok, cost(p_ok)
    for x0 in (p_ok, *starts):
        for kc in (KC, KALL):
            args = (kc, svi(lo, kc) if lo is not None else None, svi(hi, kc) if hi is not None else None)
            x = x0
            for lam in (1e3, 1e6, 1e8):
                x = least_squares(res, np.clip(x, np.array(BOUNDS[0]) + 1e-12, np.array(BOUNDS[1]) - 1e-12),
                                  args=(lam, *args), bounds=BOUNDS, x_scale="jac", xtol=1e-12, ftol=1e-12,
                                  max_nfev=1000).x
                if feasible(x, lo, hi):
                    break
            if feasible(x, lo, hi):
                break
        t = 1.0
        while t > 1e-4:
            q = p_ok + t * (x - p_ok)
            if feasible(q, lo, hi):
                if cost(q) < c0:
                    best, c0 = q, cost(q)
                break
            t *= 0.5
    return best


def repair(slices, free, backbone):
    """Shortest expiry first. Each slice is fitted to its quotes plus the wing prior; if that fit has an arbitrage
    (butterfly on its own grid, Lee, or crossing the slice before it) it is refitted with those as constraints,
    starting both from the fit itself and from a point known to be arbitrage-free: the SSVI slice for the first
    expiry, the previous repaired slice lifted to this expiry's at-the-money level for the rest."""
    out, notes = [], []
    for i, sl in enumerate(slices):
        prev = out[-1] if out else None
        prior = wing_prior(backbone[i])
        best = None
        for x0 in (free[i], backbone[i]):
            r = least_squares(objective_res, np.clip(x0, np.array(BOUNDS[0]) + 1e-12, np.array(BOUNDS[1]) - 1e-12),
                              args=(sl, prior), bounds=BOUNDS, x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=3000)
            if best is None or r.cost < best.cost:
                best = r
        reg = best.x
        if feasible(reg, prev, None):
            out.append(reg)
            notes.append("wing prior only")
            continue
        if prev is None:
            p_ok = np.array(backbone[i], float)
        else:
            p_ok = np.array(prev, float)
            p_ok[0] += max(float(svi(reg, 0.0) - svi(prev, 0.0)), 0.0) + 1e-7
            if not feasible(p_ok, prev, None):
                raise RuntimeError(f"lifted previous slice is not arbitrage-free at {sl['expiry']}")
        out.append(refit_between(sl, p_ok, prev, None, prior, starts=(reg,)))
        notes.append("constrained refit")
    # One polishing sweep, longest expiry first: the forward pass only looked backwards, so a slice could be
    # held down by nothing yet pushed up by its predecessor. Now each constrained slice moves between both of
    # its (fixed) neighbours, starting from where it is; the result is accepted only if it fits better.
    for i in reversed(range(len(slices))):
        if notes[i] == "constrained refit":
            lo = out[i - 1] if i else None
            hi = out[i + 1] if i + 1 < len(slices) else None
            out[i] = refit_between(slices[i], out[i], lo, hi, wing_prior(backbone[i]))
    return out, notes


def check_slice(p, prev, sl=None):
    """(ok, min g, Lee slope, min calendar gap, min FD density) on the fine grid plus far points out to |k| = 1e4."""
    g = g_fun(p, KALL)
    lee = p[1] * (1 + abs(p[2]))
    gap = np.min(svi(p, KALL) - svi(prev, KALL)) if prev is not None else np.inf
    fd = np.min(fd_density(p, sl["F"], sl["D"], KGRID)[0]) if sl is not None else np.nan
    ok = g.min() >= 0 and lee <= 2 and gap >= 0 and np.min(svi(p, KALL)) > 0
    return ok, float(g.min()), float(lee), float(gap), float(fd)


def vol_stats(p, sl):
    v = np.sqrt(svi(p, sl["k"]) / sl["T"])
    e = (v - sl["iv"]) * 100
    inside = np.abs(v - sl["iv"]) <= np.maximum(sl["hv"], 0) + 1e-12
    liquid = sl["mid"] >= 0.50
    rl = np.sqrt(np.mean(e[liquid] ** 2)) if liquid.any() else np.nan
    return np.sqrt(np.mean(e * e)), np.abs(e).max(), inside.mean(), rl, int(liquid.sum())


def violations(params, slices):
    """Every violation on the grid: list of (expiry, kind, k-range, worst value)."""
    out = []
    for i, (p, sl) in enumerate(zip(params, slices)):
        g = g_fun(p, KALL)
        bad = g < 0
        if bad.any():
            out.append((sl["expiry"], "butterfly", KALL[bad].min(), KALL[bad].max(), g.min(), int(bad.sum())))
        if p[1] * (1 + abs(p[2])) > 2:
            out.append((sl["expiry"], "Lee wing", np.nan, np.nan, p[1] * (1 + abs(p[2])), 0))
        if i:
            d = svi(p, KALL) - svi(params[i - 1], KALL)
            bad = d < 0
            if bad.any():
                out.append((sl["expiry"], f"calendar vs {slices[i - 1]['expiry']}", KALL[bad].min(),
                            KALL[bad].max(), d.min(), int(bad.sum())))
    return out


def print_violations(v):
    if not v:
        print("  none")
    for e, kind, lo, hi, worst, n in v:
        rng = f"k in [{lo:+.3f}, {hi:+.3f}] ({n} grid pts)" if np.isfinite(lo) else ""
        print(f"  {e} {kind:24s} {rng:34s} worst {worst:+.3e}")


if __name__ == "__main__":
    t0 = time.time()
    s1 = json.load(open(folder / "stage1.json"))
    slices = load_slices(s1)
    n = sum(len(s["k"]) for s in slices)
    print(f"SPY chain {folder.name}: quotes at {s1['quote_time']}, spot {s1['spot']}, {len(slices)} expiries, "
          f"{n} out-of-the-money European-equivalent vols")
    print(f"Arbitrage grid: k = ln(K/F) in [{KGRID[0]}, {KGRID[-1]}] step 0.001 ({len(KGRID)} points per expiry), plus "
          f"{len(KFAR)} far points out to |k| = 1e4")
    print()

    free = [fit_free(sl) for sl in slices]
    print("A. Raw SVI per expiry, unconstrained. Vol errors in vol points; 'in' = share within each quote's")
    print("   half-spread in vol; 'liq' = RMS over quotes worth 50 cents or more.")
    print(f"  {'expiry':10s} {'days':>7s} {'n':>4s} {'rms':>6s} {'max':>6s} {'in':>5s} {'liq':>6s} {'nliq':>4s}"
          f" {'min g':>10s} {'Lee':>6s} {'min cal gap':>12s}")
    for i, (p, sl) in enumerate(zip(free, slices)):
        rms, mx, ins, rl, nl = vol_stats(p, sl)
        ok, gmin, lee, gap, _ = check_slice(p, free[i - 1] if i else None)
        print(f"  {sl['expiry']} {sl['T'] * 365:7.2f} {len(sl['k']):4d} {rms:6.2f} {mx:6.2f} {ins:5.0%} {rl:6.2f} {nl:4d}"
              f" {gmin:+10.2e} {lee:6.3f} {gap:+12.2e}" + ("" if ok else "  <- arbitrage"))
    fl = [vol_stats(p, sl) for p, sl in zip(free, slices)]
    print(f"  all expiries: RMS {np.sqrt(sum(f[0] ** 2 * len(sl['k']) for f, sl in zip(fl, slices)) / n):.2f} vol pts; "
          f"quotes >= 50c: {np.sqrt(sum(f[3] ** 2 * f[4] for f in fl if f[4]) / sum(f[4] for f in fl)):.2f}")
    print("\nViolations in the unconstrained surface:")
    vf = violations(free, slices)
    print_violations(vf)
    print()

    t1 = time.time()
    backbone, ss = fit_ssvi(slices)
    print(f"B. SSVI backbone, one surface for all {len(slices)} expiries: rho {ss['rho']:+.3f}, eta {ss['eta']:.3f}, "
          f"gamma {ss['gamma']:.3f}; theta increasing")
    tb = sum(vol_stats(p, sl)[0] ** 2 * len(sl["k"]) for p, sl in zip(backbone, slices))
    print(f"   RMS {np.sqrt(tb / n):.2f} vol pts over all quotes; violations on the grid: "
          f"{len(violations(backbone, slices))} ({time.time() - t1:.0f} s)")
    print()
    fixed, notes = repair(slices, free, backbone)

    print("C. After the repair (SVI per expiry, refitted between its calendar neighbours). 'fd dens' = smallest density from")
    print("   finite second differences of out-of-the-money Black prices on the grid (independent of g); d rms = change in")
    print("   RMS vol error caused by the repair.")
    print(f"  {'expiry':10s} {'n':>4s} {'rms':>6s} {'d rms':>6s} {'max':>6s} {'in':>5s} {'liq':>6s}"
          f" {'min g':>10s} {'Lee':>6s} {'min cal gap':>12s} {'fd dens':>10s}  a, b, rho, m, s")
    rows, tot, tot_free, totl, nl_all = [], 0.0, 0.0, 0.0, 0
    for i, (p, sl) in enumerate(zip(fixed, slices)):
        rms, mx, ins, rl, nl = vol_stats(p, sl)
        rms0 = vol_stats(free[i], sl)[0]
        ok, gmin, lee, gap, fd = check_slice(p, fixed[i - 1] if i else None, sl)
        tot += rms * rms * len(sl["k"])
        tot_free += rms0 * rms0 * len(sl["k"])
        if nl:
            totl += rl * rl * nl
            nl_all += nl
        changed = notes[i]
        print(f"  {sl['expiry']} {len(sl['k']):4d} {rms:6.2f} {rms - rms0:+6.2f} {mx:6.2f} {ins:5.0%} {rl:6.2f}"
              f" {gmin:+10.2e} {lee:6.3f} {gap:+12.2e} {fd:+10.2e}  "
              + ", ".join(f"{x:+.5f}" for x in p) + f"  ({changed})")
        rows.append(dict(expiry=sl["expiry"], T=sl["T"], F=sl["F"], D=sl["D"], r=sl["r"],
                         svi=[float(x) for x in p], svi_free=[float(x) for x in free[i]],
                         n=len(sl["k"]), rms=rms, max=mx, inside=ins, rms_liquid=rl, n_liquid=nl,
                         min_g=gmin, lee=lee, min_cal_gap=gap if i else None, min_fd_density=fd,
                         repair=changed))
    print(f"\n  all expiries: RMS {np.sqrt(tot / n):.2f} vol pts (unconstrained {np.sqrt(tot_free / n):.2f}); "
          f"quotes >= 50c: {np.sqrt(totl / nl_all):.2f} over {nl_all}")
    print("\nViolations in the repaired surface:")
    vx = violations(fixed, slices)
    print_violations(vx)
    print(f"  every slice: min g {min(r['min_g'] for r in rows):+.2e}, max Lee slope "
          f"{max(r['lee'] for r in rows):.3f} (bound 2), min calendar gap "
          f"{min(r['min_cal_gap'] for r in rows[1:]):+.2e}, min finite-difference density "
          f"{min(r['min_fd_density'] for r in rows):+.2e}")

    json.dump({"source": s1["source"], "quote_time": s1["quote_time"], "spot": s1["spot"],
               "grid": [float(KGRID[0]), float(KGRID[-1]), 0.001],
               "violations_free": [dict(expiry=e, kind=kd, k_lo=None if not np.isfinite(lo) else float(lo),
                                        k_hi=None if not np.isfinite(hi) else float(hi), worst=float(w), n=nn)
                                   for e, kd, lo, hi, w, nn in vf],
               "slices": rows}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT.relative_to(HERE.parent)}; {time.time() - t0:.0f} s")
