"""Part 4 Stage 1: an arbitrage-free implied-volatility surface for the Oct 23 SPY chain.

python reference/svi_surface.py [data/spy_full_<date>]
  -> report on stdout, <folder>/svi_surface.json, docs/svi_surface.png

Input is stage1.json (chain_stage1.py): the 32 expiries with their parity forwards,
and every filtered contract with its European-equivalent vol (mid minus the
early-exercise premium, inverted on the parity forward). Only the out-of-the-money
side is fitted (put below the forward, call above): its price is all time value.

One raw SVI slice per expiry (Gatheral 2004), in total implied variance w = sigma^2 T
against log-moneyness k = ln(K / F_T):

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))

Two kinds of arbitrage, both checked on a k grid from -3 to +1.5 (strikes from $38
to $3,400 on the shortest forward) and, for the wings, in the limit k -> +-inf:

  butterfly  The risk-neutral density must be >= 0 at every strike. For a slice,
             density(K) = g(k) / (K sqrt(2 pi w)) exp(-d2^2 / 2), so the sign is the
             sign of Durrleman's g(k) = (1 - k w'/(2w))^2 - w'^2/4 (1/w + 1/4) + w''/2.
             Checked twice: g on the grid, and the repo's edge-detect kernel [1, -2, 1]
             run across Black call prices on a $0.05 strike grid (Breeden-Litzenberger).
             Lee's moment bound needs the wing slopes b (1 +- rho) <= 2.
  calendar   At a fixed k, total variance must not fall as the expiry gets longer
             (with deterministic rates and dividends, C(K F_T / F_T, T) / F_T is a call
             on a martingale). Checked as w_{i+1}(k) - w_i(k) >= 0 on the grid and
             b (1 +- rho) non-decreasing for the wings beyond it.

The fit: (1) every expiry on its own, no constraints; (2) the arbitrage checks, with
every violation reported; (3) the repair: refit with the checks as constraints, first
slice by slice from short to long (each slice arbitrage-free and on or above the one
before), then all 32 slices at once from there, so a long expiry can pull a short one
down as well as the other way round; (4) the checks again, on the slices and on the
surface between them (w linear in T at fixed k, which keeps calendar monotonicity by
construction; butterfly is checked at 19 times inside every interval).

Errors are in vol points against the mid; a quote's weight is 1 / its vol half-spread
(floor 0.25 vol points), so a $0.02-bid wing quote counts for little and a liquid
at-the-money quote counts for a lot.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize, brentq
from scipy.special import ndtr

HERE = Path(__file__).parent
folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
OUT = folder / "svi_surface.json"
PLOT = HERE.parent / "docs" / "svi_surface.png"

K_LO, K_HI = -3.0, 1.5                      # log-moneyness range of every check
KG = np.linspace(K_LO, K_HI, 4501)          # check grid, step 0.001
KC = np.union1d(np.linspace(K_LO, K_HI, 451), np.linspace(-0.3, 0.3, 601))   # constraint grid: step 0.01, 0.001 near the money
FLOOR = 0.0025                              # weight floor: 0.25 vol points
DK = 0.05                                   # edge-detect strike step, dollars


# ---------- SVI ----------

def svi(p, k):
    a, b, rho, m, s = p
    d = k - m
    return a + b * (rho * d + np.sqrt(d * d + s * s))


def svi_d1(p, k):
    a, b, rho, m, s = p
    d = k - m
    return b * (rho + d / np.sqrt(d * d + s * s))


def svi_d2(p, k):
    a, b, rho, m, s = p
    d = k - m
    return b * s * s / (d * d + s * s) ** 1.5


def durrleman(w, w1, w2, k):
    """Durrleman's g: the density is g(k) / sqrt(2 pi w) e^{-d2^2/2} / K, so density >= 0 iff g >= 0."""
    return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2


def g_svi(p, k):
    return durrleman(svi(p, k), svi_d1(p, k), svi_d2(p, k), k)


def wings(p):
    a, b, rho, m, s = p
    return b * (1 - rho), b * (1 + rho)      # slope of w as k -> -inf (as |k|) and as k -> +inf


def black_otm(F, K, w):
    """Undiscounted Black price of the out-of-the-money option (put below F, call above)."""
    v = np.sqrt(w)
    d1 = (np.log(F / K) + 0.5 * w) / v
    d2 = d1 - v
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return np.where(K >= F, call, put)


def implied_total_vol(price, F, K):
    """Total vol v = sigma sqrt T whose undiscounted OTM Black price is `price`; NaN outside the bounds."""
    hi_bound = F if K >= F else K
    if not (0 < price < hi_bound):
        return np.nan
    f = lambda v: black_otm(F, np.array(K), v * v) - price
    return brentq(f, 1e-6, 20.0, xtol=1e-14, rtol=1e-14, maxiter=300)


# ---------- data ----------

def quotes(e, cs):
    """Out-of-the-money European-equivalent quotes of one expiry, with bid and ask vols."""
    F, T, D = e["F"], e["T"], e["D"]
    otm = [c for c in cs if c["iv_true"] is not None and (c["cp"] == "call") == (c["K"] >= F)]
    K = np.array([c["K"] for c in otm])
    iv = np.array([c["iv_true"] for c in otm])
    eep = np.array([c["eep"] or 0.0 for c in otm])
    bid = np.array([c["bid"] for c in otm]) - eep
    ask = np.array([c["ask"] for c in otm]) - eep
    vb = np.array([implied_total_vol(p / D, F, k) for p, k in zip(bid, K)]) / np.sqrt(T)
    va = np.array([implied_total_vol(p / D, F, k) for p, k in zip(ask, K)]) / np.sqrt(T)
    vb = np.where(np.isfinite(vb), vb, 0.0)                 # bid at or below zero time value: vol 0
    va = np.where(np.isfinite(va), va, 3.0)
    hs = 0.5 * (va - vb)
    return dict(K=K, k=np.log(K / F), iv=iv, vb=vb, va=va, hs=hs, wt=1 / np.maximum(hs, FLOOR),
                mid=0.5 * (bid + ask), T=T, F=F, D=D)


def resid(p, q):
    w = svi(p, q["k"])
    return (np.sqrt(np.maximum(w, 1e-12) / q["T"]) - q["iv"]) * q["wt"]


def fit_free(q):
    """Unconstrained raw SVI on one expiry: multi-start least squares, bounds only."""
    T, k, iv = q["T"], q["k"], q["iv"]
    w_atm = float(np.interp(0.0, k, iv)) ** 2 * T
    lo = [-1.0, 1e-6, -0.999, -1.0, 1e-5]
    hi = [1.0, 10.0, 0.999, 1.0, 2.0]
    best = None
    for b0 in (0.05, 0.2, 0.8):
        for rho0 in (-0.9, -0.6, -0.2):
            for s0 in (0.02, 0.1):
                sc = np.sqrt(w_atm)               # scale the starting point to the slice's width
                x0 = [w_atm * 0.5, b0 * sc, rho0, 0.0, s0 * sc / 0.1 * 0.1]
                x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
                try:
                    r = least_squares(resid, x0, args=(q,), bounds=(lo, hi), xtol=1e-14, ftol=1e-14, max_nfev=4000)
                except ValueError:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    return best.x


def objective(p, q):
    r = resid(p, q)
    return float(r @ r)


# ---------- checks ----------

def edge_density(F, K, w):
    """[1, -2, 1] / dK^2 across Black prices on an even strike grid K: the density per dollar at K[1:-1].
    Prices are the out-of-the-money side (precise far from F), with the kink of (F - K)^+ added back."""
    dK = K[1] - K[0]
    C = black_otm(F, K, w)
    dens = (C[2:] - 2 * C[1:-1] + C[:-2]) / dK**2
    j = np.searchsorted(K, F)
    for i in (j - 2, j - 1):                                   # only the stencils that straddle F see the kink
        if 0 <= i < len(dens):
            dens[i] += (max(F - K[i + 2], 0) - 2 * max(F - K[i + 1], 0) + max(F - K[i], 0)) / dK**2
    return dens, C


def slice_checks(p, q):
    """Butterfly on one slice: Durrleman g on the grid, the edge-detect density on $0.05 strikes, Lee's bound."""
    g = g_svi(p, KG)
    neg = g < 0
    F, T, D = q["F"], q["T"], q["D"]
    K = np.arange(np.ceil(F * np.exp(K_LO) / DK) * DK, F * np.exp(K_HI), DK)
    dens, C = edge_density(F, K, svi(p, np.log(K / F)))
    noise = 8 * np.finfo(float).eps * np.maximum(C[1:-1], 1e-300) / DK**2 + 1e-300
    edge_neg = dens < -noise
    Kc = K[1:-1]
    lw, rw = wings(p)
    quoted = (KG >= q["k"].min()) & (KG <= q["k"].max())
    return dict(g_min=float(g.min()), g_k=float(KG[np.argmin(g)]), n_neg=int(neg.sum()), n_in=int((neg & quoted).sum()),
                neg_from=float(KG[neg].min()) if neg.any() else None, neg_to=float(KG[neg].max()) if neg.any() else None,
                edge_min=float(dens.min()), edge_neg=int(edge_neg.sum()), n_edge=len(dens),
                edge_K=(float(Kc[edge_neg].min()), float(Kc[edge_neg].max())) if edge_neg.any() else None,
                mass=float(dens.sum() * DK), lee=max(lw, rw), w_min=float(svi(p, KG).min()))


def calendar_checks(p0, p1, q0=None, q1=None):
    """w_{i+1} - w_i on the grid, and the wing slopes beyond it. n_in: grid points below 0 inside both quoted ranges."""
    dw = svi(p1, KG) - svi(p0, KG)
    neg = dw < 0
    both = np.ones_like(neg) if q0 is None else (
        (KG >= max(q0["k"].min(), q1["k"].min())) & (KG <= min(q0["k"].max(), q1["k"].max())))
    l0, r0 = wings(p0)
    l1, r1 = wings(p1)
    return dict(dw_min=float(dw.min()), n_neg=int(neg.sum()), n_in=int((neg & both).sum()),
                k_from=float(KG[neg].min()) if neg.any() else None, k_to=float(KG[neg].max()) if neg.any() else None,
                wing_ok=bool(l1 >= l0 - 1e-12 and r1 >= r0 - 1e-12))


def between_checks(P, Ts, steps=20):
    """Butterfly on the surface between expiries: w(k, T) linear in T at fixed k (from w = 0 at T = 0)."""
    worst = (np.inf, None, None)
    n_neg = 0
    Ws = [np.zeros_like(KG)] + [svi(p, KG) for p in P]
    W1 = [np.zeros_like(KG)] + [svi_d1(p, KG) for p in P]
    W2 = [np.zeros_like(KG)] + [svi_d2(p, KG) for p in P]
    T = [0.0] + list(Ts)
    for i in range(len(P)):
        for a in np.linspace(0, 1, steps + 1)[1:-1]:
            w = (1 - a) * Ws[i] + a * Ws[i + 1]
            g = durrleman(w, (1 - a) * W1[i] + a * W1[i + 1], (1 - a) * W2[i] + a * W2[i + 1], KG)
            n_neg += int((g < 0).sum())
            j = np.argmin(g)
            if g[j] < worst[0]:
                worst = (float(g[j]), (1 - a) * T[i] + a * T[i + 1], float(KG[j]))
    return n_neg, worst


# ---------- the repair ----------
#
# Constraints enter the least squares as hinge penalties lam * min(c, 0), each with a small margin so that
# the tiny overshoot a penalty leaves still lands on the right side of the true constraint. lam rises
# until the strict check (margin 0) passes on the constraint grid; the full check grid and the wings are
# checked again afterwards.

G_M, W_M, S_M = 1e-4, 2e-7, 1e-5            # margins: Durrleman g, calendar (in w), wing slopes
LO = np.array([-1.0, 1e-6, -0.999, -1.0, 1e-5])
HI = np.array([1.0, 10.0, 0.999, 1.0, 2.0])


def slice_cons(p):
    return np.concatenate([g_svi(p, KC) - G_M, 2 - S_M - np.array(wings(p)), svi(p, KC) * 1e3])


def pair_cons(p0, p1):
    sc = 1.0 / max(svi(p1, np.array([0.0]))[0], 1e-6)
    return np.concatenate([(svi(p1, KC) - svi(p0, KC) - W_M) * sc,
                           (np.array(wings(p1)) - np.array(wings(p0)) - S_M) * 10])


def strict_ok(p, below=None, above=None):
    c = [g_svi(p, KC), 2 - np.array(wings(p)), svi(p, KC)]
    if below is not None:
        c += [svi(p, KC) - svi(below, KC), np.array(wings(p)) - np.array(wings(below))]
    if above is not None:
        c += [svi(above, KC) - svi(p, KC), np.array(wings(above)) - np.array(wings(p))]
    return min(x.min() for x in c) >= 0


def refit(starts, q, below=None):
    """One slice, constrained to be arbitrage-free and on or above `below`. Best feasible over the starts."""
    def f(p, lam):
        c = [slice_cons(p)] + ([pair_cons(below, p)] if below is not None else [])
        return np.concatenate([resid(p, q), lam * np.minimum(np.concatenate(c), 0)])
    best = None
    for x0 in starts:
        x = np.clip(x0, LO + 1e-9, HI - 1e-9)
        for lam in (1e3, 1e4, 1e5):
            x = least_squares(f, x, args=(lam,), bounds=(LO, HI), xtol=1e-13, ftol=1e-13, max_nfev=3000).x
            if strict_ok(x, below):
                break
        if strict_ok(x, below) and (best is None or objective(x, q) < objective(best, q)):
            best = x
    return best


def refit_joint(P, Q):
    """All slices at once: the weighted vol residuals of every quote plus the penalties of every slice and
    every neighbouring pair. Each row depends on one or two slices, so the Jacobian is sparse."""
    from scipy.sparse import lil_matrix
    n = len(P)

    def F(x, lam):
        P = x.reshape(n, 5)
        out = [resid(P[i], Q[i]) for i in range(n)]
        out += [lam * np.minimum(slice_cons(P[i]), 0) for i in range(n)]
        out += [lam * np.minimum(pair_cons(P[i], P[i + 1]), 0) for i in range(n - 1)]
        return np.concatenate(out)
    ns, npc = len(slice_cons(P[0])), len(pair_cons(P[0], P[1]))
    J = lil_matrix((sum(len(q["k"]) for q in Q) + n * ns + (n - 1) * npc, 5 * n), dtype=int)
    r = 0
    for i in range(n):
        J[r:r + len(Q[i]["k"]), 5 * i:5 * i + 5] = 1
        r += len(Q[i]["k"])
    for i in range(n):
        J[r:r + ns, 5 * i:5 * i + 5] = 1
        r += ns
    for i in range(n - 1):
        J[r:r + npc, 5 * i:5 * i + 10] = 1
        r += npc
    x = np.concatenate(P)
    lo, hi = np.tile(LO, n), np.tile(HI, n)
    feasible = lambda P: all(strict_ok(P[i], P[i - 1] if i else None) for i in range(n))
    for lam in (1e2, 1e3, 1e4, 1e5, 1e6):
        x = least_squares(F, x, args=(lam,), bounds=(lo, hi), jac_sparsity=J, xtol=1e-12, ftol=1e-12,
                          max_nfev=500, x_scale="jac").x
        if lam >= 1e4 and feasible(x.reshape(n, 5)):
            return list(x.reshape(n, 5)), lam
    return None, lam


# ---------- report ----------

def fit_stats(p, q):
    v = np.sqrt(np.maximum(svi(p, q["k"]), 0) / q["T"])
    e = (v - q["iv"]) * 100
    inside = (v >= q["vb"] - 1e-12) & (v <= q["va"] + 1e-12)
    core = q["mid"] >= 0.50
    return dict(rms=float(np.sqrt(np.mean(e * e))), max=float(np.abs(e).max()),
                wrms=float(np.sqrt(np.sum((e * q["wt"]) ** 2) / np.sum(q["wt"] ** 2))),
                inside=float(inside.mean()), core_rms=float(np.sqrt(np.mean(e[core] ** 2))) if core.any() else np.nan,
                n_core=int(core.sum()))


if __name__ == "__main__":
    t0 = time.time()
    s1 = json.load(open(folder / "stage1.json"))
    exps = s1["expiries"]
    Q = [quotes(e, [c for c in s1["contracts"] if c["expiry"] == e["expiry"]]) for e in exps]
    Ts = [e["T"] for e in exps]
    n = len(exps)
    print(f"SPY chain {folder.name}: quotes at {s1['quote_time']}, spot {s1['spot']}, {n} expiries, "
          f"{sum(len(q['k']) for q in Q)} out-of-the-money quotes with a European-equivalent vol")
    print(f"  weights: 1 / vol half-spread, floor {FLOOR * 100:.2f} vol points. Median half-spread "
          f"{np.median(np.concatenate([q['hs'] for q in Q])) * 100:.2f} vol points")
    print(f"  checks on k = ln K/F from {K_LO} to {K_HI} (step {KG[1] - KG[0]:.3f}); edge-detect on ${DK} strikes")
    print()

    # (1) every expiry on its own
    t1 = time.time()
    P0 = [fit_free(q) for q in Q]
    print(f"(1) Raw SVI per expiry, no constraints ({time.time() - t1:.0f} s)")
    print(f"  {'expiry':10s} {'days':>6s} {'n':>4s} {'a':>9s} {'b':>8s} {'rho':>7s} {'m':>8s} {'s':>8s}"
          f" {'rms':>6s} {'wrms':>6s} {'max':>6s} {'in b/a':>6s} {'>50c':>6s}")
    for e, q, p in zip(exps, Q, P0):
        st = fit_stats(p, q)
        print(f"  {e['expiry']} {e['T'] * 365:6.2f} {len(q['k']):4d} {p[0]:9.6f} {p[1]:8.5f} {p[2]:+7.3f} {p[3]:+8.4f} {p[4]:8.5f}"
              f" {st['rms']:6.2f} {st['wrms']:6.2f} {st['max']:6.2f} {st['inside']:6.0%} {st['core_rms']:6.2f}")
    print("  rms / max: vol points against the mid, every quote. wrms: weighted by 1 / half-spread (what the fit")
    print("  minimises). in b/a: share of quotes whose fitted vol is inside the bid-ask vols. >50c: rms over quotes")
    print("  worth 50 cents or more.")
    print()

    def report_arbitrage(P, label):
        print(f"{label}")
        print("  Butterfly, slice by slice: Durrleman g (min, where, grid points below 0, how many of them inside")
        print("  the quoted strikes, their k range); the edge-detect density [1,-2,1]/dK^2 on Black prices (min per")
        print("  dollar, strikes below -roundoff, their strike range; total mass); Lee's wing slope max b(1 +- rho) <= 2")
        bad_b = 0
        for e, q, p in zip(exps, Q, P):
            c = slice_checks(p, q)
            flag = c["n_neg"] or c["edge_neg"] or c["lee"] > 2 or c["w_min"] <= 0
            bad_b += bool(flag)
            rng = f"k {c['neg_from']:+.3f}..{c['neg_to']:+.3f}" if c["n_neg"] else ""
            erng = f"K ${c['edge_K'][0]:.2f}..${c['edge_K'][1]:.2f}" if c["edge_K"] else ""
            print(f"  {e['expiry']}  g min {c['g_min']:+9.2e} at k {c['g_k']:+.3f}  g<0 {c['n_neg']:5d} ({c['n_in']:3d} quoted) {rng:18s}"
                  f" edge min {c['edge_min']:+9.2e} <0 {c['edge_neg']:5d}/{c['n_edge']} {erng:22s} mass {c['mass']:.6f}"
                  f"  Lee {c['lee']:.3f}" + ("   VIOLATION" if flag else ""))
        print("  Calendar, each pair of neighbouring expiries: min of w_{i+1} - w_i on the grid, grid points below 0")
        print("  (how many inside both expiries' quoted strikes) and their k range, and whether the wing slopes")
        print("  keep the order beyond the grid")
        bad_c = 0
        for i in range(n - 1):
            c = calendar_checks(P[i], P[i + 1], Q[i], Q[i + 1])
            flag = c["n_neg"] or not c["wing_ok"]
            bad_c += bool(flag)
            rng = f"k {c['k_from']:+.3f}..{c['k_to']:+.3f}" if c["n_neg"] else ""
            print(f"  {exps[i]['expiry']} -> {exps[i + 1]['expiry']}  min dw {c['dw_min']:+10.3e}  <0 {c['n_neg']:5d} ({c['n_in']:4d} quoted) {rng:18s}"
                  f" wings {'ok' if c['wing_ok'] else 'CROSS'}" + ("   VIOLATION" if flag else ""))
        nb, worst = between_checks(P, Ts)
        print(f"  Between expiries (w linear in T at fixed k, 19 times per interval): {nb} grid points with g < 0; "
              f"min g {worst[0]:+.2e} at T = {worst[1] * 365:.2f} days, k = {worst[2]:+.3f}")
        print(f"  => {bad_b} of {n} slices with butterfly arbitrage, {bad_c} of {n - 1} neighbouring pairs with calendar arbitrage")
        print()
        return bad_b, bad_c, nb

    b0, c0, nb0 = report_arbitrage(P0, "(2) Arbitrage in the unconstrained fit")

    # (3) the repair
    t2 = time.time()
    P1 = [None] * n
    for i in range(n):                      # pass 1: short to long, each slice on or above the one before
        below = P1[i - 1] if i else None
        starts = [P0[i]]
        if below is not None:               # the slice before, lifted in parallel, is always feasible
            lifted = below.copy()
            lifted[0] += (Ts[i] - Ts[i - 1]) * 0.02
            starts.append(lifted)
        P1[i] = refit(starts, Q[i], below)
        assert P1[i] is not None, f"pass 1 found no arbitrage-free slice at {exps[i]['expiry']}"
    t3 = time.time()
    PJ, lam = refit_joint(P1, Q)
    total = lambda P: sum(objective(p, q) for p, q in zip(P, Q))
    P = PJ if PJ is not None and total(PJ) < total(P1) else P1
    print(f"(3) Repair: refit with the checks as constraints (penalties with small margins, then a strict check)")
    print(f"  pass 1 ({t3 - t2:.0f} s): short to long, each slice arbitrage-free and on or above the one before")
    print(f"  pass 2 ({time.time() - t3:.0f} s): all 160 parameters at once from pass 1, every constraint between every pair, "
          f"penalty up to {lam:.0e}" + ("" if PJ is not None else " (did not reach a strictly feasible point: pass 1 kept)"))
    print(f"  weighted misfit (sum of squared weighted residuals): unconstrained {total(P0):.1f}, pass 1 {total(P1):.1f}, "
          f"pass 2 {total(PJ):.1f}" if PJ is not None else "")
    print(f"  {'expiry':10s} {'days':>6s} {'a':>9s} {'b':>8s} {'rho':>7s} {'m':>8s} {'s':>8s}"
          f" {'rms':>6s} {'wrms':>6s} {'max':>6s} {'in b/a':>6s} {'>50c':>6s}  change vs (1): {'rms':>6s} {'in b/a':>6s} {'max |dvol| on quotes':>21s}")
    tot = dict(n=0, se=0.0, se0=0.0, inside=0.0, inside0=0.0, core=0, cse=0.0, cse0=0.0)
    for e, q, p, p0 in zip(exps, Q, P, P0):
        st, s0 = fit_stats(p, q), fit_stats(p0, q)
        dv = (np.sqrt(svi(p, q["k"]) / q["T"]) - np.sqrt(np.maximum(svi(p0, q["k"]), 0) / q["T"])) * 100
        m = len(q["k"])
        tot["n"] += m; tot["se"] += st["rms"] ** 2 * m; tot["se0"] += s0["rms"] ** 2 * m
        tot["inside"] += st["inside"] * m; tot["inside0"] += s0["inside"] * m
        if st["n_core"]:
            tot["core"] += st["n_core"]; tot["cse"] += st["core_rms"] ** 2 * st["n_core"]; tot["cse0"] += s0["core_rms"] ** 2 * st["n_core"]
        print(f"  {e['expiry']} {e['T'] * 365:6.2f} {p[0]:9.6f} {p[1]:8.5f} {p[2]:+7.3f} {p[3]:+8.4f} {p[4]:8.5f}"
              f" {st['rms']:6.2f} {st['wrms']:6.2f} {st['max']:6.2f} {st['inside']:6.0%} {st['core_rms']:6.2f}"
              f"                {st['rms'] - s0['rms']:+6.2f} {st['inside'] - s0['inside']:+6.0%} {np.abs(dv).max():21.2f}")
    print(f"  all {tot['n']} quotes: rms {np.sqrt(tot['se'] / tot['n']):.2f} vol points (unconstrained "
          f"{np.sqrt(tot['se0'] / tot['n']):.2f}); inside the bid-ask {tot['inside'] / tot['n']:.0%} (unconstrained "
          f"{tot['inside0'] / tot['n']:.0%}); quotes worth 50c+ ({tot['core']}): rms {np.sqrt(tot['cse'] / tot['core']):.2f} "
          f"(unconstrained {np.sqrt(tot['cse0'] / tot['core']):.2f})")
    print()

    b1, c1, nb1 = report_arbitrage(P, "(4) Arbitrage in the repaired surface")

    json.dump({
        "source": s1["source"], "quote_time": s1["quote_time"], "spot": s1["spot"],
        "model": "raw SVI per expiry, w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2)), k = ln(K / F_T), w = sigma^2 T; "
                 "between expiries w is linear in T at fixed k, and w = (T / T_1) w_1 before the first",
        "k_range": [K_LO, K_HI],
        "expiries": [{**{k: e[k] for k in ("expiry", "T", "r", "D", "F")},
                      "svi": dict(zip("a b rho m s".split(), map(float, p))),
                      "svi_free": dict(zip("a b rho m s".split(), map(float, p0))),
                      **{f"fit_{k}": v for k, v in fit_stats(p, q).items()}}
                     for e, q, p, p0 in zip(exps, Q, P, P0)],
    }, open(OUT, "w"), indent=1)
    print(f"wrote {OUT.relative_to(HERE.parent)}")

    # picture: smiles, total variance, densities
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    show = [0, 4, 9, 11, 15, 21, 25, 31]
    cmap = plt.get_cmap("viridis")
    for j, i in enumerate(show):
        q, c = Q[i], cmap(j / (len(show) - 1))
        sd = np.sqrt(svi(P[i], 0.0))
        z = q["k"] / sd
        ax[0, 0].errorbar(z, q["iv"] * 100, yerr=[(q["iv"] - q["vb"]) * 100, (q["va"] - q["iv"]) * 100], fmt="o", ms=2.5,
                          color=c, alpha=0.5, elinewidth=0.6, capsize=0)
        kk = np.linspace(q["k"].min() - 0.02, q["k"].max() + 0.02, 400)
        ax[0, 0].plot(kk / sd, np.sqrt(svi(P[i], kk) / q["T"]) * 100, color=c, lw=1.4, label=f"{exps[i]['expiry']} ({q['T'] * 365:.0f}d)")
        ax[0, 0].plot(kk / sd, np.sqrt(np.maximum(svi(P0[i], kk), 0) / q["T"]) * 100, color=c, lw=0.8, ls=":")
    ax[0, 0].set(xlim=(-12, 5), ylim=(5, 90), xlabel="standard deviations from the forward, k / sqrt(w_atm)", ylabel="implied vol, %",
                 title="Smiles: quotes with bid-ask bars, repaired SVI (solid), unconstrained (dotted)")
    ax[0, 0].legend(fontsize=7, ncol=2)
    kk = np.linspace(-0.6, 0.3, 300)
    for i in range(n):
        ax[0, 1].plot(kk, svi(P[i], kk), color=cmap(i / (n - 1)), lw=0.9)
    ax[0, 1].set(xlabel="k = ln K/F", ylabel="total variance w = sigma^2 T", yscale="log", ylim=(5e-5, 0.3),
                 title="Total variance, all 32 expiries: the curves never cross")
    # the worst calendar crossing before the repair
    worst = min(range(n - 1), key=lambda i: calendar_checks(P0[i], P0[i + 1])["dw_min"])
    kk = np.linspace(K_LO, K_HI, 1000)
    for i, ls in ((worst, "-"), (worst + 1, "--")):
        ax[1, 0].plot(kk, svi(P0[i], kk), color="tab:red", ls=ls, lw=1, label=f"unconstrained {exps[i]['expiry']}")
        ax[1, 0].plot(kk, svi(P[i], kk), color="tab:blue", ls=ls, lw=1, label=f"repaired {exps[i]['expiry']}")
    ax[1, 0].set(xlabel="k = ln K/F", ylabel="w", title="The worst calendar crossing before the repair", yscale="log")
    ax[1, 0].legend(fontsize=7)
    for j, i in enumerate(show):
        q = Q[i]
        K = np.arange(np.ceil(q["F"] * 0.5), q["F"] * 1.3, DK)
        ax[1, 1].plot(K[1:-1], edge_density(q["F"], K, svi(P[i], np.log(K / q["F"])))[0], color=cmap(j / (len(show) - 1)), lw=1,
                      label=f"{exps[i]['expiry']}")
    ax[1, 1].set(yscale="log", ylim=(1e-7, 0.2), xlabel="strike, $", ylabel="density per dollar",
                 title="Edge-detect density [1,-2,1] of the repaired surface: positive everywhere")
    ax[1, 1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=110)
    print(f"wrote {PLOT.relative_to(HERE.parent)}")
    print(f"total time {time.time() - t0:.0f} s")
