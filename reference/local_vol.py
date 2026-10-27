"""Part 4 Stage 2: local volatility from the Stage 1 surface (Dupire, Gatheral's total-variance form).

python reference/local_vol.py [data/spy_full_<date>]
  -> report on stdout, <folder>/local_vol.json, docs/local_vol.png

The surface is Stage 1's (svi_surface.json): a raw SVI slice per expiry in total variance
w(k, T) = sigma_imp^2 T against k = ln(K / F_T), w linear in T at fixed k between expiries
(and w = (T / T_1) w_1 before the first). With deterministic rates and dividends, a call on S
struck at K = F_T e^k is F_T D_T c(k, T) with c(k, T) = E[(e^X_T - e^k)^+] and X = ln(S / F) a
driftless-in-price martingale log, so Dupire's equation holds in (k, T) with no drift term:

    dc/dT = 1/2 sigma_loc^2 (c_kk - c_k)

and in total variance (Gatheral, The Volatility Surface, eq. 1.10):

    sigma_loc^2(k, T) = (dw/dT) / g(k, T),
    g = 1 - k w_k / w + 1/4 (-1/4 - 1/w + k^2 / w^2) w_k^2 + 1/2 w_kk.

The denominator g is Durrleman's butterfly function from Stage 1: the density divided by a
positive factor. So Dupire divides by (a scaled) second strike derivative of the price, and
where the density is thin, g is small and every error in w_kk is magnified.

Where it is unreliable (flags, per (k, T) cell):
  E  extrapolated: k outside the strikes quoted (bid > 0) on the expiry that ends T's interval.
     The local vol inside an interval is what carries the density into that expiry, so where the
     expiry has prices, they pin it; outside them the SVI wing is a straight line chosen by the
     fit, not by any price. (Flagging by the shorter expiry's quotes too would throw away the
     wings of a long expiry whenever the one before it is thinly quoted: Stage 3's round trip
     then misses those wings by up to 6 vol points.)
  G  small denominator: g < 0.1. Half the grid cells below that move by more than 10% between
     two independent arbitrage-free fits of the same quotes (measured below).
  N  noisy in time: the quoted fit error on the two bracketing expiries, turned into an error in
     dw/dT over the gap between them, moves sigma_loc by more than 10%. Daily expiries a day
     apart do this near the money.

Regularization (what Stage 3 prices with): the raw Dupire value, held inside [2%, 300%] (and 300%
where g <= 0 makes it undefined). The flags mark where it is unreliable; they do not change it.
The cap is high because the quotes need it: inside the quoted strikes of the 3-day expiry the raw
local vol of the deep puts reaches about 450%, and a 150% cap already moves those quotes' implied
vols by 0.4 vol points in Stage 3's round trip (100%: 2 vol points). The first version of this
stage instead replaced E and G cells by the reliable local vol at the same time (linear in k
across a gap, flat beyond); that is kept as local_vol_filled() for comparison and rejected,
because paths reach a long expiry's quoted deep puts through earlier intervals where those strikes
were unquoted, and flattening there cost up to 0.45 vol points on quoted options (section 4).
After the last expiry the last interval's local vol is held flat in time.
"""
import sys
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).parent
G_MIN = 0.1            # flag G below this
G_REG = 0.1            # local_vol_filled() replaces below this
NOISE_MAX = 0.10       # flag N above this relative error in sigma_loc
LV_FLOOR, LV_CAP = 0.02, 3.0
E_FLAG, G_FLAG, N_FLAG = 1, 2, 4


def svi(p, k):
    a, b, rho, m, s = p
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + s * s))


def svi_d(p, k):
    a, b, rho, m, s = p
    x = k - m
    r = np.sqrt(x * x + s * s)
    return a + b * (rho * x + r), b * (rho + x / r), b * s * s / r**3


def durrleman(w, w1, w2, k):
    return 1 - k * w1 / w + 0.25 * (-0.25 - 1 / w + k * k / (w * w)) * w1 * w1 + 0.5 * w2


class Surface:
    """Stage 1's surface with Dupire local vol on top. P: (n, 5) raw SVI parameters, T: (n,) expiries."""

    def __init__(self, P, T, kq=None, err=None, names=None, F=None, D=None):
        self.P, self.T = np.asarray(P, float), np.asarray(T, float)
        self.kq = kq                      # (n, 2) quoted k range per expiry (bid > 0)
        self.err = err                    # (n,) fit error per expiry, vol (liquid-quote RMS)
        self.names, self.F, self.D = names, F, D

    @classmethod
    def load(cls, folder):
        s = json.load(open(Path(folder) / "svi_surface.json"))
        s1 = json.load(open(Path(folder) / "stage1.json"))
        E = s["expiries"]
        P = [[e["svi"][x] for x in ("a", "b", "rho", "m", "s")] for e in E]
        kq = []
        for e in E:
            ks = [np.log(c["K"] / e["F"]) for c in s1["contracts"] if c["expiry"] == e["expiry"]
                  and c["iv_true"] is not None and (c["cp"] == "call") == (c["K"] >= e["F"]) and c["bid"] > 0]
            kq.append((min(ks), max(ks)))
        err = np.array([e["fit_core_rms"] / 100 for e in E])
        return cls(P, [e["T"] for e in E], np.array(kq), err, [e["expiry"] for e in E],
                   np.array([e["F"] for e in E]), np.array([e["D"] for e in E]))

    def interval(self, t):
        """Index i with t in (T[i-1], T[i]] (i = 0 before the first expiry; i = n past the last)."""
        return int(np.searchsorted(self.T, t, side="left"))

    def w_all(self, k, t):
        """w, w_k, w_kk, w_T at time t on the array k (w linear in T between expiries)."""
        T, n = self.T, len(self.T)
        i = self.interval(t)
        if i == 0:
            w, w1, w2 = svi_d(self.P[0], k)
            a = t / T[0]
            return a * w, a * w1, a * w2, w / T[0]
        if i >= n:                         # past the last expiry: the last interval's slope, held
            w0, _, _ = svi_d(self.P[n - 2], k)
            w, w1, w2 = svi_d(self.P[n - 1], k)
            wT = (w - w0) / (T[n - 1] - T[n - 2])
            return w + wT * (t - T[n - 1]), w1, w2, wT
        lo, hi = svi_d(self.P[i - 1], k), svi_d(self.P[i], k)
        a = (t - T[i - 1]) / (T[i] - T[i - 1])
        return ((1 - a) * lo[0] + a * hi[0], (1 - a) * lo[1] + a * hi[1], (1 - a) * lo[2] + a * hi[2],
                (hi[0] - lo[0]) / (T[i] - T[i - 1]))

    def implied_vol(self, k, t):
        return np.sqrt(self.w_all(k, t)[0] / t)

    def dupire_raw(self, k, t):
        """(sigma_loc, g) from Gatheral's formula, no regularization; sigma_loc NaN where g <= 0."""
        w, w1, w2, wT = self.w_all(k, t)
        g = durrleman(w, w1, w2, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            lv = np.where(g > 0, np.sqrt(np.maximum(wT, 0) / g), np.nan)
        return lv, g

    def flags(self, k, t):
        """E | G | N bit flags on the array k at time t (see the module docstring)."""
        T, n = self.T, len(self.T)
        i = min(self.interval(t), n - 1 + (t > T[-1]))
        lo_i, hi_i = (max(i - 1, 0), min(i, n - 1)) if i < n else (n - 2, n - 1)
        kq = self.kq
        lo_k, hi_k = kq[hi_i]
        f = np.where((k < lo_k) | (k > hi_k), E_FLAG, 0)
        w, w1, w2, wT = self.w_all(k, t)
        g = durrleman(w, w1, w2, k)
        f = f | np.where(g < G_MIN, G_FLAG, 0)
        # dw = 2 sigma T dsigma on each bracketing expiry; relative error of sigma_loc = 1/2 rel. error of dw/dT
        def dw(j):
            wj = svi(self.P[j], k)
            return 2 * np.sqrt(wj * T[j]) * self.err[j]
        if i == 0:
            num, gap = dw(0), T[0]
        else:
            num, gap = np.hypot(dw(lo_i), dw(hi_i)), T[hi_i] - T[lo_i]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = 0.5 * num / (gap * np.maximum(wT, 1e-300))
        return f | np.where(rel > NOISE_MAX, N_FLAG, 0), g, rel

    def local_vol(self, k, t):
        """Local vol Stage 3 prices with on the array k at time t: raw Dupire held in [LV_FLOOR, LV_CAP]
        (LV_CAP where it is undefined). Returns it, the raw value and the flags."""
        raw, g = self.dupire_raw(k, t)
        f = self.flags(k, t)[0]
        return np.clip(np.nan_to_num(raw, nan=LV_CAP), LV_FLOOR, LV_CAP), raw, f

    def local_vol_filled(self, k, t):
        """The rejected alternative: E and G cells take the reliable local vol at the same time (linear in k
        across a gap, flat beyond the last reliable strike), then the same clip."""
        raw, g = self.dupire_raw(k, t)
        f = self.flags(k, t)[0]
        ok = ((f & E_FLAG) == 0) & (g >= G_REG) & np.isfinite(raw)
        if ok.any():
            lv = np.where(ok, raw, np.interp(k, k[ok], raw[ok]))
        else:                                   # nothing reliable at this time: the at-the-money implied vol
            lv = np.full_like(k, float(self.implied_vol(np.array([0.0]), t)[0]))
        return np.clip(lv, LV_FLOOR, LV_CAP), raw, f


def black_c(k, w):
    """Undiscounted call per unit forward: E[(e^X - e^k)^+] with Var X = w."""
    v = np.sqrt(w)
    d1 = (-k + 0.5 * w) / v
    return ndtr(d1) - np.exp(k) * ndtr(d1 - v)


def dupire_price_form(S, k, t, hk=None, ht=None):
    """Dupire from prices by finite differences: sigma^2 = 2 c_T / (c_kk - c_k), c from the surface.
    The k step is 1% of the at-the-money standard deviation, so it scales down with short expiries."""
    hk = hk or 0.01 * float(np.sqrt(S.w_all(np.array([0.0]), t)[0][0]))
    i = S.interval(t)
    lo = S.T[i - 1] if i > 0 else 0.0
    hi = S.T[i] if i < len(S.T) else t + 1.0
    ht = ht or 1e-3 * (hi - lo)
    c = lambda kk, tt: black_c(kk, S.w_all(kk, tt)[0])
    cT = (c(k, t + ht) - c(k, t - ht)) / (2 * ht)
    ck = (c(k + hk, t) - c(k - hk, t)) / (2 * hk)
    ckk = (c(k + hk, t) - 2 * c(k, t) + c(k - hk, t)) / hk**2
    return np.sqrt(2 * cT / (ckk - ck))


def cell_times(S, per=4):
    """Cell centres in time: `per` per interval, plus the stretch before the first expiry."""
    edges = np.concatenate([[0.0], S.T])
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out += list(a + (np.arange(per) + 0.5) * (b - a) / per)
    return np.array(out)


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
    S = Surface.load(folder)
    n = len(S.T)
    print(f"Local volatility from Stage 1's SVI surface ({folder.name}, {n} expiries, "
          f"{S.T[0] * 365:.2f} to {S.T[-1] * 365:.0f} days). Python float64.")
    print()

    # 1. The formula, checked two independent ways
    print("1. Gatheral's total-variance formula, checked")
    flat = Surface([[0.2**2 * 0.5, 0.0, 0.0, 0.0, 0.1], [0.2**2 * 1.0, 0.0, 0.0, 0.0, 0.1]], [0.5, 1.0])
    kk = np.linspace(-1, 1, 201)
    print(f"   flat 20% surface: max |sigma_loc - 20%| = {np.max(np.abs(flat.dupire_raw(kk, 0.75)[0] - 0.2)):.1e}")
    worst, cnt = 0.0, 0
    for i in range(n):
        lo = S.T[i - 1] if i else 0.0
        t = 0.5 * (lo + S.T[i])
        k = np.linspace(-0.6, 0.4, 101)
        lv, f = S.dupire_raw(k, t)[0], S.flags(k, t)[0]
        m = (f & (E_FLAG | G_FLAG)) == 0
        if m.any():
            fd = dupire_price_form(S, k[m], t)
            worst = max(worst, float(np.max(np.abs(fd / lv[m] - 1))))
            cnt += int(m.sum())
    print(f"   vs Dupire on prices (c_T / (c_kk - c_k) by finite differences), {cnt} reliable points mid-interval "
          f"in all {n} intervals: max relative difference {worst:.1e}")
    print()

    # 2. Raw local vol
    KS = np.array([-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3])
    print("2. Raw Dupire local vol, %, mid-interval (between the expiry named and the one before); in [ ] = flagged")
    print("   " + " " * 21 + "".join(f"{x:+8.2f}" for x in KS) + "   (k = ln K/F)")
    for i in range(n):
        lo = S.T[i - 1] if i else 0.0
        t = 0.5 * (lo + S.T[i])
        lv, _ = S.dupire_raw(KS, t)
        f = S.flags(KS, t)[0]
        cells = "".join((f"{v * 100:8.1f}" if not ff else f"[{v * 100:6.1f}]") if np.isfinite(v) else "     nan"
                        for v, ff in zip(lv, f))
        print(f"   {('to ' + S.names[i]):14s} {t * 365:5.1f}d{cells}")
    print("   flags: E beyond the quoted strikes, G denominator g < 0.1, N fit noise moves it > 10% "
          "(any of them puts the cell in brackets)")
    print()

    # 3. Where it is unreliable: the region, and evidence for the thresholds
    KG = np.linspace(-2.0, 1.0, 601)
    TC = cell_times(S, per=4)
    F = np.array([S.flags(KG, t)[0] for t in TC])
    RAW = np.array([S.dupire_raw(KG, t)[0] for t in TC])
    G = np.array([S.flags(KG, t)[1] for t in TC])
    REG = np.array([S.local_vol(KG, t)[0] for t in TC])
    print(f"3. Unreliable region on k in [-2, 1] (step 0.005) x {len(TC)} times (4 per interval)")
    tot = F.size
    for lab, m in [("E beyond quoted strikes", (F & E_FLAG) > 0), ("G g < 0.1", (F & G_FLAG) > 0),
                   ("G inside the quoted strikes", ((F & G_FLAG) > 0) & ((F & E_FLAG) == 0)),
                   ("N time noise > 10%", (F & N_FLAG) > 0),
                   ("N only (kept exact, marked)", F == N_FLAG), ("no flag", F == 0)]:
        print(f"   {lab:30s} {m.sum() / tot:6.1%} of cells")
    print("   By expiry interval: the quoted k range (of the expiry ending it), the part of it Dupire is used as is")
    print("   (no E or G flag; everything else is regularized), the share of that part flagged N, and min g in it:")
    for i in range(n):
        rows = slice(4 * i, 4 * i + 4)
        q = (F[rows] & E_FLAG) == 0
        ok = (F[rows] & (E_FLAG | G_FLAG)) == 0
        lo_t = S.T[i - 1] if i else 0.0
        kq = KG[q.any(axis=0)]
        line = f"     {lo_t * 365:6.1f}-{S.T[i] * 365:6.1f}d  quoted k {kq.min():+.2f}..{kq.max():+.2f}" if kq.size else \
            f"     {lo_t * 365:6.1f}-{S.T[i] * 365:6.1f}d  quoted: none"
        if ok.any():
            ks = KG[ok.any(axis=0)]
            nn = np.mean((F[rows][ok] & N_FLAG) > 0)
            gmin = np.min(np.where(ok, G[rows], np.inf))
            line += f"   used {ks.min():+.2f}..{ks.max():+.2f}   N {nn:4.0%}   min g {gmin:.2f}"
        else:
            line += "   used: none (at-the-money implied vol)"
        print(line)

    # evidence: a second, independent arbitrage-free fit of the same quotes, if available
    alt_path = HERE.parent / "data" / folder.name / "svi_surface_alt.json"
    if alt_path.exists():
        alt = json.load(open(alt_path))
        A = Surface([x["svi"] for x in alt["slices"]], [x["T"] for x in alt["slices"]])
        RA = np.array([A.dupire_raw(KG, t)[0] for t in TC])
        with np.errstate(invalid="ignore"):
            rel = np.abs(RA / RAW - 1)
        print("   Evidence: Dupire on a second, independently built arbitrage-free SVI fit of the same quotes")
        print("   (svi_surface_alt.json: different repair, same data). Share of cells where the two local vols")
        print("   differ by more than 10%, and the median difference:")
        for lab, m in [("no flag", F == 0), ("N only", F == N_FLAG), ("G (inside quotes)", ((F & G_FLAG) > 0) & ((F & E_FLAG) == 0)),
                       ("E (beyond quotes)", (F & E_FLAG) > 0)]:
            r = rel[m & np.isfinite(rel)]
            print(f"     {lab:20s} {np.mean(r > 0.1):6.1%}   median {np.median(r):6.1%}   ({m.sum()} cells)")
    print()

    # 4. Regularization
    inq = (F & E_FLAG) == 0
    print(f"4. Regularization: the raw Dupire value held in [{LV_FLOOR:.0%}, {LV_CAP:.0%}] ({LV_CAP:.0%} where g <= 0);")
    print("   the flags mark the unreliable region and do not change the value.")
    print(f"   raw local vol range {np.nanmin(RAW) * 100:.1f}% .. {np.nanmax(RAW) * 100:.0f}%; "
          f"undefined (g <= 0) in {np.mean(~np.isfinite(RAW)):.1%} of cells")
    print(f"   clipped at the cap: {np.mean(np.nan_to_num(RAW, nan=9) > LV_CAP):.1%} of all cells, "
          f"{np.mean(np.nan_to_num(RAW[inq], nan=9) > LV_CAP):.1%} of cells inside the quoted strikes; "
          f"at the floor: {np.mean(np.nan_to_num(RAW, nan=9) < LV_FLOOR):.1%}")
    big = np.where(inq, np.nan_to_num(RAW, nan=0), 0)
    a, b = np.unravel_index(np.argmax(big), big.shape)
    print(f"   largest raw value inside the quoted strikes: {big[a, b] * 100:.0f}% at t = {TC[a] * 365:.2f}d, "
          f"k = {KG[b]:+.3f} (why the cap is 300%)")
    FIL = np.array([S.local_vol_filled(KG, t)[0] for t in TC])
    print(f"   rejected alternative (local_vol_filled: E and G cells replaced by the reliable local vol at the")
    print(f"   same time) changes {np.mean(np.abs(FIL - REG) > 1e-12):.1%} of cells; Stage 3's round trip "
          f"measures its cost on quoted options.")
    for t_d in (7, 30, 90, 365):
        t = t_d / 365
        lv, raw, f = S.local_vol(KG, t)
        fl = S.local_vol_filled(KG, t)[0]
        sel = [np.argmin(np.abs(KG - x)) for x in (-0.4, -0.2, 0.0, 0.1, 0.3)]
        print(f"   {t_d:4d}d  k = -0.4, -0.2, 0, +0.1, +0.3:  used " +
              " ".join(f"{lv[j] * 100:6.1f}" for j in sel) + "   filled " + " ".join(f"{fl[j] * 100:6.1f}" for j in sel))

    out = folder / "local_vol.json"
    KJ = np.round(np.linspace(-2.0, 1.0, 151), 4)
    json.dump({"note": "Dupire local vol (Gatheral total-variance form) on Stage 1's SVI surface; k = ln(K/F_T); "
                       "t = cell centres (4 per expiry interval); flags 1 E beyond quotes, 2 G g<0.1, 4 N time noise",
               "k": KJ.tolist(), "t": [round(float(t), 8) for t in TC], "expiries": S.T.tolist(),
               "lv": [[round(float(x), 5) for x in S.local_vol(KJ, t)[0]] for t in TC],
               "lv_raw": [[None if not np.isfinite(x) else round(float(x), 5) for x in S.dupire_raw(KJ, t)[0]] for t in TC],
               "flags": [[int(x) for x in S.flags(KJ, t)[0]] for t in TC]},
              open(out, "w"), separators=(",", ":"))
    print(f"\nwrote {out.relative_to(HERE.parent)}")
