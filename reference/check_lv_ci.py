"""Part 4 in CI: the local-vol checks that fit in a few minutes. Exits 1 if any fails.

python reference/check_lv_ci.py [data/spy_full_<date>]

The full checks are reference/check_lv_blur.py (Stage 3, ~25 min on 4 processes) and reference/model_risk.py
(Stage 4, ~20 min). This one keeps a slice of each, on the committed files:
  1. the Stage 1 surface (svi_surface.json) has no arbitrage: Durrleman's g >= 0 on every slice for k in
     [-3, 1.5], the edge-detect density of its call prices >= 0 at every $0.05 strike, and total variance
     rising with expiry at every k
  2. Gatheral's total-variance Dupire agrees with Dupire on prices (finite differences) where neither is flagged
  3. the blur at a flat 20% vol matches Black-Scholes within 1e-6 (per $1 of forward) at 28 days
  4. the GPU's fixtures (src/lv_fixtures.json) are what lv_blur.py gives on the web table today, within 1e-10
  5. Stage 4's 98-day up-and-out call (H 115%, K 100%) under all three models matches model_risk.json
     within $1e-4, and local vol and Merton stay more than $3 apart on it
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import local_vol as LVM
from lv_blur import Flat, price_backward, black_c
from make_lv_data import table, GPU
import model_risk as MR

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
fails = 0


def line(ok, s):
    global fails
    fails += not ok
    print(f"   {'PASS' if ok else 'FAIL'} {s}")


def main():
    t00 = time.time()
    S = LVM.Surface.load(FOLDER)
    n = len(S.T)
    print(f"Part 4 CI checks on {FOLDER.name}: {n} expiries, {S.T[0] * 365:.1f} to {S.T[-1] * 365:.0f} days")

    print("1. Stage 1 surface: no arbitrage")
    k = np.arange(-3.0, 1.5 + 1e-9, 0.001)
    gmin, dmin, cal = np.inf, np.inf, np.inf
    for i in range(n):
        w, w1, w2 = LVM.svi_d(S.P[i], k)
        gmin = min(gmin, float(np.min(LVM.durrleman(w, w1, w2, k))))
        K = np.arange(np.ceil(S.F[i] * np.exp(-3.0) / 0.05) * 0.05, S.F[i] * np.exp(1.5), 0.05)
        kk = np.log(K / S.F[i])
        c = S.F[i] * LVM.black_c(kk, LVM.svi(S.P[i], kk))
        dens = (c[2:] - 2 * c[1:-1] + c[:-2]) / 0.05**2
        dmin = min(dmin, float(np.min(dens)) / S.F[i])
        if i:
            cal = min(cal, float(np.min(w - LVM.svi(S.P[i - 1], k))))
    line(gmin >= 0, f"butterfly: min Durrleman g over {n} slices {gmin:.2e} (>= 0)")
    line(dmin > -1e-9, f"butterfly: min edge-detect density of the call prices {dmin:.1e} per $ of forward (>= -1e-9, roundoff)")
    line(cal >= 0, f"calendar: min w(T_i+1) - w(T_i) over {n - 1} pairs {cal:.2e} (>= 0)")

    print("2. Stage 2: Gatheral's formula vs Dupire on prices, unflagged cells")
    worst, m = 0.0, 0
    for i in range(1, n):
        t = 0.5 * (S.T[i - 1] + S.T[i])
        kq = np.linspace(S.kq[i][0], S.kq[i][1], 9)
        lv, raw, f = S.local_vol(kq, t)
        ok = (f == 0) & np.isfinite(raw)
        if ok.any():
            fd = LVM.dupire_price_form(S, kq[ok], t)
            worst = max(worst, float(np.max(np.abs(fd / raw[ok] - 1))))
            m += int(ok.sum())
    line(m >= 20 and worst < 1e-3, f"{m} points mid-interval: max relative difference {worst:.1e} (< 1e-3)")

    print("3. Stage 3: the blur at a flat 20% vol vs Black-Scholes")
    i = 11
    T = S.T[i]
    worst = 0.0
    for kk in (-0.1, 0.0, 0.1):
        for call in (True, False):
            v = price_backward(Flat(0.2), kk, T, call, GPU["dt"], 0.0008, -4.0, 2.5, m=GPU["m"], knots=list(S.T))
            worst = max(worst, abs(v - float(black_c(kk, 0.04 * T, call))))
    line(worst < 1e-6, f"{S.names[i]} ({T * 365:.0f} days), k -0.1 / 0 / 0.1, calls and puts: max {worst:.1e} per $1 of forward (< 1e-6)")

    print("4. The GPU's fixtures: lv_blur.py on the web table, today")
    t0 = time.time()
    tab, _ = table(S)
    fx = json.load(open(HERE.parent / "src" / "lv_fixtures.json"))
    cases = [c for c in fx["cases"] if c["model"] == "lv" and c["i"] in (0, 4)] + \
            [c for c in fx["cases"] if c["model"] == "flat" and c["i"] == 11 and c["k"] == 0.0]
    worst = 0.0
    g = fx["settings"]
    for c in cases:
        sig = Flat(c["sigma"]) if c["model"] == "flat" else tab
        v = price_backward(sig, c["k"], S.T[c["i"]], c["call"], g["dt"], g["dx"], g["x_lo"], g["x_hi"], m=g["m"], knots=list(S.T))
        worst = max(worst, abs(100 * S.D[c["i"]] * v - c["py"]))
    line(worst < 1e-10, f"{len(cases)} of the {len(fx['cases'])} fixtures recomputed: max {worst:.1e} per $100 of forward "
                        f"(< 1e-10; {time.time() - t0:.0f} s)")

    print("5. Stage 4: model risk on the 98-day up-and-out call, H 115%, K 100%")
    mr = json.load(open(FOLDER / "model_risk.json"))
    i = MR.PICK[0]
    row = [r for r in mr["expiries"][S.names[i]]["rows"]
           if r["type"] == "up-and-out call" and r["barrier"] == 1.15 and r["strike"] == 1.0][0]
    mp = [mr["merton"][x] for x in ("sigma", "lam", "m", "delta")]
    P = MR.PROD
    got = {}
    for name, params in (("bs", mr["atm_vol"][S.names[i]]), ("merton", mp), ("lv", None)):
        t0 = time.time()
        v, _ = MR.run((name, params, i, "up", P["dt"], P["m"], P["dx"]))
        got[name] = float(v[0])                     # the first up payoff: K 100% call
        line(abs(got[name] - row[name]) < 1e-4, f"{name:6s} ${got[name]:.4f} vs committed ${row[name]:.4f} "
                                                f"(diff {got[name] - row[name]:+.1e}; {time.time() - t0:.0f} s)")
    line(got["lv"] - got["merton"] > 3, f"local vol minus Merton ${got['lv'] - got['merton']:.2f} (> $3: the models disagree)")

    print(f"\n{'all PASS' if not fails else f'{fails} FAILED'} ({time.time() - t00:.0f} s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
