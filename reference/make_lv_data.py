"""Part 4 Stage 5: the data and fixtures for the local-vol GPU step and demo panel 15.

python reference/make_lv_data.py [data/spy_full_<date>]
  -> src/lv_data.json      the local-vol table (as the GPU uses it), SPY's expiries, the OTM quotes with the
                           surface vol and the round-trip error at the GPU's settings, Stage 4's model-risk table
  -> src/lv_fixtures.json  Python prices on that same table for the GPU and the JS twin to match

The table is Stage 3's LVTable made lighter for a web page: 4 rows per expiry interval instead of 8, k every
0.005 on [-3, 1.5] plus every 0.001 on [-0.3, 0.2], vols rounded to 1e-4 (0.01 vol points) and stored as
integers. Every price here is computed on the rounded table, so the GPU, the JS twin and these fixtures all
price exactly one model. Prices are per $100 of forward, discounted: 100 D_T c(k) with c in forward units.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import local_vol as LVM
from lv_blur import LVTable, Flat, price_backward, prices_forward, implied_w, black_c

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "spy_full_2026-10-23"
GPU = dict(dt=1 / 365, m=32, dx=0.0016, x_lo=-3.0, x_hi=1.5)
LIQ_MID = 0.50


def table(S):
    t = LVTable.from_surface(S, dk=0.005, per=4, k_lo=-3.0, k_hi=1.5, fine=(-0.3, 0.2, 0.001))
    q = np.round(t.sig * 1e4).astype(int)
    return LVTable(t.t, t.k, q / 1e4, t.edges), q


def main():
    S = LVM.Surface.load(FOLDER)
    s1 = json.load(open(FOLDER / "stage1.json"))
    tab, q = table(S)
    n = len(S.T)
    g = GPU
    knots = list(S.T)
    print(f"table: {len(tab.t)} rows x {len(tab.k)} strikes; grid dx {g['dx']} on [{g['x_lo']}, {g['x_hi']}], "
          f"steps <= {g['dt'] * 365:g} day, >= {g['m']} per interval")

    # quotes and the round trip at the GPU's settings, all expiries in one forward pass
    Q = []
    for i, e in enumerate(S.names):
        F = S.F[i]
        cs = [c for c in s1["contracts"] if c["expiry"] == e and c["iv_true"] is not None and (c["cp"] == "call") == (c["K"] >= F)]
        Q.append(dict(k=np.array([np.log(c["K"] / F) for c in cs]), call=np.array([c["cp"] == "call" for c in cs]),
                      K=[c["K"] for c in cs], liq=[0.5 * (c["bid"] + c["ask"]) >= LIQ_MID for c in cs]))
    t0 = time.time()
    out = prices_forward(tab, list(S.T), [q_["k"] for q_ in Q], [q_["call"] for q_ in Q], g["dt"], g["dx"], g["x_lo"], g["x_hi"], m=g["m"])
    print(f"round trip, forward pass over {sum(len(q_['k']) for q_ in Q)} quotes: {time.time() - t0:.0f} s")
    allerr = []
    expiries = []
    for i in range(n):
        iv_s = S.implied_vol(Q[i]["k"], S.T[i])
        err = (np.sqrt(implied_w(out[i], Q[i]["k"], Q[i]["call"]) / S.T[i]) - iv_s) * 100
        liq = np.array(Q[i]["liq"])
        if liq.any():
            allerr.append(err[liq])
        expiries.append(dict(name=S.names[i], T=float(S.T[i]), F=float(S.F[i]), D=float(S.D[i]),
                             atm=float(S.implied_vol(np.array([0.0]), S.T[i])[0]),
                             k=[round(float(x), 6) for x in Q[i]["k"]], K=Q[i]["K"], call=[bool(c) for c in Q[i]["call"]],
                             liq=[bool(x) for x in liq], iv=[round(float(x), 6) for x in iv_s],
                             err=[None if not np.isfinite(x) else round(float(x), 4) for x in err]))
    a = np.concatenate(allerr)
    print(f"  liquid quotes ({a.size}): max {np.nanmax(np.abs(a)):.3f}, rms {np.sqrt(np.nanmean(a ** 2)):.3f} vol points "
          f"(Stage 3, full table and grid: max 0.073, rms 0.017)")

    mr = json.load(open(FOLDER / "model_risk.json"))
    data = dict(note="Part 4 local vol for the demo (reference/make_lv_data.py): table sig in 1e-4 units, rows at t inside "
                     "the expiry intervals (edges); quotes are SPY's OTM quotes on Oct 23 2026 with the Stage 1 surface vol "
                     "and the round-trip error (vol points) at the GPU settings",
                spot=s1["spot_implied"], settings=g, rt=dict(max=float(np.nanmax(np.abs(a))), rms=float(np.sqrt(np.nanmean(a ** 2))),
                                                             n=int(a.size)),
                table=dict(t=[round(float(x), 9) for x in tab.t], edges=[round(float(x), 9) for x in tab.edges],
                           k=[round(float(x), 6) for x in tab.k], sig=q.tolist()),
                expiries=expiries, modelRisk=mr)
    json.dump(data, open(HERE.parent / "src" / "lv_data.json", "w"), separators=(",", ":"))

    # fixtures: flat vol (also against Black) and local vol at five expiries
    cases = []
    t0 = time.time()
    for i in (11, 26):
        for k in (-0.1, 0.0, 0.1):
            for c in (True, False):
                v = price_backward(Flat(0.2), k, S.T[i], c, g["dt"], g["dx"], g["x_lo"], g["x_hi"], m=g["m"], knots=knots)
                cases.append(dict(model="flat", sigma=0.2, i=i, k=k, call=c, py=100 * S.D[i] * v,
                                  bs=100 * S.D[i] * float(black_c(k, 0.04 * S.T[i], c))))
    for i in (0, 4, 11, 18, 26):
        sd = np.sqrt(S.implied_vol(np.array([0.0]), S.T[i])[0] ** 2 * S.T[i])
        ks = [round(float(z * sd), 6) for z in (-2.5, -1.0, 0.0, 1.0)]   # as stored
        for k in ks:
            c = k >= 0
            v = price_backward(tab, k, S.T[i], c, g["dt"], g["dx"], g["x_lo"], g["x_hi"], m=g["m"], knots=knots)
            cases.append(dict(model="lv", i=i, k=round(k, 6), call=c, py=100 * S.D[i] * v,
                              iv=float(np.sqrt(implied_w(v, k, c) / S.T[i])), iv_surface=float(S.implied_vol(np.array([k]), S.T[i])[0])))
    print(f"fixtures: {len(cases)} backward prices in {time.time() - t0:.0f} s")
    for c in cases:
        extra = f"  BS {c['bs']:.8f}  |py - BS| {abs(c['py'] - c['bs']):.1e}" if c["model"] == "flat" else \
            f"  vol {c['iv'] * 100:.3f} vs surface {c['iv_surface'] * 100:.3f}"
        print(f"  {c['model']:4s} {S.names[c['i']]} k {c['k']:+.4f} {'call' if c['call'] else 'put '}  {c['py']:.8f}{extra}")
    json.dump(dict(note="Python prices (lv_blur.price_backward) on lv_data.json's table, per $100 of forward, discounted",
                   settings=g, cases=cases), open(HERE.parent / "src" / "lv_fixtures.json", "w"), indent=1)
    print("wrote src/lv_data.json, src/lv_fixtures.json")


if __name__ == "__main__":
    main()
