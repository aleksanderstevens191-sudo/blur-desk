"""Stage 2: calibrate flat Black-Scholes and Merton to the SPY snapshot in data/.

python reference/calibrate_spy.py [chain.csv]   -> prints the report (spy_calibration_results.txt)
Also writes src/spy_fit.json (market smile + fitted curves) for the demo overlay.
"""
import sys
import json
import time
import datetime as dt
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calibrate import load_chain, prepare, fit_flat, fit_merton, merton_ivs, errors

SPY_CSV = HERE.parent / "data" / "spy_chain_2026-10-22.csv"
# US Treasury constant-maturity yields on the snapshot date (FMP treasury-rates, 2026-10-22),
# interpolated to each expiry. Bond-equivalent and continuous differ by < 0.01% here.
CURVE_DAYS = np.array([30, 61, 91, 182, 365])
CURVE_RATE = np.array([4.11, 4.16, 4.22, 4.29, 4.41]) / 100


def load_spy(path=SPY_CSV):
    """The snapshot's out-of-the-money slices, each expiry at its Treasury rate (also used by check_spy_fit.py)."""
    spot, asof, rows = load_chain(path)
    rate = lambda exp: float(np.interp((exp - asof).days, CURVE_DAYS, CURVE_RATE))
    dropped = []
    slices = prepare(spot, asof, rows, rates=rate, dropped=dropped)
    return spot, asof, rows, slices, dropped


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SPY_CSV
    t0 = time.time()
    spot, asof, rows, slices, dropped = load_spy(path)

    print(f"SPY option chain {path.name}: {len(rows)} quotes, as of {asof}, SPY close {spot}")
    print("Source: Yahoo Finance delayed quotes (uploaded by Alek). Rates: US Treasury curve, same day.")
    print()
    print("Expiries used and skipped")
    for s in slices:
        print(f"  {s['expiry']}  {s['T'] * 365:3.0f} days  r {s['r']:.2%}  forward {s['F']:.2f} (parity, 4 strikes "
              f"nearest the money)  spot net of dividends {s['S']:.2f}  {len(s['K'])} OTM quotes")
    for e, why in dropped:
        print(f"  {e}  skipped: {why}")
    print()

    n = sum(len(s["K"]) for s in slices)
    half = np.concatenate([(s["iv_ask"] - s["iv_bid"]) / 2 for s in slices]) * 100
    s_flat, flat = fit_flat(slices)
    x = fit_merton(slices)
    mert = merton_ivs(slices, *x, pricer="blur")
    rms_f, max_f = errors(flat, slices)
    rms_m, max_m = errors(mert, slices)

    print(f"Least squares on implied vol, {n} out-of-the-money mids (puts below the forward, calls above)")
    print(f"Market half-spread: mean {half.mean():.2f}, max {half.max():.2f} vol points")
    print()
    print("  model                parameters                                         RMS     max   (vol points)")
    print(f"  flat Black-Scholes   sigma {s_flat:.2%}                                       {rms_f:6.2f}  {max_f:6.2f}")
    print(f"  Merton               sigma {x[0]:.2%}, lam {x[1]:.2f}/yr, m {x[2]:+.2%}, delta {x[3]:.2%}   "
          f"{rms_m:6.2f}  {max_m:6.2f}")
    kappa = np.expm1(x[2] + 0.5 * x[3] ** 2)
    print(f"  Merton's jumps: {x[1]:.2f} a year, average move {kappa:+.1%} each; the market prices in a")
    print(f"  crash-type jump about every {1 / x[1] * 12:.0f} months. Merton prices are from the blur (Merton kernel).")
    print()
    print("Per quote: market vol [bid, ask] vs model, vol points")
    for s, fl, me in zip(slices, flat, mert):
        print(f"  {s['expiry']}")
        for K, cp, iv, b, a, f, m in zip(s["K"], s["cp"], s["iv"], s["iv_bid"], s["iv_ask"], fl, me):
            print(f"    {K:5.0f} {cp:4s}  market {iv * 100:5.2f} [{b * 100:5.2f},{a * 100:5.2f}]   "
                  f"flat {f * 100:5.2f} ({(f - iv) * 100:+5.2f})   Merton {m * 100:5.2f} ({(m - iv) * 100:+5.2f})")
    print(f"\ntotal time {time.time() - t0:.0f} s")

    json.dump({
        "asof": str(asof), "spot": spot, "source": "Yahoo Finance delayed quotes",
        "flat": {"sigma": s_flat, "rms_vol_pts": rms_f, "max_vol_pts": max_f},
        "merton": {"sigma": x[0], "lam": x[1], "m": x[2], "delta": x[3], "rms_vol_pts": rms_m, "max_vol_pts": max_m},
        "slices": [{"expiry": str(s["expiry"]), "T": s["T"], "r": s["r"], "F": s["F"], "S": s["S"],
                    "K": s["K"].tolist(), "cp": s["cp"], "iv": s["iv"].tolist(),
                    "iv_bid": s["iv_bid"].tolist(), "iv_ask": s["iv_ask"].tolist(),
                    "merton_iv": me.tolist()} for s, me in zip(slices, mert)],
    }, open(HERE.parent / "src" / "spy_fit.json", "w"), indent=1)
