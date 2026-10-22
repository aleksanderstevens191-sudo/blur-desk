"""QuantLib on the CPU, for the GPU barrier benchmark (bench.html builds the same options).

Every single-barrier type x 32 barriers x 64 strikes (x scale^2), priced with
QuantLib's AnalyticBarrierEngine. Usage: python3 reference/bench_quantlib.py [scale]
"""
import sys
import time
import numpy as np
import QuantLib as ql

S0, SIGMA, T_DAYS, R, DX = 100.0, 0.2, 365, 0.05, 0.002
TYPES = ["down-and-out call", "down-and-in call", "up-and-out call", "up-and-in call",
         "down-and-out put", "down-and-in put", "up-and-out put", "up-and-in put"]


def option_set(nH, nK):
    steps = lambda m_max: [round(5 + i * (m_max - 5) / (nH - 1)) for i in range(nH)]
    down = steps(round(np.log(S0 / 70) / DX))
    up = steps(round(np.log(1.4) / DX))
    strikes = [70 + i * 65 / (nK - 1) for i in range(nK)]
    out = []
    for kind in TYPES:
        is_down = kind.startswith("down")
        for m in (down if is_down else up):
            for K in strikes:
                out.append((kind, K, S0 * np.exp((-m if is_down else m) * DX)))
    return out


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    options = option_set(32 * scale, 64 * scale)
    today = ql.Date(20, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    spot = ql.SimpleQuote(S0)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, R, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), SIGMA, dc)))
    engine = ql.AnalyticBarrierEngine(process)
    btype = {("down", "in"): ql.Barrier.DownIn, ("down", "out"): ql.Barrier.DownOut,
             ("up", "in"): ql.Barrier.UpIn, ("up", "out"): ql.Barrier.UpOut}
    exercise = ql.EuropeanExercise(today + T_DAYS)

    t0 = time.perf_counter()
    insts = []
    for kind, K, H in options:
        ud, _, io, cp = kind.replace("-", " ").split()
        o = ql.BarrierOption(btype[(ud, io)], H, 0.0,
                             ql.PlainVanillaPayoff(ql.Option.Call if cp == "call" else ql.Option.Put, K), exercise)
        o.setPricingEngine(engine)
        insts.append(o)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    first = [o.NPV() for o in insts]
    t_first = time.perf_counter() - t0

    # reprice: move the spot so every instrument recalculates
    times = []
    for bump in [1e-9, 0.0, 1e-9, 0.0, 1e-9]:
        spot.setValue(S0 + bump)
        t0 = time.perf_counter()
        for o in insts:
            o.NPV()
        times.append(time.perf_counter() - t0)
    spot.setValue(S0)
    t_re = sorted(times)[len(times) // 2]

    n = len(options)
    print(f"QuantLib {ql.__version__} AnalyticBarrierEngine, Python bindings, 1 CPU thread")
    print(f"options: 8 types x {32 * scale} barriers x {64 * scale} strikes = {n:,}")
    print(f"  build instruments:      {t_build * 1e3:8.1f} ms")
    print(f"  first pricing:          {t_first * 1e3:8.1f} ms  ->  {n / t_first:,.0f} options/sec")
    print(f"  reprice after a bump:   {t_re * 1e3:8.1f} ms  ->  {n / t_re:,.0f} options/sec (median of 5)")
    print(f"  build + price:          {(t_build + t_first) * 1e3:8.1f} ms  ->  {n / (t_build + t_first):,.0f} options/sec")
    # a few prices so the two benchmarks can be compared option by option
    for i in [0, n // 3, 2 * n // 3, n - 1]:
        kind, K, H = options[i]
        print(f"  #{i}: {kind} K={K:.4f} H={H:.4f}  {first[i]:.8f}")


if __name__ == "__main__":
    main()
