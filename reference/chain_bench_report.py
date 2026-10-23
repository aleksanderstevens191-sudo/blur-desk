"""The full-chain benchmark table: QuantLib (reference/bench_chain_quantlib.py) vs the GPU
(/chain-bench.html), both run on the same machine, from the JSON lines each one prints.

    python3 reference/chain_bench_report.py reference/chain_bench_quantlib_mac.txt reference/chain_bench_gpu_mac.txt

Rates are contracts per second at the median of the runs; the spread is [min, max].
Ratio = GPU contracts/sec over QuantLib contracts/sec (above 1: the GPU is faster).
"""
import sys
import json


def last_json(path):
    for line in reversed(open(path).read().splitlines()):
        if line.startswith("JSON "):
            return json.loads(line[5:])
    raise SystemExit(f"no JSON line in {path}")


ql_, gpu = last_json(sys.argv[1]), last_json(sys.argv[2])
n = ql_["contracts"]
assert n == gpu["contracts"], "the two runs priced different chains"
print(f"{n} SPY contracts, median of {ql_['runs']} (QuantLib) and {gpu['runs']} (GPU) runs; GPU {gpu['gpu']}; "
      f"QuantLib {ql_['quantlib']} on {ql_['cores']} cores")
print()
print("| model | method | time, ms (median [min, max]) | contracts/sec | GPU speed-up |")
print("|---|---|---|---:|---:|")
for name, label in (("bs", "Black-Scholes, own vol"), ("merton", "Merton per expiry")):
    rows = []
    q = ql_[name]
    for tag, t in (("QuantLib, 1 thread", q["one"]), (f"QuantLib, {ql_['cores']} cores", q["all"])):
        rows.append((tag, t["median"] * 1e3, t["min"] * 1e3, t["max"] * 1e3))
    g = gpu[name]
    for tag, t in (("GPU, compute only", g["compute"]), ("GPU, end to end", g["e2e"])):
        rows.append((tag, t["median"], t["min"], t["max"]))
    for tag, med, lo, hi in rows:
        print(f"| {label} | {tag} | {med:.2f} [{lo:.2f}, {hi:.2f}] | {n / med * 1e3:,.0f} | |")
    for gtag, gt in (("compute only", g["compute"]["median"]), ("end to end", g["e2e"]["median"])):
        for qtag, qt in (("1 thread", q["one"]["median"] * 1e3), (f"{ql_['cores']} cores", q["all"]["median"] * 1e3)):
            print(f"| {label} | GPU {gtag} vs QuantLib {qtag} | | | {qt / gt:.1f}x |")
print()
for name in ("bs", "merton", "both"):
    p = gpu[name]["parts"]
    print(f"GPU end to end, {name}: build {p['build']:.1f} ms, shader compile {p['compile']:.1f} ms, "
          f"upload + run + read back {p['run']:.1f} ms, finish {p['finish']:.1f} ms (medians)")
