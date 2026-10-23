"""CI: rerun the whole chain pipeline from the saved snapshot and compare with what is committed.

python reference/check_chain_fit.py      (about 6 minutes; exits 1 on any mismatch)

1. reference/chain_stage1.py on data/spy_full_2026-10-23/chain.csv: filters, early-exercise-
   adjusted parity forwards, European vols and premiums, into a scratch folder.
2. reference/chain_fit.py on that: Merton per expiry and the reference prices.
3. Compare with the committed data/.../stage1.json and src/chain_data.json: filter counts,
   every forward, every Merton parameter, every reference price, every fit error.
"""
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
SNAP = ROOT / "data" / "spy_full_2026-10-23"
fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {what}")


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for f in ("chain.csv", "yahoo_raw.csv"):
        shutil.copy(SNAP / f, tmp / f)
    print("rerunning stage 1 (filters, forwards, early exercise) and the Merton fit from the saved snapshot")
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], check=True, capture_output=True, text=True).stdout
    s1_report = run(ROOT / "reference" / "chain_stage1.py", tmp / "chain.csv")
    fit_report = run(ROOT / "reference" / "chain_fit.py", tmp, tmp / "chain_data.json")
    new1, old1 = (json.load(open(p)) for p in (tmp / "stage1.json", SNAP / "stage1.json"))
    new2, old2 = (json.load(open(p)) for p in (tmp / "chain_data.json", ROOT / "src" / "chain_data.json"))

print("\nstage 1")
check(len(new1["contracts"]) == len(old1["contracts"]), f"{len(new1['contracts'])} contracts kept by the filters "
      f"(committed {len(old1['contracts'])})")
dF = max(abs(a["F"] - b["F"]) for a, b in zip(new1["expiries"], old1["expiries"]))
check(len(new1["expiries"]) == len(old1["expiries"]) and dF < 1e-4,
      f"{len(new1['expiries'])} forwards, max change {dF:.1e} dollars (< 1e-4)")
key = lambda c: (c["expiry"], c["K"], c["cp"])
o = {key(c): c for c in old1["contracts"]}
dv = max((abs(c["iv"] - o[key(c)]["iv"]) for c in new1["contracts"] if c["iv"] is not None and key(c) in o
          and o[key(c)]["iv"] is not None), default=np.inf)
check(dv < 1e-8, f"European vols, max change {dv:.1e} (< 1e-8)")

print("\nMerton fit and reference prices")
dp = 0.0
for a, b in zip(new2["expiries"], old2["expiries"]):
    for k in ("sigma", "lam", "m", "delta"):
        dp = max(dp, abs(a["merton"][k] - b["merton"][k]) / max(abs(b["merton"][k]), 1e-3))
check(dp < 1e-4, f"Merton parameters of {len(new2['expiries'])} expiries, max relative change {dp:.1e} (< 1e-4)")
dr = max(abs(a["merton"]["rms"] - b["merton"]["rms"]) for a, b in zip(new2["expiries"], old2["expiries"]))
check(dr < 1e-3, f"fit errors, max change {dr:.1e} vol points (< 1e-3)")
for name in ("bs_ql", "merton"):
    a, b = np.array(new2["contracts"][name]), np.array(old2["contracts"][name])
    d = np.max(np.abs(a - b)) if a.shape == b.shape else np.inf
    check(d < 1e-6, f"reference prices '{name}', {len(a)} contracts, max change {d:.1e} dollars (< 1e-6)")
print("\n" + "\n".join(l for l in fit_report.splitlines() if l.startswith("  all expiries:")))
print(f"\n{'all match' if not fails else f'{fails} failed'}")
sys.exit(1 if fails else 0)
