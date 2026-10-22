"""Save an SPY option-chain snapshot from Yahoo Finance in the format calibrate.py reads.

    pip install yfinance
    python reference/fetch_spy_chain.py data/spy_chain_2026-10-23.csv

Run it while the market is open: outside trading hours Yahoo often reports zero
bids and asks, and the calibration only uses two-sided quotes.

Takes about 8 expiries spread from one week to two years (the nearest listed
expiry to each target) and strikes within 35% of spot. The first line records
spot, date and time, so the fit can be rerun later from the file alone.
"""
import sys
import datetime as dt
from zoneinfo import ZoneInfo

import yfinance as yf

TARGET_DAYS = [7, 14, 30, 60, 90, 180, 365, 730]
out = sys.argv[1] if len(sys.argv) > 1 else "spy_chain.csv"

now = dt.datetime.now(ZoneInfo("America/New_York"))
spy = yf.Ticker("SPY")
spot = float(spy.fast_info["last_price"])
listed = [dt.date.fromisoformat(e) for e in spy.options]
picked = sorted({min(listed, key=lambda e: abs((e - now.date()).days - d)) for d in TARGET_DAYS})

rows, zero = [], 0
for exp in picked:
    chain = spy.option_chain(exp.isoformat())
    for cp, df in (("call", chain.calls), ("put", chain.puts)):
        for _, q in df.iterrows():
            if abs(q["strike"] / spot - 1) > 0.35:
                continue
            bid, ask = float(q["bid"] or 0), float(q["ask"] or 0)
            zero += bid <= 0 or ask <= 0
            rows.append(f"{exp.isoformat()},{q['strike']:g},{cp},{bid:.2f},{ask:.2f}")

with open(out, "w") as f:
    f.write(f"# spot={spot:.2f} asof={now.date().isoformat()} time={now:%H:%M}ET source=yahoo\n")
    f.write("expiry,strike,type,bid,ask\n")
    f.write("\n".join(rows) + "\n")
print(f"wrote {out}: spot {spot:.2f}, {len(picked)} expiries "
      f"({picked[0]} to {picked[-1]}), {len(rows)} quotes")
if zero > len(rows) / 2:
    print(f"warning: {zero} of {len(rows)} quotes have a zero bid or ask; "
          "rerun during market hours (9:30-16:00 ET)")
