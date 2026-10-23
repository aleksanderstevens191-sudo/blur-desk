"""Save a complete SPY option-chain snapshot: every listed expiry, every strike, calls and puts.

    pip install yfinance
    python reference/fetch_spy_chain_full.py            # -> data/spy_full_<date>/

Run it while the market is open (9:30-16:00 ET). Two independent sources, each
saved untouched next to one normalized CSV:

  cboe_raw.json    Cboe's delayed-quotes feed for SPY: the whole chain in one
                   response, stamped by Cboe (15-minute delayed).
  yahoo_raw.csv    Yahoo Finance via yfinance, one request per expiry, every column.
  chain.csv        what the stage 1 script reads, built from Cboe when it worked,
                   otherwise from Yahoo. Header line records source, timestamp,
                   spot and contract count.

Nothing is filtered here; reference/chain_stage1.py applies and reports the filters.
"""
import sys
import json
import ssl
import datetime as dt
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
now = dt.datetime.now(ET)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data" / f"spy_full_{now.date()}"
out.mkdir(parents=True, exist_ok=True)
COLS = "expiry,strike,type,bid,ask,bid_size,ask_size,last,volume,open_interest,last_trade_utc,vendor_iv"


def fetch_cboe():
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request("https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json",
                                 headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60, context=ctx).read()
    (out / "cboe_raw.json").write_bytes(raw)
    doc = json.loads(raw)
    d = doc["data"]
    rows = []
    for o in d["options"]:
        sym = o["option"]                  # e.g. SPY261120C00740000
        exp = dt.date(2000 + int(sym[-15:-13]), int(sym[-13:-11]), int(sym[-11:-9]))
        cp = "call" if sym[-9] == "C" else "put"
        K = int(sym[-8:]) / 1000
        g = lambda k: o.get(k) if o.get(k) is not None else ""
        rows.append(f"{exp},{K:g},{cp},{g('bid')},{g('ask')},{g('bid_size')},{g('ask_size')},"
                    f"{g('last_trade_price')},{g('volume')},{g('open_interest')},{g('last_trade_time')},{g('iv')}")
    spot = d.get("current_price") or d.get("close")
    stamp = doc.get("timestamp") or d.get("last_trade_time") or ""
    return float(spot), str(stamp), rows


def fetch_yahoo():
    import yfinance as yf
    spy = yf.Ticker("SPY")
    spot = float(spy.fast_info["last_price"])
    frames = []
    for e in spy.options:
        ch = spy.option_chain(e)
        for cp, df in (("call", ch.calls), ("put", ch.puts)):
            df = df.copy()
            df.insert(0, "type", cp)
            df.insert(0, "expiry", e)
            frames.append(df)
    import pandas as pd
    allq = pd.concat(frames, ignore_index=True)
    allq.to_csv(out / "yahoo_raw.csv", index=False)
    num = lambda v: "" if pd.isna(v) else v
    rows = [f"{q.expiry},{q.strike:g},{q.type},{num(q.bid)},{num(q.ask)},,,{num(q.lastPrice)},{num(q.volume)},"
            f"{num(q.openInterest)},{num(q.lastTradeDate)},{num(q.impliedVolatility)}" for q in allq.itertuples()]
    return spot, dt.datetime.now(ET).isoformat(timespec="seconds"), rows


got = {}
for name, fn in (("cboe", fetch_cboe), ("yahoo", fetch_yahoo)):
    try:
        got[name] = fn()
        spot, stamp, rows = got[name]
        exps = sorted({r.split(",")[0] for r in rows})
        print(f"{name}: {len(rows)} contracts, {len(exps)} expiries ({exps[0]} to {exps[-1]}), "
              f"spot {spot:.2f}, stamp {stamp}")
    except Exception as err:                       # keep going: one source is enough
        print(f"{name}: failed ({type(err).__name__}: {err})")

if not got:
    sys.exit("no source worked; check the network and rerun")
src = "cboe" if "cboe" in got else "yahoo"
spot, stamp, rows = got[src]
label = {"cboe": "Cboe delayed quotes (cdn.cboe.com, 15-min delayed)", "yahoo": "Yahoo Finance via yfinance"}[src]
with open(out / "chain.csv", "w") as f:
    f.write(f"# spot={spot:.4f} asof={now.date()} fetched={now.isoformat(timespec='seconds')} "
            f"source_stamp={stamp.replace(' ', 'T')} source={src} contracts={len(rows)} label={label}\n")
    f.write(COLS + "\n" + "\n".join(rows) + "\n")
print(f"\nwrote {out}/chain.csv from {src}: {len(rows)} contracts")
print("then:  git add data && git commit -m 'SPY full chain snapshot' && git push")
