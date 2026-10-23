"""Save a complete SPY option-chain snapshot: every listed expiry, every strike, calls and puts.

    pip install yfinance
    python reference/fetch_spy_chain_full.py            # -> data/spy_full_<date>/
    python reference/fetch_spy_chain_full.py --rebuild data/spy_full_2026-10-23

Run it while the market is open (9:30-16:00 ET). Two independent sources, each
saved untouched next to one normalized CSV:

  cboe_raw.json    Cboe's delayed-quotes feed for SPY: the whole chain in one
                   response. Quotes are 15 minutes delayed; the underlying's
                   last_trade_time (US Eastern) is the moment they describe.
  yahoo_raw.csv    Yahoo Finance via yfinance, one request per expiry, every column
                   (also delayed; lists fewer expiries than Cboe).
  chain.csv        what reference/chain.py reads, built from cboe_raw.json when
                   Cboe worked, otherwise from Yahoo. The header records source,
                   fetch time, quote time, spot and contract count.

--rebuild regenerates chain.csv from the raw files already in a folder.
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
COLS = "expiry,strike,type,bid,ask,bid_size,ask_size,last,volume,open_interest,last_trade,vendor_iv"
LABEL = {"cboe": "Cboe_delayed_quotes_(cdn.cboe.com,_15-min_delayed)", "yahoo": "Yahoo_Finance_via_yfinance"}


def et(s):
    """Cboe times are naive US Eastern strings; return ISO with the offset."""
    return dt.datetime.fromisoformat(s).replace(tzinfo=ET).isoformat() if s else ""


def parse_cboe(raw):
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
                    f"{g('last_trade_price')},{g('volume')},{g('open_interest')},{et(o.get('last_trade_time'))},{g('iv')}")
    spot = float(d.get("current_price") or d.get("close"))
    return spot, et(d.get("last_trade_time")), rows


def fetch_cboe(out):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request("https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json",
                                 headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60, context=ctx).read()
    (out / "cboe_raw.json").write_bytes(raw)
    return parse_cboe(raw)


def parse_yahoo(path):
    import csv
    rows, last = [], ""
    for q in csv.DictReader(open(path)):
        rows.append(f"{q['expiry']},{float(q['strike']):g},{q['type']},{q['bid']},{q['ask']},,,{q['lastPrice']},"
                    f"{q['volume']},{q['openInterest']},{q['lastTradeDate'].replace(' ', 'T')},{q['impliedVolatility']}")
        last = max(last, q["lastTradeDate"])
    return rows, last.replace(" ", "T")


def fetch_yahoo(out):
    import yfinance as yf
    import pandas as pd
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
    pd.concat(frames, ignore_index=True).to_csv(out / "yahoo_raw.csv", index=False)
    rows, last = parse_yahoo(out / "yahoo_raw.csv")
    return spot, last, rows


def write_chain(out, src, spot, qtime, rows, fetched):
    with open(out / "chain.csv", "w") as f:
        f.write(f"# spot={spot:.4f} asof={fetched[:10]} fetched={fetched} quote_time={qtime} "
                f"source={src} contracts={len(rows)} label={LABEL[src]}\n")
        f.write(COLS + "\n" + "\n".join(rows) + "\n")
    print(f"wrote {out}/chain.csv from {src}: {len(rows)} contracts, quotes as of {qtime}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rebuild"]:
        out = Path(sys.argv[2])
        head = (out / "chain.csv").read_text().split("\n", 1)[0]
        meta = dict(t.split("=", 1) for t in head[1:].split() if "=" in t)
        spot, qtime, rows = parse_cboe((out / "cboe_raw.json").read_bytes())
        write_chain(out, "cboe", spot, qtime, rows, meta["fetched"])
        sys.exit()

    now = dt.datetime.now(ET)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data" / f"spy_full_{now.date()}"
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for name, fn in (("cboe", fetch_cboe), ("yahoo", fetch_yahoo)):
        try:
            got[name] = fn(out)
            spot, qtime, rows = got[name]
            exps = sorted({r.split(",")[0] for r in rows})
            print(f"{name}: {len(rows)} contracts, {len(exps)} expiries ({exps[0]} to {exps[-1]}), "
                  f"spot {spot:.2f}, quotes as of {qtime}")
        except Exception as err:                       # keep going: one source is enough
            print(f"{name}: failed ({type(err).__name__}: {err})")
    if not got:
        sys.exit("no source worked; check the network and rerun")
    src = "cboe" if "cboe" in got else "yahoo"
    write_chain(out, src, *got[src], now.isoformat(timespec="seconds"))
    print("then:  git add data && git commit -m 'SPY full chain snapshot' && git push")
