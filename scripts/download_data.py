#!/usr/bin/env python
"""Download daily, dividend- and split-adjusted OHLCV for a multi-asset ETF universe.

Source: Yahoo Finance via the ``yfinance`` package (``auto_adjust=True``: open, high,
low and close are adjusted for splits and distributions, so close-to-close returns
are total returns). Output is the long format read by ``src.data.load_csv_market``.

Usage:
    python scripts/download_data.py --out data/etf_daily.csv --start 2006-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_TICKERS = ["SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "HYG", "GLD", "DBC", "VNQ"]


def download(tickers: Sequence[str], start: str, end: str | None) -> pd.DataFrame:
    """Download adjusted OHLCV and return a long-format frame.

    Args:
        tickers: Yahoo Finance symbols.
        start: First date (inclusive).
        end: Last date (exclusive); ``None`` = today.

    Returns:
        Frame with columns ``date, ticker, open, high, low, close, volume``.

    Raises:
        RuntimeError: If any ticker returns no data.
    """
    import yfinance as yf

    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, actions=False)
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {t}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date", "index": "date"})
        df["ticker"] = t
        frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.date
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> dict:
    """Basic data-quality report: coverage, gaps, non-positive prices, extreme moves."""
    report = {}
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date")
        r = g["close"].pct_change()
        report[t] = {
            "first": str(g["date"].iloc[0]),
            "last": str(g["date"].iloc[-1]),
            "rows": int(len(g)),
            "non_positive_close": int((g["close"] <= 0).sum()),
            "missing_close": int(g["close"].isna().sum()),
            "zero_volume_days": int((g["volume"] <= 0).sum()),
            "max_abs_daily_return": float(r.abs().max()),
            "ann_return_pct": float(100 * ((g["close"].iloc[-1] / g["close"].iloc[0]) ** (252 / len(g)) - 1)),
            "ann_vol_pct": float(100 * r.std() * 252**0.5),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="data/etf_daily.csv")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = download(args.tickers, args.start, args.end)
    df.to_csv(out, index=False)
    report = validate(df)
    meta = {
        "source": "Yahoo Finance via yfinance (auto_adjust=True: split- and distribution-adjusted OHLC)",
        "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "tickers": list(args.tickers),
        "start": args.start,
        "end": args.end or "latest",
        "rows": int(len(df)),
        "quality": report,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(report).T.to_string())
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.2f} MB) and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
