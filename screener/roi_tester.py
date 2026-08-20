#!/usr/bin/env python3
"""Track the real ROI of the screener's picks since the day they were screened.

Reads the ranked picks file (top_candidates.csv from select_top15.py by
default), takes the top N rows (default 5), and measures how each has actually
performed using Alpaca daily bars, with the same trade mechanics the backtest
assumes:

- entry at the OPEN of the first session after the screen date (+slippage),
- exit at the CLOSE of the session --hold-days trading days later (-slippage)
  if that day has arrived, otherwise marked OPEN and valued at the latest close,
- each pick compared against SPY over its own exact window.

Run it any time after a screen; positions inside the hold window show as OPEN
with day k/N.  Writes roi_report.csv and prints a summary.

Usage:
    python3 roi_tester.py                    # top 5 of top_candidates.csv
    python3 roi_tester.py --top 10 --picks short_term_candidates.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import backtest as bt
import run_report as rr


def finite_number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _get_alpaca_client(api_key: str | None, secret_key: str | None):
    from alpaca.data.historical.stock import StockHistoricalDataClient
    _load_dotenv()
    key = api_key or os.environ.get("ALPACA_API_KEY")
    secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Alpaca API key and secret required. Set ALPACA_API_KEY / ALPACA_SECRET_KEY "
            "(env vars or a .env file), or pass --alpaca-key / --alpaca-secret."
        )
    return StockHistoricalDataClient(key, secret)


def fetch_daily_bars(client, symbols: list[str], start: datetime) -> dict[str, pd.DataFrame]:
    """Daily bars per symbol, tz-naive normalized index; splits batches on an
    invalid symbol so one bad ticker cannot sink the rest."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    out: dict[str, pd.DataFrame] = {}

    def salvage(syms: list[str]) -> None:
        req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
                               start=start, feed=DataFeed.IEX, adjustment="all")
        try:
            df = client.get_stock_bars(req).df
            if df is None or df.empty:
                return
            for sym in syms:
                try:
                    sym_df = df.loc[sym].copy()
                except KeyError:
                    continue
                if sym_df.empty:
                    continue
                idx = pd.to_datetime(sym_df.index)
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                sym_df.index = idx.normalize()
                out[sym] = sym_df
        except Exception:
            if len(syms) == 1:
                return
            mid = len(syms) // 2
            time.sleep(0.1)
            salvage(syms[:mid])
            salvage(syms[mid:])

    salvage(symbols)
    return out


def fetch_daily_bars_cached(client, symbols: list[str], start: datetime,
                            refresh: bool = False) -> dict[str, pd.DataFrame]:
    """fetch_daily_bars, reused for the rest of today once fetched.

    Shares the .cache/ directory and day-scoped key/refresh mechanism with
    backtest.py, but uses its own key tag ("ohlcv-roi") since this function's
    bars keep Alpaca's raw lowercase columns and an open-ended (no --end)
    range, a different shape from backtest.fetch_price_data_cached's."""
    today = bt._today_str()
    key = bt._cache_key("bars", "ohlcv-roi", today, sorted(symbols), start.date())
    path = bt.CACHE_DIR / f"bars_{key}.pkl"
    if not refresh:
        cached = bt._cache_load(path, today)
        if cached is not None:
            print(f"  -> using today's cached Alpaca bars for {len(cached)} tickers "
                  f"({path.name}; pass --refresh-cache to force a refetch)")
            return cached
    data = fetch_daily_bars(client, symbols, start)
    bt._cache_save(path, today, data)
    return data


def evaluate_pick(ticker: str, screen_date: pd.Timestamp, bars: pd.DataFrame | None,
                  spy: pd.DataFrame | None, hold_days: int, slippage: float,
                  stop_pct: float = 0.0) -> dict[str, Any]:
    """ROI for one pick using next-session-open entry / close exit mechanics."""
    row: dict[str, Any] = {"ticker": ticker, "screen_date": screen_date.date().isoformat(),
                           "status": "NO_DATA", "entry_date": None, "entry_price": None,
                           "exit_date": None, "exit_price": None, "roi_pct": None,
                           "spy_roi_pct": None, "excess_pct": None, "days_held": 0}
    if bars is None or bars.empty:
        return row

    future = bars.index[bars.index > screen_date]
    if len(future) == 0:
        row["status"] = "PENDING (no session after screen date yet)"
        return row

    entry_ts = future[0]
    entry_bar = bars.loc[entry_ts]
    entry_fill = finite_number(entry_bar.get("open")) or finite_number(entry_bar.get("close"))
    if not entry_fill or entry_fill <= 0:
        return row
    entry_price = entry_fill * (1 + slippage / 100)

    entry_idx = bars.index.get_loc(entry_ts)

    # Did a stop trigger inside the window? Checked before the time exit.
    stop_idx = None
    if stop_pct > 0:
        stop_price = entry_price * (1 - stop_pct / 100)
        last = min(entry_idx + hold_days, len(bars.index) - 1)
        for i in range(entry_idx + 1, last + 1):
            if float(bars.iloc[i]["low"]) <= stop_price:
                stop_idx = i
                break

    if stop_idx is not None:
        exit_ts = bars.index[stop_idx]
        fill = min(float(bars.iloc[stop_idx]["open"]), entry_price * (1 - stop_pct / 100))
        exit_price = fill * (1 - slippage / 100)
        sessions_held = stop_idx - entry_idx
        status = f"STOPPED (day {sessions_held})"
    else:
        exit_idx = entry_idx + hold_days
        if exit_idx < len(bars.index):
            exit_ts = bars.index[exit_idx]
            status = "CLOSED"
            sessions_held = hold_days
        else:
            exit_ts = bars.index[-1]
            sessions_held = len(bars.index) - 1 - entry_idx
            status = f"OPEN (day {sessions_held}/{hold_days})"
        exit_price = float(bars.loc[exit_ts, "close"]) * (1 - slippage / 100)
    roi = (exit_price / entry_price - 1) * 100

    spy_roi = None
    if spy is not None and entry_ts in spy.index:
        spy_entry = finite_number(spy.loc[entry_ts].get("open")) or finite_number(spy.loc[entry_ts].get("close"))
        spy_exit_ts = exit_ts if exit_ts in spy.index else spy.index[spy.index <= exit_ts][-1]
        if spy_entry and spy_entry > 0:
            spy_roi = (float(spy.loc[spy_exit_ts, "close"]) / spy_entry - 1) * 100

    row.update(status=status, entry_date=entry_ts.date().isoformat(), entry_price=round(entry_price, 4),
               exit_date=exit_ts.date().isoformat(), exit_price=round(exit_price, 4),
               roi_pct=round(roi, 2), days_held=sessions_held,
               spy_roi_pct=round(spy_roi, 2) if spy_roi is not None else None,
               excess_pct=round(roi - spy_roi, 2) if spy_roi is not None else None)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks", type=Path, default=Path("top_candidates.csv"),
                        help="Ranked picks CSV (default: top_candidates.csv from select_top15.py)")
    parser.add_argument("--top", type=int, default=5,
                        help="How many picks from the top of the file to test (default: 5)")
    parser.add_argument("--hold-days", type=int, default=50,
                        help="Trading-day hold used for the CLOSED exit (default: 50, ~10 weeks)")
    parser.add_argument("--slippage", type=float, default=0.05,
                        help="One-way slippage %% applied to entry and exit (default: 0.05)")
    parser.add_argument("--stop-loss-pct", type=float, default=15.0,
                        help="Flag/exit a pick whose low breached this %% below entry "
                             "(0 disables; default: 15 — 25 helped the sweep but hurt the "
                             "portfolio backtest)")
    parser.add_argument("--alpaca-key", type=str, default=None)
    parser.add_argument("--alpaca-secret", type=str, default=None)
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Ignore today's cached Alpaca bars and refetch (cache normally "
                             "makes a same-day rerun skip the network call)")
    parser.add_argument("--output", type=Path, default=Path("roi_report.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rr.begin("roi_tester", args)
    if args.top < 1:
        raise SystemExit("--top must be positive.")
    if not args.picks.exists():
        raise SystemExit(
            f"'{args.picks}' not found. Run the screener then select_top15.py first, "
            f"or pass --picks <file>."
        )
    df = pd.read_csv(args.picks)
    if df.empty:
        raise SystemExit(f"'{args.picks}' has no rows to test.")
    if "screened_at_utc" not in df.columns or "ticker" not in df.columns:
        raise SystemExit(f"'{args.picks}' is missing ticker/screened_at_utc columns.")

    picks = df.head(args.top).copy()
    picks["screen_date"] = (pd.to_datetime(picks["screened_at_utc"], errors="coerce", utc=True)
                            .dt.tz_localize(None).dt.normalize())
    if picks["screen_date"].isna().any():
        raise SystemExit("Could not parse screened_at_utc for one or more picks.")

    earliest = picks["screen_date"].min()
    client = _get_alpaca_client(args.alpaca_key, args.alpaca_secret)
    symbols = sorted(set(picks["ticker"])) + ["SPY"]
    print(f"Testing ROI of {len(picks)} picks screened {earliest.date()} "
          f"(hold {args.hold_days} trading days) …")
    bars = fetch_daily_bars_cached(client, symbols,
                                   (earliest - timedelta(days=5)).to_pydatetime().replace(tzinfo=timezone.utc),
                                   refresh=args.refresh_cache)
    spy = bars.get("SPY")
    if spy is None:
        print("WARNING: no SPY data; benchmark columns will be empty", file=sys.stderr)

    results = [evaluate_pick(r.ticker, r.screen_date, bars.get(r.ticker), spy,
                             args.hold_days, args.slippage, args.stop_loss_pct)
               for r in picks.itertuples()]

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)

    header = (f"{'ticker':<7} {'status':<22} {'entry':>10} {'exit/last':>10} "
              f"{'ROI%':>7} {'SPY%':>7} {'excess':>7}")
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        roi = f"{r['roi_pct']:>7.2f}" if r["roi_pct"] is not None else "      -"
        spyp = f"{r['spy_roi_pct']:>7.2f}" if r["spy_roi_pct"] is not None else "      -"
        exc = f"{r['excess_pct']:>7.2f}" if r["excess_pct"] is not None else "      -"
        ep = f"{r['entry_price']:>10.2f}" if r["entry_price"] is not None else "         -"
        xp = f"{r['exit_price']:>10.2f}" if r["exit_price"] is not None else "         -"
        print(f"{r['ticker']:<7} {r['status']:<22} {ep} {xp} {roi} {spyp} {exc}")

    scored = [r for r in results if r["roi_pct"] is not None]
    if scored:
        rois = [r["roi_pct"] for r in scored]
        wins = sum(1 for x in rois if x > 0)
        print(f"\nPicks with data : {len(scored)}/{len(results)}")
        print(f"Average ROI     : {sum(rois) / len(rois):+.2f}%")
        print(f"Win rate        : {wins}/{len(scored)} ({wins / len(scored) * 100:.0f}%)")
        excesses = [r["excess_pct"] for r in scored if r["excess_pct"] is not None]
        if excesses:
            print(f"Avg excess vs SPY: {sum(excesses) / len(excesses):+.2f}%")
        closed = sum(1 for r in scored if r["status"] == "CLOSED")
        if closed < len(scored):
            print(f"NOTE: {len(scored) - closed} position(s) still inside the hold window — "
                  f"their ROI is unrealised.")
    rr.rows("roi_tester", "picks", results)
    if scored:
        rois = [r["roi_pct"] for r in scored]
        excesses = [r["excess_pct"] for r in scored if r["excess_pct"] is not None]
        rr.metrics("roi_tester", "headline", {
            "picks_tested": len(results), "picks_with_data": len(scored),
            "hold_days": args.hold_days, "slippage_pct": args.slippage,
            "avg_roi_pct": round(sum(rois) / len(rois), 4),
            "win_rate_pct": round(sum(1 for x in rois if x > 0) / len(scored) * 100, 2),
            "avg_excess_vs_spy_pct": round(sum(excesses) / len(excesses), 4) if excesses else None,
            "closed_positions": sum(1 for r in scored if r["status"] == "CLOSED"),
            "still_open": sum(1 for r in scored if r["status"] != "CLOSED"),
        })
    else:
        rr.note("roi_tester", "headline",
                "No pick had usable data yet — likely screened too recently for a session to "
                "have elapsed since the screen date.")

    print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
