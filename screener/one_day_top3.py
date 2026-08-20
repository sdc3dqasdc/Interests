#!/usr/bin/env python3
"""Pick the N (default 3) best stocks for a ONE-DAY hold — and backtest the rule.

Scores every eligible name in the universe with the 1-day mean-reversion score
(day_trade_signals.py), takes the top N for the most recent date as today's
picks, and — so the picks are not taken on faith — replays "each day, buy the
top N, sell at the next close" over history, NET OF SLIPPAGE, versus SPY.

READ THIS: a 1-day strategy lives or dies on slippage. Buying and selling every
single day means ~0.1% round-trip cost DAILY; over ~250 trading days that is a
~25% headwind before any edge. The backtest below subtracts it, so look at the
NET number and the comparison to SPY. If net barely beats or trails SPY, the
edge is not worth the risk and churn — that is the likely outcome, and the tool
is built to show it rather than hide it.

Usage:
    python3 one_day_top3.py --start 2023-01-01 --end 2026-07-21
    python3 one_day_top3.py --top 3 --tickers screener_universe.txt --max-tickers 800
"""

from __future__ import annotations

import argparse
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import day_trade_signals as sig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", type=Path, default=Path("screener_universe.txt"),
                   help="Universe file, one symbol per line (default: screener_universe.txt)")
    p.add_argument("--top", type=int, default=3, help="Stocks to pick per day (default: 3)")
    p.add_argument("--start", default=None, help="Backtest start (default: ~3y ago)")
    p.add_argument("--end", default=None, help="Backtest end (default: today)")
    p.add_argument("--max-tickers", type=int, default=800,
                   help="Cap universe size for a tractable fetch (default: 800)")
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-volume", type=float, default=10_000_000,
                   help="Liquidity floor — a 1-day strategy must exit fast (default: 10M)")
    p.add_argument("--slippage", type=float, default=0.05,
                   help="One-way slippage %% (round-trip = 2x, charged daily; default: 0.05)")
    p.add_argument("--output", type=Path, default=Path("one_day_picks.csv"))
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-secret", default=None)
    p.add_argument("--refresh-cache", action="store_true")
    return p.parse_args()


def build_panel(bars: dict[str, pd.DataFrame], args) -> pd.DataFrame:
    """Long table of (date, ticker, score, next_ret, eligible) for every bar."""
    frames = []
    for t, df in bars.items():
        if df is None or len(df) < 220:
            continue
        feats = sig.compute_features(df)
        score = sig.next_day_up_score(df)
        nret = sig.next_day_return(df)
        eligible = (
            (df["Close"] >= args.min_price)
            & (feats["dollar_vol"] >= args.min_dollar_volume)
            & feats["uptrend"].fillna(False)
            & score.notna()
        )
        frames.append(pd.DataFrame({
            "date": df.index, "ticker": t, "close": df["Close"].to_numpy(),
            "score": score.to_numpy(), "next_ret": nret.to_numpy(),
            "eligible": eligible.to_numpy(),
        }))
    if not frames:
        raise SystemExit("No ticker had enough history to score.")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz=timezone.utc).normalize().tz_localize(None)
    start = pd.Timestamp(args.start) if args.start else end - timedelta(days=3 * 365 + 300)

    tickers = bt.read_tickers(args.tickers)[:args.max_tickers]
    print(f"Scoring {len(tickers)} tickers for a {args.top}-name 1-day hold "
          f"({start.date()} → {end.date()}) …")
    client = bt._get_alpaca_client(args.api_key, args.api_secret)
    bars = bt.fetch_price_data_cached(tickers, client,
                                      start.to_pydatetime(), end.to_pydatetime(),
                                      refresh=args.refresh_cache)
    spy = bt.fetch_price_data_cached(["SPY"], client,
                                     start.to_pydatetime(), end.to_pydatetime(),
                                     refresh=args.refresh_cache).get("SPY")

    panel = build_panel(bars, args)
    rt_cost = 2 * args.slippage / 100.0  # daily round-trip slippage as a fraction

    # --- today's live picks: latest date's top-N eligible by score -----------
    latest = panel["date"].max()
    live = (panel[(panel["date"] == latest) & panel["eligible"]]
            .sort_values("score", ascending=False).head(args.top))

    # --- backtest: each day take top-N eligible, hold to next close ----------
    hist = panel[panel["eligible"] & panel["next_ret"].notna()].copy()
    daily = []
    for date, grp in hist.groupby("date"):
        picks = grp.sort_values("score", ascending=False).head(args.top)
        if picks.empty:
            continue
        gross = picks["next_ret"].mean()
        daily.append({
            "date": date, "n": len(picks),
            "gross_ret": gross, "net_ret": gross - rt_cost,
            "win": (picks["next_ret"] > 0).mean(),
        })
    bt_df = pd.DataFrame(daily)

    print(f"\nTODAY'S TOP {args.top} FOR A 1-DAY HOLD  (as of {latest.date()})")
    print("=" * 60)
    if live.empty:
        print("  no eligible names today (all failed the uptrend/liquidity gate).")
    else:
        for _, r in live.iterrows():
            print(f"  {r['ticker']:<6} score {r['score']:+.2f}   ${r['close']:.2f}")

    if bt_df.empty:
        print("\nNo backtest days available — cannot validate. Treat picks as untested.")
        live.to_csv(args.output, index=False)
        return 0

    # Compound net daily returns; compare to SPY over the same dates.
    net_cum = float(np.prod(1 + bt_df["net_ret"]) - 1) * 100
    gross_cum = float(np.prod(1 + bt_df["gross_ret"]) - 1) * 100
    win_rate = bt_df["win"].mean() * 100
    avg_net = bt_df["net_ret"].mean() * 100
    spy_cum = None
    if spy is not None:
        spy_ret = spy["Close"].pct_change().reindex(bt_df["date"]).dropna()
        spy_cum = float(np.prod(1 + spy_ret) - 1) * 100

    print(f"\n--- BACKTEST: top-{args.top}, 1-day hold, {len(bt_df)} trading days ---")
    print(f"  daily win rate (picks up) : {win_rate:.1f}%")
    print(f"  avg net daily return      : {avg_net:+.3f}%   (after {rt_cost*100:.2f}% round-trip)")
    print(f"  cumulative GROSS          : {gross_cum:+.1f}%")
    print(f"  cumulative NET (slippage) : {net_cum:+.1f}%")
    if spy_cum is not None:
        verdict = "BEATS" if net_cum > spy_cum else "TRAILS"
        print(f"  SPY buy & hold (same days): {spy_cum:+.1f}%   -> net {verdict} SPY "
              f"by {net_cum - spy_cum:+.1f}pp")
    print("\n  NOTE: slippage is an estimate; real 1-day fills, borrow, and the "
          "overnight gap add risk this backtest does not fully capture.")

    live.assign(as_of=latest.date()).to_csv(args.output, index=False)
    print(f"\nToday's picks written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
