#!/usr/bin/env python3
"""Historical win-rate test of the FULL pipeline: screen -> rank -> top-N picks.

backtest.py simulates a portfolio that buys everything the screen passes.
This tool instead replays the model exactly the way it is used live:

    at each pick date:  apply the screen (backtest.passes_screen, same rules)
                        rank survivors by 6-month relative strength vs SPY
                        take the top N (default 5)
    for each pick:      enter at the next session's open (+slippage)
                        exit at the close 50 trading days later (-slippage)

and then reports pick-level statistics: win rate with a 95% confidence
interval, average/median ROI, excess vs SPY over each pick's exact window,
and breakdowns by year and by rank (does rank 1 beat rank 5?).

The screen rules and the ranking are IMPORTED from backtest.py and
select_top15.py, so this tester cannot drift out of sync with the model.

Same caveats as the backtest: fundamentals are today's values (survivorship +
lookahead bias), the earnings blackout and news gate are not simulated, and
with --pick-freq shorter than the hold, consecutive picks overlap in time so
samples are not fully independent.

Usage:
    python3 win_rate_tester.py --start 2021-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import backtest as bt
import run_report as rr
import select_top15 as sel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=Path, default=Path("screener_universe.txt"))
    parser.add_argument("--start", type=str, required=True, help="First pick date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Last session (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=5,
                        help="Picks taken per pick date, mirroring live use (default: 5)")
    parser.add_argument("--pick-freq", type=int, default=5,
                        help="Trading days between pick dates (default: 5; note picks "
                             "overlap when this is shorter than the hold)")
    parser.add_argument("--hold-days", type=int, default=50)  # 10-week hold (was 20 = 4wk)
    parser.add_argument("--slippage", type=float, default=0.05)
    parser.add_argument("--stop-loss-pct", type=float, default=15.0,
                        help="Hard stop below entry, %% (0 disables; default: 15). NOTE: the "
                             "ab_sweep liked a wider 25%% stop on per-pick avg ROI, but the "
                             "portfolio backtest showed it CUT excess return from +38%% to "
                             "+23%% (deeper losers drag compounding) — so 15 stands.")
    parser.add_argument("--trailing-stop-pct", type=float, default=0.0)
    parser.add_argument("--take-profit-pct", type=float, default=0.0)
    parser.add_argument("--warmup-days", type=int, default=400)
    parser.add_argument("--alpaca-key", type=str, default=None)
    parser.add_argument("--alpaca-secret", type=str, default=None)
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Ignore today's cached Alpaca bars / fundamentals and refetch "
                             "(cache normally makes a same-day rerun skip both network calls)")
    parser.add_argument("--output", type=Path, default=Path("win_rate_picks.csv"))
    parser.add_argument("--alpha191-weight", type=float, default=0.0,
                        help="Blend weight for the Guotai Junan Alpha-191 composite in the "
                             "pick ranking: 0 = pure relative strength (default), 1 = pure "
                             "Alpha-191. Use this to A/B whether the factors add value.")

    # Screen thresholds — identical names/defaults to backtest.py so
    # bt.passes_screen can be reused verbatim.
    parser.add_argument("--above-200d-sma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-golden-cross", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-rising-200d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-6m-return", type=float, default=5.0)
    parser.add_argument("--min-rs-6m", type=float, default=0.0)
    parser.add_argument("--momentum-window", type=int, choices=[6, 12], default=6,
                        help="Momentum horizon in months, 6 (default) or 12")
    parser.add_argument("--max-ext-sma50", type=float, default=15.0)  # widening to 25 hurt the backtest
    parser.add_argument("--min-ext-sma50", type=float, default=-3.0)  # widening to -8 hurt the backtest
    parser.add_argument("--max-5d-return", type=float, default=8.0)
    parser.add_argument("--min-rsi", type=float, default=40.0)
    parser.add_argument("--max-rsi", type=float, default=65.0)
    parser.add_argument("--min-atr-pct", type=float, default=1.0)
    parser.add_argument("--max-atr-pct", type=float, default=6.0)
    parser.add_argument("--min-price", type=float, default=10.0)
    parser.add_argument("--min-dollar-volume", type=float, default=10_000_000)
    parser.add_argument("--min-volume-ratio", type=float, default=0.0)
    parser.add_argument("--min-close-strength", type=float, default=0.0)
    # --- RSI(2) / MACD gates (off by default) — same names as backtest.py ----
    parser.add_argument("--max-rsi2", type=float, default=None)
    parser.add_argument("--min-rsi2", type=float, default=None)
    parser.add_argument("--min-macd-hist-pct", type=float, default=None)
    parser.add_argument("--require-macd-positive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--macd-weight", type=float, default=0.0,
                        help="Blend weight for the MACD-histogram percentile in the pick "
                             "ranking: 0 = pure relative strength (default), 1 = pure MACD "
                             "histogram. Use to A/B whether MACD adds rank value.")
    parser.add_argument("--market-regime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-market-cap", type=float, default=2_000_000_000)
    parser.add_argument("--require-positive-fcf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-roe", type=float, default=0.10)
    parser.add_argument("--max-workers", type=int, default=10)
    return parser.parse_args()


def load_market_data(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Fetch bars, compute signals and fetch fundamentals — the expensive part.

    Split out from main() so a multi-variant sweep (ab_sweep.py) can pay this
    cost ONCE and then replay run_picks() against many parameter sets, instead
    of refetching and recomputing per variant."""
    if not args.tickers.exists():
        raise SystemExit(f"Ticker file '{args.tickers}' not found — run the screener first.")
    tickers = bt.read_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No tickers found in file.")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    fetch_start = start - timedelta(days=args.warmup_days)

    client = bt._get_alpaca_client(args.alpaca_key, args.alpaca_secret)
    hist = bt.fetch_price_data_cached(tickers, client, fetch_start, end, refresh=args.refresh_cache)
    if not hist:
        raise SystemExit("No price data returned from Alpaca.")
    spy = bt.fetch_price_data_cached(["SPY"], client, fetch_start, end, refresh=args.refresh_cache).get("SPY")
    if spy is None or spy.empty:
        raise SystemExit("No SPY data — needed for the calendar, regime gate and benchmark.")

    print("Computing technical signals …")
    too_short = 0
    for t in list(hist.keys()):
        if len(hist[t]) < 200:
            del hist[t]
            too_short += 1
            continue
        hist[t] = bt.compute_signals(hist[t])
    if too_short:
        print(f"  -> skipped {too_short} tickers with <200 bars (cannot pass the trend rules)")
    if not hist:
        raise SystemExit("No tickers with enough history to screen.")

    info_map = bt.fetch_info_map(list(hist.keys()), max_workers=args.max_workers,
                                 max_age_days=0 if args.refresh_cache else 1.0)
    if not info_map:
        print("WARNING: fundamentals unavailable — screening on technicals only.", file=sys.stderr)
        info_map = {t: {} for t in hist}
    return hist, spy, info_map


def run_picks(hist: dict[str, pd.DataFrame], spy: pd.DataFrame, info_map: dict,
              args: argparse.Namespace, verbose: bool = True) -> list[dict[str, Any]]:
    """Replay screen -> rank -> top-N over the window, returning scored picks.

    Pure function of (data, args): no fetching, so it is cheap to call many
    times with different parameters."""
    start_dt = pd.Timestamp(args.start)
    spy_sma200 = spy["Close"].rolling(200).mean()
    # Risk-off = below the 200-day OR the 200-day is falling (bear-rally trap).
    spy_risk_off = (spy["Close"] <= spy_sma200) | (spy_sma200 <= spy_sma200.shift(20))
    risk_off_dates = set(spy_risk_off.index[spy_risk_off.fillna(False)]) if args.market_regime else set()
    # Benchmark momentum must use the SAME horizon as the screen rule, or the
    # relative-strength comparison comes from two different windows.
    mom_lag = 252 if args.momentum_window == 12 else 126
    mom_col = "12m_return" if args.momentum_window == 12 else "6m_return"
    spy_6m_series = (spy["Close"].shift(21) / spy["Close"].shift(mom_lag) - 1) * 100
    spy_6m_by_date = {d: v for d, v in spy_6m_series.items() if pd.notna(v)}
    all_dates = [d for d in sorted(spy.index) if d >= start_dt]
    if len(all_dates) < args.hold_days + 2:
        raise SystemExit("Window too short for even one completed hold.")

    pick_dates = all_dates[::args.pick_freq]
    skipped_risk_off = sum(1 for d in pick_dates if d in risk_off_dates)
    if verbose:
        print(f"{len(pick_dates)} pick dates every {args.pick_freq} sessions "
              f"({skipped_risk_off} skipped as risk-off) …")

    picks: list[dict[str, Any]] = []
    for date in pick_dates:
        if date in risk_off_dates:
            continue
        spy_6m = spy_6m_by_date.get(date)
        survivors = []
        for ticker, df in hist.items():
            if ticker not in info_map or date not in df.index:
                continue
            row = df.loc[date]
            if bt.passes_screen(row, info_map[ticker], args, spy_6m):
                # Rank by relative strength vs SPY — the documented cross-
                # sectional momentum signal.  (The old multi-factor composite
                # showed no rank separation in testing.)
                ret6m = bt.finite_number(row.get(mom_col)) or 0.0
                survivors.append((ret6m - (spy_6m or 0.0), ticker, row))

        if args.macd_weight > 0 and survivors:
            # Blend percentile ranks of relative strength and the MACD
            # histogram (%% of price — the cross-sectionally comparable form).
            w = min(args.macd_weight, 1.0)
            frame = pd.DataFrame([
                {"rs": rs, "macd": bt.finite_number(row.get("macd_hist_pct"))}
                for rs, _, row in survivors])
            if frame["macd"].notna().any():
                blended = ((1 - w) * frame["rs"].rank(pct=True)
                           + w * frame["macd"].rank(pct=True))
                survivors = [(float(blended.iloc[i]), survivors[i][1], survivors[i][2])
                             for i in range(len(survivors))]

        if args.alpha191_weight > 0 and survivors:
            # Blend percentile ranks of relative strength and the Alpha-191
            # composite, cross-sectionally across this date's survivors.
            frame = pd.DataFrame(
                [{"rs": rs, **{c: bt.finite_number(row.get(c)) for c in sel.ALPHA191_COLS}}
                 for rs, _, row in survivors])
            alpha_pct = sel.alpha191_percentile(frame)
            if alpha_pct is not None:
                w = min(args.alpha191_weight, 1.0)
                blended = (1 - w) * frame["rs"].rank(pct=True) + w * alpha_pct
                survivors = [(float(blended.iloc[i]), survivors[i][1], survivors[i][2])
                             for i in range(len(survivors))]
        scored = survivors
        scored.sort(key=lambda x: -x[0])

        for rank, (score, ticker, row) in enumerate(scored[:args.top], start=1):
            df = hist[ticker]
            entry_idx = df.index.get_loc(date) + 1
            exit_idx = entry_idx + args.hold_days
            if exit_idx >= len(df.index):
                continue  # hold would run past the data — incomplete, don't score
            entry_bar = df.iloc[entry_idx]
            entry_fill = bt.finite_number(entry_bar.get("Open")) or bt.finite_number(entry_bar.get("Close"))
            if not entry_fill or entry_fill <= 0:
                continue
            entry_price = entry_fill * (1 + args.slippage / 100)
            # Risk exits (stop / trailing / target) can end the hold early.
            exit_idx, exit_price, exit_reason = bt.resolve_exit(
                df, entry_idx, exit_idx, entry_price,
                args.stop_loss_pct, args.take_profit_pct, args.trailing_stop_pct, args.slippage)
            roi = (exit_price / entry_price - 1) * 100

            entry_ts, exit_ts = df.index[entry_idx], df.index[exit_idx]
            spy_roi = None
            spy_pos = spy.index.searchsorted(entry_ts)
            spy_exit_pos = min(spy.index.searchsorted(exit_ts), len(spy.index) - 1)
            if spy_pos < len(spy.index):
                spy_entry = bt.finite_number(spy.iloc[spy_pos].get("Open")) or \
                    bt.finite_number(spy.iloc[spy_pos].get("Close"))
                if spy_entry and spy_entry > 0:
                    spy_roi = (float(spy.iloc[spy_exit_pos]["Close"]) / spy_entry - 1) * 100

            picks.append({
                "pick_date": date.date().isoformat(), "rank": rank, "ticker": ticker,
                "rank_score": round(score, 4),
                "entry_date": entry_ts.date().isoformat(), "exit_date": exit_ts.date().isoformat(),
                "exit_reason": exit_reason,
                "roi_pct": round(roi, 2),
                "spy_roi_pct": round(spy_roi, 2) if spy_roi is not None else None,
                "excess_pct": round(roi - spy_roi, 2) if spy_roi is not None else None,
                "win": roi > 0,
                # Entry-time diagnostics: let downstream analysis recover the
                # in-band shape of each indicator vs realised return without
                # re-running the whole screen.
                "rsi_14": bt.finite_number(row.get("rsi_14")),
                "rsi_2": bt.finite_number(row.get("rsi_2")),
                "macd_hist_pct": bt.finite_number(row.get("macd_hist_pct")),
                "ext_vs_sma50": bt.finite_number(row.get("ext_vs_sma50")),
                "atr_pct": bt.finite_number(row.get("atr_pct")),
            })
    return picks


def main() -> int:
    args = parse_args()
    rr.begin("win_rate_tester", args)
    hist, spy, info_map = load_market_data(args)
    args.fundamentals_available = bool(info_map) and any(info_map.values())

    picks = run_picks(hist, spy, info_map, args)
    if not picks:
        raise SystemExit("No completed picks in the window — widen the dates or loosen the screen.")

    out = pd.DataFrame(picks)
    out.to_csv(args.output, index=False)

    n = len(out)
    wins = int(out["win"].sum())
    p = wins / n
    se = math.sqrt(p * (1 - p) / n)
    lo, hi = max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║              PIPELINE WIN-RATE TEST (top {args.top} picks)          ║
╚══════════════════════════════════════════════════════════╝
  Picks scored      : {n}   (hold {args.hold_days} trading days)
  Win rate          : {p * 100:.1f}%   [95% CI {lo * 100:.1f}% – {hi * 100:.1f}%]{'  << CI includes 50%' if lo <= 0.5 else ''}
  Avg ROI / pick    : {out['roi_pct'].mean():+.2f}%
  Median ROI        : {out['roi_pct'].median():+.2f}%
  Best / worst      : {out['roi_pct'].max():+.2f}% / {out['roi_pct'].min():+.2f}%""")
    excess = out["excess_pct"].dropna()
    if len(excess):
        beat = (excess > 0).mean() * 100
        print(f"  Avg excess vs SPY : {excess.mean():+.2f}%   (beat SPY on {beat:.0f}% of picks)")

    rr.metrics("win_rate_tester", "headline", {
        "picks_scored": n, "hold_days": args.hold_days, "top_n": args.top,
        "win_rate_pct": round(p * 100, 2),
        "win_rate_ci_low_pct": round(lo * 100, 2), "win_rate_ci_high_pct": round(hi * 100, 2),
        "ci_includes_50pct": bool(lo <= 0.5),
        "avg_roi_pct": round(out["roi_pct"].mean(), 4),
        "median_roi_pct": round(out["roi_pct"].median(), 4),
        "best_roi_pct": round(out["roi_pct"].max(), 4),
        "worst_roi_pct": round(out["roi_pct"].min(), 4),
        "avg_excess_vs_spy_pct": round(excess.mean(), 4) if len(excess) else None,
        "beat_spy_pct_of_picks": round((excess > 0).mean() * 100, 2) if len(excess) else None,
    })
    rr.rows("win_rate_tester", "by_year", [
        {"year": y, "picks": len(g), "win_pct": round(g["win"].mean() * 100, 2),
         "avg_roi_pct": round(g["roi_pct"].mean(), 3)}
        for y, g in out.groupby(out["pick_date"].str[:4])])
    if "exit_reason" in out.columns:
        rr.rows("win_rate_tester", "by_exit_reason", [
            {"exit_reason": r, "picks": len(g), "win_pct": round(g["win"].mean() * 100, 2),
             "avg_roi_pct": round(g["roi_pct"].mean(), 3),
             "worst_roi_pct": round(g["roi_pct"].min(), 3)}
            for r, g in out.groupby("exit_reason")])
    rr.rows("win_rate_tester", "by_rank", [
        {"rank": r, "picks": len(g), "win_pct": round(g["win"].mean() * 100, 2),
         "avg_roi_pct": round(g["roi_pct"].mean(), 3)}
        for r, g in out.groupby("rank")])
    if args.pick_freq < args.hold_days:
        rr.note("win_rate_tester", "caveats",
                f"Pick dates every {args.pick_freq} sessions with a {args.hold_days}-session "
                f"hold: windows overlap, picks are correlated, so the CI is optimistic.")

    print("\n  --- by year ---")
    year = out["pick_date"].str[:4]
    for y, grp in out.groupby(year):
        print(f"  {y}: {len(grp):>4} picks   win {grp['win'].mean() * 100:5.1f}%   "
              f"avg {grp['roi_pct'].mean():+6.2f}%")

    if "exit_reason" in out.columns:
        print("\n  --- by exit reason (is the stop doing its job?) ---")
        for reason, grp in out.groupby("exit_reason"):
            print(f"  {reason:<7}: {len(grp):>4} picks   win {grp['win'].mean() * 100:5.1f}%   "
                  f"avg {grp['roi_pct'].mean():+6.2f}%   worst {grp['roi_pct'].min():+6.2f}%")

    print("\n  --- by rank (is the ranking adding value?) ---")
    for r, grp in out.groupby("rank"):
        print(f"  rank {r}: {len(grp):>4} picks   win {grp['win'].mean() * 100:5.1f}%   "
              f"avg {grp['roi_pct'].mean():+6.2f}%")

    if args.pick_freq < args.hold_days:
        print(f"\n  NOTE: pick dates every {args.pick_freq} sessions with a {args.hold_days}-session "
              f"hold means overlapping windows — picks are correlated, so the CI is optimistic.")
    print(f"\nPer-pick detail written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
