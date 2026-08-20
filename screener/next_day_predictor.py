#!/usr/bin/env python3
"""Predict whether given stocks rise the NEXT session, with buy/sell levels.

Give it one or more tickers. For each it computes the 1-day mean-reversion score
(see day_trade_signals.py), turns it into a lean, and proposes a trade plan:
  - a suggested BUY-in price (a limit at/below the last close, for the dip-buy),
  - a suggested SELL price set to make at least --target-pct (default 2%).

Because a +2% next-day move is ambitious (the average next-day move is a few
tenths of a percent), it ALSO reports, from history, how often that target was
actually reachable — the fraction of similar setups whose NEXT-day high cleared
+2%. If that odds number is low (it usually is), the 2% target will rarely fill
and you would exit most trades some other way; the tool shows this rather than
implying the target is easy.

Usage:
    python3 next_day_predictor.py AAPL MSFT NVDA
    python3 next_day_predictor.py AAPL --target-pct 2 --start 2023-01-01
"""

from __future__ import annotations

import argparse
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import day_trade_signals as sig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+", help="One or more stock symbols")
    p.add_argument("--target-pct", type=float, default=2.0,
                   help="Minimum %% profit the SELL price must lock in (default: 2.0)")
    p.add_argument("--start", default=None,
                   help="History start for the hit-rate backtest (default: ~3y ago)")
    p.add_argument("--end", default=None, help="History end (default: today)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-secret", default=None)
    p.add_argument("--refresh-cache", action="store_true")
    return p.parse_args()


def lean(score: float, uptrend: bool) -> str:
    if not uptrend:
        return "AVOID (not in uptrend — no mean-reversion bet)"
    if score >= 1.0:
        return "LEAN UP"
    if score <= -1.0:
        return "LEAN DOWN"
    return "NEUTRAL"


def main() -> int:
    args = parse_args()
    tickers = [t.upper() for t in args.tickers]
    target = args.target_pct / 100.0
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz=timezone.utc).normalize().tz_localize(None)
    start = pd.Timestamp(args.start) if args.start else end - timedelta(days=3 * 365 + 300)

    client = bt._get_alpaca_client(args.api_key, args.api_secret)
    bars = bt.fetch_price_data_cached(tickers, client,
                                      start.to_pydatetime(), end.to_pydatetime(),
                                      refresh=args.refresh_cache)

    # Pool every ticker's realised (score, next-day close return, next-day high
    # gain) triples for the shared hit-rate / target-odds tables.
    pooled_score, pooled_ret, pooled_gain = [], [], []
    per_ticker = {}
    for t in tickers:
        df = bars.get(t)
        if df is None or len(df) < 220:
            per_ticker[t] = None
            continue
        score = sig.next_day_up_score(df)
        nret = sig.next_day_return(df)
        gain = sig.next_day_high_gain(df)
        feats = sig.compute_features(df)
        last_close = float(df["Close"].iloc[-1])
        per_ticker[t] = {
            "date": df.index[-1],
            "score": float(score.iloc[-1]),
            "uptrend": bool(feats["uptrend"].iloc[-1]),
            "rsi2": float(feats["rsi2"].iloc[-1]),
            "close": last_close,
            "atr_pct": float(sig.compute_atr_pct(df).iloc[-1]),
        }
        pooled_score.append(score.iloc[:-1])   # drop last bar (no realised next day)
        pooled_ret.append(nret.iloc[:-1])
        pooled_gain.append(gain.iloc[:-1])

    # Build pooled tables: up-rate by score bucket, and +target hit-rate by bucket.
    up_table = None
    hit_by_bucket = base_hit = edges = None
    if pooled_score:
        alls = pd.concat(pooled_score)
        allr = pd.concat(pooled_ret)
        allg = pd.concat(pooled_gain)
        up_table = sig.hit_rate_by_bucket(alls, allr, n_buckets=5)
        pooled = pd.DataFrame({"score": alls, "gain": allg}).dropna()
        if len(pooled) >= 100:
            pooled["hit"] = pooled["gain"] >= target
            base_hit = pooled["hit"].mean() * 100
            try:
                pooled["bucket"], edges = pd.qcut(pooled["score"], 5, labels=False,
                                                  retbins=True, duplicates="drop")
                hit_by_bucket = pooled.groupby("bucket")["hit"].mean() * 100
            except ValueError:
                hit_by_bucket = None

    def target_odds(score: float) -> float | None:
        """Historical %% of setups in this score's bucket whose +target filled."""
        if hit_by_bucket is None or edges is None:
            return base_hit
        b = int(np.clip(np.digitize(score, edges[1:-1]), 0, len(hit_by_bucket) - 1))
        return float(hit_by_bucket.get(b, base_hit))

    print(f"\n1-DAY PLAN  (target +{args.target_pct:.1f}%, history {start.date()} → {end.date()})")
    print("=" * 70)

    for t in tickers:
        info = per_ticker[t]
        if info is None:
            print(f"\n{t}: not enough history (<220 bars) — cannot score.")
            continue
        verdict = lean(info["score"], info["uptrend"])
        buy = info["close"]
        sell = round(buy * (1 + target), 2)
        odds = target_odds(info["score"])
        long_favoured = verdict in ("LEAN UP", "NEUTRAL")
        print(f"\n{t}  (as of {info['date'].date()})")
        print(f"  next-day score : {info['score']:+.2f}   RSI(2) {info['rsi2']:.0f}   "
              f"{'uptrend' if info['uptrend'] else 'NOT uptrend'}   ATR {info['atr_pct']:.1f}%/day")
        print(f"  lean           : {verdict}")
        plan_tag = "suggested trade" if long_favoured else "levels (long NOT favoured)"
        print(f"  {plan_tag:<15}: BUY ≤ ${buy:,.2f}   SELL @ ${sell:,.2f}  (+{args.target_pct:.1f}%)")
        if not long_favoured:
            print("                   ^ setup argues against a long here — shown for reference only.")
        if odds is not None:
            realistic = "plausible" if odds >= 25 else "unlikely to fill most days"
            print(f"  +{args.target_pct:.0f}% odds      : next-day high cleared the target on "
                  f"{odds:.0f}% of similar setups ({realistic}; base rate {base_hit:.0f}%)")

    if up_table is not None:
        print("\n--- score vs realised next-day return (pooled) ---")
        print(up_table.to_string())
        spread = up_table.iloc[-1]["up_rate_pct"] - up_table.iloc[0]["up_rate_pct"]
        msg = ("essentially NO 1-day edge — do not trade on it" if spread < 3
               else f"a faint {spread:+.1f}pp edge; daily slippage likely eats it")
        print(f"\nVERDICT: top-vs-bottom up-rate spread {spread:+.1f}pp — {msg}.")
        if base_hit is not None and base_hit < 25:
            print(f"NOTE: a +{args.target_pct:.0f}% next-day target filled only ~{base_hit:.0f}% of "
                  "the time historically — set a stop/time-exit for the majority that miss.")
    else:
        print("\nNot enough pooled history to measure a hit rate — treat plan as untested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
