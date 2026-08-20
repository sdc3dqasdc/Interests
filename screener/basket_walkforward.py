#!/usr/bin/env python3
"""Run the self-optimizing walk-forward across a BASKET, to test generalization.

A stable hold/target that earns out-of-sample across MANY names is real; one that
only works on AAOI is a story fit to one history. For each ticker this runs the
same rolling walk-forward as aaoi_next_day.py (long and short), reports the
recommended side's pooled out-of-sample result and verdict, then summarizes:
  - how often the chosen (hold, target) clusters on one setup, and
  - how many names actually show a ROBUST out-of-sample edge.

Usage:
    python3 basket_walkforward.py
    python3 basket_walkforward.py AAOI NVDA AMD SMCI --start 2020-01-01
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta, timezone

import pandas as pd

import backtest as bt
import day_trade_signals as sig
import aaoi_next_day as aa

DEFAULT_BASKET = ["AAOI", "COHR", "LITE", "AAPL", "NVDA", "AMD", "MU", "MRVL",
                  "AVGO", "SMCI", "ANET", "TSLA", "META", "AMZN", "NFLX",
                  "CRWD", "PLTR", "INTC", "MSFT", "SOFI"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", default=None,
                   help=f"Symbols (default: a {len(DEFAULT_BASKET)}-name tech/semis basket)")
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--live-lookback", type=int, default=500)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-secret", default=None)
    p.add_argument("--refresh-cache", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    tickers = [t.upper() for t in (a.tickers or DEFAULT_BASKET)]
    slip_rt = 2 * a.slippage / 100.0
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp.now(tz=timezone.utc).normalize().tz_localize(None)
    start = pd.Timestamp(a.start) if a.start else end - timedelta(days=5 * 365 + 300)

    client = bt._get_alpaca_client(a.api_key, a.api_secret)
    bars = bt.fetch_price_data_cached(tickers, client, start.to_pydatetime(),
                                      end.to_pydatetime(), refresh=a.refresh_cache)

    print(f"\nBASKET WALK-FORWARD  ({len(tickers)} names, {start.date()} → {end.date()})")
    print("=" * 78)
    print(f"{'ticker':<7}{'side':<6}{'live H/tgt':<12}{'OOS%/tr':>8}{'win':>6}"
          f"{'folds+':>8}   verdict")
    print("-" * 78)

    rows, param_votes = [], Counter()
    for t in tickers:
        df = bars.get(t)
        if df is None or len(df) < 400:
            print(f"{t:<7}(insufficient history)")
            continue
        atr_pct = float(sig.compute_atr_pct(df).iloc[-1])
        stop = atr_pct / 100.0
        score_s = sig.next_day_up_score(df)
        cur = float(score_s.iloc[-1])
        res = [r for r in (aa.walk_forward(df, score_s, False, stop, slip_rt),
                           aa.walk_forward(df, score_s, True, stop, slip_rt)) if r]
        if not res:
            print(f"{t:<7}(too few signals)")
            continue
        want_short = cur <= -aa.SIGNAL_THR
        rec = next((r for r in res if r["short"] == want_short), None) or \
            max(res, key=lambda r: r["oos_avg"])
        verdict, pos = aa.classify(rec)
        H, tgt, _ = aa.pick_live_params(df, score_s, rec["short"], stop, slip_rt, a.live_lookback)
        side = "SHORT" if rec["short"] else "LONG"
        param_votes[(H, tgt)] += 1
        rows.append({"t": t, "verdict": verdict, "oos": rec["oos_avg"]})
        folds_str = f"{pos}/{len(rec['folds'])}"
        print(f"{t:<7}{side:<6}{f'{H}d/+{int(tgt*100)}%':<12}{rec['oos_avg']:>8.2f}"
              f"{rec['oos_win']:>5.0f}%{folds_str:>8}   {verdict}")

    # --- summary: does it generalize? ---
    robust = [r for r in rows if r["verdict"] == "ROBUST"]
    lumpy = [r for r in rows if r["verdict"].startswith("POSITIVE")]
    dead = [r for r in rows if r["verdict"] == "NO OOS EDGE"]
    print("-" * 78)
    print(f"SUMMARY: {len(robust)} ROBUST, {len(lumpy)} lumpy-positive, {len(dead)} no-edge "
          f"(of {len(rows)} scored)")
    if param_votes:
        common = param_votes.most_common(3)
        print("  param clustering (live pick): " +
              ", ".join(f"{h}d/+{int(t*100)}% ×{c}" for (h, t), c in common))
    robust_pct = 100 * len(robust) / len(rows) if rows else 0
    if robust_pct >= 40:
        print(f"  -> {robust_pct:.0f}% robust: the setup GENERALIZES beyond one name.")
    else:
        print(f"  -> only {robust_pct:.0f}% robust: the edge is mostly name-specific / lumpy. "
              "Treat single-ticker 'wins' as regime luck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
