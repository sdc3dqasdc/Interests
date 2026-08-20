#!/usr/bin/env python3
"""Narrow the screener's BUY candidates to the N with the most 10-week potential.

Reads short_term_candidates.csv (written by short_term_screener_alpaca.py)
directly — no re-screening and no network calls — scores each candidate on
the metrics the screener already computed, and writes the ranked top N
(default 5) to top_candidates.csv with per-component scores so every pick
is explainable.

Scoring philosophy matches the screener's win-rate tuning: reward setups
whose components sit in their historical sweet spots (steady momentum near
the highs on confirmed volume) and penalise the edges of the allowed bands,
where mean-reversion risk concentrates.  Each component is normalised to
0-1, weighted, and summed to a 0-100 potential score.

Usage:
    python3 select_top15.py                       # top 5 from default CSV
    python3 select_top15.py --top 10
    python3 select_top15.py --input my_candidates.csv --output my_top.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

import run_report as rr

# Component weights (sum 100), tuned for a ~4-week hold and for hit rate rather
# than maximum upside: the durable 6-month momentum factor and entry quality
# dominate, while extension, high volatility, and event risk are penalised.
WEIGHTS = {
    "momentum_6m": 25.0,    # 6-month momentum — the horizon where it persists
    "entry_quality": 20.0,  # buying consolidation, not extension above the 50-day
    "trend": 15.0,          # healthy (not euphoric) margin above the 200-day SMA
    "rsi_zone": 10.0,       # higher RSI within the band: 45-50 measured worst
    "low_vol": 10.0,        # lower ATR carries the higher hit rate
    "quality": 10.0,        # stronger ROE = steadier holds
    "earnings_gap": 5.0,    # more days until earnings = less gap risk
    "liquidity": 5.0,       # deeper dollar volume = cleaner fills
}


ALPHA191_COLS = ["alpha002", "alpha111", "alpha054"]


def alpha191_percentile(df: pd.DataFrame) -> pd.Series | None:
    """Cross-sectional Alpha-191 composite: mean percentile rank of each factor
    across the candidate set, 0-1 with higher = better.  Returns None when the
    columns are absent (older CSVs) so ranking falls back cleanly."""
    present = [c for c in ALPHA191_COLS if c in df.columns and df[c].notna().any()]
    if not present:
        return None
    ranks = pd.DataFrame({c: df[c].rank(pct=True) for c in present})
    return ranks.mean(axis=1)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _triangle(value: float, low: float, peak: float, high: float) -> float:
    """1.0 at peak, falling linearly to 0.0 at low/high (sweet-spot shape)."""
    if value <= low or value >= high:
        return 0.0
    if value <= peak:
        return (value - low) / (peak - low)
    return (high - value) / (high - peak)


def component_scores(row: pd.Series) -> dict[str, float]:
    """Each component normalised to 0-1; missing data scores a neutral 0.5."""
    s: dict[str, float] = {}

    v = row.get("six_month_return")
    # Screen floor is 5%; reward more, but peak at ~30% since the very hottest
    # names are the most reversal-prone over the next month.
    s["momentum_6m"] = _triangle(v, 5.0, 30.0, 100.0) if pd.notna(v) else 0.5

    v = row.get("ext_vs_sma50")
    # Allowed up to 15% above the 50-day SMA; nearer (or just below) is a much
    # better 4-week entry than buying an extended chart.
    s["entry_quality"] = _clamp01((15.0 - v) / 20.0) if pd.notna(v) else 0.5

    price, sma200 = row.get("price"), row.get("sma_200")
    # Healthy margin above the 200-day SMA — enough to confirm the trend, not so
    # much that the move is already exhausted.
    if pd.notna(price) and pd.notna(sma200) and sma200 > 0:
        s["trend"] = _triangle((price / sma200 - 1) * 100, 0.0, 12.0, 60.0)
    else:
        s["trend"] = 0.5

    v = row.get("rsi_14")
    # MEASURED, not assumed.  Inside the screen's own 40-65 band the 45-50
    # bucket is the WORST performer (avg -0.05%, 50.0% win over 685 picks --
    # see ab_sweep.py "rsi_shape_in_band"), while 60-65 wins most often
    # (64.3%).  alpha_lab agrees at the universe level: |RSI-50| scores
    # ICIR -0.238 (t=-3.38), so proximity to 50 predicts LOWER returns.
    # The old triangle peaked at 50 and therefore handed full marks to the
    # weakest zone.  A monotone ramp fits both measurements.  The 40-45 bucket
    # also did well but on only 54 picks, too thin to justify a second peak.
    s["rsi_zone"] = _clamp01((v - 40.0) / 25.0) if pd.notna(v) else 0.5

    v = row.get("atr_pct")
    # Allowed band [1, 6]; lower volatility scores higher for hit rate.
    s["low_vol"] = _clamp01((6.0 - v) / 5.0) if pd.notna(v) else 0.5

    v = row.get("return_on_equity")
    # Screen floor is 0.10; 35%+ ROE is full marks.
    s["quality"] = _clamp01((v - 0.10) / 0.25) if pd.notna(v) else 0.5

    v = row.get("avg_dollar_volume")
    # log-scale from the screen's $10M floor to ~$1B.
    if pd.notna(v) and v > 0:
        s["liquidity"] = _clamp01((math.log10(v) - 7.0) / 2.0)
    else:
        s["liquidity"] = 0.5

    v = row.get("days_to_earnings")
    # Blackout already rejected anything inside the 28-day hold window; a full
    # quarter of clear air is fully derisked.
    s["earnings_gap"] = _clamp01((v - 28.0) / 62.0) if pd.notna(v) and v >= 0 else 0.5

    return s


def potential_score(row: pd.Series) -> tuple[float, dict[str, float]]:
    """0-100 technical score, then the sentiment adjustment.

    net_sentiment_adjustment (0 to -20) comes from the screener: a CNN
    Fear & Greed penalty plus a negative-news-flow penalty.  It is SUBTRACTED
    from the technical score rather than gating the name out, so an
    exceptional setup can still outrank a mediocre one with clean news."""
    comps = component_scores(row)
    total = sum(WEIGHTS[name] * comps[name] for name in WEIGHTS)
    adj = row.get("net_sentiment_adjustment")
    if pd.notna(adj):
        total = max(0.0, total + float(adj))
    return round(total, 2), comps


def position_weight(row: pd.Series, base: float = 1.0) -> float:
    """Suggested position size multiplier: the screener's sentiment_scale.

    Kept separate from the score on purpose — sentiment says how much capital
    to commit to the whole book, not which names are better than which."""
    scale = row.get("sentiment_scale")
    return round(base * float(scale), 3) if pd.notna(scale) else base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("short_term_candidates.csv"),
                        help="Candidates CSV from the screener (default: short_term_candidates.csv)")
    parser.add_argument("--output", type=Path, default=Path("top_candidates.csv"),
                        help="Ranked output CSV (default: top_candidates.csv)")
    parser.add_argument("--top", type=int, default=5,
                        help="How many candidates to keep (default: 5)")
    parser.add_argument("--alpha191-weight", type=float, default=0.0,
                        help="Blend weight for the Guotai Junan Alpha-191 composite in the "
                             "ranking: 0 = pure 6-month relative strength (default), "
                             "1 = pure Alpha-191. UNVALIDATED on this pipeline — A/B it with "
                             "win_rate_tester.py before trusting it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rr.begin("select_top15", args)
    if args.top < 1:
        raise SystemExit("--top must be positive.")
    if not args.input.exists():
        raise SystemExit(
            f"'{args.input}' not found. Run short_term_screener_alpaca.py first "
            f"(it writes this file), or pass --input <file>."
        )

    df = pd.read_csv(args.input)
    if df.empty:
        raise SystemExit(
            f"'{args.input}' has no candidates — the screen passed nothing on its "
            f"last run, so there is nothing to rank."
        )

    scores, comp_rows = [], []
    for _, row in df.iterrows():
        total, comps = potential_score(row)
        scores.append(total)
        comp_rows.append({f"score_{k}": round(v, 3) for k, v in comps.items()})

    df["potential_score"] = scores
    df = pd.concat([df, pd.DataFrame(comp_rows, index=df.index)], axis=1)
    if "sentiment_scale" in df.columns:
        df["position_weight"] = [position_weight(r) for _, r in df.iterrows()]
    # Primary rank: 6-month relative strength vs SPY (rs_6m) — the documented
    # cross-sectional momentum signal; historical testing showed the composite
    # score alone gave no rank separation.  Composite kept as the tie-breaker
    # and for the per-component columns.
    alpha_pct = alpha191_percentile(df)
    if alpha_pct is not None:
        df["alpha191_pct"] = alpha_pct.round(3)
    if args.alpha191_weight > 0 and alpha_pct is not None and "rs_6m" in df.columns:
        # Blend percentile ranks: 0 = pure relative strength, 1 = pure Alpha-191.
        w = min(args.alpha191_weight, 1.0)
        rs_pct = df["rs_6m"].rank(pct=True)
        df["blended_rank"] = ((1 - w) * rs_pct + w * alpha_pct).round(4)
        df = df.sort_values(["blended_rank", "potential_score"], ascending=False).reset_index(drop=True)
    elif "rs_6m" in df.columns and df["rs_6m"].notna().any():
        df = df.sort_values(["rs_6m", "potential_score"], ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values("potential_score", ascending=False).reset_index(drop=True)

    kept = df.head(args.top)
    kept.to_csv(args.output, index=False)

    print(f"Ranked {len(df)} candidates from {args.input}; kept top {len(kept)}.")
    if len(df) <= args.top:
        print(f"NOTE: only {len(df)} candidates available (≤ --top {args.top}); kept them all.")
    print()
    def _f(row, col, width, dec):
        v = row.get(col)
        return f"{v:>{width}.{dec}f}" if pd.notna(v) else " " * (width - 1) + "-"

    header = (f"{'#':>2}  {'ticker':<6} {'score':>6}  {'price':>8}  {'6m%':>7}  "
              f"{'RSI':>5}  {'vs50d%':>7}  {'ATR%':>5}  {'ROE':>6}  {'ern_d':>5}")
    print(header)
    print("-" * len(header))
    for i, row in kept.iterrows():
        roe = row.get("return_on_equity")
        print(f"{i + 1:>2}  {row['ticker']:<6} {row['potential_score']:>6.1f}  "
              f"{_f(row,'price',8,2)}  {_f(row,'six_month_return',7,1)}  {_f(row,'rsi_14',5,1)}  "
              f"{_f(row,'ext_vs_sma50',7,1)}  {_f(row,'atr_pct',5,1)}  "
              f"{(f'{roe:>6.1%}' if pd.notna(roe) else '     -')}  {_f(row,'days_to_earnings',5,0)}")
    rr.metrics("select_top15", "headline", {
        "candidates_in": len(df), "kept": len(kept),
        "alpha191_weight": args.alpha191_weight,
        "ranked_by": ("blended_rank" if "blended_rank" in df.columns
                      else "rs_6m" if "rs_6m" in df.columns else "potential_score"),
        "top_score": kept["potential_score"].max() if len(kept) else None,
        "median_score": round(kept["potential_score"].median(), 2) if len(kept) else None,
        "market_overlay": (str(df.iloc[0]["market_overlay"])
                           if "market_overlay" in df.columns and len(df) else None),
    })
    rr.rows("select_top15", "picks", [
        {"rank": i + 1, "ticker": r["ticker"], "score": r["potential_score"],
         "price": r.get("price"), "six_month_return": r.get("six_month_return"),
         "rs_6m": r.get("rs_6m"), "rsi_14": r.get("rsi_14"),
         "ext_vs_sma50": r.get("ext_vs_sma50"), "atr_pct": r.get("atr_pct"),
         "selection": r.get("company_selection")}
        for i, (_, r) in enumerate(kept.iterrows())])

    print(f"\nFull ranking with per-component scores written to {args.output}")

    if "position_weight" in kept.columns and len(kept):
        w = float(kept.iloc[0]["position_weight"])
        if abs(w - 1.0) > 1e-6:
            print(f"Sentiment position-size weight: x{w:.2f} — size each position at "
                  f"{w:.0%} of normal (the tape is stretched, but good setups still qualify).")
    if "net_sentiment_adjustment" in kept.columns:
        pen = kept["net_sentiment_adjustment"].dropna()
        if len(pen) and (pen < 0).any():
            print(f"Sentiment/news penalties applied to {(pen < 0).sum()} of {len(pen)} picks "
                  f"(worst {pen.min():+.1f} pts).")
    if "market_overlay" in df.columns and len(df) and str(df.iloc[0]["market_overlay"]) != "BUY_ALLOWED":
        print(f"WARNING: market overlay is {df.iloc[0]['market_overlay']} — "
              f"the screener flagged a risk-off tape on this run.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
