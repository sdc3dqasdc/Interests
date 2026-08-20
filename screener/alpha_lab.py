#!/usr/bin/env python3
"""Alpha factor lab: clean the data, measure IC/ICIR on YOUR universe, keep the best.

Published ICIRs are not transferable — they are measured on other universes,
periods and horizons (Alpha191 on A-shares, Alpha101 on a different era).  So
this tool computes everything empirically on your own Alpaca data:

  1. DATA CLEANING (drop what is untradeable, tame what is extreme)
     - drop tickers with too little history
     - drop bars that are stale/untradeable: non-positive volume, non-finite or
       non-positive prices, high<low, and frozen runs (price unchanged for
       --stale-run sessions), which are halts/delistings masquerading as data
     - enforce a per-date eligibility floor (price, dollar volume) so illiquid
       names cannot dominate a cross-section
     - winsorize each cross-section by MAD (median +/- k*1.4826*MAD) then
       z-score it, so one blown-up value cannot swing the factor

  2. EVALUATION
     - cross-sectional RankIC of each factor vs forward --horizon-day returns
     - IC mean / std / ICIR (mean/std) / t-stat / share of positive days
     - DECAY CHECK: IC on the first vs second half of the window

  3. SELECTION (drop the weak and the unstable)
     - drop factors below --min-icir
     - drop factors whose IC SIGN FLIPS between halves (unstable / decayed)
     - keep the --top N survivors by |ICIR|, writing alpha_selection.json

Factor sign convention: every factor is oriented so that HIGHER = predicted
better forward return, so a positive IC means the factor works as written.

Usage:
    python3 alpha_lab.py --start 2019-01-01 --end 2024-12-31
    python3 alpha_lab.py --start 2019-01-01 --end 2024-12-31 --horizon 20 --top 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import backtest as bt
import run_report as rr

# ---------------------------------------------------------------------------
# Factor pool — common, well-documented alphas computable from daily OHLCV.
# Each returns a Series aligned to df.index, oriented HIGHER = BETTER.
# ---------------------------------------------------------------------------
def _clv(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, float("nan"))
    return (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng).astype(float)


def _ret(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI, same smoothing as backtest.compute_rsi."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Standard MACD(12,26,9) -> (line, signal, histogram).

    NOTE: line/signal/histogram are in PRICE units, so every MACD factor below
    divides by close.  Raw MACD on a $500 stock dwarfs the same *relative* move
    on a $15 one, so an unnormalised MACD ranks the cross-section mostly by
    price level — the same trap the Alpha-191 adaptation already documents."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


FACTORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # --- GTJA Alpha-191 survivors (already used by the screener) -----------
    # dropped gtja002_reversal (ICIR -0.018) and gtja111_volprice (-0.043,
    # sign-flipping): both near-zero and unstable in run_report.md.
    "gtja054_pxaction": lambda d: -(
        ((d["Close"] - d["Open"]) / d["Close"]).abs().rolling(10).std()
        + (d["Close"] - d["Open"]) / d["Close"]
        + d["Close"].rolling(10).corr(d["Open"])
    ),

    # --- Alpha101-style price/volume forms ---------------------------------
    # WQ#101: intraday drive, normalised by the day's range.
    "wq101_intraday": lambda d: (d["Close"] - d["Open"]) / ((d["High"] - d["Low"]) + 0.001),
    # (dropped wq012_volshock: ICIR 0.021, sign-flipping — noise.)
    # Price-volume correlation: falling correlation precedes reversals.
    "pv_corr_10": lambda d: -d["Close"].rolling(10).corr(d["Volume"]),

    # --- Classic academic factors ------------------------------------------
    # Short-term (1-month) reversal — Jegadeesh 1990 / Lehmann 1990.
    "st_reversal_21": lambda d: -d["Close"].pct_change(21) * 100,
    # 12-1 momentum — Jegadeesh & Titman 1993, skipping the last month.
    "momentum_12_1": lambda d: (d["Close"].shift(21) / d["Close"].shift(252) - 1) * 100,
    # 6-1 momentum, the screener's horizon.
    "momentum_6_1": lambda d: (d["Close"].shift(21) / d["Close"].shift(126) - 1) * 100,
    # Low-volatility anomaly — lower realised vol has earned better risk-adj returns.
    "low_vol_20": lambda d: -_ret(d).rolling(20).std() * 100,
    # (dropped amihud_illiq: ICIR 0.023, sign-flipping — liquidity floor
    # already handled by the per-date eligibility gate anyway.)
    # Negative-skew preference: lottery-like right skew underperforms (Bali 2011).
    "neg_skew_60": lambda d: -_ret(d).rolling(60).skew(),
    # Proximity to the 52-week high — George & Hwang 2004.
    "high52_prox": lambda d: d["Close"] / d["Close"].rolling(252).max() * 100,
    # Volume trend: rising participation.
    "vol_trend_20": lambda d: d["Volume"].rolling(5).mean() / d["Volume"].rolling(20).mean(),
    # Distance below the 50-day SMA (consolidation, not breakdown).
    "ext_sma50": lambda d: -((d["Close"] / d["Close"].rolling(50).mean() - 1) * 100).abs(),

    # --- RSI family ---------------------------------------------------------
    # Connors RSI(2): the canonical short-term mean-reversion setup — a deeply
    # oversold 2-period RSI inside an established uptrend.  Negated so that a
    # LOWER RSI(2) (more oversold = the buy signal) scores HIGHER.
    "rsi2_oversold": lambda d: -_rsi(d["Close"], 2),
    # Same idea at the standard period, to see whether the effect is specific
    # to the very short lookback or survives at RSI(14).
    "rsi14_oversold": lambda d: -_rsi(d["Close"], 14),
    # The screener's implicit thesis: RSI near 50 = a pullback inside an
    # uptrend, better than either extreme.  Higher = closer to 50.
    "rsi14_near50": lambda d: -(_rsi(d["Close"], 14) - 50).abs(),
    # RSI momentum: is the oscillator turning up over the last week?
    "rsi14_slope5": lambda d: _rsi(d["Close"], 14).diff(5),

    # --- MACD family (12,26,9), all normalised by close ---------------------
    # dropped macd_hist_slope (ICIR -0.004) and macd_hist_rev (0.023, the
    # redundant inverse of macd_hist) — both dead in run_report.md.
    # Histogram = MACD - signal.  Positive/widening = bullish momentum.
    "macd_hist": lambda d: _macd(d["Close"])[2] / d["Close"] * 100,
    # MACD line itself vs zero — the trend-regime reading.
    "macd_line": lambda d: _macd(d["Close"])[0] / d["Close"] * 100,

    # --- grab-bag additions, untested, just to see what sticks --------------
    # Bollinger %B(20,2): position within the band, 0=lower band, 1=upper.
    "boll_pctb_20": lambda d: (
        (d["Close"] - (d["Close"].rolling(20).mean() - 2 * d["Close"].rolling(20).std()))
        / ((4 * d["Close"].rolling(20).std()).replace(0, float("nan")))
    ),
    # OBV trend: is on-balance volume rising relative to its own average?
    "obv_trend_20": lambda d: (
        (np.sign(d["Close"].diff()) * d["Volume"]).cumsum().rolling(5).mean()
        / (np.sign(d["Close"].diff()) * d["Volume"]).cumsum().rolling(20).mean()
    ),
    # ATR(14) as a percent of price: pure volatility level, no direction.
    "atr_pct_14": lambda d: -(
        pd.concat([
            d["High"] - d["Low"],
            (d["High"] - d["Close"].shift()).abs(),
            (d["Low"] - d["Close"].shift()).abs(),
        ], axis=1).max(axis=1).ewm(alpha=1 / 14, adjust=False).mean() / d["Close"] * 100
    ),
    # Stochastic %K(14): close's position in the 14-day high/low range.
    "stoch_k_14": lambda d: (
        (d["Close"] - d["Low"].rolling(14).min())
        / ((d["High"].rolling(14).max() - d["Low"].rolling(14).min()).replace(0, float("nan")))
        * 100
    ),

    # --- second grab-bag batch, also untested ------------------------------
    # Williams %R(14): same range as stoch_k_14 but anchored off the high
    # (0 = at the high, -100 = at the low); flipped so higher = better.
    "williams_r_14": lambda d: (
        (d["Close"] - d["High"].rolling(14).max())
        / ((d["High"].rolling(14).max() - d["Low"].rolling(14).min()).replace(0, float("nan")))
        * 100
    ),
    # Rate of change over 10 sessions: simple short-term momentum.
    "roc_10": lambda d: d["Close"].pct_change(10) * 100,
    # Donchian channel position over 20 sessions: 0=at the low, 1=at the high.
    "donchian_pct_20": lambda d: (
        (d["Close"] - d["Low"].rolling(20).min())
        / ((d["High"].rolling(20).max() - d["Low"].rolling(20).min()).replace(0, float("nan")))
    ),
    # Chaikin Money Flow(20): volume-weighted accumulation/distribution.
    "cmf_20": lambda d: (
        (_clv(d) * d["Volume"]).rolling(20).sum() / d["Volume"].rolling(20).sum()
    ),

    # --- third grab-bag batch, also untested -------------------------------
    # TRIX(15): triple-smoothed EMA rate of change — a de-noised momentum read.
    "trix_15": lambda d: (
        lambda e: e.pct_change() * 100
    )(
        d["Close"].ewm(span=15, adjust=False).mean()
                  .ewm(span=15, adjust=False).mean()
                  .ewm(span=15, adjust=False).mean()
    ),
    # Aroon oscillator(25): trend strength from time-since-high vs time-since-low.
    "aroon_osc_25": lambda d: (
        d["High"].rolling(26).apply(lambda w: w.argmax(), raw=True)
        - d["Low"].rolling(26).apply(lambda w: w.argmin(), raw=True)
    ) * (100.0 / 25.0),
    # Money Flow Index(14): volume-weighted RSI on the typical price.
    "mfi_14": lambda d: (
        lambda tp: (
            (tp.diff() > 0) * tp * d["Volume"]
        ).rolling(14).sum() / (
            (tp.diff() < 0) * tp * d["Volume"]
        ).rolling(14).sum().replace(0, float("nan"))
    )((d["High"] + d["Low"] + d["Close"]) / 3),
    # Overnight gap: today's open vs yesterday's close (open-drift effect).
    "overnight_gap": lambda d: (d["Open"] / d["Close"].shift(1) - 1) * 100,
}


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
def clean_bars(df: pd.DataFrame, stale_run: int) -> pd.DataFrame:
    """Drop untradeable/stale bars from one ticker's history."""
    d = df.copy()
    price_cols = ["Open", "High", "Low", "Close"]
    ok = np.isfinite(d[price_cols].to_numpy(dtype=float)).all(axis=1)
    ok &= (d[price_cols] > 0).all(axis=1).to_numpy()
    ok &= (d["High"] >= d["Low"]).to_numpy()
    ok &= (d["Volume"].fillna(0) > 0).to_numpy()
    d = d[ok]
    if d.empty or stale_run <= 0:
        return d
    # Frozen price runs: identical close for stale_run+ consecutive sessions is
    # a halt or a stale feed, not a market.  Drop the whole run.
    grp = (d["Close"] != d["Close"].shift(1)).cumsum()
    run_len = d.groupby(grp)["Close"].transform("size")
    return d[run_len < stale_run]


def winsorize_zscore(x: pd.Series, mad_k: float) -> pd.Series:
    """MAD-winsorize then z-score one cross-section."""
    v = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 5:
        return pd.Series(dtype=float)
    med = v.median()
    mad = (v - med).abs().median()
    if mad > 0:
        lo, hi = med - mad_k * 1.4826 * mad, med + mad_k * 1.4826 * mad
        v = v.clip(lo, hi)
    sd = v.std()
    return (v - v.mean()) / sd if sd and sd > 0 else pd.Series(dtype=float)


def rank_ic(factor: pd.Series, fwd: pd.Series) -> float | None:
    """Cross-sectional Spearman (rank) IC for one date."""
    joined = pd.concat([factor, fwd], axis=1).dropna()
    if len(joined) < 20:
        return None
    a, b = joined.iloc[:, 0].rank(), joined.iloc[:, 1].rank()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(a.corr(b))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", type=Path, default=Path("screener_universe.txt"))
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--horizon", type=int, default=50,
                   help="Forward return horizon in trading days (default: 50, the 10-week hold)")
    p.add_argument("--top", type=int, default=5, help="Factors to keep (default: 5)")
    p.add_argument("--min-icir", type=float, default=0.02,
                   help="Kick out factors whose |ICIR| is below this (default: 0.02)")
    p.add_argument("--sample-every", type=int, default=5,
                   help="Evaluate every Nth session (default: 5, keeps overlap manageable)")
    p.add_argument("--max-tickers", type=int, default=1500,
                   help="Cap universe size for speed (default: 1500)")
    p.add_argument("--min-bars", type=int, default=300, help="Minimum clean bars per ticker")
    p.add_argument("--stale-run", type=int, default=5,
                   help="Drop runs of this many identical closes as stale (0 disables)")
    p.add_argument("--mad-k", type=float, default=5.0, help="MAD winsorization width")
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-volume", type=float, default=1_000_000)
    p.add_argument("--warmup-days", type=int, default=500)
    p.add_argument("--alpaca-key", type=str, default=None)
    p.add_argument("--alpaca-secret", type=str, default=None)
    p.add_argument("--refresh-cache", action="store_true",
                   help="Ignore today's cached Alpaca bars and refetch (cache normally makes "
                        "a same-day rerun skip the network call)")
    p.add_argument("--report", type=Path, default=Path("alpha_report.csv"))
    p.add_argument("--selection", type=Path, default=Path("alpha_selection.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rr.begin("alpha_lab", args)
    if not args.tickers.exists():
        raise SystemExit(f"Ticker file '{args.tickers}' not found — run the screener first.")
    tickers = bt.read_tickers(args.tickers)[:args.max_tickers]

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    client = bt._get_alpaca_client(args.alpaca_key, args.alpaca_secret)
    raw = bt.fetch_price_data_cached(tickers, client, start - timedelta(days=args.warmup_days), end,
                                     refresh=args.refresh_cache)
    if not raw:
        raise SystemExit("No price data returned from Alpaca.")

    # --- 1. clean -----------------------------------------------------------
    print("Cleaning bars (stale/untradeable removal) …")
    hist, dropped_bars, dropped_tickers = {}, 0, 0
    for t, df in raw.items():
        before = len(df)
        c = clean_bars(df, args.stale_run)
        dropped_bars += before - len(c)
        if len(c) < args.min_bars:
            dropped_tickers += 1
            continue
        hist[t] = c
    print(f"  -> {len(hist)} tickers kept, {dropped_tickers} dropped for short history, "
          f"{dropped_bars:,} stale/untradeable bars removed")
    if len(hist) < 30:
        raise SystemExit("Too few tickers survived cleaning to measure a cross-section.")

    # --- 2. panels ----------------------------------------------------------
    print(f"Computing {len(FACTORS)} factors …")
    close = pd.DataFrame({t: d["Close"] for t, d in hist.items()}).sort_index()
    dollar_vol = pd.DataFrame(
        {t: (d["Close"] * d["Volume"]).rolling(20).mean() for t, d in hist.items()}).sort_index()
    fwd = close.shift(-args.horizon) / close - 1          # forward return, no lookahead in factors
    eligible = (close >= args.min_price) & (dollar_vol >= args.min_dollar_volume)

    panels: dict[str, pd.DataFrame] = {}
    for i, (name, fn) in enumerate(FACTORS.items(), 1):
        cols = {}
        for t, d in hist.items():
            try:
                s = fn(d)
                if isinstance(s, pd.Series):
                    cols[t] = s.astype(float)
            except Exception:
                continue
        if cols:
            panels[name] = pd.DataFrame(cols).reindex(close.index)
        print(f"  [{i}/{len(FACTORS)}] {name}", flush=True)

    # --- 3. evaluate --------------------------------------------------------
    start_dt = pd.Timestamp(args.start)
    dates = [d for d in close.index if d >= start_dt][::args.sample_every]
    dates = [d for d in dates if d in fwd.index and fwd.loc[d].notna().sum() >= 20]
    print(f"Evaluating RankIC on {len(dates)} cross-sections "
          f"(horizon {args.horizon}d, every {args.sample_every} sessions) …")

    rows = []
    for name, panel in panels.items():
        ics: list[tuple[pd.Timestamp, float]] = []
        for date in dates:
            if date not in panel.index:
                continue
            mask = eligible.loc[date] if date in eligible.index else None
            f = panel.loc[date]
            if mask is not None:
                f = f.where(mask)
            f = winsorize_zscore(f, args.mad_k)      # clean the cross-section
            if f.empty:
                continue
            ic = rank_ic(f, fwd.loc[date])
            if ic is not None and math.isfinite(ic):
                ics.append((date, ic))
        if len(ics) < 20:
            continue
        ser = pd.Series({d: v for d, v in ics}).sort_index()
        mean, sd = ser.mean(), ser.std()
        icir = mean / sd if sd and sd > 0 else 0.0
        half = len(ser) // 2
        ic1, ic2 = ser.iloc[:half].mean(), ser.iloc[half:].mean()
        rows.append({
            "factor": name, "n_dates": len(ser),
            "ic_mean": round(mean, 5), "ic_std": round(sd, 5),
            "icir": round(icir, 4),
            "t_stat": round(mean / sd * math.sqrt(len(ser)), 2) if sd and sd > 0 else 0.0,
            "pct_positive": round((ser > 0).mean() * 100, 1),
            "ic_first_half": round(ic1, 5), "ic_second_half": round(ic2, 5),
            "sign_flip": bool(np.sign(ic1) != np.sign(ic2)),
            "decay_pct": round((abs(ic2) / abs(ic1) - 1) * 100, 1) if ic1 else None,
        })

    if not rows:
        raise SystemExit("No factor produced enough cross-sections — widen the window.")
    rep = pd.DataFrame(rows)
    rep["abs_icir"] = rep["icir"].abs()
    rep = rep.sort_values("abs_icir", ascending=False).reset_index(drop=True)
    rep.drop(columns=["abs_icir"]).to_csv(args.report, index=False)

    # --- 4. select: drop the weak and the sign-unstable ----------------------
    kept = rep[(rep["icir"].abs() >= args.min_icir) & (~rep["sign_flip"])]
    dropped_weak = rep[rep["icir"].abs() < args.min_icir]["factor"].tolist()
    dropped_flip = rep[rep["sign_flip"]]["factor"].tolist()
    top = kept.head(args.top)

    print(f"\n{'factor':<20} {'ICIR':>7} {'IC mean':>9} {'t':>6} {'%pos':>6} "
          f"{'IC 1st':>8} {'IC 2nd':>8}  flag")
    print("-" * 78)
    for _, r in rep.iterrows():
        flag = "SIGN FLIP" if r["sign_flip"] else ("weak" if abs(r["icir"]) < args.min_icir else "")
        if r["factor"] in set(top["factor"]):
            flag = "** KEPT **"
        print(f"{r['factor']:<20} {r['icir']:>7.4f} {r['ic_mean']:>9.5f} {r['t_stat']:>6.2f} "
              f"{r['pct_positive']:>6.1f} {r['ic_first_half']:>8.5f} {r['ic_second_half']:>8.5f}  {flag}")

    print(f"\nKicked out: {len(dropped_weak)} weak (|ICIR| < {args.min_icir}), "
          f"{len(dropped_flip)} sign-flipped (decayed/unstable)")
    if dropped_flip:
        print(f"  sign-flipped: {', '.join(dropped_flip)}")

    selection = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": args.start, "end": args.end, "horizon_days": args.horizon},
        "criteria": {"min_icir": args.min_icir, "reject_sign_flip": True},
        "selected": [
            {"factor": r["factor"], "icir": r["icir"], "ic_mean": r["ic_mean"],
             "t_stat": r["t_stat"], "pct_positive": r["pct_positive"]}
            for _, r in top.iterrows()
        ],
    }
    args.selection.write_text(json.dumps(selection, indent=2))

    rr.metrics("alpha_lab", "headline", {
        "window_start": args.start, "window_end": args.end,
        "horizon_days": args.horizon, "sample_every": args.sample_every,
        "tickers_kept": len(hist), "cross_sections": len(dates),
        "factors_evaluated": len(rep), "factors_kept": len(top),
        "dropped_weak": len(dropped_weak), "dropped_sign_flip": len(dropped_flip),
        "best_abs_icir": round(rep["icir"].abs().max(), 4),
    })
    rr.rows("alpha_lab", "factors", rep.drop(columns=["abs_icir"]).to_dict("records"))
    rr.rows("alpha_lab", "selected", top[["factor", "icir", "ic_mean", "t_stat"]].to_dict("records"))
    rr.note("alpha_lab", "caveats",
            "Single-factor ICIRs of 0.02-0.05 are normal and weak; they are meant to be "
            "combined across a wide universe, not to carry a 5-pick portfolio. ICs here are "
            "measured on the FULL eligible universe, not on the post-screen survivor set.")

    print(f"\nTop {len(top)} by |ICIR| -> {args.selection}")
    print(f"Full report -> {args.report}")
    if len(top) and abs(top.iloc[0]["icir"]) < 0.05:
        print("\nNOTE: even the best |ICIR| here is small. Single-factor ICIRs of 0.02-0.05 are "
              "normal and weak — they are meant to be combined across a wide universe with "
              "frequent rebalancing, not to carry a 5-pick portfolio on their own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
