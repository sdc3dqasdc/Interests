#!/usr/bin/env python3
"""Shared 1-day-horizon signal core for the day-trade tools.

next_day_predictor.py and one_day_top3.py both answer a 1-day question ("does
this go up tomorrow?"), so the feature/score logic lives here once.

HONESTY NOTE — read before trusting any output built on this:
  The 1-day horizon is close to a random walk. The only edge with real academic
  support at 1-2 days is SHORT-TERM MEAN REVERSION: a name that just sold off
  inside an established uptrend tends to bounce (Connors' RSI(2) setup, Lehmann
  1990 reversal). That edge is small (a few percent of hit rate over 50/50) and
  it evaporates under daily-turnover slippage. So the score below is oriented to
  that effect, and the tools MEASURE its realised hit rate rather than asserting
  it. If the measured hit rate is ~50%, the signal is worthless on your data —
  that is the expected null result, not a bug.

Columns are the Capitalized OHLCV that backtest.fetch_price_data_cached returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the short-horizon features the 1-day score is built from."""
    d = df.copy()
    close = d["Close"]
    d["rsi2"] = bt.compute_rsi(close, 2)              # 2-day RSI: the mean-reversion gauge
    d["ret1"] = close.pct_change(1) * 100             # yesterday's move (reversion input)
    ema10 = close.ewm(span=10, adjust=False).mean()
    d["ext_ema10"] = (close / ema10 - 1) * 100        # stretch vs the 10-day EMA
    d["sma200"] = close.rolling(200).mean()
    d["uptrend"] = close > d["sma200"]                # only fade dips inside an uptrend
    d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean().shift(1)
    d["dollar_vol"] = (close * d["Volume"]).rolling(20).mean()
    return d


def _zscore(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Rolling per-ticker z-score, so one ticker can be scored on its own history."""
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def next_day_up_score(df: pd.DataFrame) -> pd.Series:
    """Score oriented so HIGHER = more likely to rise the next session.

    A blend of mean-reversion signals (oversold RSI(2), a down day, price stretched
    BELOW its 10-day EMA, capitulation volume), gated to only fire inside an
    uptrend. Outside an uptrend the score is damped toward neutral — fading dips
    in a downtrend is knife-catching."""
    d = compute_features(df)
    z_oversold = _zscore(-d["rsi2"])        # higher when more oversold
    z_reversal = _zscore(-d["ret1"])        # higher when yesterday fell
    z_pullback = _zscore(-d["ext_ema10"])   # higher when below the 10-EMA
    z_volume = _zscore(d["vol_ratio"])      # higher on a volume spike (capitulation)
    score = z_oversold + z_reversal + 0.5 * z_pullback + 0.25 * z_volume
    # Connors filter: only take the bounce bet in an uptrend; else push it down.
    return score.where(d["uptrend"], score - 3.0)


def next_day_return(df: pd.DataFrame) -> pd.Series:
    """Close-to-close return of the NEXT session (what the score tries to predict).

    NaN on the last bar (no next day yet) — that bar is the live prediction."""
    return df["Close"].shift(-1) / df["Close"] - 1.0


def next_day_high_gain(df: pd.DataFrame) -> pd.Series:
    """Best-case gain from today's close to TOMORROW's high.

    This is what decides whether a +2% take-profit limit could have filled the
    next day: target hit  <=>  next_day_high_gain >= 0.02.  It is optimistic (it
    assumes a limit at the high fills), so realised results will be worse."""
    return df["High"].shift(-1) / df["Close"] - 1.0


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a %% of close — the daily move budget, for sizing a target."""
    atr = bt.compute_atr(df["High"], df["Low"], df["Close"], period)
    return atr / df["Close"] * 100


def hit_rate_by_bucket(scores: pd.Series, next_rets: pd.Series,
                       n_buckets: int = 5) -> pd.DataFrame | None:
    """Bucket realised (score, next-day return) pairs to show the score's edge.

    This is the honesty check: if the top bucket's up-rate is no better than the
    bottom's, the score has no 1-day predictive value on this data."""
    d = pd.DataFrame({"score": scores, "ret": next_rets}).dropna()
    if len(d) < n_buckets * 20:
        return None
    try:
        d["bucket"] = pd.qcut(d["score"], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return None
    g = d.groupby("bucket").agg(
        n=("ret", "size"),
        up_rate_pct=("ret", lambda s: (s > 0).mean() * 100),
        avg_next_ret_pct=("ret", lambda s: s.mean() * 100),
    )
    return g.round(2)
