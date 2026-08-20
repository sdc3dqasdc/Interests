"""KDI — the Kechuang-Dibo Index: STAR50 (科创50) vs Dividend Low Volatility (红利低波).

Same seed idea as tech_rotation/ (its README has the full derivation): when the
growth board is cheap against the steady dividend index, tilt toward growth;
when it's expensive, tilt back to the steady one.  And the same finding from
that test carries the warning label here: the raw ratio-mean-reversion signal
was the ONE thing that lost money on 27 years of US data.  What survived there
was a regime/momentum/vol/rates read, not the ratio itself.

This module is built the same way and with the same components, substituted
for onshore analogues:

  * valuation   KC50/DIBO ratio's gap from its own trend, INVERTED (the raw idea)
  * momentum    6-month KC50-vs-DIBO relative return (the brake on the value trap)
  * reversal    1-month relative return, inverted
  * regime      CSI300's own 200-day trend (market-wide risk-on/off gate)
  * rel_vol     KC50's realised vol vs DIBO's, inverted
  * rates       China treasury bond ETF's 3-month return (falling yields help growth)
  * semis       China semiconductor ETF vs KC50, 3-month relative return
  * concentration  SSE50-vs-CSI300 3-month relative return (mega-cap concentration)

There is no `credit` component here — there is no liquid onshore high-yield ETF
on yfinance to build HYG/IEF's role, and that factor didn't survive the US test
either, so it isn't missed.

CRITICAL CAVEAT, unlike the US module: STAR50 only has ~5 years of tradeable
history (lists Sep-2020).  The US index was validated on 27 years split into two
13-year halves; this one can only be split into two ~2.5-year halves.  That is
not enough data to distinguish a real regime effect from one lucky/unlucky
stretch.  Treat every number in this module as a much weaker read than its US
counterpart — see backtest.py's docstring and the README before trusting a
component selection here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

Z_WINDOW = 504     # ~2 years — shorter than the US module's 5, because there
Z_MIN = 252        # simply isn't 5 years of history before STAR50 even lists
Z_CLIP = 3.0

DEFAULT_WEIGHTS: dict[str, float] = {
    "valuation": 1.00,
    "momentum": 1.00,
    "reversal": 0.50,
    "regime": 1.00,
    "rel_vol": 0.50,
    "rates": 0.50,
    "semis": 0.75,
    "concentration": 0.25,
}

# NOT a re-derivation of tech_rotation.CORE_WEIGHTS -- porting the US subset
# verbatim was tried first and it LOSES here in every window (Sharpe 0.57 vs
# 0.94 for the full 8-factor index, in the full sample and in both halves).
# The four-factor idea does not transfer; onshore growth-vs-dividend dynamics
# are not the same shape as QQQ-vs-SPY.
#
# A from-scratch "China core" is deliberately NOT built to replace it.  ~5.6
# years of STAR50 history splits into two ~2.8-year halves -- not enough
# statistical power to trust a same-sign-in-both-halves filter (with 8
# components and 2 halves, a few passing by chance is expected).  The one
# item with a real, low-power signal against it: `momentum` is negative in
# both halves here (t=-0.67, t=-1.59) -- the opposite of its role in the US
# index -- but dropping it changes results by less than the noise band, so it
# is kept rather than hand-edited on two data points.
#
# DEFAULT_WEIGHTS (all 8) is the recommended reading.  CORE_WEIGHTS is kept
# only so the module can show, side by side, that porting the US answer here
# would have been a mistake.
CORE_WEIGHTS: dict[str, float] = {
    "regime": 1.00,
    "momentum": 1.00,
    "rel_vol": 0.50,
    "rates": 0.50,
}

BUY_THRESHOLD = -0.5
RISK_ON_WEIGHT = 1.00
RISK_OFF_WEIGHT = 0.20


def _z(series: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-Z_CLIP, Z_CLIP)


def _realised_vol(close: pd.Series, window: int = 21) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * np.sqrt(252)


def _regime(broad: pd.Series, window: int = 200) -> pd.Series:
    sma = broad.rolling(window).mean()
    above = (broad > sma).astype(float)
    rising = (sma > sma.shift(21)).astype(float)
    return ((above + rising) - 1.0) * 1.5


def build_components(px: pd.DataFrame) -> pd.DataFrame:
    """One column per component, positive = favour KC50 (STAR50)."""
    # Ratio legs run off the ChiNext/CSI300 proxy pair before STAR50 lists, and
    # off the actual STAR50 ETF once it exists -- same splicing trick as
    # tech_rotation, but over a much shorter proxy gap (2015-2020, not 1985-1999).
    growth = px[dt.KC50].fillna(px[dt.KC50_PROXY])
    broad = px[dt.CSI300]
    rel = np.log(growth / broad)

    comps = {}
    comps["valuation"] = -_z(rel - rel.ewm(span=200, min_periods=100).mean())
    comps["momentum"] = _z(rel.diff(126))
    comps["reversal"] = -_z(rel.diff(21))
    comps["regime"] = _regime(broad)
    comps["rel_vol"] = -_z(_realised_vol(growth) / _realised_vol(broad))
    comps["rates"] = _z(np.log(px[dt.BONDS]).diff(63))
    comps["semis"] = _z(np.log(px[dt.SEMIS] / growth).diff(63))
    comps["concentration"] = _z(np.log(px[dt.SSE50] / broad).diff(63))
    return pd.DataFrame(comps, index=px.index)


def composite(comps: pd.DataFrame, weights: dict[str, float] | None = None,
             rescale_window: int = 378, rescale_min: int = 126) -> pd.Series:
    weights = weights or DEFAULT_WEIGHTS
    cols = [c for c in comps.columns if c in weights]
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)

    present = comps[cols].notna().astype(float)
    weight_present = present.mul(w, axis=1).sum(axis=1)
    raw = comps[cols].mul(w, axis=1).sum(axis=1, min_count=1) / weight_present.replace(0.0, np.nan)
    raw = raw.where(weight_present >= 0.5 * w.sum())

    scale = raw.rolling(rescale_window, min_periods=rescale_min).std()
    return (raw / scale.replace(0.0, np.nan)).clip(-Z_CLIP, Z_CLIP)


def build_index(px: pd.DataFrame, weights: dict[str, float] | None = None
                ) -> tuple[pd.Series, pd.DataFrame]:
    comps = build_components(px)
    return composite(comps, weights), comps


def ladder_weight(nti: pd.Series, threshold: float = BUY_THRESHOLD,
                  risk_on: float = RISK_ON_WEIGHT, risk_off: float = RISK_OFF_WEIGHT
                  ) -> pd.Series:
    return pd.Series(np.where(nti > threshold, risk_on, risk_off),
                     index=nti.index).where(nti.notna())


def target_weight(nti: pd.Series, k: float = 0.25, floor: float = 0.0, cap: float = 1.0
                  ) -> pd.Series:
    return (0.5 + k * nti).clip(floor, cap)
