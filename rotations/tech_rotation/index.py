"""NTI — the Nasdaq Tilt Index: when to own the Nasdaq instead of the S&P 500.

The seed idea is the CSI 300 / ChiNext ratio trade: when the growth index is
cheap relative to the blue-chip index, rotate into growth; when it is expensive,
rotate back into the steady one.  The US analogue is QQQ / SPY.

That ratio on its own is a pure mean-reversion bet, and mean reversion alone is
what got people short the Nasdaq for the whole of 2013-2021 — the ratio was
"expensive" for eight straight years while QQQ beat SPY by ~200%.  So the ratio
is one input of several, not the signal:

  * valuation   the QQQ/SPY ratio's gap from its own long trend, INVERTED
                (the seed idea: a cheap ratio favours tech)
  * momentum    6-month QQQ-vs-SPY relative return, straight (a cheap ratio
                that is still falling is a value trap; this is the brake)
  * reversal    1-month relative return, inverted (fade the blow-off)
  * regime      SPY's own 200-day trend (tech leadership dies in bear markets)
  * rel vol     QQQ's realised vol vs SPY's, inverted (tech leads when it is
                not the volatile leg)
  * rates       TLT's 3-month return (long-duration growth is discounted at the
                long rate; falling yields are a tailwind, 2022 was the proof)
  * credit      HYG-vs-IEF 3-month spread move (risk appetite)
  * semis       SMH-vs-QQQ 3-month relative return (semis lead the tech tape)
  * concentr.   SPY-vs-RSP 3-month relative return (mega-cap concentration
                regimes are the ones QQQ wins)

Every component is turned into a trailing z-score and signed so that POSITIVE
means "favour the Nasdaq".  The composite is a weighted mean of whichever
components exist on that date, re-scaled by its own trailing standard deviation
so a reading is in sigma units and comparable across decades.

Nothing here uses a future bar: every mean, standard deviation and rank comes
from a trailing window ending on the day the reading is stamped.  Weights are
set a priori by role, not fitted — the backtest exists to test them, and
`--ablate` reports what each one is actually worth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

# Trailing windows for standardisation.  Five years is long enough to span a
# full leadership cycle and short enough that the 1999 bubble is not still
# setting the scale in 2020; two years minimum keeps the early history usable.
Z_WINDOW = 1260
Z_MIN = 504
Z_CLIP = 3.0

# A priori weights by role, not fitted.  The two core disagreeing views
# (valuation vs momentum) and the regime gate carry full weight; confirmations
# carry half.
DEFAULT_WEIGHTS: dict[str, float] = {
    "valuation": 1.00,
    "momentum": 1.00,
    "reversal": 0.50,
    "regime": 1.00,
    "rel_vol": 0.50,
    "rates": 0.50,
    "credit": 0.50,
    "semis": 0.75,
    "concentration": 0.25,
}

# Candidates under test, not yet promoted to CORE_WEIGHTS. Weighted the same
# as the other "confirmation" components (0.50) so they get to earn their
# place rather than being handed extra influence. `sentiment` only has data
# back to 2021 (see data.fetch_fear_greed) so it cannot be checked against the
# pre-2013 half the way everything else here is -- ablate() reports its own
# window, read it as OOS-only evidence, not a full both-halves pass.
CANDIDATE_WEIGHTS: dict[str, float] = {
    "sentiment": 0.50,
    "trend_50": 0.50,
}


# The trimmed index.  Chosen by a rule, not by picking the best combination:
# keep only components whose timing contribution has the SAME SIGN in both
# disjoint halves of the sample (1999-2012 and 2013-now), and whose removal
# hurts both halves.  That leaves regime, momentum, rel_vol and rates.
#
# The five dropped ones fail in two distinguishable ways.  valuation and
# reversal -- both mean-reversion bets, and the valuation leg is the seed
# idea itself -- are negative in BOTH halves: in the US the stretched QQQ/SPY
# ratio has been a reason to keep holding, not to sell.  credit, semis and
# concentration flip sign between halves, which is the signature of a factor
# that was never there.  Relative weights of the survivors are unchanged from
# DEFAULT_WEIGHTS, so nothing here is re-fitted.
CORE_WEIGHTS: dict[str, float] = {
    "regime": 1.00,
    "momentum": 1.00,
    "rel_vol": 0.50,
    "rates": 0.50,
}


def _z(series: pd.Series, window: int = Z_WINDOW, min_periods: int = Z_MIN) -> pd.Series:
    """Trailing z-score, clipped.  Clipping matters: the March-2000 and
    March-2020 tails are 6-8 sigma events that would otherwise let one
    component dictate the whole composite for months."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-Z_CLIP, Z_CLIP)


def _realised_vol(close: pd.Series, window: int = 21) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * np.sqrt(252)


def _regime(spy: pd.Series, window: int = 200) -> pd.Series:
    """SPY's own trend, as a bounded -1..+1 score rather than a z.

    +1  above a rising 200-day SMA      (the regime QQQ wins in)
     0  above a falling one, or below a rising one (transition)
    -1  below a falling 200-day SMA     (drawdowns hit QQQ ~1.3x as hard)

    Scaled by 1.5 so a bounded score carries weight comparable to a z-score
    that ranges over +-3."""
    sma = spy.rolling(window).mean()
    above = (spy > sma).astype(float)
    rising = (sma > sma.shift(21)).astype(float)
    return ((above + rising) - 1.0) * 1.5


def build_components(px: pd.DataFrame, fear_greed: pd.Series | None = None) -> pd.DataFrame:
    """One column per component, each signed so positive = favour the Nasdaq.

    `fear_greed` is optional: pass data.fetch_fear_greed() to include the
    `sentiment` candidate. Without it, comps has the 9 price-only components
    plus `trend_50`; sentiment is simply absent, same as any other component
    whose source series hasn't started yet."""
    # Ratio legs run off the underlying indices, which predate the ETFs by 14
    # years, so the z-scores are warm when QQQ starts trading.  The ETFs are
    # what actually gets held; see data.TECH_IDX for why this is not cheating.
    tech = px[dt.TECH_IDX].fillna(px[dt.TECH])
    steady = px[dt.STEADY_IDX].fillna(px[dt.STEADY])
    rel = np.log(tech / steady)  # the Nasdaq-100 / S&P-500 ratio, in logs

    comps = {}

    comps["valuation"] = -_z(rel - rel.ewm(span=200, min_periods=100).mean())

    # Six months is the horizon at which relative momentum persists.
    comps["momentum"] = _z(rel.diff(126))

    comps["reversal"] = -_z(rel.diff(21))

    comps["regime"] = _regime(steady)

    # A QQQ/SPY vol ratio well above its norm has marked every tech unwind
    # since 2000.
    comps["rel_vol"] = -_z(_realised_vol(tech) / _realised_vol(steady))

    # TLT up = yields down = a tailwind for long-duration growth.
    comps["rates"] = _z(np.log(px[dt.LONG_BOND]).diff(63))

    # High yield beating duration-matched Treasuries is risk appetite, which
    # shows up in the growth leg first.
    comps["credit"] = _z(np.log(px[dt.CREDIT] / px[dt.INT_BOND]).diff(63))

    # An internal confirmation, not a duplicate of QQQ-vs-SPY momentum.
    comps["semis"] = _z(np.log(px[dt.SEMIS] / px[dt.TECH]).diff(63))

    # Cap-weighted beating equal-weight = a concentration regime, the kind QQQ
    # wins.  Smallest weight: the noisiest of the confirmations.
    comps["concentration"] = _z(np.log(px[dt.STEADY] / px[dt.EQUAL_WT]).diff(63))

    # Technical candidate: short-horizon trend confirmation. `momentum` (126d)
    # and `regime` (200d SMA) are both slow; this is the ratio's distance from
    # its own 50-day EMA, NOT inverted -- a trend-following read at a horizon
    # short enough to catch what the slower two miss, distinct from
    # `valuation`'s 200-day mean-reversion read on the same ratio.
    comps["trend_50"] = _z(rel - rel.ewm(span=50, min_periods=25).mean())

    # Sentiment candidate: CNN Fear & Greed, contrarian -- the `reversal`
    # construction applied to whole-market sentiment rather than the QQQ/SPY
    # ratio.  Panic (a low reading against its own trailing average) favours
    # the higher-beta leg; crowded greed favours stepping back.  Data starts
    # 2021-02-01; earlier dates are absent, same handling as HYG/RSP above.
    if fear_greed is not None:
        fg = fear_greed.reindex(px.index).ffill()
        comps["sentiment"] = -_z(fg - fg.rolling(252, min_periods=60).mean(), window=504, min_periods=120)

    return pd.DataFrame(comps, index=px.index)


def composite(
    comps: pd.DataFrame,
    weights: dict[str, float] | None = None,
    rescale_window: int = 756,
    rescale_min: int = 252,
) -> pd.Series:
    """Weighted mean of the available components, re-scaled to sigma units.

    Components go missing at the start of history (HYG lists in 2007, and every
    z needs two years of warm-up), so the mean is taken over whatever exists on
    each date and re-normalised by the weight actually present.  A date with
    less than half the total weight available yields no reading."""
    weights = weights or DEFAULT_WEIGHTS
    cols = [c for c in comps.columns if c in weights]
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)

    present = comps[cols].notna().astype(float)
    weight_present = present.mul(w, axis=1).sum(axis=1)
    raw = comps[cols].mul(w, axis=1).sum(axis=1, min_count=1) / weight_present.replace(0.0, np.nan)
    raw = raw.where(weight_present >= 0.5 * w.sum())

    # Averaging ~9 partly independent z-scores shrinks the spread well below 1
    # sigma, and by how much depends on how correlated they happen to be in that
    # era.  Re-scaling by the composite's own trailing std keeps a "+2 reading"
    # meaning the same thing in 2004 and 2024.  Trailing only -- no lookahead.
    scale = raw.rolling(rescale_window, min_periods=rescale_min).std()
    return (raw / scale.replace(0.0, np.nan)).clip(-Z_CLIP, Z_CLIP)


def build_index(px: pd.DataFrame, weights: dict[str, float] | None = None
                ) -> tuple[pd.Series, pd.DataFrame]:
    comps = build_components(px)
    return composite(comps, weights), comps


# Where the buy decision actually sits.  Bucketing every historical reading and
# measuring the NEXT month of QQQ-minus-SPY return gives a table that is flat
# above the line and negative below it -- not a smooth ladder:
#
#     reading        % of time   next 21d QQQ-SPY (in-sample / out-of-sample)
#     > +1.5              22%        +0.80%  /  +0.70%
#     +0.5 .. +1.5        39%        +0.44%  /  +0.48%
#     -0.5 .. +0.5        20%        +0.76%  /  +0.60%
#     -1.5 .. -0.5        14%        -0.86%  /  -0.26%
#     < -1.5               6%        -3.12%  /  +0.25%   <- IS only, do not trust
#
# Two things follow.  Above -0.5 the Nasdaq wins by roughly the same margin
# whatever the reading, so paying turnover to distinguish +0.6 from +2.0 buys
# nothing.  Below -0.5 it loses in both halves.  And the deepest bucket's huge
# in-sample number is the dot-com bust alone -- out of sample it is positive, so
# "maximum bearish = maximum short" is not supported.
#
# Hence a switch, not a dial.  The threshold is deliberately not tuned: the
# backtest gives Sharpe 0.61-0.63 anywhere from -0.75 to 0.0.
BUY_THRESHOLD = -0.5
RISK_ON_WEIGHT = 1.00
RISK_OFF_WEIGHT = 0.20


def ladder_weight(nti: pd.Series, threshold: float = BUY_THRESHOLD,
                  risk_on: float = RISK_ON_WEIGHT, risk_off: float = RISK_OFF_WEIGHT
                  ) -> pd.Series:
    """The two-state allocation: heavy Nasdaq above the threshold, defensive below.

    Beats a static blend of the same average exposure in the full sample and in
    both halves separately, at lower turnover than the continuous tilt."""
    return pd.Series(np.where(nti > threshold, risk_on, risk_off),
                     index=nti.index).where(nti.notna())


def target_weight(nti: pd.Series, k: float = 0.25, floor: float = 0.0, cap: float = 1.0
                  ) -> pd.Series:
    """Map a reading to a Nasdaq weight; the remainder goes to the S&P.

    50/50 at a neutral reading, all-in at +2 sigma, all-out at -2 sigma.  A
    continuous tilt rather than a flip: the index is a probabilistic read, and
    binary switching pays the spread on every wobble around zero (the backtest
    reports the binary variant for comparison)."""
    return (0.5 + k * nti).clip(floor, cap)
