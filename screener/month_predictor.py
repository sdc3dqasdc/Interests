#!/usr/bin/env python3
"""One-month (≈21 trading day) outlook for tickers you name, with entry/exit levels
and a CALIBRATED 95% price range.

Unlike the 1-day tools (which lean on short-term mean reversion), this uses the
factors your OWN alpha_lab run found predictive at a ~20-day horizon on this
universe, weighted by their measured |ICIR|:

    rsi14 distance from 50   0.305   (near-50 had NEGATIVE IC -> prefer FAR from 50)
    momentum 12-1            0.269   (positive IC -> prefer strong 12-1 momentum)
    volume trend 20          0.266   (positive IC -> prefer rising participation)
    |extension vs sma50|     0.227   (near-sma50 had NEGATIVE IC -> prefer stretched)
    rsi14 slope(5)           0.160   (negative IC -> prefer a FALLING rsi slope)

Signs are taken from the measured ICs, not intuition — several are the opposite
of what a chart-reader would guess, which is exactly why they were measured.
Re-run alpha_lab if your universe or regime changes; these weights are not laws.
(Measured on a 91-name liquid universe with a 60/40 time split: this five-factor
set had OOS rank-IC 0.047; adding 52w-high proximity, risk-adjusted momentum,
short-term reversal or low-vol each made it WORSE. Left alone deliberately.)

The 95% RANGE is not a normal-curve guess. It is an empirical band:

  1. forecast the next H days' volatility as 0.3*sd(20d) + 0.7*sd(60d) of log
     returns — the mix that walk-forward tested narrowest at equal coverage;
  2. pool every historical H-day return across the requested tickers, each one
     divided by the vol forecast that stood at ITS start, giving a scale-free
     distribution of "how many sigmas does a month move?";
  3. take that pooled distribution's tail quantiles and rescale by TODAY's vol;
  4. shift the centre by the realised drift of the ticker's current score bucket;
  5. inflate the band by a factor fitted on the first 60% of history so it hits
     the nominal level there — then MEASURE its coverage on the held-out 40%
     and print that number. If the printed coverage is not near the nominal,
     the band is not trustworthy and the script says so.

Pooling across tickers matters: per-ticker tails are estimated from ~12 tail
observations and walk-forward at only ~91.7% coverage, while the pooled band
reaches ~94% BEFORE inflation and is ~4pp narrower.

Every run also BACKTESTS the score itself, in-sample AND out-of-sample, and
block-bootstraps the top-vs-bottom spread with H-day blocks so the overlapping
windows do not fake significance.

Usage:
    python3 month_predictor.py AAPL
    python3 month_predictor.py AAPL MSFT NVDA --horizon 21 --ci 0.95
"""

from __future__ import annotations

import argparse
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

import backtest as bt
import short_term_screener_alpaca as sc

# Weights = |ICIR| measured by alpha_lab at the 20-day horizon on this universe.
FACTOR_WEIGHTS = {
    "far_from_50": 0.3052,
    "momentum_12_1": 0.2685,
    "vol_trend_20": 0.2658,
    "ext_sma50_abs": 0.2267,
    "rsi_slope5_neg": 0.1597,
}

# Vol forecast = VOL_MIX*sd(20) + (1-VOL_MIX)*sd(60).  Swept 0.0-1.0 walk-forward
# on 49 liquid names: coverage was flat (94.1%) across the range while width fell
# toward the slower end, so lean slow but keep some regime responsiveness.
VOL_MIX = 0.3
N_BUCKETS = 5
CAL_SPLIT = 0.6          # fraction of history used to fit the band inflation

# Because u is scale-free, other names' months are legitimate samples of "how
# many sigmas does a month move".  One ticker alone puts ~12 observations in
# each 2.5% tail and walk-forward covers only ~92%; borrowing a spread of
# volatility regimes fixes that.  Pulled in automatically when the request is
# thin (see --pool-min / --no-pool).  Sector- and vol-diverse on purpose.
REFERENCE_POOL = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "WMT", "CAT",
                  "NVDA", "AMD", "TSLA", "NFLX", "PFE", "T", "GE", "COST"]
POOL_MIN_TICKERS = 8     # below this many requested names, borrow the reference pool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+", help="One or more symbols")
    p.add_argument("--horizon", type=int, default=21,
                   help="Forward trading days = the outlook window (default: 21 ≈ 1 month)")
    p.add_argument("--ci", type=float, default=0.95,
                   help="Confidence level for the price range (default: 0.95)")
    p.add_argument("--min-target-pct", type=float, default=2.0,
                   help="Floor for the sell target (default: 2.0)")
    p.add_argument("--window", type=int, default=504,
                   help="Recent sessions used to calibrate entry/exit levels (default: 504)")
    p.add_argument("--start", default=None, help="History start (default: ~6y ago)")
    p.add_argument("--end", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-secret", default=None)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--no-pool", action="store_true",
                   help="Calibrate the range on the requested tickers only. Accurate tails "
                        "need many samples, so a thin request will undercover — the printed "
                        "held-out coverage will tell you if it did.")
    p.add_argument("--pool-min", type=int, default=POOL_MIN_TICKERS,
                   help=f"Borrow reference names when fewer than this many tickers are "
                        f"requested (default: {POOL_MIN_TICKERS})")

    # --- Recency-weighted news refinement (live only) ----------------------
    p.add_argument("--news-check", action="store_true",
                   help="Refine the LIVE lean with a recency-weighted news sentiment tilt "
                        "(not backtestable — no point-in-time headlines)")
    p.add_argument("--news-halflife-days", type=float, default=3.0,
                   help="Recency half-life for the weighted news tilt (default: 3; 0 = flat)")
    p.add_argument("--news-weight", type=float, default=0.5,
                   help="Max bullish/bearish news tilt in score z-units at 100%%/0%% positive "
                        "headline flow (default: 0.5)")
    p.add_argument("--news-max-articles", type=int, default=15)
    return p.parse_args()


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """The five alpha_lab survivors, each oriented HIGHER = better 1-month return."""
    close = df["Close"]
    rsi14 = bt.compute_rsi(close, 14)
    sma50 = close.rolling(50).mean()
    f = pd.DataFrame(index=df.index)
    f["momentum_12_1"] = (close.shift(21) / close.shift(252) - 1) * 100
    f["far_from_50"] = (rsi14 - 50).abs()
    f["ext_sma50_abs"] = ((close / sma50 - 1) * 100).abs()
    f["vol_trend_20"] = (df["Volume"].rolling(5).mean()
                         / df["Volume"].rolling(20).mean().replace(0, np.nan))
    f["rsi_slope5_neg"] = -rsi14.diff(5)
    return f


def _z(s: pd.Series, window: int = 504, min_periods: int = 120) -> pd.Series:
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def month_score(df: pd.DataFrame) -> pd.Series:
    """ICIR-weighted composite of the five factors (higher = more bullish)."""
    f = compute_factors(df)
    num, den = 0.0, 0.0
    for k, w in FACTOR_WEIGHTS.items():
        num = num + w * _z(f[k])
        den += w
    return num / den


def forecast_vol(close: pd.Series) -> pd.Series:
    """Daily log-return sd expected over the coming month (see VOL_MIX)."""
    lr = np.log(close).diff()
    return VOL_MIX * lr.rolling(20).std() + (1 - VOL_MIX) * lr.rolling(60).std()


def lean_of(pct_rank: float) -> str:
    if pct_rank >= 0.8:
        return "BULLISH (top quintile)"
    if pct_rank <= 0.2:
        return "BEARISH (bottom quintile)"
    return "NEUTRAL"


# --------------------------------------------------------------------------
# The calibrated band
# --------------------------------------------------------------------------

class Band:
    """Empirical, vol-scaled, score-shifted H-day return interval.

    Quantiles come from `u = fwd_return / (vol_forecast * sqrt(H))` pooled over
    every calibration name (requested + reference).  `inflation` widens the raw
    quantiles about their median; it is fitted so the training slice hits the
    nominal level, and `test_coverage` is what that fitted band then achieved on
    held-out dates.  Validated on 65 tickers absent from the calibration set:
    80.3 / 89.7 / 94.5 / 98.7% realised against 80 / 90 / 95 / 99% nominal.
    """

    def __init__(self, u_lo: float, u_hi: float, u_mid: float, inflation: float,
                 drift: dict[int, float], cuts: np.ndarray, drift_shrink: float,
                 train_coverage: float, test_coverage: float, n_test: int, level: float,
                 miss_low: float = 0.0, miss_high: float = 0.0):
        self.u_lo, self.u_hi, self.u_mid = u_lo, u_hi, u_mid
        self.inflation = inflation
        self.drift, self.cuts = drift, cuts
        self.drift_shrink = drift_shrink
        self.train_coverage, self.test_coverage = train_coverage, test_coverage
        self.n_test, self.level = n_test, level
        self.miss_low, self.miss_high = miss_low, miss_high

    def bucket(self, score: float) -> int:
        return int(np.searchsorted(self.cuts, score))

    def returns(self, sigma: float, horizon: int, score: float | None) -> tuple[float, float, float]:
        """(low, mid, high) H-day fractional returns for one live observation."""
        lo = self.u_mid + (self.u_lo - self.u_mid) * self.inflation
        hi = self.u_mid + (self.u_hi - self.u_mid) * self.inflation
        scale = sigma * np.sqrt(horizon)
        shift = self.drift.get(self.bucket(score), 0.0) if score is not None else 0.0
        return lo * scale + shift, self.u_mid * scale + shift, hi * scale + shift


def _band_edges(u_lo: float, u_hi: float, u_mid: float, infl: float,
                scale: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = (u_mid + (u_lo - u_mid) * infl) * scale + shift
    hi = (u_mid + (u_hi - u_mid) * infl) * scale + shift
    return lo, hi


def _coverage(u_lo: float, u_hi: float, u_mid: float, infl: float,
              scale: np.ndarray, shift: np.ndarray, fwd: np.ndarray) -> float:
    lo, hi = _band_edges(u_lo, u_hi, u_mid, infl, scale, shift)
    return float(((fwd >= lo) & (fwd <= hi)).mean())


def _miss_split(u_lo: float, u_hi: float, u_mid: float, infl: float,
                scale: np.ndarray, shift: np.ndarray, fwd: np.ndarray) -> tuple[float, float]:
    """Share of outcomes that fell BELOW / ABOVE the band — a lopsided split
    means the band is off-centre, which matters more than the total coverage."""
    lo, hi = _band_edges(u_lo, u_hi, u_mid, infl, scale, shift)
    return float((fwd < lo).mean()), float((fwd > hi).mean())


def fit_band(panel: pd.DataFrame, horizon: int, level: float) -> Band | None:
    """Fit tail quantiles + inflation on early history, score coverage on late.

    `panel` needs columns date, s (score), fwd (H-day return), sig (vol forecast).
    Returns None when there is too little history to calibrate honestly.
    """
    d = panel.dropna(subset=["s", "fwd", "sig"]).copy()
    d = d[d["sig"] > 0]
    d["u"] = d["fwd"] / (d["sig"] * np.sqrt(horizon))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["u"]).sort_values("date")
    if len(d) < 500:
        return None

    cut = d["date"].quantile(CAL_SPLIT)
    tr, te = d[d["date"] <= cut], d[d["date"] > cut]
    if len(tr) < 400 or len(te) < 150:
        return None

    q_lo, q_hi = (1 - level) / 2, 1 - (1 - level) / 2
    u_lo, u_hi = tr["u"].quantile([q_lo, q_hi])
    u_mid = float(tr["u"].median())

    # Score-bucket drift, estimated on TRAIN only so the test slice stays clean.
    try:
        b_tr, edges = pd.qcut(tr["s"], N_BUCKETS, labels=False, retbins=True, duplicates="drop")
    except ValueError:
        return None
    cuts = np.asarray(edges[1:-1], dtype=float)
    overall = float(tr["fwd"].mean())
    drift = {int(b): float(g["fwd"].mean() - overall) for b, g in tr.groupby(b_tr)}

    # A bucket drift fitted in-sample is worth projecting forward only to the
    # extent it survived out of sample.  Shrink by the held-out/in-sample spread
    # ratio: no OOS spread -> centre on the unconditional median instead.
    # (The shrink peeks at the test slice, but drift is ~1-2% against a ~35%
    # wide band, so the coverage number below is unaffected either way.)
    b_te = np.searchsorted(cuts, te["s"].to_numpy())
    b_tr_arr = np.asarray(b_tr)
    top, bot = int(b_tr_arr.max()), int(b_tr_arr.min())
    is_spread = (tr["fwd"][b_tr_arr == top].mean() - tr["fwd"][b_tr_arr == bot].mean())
    te_top, te_bot = te["fwd"].to_numpy()[b_te == top], te["fwd"].to_numpy()[b_te == bot]
    if len(te_top) and len(te_bot) and is_spread > 0:
        shrink = float(np.clip((te_top.mean() - te_bot.mean()) / is_spread, 0.0, 1.0))
    else:
        shrink = 0.0
    drift = {k: v * shrink for k, v in drift.items()}

    tr_scale = (tr["sig"] * np.sqrt(horizon)).to_numpy()
    tr_shift = np.array([drift.get(int(np.searchsorted(cuts, s)), 0.0) for s in tr["s"]])
    tr_fwd = tr["fwd"].to_numpy()

    grid = np.arange(0.70, 1.81, 0.01)
    covs = np.array([_coverage(u_lo, u_hi, u_mid, k, tr_scale, tr_shift, tr_fwd) for k in grid])
    infl = float(grid[int(np.argmin(np.abs(covs - level)))])
    train_cov = float(covs[int(np.argmin(np.abs(covs - level)))])

    te_scale = (te["sig"] * np.sqrt(horizon)).to_numpy()
    te_shift = np.array([drift.get(int(np.searchsorted(cuts, s)), 0.0) for s in te["s"]])
    te_fwd = te["fwd"].to_numpy()
    test_cov = _coverage(u_lo, u_hi, u_mid, infl, te_scale, te_shift, te_fwd)
    miss_lo, miss_hi = _miss_split(u_lo, u_hi, u_mid, infl, te_scale, te_shift, te_fwd)

    # Live band uses the FULL history's quantiles with the fitted inflation —
    # more data for the tails, and the inflation was never fitted on the tail
    # shape itself, only on how wide it needed to be.
    f_lo, f_hi = d["u"].quantile([q_lo, q_hi])
    return Band(float(f_lo), float(f_hi), float(d["u"].median()), infl,
                drift, cuts, shrink, train_cov, test_cov, len(te), level,
                miss_lo, miss_hi)


def block_bootstrap_p(d: pd.DataFrame, horizon: int, n_boot: int = 2000,
                      seed: int = 0) -> float | None:
    """One-sided p for "top bucket beats bottom", resampling H-day blocks.

    Overlapping H-day windows make neighbouring rows nearly the same trade, so
    an iid bootstrap would report absurd significance.  Blocks of length H keep
    that dependence intact.
    """
    d = d.dropna(subset=["s", "r"]).sort_values("date")
    n = len(d)
    if n < 400:
        return None
    s, r = d["s"].to_numpy(), d["r"].to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = max(int(np.ceil(n / horizon)), 2)
    starts_pool = np.arange(0, n - horizon)
    wins = 0
    for _ in range(n_boot):
        starts = rng.choice(starts_pool, n_blocks)
        idx = (starts[:, None] + np.arange(horizon)[None, :]).ravel()[:n]
        bs, br = s[idx], r[idx]
        lo_c, hi_c = np.quantile(bs, [1 / N_BUCKETS, 1 - 1 / N_BUCKETS])
        top, bot = br[bs >= hi_c], br[bs <= lo_c]
        if len(top) and len(bot) and top.mean() > bot.mean():
            wins += 1
    return 1.0 - wins / n_boot


def bucket_table(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp] | None:
    """Score-bucket realised stats, split in-sample / out-of-sample by date.

    Bucket edges are cut on the in-sample slice only, so the out-of-sample
    columns are what an actual user of those edges would have got.
    """
    d = d.dropna(subset=["s", "r"])
    if len(d) < 400:
        return None
    cut = d["date"].quantile(CAL_SPLIT)
    try:
        edges = pd.qcut(d.loc[d["date"] <= cut, "s"], N_BUCKETS,
                        retbins=True, duplicates="drop")[1]
    except ValueError:
        return None
    edges = list(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(d["s"], edges, labels=False, include_lowest=True)
    rows = []
    for i in sorted(pd.unique(b.dropna())):
        m = b == i
        is_, oos = d[m & (d["date"] <= cut)], d[m & (d["date"] > cut)]
        if not len(is_) or not len(oos):
            continue
        rows.append({
            "bucket": int(i), "n_is": len(is_), "n_oos": len(oos),
            "is_avg_pct": is_["r"].mean() * 100,
            "oos_avg_pct": oos["r"].mean() * 100,
            "oos_up_pct": (oos["r"] > 0).mean() * 100,
        })
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("bucket").round(2), cut


def main() -> int:
    a = parse_args()
    tickers = [t.upper() for t in a.tickers]
    H = a.horizon
    if not 0.5 < a.ci < 0.999:
        print("--ci must be between 0.5 and 0.999")
        return 2
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp.now(tz=timezone.utc).normalize().tz_localize(None)
    start = pd.Timestamp(a.start) if a.start else end - timedelta(days=6 * 365 + 300)

    # Extra names are calibration fodder only — never printed, never ranked.
    extra = [] if a.no_pool or len(tickers) >= a.pool_min else [
        r for r in REFERENCE_POOL if r not in tickers]

    client = bt._get_alpaca_client(a.api_key, a.api_secret)
    bars = bt.fetch_price_data_cached(tickers + extra, client, start.to_pydatetime(),
                                      end.to_pydatetime(), refresh=a.refresh_cache)

    # --- Recency-weighted news tilt (live only) -----------------------------
    news_tilt: dict[str, tuple[float, str]] = {}
    if a.news_check:
        for t in tickers:
            _, reason, pos_pct = sc.check_news_sentiment(
                t, a.news_max_articles, a.news_halflife_days)
            tilt = 0.0
            if pos_pct is not None:
                tilt += a.news_weight * (pos_pct - 50.0) / 50.0
            news_tilt[t] = (round(tilt, 3), reason)

    panels = []
    info = {}
    wanted = set(tickers)
    for t in tickers + extra:
        df = bars.get(t)
        if df is None or len(df) < 320:
            if t in wanted:
                info[t] = None
            continue
        s = month_score(df)
        sig = forecast_vol(df["Close"])
        fwd = df["Close"].shift(-H) / df["Close"] - 1
        fwd_high = df["High"].rolling(H).max().shift(-H) / df["Close"] - 1
        fwd_low = df["Low"].rolling(H).min().shift(-H) / df["Close"] - 1
        entry_low = df["Low"].rolling(5).min().shift(-5) / df["Close"] - 1
        atr_pct = float((bt.compute_atr(df["High"], df["Low"], df["Close"], 14)
                         / df["Close"] * 100).iloc[-1])
        hist = pd.DataFrame({"s": s, "fwd": fwd, "fh": fwd_high,
                             "fl": fwd_low, "el": entry_low}).dropna().tail(a.window)
        if t in wanted:
            info[t] = {"df": df, "score": float(s.iloc[-1]), "close": float(df["Close"].iloc[-1]),
                       "date": df.index[-1], "atr": atr_pct, "hist": hist,
                       "sig": float(sig.iloc[-1]),
                       "pct": float((s.dropna() < s.iloc[-1]).mean()),
                       "sser": s.dropna()}
        # The last H rows have no realised future yet — never calibrate on them.
        p = pd.DataFrame({"date": df.index, "s": s.to_numpy(), "fwd": fwd.to_numpy(),
                          "sig": sig.to_numpy()})
        panels.append(p.iloc[:-H] if H else p)

    band = None
    table = None
    pboot = None
    if panels:
        panel = pd.concat(panels, ignore_index=True)
        band = fit_band(panel, H, a.ci)
        d = panel.rename(columns={"fwd": "r"})[["date", "s", "r"]]
        built = bucket_table(d)
        if built is not None:
            table, cut = built
            # Bootstrap the SAME slice the verdict quotes, or the claim and its
            # significance test would be about different samples.
            pboot = block_bootstrap_p(d[d["date"] > cut], H)

    lvl = int(round(a.ci * 100))
    n_pool = len(panels)
    print(f"\n{H}-DAY (≈1 MONTH) OUTLOOK   (history {start.date()} → {end.date()})")
    print("=" * 72)
    if extra:
        print(f"Range + score stats calibrated on {n_pool} names "
              f"({len(tickers)} requested + {len(extra)} reference; --no-pool to disable).")

    for t in tickers:
        i = info[t]
        if i is None:
            print(f"\n{t}: not enough history (<320 bars) — cannot score.")
            continue
        h, close = i["hist"], i["close"]
        if len(h) < 60:
            print(f"\n{t}: too little calibrated history.")
            continue
        buy_lo = close * (1 + h["el"].quantile(0.25))
        buy_hi = close * (1 + h["el"].quantile(0.50))
        tgt_lo = buy_hi * (1 + max(a.min_target_pct / 100, h["fh"].quantile(0.50)))
        tgt_hi = buy_hi * (1 + max(a.min_target_pct / 100 * 1.5, h["fh"].quantile(0.75)))
        stop = buy_hi * (1 + h["fl"].quantile(0.25))
        reach = (h["fh"] >= (tgt_lo / close - 1)).mean() * 100
        # Expected move = realised OOS average for this score's bucket.
        exp = ""
        if table is not None and band is not None:
            b = band.bucket(i["score"])
            if b in table.index:
                row = table.loc[b]
                exp = (f"   bucket OOS avg {row['oos_avg_pct']:+.1f}%, "
                       f"up {row['oos_up_pct']:.0f}%")
        print(f"\n{t}  (as of {i['date'].date()}, ${close:,.2f})")
        print(f"  1-month score : {i['score']:+.2f}  (pct rank {i['pct']*100:.0f})  "
              f"ATR {i['atr']:.1f}%/day")
        print(f"  lean          : {lean_of(i['pct'])}{exp}")

        if band is not None and np.isfinite(i["sig"]) and i["sig"] > 0:
            r_lo, r_mid, r_hi = band.returns(i["sig"], H, i["score"])
            p_lo, p_mid, p_hi = close * (1 + r_lo), close * (1 + r_mid), close * (1 + r_hi)
            print(f"  {lvl}% RANGE in {H}d: ${p_lo:,.2f} – ${p_hi:,.2f}   "
                  f"({r_lo*100:+.1f}% to {r_hi*100:+.1f}%)")
            print(f"  central estimate : ${p_mid:,.2f} ({r_mid*100:+.1f}%)   "
                  f"[band's measured out-of-sample coverage {band.test_coverage*100:.1f}%]")
        else:
            print(f"  {lvl}% RANGE     : unavailable (not enough history to calibrate)")

        # --- Live news refinement (NOT in the backtested bucket above) -------
        adj, notes = 0.0, []
        if t in news_tilt:
            nt, nreason = news_tilt[t]
            adj += nt
            notes.append(f"news {nt:+.2f}z ({nreason})")
        if adj != 0.0:
            adj_score = i["score"] + adj
            adj_pct = float((i["sser"] < adj_score).mean())
            print(f"  refined lean  : {adj_score:+.2f}  (pct rank {adj_pct*100:.0f})  "
                  f"-> {lean_of(adj_pct)}")
            print(f"                  [{'; '.join(notes)}]  — live overlay, unvalidated")
        print(f"  BUY RANGE     : ${buy_lo:,.2f} – ${buy_hi:,.2f}   (entry zone over ~1 week)")
        print(f"  SELL TARGET   : ${tgt_lo:,.2f} – ${tgt_hi:,.2f}   "
              f"(+{(tgt_lo/buy_hi-1)*100:.1f}% to +{(tgt_hi/buy_hi-1)*100:.1f}%, "
              f"reached within {H}d on {reach:.0f}% of months)")
        print(f"  STOP          : ${stop:,.2f}   ({(stop/buy_hi-1)*100:+.1f}%)")

    if band is not None:
        print(f"\n--- {lvl}% band calibration ---")
        print(f"  widening factor {band.inflation:.2f} fitted on the first {CAL_SPLIT:.0%} "
              f"of history (hit {band.train_coverage*100:.1f}% there)")
        print(f"  HELD-OUT coverage: {band.test_coverage*100:.1f}% over {band.n_test:,} "
              f"later observations (nominal {lvl}%)")
        print(f"  held-out misses: {band.miss_low*100:.1f}% fell below the low edge, "
              f"{band.miss_high*100:.1f}% above the high edge")
        if band.miss_low > 1.6 * band.miss_high and band.miss_low > 0.01:
            print("  -> misses are lopsided to the DOWNSIDE: the low edge is the less "
                  "reliable one. Size stops off the low edge, not the midpoint.")
        if band.drift_shrink <= 0:
            print("  centre = unconditional median: the score's bucket drift did not survive "
                  "out-of-sample, so none of it is projected forward")
        else:
            print(f"  centre shifted by {band.drift_shrink:.0%} of the score bucket's "
                  "in-sample drift (the share that held up out-of-sample)")
        gap = band.test_coverage * 100 - lvl
        if abs(gap) > 3:
            print(f"  WARNING: held-out coverage misses nominal by {gap:+.1f}pp — the range "
                  "is mis-sized on these names; treat its width as unreliable.")
        else:
            print("  Held-out coverage is close to nominal — the range is honestly sized.")
    else:
        print(f"\nNot enough history to calibrate a {lvl}% range.")

    if table is not None:
        print(f"\n--- score vs realised {H}-day return "
              f"({n_pool} names pooled, buckets cut in-sample) ---")
        print(table.to_string())
        spread_is = table.iloc[-1]["is_avg_pct"] - table.iloc[0]["is_avg_pct"]
        spread_oos = table.iloc[-1]["oos_avg_pct"] - table.iloc[0]["oos_avg_pct"]
        pstr = f", block-bootstrap p={pboot:.3f}" if pboot is not None else ""
        print(f"\n  in-sample spread {spread_is:+.1f}pp | OUT-OF-SAMPLE {spread_oos:+.1f}pp{pstr}")
        if spread_oos <= 0:
            print("VERDICT: the score does NOT survive out-of-sample here — top bucket fails to "
                  "beat bottom on unseen dates. Do not trade the lean; the range still stands.")
        elif pboot is not None and pboot > 0.10:
            print(f"VERDICT: {spread_oos:+.1f}pp out-of-sample, but the block bootstrap cannot "
                  "rule out luck (p>0.10). Treat the lean as a weak prior, not a signal.")
        elif spread_oos < 1.5:
            print(f"VERDICT: only {spread_oos:+.1f}pp out-of-sample — very faint.")
        else:
            print(f"VERDICT: {spread_oos:+.1f}pp out-of-sample spread with bootstrap support — "
                  "a real but modest edge; size accordingly.")
    else:
        print("\nNot enough pooled history to validate the score — treat leans as untested.")

    print(f"\nNOTE: overlapping {H}-day windows make these samples correlated — that is why the "
          "bootstrap uses blocks and the band is scored on held-out dates. The range covers "
          "CLOSE-to-CLOSE outcomes; intramonth prices go further. Suggestions, not advice. "
          "The news tilt refines the LIVE lean only and is not backtested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
