#!/usr/bin/env python3
"""Swing-trade stock screener using Alpaca Market Data API (IEX feed, free tier).

Targets a ~10-week (50 trading day) holding period.  NOTE: the entry rules
(ATR cap, RSI/extension bands, momentum windows) were originally calibrated for
a 4-week hold and have NOT been re-tuned for 10 weeks — revisit them if the
longer hold underperforms.  The rules are built for hit
rate rather than maximum upside: a quality company in a confirmed long-term
uptrend (above its 200-day SMA, 50-day above 200-day) with real 6-month
momentum, entered while consolidating rather than after a spike — because over
a ~1-month horizon short-term strength tends to revert — with earnings kept
outside the hold window and new longs suppressed when SPY is below its own
200-day SMA.

Prerequisites:
    pip install alpaca-py pandas numpy

Set environment variables:
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET_KEY="your_secret"

Or pass --alpaca-key and --alpaca-secret flags.

This replaces yfinance with Alpaca's StockHistoricalDataClient for reliable,
rate-limit-free daily OHLCV data.  Fundamentals still come from yfinance
(market cap, ROE, FCF) since Alpaca does not provide fundamental data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import run_report as rr


# ---------------------------------------------------------------------------
# Embedded lexicon for headline sentiment (zero extra deps)
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = frozenset({
    "beat", "beats", "beaten", "surpass", "exceed", "exceeds", "outperform",
    "outperforms", "upgrade", "upgrades", "upgraded", "buy", "strong", "bullish",
    "rally", "rallies", "rallied", "surge", "surges", "surged", "soar", "soars",
    "soared", "rocket", "rockets", "rocketed", "moon", "gain", "gains", "gained",
    "jump", "jumps", "jumped", "rise", "rises", "rose", "risen", "up", "higher",
    "high", "record", "breakthrough", "innovation", "growth", "profit", "profits",
    "profitable", "revenue", "earnings", "success", "successful", "opportunity",
    "opportunities", "momentum", "leader", "leading", "premium", "advantage",
    "recovery", "rebound", "recovers", "recovered", "rebounds", "rebounded",
    "boost", "boosts", "boosted", "expand", "expands", "expanded", "expansion",
    "partnership", "partnerships", "merger", "acquisition", "acquires", "acquired",
    "approval", "approved", "launch", "launches", "launched", "deal", "deals",
    "contract", "contracts", "dividend", "dividends", "raise", "raises", "raised",
    "target", "targets", "crush", "crushes", "crushed", "pop", "pops",
    "popped", "rip", "rips", "ripped", "explode", "explodes", "exploded",
    "double", "doubles", "triples", "triple", "quadruple", "quadruples",
    " ATH", " all-time", "52-week high", "new high", "best", "top", "outlook",
    "optimistic", "positive", "favorable", "upside", "run", "runs", "running",
})

_NEGATIVE_WORDS = frozenset({
    "miss", "misses", "missed", "loss", "losses", "lose", "loses", "lost",
    "decline", "declines", "declined", "declining", "drop", "drops", "dropped",
    "fall", "falls", "fell", "fallen", "down", "lower", "low", "downgrade",
    "downgrades", "downgraded", "sell", "weak", "weakness", "bearish", "underperform",
    "underperforms", "underperformed", "cut", "cuts", "layoff", "layoffs", "fire",
    "fired", "fires", "resign", "resigns", "resigned", "depart", "departs",
    "departed", "exit", "exits", "exited", "debt", "bankruptcy", "bankrupt",
    "investigation", "investigations", "lawsuit", "lawsuits", "recall", "recalls",
    "warning", "warnings", "risk", "risks", "concern", "concerns", "concerned",
    "trouble", "struggle", "struggles", "struggled", "plunge", "plunges", "plunged",
    "tumble", "tumbles", "tumbled", "sink", "sinks", "sank", "sunk", "slump",
    "slumps", "slumped", "crash", "crashes", "crashed", "dump", "dumps", "dumped",
    "fraud", "scandal", "scandals", "delay", "delays", "delayed", "cancel",
    "cancels", "cancelled", "canceled", "slash", "slashes", "slashed", "trim",
    "trims", "trimmed", "reduce", "reduces", "reduced", "disappoint", "disappoints",
    "disappointed", "disappointing", "warns", "warned", "alert",
    "negative", "pessimistic", "bleak", "dire", "ugly", "worst", "bottom",
    "52-week low", "new low", "bear", "bears", "short", "shorts", "shorted",
    "overvalued", "bubble", "bust", "collapse", "collapses", "collapsed",
    "crisis", "crises", "recession", "inflation", "loss-making", "unprofitable",
})


@dataclass(frozen=True)
class Candidate:
    ticker: str
    price: float
    exchange: str
    market_cap: float | None
    rsi_14: float | None
    price_vs_20d_high: float | None
    volume_ratio: float | None
    atr_14: float | None
    ema_10: float | None
    ema_20: float | None
    five_day_return: float | None
    trailing_pe: float | None
    return_on_equity: float | None
    free_cash_flow: float | None
    avg_dollar_volume: float | None
    sma_50: float | None
    days_to_earnings: float | None
    ext_vs_ema10: float | None
    close_loc: float | None
    sma_200: float | None
    six_month_return: float | None
    ext_vs_sma50: float | None
    sma200_rising: bool | None
    rs_6m: float | None
    alpha002: float | None
    alpha111: float | None
    alpha054: float | None
    rsi_2: float | None
    macd_hist_pct: float | None
    macd_line_pct: float | None


def finite_number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_tickers(path: Path) -> list[str]:
    tickers = []
    for line in path.read_text().splitlines():
        ticker = line.split("#", 1)[0].strip().upper()
        if ticker:
            tickers.append(ticker)
    return list(dict.fromkeys(tickers))


def current_nasdaq_nyse_universe() -> list[str]:
    """Download current non-ETF Nasdaq and NYSE listings from Nasdaq Trader."""
    try:
        import requests
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install requests") from error

    urls = {
        "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-stock-screen/1.0)"}
    symbols: set[str] = set()
    for market, url in urls.items():
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        for row in csv.DictReader(io.StringIO(response.text), delimiter="|"):
            symbol = (row.get("Symbol") if market == "nasdaq" else row.get("ACT Symbol")) or ""
            is_test = row.get("Test Issue") == "Y"
            is_etf = row.get("ETF") == "Y"
            is_nyse = market == "other" and row.get("Exchange") == "N"
            if is_test or is_etf or (market == "other" and not is_nyse):
                continue
            symbol = symbol.strip().upper().replace(".", "-")
            if re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", symbol):
                symbols.add(symbol)
    if not symbols:
        raise RuntimeError("Nasdaq Trader returned no usable Nasdaq/NYSE symbols")
    return sorted(symbols)


# ---------------------------------------------------------------------------
# Alpaca data client setup
# ---------------------------------------------------------------------------
def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a .env file for any keys not already set.

    Already-exported real environment variables win, so this only fills gaps.
    No dependency on python-dotenv; missing file is a silent no-op."""
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _get_alpaca_client(api_key: str | None, secret_key: str | None):
    from alpaca.data.historical.stock import StockHistoricalDataClient
    _load_dotenv()
    key = api_key or os.environ.get("ALPACA_API_KEY")
    secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Alpaca API key and secret required. Set ALPACA_API_KEY / ALPACA_SECRET_KEY "
            "(env vars or a .env file), or pass --alpaca-key / --alpaca-secret."
        )
    return StockHistoricalDataClient(key, secret)


# ---------------------------------------------------------------------------
# Bulk fetch prices from Alpaca (IEX feed, free tier)
# ---------------------------------------------------------------------------
def _latest_contiguous_segment(index: pd.DatetimeIndex, max_gap_days: int = 20) -> pd.DatetimeIndex:
    """Keep only the most recent run of bars with no gap larger than
    max_gap_days between consecutive sessions.

    A symbol later reused for an unrelated company (e.g. LB: L Brands until
    its August 2021 rename to BBWI, then reassigned to a different listing)
    shows up as a years-long gap in an otherwise near-daily series.  Without
    this, a hold measured by row position can silently span two unrelated
    companies and years of calendar time.  20 calendar days is comfortably
    above any real market closure (long weekends/holidays run 3-5 days) and
    comfortably below a delisting-driven gap."""
    if len(index) < 2:
        return index
    gaps = index.to_series().diff().dt.days
    breaks = gaps[gaps > max_gap_days].index
    return index[index >= breaks[-1]] if len(breaks) else index


def _collect_bars(df: pd.DataFrame, syms: list[str], all_bars: dict[str, pd.DataFrame]) -> int:
    """Pull each symbol's frame out of a multi-indexed (symbol, timestamp) result.
    Returns how many symbols were trimmed for a symbol-reuse gap."""
    trimmed = 0
    for sym in syms:
        try:
            sym_df = df.loc[sym].copy()
        except KeyError:
            continue
        if sym_df.empty or len(sym_df) < 25:
            continue
        sym_df.index = pd.to_datetime(sym_df.index)
        kept = _latest_contiguous_segment(sym_df.index)
        if len(kept) < len(sym_df.index):
            sym_df = sym_df.loc[kept]
            trimmed += 1
        if len(sym_df) < 25:
            continue
        all_bars[sym] = sym_df
    return trimmed


def _fetch_bars_salvage(client, syms: list[str], start, end, all_bars: dict[str, pd.DataFrame],
                        trimmed: list[int]) -> int:
    """Fetch one batch; if Alpaca rejects it for a bad symbol, split and retry so
    a single invalid symbol (share class, warrant, unit) cannot sink the batch.
    Recurses down to single symbols.  Returns the count of symbols dropped;
    accumulates the symbol-reuse-gap trim count into trimmed[0]."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    req = StockBarsRequest(
        symbol_or_symbols=syms,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment="all",
    )
    try:
        df = client.get_stock_bars(req).df
        if df is not None and not df.empty:
            trimmed[0] += _collect_bars(df, syms, all_bars)
        return 0
    except Exception:
        if len(syms) == 1:
            return 1  # genuinely invalid single symbol — drop it quietly
        mid = len(syms) // 2
        time.sleep(0.1)
        dropped = _fetch_bars_salvage(client, syms[:mid], start, end, all_bars, trimmed)
        time.sleep(0.1)
        dropped += _fetch_bars_salvage(client, syms[mid:], start, end, all_bars, trimmed)
        return dropped


# ---------------------------------------------------------------------------
# Day-scoped local cache — Alpaca bars and CNN sentiment barely change within
# a single trading day, so a second run today reuses the first run's fetch
# instead of hitting the network again.  Keyed by date + inputs, so a
# different universe/history-days/window on the same day still refetches.
# ---------------------------------------------------------------------------
CACHE_DIR = Path(".cache")


def _today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _cache_key(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _cache_load(path: Path, today: str):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        return cached["data"] if cached.get("date") == today else None
    except Exception:
        return None


def _cache_save(path: Path, today: str, data: Any) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        # Shared pruning: bar caches are 100-300MB each and the key includes
        # the exact ticker list, so stale days must be swept or .cache grows
        # without bound.
        import backtest as _bt
        removed = _bt.prune_cache(today)
        if removed:
            print(f"  -> pruned {removed} cache entries from earlier days")
        with open(path, "wb") as handle:
            pickle.dump({"date": today, "data": data}, handle)
    except Exception as exc:
        print(f"WARNING: could not write cache {path}: {exc}", file=sys.stderr)


def fetch_alpaca_bars_cached(
    tickers: list[str],
    client,
    start: datetime,
    end: datetime | None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """fetch_alpaca_bars, reused for the rest of today once fetched."""
    today = _today_str()
    key = _cache_key("bars", today, sorted(tickers), start.date(), end.date() if end else None)
    path = CACHE_DIR / f"bars_{key}.pkl"
    if not refresh:
        cached = _cache_load(path, today)
        if cached is not None:
            print(f"  -> using today's cached Alpaca bars for {len(cached)} tickers "
                  f"({path.name}; pass --refresh-cache to force a refetch)")
            return cached
    bars = fetch_alpaca_bars(tickers, client, start, end)
    _cache_save(path, today, bars)
    return bars


def cnn_sentiment_cached(history_days: int, refresh: bool = False) -> tuple[float, float, float, float]:
    """cnn_sentiment, reused for the rest of today once fetched."""
    today = _today_str()
    key = _cache_key("sentiment", today, history_days)
    path = CACHE_DIR / f"sentiment_{key}.pkl"
    if not refresh:
        cached = _cache_load(path, today)
        if cached is not None:
            print(f"  -> using today's cached CNN sentiment ({path.name}; "
                  f"pass --refresh-cache to force a refetch)")
            return cached
    result = cnn_sentiment(history_days)
    _cache_save(path, today, result)
    return result


def fetch_alpaca_bars(
    tickers: list[str],
    client,
    start: datetime,
    end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for all tickers in one or few Alpaca calls.
    Alpaca supports up to ~100 symbols per request; we chunk automatically.
    A batch containing an invalid symbol is retried in halves so the other
    ~99 valid symbols are not lost with it."""
    chunk_size = 100
    all_bars: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    show_progress = total > chunk_size
    dropped = 0
    trimmed = [0]

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        dropped += _fetch_bars_salvage(client, chunk, start, end, all_bars, trimmed)
        time.sleep(0.3)
        if show_progress:
            done = min(i + chunk_size, total)
            print(f"  Alpaca bars {done}/{total} ({done * 100 // total}%)", flush=True)

    if dropped:
        print(f"  ({dropped} symbols dropped as invalid for Alpaca — share classes/warrants/units)",
              file=sys.stderr)
    if trimmed[0]:
        print(f"  ({trimmed[0]} symbols trimmed to their most recent listing — a large gap "
              f"means the symbol was reused after an unrelated delisting)", file=sys.stderr)
    return all_bars


# ---------------------------------------------------------------------------
# Technical indicators (vectorised on DataFrames)
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Standard MACD(12,26,9) -> (line, signal, histogram), in price units.

    Normalised by close before any cross-sectional comparison — raw MACD
    scales with price level, so the unnormalised histogram would rank a $500
    stock above a $15 one regardless of relative momentum."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["rsi_2"] = compute_rsi(df["close"], 2)
    macd_line, macd_signal, macd_hist = compute_macd(df["close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["macd_hist_pct"] = macd_hist / df["close"] * 100
    df["macd_line_pct"] = macd_line / df["close"] * 100
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
    # Compare today's volume against the prior 20 sessions (exclude today so the
    # spike being measured does not dilute its own baseline).
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().shift(1)
    df["dollar_volume_20"] = (df["close"] * df["volume"]).rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    # 200-day slope: rising long-term trend distinguishes a consolidation from
    # a top rolling over.
    df["sma200_prev20"] = df["sma_200"].shift(20)
    df["5d_return"] = df["close"].pct_change(5) * 100
    # Intermediate (6-month) momentum — the horizon where the momentum factor is
    # actually documented, skipping the most recent month to sidestep the
    # well-known 1-month reversal effect.
    df["6m_return"] = (df["close"].shift(21) / df["close"].shift(126) - 1) * 100
    # How stretched price is above its 50-day SMA (extension mean-reverts over weeks).
    df["ext_vs_sma50"] = (df["close"] / df["sma_50"] - 1) * 100
    # How stretched price is above its 10-day EMA (buying extension mean-reverts).
    df["ext_vs_ema10"] = (df["close"] / df["ema_10"] - 1) * 100
    # Where the close sits in the day's range: 1.0 = closed at the high.
    day_range = df["high"] - df["low"]
    df["close_loc"] = ((df["close"] - df["low"]) / day_range).where(day_range > 0, 0.5)
    high_20 = df["close"].rolling(20).max()
    df["price_vs_20d_high"] = (df["close"] / high_20 - 1) * 100
    df = add_alpha191(df)
    return df


def add_alpha191(df: pd.DataFrame) -> pd.DataFrame:
    """Three Guotai Junan Alpha-191 factors, oriented so higher = better.

    Chosen because (a) each survived the 2018-25 CSI-300 decay study, and
    (b) each belongs to a family that also survived double-selection LASSO
    testing on the S&P 500 — volume-price interaction, short-term mean
    reversion, and volatility/price-action.  That US study aggregated signals
    to ~21 trading days, which matches this screen's 4-week hold.

    Two deliberate adaptations from the published A-share formulas: the
    price- and volume-scaled terms are normalised (by close and by 20-day
    average volume).  Raw dollar moves and raw share volume are not
    comparable across a US universe spanning $10 and $500 stocks, so the
    unnormalised versions would rank mostly by price level and volume size.
    """
    rng = (df["high"] - df["low"]).replace(0, float("nan"))
    # Close location value: +1 = closed at the high, -1 = at the low.
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng).astype(float)

    # Alpha002: -1 * DELTA(CLV, 1) — short-term reversal of intraday pressure.
    df["alpha002"] = -clv.diff(1)

    # Alpha111: SMA(V*CLV,11,2) - SMA(V*CLV,4,2) — volume-price interaction.
    # The A-share SMA(X,N,M) is an EMA with alpha = M/N.
    money_flow = df["volume"] * clv
    fast = money_flow.ewm(alpha=2 / 4, adjust=False).mean()
    slow = money_flow.ewm(alpha=2 / 11, adjust=False).mean()
    df["alpha111"] = (slow - fast) / df["volume"].rolling(20).mean()

    # Alpha054: -1 * RANK(STD(|C-O|,10) + (C-O) + CORR(C,O,10)) — the cross-
    # sectional RANK is applied later, across the candidate set.
    body = (df["close"] - df["open"]) / df["close"]
    df["alpha054"] = -(body.abs().rolling(10).std() + body
                       + df["close"].rolling(10).corr(df["open"]))
    return df


# ---------------------------------------------------------------------------
# Fundamentals from yfinance (still needed — Alpaca has none)
# ---------------------------------------------------------------------------
def _fetch_yf_info(ticker: str) -> tuple[str, dict[str, Any] | None, Exception | None]:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).get_info()
        return ticker, {
            "exchange": str(info.get("exchange") or "").upper(),
            "market_cap": finite_number(info.get("marketCap")),
            "return_on_equity": finite_number(info.get("returnOnEquity")),
            "free_cash_flow": finite_number(info.get("freeCashflow")),
            "trailing_pe": finite_number(info.get("trailingPE")),
            "next_earnings_ts": finite_number(
                info.get("earningsTimestampStart") or info.get("earningsTimestamp")
            ),
        }, None
    except Exception as exc:
        return ticker, None, exc


def fetch_info_map(tickers: list[str], max_workers: int = 10,
                   cache_path: Path | None = Path("fundamentals_cache.json"),
                   max_age_days: float = 1.0) -> dict[str, dict[str, Any]]:
    """Fetch fundamentals with a local JSON cache.

    Entries younger than max_age_days are reused without hitting Yahoo (a big
    speed-up on reruns).  When a fresh fetch fails — e.g. Yahoo rate-limiting —
    a stale cache entry is still used as a fallback, so one bad Yahoo day
    cannot zero out the whole screen.  Default max_age_days=1 refreshes once
    per calendar day, same cadence as the Alpaca bars cache."""
    now = datetime.now(timezone.utc)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    out: dict[str, dict[str, Any]] = {}
    to_fetch: list[str] = []
    for t in tickers:
        entry = cache.get(t)
        if entry and max_age_days > 0:
            try:
                age = (now - datetime.fromisoformat(entry["cached_at"])).total_seconds() / 86400
            except Exception:
                age = None
            if age is not None and age <= max_age_days:
                out[t] = entry["data"]
                continue
        to_fetch.append(t)
    if out:
        print(f"  -> {len(out)} fundamentals from cache (≤{max_age_days:.0f}d old)")

    total = len(to_fetch)
    if total:
        print(f"Fetching fundamentals for {total} tickers (yfinance, {max_workers} workers) …")
        done = 0
        step = max(1, total // 100)  # report roughly every 1%
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_yf_info, t): t for t in to_fetch}
            for fut in as_completed(futs):
                t, data, err = fut.result()
                if data is not None:
                    out[t] = data
                    cache[t] = {"cached_at": now.isoformat(), "data": data}
                done += 1
                if done % step == 0 or done == total:
                    print(f"  fundamentals {done}/{total} ({done * 100 // total}%) — {len(out)} usable",
                          flush=True)

        # Stale-cache fallback for tickers whose fresh fetch failed.
        stale_used = 0
        for t in to_fetch:
            if t not in out and t in cache:
                out[t] = cache[t]["data"]
                stale_used += 1
        if stale_used:
            print(f"  -> reused {stale_used} stale cache entries for tickers Yahoo failed on")

    if cache_path:
        try:
            cache_path.write_text(json.dumps(cache))
        except Exception as exc:
            print(f"WARNING: could not write {cache_path}: {exc}", file=sys.stderr)

    print(f"  -> fundamentals for {len(out)} tickers")
    return out


# ---------------------------------------------------------------------------
# Build Candidate from pre-fetched DataFrame
# ---------------------------------------------------------------------------
def build_candidate(ticker: str, df: pd.DataFrame, info: dict[str, Any],
                    spy_6m_return: float | None = None) -> Candidate | None:
    if df.empty or len(df) < 25:
        return None
    latest = df.iloc[-1]
    price = finite_number(latest.get("close"))
    if not price or price <= 0:
        return None

    earnings_ts = info.get("next_earnings_ts")
    days_to_earnings = None
    if earnings_ts is not None:
        delta = datetime.fromtimestamp(earnings_ts, tz=timezone.utc) - datetime.now(timezone.utc)
        days_to_earnings = delta.total_seconds() / 86400.0

    return Candidate(
        ticker=ticker,
        price=price,
        exchange=info.get("exchange", ""),
        market_cap=info.get("market_cap"),
        rsi_14=finite_number(latest.get("rsi_14")),
        price_vs_20d_high=finite_number(latest.get("price_vs_20d_high")),
        volume_ratio=finite_number(latest.get("vol_ratio")),
        atr_14=finite_number(latest.get("atr_14")),
        ema_10=finite_number(latest.get("ema_10")),
        ema_20=finite_number(latest.get("ema_20")),
        five_day_return=finite_number(latest.get("5d_return")),
        trailing_pe=info.get("trailing_pe"),
        return_on_equity=info.get("return_on_equity"),
        free_cash_flow=info.get("free_cash_flow"),
        avg_dollar_volume=finite_number(latest.get("dollar_volume_20")),
        sma_50=finite_number(latest.get("sma_50")),
        days_to_earnings=days_to_earnings,
        ext_vs_ema10=finite_number(latest.get("ext_vs_ema10")),
        close_loc=finite_number(latest.get("close_loc")),
        sma_200=finite_number(latest.get("sma_200")),
        six_month_return=finite_number(latest.get("6m_return")),
        ext_vs_sma50=finite_number(latest.get("ext_vs_sma50")),
        sma200_rising=(
            finite_number(latest.get("sma_200")) > finite_number(latest.get("sma200_prev20"))
            if finite_number(latest.get("sma_200")) is not None
            and finite_number(latest.get("sma200_prev20")) is not None else None
        ),
        rs_6m=(
            finite_number(latest.get("6m_return")) - spy_6m_return
            if finite_number(latest.get("6m_return")) is not None
            and spy_6m_return is not None else None
        ),
        alpha002=finite_number(latest.get("alpha002")),
        alpha111=finite_number(latest.get("alpha111")),
        alpha054=finite_number(latest.get("alpha054")),
        rsi_2=finite_number(latest.get("rsi_2")),
        macd_hist_pct=finite_number(latest.get("macd_hist_pct")),
        macd_line_pct=finite_number(latest.get("macd_line_pct")),
    )


# ---------------------------------------------------------------------------
# Screening rules
# ---------------------------------------------------------------------------
def buy_rule(candidate: Candidate, args: argparse.Namespace,
             fundamentals_available: bool = True) -> list[str]:
    """Rules for a ~4-week (20 trading day) hold.

    Structure: a quality company in a confirmed long-term uptrend, entered on a
    consolidation rather than a spike.  The horizon matters — over ~1 month the
    documented effect is short-term *reversal*, so chasing recent strength hurts,
    while 6-month momentum and the 200-day trend are what persist.

    fundamentals_available=False (total yfinance outage) skips the checks that
    need Yahoo data — exchange, market cap, FCF, ROE — rather than rejecting
    the whole universe; the technical rules still apply in full.
    """
    failures = []

    # --- Listing, size, liquidity -----------------------------------------
    if fundamentals_available:
        if candidate.exchange not in {"NMS", "NGM", "NCM", "NAS", "NYQ", "NYS"}:
            failures.append(f"not a recognised Nasdaq/NYSE exchange ({candidate.exchange or 'missing'})")
        if candidate.market_cap is None or candidate.market_cap <= args.min_market_cap:
            failures.append(f"market cap not above ${args.min_market_cap:,.0f}")
    if candidate.price < args.min_price:
        failures.append(f"price ${candidate.price:.2f} below ${args.min_price:.2f} floor")
    if candidate.avg_dollar_volume is None or candidate.avg_dollar_volume < args.min_dollar_volume:
        have = f"${candidate.avg_dollar_volume:,.0f}" if candidate.avg_dollar_volume is not None else "missing"
        failures.append(f"avg daily $ volume {have} below ${args.min_dollar_volume:,.0f}")

    # --- Long-term trend: the single most reliable win-rate filter ---------
    if args.above_200d_sma:
        if candidate.sma_200 is None:
            failures.append("missing 200-day SMA (needs ~1y of history)")
        elif candidate.price <= candidate.sma_200:
            failures.append("price not above 200-day SMA (not a long-term uptrend)")
    if args.require_golden_cross:
        if candidate.sma_50 is None or candidate.sma_200 is None:
            failures.append("missing SMA data for trend structure")
        elif candidate.sma_50 <= candidate.sma_200:
            failures.append("50-day SMA below 200-day SMA (trend structure not confirmed)")

    if args.require_rising_200d:
        if candidate.sma200_rising is None:
            failures.append("missing 200-day SMA slope")
        elif not candidate.sma200_rising:
            failures.append("200-day SMA falling (top rolling over, not a consolidation)")

    # --- Intermediate momentum (the factor that actually persists) ---------
    if candidate.six_month_return is None:
        failures.append("missing 6-month return")
    elif candidate.six_month_return < args.min_6m_return:
        failures.append(
            f"6-month return {candidate.six_month_return:.1f}% below {args.min_6m_return:.1f}%"
        )
    # Relative strength: momentum is a cross-sectional effect — beating the
    # market matters, not the absolute number.  Skipped when SPY data missing.
    if candidate.rs_6m is not None and candidate.rs_6m < args.min_rs_6m:
        failures.append(
            f"6-month return trails SPY by {-candidate.rs_6m:.1f}% "
            f"(need ≥{args.min_rs_6m:+.1f}% relative strength)"
        )

    # --- Reversal guards: do not buy a stretched or spiking chart ----------
    if candidate.ext_vs_sma50 is None:
        failures.append("missing 50-day SMA extension")
    elif candidate.ext_vs_sma50 > args.max_ext_sma50:
        failures.append(
            f"price {candidate.ext_vs_sma50:.1f}% above 50-day SMA "
            f"(overextended, need ≤{args.max_ext_sma50:.1f}%)"
        )
    elif candidate.ext_vs_sma50 < args.min_ext_sma50:
        failures.append(
            f"price {candidate.ext_vs_sma50:.1f}% below 50-day SMA "
            f"(breakdown, not a consolidation; need ≥{args.min_ext_sma50:.1f}%)"
        )
    if candidate.five_day_return is None:
        failures.append("missing 5-day return")
    elif candidate.five_day_return > args.max_5d_return:
        failures.append(
            f"5-day return {candidate.five_day_return:.1f}% above {args.max_5d_return:.1f}% "
            f"(recent spike — 1-month reversal risk)"
        )

    # --- Entry zone: strong but not overbought -----------------------------
    if candidate.rsi_14 is None:
        failures.append("missing RSI(14)")
    elif not (args.min_rsi <= candidate.rsi_14 <= args.max_rsi):
        failures.append(f"RSI(14) {candidate.rsi_14:.1f} not in [{args.min_rsi:.1f}, {args.max_rsi:.1f}]")

    # --- Volatility: lower vol carries higher hit rates --------------------
    if candidate.atr_14 is None:
        failures.append("missing ATR(14)")
    else:
        atr_pct = candidate.atr_14 / candidate.price * 100
        if atr_pct < args.min_atr_pct:
            failures.append(f"ATR {atr_pct:.1f}% below {args.min_atr_pct:.1f}% (too inert)")
        elif atr_pct > args.max_atr_pct:
            failures.append(f"ATR {atr_pct:.1f}% above {args.max_atr_pct:.1f}% (too volatile for a 4-week hold)")

    # --- Event risk: earnings inside the hold window is the biggest gap risk
    if args.earnings_blackout_days > 0 and candidate.days_to_earnings is not None:
        if 0 <= candidate.days_to_earnings <= args.earnings_blackout_days:
            failures.append(
                f"earnings in ~{candidate.days_to_earnings:.0f}d within the "
                f"{args.earnings_blackout_days}d hold window"
            )

    # --- Quality: matters more the longer you hold -------------------------
    if fundamentals_available:
        if args.require_positive_fcf:
            if candidate.free_cash_flow is None or candidate.free_cash_flow <= 0:
                failures.append("free cash flow not positive (quality backstop)")
        if args.require_roe is not None and args.require_roe > 0:
            if candidate.return_on_equity is None or candidate.return_on_equity < args.require_roe:
                failures.append(f"ROE below {args.require_roe:.0%} (quality backstop)")

    # --- Optional extras (disabled by default) -----------------------------
    if args.min_volume_ratio > 0:
        if candidate.volume_ratio is None or candidate.volume_ratio < args.min_volume_ratio:
            failures.append(f"volume ratio below {args.min_volume_ratio:.2f}x")
    if args.min_close_strength > 0:
        if candidate.close_loc is None or candidate.close_loc < args.min_close_strength:
            failures.append(f"weak close (need ≥{args.min_close_strength:.0%} of day range)")
    return failures


# ---------------------------------------------------------------------------
# CNN Fear & Greed
# ---------------------------------------------------------------------------
def cnn_sentiment(history_days: int) -> tuple[float, float, float, float]:
    try:
        import requests
    except ModuleNotFoundError as error:
        raise RuntimeError("Missing dependency: install requests") from error

    try:
        response = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-stock-screen/1.0)"},
        )
        response.raise_for_status()
        payload = response.json()
        current = finite_number(payload.get("fear_and_greed", {}).get("score"))
        historical = payload.get("fear_and_greed_historical", {})
        points = historical.get("data", historical) if isinstance(historical, dict) else historical
        scores = [
            score for point in points if isinstance(point, dict)
            if (score := finite_number(point.get("y", point.get("score")))) is not None
        ]
        if current is None or not scores:
            raise RuntimeError("CNN response did not contain usable current and historical scores")
    except (requests.RequestException, RuntimeError) as error:
        print(f"WARNING: CNN endpoint unavailable ({error}); using public archive", file=sys.stderr)
        archive = requests.get(
            "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/datasets/cnn_fear_greed.csv",
            timeout=30,
        )
        archive.raise_for_status()
        dated_scores: dict[str, float] = {}
        for row in csv.DictReader(archive.text.splitlines()):
            score = finite_number(row.get("Fear Greed"))
            if score is not None:
                dated_scores[row["Date"]] = score
        if not dated_scores:
            raise RuntimeError("Fear & Greed archive did not contain usable scores")
        scores = [dated_scores[day] for day in sorted(dated_scores)]
        current = scores[-1]

    window = scores[-history_days:]
    greed_average = sum(window) / len(window)
    return current, greed_average, 100 - current, 100 - greed_average


def spy_regime_risk_off(spy_df: pd.DataFrame | None, window: int = 200) -> tuple[bool, str] | None:
    """Market-regime gate: SPY must be above a RISING 200-day SMA.

    Price above a falling 200-day is the classic bear-market-rally trap (the
    2022 windows that produced 30%-win picks in testing), so the slope
    condition matters as much as the level.  Returns (risk_off, reason), or
    None when there is not enough SPY history (caller should skip)."""
    if spy_df is None or len(spy_df) < window + 20:
        return None
    sma_series = spy_df["close"].rolling(window).mean()
    sma = float(sma_series.iloc[-1])
    sma_prev = float(sma_series.iloc[-21])
    close = float(spy_df["close"].iloc[-1])
    if close <= sma:
        return True, f"SPY {close:.2f} at/below its {window}-day SMA {sma:.2f} (risk-off regime)"
    if sma <= sma_prev:
        return True, (f"SPY above but its {window}-day SMA is falling "
                      f"({sma:.2f} ≤ {sma_prev:.2f} 20 sessions ago) — bear-rally risk")
    return False, f"SPY {close:.2f} above its rising {window}-day SMA {sma:.2f}"


def _taper(excess_pct: float, start: float, full: float, floor: float) -> float:
    """1.0 until excess_pct reaches start, then linearly down to floor at full."""
    if excess_pct <= start:
        return 1.0
    t = min(1.0, (excess_pct - start) / max(full - start, 1e-9))
    return 1.0 - t * (1.0 - floor)


def sentiment_scale(args: argparse.Namespace) -> tuple[float, float, str]:
    """Continuous position-size weight from CNN Fear & Greed.

    Replaces the old binary BUY_ALLOWED / SELL_RISK_OFF gate.  A stretched
    tape is a reason to commit less capital, not to skip an otherwise good
    setup entirely — the binary gate threw away whole rebalance dates.

    Both extremes taper the weight down: greed above --sentiment-taper-start-pct
    over its average (crowded, reversal-prone) and panic the same distance
    below its average (disorderly tape).  Normal conditions score 1.0.

    Returns (scale, score_penalty, reason).  score_penalty maps the weight onto
    the same 0..-10 point range select_top15 uses for the news penalty, so the
    two are directly comparable when ranking.

    NOTE: the rules as specified only ever scale DOWN, so scale lands in
    [floor, 1.0].  The cap above 1.0 exists so a future "size up when the tape
    is calm" rule has a defined ceiling; nothing here reaches it today."""
    greed_excess = (args.greed - args.greed_average) / args.greed_average * 100
    panic_deficit = (args.panic_average - args.panic) / args.panic_average * 100

    g = _taper(greed_excess, args.sentiment_taper_start_pct,
               args.sentiment_taper_full_pct, args.sentiment_scale_floor)
    p = _taper(panic_deficit, args.sentiment_taper_start_pct,
               args.sentiment_taper_full_pct, args.sentiment_scale_floor)
    scale = max(args.sentiment_scale_floor, min(args.sentiment_scale_cap, min(g, p)))

    # Report whichever extreme actually binds.  Branch on the scale, not on
    # g vs p: greed and panic are near-mirror series, so they frequently taper
    # by the SAME amount and a g/p comparison then falls through to the
    # "within range" message while the weight is well below 1.0.
    if scale >= 1.0:
        reason = (f"sentiment within {args.sentiment_taper_start_pct:.0f}% of average "
                  f"-> full size x{scale:.2f}")
    elif g <= p:
        reason = (f"greed {args.greed:.1f} is {greed_excess:+.0f}% vs its average "
                  f"{args.greed_average:.1f} -> size x{scale:.2f}")
    else:
        reason = (f"panic {args.panic:.1f} is {-panic_deficit:+.0f}% vs its average "
                  f"{args.panic_average:.1f} -> size x{scale:.2f}")
    # Weight 1.0 -> 0 points, floor (0.5) -> -10 points.
    span = max(1.0 - args.sentiment_scale_floor, 1e-9)
    penalty = round(-10.0 * (1.0 - scale) / span, 2)
    return round(scale, 3), penalty, reason


def news_penalty(positive_pct: float | None, start_pct: float, max_points: float) -> float:
    """Score penalty (0 to -max_points) for a negative-skewed news flow.

    Replaces the old hard reject: a name with ugly headlines is downweighted,
    not eliminated, so an exceptional technical setup can still overrule it.
    Zero until negative share reaches start_pct, then linear to -max_points at
    100% negative."""
    if positive_pct is None:
        return 0.0
    negative_pct = 100.0 - positive_pct
    if negative_pct <= start_pct:
        return 0.0
    t = (negative_pct - start_pct) / max(100.0 - start_pct, 1e-9)
    return round(-max_points * min(1.0, t), 2)


def sentiment_action(args: argparse.Namespace) -> tuple[str, str]:
    greed_limit = args.greed_average * (1 + args.greed_above_average_pct / 100)
    panic_limit = args.panic_average * (1 - args.panic_below_average_pct / 100)
    if args.greed > greed_limit:
        return ("SELL", f"greed {args.greed:.1f} exceeds {args.greed_above_average_pct:.0f}% above its average ({greed_limit:.1f})")
    if args.panic < panic_limit:
        return ("SELL", f"panic {args.panic:.1f} is more than {args.panic_below_average_pct:.0f}% below its average ({panic_limit:.1f})")
    return ("BUY", "sentiment did not trigger either market-wide sell rule")


# ---------------------------------------------------------------------------
# News sentiment (contrarian gate)
# ---------------------------------------------------------------------------
def _headline_sentiment(title: str) -> int:
    lowered = title.lower()
    pos_hits = sum(1 for w in _POSITIVE_WORDS if w in lowered)
    neg_hits = sum(1 for w in _NEGATIVE_WORDS if w in lowered)
    if pos_hits > neg_hits:
        return 1
    if neg_hits > pos_hits:
        return -1
    return 0


def _article_title(item: dict[str, Any]) -> str:
    """Pull the headline across yfinance's old and new news schemas.

    Older yfinance put the title at the top level; newer versions nest the
    article body under item['content']."""
    title = item.get("title") or item.get("headline")
    if not title and isinstance(item.get("content"), dict):
        title = item["content"].get("title")
    return title or ""


def _article_age_days(item: dict[str, Any], now: datetime) -> float | None:
    """Age of a yfinance news item in days, or None when no timestamp is present.

    Handles the old schema (providerPublishTime, unix seconds) and the new one
    (content.pubDate / content.displayTime, ISO 8601)."""
    ts = item.get("providerPublishTime")
    if ts is not None:
        try:
            pub = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            return max(0.0, (now - pub).total_seconds() / 86400.0)
        except (TypeError, ValueError, OSError, OverflowError):
            pass
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content:
        for key in ("pubDate", "displayTime"):
            raw = content.get(key)
            if not raw:
                continue
            try:
                pub = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                return max(0.0, (now - pub).total_seconds() / 86400.0)
            except ValueError:
                continue
    return None


def check_news_sentiment(ticker: str, max_articles: int = 15,
                         halflife_days: float = 3.0) -> tuple[bool, str, float | None]:
    """Recency-weighted directional news gate for a momentum strategy.

    Returns (should_buy, reason, positive_pct).  positive_pct is the
    *recency-weighted* share of directional (non-neutral) headlines that are
    positive, or None when there are no directional headlines to judge — each
    headline's ±1 sign is weighted by 0.5 ** (age_days / halflife_days) so a
    fresh headline moves the flow far more than a stale one (a week-old scare is
    largely priced in).  A headline with no timestamp gets full weight.  The
    caller downweights names whose weighted news flow skews negative."""
    try:
        import yfinance as yf
    except ModuleNotFoundError:
        return (True, "yfinance missing, skipping news check", None)
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return (True, "no news available", None)
    except Exception as exc:
        return (True, f"news fetch failed: {exc}", None)

    recent = news[:max_articles]
    now = datetime.now(timezone.utc)
    pos_w = neg_w = 0.0
    positive = negative = neutral = parsed = 0
    for item in recent:
        title = _article_title(item)
        if not title:
            continue
        parsed += 1
        age = _article_age_days(item, now)
        weight = 0.5 ** (age / halflife_days) if (age is not None and halflife_days > 0) else 1.0
        sign = _headline_sentiment(title)
        if sign > 0:
            pos_w += weight
            positive += 1
        elif sign < 0:
            neg_w += weight
            negative += 1
        else:
            neutral += 1
    if not parsed:
        return (True, "no parseable headlines", None)

    directional = pos_w + neg_w
    positive_pct = (pos_w / directional * 100) if directional > 0 else None
    reason = (f"news: recency-weighted {pos_w:.1f}+/{neg_w:.1f}- "
              f"({positive}+{negative}-{neutral}0 raw over {parsed} articles)")
    return (True, reason, positive_pct)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _momentum_score(row: dict[str, Any]) -> float:
    """Coarse ordering for the candidates CSV — select_top15.py does the
    real multi-factor ranking.  Rewards 6-month momentum (capped, since the
    hottest names revert) and penalises extension above the 50-day SMA."""
    s = 0.0
    if row.get("six_month_return") is not None:
        s += min(row["six_month_return"], 60.0)
    if row.get("ext_vs_sma50") is not None:
        s -= max(row["ext_vs_sma50"], 0.0)
    return s


def promote_near_misses(candidates: list[dict[str, Any]], rejected: list[dict[str, Any]],
                        min_candidates: int) -> int:
    """Top the candidate list up to min_candidates with the closest rejects.

    Near-misses are rejects with full data (early rejects with no price are
    excluded), ordered by fewest failed rules, then momentum score.  Promoted
    rows keep their failure reasons and are labelled NEAR_MISS — they did NOT
    pass the screen, and the label must survive into the CSV so nothing
    downstream mistakes them for full candidates.  Returns how many were moved."""
    def fail_count(row: dict[str, Any]) -> int:
        return str(row.get("selection_reason", "")).count(";") + 1

    pool = [r for r in rejected if r.get("price") is not None]
    pool.sort(key=lambda r: (fail_count(r), -_momentum_score(r)))
    moved = 0
    for row in pool[:max(0, min_candidates - len(candidates))]:
        row["company_selection"] = "NEAR_MISS"
        row["combined_research_action"] = "NEAR_MISS_REVIEW"
        rejected.remove(row)
        candidates.append(row)
        moved += 1
    return moved


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "ticker", "price", "market_cap", "exchange",
        "rsi_14", "price_vs_20d_high", "volume_ratio", "atr_14", "atr_pct",
        "ext_vs_ema10", "close_loc", "sma_50", "sma_200", "six_month_return", "ext_vs_sma50",
        "rs_6m", "sma200_rising", "rsi_2", "macd_hist_pct", "macd_line_pct",
        "alpha002", "alpha111", "alpha054",
        "ema_10", "ema_20", "five_day_return", "avg_dollar_volume", "days_to_earnings",
        "trailing_pe", "return_on_equity", "free_cash_flow",
        "company_selection", "selection_reason",
        "market_overlay", "market_reason",
        "sentiment_scale", "sentiment_penalty", "news_penalty", "net_sentiment_adjustment",
        "news_sentiment_pct", "news_gate_reason",
        "combined_research_action", "screened_at_utc",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", type=Path, nargs="?", help="Text file with one ticker per line")
    parser.add_argument("--universe", choices=["nasdaq-nyse"],
                        help="Download the current non-ETF Nasdaq/NYSE universe")
    parser.add_argument("--max-tickers", type=int,
                        help="Optional cap for a trial run; applied alphabetically")

    # Alpaca credentials
    parser.add_argument("--alpaca-key", type=str, default=None,
                        help="Alpaca API key (or set ALPACA_API_KEY env var)")
    parser.add_argument("--alpaca-secret", type=str, default=None,
                        help="Alpaca API secret (or set ALPACA_SECRET_KEY env var)")
    parser.add_argument("--alpaca-history-days", type=int, default=420,
                        help="Calendar days of price history to fetch; needs ~1y so the "
                             "200-day SMA and 6-month momentum are defined (default: 420)")

    # --- Trend (core win-rate filters for a 4-week hold) -------------------
    parser.add_argument("--above-200d-sma", action=argparse.BooleanOptionalAction, default=True,
                        help="Require close above the 200-day SMA (default: on)")
    parser.add_argument("--require-golden-cross", action=argparse.BooleanOptionalAction, default=True,
                        help="Require 50-day SMA above 200-day SMA (default: on)")
    parser.add_argument("--min-6m-return", type=float, default=5.0,
                        help="Minimum 6-month return %%, skipping the last month — the "
                             "horizon where momentum actually persists (default: 5)")
    parser.add_argument("--require-rising-200d", action=argparse.BooleanOptionalAction, default=True,
                        help="Require the stock's own 200-day SMA to be rising (default: on)")
    parser.add_argument("--min-rs-6m", type=float, default=0.0,
                        help="Minimum 6-month return relative to SPY, %% points — momentum is "
                             "a cross-sectional effect (default: 0 = must beat SPY)")

    # --- Reversal guards ---------------------------------------------------
    parser.add_argument("--max-ext-sma50", type=float, default=15.0,
                        help="Maximum %% price may stretch above its 50-day SMA (default: 15). "
                             "A wider 25%% band lifted ab_sweep's per-pick avg ROI but hurt "
                             "the portfolio backtest (admits more extended, riskier names), "
                             "so the tight band stands.")
    parser.add_argument("--min-ext-sma50", type=float, default=-3.0,
                        help="Minimum %% vs the 50-day SMA — a consolidation holds near the "
                             "50-day; deep breakdowns are knife-catches (default: -3; widening "
                             "to -8 cost ~3pp of backtest excess)")
    parser.add_argument("--max-5d-return", type=float, default=8.0,
                        help="Reject recent spikes above this 5-day return %%; over ~1 month "
                             "short-term strength tends to revert (default: 8)")

    # --- Entry zone / volatility -------------------------------------------
    parser.add_argument("--min-rsi", type=float, default=40.0,
                        help="Lower RSI bound; allows buying a pullback inside an uptrend "
                             "rather than only breakouts (default: 40)")
    parser.add_argument("--max-rsi", type=float, default=65.0,
                        help="Upper RSI bound; overbought entries fade (default: 65)")
    parser.add_argument("--min-atr-pct", type=float, default=1.0,
                        help="Minimum ATR(14) as %% of price (default: 1.0)")
    parser.add_argument("--max-atr-pct", type=float, default=6.0,
                        help="Maximum ATR(14) as %% of price — lower-volatility names carry "
                             "higher hit rates (default: 6.0)")

    # --- Liquidity / size ---------------------------------------------------
    parser.add_argument("--min-price", type=float, default=10.0,
                        help="Minimum share price (default: 10.0)")
    parser.add_argument("--min-dollar-volume", type=float, default=10_000_000,
                        help="Minimum 20-day average daily dollar volume (default: 10,000,000)")

    # --- Event risk ---------------------------------------------------------
    parser.add_argument("--earnings-blackout-days", type=int, default=28,
                        help="Reject if earnings fall within this many days — defaults to the "
                             "whole 4-week hold window (0 disables; default: 28)")

    # --- Market regime ------------------------------------------------------
    parser.add_argument("--market-regime", action=argparse.BooleanOptionalAction, default=True,
                        help="Force risk-off when SPY is at/below its 200-day SMA; disable "
                             "with --no-market-regime")

    # --- Optional extras (off by default; kept for experimentation) ---------
    parser.add_argument("--min-volume-ratio", type=float, default=0.0,
                        help="Optional volume-spike floor, 0 disables (default: 0)")
    parser.add_argument("--min-close-strength", type=float, default=0.0,
                        help="Optional closing-strength floor 0-1, 0 disables (default: 0)")

    # Quality backstop
    parser.add_argument("--min-market-cap", type=float, default=2_000_000_000,
                        help="Minimum market cap; larger caps carry fewer blow-ups "
                             "(default: 2,000,000,000)")
    parser.add_argument("--require-positive-fcf", action=argparse.BooleanOptionalAction, default=True,
                        help="Require positive free cash flow (default: on)")
    parser.add_argument("--require-roe", type=float, default=0.10,
                        help="Minimum return on equity; quality matters more over a 4-week "
                             "hold (default: 0.10; pass 0 or a negative to disable)")

    # Sentiment overlay
    parser.add_argument("--greed", type=float)
    parser.add_argument("--greed-average", type=float)
    parser.add_argument("--panic", type=float)
    parser.add_argument("--panic-average", type=float)
    parser.add_argument("--greed-above-average-pct", type=float, default=30.0)
    parser.add_argument("--panic-below-average-pct", type=float, default=35.0)
    # Continuous position-size weight from sentiment (replaces the binary gate).
    parser.add_argument("--sentiment-taper-start-pct", type=float, default=20.0,
                        help="Sentiment this %% away from its average starts shrinking the "
                             "position-size weight (default: 20)")
    parser.add_argument("--sentiment-taper-full-pct", type=float, default=60.0,
                        help="Distance from average at which the weight hits its floor "
                             "(default: 60)")
    parser.add_argument("--sentiment-scale-floor", type=float, default=0.5,
                        help="Smallest position-size weight (default: 0.5)")
    parser.add_argument("--sentiment-scale-cap", type=float, default=1.5,
                        help="Largest position-size weight (default: 1.5)")
    parser.add_argument("--sentiment-gate", action=argparse.BooleanOptionalAction, default=False,
                        help="Restore the old binary SELL_RISK_OFF sentiment gate instead of "
                             "the continuous weight (default: off — weight only)")
    parser.add_argument("--cnn-sentiment", action="store_true")
    parser.add_argument("--sentiment-history-days", type=int, default=252)

    # News contrarian gate
    parser.add_argument("--news-check", action="store_true")
    parser.add_argument("--news-halflife-days", type=float, default=3.0,
                        help="Recency half-life for the weighted news algorithm: each headline's "
                             "sentiment is weighted by 0.5**(age_days/halflife), so a fresh "
                             "headline outweighs a stale one (default: 3; 0 disables weighting)")
    parser.add_argument("--news-penalty-start-pct", type=float, default=50.0,
                        help="Negative-headline share at which the news score penalty starts "
                             "(default: 50)")
    parser.add_argument("--news-penalty-max", type=float, default=10.0,
                        help="Score points deducted at 100%% negative headlines (default: 10)")
    parser.add_argument("--news-negative-threshold", type=float, default=None,
                        help="Legacy HARD REJECT above this %% negative headlines. Default off "
                             "— news now applies a score penalty instead of eliminating a name.")
    parser.add_argument("--news-max-articles", type=int, default=15)

    parser.add_argument("--yf-workers", type=int, default=10,
                        help="Parallel workers for yfinance fundamentals (default: 10; "
                             "raising much higher risks Yahoo rate-limiting)")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Ignore today's cached Alpaca bars / CNN sentiment and refetch "
                             "(cache normally makes a same-day rerun skip both network calls)")

    parser.add_argument("--min-candidates", type=int, default=5,
                        help="If fewer names pass every rule, top the list up to this many "
                             "with the closest rejects, labelled NEAR_MISS (0 disables; default: 5)")
    parser.add_argument("--candidates-output", type=Path, default=Path("short_term_candidates.csv"))
    parser.add_argument("--rejected-output", type=Path, default=Path("rejected_companies.csv"))
    parser.add_argument("--ticker-output", type=Path, default=Path("screener_universe.txt"),
                        help="Write the screened universe (one symbol per line) here so "
                             "backtest.py can reuse the exact same list (pass '' to disable)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    rr.begin("screener", args)
    if bool(args.tickers) == bool(args.universe):
        raise SystemExit("Provide either a ticker file or --universe nasdaq-nyse, but not both.")
    if args.max_tickers is not None and args.max_tickers < 1:
        raise SystemExit("--max-tickers must be positive.")

    manual_sentiment = (args.greed, args.greed_average, args.panic, args.panic_average)
    if args.cnn_sentiment:
        if any(value is not None for value in manual_sentiment):
            raise SystemExit("Use either --cnn-sentiment or all four manual sentiment values, not both.")
        if args.sentiment_history_days < 1:
            raise SystemExit("--sentiment-history-days must be positive.")
        args.greed, args.greed_average, args.panic, args.panic_average = cnn_sentiment_cached(
            args.sentiment_history_days, refresh=args.refresh_cache
        )
    elif any(value is None for value in manual_sentiment):
        raise SystemExit(
            "Provide --greed, --greed-average, --panic, and --panic-average, "
            "or use --cnn-sentiment."
        )
    if min(args.greed_average, args.panic_average) <= 0:
        raise SystemExit("Sentiment averages must be positive.")
    if not 0 <= args.panic_below_average_pct < 100:
        raise SystemExit("--panic-below-average-pct must be in [0, 100).")

    # Continuous size weight is the default; the binary gate is opt-in legacy.
    size_scale, size_penalty, size_reason = sentiment_scale(args)
    if args.sentiment_gate:
        market_action, market_reason = sentiment_action(args)
    else:
        market_action, market_reason = "BUY", size_reason
    print(f"Sentiment position-size weight: x{size_scale:.2f} ({size_reason})")

    tickers = current_nasdaq_nyse_universe() if args.universe else read_tickers(args.tickers)
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]

    # Path("") normalises to Path("."), which is truthy — so the documented
    # "pass '' to disable" needs an explicit check or we try to write the cwd.
    if args.ticker_output is not None and str(args.ticker_output) not in ("", "."):
        args.ticker_output.write_text("\n".join(tickers) + "\n")
        print(f"Wrote {len(tickers)} screened symbols to {args.ticker_output}")

    print(f"Screening {len(tickers)} symbols")
    t0 = time.perf_counter()

    # --- Phase 1: fetch all price data from Alpaca in bulk ---
    client = _get_alpaca_client(args.alpaca_key, args.alpaca_secret)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.alpaca_history_days)

    spy_df = fetch_alpaca_bars_cached(["SPY"], client, start, end, refresh=args.refresh_cache).get("SPY")
    spy_6m_return = None
    if spy_df is not None and len(spy_df) >= 127:
        c = spy_df["close"]
        spy_6m_return = float((c.iloc[-22] / c.iloc[-127] - 1) * 100)
    else:
        print("WARNING: not enough SPY history — relative-strength rule skipped this run",
              file=sys.stderr)
    if args.market_regime:
        regime = spy_regime_risk_off(spy_df)
        if regime is None:
            print("WARNING: not enough SPY history for the market-regime gate; skipping it",
                  file=sys.stderr)
        elif regime[0]:
            market_action, market_reason = "SELL", regime[1]
        else:
            market_reason = f"{market_reason}; {regime[1]}"

    print(f"Fetching {args.alpaca_history_days} days of Alpaca daily bars …")
    raw_bars = fetch_alpaca_bars_cached(tickers, client, start, end, refresh=args.refresh_cache)
    if not raw_bars:
        raise SystemExit("No price data returned from Alpaca.")
    print(f"  -> bars for {len(raw_bars)} tickers")

    # --- Phase 2: compute technicals ---
    print("Computing technical signals …")
    enriched: dict[str, pd.DataFrame] = {}
    too_short = 0
    for ticker, df in raw_bars.items():
        # Names without ~200 bars can never pass (no 200-day SMA / 6-month
        # momentum), so drop them now instead of wasting slow yfinance calls.
        if len(df) < 200:
            too_short += 1
            continue
        enriched[ticker] = enrich_df(df)
    if too_short:
        print(f"  -> skipped {too_short} tickers with <200 bars of history (cannot pass the trend rules)")

    # --- Phase 3: fetch fundamentals (yfinance, limited parallel) ---
    info_map = fetch_info_map(list(enriched.keys()), max_workers=args.yf_workers,
                              max_age_days=0 if args.refresh_cache else 1.0)
    fundamentals_available = bool(info_map)
    if not fundamentals_available:
        print("WARNING: fundamentals unavailable for EVERY ticker (Yahoo outage/rate limit?) — "
              "screening on technicals only; exchange, market-cap, FCF and ROE gates skipped "
              "this run.", file=sys.stderr)

    # --- Phase 4: screen ---
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for ticker, df in enriched.items():
        info = info_map.get(ticker)
        if info is None and not fundamentals_available:
            info = {}
        if info is None:
            rejected.append({
                "ticker": ticker, "price": None, "market_cap": None, "exchange": None,
                "rsi_14": None, "price_vs_20d_high": None, "volume_ratio": None,
                "atr_14": None, "ema_10": None, "ema_20": None, "five_day_return": None,
                "trailing_pe": None, "return_on_equity": None, "free_cash_flow": None,
                "company_selection": "REJECT", "selection_reason": "missing fundamentals",
                "market_overlay": market_action, "market_reason": market_reason,
                "sentiment_scale": size_scale, "sentiment_penalty": size_penalty,
                "news_penalty": 0.0, "net_sentiment_adjustment": size_penalty,
                "news_sentiment_pct": None, "news_gate_reason": "n/a",
                "combined_research_action": "REJECT",
                "screened_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            continue

        candidate = build_candidate(ticker, df, info, spy_6m_return)
        if candidate is None:
            rejected.append({
                "ticker": ticker, "price": None, "market_cap": None, "exchange": None,
                "rsi_14": None, "price_vs_20d_high": None, "volume_ratio": None,
                "atr_14": None, "ema_10": None, "ema_20": None, "five_day_return": None,
                "trailing_pe": None, "return_on_equity": None, "free_cash_flow": None,
                "company_selection": "REJECT", "selection_reason": "missing valid current price",
                "market_overlay": market_action, "market_reason": market_reason,
                "sentiment_scale": size_scale, "sentiment_penalty": size_penalty,
                "news_penalty": 0.0, "net_sentiment_adjustment": size_penalty,
                "news_sentiment_pct": None, "news_gate_reason": "n/a",
                "combined_research_action": "REJECT",
                "screened_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            continue

        failures = buy_rule(candidate, args, fundamentals_available)
        selected = not failures
        row = {
            "ticker": candidate.ticker,
            "price": round(candidate.price, 2),
            "market_cap": candidate.market_cap,
            "exchange": candidate.exchange,
            "rsi_14": round(candidate.rsi_14, 2) if candidate.rsi_14 is not None else None,
            "price_vs_20d_high": round(candidate.price_vs_20d_high, 2) if candidate.price_vs_20d_high is not None else None,
            "volume_ratio": round(candidate.volume_ratio, 2) if candidate.volume_ratio is not None else None,
            "atr_14": round(candidate.atr_14, 4) if candidate.atr_14 is not None else None,
            "atr_pct": round(candidate.atr_14 / candidate.price * 100, 2) if candidate.atr_14 is not None and candidate.price else None,
            "ext_vs_ema10": round(candidate.ext_vs_ema10, 2) if candidate.ext_vs_ema10 is not None else None,
            "close_loc": round(candidate.close_loc, 2) if candidate.close_loc is not None else None,
            "sma_50": round(candidate.sma_50, 2) if candidate.sma_50 is not None else None,
            "sma_200": round(candidate.sma_200, 2) if candidate.sma_200 is not None else None,
            "six_month_return": round(candidate.six_month_return, 2) if candidate.six_month_return is not None else None,
            "ext_vs_sma50": round(candidate.ext_vs_sma50, 2) if candidate.ext_vs_sma50 is not None else None,
            "rs_6m": round(candidate.rs_6m, 2) if candidate.rs_6m is not None else None,
            "sma200_rising": candidate.sma200_rising,
            "rsi_2": round(candidate.rsi_2, 2) if candidate.rsi_2 is not None else None,
            "macd_hist_pct": round(candidate.macd_hist_pct, 4) if candidate.macd_hist_pct is not None else None,
            "macd_line_pct": round(candidate.macd_line_pct, 4) if candidate.macd_line_pct is not None else None,
            "alpha002": round(candidate.alpha002, 5) if candidate.alpha002 is not None else None,
            "alpha111": round(candidate.alpha111, 5) if candidate.alpha111 is not None else None,
            "alpha054": round(candidate.alpha054, 5) if candidate.alpha054 is not None else None,
            "ema_10": round(candidate.ema_10, 2) if candidate.ema_10 is not None else None,
            "ema_20": round(candidate.ema_20, 2) if candidate.ema_20 is not None else None,
            "five_day_return": round(candidate.five_day_return, 2) if candidate.five_day_return is not None else None,
            "avg_dollar_volume": round(candidate.avg_dollar_volume, 0) if candidate.avg_dollar_volume is not None else None,
            "days_to_earnings": round(candidate.days_to_earnings, 1) if candidate.days_to_earnings is not None else None,
            "trailing_pe": round(candidate.trailing_pe, 2) if candidate.trailing_pe is not None else None,
            "return_on_equity": candidate.return_on_equity,
            "free_cash_flow": candidate.free_cash_flow,
            "company_selection": "BUY_CANDIDATE" if selected else "REJECT",
            "selection_reason": "all short-term buy rules passed" if selected else "; ".join(failures),
            "market_overlay": "BUY_ALLOWED" if market_action == "BUY" else "SELL_RISK_OFF",
            "market_reason": market_reason,
            "sentiment_scale": size_scale,
            "sentiment_penalty": size_penalty,
            "news_penalty": 0.0,
            "net_sentiment_adjustment": size_penalty,
            "news_sentiment_pct": None,
            "news_gate_reason": "pending" if selected else "n/a (technical reject)",
            "combined_research_action": (
                "BUY_RESEARCH_CANDIDATE" if selected and market_action == "BUY"
                else "HOLD_OFF" if selected else "REJECT"
            ),
            "screened_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (candidates if selected else rejected).append(row)

    # --- Phase 5: contrarian news gate (only for technical candidates) ---
    if args.news_check and candidates:
        print(f"Running news sentiment gate on {len(candidates)} technical candidates …")
        with ThreadPoolExecutor(max_workers=min(args.yf_workers, 8)) as executor:
            future_to_row = {
                executor.submit(check_news_sentiment, row["ticker"], args.news_max_articles,
                                args.news_halflife_days): row
                for row in candidates
            }
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                should_buy, news_reason, positive_pct = future.result()
                row["news_sentiment_pct"] = round(positive_pct, 1) if positive_pct is not None else None
                negative_pct = 100 - positive_pct if positive_pct is not None else None

                # Score penalty, not elimination: a strong setup can outweigh
                # an ugly headline tape.
                penalty = news_penalty(positive_pct, args.news_penalty_start_pct,
                                       args.news_penalty_max)
                row["news_penalty"] = penalty
                row["net_sentiment_adjustment"] = round(row["sentiment_penalty"] + penalty, 2)
                row["news_gate_reason"] = (
                    f"{news_reason}; penalty {penalty:+.1f} pts" if penalty
                    else f"{news_reason}; no penalty")

                # Legacy hard reject, applied only when the flag is set.
                if (args.news_negative_threshold is not None and negative_pct is not None
                        and negative_pct >= args.news_negative_threshold):
                    row["company_selection"] = "REJECT"
                    row["selection_reason"] = (
                        f"news gate: {negative_pct:.0f}% negative headlines ≥ "
                        f"{args.news_negative_threshold:.0f}% threshold (momentum: avoid negative news flow)"
                    )
                    row["combined_research_action"] = "REJECT"
                    rejected.append(row)
        candidates = [r for r in candidates if r["company_selection"] == "BUY_CANDIDATE"]

    # --- Phase 5.5: floor the candidate count with labelled near-misses ---
    if args.min_candidates > 0 and len(candidates) < args.min_candidates:
        promoted = promote_near_misses(candidates, rejected, args.min_candidates)
        if promoted:
            print(f"Only {len(candidates) - promoted} passed every rule; promoted {promoted} "
                  f"nearest-miss names (labelled NEAR_MISS) to reach {args.min_candidates}.")

    # --- Phase 6: rank & write ---
    candidates.sort(key=_momentum_score, reverse=True)
    write_results(args.candidates_output, candidates)
    write_results(args.rejected_output, rejected)

    elapsed = time.perf_counter() - t0

    near_miss = sum(1 for r in candidates if r.get("company_selection") == "NEAR_MISS")
    rr.metrics("screener", "headline", {
        "universe_size": len(tickers), "tickers_with_bars": len(raw_bars),
        "tickers_with_enough_history": len(enriched),
        "fundamentals_available": fundamentals_available,
        "candidates": len(candidates), "passed_every_rule": len(candidates) - near_miss,
        "near_miss_promoted": near_miss, "rejected": len(rejected),
        "market_overlay": "BUY_ALLOWED" if market_action == "BUY" else "SELL_RISK_OFF",
        "market_reason": market_reason,
        "greed": args.greed, "greed_average": args.greed_average,
        "spy_6m_return_pct": round(spy_6m_return, 2) if spy_6m_return is not None else None,
        "elapsed_seconds": round(elapsed, 1),
    })
    rr.rows("screener", "candidates", [
        {"ticker": r["ticker"], "price": r["price"], "rsi_14": r["rsi_14"],
         "six_month_return": r["six_month_return"], "rs_6m": r["rs_6m"],
         "ext_vs_sma50": r["ext_vs_sma50"], "atr_pct": r["atr_pct"],
         "macd_hist_pct": r.get("macd_hist_pct"), "rsi_2": r.get("rsi_2"),
         "selection": r["company_selection"]}
        for r in candidates], limit=30)
    # Which rules bind hardest — the most useful diagnostic when nothing passes.
    reasons: dict[str, int] = {}
    for r in rejected:
        for reason in str(r.get("selection_reason", "")).split(";"):
            key = reason.strip().split("(")[0].strip()
            if key:
                reasons[key] = reasons.get(key, 0) + 1
    rr.rows("screener", "top_rejection_reasons", [
        {"reason": k, "count": v}
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:15]])

    print(f"Wrote {len(candidates)} BUY_CANDIDATE rows to {args.candidates_output}")
    print(f"Wrote {len(rejected)} rejected rows to {args.rejected_output}")
    print(f"Market overlay: {market_action} ({market_reason})")
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
