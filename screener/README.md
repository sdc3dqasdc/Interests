# Short-Term Screener (Alpaca)

A ~10-week swing-trading pipeline: screen the market for quality names in a
confirmed long-term uptrend, rank the survivors, and measure whether any of it
actually works — on historical data and on your own live picks.

Price data comes from **Alpaca** (IEX feed, free tier). Fundamentals (market
cap, ROE, free cash flow, earnings date) come from **yfinance**, since Alpaca
does not provide them.

This document describes the real mechanics — what each piece actually does,
not an aspirational summary. Where something is unvalidated or a known gap,
it's called out as such rather than smoothed over.

## Strategy

Buy quality companies while they're consolidating inside an established
uptrend — not chasing a breakout. The two effects the entry rules lean on are
intermediate (6-month) momentum and the long-term trend; the effect they
avoid is short-term reversal, so the screen skips anything that just spiked.

- Price above a **rising 200-day SMA**, 50-day above 200-day (golden cross)
- **6-month return beats SPY's** (relative strength, not an absolute number)
- Price sits **near its 50-day SMA** — a consolidation, not a breakout chase
  or a breakdown
- RSI in a **pullback zone** (40–65), not overbought
- Earnings **outside the hold window**
- Quality backstop: positive free cash flow, ROE ≥ 10%, market cap ≥ $2B
- New entries **suppressed when SPY itself is below its own rising 200-day
  SMA** (market-regime gate)
- A **stop-loss** (15% default) caps any single position's damage

**Hold period is 10 weeks (50 trading days)** — it started at 4 weeks (20
days) and was widened without re-tuning the entry rules. The ATR cap,
RSI/extension bands, and momentum windows above were calibrated for the
4-week version; the screener's own docstring flags this as unfinished, and
`ab_sweep.py` (below) exists partly to re-check them at the new horizon.

None of this is fitted to a backtest — the thresholds come from the
trend-template/momentum literature. **Not everything here has been measured,
though.** `alpha_lab.py` only tests price/volume factors — it has nothing to
say about the ROE/FCF quality gates or the SPY regime gate; those are
untested, not validated, and "untested" and "shown not to work" call for
opposite reactions. The tools below exist to close that gap, not to assume
the strategy works.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install alpaca-py yfinance pandas numpy matplotlib requests

cp .env.example .env   # then edit .env with your real Alpaca keys
```

Get a free Alpaca account (paper trading is enough for market data) at
[alpaca.markets](https://alpaca.markets). Both `ALPACA_API_KEY` and
`ALPACA_SECRET_KEY` are read from `.env` automatically — no need to `export`
them in your shell.

## The pipeline

```
screener  ->  selector  ->  paper_trader (live orders)
   |             |     ->  (live) roi_tester
   |             |     ->  (history) win_rate_tester  ->  ab_sweep (many variants, one data load)
   -> backtest              (research) alpha_lab

run_all.py drives screener/selector/roi_tester/alpha_lab/win_rate_tester/backtest
in order and merges all their output into one run_report.md.
```

### 1. `short_term_screener_alpaca.py` — screen the market today

```bash
.venv/bin/python short_term_screener_alpaca.py --universe nasdaq-nyse --cnn-sentiment
```

Downloads the current Nasdaq/NYSE universe, pulls ~14 months of Alpaca bars,
fetches fundamentals (cached locally, `fundamentals_cache.json`, 7-day TTL),
and applies the rules above. Alpaca bars and the CNN Fear & Greed read are
also cached per calendar day under `.cache/` — a second run the same day
reuses them instead of refetching (`--refresh-cache` to force a live pull).
Writes:

- `short_term_candidates.csv` — everything that passed (or the closest
  near-misses, labelled `NEAR_MISS`, if fewer than 5 pass — see
  `--min-candidates`), including a `sentiment_scale` column derived from the
  CNN reading
- `rejected_companies.csv` — everything else, with the exact reason it failed
- `screener_universe.txt` — the ticker list, reused by every other tool

Survives a total Yahoo outage: if fundamentals fail for every ticker, it
screens on technicals alone rather than returning nothing. Also survives
Alpaca returning a symbol whose price history has a multi-year gap (a ticker
reused after an unrelated delisting, e.g. `LB`) — the series is trimmed to
its most recent contiguous listing instead of silently spanning two
different companies.

### 2. `select_top15.py` — narrow to the best picks

```bash
.venv/bin/python select_top15.py
```

Reads `short_term_candidates.csv`, ranks by 6-month relative strength vs SPY
(the composite/Alpha-191 blend is available but off by default — see below),
and writes `top_candidates.csv`, including a `position_weight` column derived
from the screener's `sentiment_scale` — sentiment sizes how much capital goes
into the book as a whole, kept separate from which names rank better than
which.

### 2b. `paper_trader.py` — put the picks on an Alpaca paper account

```bash
.venv/bin/python paper_trader.py           # dry run: print the orders
.venv/bin/python paper_trader.py --live    # submit them to the paper account
```

Reconciles the paper account against `top_candidates.csv` with the backtest's
mechanics: `--max-positions 5` names at 1/5 of equity each (scaled by
`position_weight` from step 2), a GTC `--stop-loss-pct 15` stop placed
server-side, and a sale after `--hold-days 50` trading days. Entry dates live
in `paper_positions.json`, since Alpaca positions don't carry one — delete it
to reset the hold clock.

Defaults to a dry run; nothing is sent without `--live`, and it only talks to
`paper-api.alpaca.markets` unless you pass `--real` and type a confirmation.
Run it on the same cadence as the backtest's `--rebalance-freq` (every ~10
trading days), during market hours.

### 3a. `roi_tester.py` — how are your live picks actually doing?

```bash
.venv/bin/python roi_tester.py
```

Reads `top_candidates.csv`, replays each pick with the same entry/exit/stop
mechanics as the backtest against real Alpaca data, and reports CLOSED / OPEN
/ STOPPED / PENDING with ROI vs SPY. Run this anytime — the day after a
screen, mid-hold, or after the hold closes out. This is the bias-free number
that actually answers "does this work."

### 3b. `backtest.py` — portfolio-level historical test

```bash
.venv/bin/python backtest.py --start 2022-01-01 --end 2024-12-31
```

Simulates buying everything the screen passes (capped at `--max-positions`,
default 5, sized 1/N of equity) over history. Reports total/annualised return,
win rate, max drawdown vs SPY buy-and-hold.

### 3c. `win_rate_tester.py` — does the *actual model* (top 5 picks) win?

```bash
.venv/bin/python win_rate_tester.py --start 2021-01-01 --end 2024-12-31
```

Different question from the backtest: replays screen → rank → **top 5 only**
at each historical pick date, the way you actually use it, and reports
pick-level win rate with a 95% confidence interval, per-year and per-rank
breakdowns (does rank 1 beat rank 5?), and a by-exit-reason table (is the
stop-loss earning its keep?).

### 4. `alpha_lab.py` — which factors have real signal on your data?

```bash
.venv/bin/python alpha_lab.py --start 2019-01-01 --end 2024-12-31
```

Cleans the data (drops halted/stale bars, winsorizes outliers), then measures
IC/ICIR for candidate alpha factors (GTJA Alpha-191 survivors, Alpha101-style
forms, classic academic factors, RSI/MACD/Bollinger/Aroon/MFI-style forms)
against your own universe — because published ICIRs don't transfer across
markets. Kicks out weak or sign-flipped (decayed) factors and keeps the top
N. Writes `alpha_report.csv` and `alpha_selection.json`. This only ever
measures price/volume factors: it says nothing about whether the ROE/FCF
quality gates or the SPY regime gate help.

### 5. `ab_sweep.py` — A/B many rule variants against one data load

```bash
.venv/bin/python ab_sweep.py --start 2021-01-01 --end 2024-12-31
```

`win_rate_tester.py` answers "how did this one configuration do?" — running
it once per variant re-fetches bars, recomputes every signal, and re-fetches
fundamentals each time, for minutes of identical work just to change one
threshold. This loads the market data once and replays every variant against
it, so a 14-way comparison costs one setup instead of fourteen. Built to
answer specific questions with a measurement instead of an assumption — e.g.
"does the market-regime gate actually help?" — and to re-check the entry
thresholds now that the hold is 10 weeks instead of the 4 they were tuned
for.

### 6. `run_all.py` / `run_report.py` — one consolidated report

```bash
.venv/bin/python run_all.py --start 2021-01-01 --end 2024-12-31
```

Runs the screener, `select_top15`, `roi_tester`, `alpha_lab`,
`win_rate_tester`, and `backtest` in dependency order (any can be skipped
with `--skip`), all writing into **one shared report** instead of each
overwriting the last tool's output. `run_report.py` is the shared writer:
every tool appends its parameters and headline metrics as one fact per row
(`run_id, tool, section, kind, name, value`) to `run_report.csv`, rendered
alongside as `run_report.md`. Long format because the six tools' outputs are
wildly different shapes — the screener has candidate counts, the backtest has
an equity curve, `alpha_lab` has a factor table — and a wide CSV would be
mostly empty columns for each one.

### 7. `tech_rotation/` — QQQ or SPY? the Nasdaq Tilt Index

```bash
.venv/bin/python -m tech_rotation.daily --current 0.70    # today's decision
.venv/bin/python -m tech_rotation.backtest --ablate --sensitivity
```

A separate question from the screener: not which stock, but which of the two
index ETFs to hold. A nine-factor index (trend regime, relative momentum and
volatility, rate pressure, credit, semis, concentration, and the QQQ/SPY ratio
itself) maps to a QQQ weight, backtested against holding either ETF and against
a beta-matched static blend. See `tech_rotation/README.md` — including the
finding that the ratio-mean-reversion rule it started from is the one component
that loses money.

The China counterpart, `china_rotation/` (STAR50 `科创50` vs Dividend Low
Volatility `红利低波`), deliberately does NOT reuse the US answer: on ~5.5
years of STAR50 history the ported 4-factor subset underperforms the full
8-factor index, and the ratio-mean-reversion idea that lost in the US is
actually competitive here. See `china_rotation/README.md`.

## Experimental — not part of the validated pipeline

`day_trade_signals.py`, `next_day_predictor.py`, `one_day_top3.py`,
`month_predictor.py`, and `basket_walkforward.py` are a separate line of
1-day and ~1-month mean-reversion experiments living in this same directory.
They are **not wired into `run_all.py`, not covered by `alpha_lab.py`'s
factor testing, and not the strategy the rest of this README describes.**
`day_trade_signals.py`'s own docstring carries an explicit honesty note to
read before trusting anything built on it. Treat these as scratch research,
not as something to run for real picks — if one of them earns its way into
the real pipeline, it belongs in the sections above, backed by the same
IC/win-rate evidence the rest of this system requires.

## Honest limitations

- **Fundamentals are today's values** applied across historical dates in the
  backtests — survivorship and lookahead bias. Read backtest "alpha" numbers
  as an optimistic ceiling, not an expectation.
- **The earnings blackout and news gate are not simulated** in the backtest —
  no free point-in-time source for either.
- **The entry rule thresholds were tuned for a 4-week hold and never
  re-validated at the current 10-week hold.** `ab_sweep.py` exists to close
  this gap; until it's run and the results acted on, treat the current
  defaults as inherited, not re-confirmed.
- **The ROE/FCF quality gates and the SPY market-regime gate are untested.**
  `alpha_lab.py` only measures price/volume factors, so it has never checked
  whether these rules help, hurt, or do nothing.
- **`roi_tester.py` is the one number without survivorship/lookahead bias** —
  it's real forward performance on real data. Let it accumulate before
  trusting a win rate.
- Single-factor ICIRs of 0.02–0.05 in `alpha_lab.py` are normal and weak by
  design; they're meant to be combined across a wide universe, not to carry a
  5-pick portfolio alone.

## Files not in this repo

`.env` (your API keys — including, if set, `ALERT_WEBHOOK_URL`) and every
generated CSV/PNG/JSON/lock file (`screener_universe.txt`, `*_candidates.csv`,
`*_report.csv`, `run_report.csv`/`.md`, backtest/roi/win-rate/alpha-lab
outputs, plots, `paper_positions.json`) are gitignored, along with the
`.cache/` and `fundamentals_cache.json` caches and `__pycache__/`. All of it
is either a secret or fully reproducible by running the tools above.
