# NTI — the Nasdaq Tilt Index

*When is it time to own the Nasdaq instead of the S&P 500?*

The seed is the CSI 300 / ChiNext ratio trade: when the growth index is cheap
against the blue-chip index, rotate into growth; when it is expensive, rotate
back into the steady one. The US analogue is QQQ vs SPY.

**The headline finding: on US data, that ratio rule is the part that does not
work.** Traded alone it is the worst variant in the file — worse than holding
either ETF, in both halves of the sample. What does work is a regime read:
market trend, relative momentum, relative volatility and rate pressure.

```bash
.venv/bin/python -m tech_rotation.daily                         # today's decision
.venv/bin/python -m tech_rotation.daily --current 0.70 --plot   # trade vs what you hold
.venv/bin/python -m tech_rotation.backtest                      # main report
.venv/bin/python -m tech_rotation.backtest --ablate             # what each factor is worth
.venv/bin/python -m tech_rotation.backtest --sensitivity        # parameter robustness
.venv/bin/python -m tech_rotation.backtest --selftest           # lookahead check
.venv/bin/python -m tech_rotation.backtest --ladder             # what each reading paid
.venv/bin/python -m tech_rotation.backtest --profit --plot      # profit % + equity charts
```

## Daily use

`tech_rotation.daily` reads the last row of the index and stops — no equity
curves. Pass `--current` with the QQQ weight you actually hold and it returns a
trade or a hold:

```
NTI-core  2026-07-28   -0.02 sigma
  buy:     100% QQQ  /    0% SPY   (switch rule, threshold -0.5)
  trend:   -0.76 vs a week ago, -1.52 vs a month ago
  you hold:  70% QQQ
  ACTION:   sell 20pt of QQQ (70% -> 50%), funded from SPY
```

Checking it daily is fine; **acting** daily is not. The backtest is flat across
rebalance frequencies from 1 to 21 days, so daily trading only buys turnover.
The `--band` default of 10pt is the no-trade zone — below that, hold.

## The buy ladder — what to hold at what reading

Bucket every historical reading and measure the **next month** of QQQ-minus-SPY
return (`--ladder`):

| reading | % of time | next 21d, IS | next 21d, OOS | hold |
|---|---|---|---|---|
| > +1.5 | 22% | +0.80% | +0.70% | 100% QQQ |
| +0.5 .. +1.5 | 39% | +0.44% | +0.48% | 100% QQQ |
| -0.5 .. +0.5 | 20% | +0.76% | +0.60% | 100% QQQ |
| -1.5 .. -0.5 | 14% | **-0.86%** | **-0.26%** | 20% QQQ |
| < -1.5 | 6% | -3.12% | *+0.25%* | 20% QQQ |

It is **not a smooth ladder**, and that is the useful part. Above −0.5σ the
Nasdaq wins by roughly the same margin whatever the reading — paying turnover to
tell +0.6 from +2.0 buys nothing. Below −0.5σ it loses in both halves. The
deepest bucket's spectacular in-sample number is the dot-com bust alone; out of
sample it is *positive*, so "maximum bearish = maximum defensive" is not
supported by anything but one episode.

So the rule is a **switch, not a dial**: `NTI-core > -0.5σ → 100% QQQ,
otherwise 20% QQQ / 80% SPY`. The threshold is not tuned — anything from -0.75
to 0.0 gives Sharpe 0.61-0.63. It beats a static blend of the same 85% average
exposure in the full sample (0.63 vs 0.52) and in both halves (0.36 vs 0.23,
1.00 vs 0.96), at **lower** turnover than the continuous tilt (3.8x vs 6.0x/yr).

## The index

Nine components, each a trailing z-score signed so **positive = favour the
Nasdaq**, combined by a priori weights and re-scaled by their own trailing
standard deviation so a reading is in sigma units.

| component | what it is | weight |
|---|---|---|
| `valuation` | QQQ/SPY ratio's gap from its 200-day trend, **inverted** — the video's idea | 1.00 |
| `momentum` | 6-month QQQ-vs-SPY relative return — the brake on the value trap | 1.00 |
| `reversal` | 1-month relative return, inverted | 0.50 |
| `regime` | SPY vs its own rising/falling 200-day SMA | 1.00 |
| `rel_vol` | QQQ's realised vol over SPY's, inverted | 0.50 |
| `rates` | TLT 3-month return — the discount rate on long-duration growth | 0.50 |
| `credit` | HYG-vs-IEF 3-month spread move | 0.50 |
| `semis` | SMH-vs-QQQ 3-month relative return | 0.75 |
| `concentration` | SPY-vs-RSP 3-month relative return | 0.25 |

Reading → position, continuous version: `w_QQQ = clip(0.5 + 0.25 × NTI, 0, 1)`,
remainder in SPY. Neutral is 50/50; ±2 sigma is all-in or all-out. The switch
rule above is the tested default; this one is the smoother alternative.

**NTI-core** keeps only `regime`, `momentum`, `rel_vol`, `rates` at their
original relative weights. The four were not picked as the best-performing
subset — they are the ones whose timing contribution has the **same sign in
both disjoint halves** of the sample and whose removal hurts both halves.

## Harness

- The reading for day *t* uses only bars through *t* and is traded at *t+1*'s
  close. `--selftest` rebuilds the index from truncated history and checks the
  point-in-time value matches the full-history one to 1e-10.
- Weekly rebalance, 5bp per unit of turnover.
- Signal history comes from `^NDX` / `^GSPC` (1985+) so the z-scores are warm
  when QQQ starts trading in March 1999 — otherwise the backtest would miss the
  dot-com bust, the one episode that matters most here. Only the ETFs are held.
- Ablations are measured over a **common window**: HYG lists in 2007 and RSP in
  2003, so a credit-only variant measured from its own inception simply skips
  the bust and posts a meaningless number.

## Results — 1999-03 to 2026-07

Sharpe, net of costs:

| | full sample | 1999–2012 | 2013–now |
|---|---|---|---|
| **NTI-core 2-state (switch)** | **0.63** | **0.36** | **1.00** |
| NTI-core (4 factors, continuous) | 0.61 | 0.33 | 1.00 |
| NTI (all 9) | 0.58 | 0.30 | 0.96 |
| hold QQQ | 0.51 | 0.23 | 0.96 |
| hold SPY | 0.52 | 0.22 | 0.90 |
| 50/50 rebalanced | 0.53 | 0.23 | 0.95 |
| static 64% QQQ *(beta-matched)* | 0.53 | 0.23 | 0.96 |
| ratio only (the video's rule) | 0.45 | 0.14 | 0.90 |

The beta-matched line is the control that matters: a rule sitting at 64% QQQ on
average beats SPY in any period QQQ won, with no skill involved. NTI-core is
above it in both halves; the ratio-only rule is below everything.

Worst drawdown over the full sample: NTI-core **-64.6%** vs QQQ **-83.0%** and
the beta-matched static blend **-73.5%**. Out of sample it is -28.8% vs -35.1%
for QQQ. Most of the value is in what it avoids.

By episode (total return):

| | dot-com 00-02 | GFC 07-09 | tech run 13-21 | 2022 | AI run 23-now |
|---|---|---|---|---|---|
| NTI-core | -61% | -57% | +459% | -24% | +134% |
| hold QQQ | -81% | -52% | +544% | -33% | +161% |
| hold SPY | -42% | -55% | +285% | -19% | +104% |

## Honest reading of the evidence

Strip out the average tilt and measure only the timing — `(w_t − w̄) × (QQQ − SPY)`:

| NTI-core | timing CAGR | IR | t-stat | IC(21d) |
|---|---|---|---|---|
| full sample | +1.6% | 0.28 | 1.45 | 0.066 |
| 1999–2012 | +2.4% | 0.31 | 1.16 | 0.081 |
| 2013–now | +0.8% | 0.31 | 1.14 | 0.056 |

The timing edge is **positive and consistent in both halves, but not
statistically significant** — t ≈ 1.4 over 27 years is not proof of skill. What
the index does reliably is size exposure: it is a defensible way to decide *how
much* Nasdaq to hold, and it survives every parameter it could have been fitted
to (tilt size 0.10–0.50, rebalance 1–21 days, execution lag 1–10 days all give
Sharpe 0.57–0.62). Treat it as an allocation dial, not a market-timing signal.

The one strong claim the data does support is the negative one: the ratio rule
from the video, applied to QQQ/SPY, loses money against every alternative here,
in both halves independently.
