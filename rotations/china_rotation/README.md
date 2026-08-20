# KDI — Kechuang50 vs Hongli-Dibo

*Should I hold 科创50 (STAR50, growth/tech) or 红利低波 (Dividend Low Volatility)?*

This is the China counterpart to [`tech_rotation/`](../tech_rotation/README.md)
(QQQ vs SPY). Read that README first — this one assumes it and exists mainly
to document where China **disagrees** with the US result.

```bash
.venv/bin/python -m china_rotation.daily                          # today's decision
.venv/bin/python -m china_rotation.daily --current 0.50 --plot    # trade vs what you hold
.venv/bin/python -m china_rotation.backtest --ablate --ladder --profit
.venv/bin/python -m china_rotation.backtest --selftest             # lookahead check
```

## Read this before trusting anything below

**The traded legs have ~5.5 years of history, not 27.** STAR50 (588000.SS)
lists 2020-09-28 — the STAR board itself is only a few years old. Splitting
that into halves gives ~2.8 years each: one bear leg (2021-22) and one bull
leg (2023-25 AI rally), not two independent market regimes. Every number here
should be read as a much weaker signal than its US counterpart, which passed
the same tests on 27 years split into two 13-year halves.

## The two US findings do NOT transfer

1. **The full 8-factor index beats every trimmed subset here** — including
   the exact 4-factor subset (`regime`, `momentum`, `rel_vol`, `rates`) that
   won on US data. Porting that subset verbatim was tried first:

   | | full sample Sharpe | 1st half | 2nd half |
   |---|---|---|---|
   | **KDI (all 8)** | **0.93** | 0.45 | **1.23** |
   | KDI-core (US subset, ported) | 0.54 | 0.38 | 0.65 |
   | ratio only | 0.77 | 0.40 | 1.00 |

2. **The ratio-mean-reversion idea — the one thing that lost in the US — is
   competitive here**, beating a 50/50 blend and the beta-matched static
   blend in both halves. Onshore growth-vs-dividend dynamics do not have the
   same shape as QQQ-vs-SPY; state-directed capital flows into strategic
   sectors (semis, STAR-board IPOs) plausibly work differently than the
   momentum-driven leadership persistence that dominates in the US.

A from-scratch "China core" is deliberately **not** built to replace the
ported one. Two ~2.8-year halves is not enough statistical power to trust a
same-sign-in-both-halves filter across 8 components — some passing by chance
is expected. `momentum` is negative in both halves here (the opposite of its
role in the US index), which is a real, if weak, signal against it — but
dropping it changes results by less than the noise band, so it stays.

**Recommendation: use the full 8-factor `KDI`, continuous tilt.** Don't use
the 4-factor subset that worked in the US.

## The buy ladder does NOT transfer either

Bucketing readings against next-month KC50-minus-DIBO return, the way
`tech_rotation` derived its −0.5σ switch threshold:

| reading | % of time | next 21d |
|---|---|---|
| > +1.5σ | 9% | **−0.41%** |
| +0.5 .. +1.5σ | 27% | +1.43% |
| −0.5 .. +0.5σ | 18% | +0.95% |
| −1.5 .. −0.5σ | 33% | **−2.22%** |
| < −1.5σ | 13% | **+1.68%** |

This is **not monotonic** — the most bullish bucket pays negative, and the
most bearish bucket pays positive. With ~5.5 years of data this is much more
likely to be one or two dominant episodes (the 2021-22 STAR50 selloff was a
long grind through the "−1.5..−0.5" band, and its bottom coincided with the
"< −1.5" band right before the recovery) than a real, exploitable shape. **No
switch-rule threshold is recommended for China.** Use the continuous tilt.

## Components

Same eight roles as `tech_rotation`, substituted for what's actually liquid
onshore (no `credit` component — no clean onshore high-yield ETF on yfinance,
and that factor didn't survive the US test either):

| component | China proxy | US analogue |
|---|---|---|
| `valuation` | KC50/DIBO ratio's 200d-trend gap, inverted | QQQ/SPY ratio |
| `momentum` | 6-month KC50-vs-DIBO relative return | QQQ-vs-SPY momentum |
| `reversal` | 1-month relative return, inverted | same |
| `regime` | CSI300 (510300.SS) 200-day trend | SPY trend |
| `rel_vol` | KC50 realised vol vs DIBO's, inverted | QQQ vs SPY vol |
| `rates` | China treasury bond ETF (511010.SS), 3-month return | TLT |
| `semis` | China semiconductor ETF (512480.SS) vs KC50 | SMH vs QQQ |
| `concentration` | SSE50 (510050.SS) vs CSI300 | SPY vs RSP |

Signal history before STAR50 existed is spliced from the ChiNext ETF
(159915.SZ, since 2015) standing in for the growth leg — same trick
`tech_rotation` uses with `^NDX`/`^GSPC`, but over a 5-year proxy gap
(2015-2020) instead of 14.

## Harness

Identical rules to `tech_rotation/backtest.py`: reading at close *t*, traded
at *t+1*'s close, weekly rebalance, 5bp/side cost, every z-score trailing
only. `--selftest` rebuilds the index from truncated history and confirms the
point-in-time reading matches the full-history one to 1e-10 — passes here too.

## Results — 2020-09 to 2026-08 (traded window)

| | Sharpe full | 1st half | 2nd half | MaxDD |
|---|---|---|---|---|
| **KDI (all 8)** | **0.93** | 0.45 | **1.23** | -22.6% |
| KDI-core (ported US subset) | 0.54 | 0.38 | 0.65 | -23.3% |
| ratio only | 0.77 | 0.40 | 1.00 | -26.8% |
| hold DIBO | 0.71 | 0.79 | 0.65 | -16.5% |
| hold KC50 | 0.23 | -0.32 | 0.53 | -59.6% |
| static 46% KC50 (beta-matched) | 0.54 | 0.25 | 0.72 | -24.8% |

KDI (all 8) beats its own beta-matched static blend in the full sample and in
both halves — the one result here that's at least directionally consistent
with the US finding (a regime-aware rotation beats a fixed blend of the same
average exposure).

Timing-only t-stat (average tilt stripped out): **2.4 full / 0.6 / 2.4** —
looks stronger than the US number (1.4), but on a fifth of the data and one
regime transition, that is a wider confidence interval producing a bigger
number, not more evidence.

## Today

```
KDI  2026-08-03   -0.83 sigma
  target:   29% 588000.SS (STAR50)  /   71% 512890.SS (Div. Low Vol)
```

Driven mostly by `semis` at its -3.0σ floor and a strong `reversal` reading
(+2.72σ, i.e. KC50 has just underperformed sharply over the past month) —
those two are pulling in opposite directions, which is exactly the kind of
disagreement the index exists to net out.
