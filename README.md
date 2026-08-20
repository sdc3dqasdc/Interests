# Quant Research

Systematic equity research I build and test end to end: form a rule, backtest
it honestly (out-of-sample split, transaction costs, lookahead self-tests),
and keep only what survives. Each folder is a self-contained project with its
own README describing the real mechanics — including the parts that did *not*
work.

## [`screener/`](screener/) — short-term screener (Alpaca)

A ~10-week swing pipeline: screen the market for quality names in a confirmed
long-term uptrend, rank the survivors, and measure whether any of it actually
works — on historical data and on live picks. Includes a portfolio backtest,
win-rate and ROI testers, an A/B factor sweep, a month-ahead predictor, and a
paper trader. Price data from Alpaca (IEX), fundamentals from yfinance.

```bash
cd screener
cp .env.example .env          # add your Alpaca keys
python run_all.py
```

## [`rotations/`](rotations/) — two-asset rotation indices

*"Which of these two should I be holding right now?"* — the same regime
framework applied to two markets, with opposite conclusions about which
factors transfer.

- [`tech_rotation/`](rotations/tech_rotation/) — **NTI**, QQQ vs SPY. The
  ratio-mean-reversion seed is the part that fails on US data; what works is
  a regime read (trend, relative momentum, relative volatility, rate pressure).
- [`china_rotation/`](rotations/china_rotation/) — **KDI**, 科创50 (STAR50) vs
  红利低波 (Dividend Low Vol). The US four-factor answer does not transfer;
  documents where China disagrees.

```bash
cd rotations
python -m tech_rotation.daily
python -m china_rotation.daily
```

---

Research code, not investment advice. Backtested results are not a promise of
future returns.
