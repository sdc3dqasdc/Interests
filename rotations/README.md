# Rotation indices

Two instances of the same question — *hold the growth asset or the defensive
one?* — built on a shared shape: a daily data layer with an on-disk cache, a
factor index, a backtest with an out-of-sample split and cost model, and a
`daily` entry point that prints today's decision.

Run them as packages **from this directory**, so the relative imports resolve:

```bash
python -m tech_rotation.daily                 # QQQ vs SPY  (NTI)
python -m tech_rotation.backtest --ablate
python -m china_rotation.daily                # STAR50 vs Dividend Low Vol (KDI)
python -m china_rotation.backtest --ablate --ladder --profit
```

| | [`tech_rotation/`](tech_rotation/README.md) | [`china_rotation/`](china_rotation/README.md) |
|---|---|---|
| Pair | QQQ vs SPY | 科创50 vs 红利低波 |
| Result | 4 factors survive both halves | US answer does **not** transfer; needs all 8 |
| History | full | ~5.5y traded |
