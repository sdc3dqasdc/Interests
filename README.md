# Quant Research

Three systematic equity projects that share one question: **given a pile of
plausible signals, which ones do you actually keep?**

Every folder here answers it with a measurement rather than an opinion, and
each uses a *different* selection procedure because the statistical setting is
different. This README is about those procedures and the math behind them; the
per-project READMEs cover mechanics.

| Folder | Setting | Selection procedure |
|---|---|---|
| [`screener/`](screener/) | ~1500-name cross-section, daily | Rank-IC / ICIR with a sign-stability filter |
| [`rotations/tech_rotation/`](rotations/tech_rotation/) | 1 spread (QQQ−SPY), 25y | A-priori weights + leave-one-out ablation, two halves |
| [`rotations/china_rotation/`](rotations/china_rotation/) | 1 spread (STAR50−DivLowVol), ~5.5y | Same, and it reaches the opposite conclusion |

---

## 1. Cross-sectional selection: IC, ICIR, and what they don't tell you

[`screener/alpha_lab.py`](screener/alpha_lab.py) defines a pool of **29**
candidate factors — GTJA Alpha-191 forms, Alpha-101-style price/volume forms,
classic academic anomalies (12-1 momentum, short-term reversal, low-vol,
52-week-high proximity, negative skew), and a deliberate grab-bag of technical
indicators (RSI family, MACD family, Bollinger %B, CMF, TRIX, Williams %R).
The grab-bag is there on purpose: a selection rule you only ever feed good
candidates is a rule you have not tested.

Published ICIRs are not transferable — they were measured on other universes,
eras and horizons — so everything is recomputed on this universe.

### 1.1 Conditioning the cross-section

Each factor is computed per ticker, then, **per date**, the cross-section is
winsorized by median absolute deviation and standardised:

$$
m = \operatorname{med}(f), \qquad
\mathrm{MAD} = \operatorname{med}\left(\lvert f_i - m \rvert\right), \qquad
\hat{\sigma}_{\mathrm{MAD}} = 1.4826 \cdot \mathrm{MAD}
$$

$$
\tilde{f}_i = \operatorname{clip}\!\left(f_i,\; m \pm k\,\hat{\sigma}_{\mathrm{MAD}}\right),
\qquad
z_i = \frac{\tilde{f}_i - \overline{\tilde{f}}}{s(\tilde{f})}, \qquad k = 5
$$

The constant $1.4826 = 1/\Phi^{-1}(0.75)$ is what makes the MAD a consistent
estimator of $\sigma$ for Gaussian data, so $k$ is interpretable in sigma units
while the estimator itself has a 50% breakdown point — one blown-up print
cannot move the clip bounds, which a mean/std winsorization would let it do.

Before that, bars are cleaned (non-positive volume or price, `high < low`,
non-finite values, and frozen runs of $\ge 5$ identical closes — halts and
stale feeds masquerading as data), and a per-date eligibility gate
($P \ge \$5$, dollar volume $\ge \$1\mathrm{M}$) keeps illiquid names from
dominating a cross-section they could never be traded in.

### 1.2 The IC series

For each sampled date $t$, the **rank IC** is the Spearman correlation between
the conditioned factor and the forward $h$-day return:

$$
\mathrm{IC}_t \;=\; \operatorname{corr}\!\left(\operatorname{rank}(z_{t}),\;
\operatorname{rank}(r_{t \to t+h})\right), \qquad h = 50
$$

Ranks, not levels: the payoff of a ranking model is invariant to any monotone
transform of the factor, and the return cross-section has heavy enough tails
that a Pearson IC would largely report where the two biggest movers landed.
Dates with fewer than 20 eligible names are skipped. Every factor is written
so that **higher = predicted better**, so a positive IC means it works as
stated and a negative one means it works inverted.

Summarising the series $\{\mathrm{IC}_t\}_{t=1}^{T}$:

$$
\mathrm{ICIR} = \frac{\overline{\mathrm{IC}}}{s(\mathrm{IC})},
\qquad
t = \mathrm{ICIR}\sqrt{T}
$$

ICIR is the information ratio *of the signal itself* — how reliably the factor
ranks, not how much it pays. It connects to portfolio performance through
Grinold's fundamental law, $\mathrm{IR} \approx \mathrm{IC}\sqrt{\mathrm{BR}}$:
a small per-name edge is only worth something applied across many independent
bets, which is precisely why the screener is a cross-sectional ranker holding a
basket, not a single-name call.

### 1.3 The selection rule

Split the IC series into two disjoint halves and keep a factor only if:

$$
\lvert \mathrm{ICIR} \rvert \ge 0.02
\quad\text{and}\quad
\operatorname{sign}\!\left(\overline{\mathrm{IC}}_{1}\right)
= \operatorname{sign}\!\left(\overline{\mathrm{IC}}_{2}\right)
$$

then take the top $N=5$ survivors by $\lvert\mathrm{ICIR}\rvert$.

The second condition does nearly all the work. The $\lvert\mathrm{ICIR}\rvert$
floor is a noise gate that almost everything clears; the sign test is a crude
but cheap stationarity check. Under a true constant edge, both halves agree
with high probability; under the null $\mathrm{IC}=0$, they agree with
probability $\tfrac{1}{2}$. So the filter roughly **halves the pass rate of
pure noise while barely touching a real effect** — a favourable asymmetry when
29 candidates are being screened at once. The report also records a decay
figure, $\lvert \mathrm{IC}_2\rvert / \lvert \mathrm{IC}_1\rvert - 1$, so a
factor that keeps its sign but loses its magnitude is visible rather than
silently promoted.

Measured on 2021-01-01 → 2026-07-21, $h=50$, 269 sampled cross-sections
(`alpha_report.csv` is regenerated per run and gitignored):

| factor | ICIR | IC mean | t | % positive | half 1 → half 2 | verdict |
|---|---:|---:|---:|---:|---|---|
| `neg_skew_60` | 0.375 | 0.0322 | 6.15 | 65.8 | 0.060 → 0.005 | kept |
| `momentum_12_1` | 0.341 | 0.0533 | 5.28 | 65.4 | 0.031 → 0.075 | kept |
| `high52_prox` | 0.301 | 0.0540 | 4.67 | 68.5 | 0.045 → 0.063 | kept |
| `low_vol_20` | 0.294 | 0.0633 | 4.82 | 59.1 | 0.121 → 0.006 | kept |
| `atr_pct_14` | 0.293 | 0.0683 | 4.80 | 58.4 | 0.125 → 0.012 | kept |
| `ext_sma50` | 0.222 | 0.0289 | 3.64 | 58.0 | 0.068 → −0.010 | **sign flip** |
| `macd_line` | −0.174 | −0.0247 | −2.85 | 42.8 | −0.052 → 0.003 | **sign flip** |
| `rsi14_near50` | −0.088 | −0.0082 | −1.44 | 46.8 | 0.004 → −0.021 | **sign flip** |
| `macd_hist` | −0.006 | −0.0008 | −0.10 | 49.4 | −0.009 → 0.008 | **sign flip** |

Most of the RSI and MACD family — the indicators a discretionary screen would
lean on hardest — sit at $\lvert\mathrm{ICIR}\rvert < 0.1$ **and** flip sign
between halves. That is the signature of an effect that was never there.

### 1.4 Three things this procedure gets wrong, stated plainly

**The t-stats are inflated by overlap.** $t = \mathrm{ICIR}\sqrt{T}$ assumes
independent observations. With $h = 50$ trading days of forward return sampled
every 5 sessions, each observation shares roughly $50/5 = 10$ neighbours'
returns. Under a Hansen–Hodrick / Newey–West style correction the effective
sample is closer to $T/10$, so the honest scale is $t/\sqrt{10}$:

| factor | naive $t$ | overlap-adjusted $t$ |
|---|---:|---:|
| `neg_skew_60` | 6.15 | **1.94** |
| `momentum_12_1` | 5.28 | **1.67** |
| `low_vol_20` | 4.82 | **1.52** |
| `atr_pct_14` | 4.80 | **1.52** |
| `high52_prox` | 4.67 | **1.48** |

**And they are not corrected for multiple testing.** Screening 29 candidates
at $\alpha = 0.05$ expects $29 \times 0.05 \approx 1.5$ false discoveries by
construction. A Bonferroni threshold of $0.05/29$ needs $\lvert t\rvert
\gtrsim 3.1$; *nothing in the pool clears that once overlap is accounted for.*
So none of these are treated as established alpha — they are ranking
ingredients whose value is decided downstream by the portfolio backtest, which
models position limits, rebalancing, costs and compounding. That arbiter has
overruled the per-pick sweep before, which is why
[`screener/ab_sweep.py`](screener/ab_sweep.py) carries a warning banner about
its own metric.

**ICIR ranking ignores redundancy.** `low_vol_20` (negated 20-day return std)
and `atr_pct_14` (negated ATR/price) are the same volatility view computed two
ways, and they land at ICIR 0.294 and 0.293. Ranking marginally means the
top-5 hands 40% of its weight to one idea. If the five survivors were
genuinely independent, the combined signal would be

$$
\mathrm{ICIR}_{\text{combined}} = \sqrt{\textstyle\sum_i \mathrm{ICIR}_i^2} \approx 0.72
$$

but with two of them nearly collinear the real figure is lower, and the naive
sum-of-squares is the number to distrust. A correlation-aware step —
clustering the pool and keeping one representative per cluster, or a full
$\Sigma^{-1}$ weighting — is the honest next iteration.

### 1.5 From factors to a score

Two downstream consumers weight factors in deliberately different ways.

[`screener/month_predictor.py`](screener/month_predictor.py) weights **by
measured $\lvert\mathrm{ICIR}\rvert$** at the 20-day horizon:

$$
S = \sum_i \lvert\mathrm{ICIR}_i\rvert \cdot z_i
$$

which is the mean-variance optimum $w \propto \Sigma^{-1}\mu$ under the
assumption $\Sigma = \mathrm{diag}(\sigma_i^2)$ — i.e. exactly the independence
assumption §1.4 says is violated. It is used anyway because estimating a
$5\times 5$ factor covariance from this sample would add more estimation error
than it removes; the assumption is a choice, not an oversight.

[`screener/select_top15.py`](screener/select_top15.py) does **not** use ICIR
weights. Its eight components are weighted a priori (momentum 25, entry
quality 20, trend 15, RSI zone 10, low vol 10, quality 10, earnings gap 5,
liquidity 5) and — the important part — most are **non-monotone** in the
underlying variable:

$$
\mathrm{tri}(x; a, p, b) =
\begin{cases}
\dfrac{x-a}{p-a}, & a < x \le p \\[2ex]
\dfrac{b-x}{b-p}, & p < x < b \\[1ex]
0, & \text{otherwise}
\end{cases}
$$

A triangular sweet-spot, not a rank. The thesis is *buy consolidation inside an
uptrend*, so more 6-month momentum is better only up to ~30%, and a stock 25%
above its 50-day SMA is worse than one sitting on it. Rank-IC cannot express
that shape — a factor that is good in the middle and bad at both ends has an IC
near zero — which is why this score is validated by backtest rather than by IC,
and why the two files disagree by design.

---

## 2. Time-series selection: ablation instead of IC

The [rotation indices](rotations/) face the opposite problem. There is no
cross-section — one spread, one observation per day — so IC is unavailable and
$T$ is small in the units that matter (leadership regimes last years, so 25
years of daily data is perhaps a handful of independent events). Fitting
weights here would be curve-fitting with extra steps.

So the weights are **set a priori by role and never fitted**: the two
deliberately disagreeing views (valuation vs momentum) and the regime gate get
1.00, confirmations get 0.50–0.75. Each component is a clipped trailing
z-score,

$$
z_t = \operatorname{clip}\!\left(\frac{x_t - \mu_{t-W:t}}{\sigma_{t-W:t}},\, \pm 3\right),
\qquad W = 1260 \text{ sessions } (\approx 5\text{y})
$$

trailing-only, so no reading uses a bar that had not printed yet — a `--selftest`
lookahead check enforces this. Clipping at $3\sigma$ matters because March 2000
and March 2020 are 6–8 sigma events that would otherwise let one day set the
scale for a decade.

The composite is a weight-renormalised mean over whichever components exist
(HYG lists in 2007, every z needs two years of warm-up), then **re-scaled by
its own trailing standard deviation**. That second step is not cosmetic. For
$n$ standardised components with average pairwise correlation $\rho$,

$$
\operatorname{sd}\left(\frac{1}{n}\sum_i z_i\right)
= \sqrt{\frac{1 + (n-1)\rho}{n}}
$$

so the spread of the raw average depends on how correlated the components
happen to be *in that era* — and $\rho$ drifts. Dividing by the trailing std
keeps "+2 sigma" meaning the same thing in 2004 and in 2024, which is a
precondition for a fixed threshold to be meaningful at all.

**Selection is leave-one-out ablation, judged on both halves.** For every
component: rebuild the index without it, and run it alone; measure Sharpe over
a common window (measuring a HYG-only variant from HYG's 2007 inception would
skip the dot-com bust and flatter it). Keep a component only if dropping it
hurts **in both disjoint halves** (1999–2012 and 2013–now). Survivors keep
their original relative weights, so nothing is re-fitted.

On US data (QQQ vs SPY) that leaves **regime, momentum, rel_vol, rates**, and
the headline result is a negative one: the seed idea — the QQQ/SPY ratio's gap
from trend, a pure mean-reversion bet — is *negative in both halves*. A
stretched ratio has been a reason to keep holding, not to sell. `credit`,
`semis` and `concentration` flip sign between halves: never there.

The allocation rule gets the same treatment. Bucketing every historical reading
against the next 21 days of QQQ−SPY return gives a table that is **flat above
the line and negative below it**, in and out of sample — not a smooth ladder.
Above −0.5σ the Nasdaq wins by roughly the same margin whatever the reading, so
paying turnover to distinguish +0.6 from +2.0 buys nothing. Hence a two-state
switch rather than a continuous tilt, at a threshold deliberately left untuned
(Sharpe is 0.61–0.63 anywhere from −0.75 to 0.0 — a flat optimum is the only
kind worth trusting).

**And the same procedure on China disagrees.**
[`china_rotation/`](rotations/china_rotation/) runs STAR50 vs Dividend-Low-Vol
through the identical framework, and the US four-factor answer does *not*
transfer — the recommended reading keeps all eight components. With only ~5.5
years of traded history there is not enough sample to justify trimming, and
porting a subset derived from a different market would be borrowing a
conclusion rather than deriving one. Two markets, one procedure, two answers,
is the point of running both.

---

## Running it

```bash
cd screener
cp .env.example .env               # Alpaca keys; fundamentals come from yfinance
python alpha_lab.py --start 2021-01-01 --end 2026-07-21   # -> alpha_report.csv
python run_all.py                                          # screen + backtest + report
```

```bash
cd rotations
python -m tech_rotation.daily                    # today's decision
python -m tech_rotation.backtest --ablate        # what each component is worth
python -m china_rotation.backtest --ablate --ladder --profit
python -m china_rotation.backtest --selftest     # lookahead check
```

---

Research code, not investment advice. Backtested results are not a promise of
future returns, and §1.4 is the honest summary of how much of this is
statistically established: not much yet.
