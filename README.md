# Quant Research

Three systematic equity projects I built around one question, given a pile of
plausible signals which ones do I actually keep?

I use a different selection procedure in each folder because the statistical
setting is different. This README covers those procedures and the math like the
per-project READMEs cover mechanics.

| Folder | Setting | How I select |
|---|---|---|
| [`screener/`](screener/) | ~1500-name cross-section, daily | Rank IC / ICIR with a sign-stability filter |
| [`rotations/tech_rotation/`](rotations/tech_rotation/) | 1 spread (QQQ vs SPY), 25y | A priori weights plus leave-one-out ablation, two halves |
| [`rotations/china_rotation/`](rotations/china_rotation/) | 1 spread (STAR50 vs DivLowVol), ~5.5y | Same, and I get the opposite answer |

---

## 1. Cross-sectional selection: IC, ICIR, and what they don't tell me

In [`screener/alpha_lab.py`](screener/alpha_lab.py) I test **29** candidate
factors, GTJA(Guotai) Alpha191, Alpha101 price/volume forms, classic anomalies
(12-1 momentum, short-term reversal, low-vol, 52-week-high proximity, negative
skew), and technical indicators like (RSI family, MACD family, Bollinger %B,
CMF, TRIX, Williams %R). I include this on purpose a selection rule I only ever
feed good candidates is a rule I haven't tested.

Published ICIRs I don't reuse. They were measured on other universes and eras
and horizons so I recompute everything on my own universe.

### 1.1 Conditioning the cross-section

I compute each factor for each ticker then **per date** winsorize by median
absolute deviation and standardise it:

$$
m = \mathrm{med}(f), \qquad
\mathrm{MAD} = \mathrm{med}\left(\lvert f_i - m \rvert\right), \qquad
\hat{\sigma}_{\mathrm{MAD}} = 1.4826 \cdot \mathrm{MAD}
$$

$$
\tilde{f}_i = \mathrm{clip}\!\left(f_i,\; m \pm k\,\hat{\sigma}_{\mathrm{MAD}}\right),
\qquad
z_i = \frac{\tilde{f}_i - \overline{\tilde{f}}}{s(\tilde{f})}, \qquad k = 5
$$

I use MAD rather than mean/std because $1.4826 = 1/\Phi^{-1}(0.75)$ makes it a
consistent estimator of $\sigma$ for Gaussian data so $k$ stays interpretable
in sigma units while the estimator has a 50% breakdown point and one blown-up
print cannot move my clip bounds which is exactly the thing a mean/std
winsorization would sit there and let it do.

The bars get cleaned first. Non-positive volume or price, `high < low`,
non-finite values, and frozen runs of $\ge 5$ identical closes like halts and
stale feeds masquerading as data, then a per-date eligibility gate
($P \ge \$5$, dollar volume $\ge \$1\mathrm{M}$) so illiquid names can't
dominate a cross-section I could never have traded.

### 1.2 The IC series

For each sampled date $t$ I take the **rank IC** the Spearman correlation
between the conditioned factor and the forward $h$-day return:

$$
\mathrm{IC}_t \;=\; \mathrm{corr}\!\left(\mathrm{rank}(z_{t}),\;
\mathrm{rank}(r_{t \to t+h})\right), \qquad h = 50
$$

Ranks not levels. My payoff is from a ranking which is invariant to any
monotone transform of the factor and the return has heavy enough tails that a
Pearson IC would largely tell me where the two biggest movers landed that day
and nothing else. Fewer than 20 eligible names and I skip the date. Every
factor is written so **higher = predicted better** so a positive IC means it
works as stated and a negative one means it works inverted.

I summarise the series $\{\mathrm{IC}_t\}_{t=1}^{T}$ with:

$$
\mathrm{ICIR} = \frac{\overline{\mathrm{IC}}}{s(\mathrm{IC})},
\qquad
t = \mathrm{ICIR}\sqrt{T}
$$

ICIR is the information ratio *of the signal itself* like how reliably the
factor ranks not how much it pays. It reaches portfolio performance through
Grinold's fundamental law $\mathrm{IR} \approx \mathrm{IC}\sqrt{\mathrm{BR}}$ a
small edge is only worth something spread across many independent bets which is
why I built the screener as a cross-sectional ranker holding a basket rather
than a single-name call.

### 1.3 My selection rule

I split the ICs into two halfs and keep a factor only if:

$$
\lvert \mathrm{ICIR} \rvert \ge 0.02
\quad\text{and}\quad
\mathrm{sign}\!\left(\overline{\mathrm{IC}}_{1}\right)
= \mathrm{sign}\!\left(\overline{\mathrm{IC}}_{2}\right)
$$

then take the top 5 survivors by $\lvert\mathrm{ICIR}\rvert$.

The second condition does nearly all the work. My $\lvert\mathrm{ICIR}\rvert$
floor is a noise gate that almost everything clears the sign test is the crude
but cheap stationarity check. Under a true constant edge both halves agree with
high probability under the null $\mathrm{IC}=0$ they agree with probability
$\tfrac{1}{2}$ so it roughly **halves the pass rate of pure noise while barely
touching a real effect** and that asymmetry is what I want when I'm screening
29 candidates at once. There is a decay figure too,
$\lvert \mathrm{IC}_2\rvert / \lvert \mathrm{IC}_1\rvert - 1$, so a factor that
keeps its sign but loses its magnitude is visible to me rather than silently
promoted.

Measured on 2021-01-01 to 2026-07-21, $h=50$, 269 sampled cross-sections:

| factor | ICIR | IC mean | t | % positive | half 1 to half 2 | verdict |
|---|---:|---:|---:|---:|---|---|
| `neg_skew_60` | 0.375 | 0.0322 | 6.15 | 65.8 | 0.060 to 0.005 | kept |
| `momentum_12_1` | 0.341 | 0.0533 | 5.28 | 65.4 | 0.031 to 0.075 | kept |
| `high52_prox` | 0.301 | 0.0540 | 4.67 | 68.5 | 0.045 to 0.063 | kept |
| `low_vol_20` | 0.294 | 0.0633 | 4.82 | 59.1 | 0.121 to 0.006 | kept |
| `atr_pct_14` | 0.293 | 0.0683 | 4.80 | 58.4 | 0.125 to 0.012 | kept |
| `ext_sma50` | 0.222 | 0.0289 | 3.64 | 58.0 | 0.068 to −0.010 | **sign flip** |
| `macd_line` | −0.174 | −0.0247 | −2.85 | 42.8 | −0.052 to 0.003 | **sign flip** |
| `rsi14_near50` | −0.088 | −0.0082 | −1.44 | 46.8 | 0.004 to −0.021 | **sign flip** |
| `macd_hist` | −0.006 | −0.0008 | −0.10 | 49.4 | −0.009 to 0.008 | **sign flip** |

Most of the RSI and MACD family like the indicators a discretionary screen
leans on hardest sit at $\lvert\mathrm{ICIR}\rvert < 0.1$ **and** flip sign
between halves. Never there in the first place.

### 1.4 Three things I know are wrong with this procedure

**My t-stats are inflated by overlap.** $t = \mathrm{ICIR}\sqrt{T}$ assumes
independent observations. With $h = 50$ trading days of forward return sampled
every 5 sessions each observation shares roughly $50/5 = 10$ neighbours'
returns so under a correction like Hansen-Hodrick or Newey-West my effective
sample is closer to $T/10$ and the honest scale is $t/\sqrt{10}$:

| factor | naive $t$ | overlap-adjusted $t$ |
|---|---:|---:|
| `neg_skew_60` | 6.15 | **1.94** |
| `momentum_12_1` | 5.28 | **1.67** |
| `low_vol_20` | 4.82 | **1.52** |
| `atr_pct_14` | 4.80 | **1.52** |
| `high52_prox` | 4.67 | **1.48** |

**And I don't correct for multiple testing.** Screening 29 candidates at
$\alpha = 0.05$ expects $29 \times 0.05 \approx 1.5$ false discoveries just by
construction and a Bonferroni threshold of $0.05/29$ needs $\lvert t\rvert
\gtrsim 3.1$ and *nothing in my pool clears that once overlap is accounted for*
so none of these are established alpha to me. They are ranking ingredients and
the portfolio backtest settles them downstream like it models position limits,
rebalancing, costs and compounding. That arbiter has overruled my per-pick
sweep before. It is why [`screener/ab_sweep.py`](screener/ab_sweep.py) warns
about its own metric.

**Ranking by ICIR ignores redundancy.** `low_vol_20` (negated 20-day return
std) and `atr_pct_14` (negated ATR/price) are the same volatility view computed
two ways and they land at ICIR 0.294 and 0.293 so because I rank marginally my
top 5 hands 40% of its weight to one idea. If the five survivors were really
independent I would get

$$
\mathrm{ICIR}_{\text{combined}} = \sqrt{\sum_i \mathrm{ICIR}_i^2} \approx 0.72
$$

but with two of them nearly collinear the real figure is lower. A
correlation-aware step like clustering the pool and keeping one representative
per cluster or a full $\Sigma^{-1}$ weighting is my next iteration.

### 1.5 From factors to a score

Two consumers downstream and I weight the factors differently in each on
purpose.

[`screener/month_predictor.py`](screener/month_predictor.py) weights **by
measured $\lvert\mathrm{ICIR}\rvert$** at the 20-day horizon:

$$
S = \sum_i \lvert\mathrm{ICIR}_i\rvert \cdot z_i
$$

which is the mean-variance optimum $w \propto \Sigma^{-1}\mu$ under
$\Sigma = \mathrm{diag}(\sigma_i^2)$ or exactly the independence assumption
§1.4 says I'm violating. I use it anyway because estimating a $5\times 5$
factor covariance from this sample would add more estimation error than it
removes.

[`screener/select_top15.py`](screener/select_top15.py) does **not** use ICIR
weights. Its eight component weights are set a priori (momentum 25, entry
quality 20, trend 15, RSI zone 10, low vol 10, quality 10, earnings gap 5,
liquidity 5) and most of them are **non-monotone** in the underlying variable
which is the important part:

$$
\mathrm{tri}(x; a, p, b) =
\begin{cases}
\dfrac{x-a}{p-a}, & a < x \le p \\[2ex]
\dfrac{b-x}{b-p}, & p < x < b \\[1ex]
0, & \text{otherwise}
\end{cases}
$$

A triangular sweet-spot not a rank. My thesis here is *buy consolidation inside
an uptrend* so more 6-month momentum is better only up to about 30% and a stock
25% above its 50-day SMA is worse than one sitting on it. Rank IC cannot
express that shape a factor that is good in the middle and bad at both ends has
an IC near zero so this score gets validated by backtest instead and the two
files disagree by design.

---

## 2. Time-series selection: ablation instead of IC

The [rotation indices](rotations/) put me in the opposite position. There is no
cross-section, one spread and one observation per day, so IC isn't available to
me and $T$ is small in the units that matter like leadership regimes last years
so 25 years of daily data is maybe a handful of independent events. Fit weights
here and I am curve-fitting with extra steps.

So the weights are a priori by role and never fitted. The two deliberately
disagreeing views (valuation vs momentum) and the regime gate get 1.00 and the
confirmations get 0.50 to 0.75. Each component is a clipped trailing z-score,

$$
z_t = \mathrm{clip}\!\left(\frac{x_t - \mu_{t-W:t}}{\sigma_{t-W:t}},\, \pm 3\right),
\qquad W = 1260 \text{ sessions } (\approx 5\text{y})
$$

trailing only so no reading uses a bar that hadn't printed yet and a
`--selftest` lookahead check enforces that. I clip at $3\sigma$ because March
2000 and March 2020 are 6 to 8 sigma events that would otherwise let one day
set my scale for a decade.

The composite is a weight-renormalised mean over whichever components exist
like HYG only lists in 2007 and every z needs two years of warm-up, then it
gets **re-scaled by its own trailing standard deviation**. That second step
isn't cosmetic. For $n$ standardised components with average pairwise
correlation $\rho$,

$$
\mathrm{sd}\left(\frac{1}{n}\sum_i z_i\right)
= \sqrt{\frac{1 + (n-1)\rho}{n}}
$$

so the spread of my raw average depends on how correlated the components happen
to be *in that era* and $\rho$ drifts around. Dividing by the trailing std
keeps "+2 sigma" meaning the same thing in 2004 and in 2024 which I need before
a fixed threshold means anything at all.

**I select by leave-one-out ablation judged on both halves.** For every
component I rebuild the index without it and run it alone and measure Sharpe
over a common window (measuring something like a HYG-only variant from HYG's
own 2007 inception would skip the dot-com bust and flatter it) and a component
stays only if dropping it hurts **in both halves** (1999 to 2012 and 2013 to
now). Survivors keep their original relative weights so nothing gets re-fitted.

On US data (QQQ vs SPY) that leaves **regime, momentum, rel_vol, rates**. My
headline result is a negative one. The seed idea, the QQQ/SPY ratio's gap from
trend which is a pure mean-reversion bet, is *negative in both halves* and a
stretched ratio has been a reason to keep holding not to sell. `credit`,
`semis` and `concentration` flip sign between halves.

The allocation rule gets the same treatment. Bucketing every historical reading
against the next 21 days of QQQ minus SPY return gives a table that is **flat
above the line and negative below it** in and out of sample and not a smooth
ladder at all so above −0.5σ the Nasdaq wins by roughly the same margin
whatever the reading and paying turnover to distinguish +0.6 from +2.0 buys me
nothing. Hence a two-state switch rather than a continuous tilt at a threshold
I deliberately left untuned (Sharpe is 0.61 to 0.63 anywhere from −0.75 to
0.0).

**And the same procedure on China disagrees with me.**
[`china_rotation/`](rotations/china_rotation/) runs STAR50 vs Dividend Low Vol
through the identical framework and my US four-factor answer does *not*
transfer so the reading I recommend keeps all eight components. About 5.5 years
of traded history is not the sample to justify trimming anything and porting a
subset I derived on a different market would be borrowing a conclusion rather
than deriving one.

---

## Running it

```bash
cd screener
cp .env.example .env               # Alpaca keys, fundamentals come from yfinance
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

## Tooling

I wrote this with [Claude Code](https://claude.com/claude-code) as a coding
assistant. The research questions, the selection rules and the decisions about
what to keep are mine and the assistant implemented them and iterated on them.

---

Research code not investment advice. Backtested results are not a promise of
future returns, §1.4 is my summary of how much of this is actually established.
