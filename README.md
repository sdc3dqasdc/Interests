# Quant Research

Three systematic equity projects I built around one question, given a pile of
plausible signals, which ones do I actually keep?

I use a different selection procedure in each folder because the statistical
setting is different in each case. This README covers those procedures and the
math behind them, the per-project READMEs cover the mechanics.

| Folder | Setting | How I select |
|---|---|---|
| [`screener/`](screener/) | ~1500 name cross section, daily | Rank IC / ICIR with a sign stability filter |
| [`rotations/tech_rotation/`](rotations/tech_rotation/) | 1 spread (QQQ minus SPY), 25y | A priori weights plus leave one out ablation, two halves |
| [`rotations/china_rotation/`](rotations/china_rotation/) | 1 spread (STAR50 minus DivLowVol), ~5.5y | Same, and I get the opposite answer |

---

## 1. Cross sectional selection, IC, ICIR, and what they don't tell me

In [`screener/alpha_lab.py`](screener/alpha_lab.py) I test **29** candidate
factors, GTJA Alpha 191 forms, Alpha 101 style price/volume forms, classic
academic anomalies (12-1 momentum, short term reversal, low vol, 52 week high
proximity, negative skew), and a deliberate grab bag of technical indicators
(RSI family, MACD family, Bollinger %B, CMF, TRIX, Williams %R). I include the
grab bag on purpose, a selection rule that I only ever feed good candidates
into is a rule I haven't really tested.

I don't reuse published ICIRs. They were measured on other universes, other
eras and other horizons, so I recompute everything on my own universe.

### 1.1 Conditioning the cross section

I compute each factor per ticker, then **per date** I winsorize the cross
section by median absolute deviation and standardise it:

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

I use MAD instead of mean and std because $1.4826 = 1/\Phi^{-1}(0.75)$ makes it
a consistent estimator of $\sigma$ for Gaussian data, which keeps $k$
interpretable in sigma units, and the estimator itself has a 50% breakdown
point so one blown up print cannot move my clip bounds, which is exactly the
thing a mean and std winsorization would happily let it do.

Before any of that I clean the bars (non positive volume or price,
`high < low`, non finite values, and frozen runs of $\ge 5$ identical closes,
which are halts and stale feeds masquerading as data), and I apply a per date
eligibility gate ($P \ge \$5$, dollar volume $\ge \$1\mathrm{M}$) so that
illiquid names can't dominate a cross section I could never have traded in
anyway.

### 1.2 The IC series

For each sampled date $t$ I take the **rank IC**, the Spearman correlation
between the conditioned factor and the forward $h$ day return:

$$
\mathrm{IC}_t \;=\; \mathrm{corr}\!\left(\mathrm{rank}(z_{t}),\;
\mathrm{rank}(r_{t \to t+h})\right), \qquad h = 50
$$

Ranks and not levels. My payoff comes from a ranking, which is invariant to any
monotone transform of the factor, and the return cross section has heavy enough
tails that a Pearson IC would mostly just tell me where the two biggest movers
landed that day. I skip any date with fewer than 20 eligible names. Every
factor is written so that **higher = predicted better**, so a positive IC means
it works as stated and a negative one means it works inverted.

I summarise the series $\{\mathrm{IC}_t\}_{t=1}^{T}$ with:

$$
\mathrm{ICIR} = \frac{\overline{\mathrm{IC}}}{s(\mathrm{IC})},
\qquad
t = \mathrm{ICIR}\sqrt{T}
$$

ICIR is the information ratio *of the signal itself*, how reliably the factor
ranks and not how much it pays. It reaches actual portfolio performance through
Grinold's fundamental law, $\mathrm{IR} \approx \mathrm{IC}\sqrt{\mathrm{BR}}$,
a small per name edge is only worth something once it is spread across many
independent bets, which is the reason I built the screener as a cross sectional
ranker holding a basket rather than as a single name call.

### 1.3 My selection rule

I split the IC series into two disjoint halves and I keep a factor only if:

$$
\lvert \mathrm{ICIR} \rvert \ge 0.02
\quad\text{and}\quad
\mathrm{sign}\!\left(\overline{\mathrm{IC}}_{1}\right)
= \mathrm{sign}\!\left(\overline{\mathrm{IC}}_{2}\right)
$$

and then I take the top $N=5$ survivors by $\lvert\mathrm{ICIR}\rvert$.

The second condition is doing nearly all of the work here. My
$\lvert\mathrm{ICIR}\rvert$ floor is a noise gate that almost everything
clears, the sign test is crude but it is cheap and it is a real stationarity
check. Under a true constant edge both halves agree with high probability,
under the null $\mathrm{IC}=0$ they agree with probability $\tfrac{1}{2}$, so
the filter roughly **halves the pass rate of pure noise while barely touching a
real effect**, and that asymmetry is what I want when I am screening 29
candidates at once. I also record a decay figure,
$\lvert \mathrm{IC}_2\rvert / \lvert \mathrm{IC}_1\rvert - 1$, so a factor that
keeps its sign but quietly loses its magnitude is visible to me instead of
getting promoted anyway.

Measured on 2021-01-01 to 2026-07-21, $h=50$, 269 sampled cross sections:

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

Most of the RSI and MACD family, which are the indicators a discretionary
screen leans on hardest, sit at $\lvert\mathrm{ICIR}\rvert < 0.1$ **and** they
flip sign between the halves. I read that as the signature of an effect that
was never there in the first place.

### 1.4 Three things I know are wrong with this procedure

**My t stats are inflated by overlap.** $t = \mathrm{ICIR}\sqrt{T}$ assumes
independent observations. With $h = 50$ trading days of forward return sampled
every 5 sessions each observation shares roughly $50/5 = 10$ neighbours'
returns, so under a Hansen Hodrick or Newey West style correction my effective
sample is closer to $T/10$ and the honest scale is $t/\sqrt{10}$:

| factor | naive $t$ | overlap adjusted $t$ |
|---|---:|---:|
| `neg_skew_60` | 6.15 | **1.94** |
| `momentum_12_1` | 5.28 | **1.67** |
| `low_vol_20` | 4.82 | **1.52** |
| `atr_pct_14` | 4.80 | **1.52** |
| `high52_prox` | 4.67 | **1.48** |

**And I don't correct for multiple testing.** Screening 29 candidates at
$\alpha = 0.05$ expects $29 \times 0.05 \approx 1.5$ false discoveries just by
construction. A Bonferroni threshold of $0.05/29$ needs $\lvert t\rvert
\gtrsim 3.1$ and *nothing in my pool clears that once the overlap is accounted
for*, so I don't treat any of these as established alpha. I treat them as
ranking ingredients whose value gets settled downstream by the portfolio
backtest, which models position limits, rebalancing, costs and compounding.
That arbiter has overruled my per pick sweep before, which is why
[`screener/ab_sweep.py`](screener/ab_sweep.py) warns about its own metric.

**Ranking by ICIR ignores redundancy.** `low_vol_20` (negated 20 day return
std) and `atr_pct_14` (negated ATR over price) are the same volatility view
computed two different ways and they land at ICIR 0.294 and 0.293. Because I
rank marginally, my top 5 ends up handing 40% of its weight to one idea. If the
five survivors really were independent I would get

$$
\mathrm{ICIR}_{\text{combined}} = \sqrt{\sum_i \mathrm{ICIR}_i^2} \approx 0.72
$$

but with two of them nearly collinear the real figure is lower than that. A
correlation aware step, clustering the pool and keeping one representative per
cluster, or a full $\Sigma^{-1}$ weighting, is my next iteration.

### 1.5 From factors to a score

I weight factors two different ways downstream and I do it on purpose.

[`screener/month_predictor.py`](screener/month_predictor.py) weights **by
measured $\lvert\mathrm{ICIR}\rvert$** at the 20 day horizon:

$$
S = \sum_i \lvert\mathrm{ICIR}_i\rvert \cdot z_i
$$

which is the mean variance optimum $w \propto \Sigma^{-1}\mu$ under
$\Sigma = \mathrm{diag}(\sigma_i^2)$, in other words exactly the independence
assumption that §1.4 says I am violating. I use it anyway, because estimating a
$5\times 5$ factor covariance out of this sample would add more estimation
error than it takes away.

[`screener/select_top15.py`](screener/select_top15.py) does **not** use ICIR
weights at all. I set its eight component weights a priori (momentum 25, entry
quality 20, trend 15, RSI zone 10, low vol 10, quality 10, earnings gap 5,
liquidity 5) and, the important part, most of them are **non monotone** in the
underlying variable:

$$
\mathrm{tri}(x; a, p, b) =
\begin{cases}
\dfrac{x-a}{p-a}, & a < x \le p \\[2ex]
\dfrac{b-x}{b-p}, & p < x < b \\[1ex]
0, & \text{otherwise}
\end{cases}
$$

A triangular sweet spot and not a rank. My thesis here is *buy consolidation
inside an uptrend*, so more 6 month momentum is only better up to about 30%,
and a stock sitting 25% above its 50 day SMA is worse than one sitting right on
it. Rank IC cannot express that shape at all, a factor that is good in the
middle and bad at both ends has an IC of about zero, so I validate this score
by backtest instead and the two files disagree by design.

---

## 2. Time series selection, ablation instead of IC

The [rotation indices](rotations/) put me in the opposite position. There is no
cross section, one spread and one observation per day, so IC is not available
to me at all, and $T$ is small in the units that actually matter, leadership
regimes last for years so 25 years of daily data is maybe a handful of
independent events. If I fitted weights here I would be curve fitting with
extra steps.

So I set the weights a priori by role and I never fit them. The two
deliberately disagreeing views (valuation against momentum) and the regime gate
get 1.00, the confirmations get 0.50 to 0.75. Each component is a clipped
trailing z score,

$$
z_t = \mathrm{clip}\!\left(\frac{x_t - \mu_{t-W:t}}{\sigma_{t-W:t}},\, \pm 3\right),
\qquad W = 1260 \text{ sessions } (\approx 5\text{y})
$$

trailing only, so no reading uses a bar that hadn't printed yet, and I enforce
that with a `--selftest` lookahead check. I clip at $3\sigma$ because March
2000 and March 2020 are 6 to 8 sigma events that would otherwise let a single
day set my scale for a whole decade.

The composite is a weight renormalised mean over whichever components exist on
that date (HYG lists in 2007, and every z needs two years of warm up), and then
it gets **re scaled by its own trailing standard deviation**. That second step
is not cosmetic. For $n$ standardised components with average pairwise
correlation $\rho$,

$$
\mathrm{sd}\left(\frac{1}{n}\sum_i z_i\right)
= \sqrt{\frac{1 + (n-1)\rho}{n}}
$$

so the spread of my raw average depends on how correlated the components happen
to be *in that particular era*, and $\rho$ drifts around over time. Dividing by
the trailing std is what keeps "+2 sigma" meaning the same thing in 2004 and in
2024, which I need to be true before a fixed threshold means anything at all.

**I select by leave one out ablation, judged on both halves.** For every
component I rebuild the index without it, and I also run it alone, and I
measure Sharpe over a common window (measuring a HYG only variant from HYG's
own 2007 inception would skip the dot com bust entirely and flatter it). I keep
a component only if dropping it hurts **in both disjoint halves** (1999 to 2012
and 2013 to now). The survivors keep their original relative weights so I re
fit nothing.

On US data (QQQ against SPY) that leaves me with **regime, momentum, rel_vol,
rates**, and my headline result is a negative one. The seed idea, the QQQ/SPY
ratio's gap from trend, which is a pure mean reversion bet, is *negative in
both halves*. In the US a stretched ratio has been a reason to keep holding and
not a reason to sell. `credit`, `semis` and `concentration` flip sign between
the halves, they were never there.

I give the allocation rule the same treatment. Bucketing every historical
reading against the next 21 days of QQQ minus SPY return gives a table that is
**flat above the line and negative below it**, in sample and out of sample,
which is not a smooth ladder at all. Above −0.5σ the Nasdaq wins by roughly the
same margin whatever the reading is, so paying turnover to distinguish +0.6
from +2.0 buys me nothing. Hence a two state switch rather than a continuous
tilt, at a threshold I deliberately left untuned (Sharpe is 0.61 to 0.63
anywhere from −0.75 to 0.0).

**And the same procedure on China disagrees with me.**
[`china_rotation/`](rotations/china_rotation/) runs STAR50 against Dividend Low
Vol through the identical framework and my US four factor answer does *not*
transfer, the reading I recommend there keeps all eight components. With only
about 5.5 years of traded history I don't have the sample to justify trimming
anything, and porting over a subset I derived on a different market would be
borrowing a conclusion instead of deriving one.

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
what to keep are mine, the assistant implemented them and iterated on them.

---

Research code and not investment advice. Backtested results are not a promise
of future returns, §1.4 is my own summary of how much of this is actually
established.
