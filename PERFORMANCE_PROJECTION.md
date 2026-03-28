# ClawdBot — Theoretical Performance Projection
### Expected Value & ROI Analysis

*Date: 2026-03-28 | Author: Financial Risk Analysis*

---

## 1. System Parameters

| Parameter | Value |
|---|---|
| Starting Capital | $10,500.00 USDT |
| Max Concurrent Positions | 3 |
| Margin per Position | ~$700.00 USDT (≈ 6.7% of capital) |
| Leverage | 5× |
| **Notional Position Size** | **$3,500.00 USDT** |
| Assumed Win Rate | 55% (ML Conf > 55%, Gemini Sentiment > −0.1) |
| Take-Profit Target | +1.5% unleveraged market move |
| Stop-Loss Target | −0.5% unleveraged market move |
| Binance Open Fee | 0.10% of notional |
| Binance Close Fee | 0.10% of notional |

---

## 2. The Math Per Trade

### Round-Trip Fee Cost

```
Fee per trade = (Open Fee + Close Fee) × Notional Position Size
             = (0.10% + 0.10%) × $3,500.00
             = 0.20% × $3,500.00
             = $7.00 USDT (flat, regardless of win or loss)
```

### Average Winning Trade

```
Gross Profit  = Take-Profit % × Notional Position Size
              = 1.50% × $3,500.00
              = $52.50 USDT

Net Profit    = Gross Profit − Round-Trip Fee
              = $52.50 − $7.00
              = +$45.50 USDT
```

### Average Losing Trade

```
Gross Loss    = Stop-Loss % × Notional Position Size
              = 0.50% × $3,500.00
              = $17.50 USDT

Net Loss      = Gross Loss + Round-Trip Fee
              = $17.50 + $7.00
              = −$24.50 USDT
```

### Effective Risk/Reward Ratio (Net of Fees)

```
Gross R:R  = $52.50 / $17.50 = 3.00 : 1
Net R:R    = $45.50 / $24.50 = 1.86 : 1
```

> **Key Observation:** Fees compress the risk/reward ratio from 3.00:1 (gross)
> down to 1.86:1 (net). The $7.00 fee adds 40% to the cost of a losing trade
> while removing 13.3% from the reward of a winning trade.

---

## 3. Expected Value (EV) Per Trade

The mathematical expectancy of the system on a per-trade basis:

```
EV = (Win Rate × Net Profit per Win) + (Loss Rate × Net Loss per Loss)
   = (0.55 × $45.50)  +  (0.45 × −$24.50)
   = $25.025           −  $11.025
   = +$14.00 USDT per trade
```

### Break-Even Win Rate Analysis

The minimum win rate required to remain profitable, with and without fees:

```
Without fees:  W × $52.50 = (1 − W) × $17.50  →  W = 17.50 / 70.00 = 25.0%
With fees:     W × $45.50 = (1 − W) × $24.50  →  W = 24.50 / 70.00 = 35.0%
```

> Fees raise the break-even win rate from **25%** to **35%**. At the assumed
> 55% win rate the system has a healthy **20 percentage-point cushion** above
> break-even.

---

## 4. The 100-Trade Scenario

Assumptions: exactly 100 trades executed — **55 wins**, **45 losses**.

| Metric | Calculation | Result |
|---|---|---|
| Gross wins | 55 × $52.50 | +$2,887.50 |
| Gross losses | 45 × $17.50 | −$787.50 |
| Total fees | 100 × $7.00 | −$700.00 |
| **Net P&L** | $2,887.50 − $787.50 − $700.00 | **+$1,400.00** |

*Alternatively, using net-per-trade figures:*

```
Profit from wins  = 55 × $45.50 = $2,502.50
Loss from losses  = 45 × $24.50 = $1,102.50
                                 ----------
Net P&L           =              +$1,400.00 USDT
```

### Final Account Balance

```
Final Balance = Starting Capital + Net P&L
             = $10,500.00 + $1,400.00
             = $11,900.00 USDT
```

### Return on Capital

```
ROI (100 trades) = Net P&L / Starting Capital
                 = $1,400.00 / $10,500.00
                 = +13.33%
```

### Summary Table

| Metric | Value |
|---|---|
| Starting Balance | $10,500.00 USDT |
| Total Gross Revenue (55 wins) | +$2,887.50 USDT |
| Total Gross Losses (45 losses) | −$787.50 USDT |
| Total Fees (100 trades × $7.00) | −$700.00 USDT |
| **Net P&L** | **+$1,400.00 USDT** |
| **Final Balance** | **$11,900.00 USDT** |
| **ROI** | **+13.33%** |
| EV per Trade | +$14.00 USDT |

---

## 5. Risk Analysis & Fee-Impact Assessment

### Is the Fee Structure Eating Too Much of Our Profits?

**Yes — fees are material and disproportionately hurt losing trades.**

| | Winning Trade | Losing Trade |
|---|---|---|
| Gross P&L | +$52.50 | −$17.50 |
| Fee | −$7.00 | −$7.00 |
| Net P&L | +$45.50 | −$24.50 |
| Fee as % of gross move | **13.3%** | **40.0%** |

The $7.00 round-trip fee represents **40% of the gross stop-loss amount**. On a
mean-reversion strategy where the stop-loss is intentionally tight (0.5%), this
is extremely significant. Every losing trade costs 40% more than the raw market
move warrants.

### Should We Increase the Minimum Take-Profit Target?

**Yes — raising the TP target meaningfully improves the net risk/reward ratio.**

The table below illustrates the effect of increasing the TP target while keeping
all other parameters constant:

| TP Target | Gross Win | Net Win | Net R:R | EV/Trade | 100-Trade Net P&L | ROI |
|---|---|---|---|---|---|---|
| +1.0% | $35.00 | $28.00 | 1.14:1 | +$3.83 | +$383.33 | +3.65% |
| **+1.5% (current)** | **$52.50** | **$45.50** | **1.86:1** | **+$14.00** | **+$1,400.00** | **+13.33%** |
| +2.0% | $70.00 | $63.00 | 2.57:1 | +$23.63 | +$2,362.50 | +22.50% |
| +2.5% | $87.50 | $80.50 | 3.29:1 | +$33.28 | +$3,277.50 | +31.21% |

> Moving the TP from **+1.5% → +2.0%** improves the 100-trade net P&L by
> **+$962.50** (+68.75%) and ROI from 13.33% to 22.50% — a substantial gain
> for a single 0.5% tweak to the target.

### Minimum TP to Restore the 3:1 Gross Risk/Reward Net of Fees

If the goal is to maintain a **3:1 net** risk/reward ratio:

```
Required Net Win = 3 × Net Loss
                 = 3 × $24.50
                 = $73.50

Required Gross Win = Net Win + Fee
                   = $73.50 + $7.00
                   = $80.50

Required TP %     = Gross Win / Notional
                  = $80.50 / $3,500.00
                  = 2.30%
```

> **Recommendation:** Set a minimum Take-Profit target of **+2.3%** to restore
> the pre-fee 3:1 risk/reward ratio on a net basis.

---

## 6. Executive Summary

| Finding | Detail |
|---|---|
| System is **mathematically profitable** | Positive EV of +$14.00/trade at 55% win rate |
| 100-trade projected return | **+13.33% ROI** ($1,400 net P&L on $10,500 capital) |
| Fee cost is significant | $700 total fees consume **50%** of gross losses and **24.2%** of total gross wins |
| Tight stop-loss amplifies fee drag | $7.00 fee = 40% of the $17.50 gross SL amount |
| **Recommended action** | Raise minimum TP target to **+2.0% – +2.3%** to meaningfully improve net R:R and reduce fee drag as a proportion of profits |

---

*All figures are deterministic projections based on the stated parameters.
Actual results will vary due to slippage, partial fills, market gaps, and
real-world win-rate deviation from the assumed 55% target.*
