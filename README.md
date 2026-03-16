
# Market Sentiment Analysis: Fear/Greed Index × Hyperliquid Trader Performance

![Status](https://img.shields.io/badge/status-complete-brightgreen) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Overview

This project investigates **how cryptocurrency market sentiment (measured by the Fear & Greed Index) impacts trader behavior and profitability** on the Hyperliquid perpetual futures exchange. Through comprehensive data preparation, exploratory analysis, predictive modeling, and trader segmentation, we uncover actionable strategies for different trader archetypes.

**Key Finding:** Market sentiment measurably affects trader performance and behavioral patterns. Fear days show 23.6% higher average PnL vs Greed days ($39K vs $16K), with traders significantly adjusting trade frequency, position sizing, and long/short bias.

---

## 🔍 Part A: Data Preparation & Feature Engineering

### Datasets
- **Fear & Greed Index (2,644 daily observations, Feb 2018 – May 2025)**
  - Sentiment classifications: Extreme Fear, Fear, Neutral, Greed, Extreme Greed
  - FGI values: 0–100 scale
  - Source: Crypto Fear & Greed Index

- **Hyperliquid Trader Data (211,224 trades, 32 accounts, 246 coins)**
  - Individual trade data: entry/exit prices, sizes, PnL, timestamps
  - Features: trade frequency, win rate, leverage usage, long/short ratio

### Data Cleaning & Preparation
✅ **Zero missing values** after merging  
✅ **No duplicates** detected  
✅ **Timestamp alignment:** All trades matched to daily FGI values  
✅ **Outlier handling:** Valid trades retained; only 6 records without sentiment matched were dropped (211,218 final records)

### Key Features Engineered

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| **Daily PnL per Trader** | Sum of Closed PnL per account/day | Performance metric |
| **Win Rate** | Profitable Trades / Total Trades | Trading quality |
| **Leverage Proxy** | Position Size × Entry Price / Account Equity | Risk exposure |
| **Long/Short Ratio** | Buy Orders / Sell Orders | Directional bias |
| **Trade Frequency** | Trades per day per account | Activity level |
| **Average Trade Size (USD)** | Σ(Trade Size) / # Trades | Position sizing |

---

## 📈 Part B: Analysis & Key Insights

### **Insight 1: Sentiment Has Measurable Impact on Performance**

**Evidence:**
- **Fear Days** → Avg PnL: **$39,012** | Win Rate: **84.2%** | Trades: 105/day
- **Greed Days** → Avg PnL: **$15,848** | Win Rate: **85.6%** | Trades: 77/day
- **Mann-Whitney U Test:** p-value = 0.0162 ✓ *Statistically significant*

**Interpretation:** During fear-driven markets, traders collectively make more money per day despite lower win rates. This suggests more disciplined risk-taking and selective trade entry during fearful periods.

### **Insight 2: Traders Dynamically Adjust Behavior by Sentiment**

**Trade Frequency Shift:**
- Extreme Fear → 133.8 trades/day
- Greed → 77.6 trades/day (**−42% reduction**)

**Long/Short Ratio (Median):**
- Fear → 0.17 (heavily SHORT-biased)
- Greed → 0.11 (even more SHORT-biased)

**Leverage Proxy:**
- Extreme Fear → 7.1x (high risk-taking)
- Greed → 4.2x (reduced leverage)

**Interpretation:** Traders paradoxically *increase* activity and leverage during market fear—betting on reversals—while reducing both during greedy periods (profit-taking behavior).

### **Insight 3: Trader Segmentation Reveals Distinct Archetypes**

**High-Leverage Traders:**
- Fear days: +$8,988 avg PnL
- Greed days: +$5,801 avg PnL
- **Most vulnerable in neutral/greed regimes** (60% decline)

**Infrequent Traders:**
- Extreme Fear: +$3,661 avg PnL (best performance window)
- **Contrarian advantage:** Patient capital rewards panic selling

**Consistent Winners (13% of traders):**
- 91.6% avg win rate
- $9,731 avg daily PnL
- Low consistency variance (0.48 Sharpe-like ratio)

---

## 🤖 Part C: Predictive Modeling & Clustering

### Machine Learning Models

**Objective:** Predict next-day profitability (binary: Profitable/Unprofitable)

**Models Trained:**
1. **Random Forest Classifier** (n_estimators=200, max_depth=8)
   - Accuracy: 85%
   - Precision (Profitable): 85%
   - Recall: 100% (catches all wins)

2. **Gradient Boosting** (n_estimators=200, max_depth=4)
   - Identical performance due to class imbalance (80% profitable days)
   - Better generalization on minority class

**Top 5 Feature Importances:**
1. FGI Value (Fear & Greed Index)
2. 7-day rolling PnL mean
3. Lagged Win Rate (t-1)
4. Active trader count
5. Sentiment encoded (5 categories)

### Trader Behavioral Archetypes (K-Means Clustering)

| Archetype | Size | Trades/Day | Avg PnL | Win Rate | Consistency |
|-----------|------|-----------|---------|----------|-------------|
| **Conservative Winners** | 13 | 75.6 | $9,731 | 91.6% | 0.482 |
| **Aggressive Mid-Tier** | 17 | 91.2 | $2,549 | 77.2% | 0.130 |
| **Ultra-High Frequency** | 2 | 535.3 | $26,430 | 90.8% | 0.462 |

**Key Finding:** The 2 ultra-high-frequency traders dominate by volume and profit, suggesting *scale-based advantages* (possibly algorithmic/bot trading).

---

## 🎯 Actionable Strategies

### Strategy 1: Sentiment-Adaptive Position Sizing

**Rule:**
```
IF sentiment IN [Extreme Fear, Fear]:
  REDUCE position size by 30% for high-leverage traders
  INCREASE stop-loss frequency to catch reversals
ELSE IF sentiment IN [Greed]:
  TIGHTEN stop-losses by 15% (override confidence)
  REDUCE leverage by 25% (crowd is overleveraged)
```

**Evidence:** High-leverage traders show 5.4x higher PnL volatility in fear regimes.

### Strategy 2: Contrarian Frequency Play

**For Infrequent Traders:**
```
IF sentiment == Extreme Fear:
  INCREASE trade count by 25% (historically $3,661 avg PnL)
  FOCUS on counter-trend entries
  
IF sentiment == Extreme Greed:
  HOLD or reduce activity (avg PnL drops to $1,882)
```

**For Frequent Traders:**
```
IF sentiment == Extreme Greed:
  REDUCE daily trades by 30% (crowd over-trading)
  MOVE to 6h/daily timeframes (avoid noise)
  
IF sentiment == Extreme Fear:
  MAINTAIN activity but reduce size (stay fluid)
```

### Strategy 3: Segment-Specific Risk Management

- **Consistent Winners:** Trade normally (91.6% win rate); no adjustments needed
- **Mid-Tier Traders:** Lock in profits on +40% return days; reduce leverage in greed
- **Net Losers:** STRICT position sizing + win rate thresholds before trading

---

## 📊 Visualizations Summary

### Part A: Data Overview (`part_a_data_overview.png`)
- Trade count by account
- Coin distribution (246 unique assets)
- Date range alignment (2023-2025 overlap with FGI)
- Missing value heatmap

### Part B: Performance Analysis (`part_b_performance_by_sentiment.png`)
- **4-panel layout:**
  1. Average daily PnL by sentiment category
  2. Win rate across sentiments
  3. 5th percentile drawdown (risk metric)
  4. PnL distribution (fear vs greed comparison)

### Part B: Behavioral Patterns (`part_b_behavior_by_sentiment.png`)
- Trade frequency by sentiment
- Average trade size trends
- Long/short ratio deviations
- Median leverage proxy

### Part B: Segmentation (`part_b_segmentation.png`)
- 6-panel trader segment analysis
- PnL by leverage/frequency/performance × sentiment
- Win rate heatmaps
- Trader scatter plot (win rate vs total PnL)

### Part C: Predictive Model (`part_c_predictive_model.png`)
- Feature importance barplot (top 10)
- Confusion matrix (RF & GB models)
- Class balance issues highlighted

### Part C: Clustering (`part_c_clustering.png`)
- Elbow curve (K=2 to 8)
- PCA projection of 3 clusters
- Archetype identification

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pandas, numpy, matplotlib, seaborn
scikit-learn (clustering, predictive models)
streamlit (for dashboard)
```

### Installation & Usage

**1. Clone & Setup**
```bash
git clone https://github.com/ShubhamP1028/Market-Sentiment-Analysis.git
cd Market-Sentiment-Analysis
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
```

**2. Run Notebooks (In Order)**
```bash
# Data preparation
jupyter notebook Part_A_Data_Preparation.ipynb

# Analysis & insights
jupyter notebook Part_B_Analysis.ipynb

# Modeling & actionable output
jupyter notebook Part_C_Actionable_Output_Bonus.ipynb
```

**3. Launch Interactive Dashboard**
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

**4. Key Outputs Generated**
- `daily_aggregate_metrics.csv` — Market-level summary (use in models)
- `trader_segments.csv` — Trader profiles (for segmentation)
- `daily_segmented.csv` — Daily data with segment labels (for filtering)
- PNG plots — Visualization reports (save for presentations)

---

## 📖 File Descriptions

### CSVs
- **`merged_data.csv`** (211K rows): Individual trades with FGI sentiment matched
- **`daily_aggregate_metrics.csv`** (479 rows): Daily totals (all traders combined)
- **`daily_trader_metrics.csv`** (2,340 rows): Per-trader daily rollup
- **`trader_segments.csv`** (32 rows): Account-level stats + cluster assignment
- **`daily_segmented.csv`** (2,340 rows): Daily data + leverage/frequency/performance labels

### Notebooks
| Notebook | Runtime | Output |
|----------|---------|--------|
| Part A | ~10s | Feature-engineered CSVs |
| Part B | ~30s | Analysis plots + segment tables |
| Part C | ~1m | Predictive models + clustering + dashboard.py |

---

## 🔬 Technical Details

### Sentiment Mapping
```python
Extreme Fear (0-25)   → "Fear"
Fear (25-45)          → "Fear"
Neutral (45-55)       → "Neutral"
Greed (55-75)         → "Greed"
Extreme Greed (75+)   → "Greed"
```

### Consistency Metric (Sharpe-like)
```
consistency = avg_pnl / std_pnl
Higher = more predictable returns
```

### Leverage Proxy
```
leverage_proxy = (total_position_notional / account_equity)
Approximates effective leverage from trading size patterns
```

---

## ⚠️ Limitations & Caveats

1. **Selection Bias:** Only 32 Hyperliquid accounts analyzed (may not represent all traders)
2. **Lookback Period:** May 2023 – May 2025 (2 years of data; limited bull/bear coverage)
3. **Confounding Factors:** Market volatility, liquidation cascades not isolated
4. **PnL Attribution:** Cannot distinguish skill vs. luck in short windows
5. **Class Imbalance:** 80% of days were profitable (oversampling risk in models)

---

## 📚 References

- **Fear & Greed Index:**
- **Hyperliquid API:** 
- **Clustering:** K-Means on standardized trader features
- **Models:** Scikit-learn RandomForest & GradientBoosting classifiers

---

## 📧 Contact & Feedback
  
**Linkedin:** [Shubham1028](https://linkedin.com/in/Shubham1028)
