# Big Tech Stock Classifier

A machine learning pipeline that predicts which of the **8 most prominent US technology stocks** (MAG-7 + AMD) will be the best performer the following trading day, framed as an 8-class daily classification problem.

---

## Overview

| | |
|---|---|
| **Type** | Multi-class Classification |
| **Stocks** | AAPL, AMD, AMZN, GOOGL, META, MSFT, NVDA, TSLA |
| **Data** | 6 years of daily OHLCV (May 2020 – May 2026) |
| **Features** | 120 (percent change windows + technical indicators) |
| **Validation** | Expanding window time series cross-validation |

---

## Models Evaluated

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Softmax Regression | 14.7% | 14.3% |
| MLP Neural Network | 14.5% | 14.2% |
| **XGBoost** | **19.0%** | **17.0%** |
| | | |
| *Random Baseline* | *12.5%* | *N/A* |
| *Momentum Baseline* | *16.3%* | *N/A* |
| *Reversion Baseline* | *15.6%* | *N/A* |

**XGBoost outperforms all baselines**, achieving a ~52% lift over random and beating both naive momentum and reversion strategies.

---

## Technical Highlights

- **Temporal data integrity:** expanding window validation ensures no future data leaks into any training window, respecting the time series structure of financial data
- **Leakage-free scaling:** `StandardScaler` is refit on each training window and applied to the corresponding test point, preventing mean/std leakage across the temporal split
- **Feature engineering:** per-ticker percent change windows (1d, 5d, 21d) and technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV) computed via `pandas_ta`
- **Class imbalance analysis:** label distribution spans from TSLA (356 wins) to MSFT (84 wins); class reweighting was tested and empirically ruled out as it penalized the model for learning real market patterns
- **Live retraining:** designed to retrain from scratch on any given day using up-to-date data fetched directly from the `yfinance` API

---

## Stack

```
yfinance       # market data ingestion
pandas-ta      # technical indicator computation
scikit-learn   # Softmax Regression, MLP, preprocessing, metrics
xgboost        # gradient boosted classifier
pandas/numpy   # data manipulation
matplotlib     # confusion matrix visualization
pickle         # result persistence
```

---

## Project Structure

```
big-tech-stock-classifier/
├── data/
│   ├── raw_master.csv              # engineered feature dataset
│   └── results.pkl                 # serialized model predictions
├── src/
│   ├── 1_data.ipynb                    # data ingestion + feature engineering
│   └── 2_models.ipynb                  # expanding window training + evaluation
└── docs/
    └── Kuchar_ML_Final_Report.pdf  # full technical write-up
```

---

## Quickstart

```bash
git clone https://github.com/gkuchar/big-tech-stock-classifier
cd big-tech-stock-classifier
pip install -r requirements.txt
```

Run `1_data.ipynb` to fetch current data, then `2_models.ipynb` to retrain and evaluate. Set `RETRAIN = False` in notebook 2 to load saved results without retraining.

---

## Key Takeaways

- Tree-based models (XGBoost) outperform linear and neural approaches on small, noisy tabular financial data
- Temporal validation is non-negotiable for time series ML; random splits introduce data leakage and produce artificially inflated accuracy
- A 52% lift over random in an 8-class daily stock prediction task is a meaningful result, not noise