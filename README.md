# 📈 SPY Live Volatility Surface System
### Quantitative Finance | Risk Management | AI-Powered Analytics

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Plotly](https://img.shields.io/badge/Plotly-5.x-green)
![Dash](https://img.shields.io/badge/Dash-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

A professional-grade **Live Implied Volatility Surface** system
built entirely in Python. Fetches live SPY options data,
calculates implied volatilities using Black-Scholes,
constructs a smooth 3D volatility surface, computes all
option Greeks, runs AI-powered anomaly detection, forecasts
next-day volatility using ML ensemble models, and displays
everything in a live auto-refreshing dashboard.

> Built as a portfolio project demonstrating quantitative
> finance, risk management, and AI/ML skills.

---

## 🖥️ Live Dashboard
```
┌─────────────────────────────────────────────┐
│  📈 SPY Live Volatility Surface             │
│  Spot: $662.29  |  Last Update: 16:51:04   │
├──────────┬─────────┬─────────┬──────┬───────┤
│  SPOT    │CALL IV  │ PUT IV  │SKEW  │OPTIONS│
│ $662.29  │ 27.8%   │ 32.3%   │+4.5% │  985  │
├──────────┴─────────┴─────────┴──────┴───────┤
│         3D VOLATILITY SURFACE               │
│         Red=High IV  |  Green=Low IV        │
├─────────────────────┬───────────────────────┤
│  ATM Term Structure │  Vol Smile            │
├─────────────────────┴───────────────────────┤
│         ATM Greeks vs DTE                   │
│  Delta | Gamma | Vega | Theta | Rho         │
└─────────────────────────────────────────────┘
```

---

## 🏗️ Architecture
```
vol_surface_project/
├── config/
│   └── settings.py          # Central config
├── data/
│   └── fetcher_yfinance.py  # Live options chain
├── volatility/
│   ├── black_scholes.py     # BS pricer + Greeks
│   ├── iv_solver.py         # Brent IV solver
│   └── surface_builder.py   # RBF surface
├── visualization/
│   ├── surface_plot.py      # Static 3D plot
│   └── dashboard.py         # Live Dash app
├── risk/
│   ├── var_calculator.py    # VaR / CVaR
│   ├── anomaly_detector.py  # AI spike detection
│   ├── greeks_surface.py    # Greeks surfaces
│   └── vol_forecaster.py    # ML forecasting
├── data_store/
│   ├── storage.py           # SQLite persistence
│   └── vol_history.db       # Historical data
├── main.py                  # Master controller
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start
```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
python main.py

# 3. Select mode
# [4] Full system — scheduler + dashboard

# 4. Open browser
# http://localhost:8050
```

---

## 🧮 Mathematical Foundation

### Black-Scholes
```
Call = S·N(d1) - K·e^(-rT)·N(d2)
d1   = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2   = d1 - σ·√T
```

### Implied Volatility
```
IV = σ : BS(S,K,T,r,σ) = Market Price
Solved via Brent's method
```

### Value at Risk
```
VaR(95%) = σ_daily · √T · Φ⁻¹(0.05) · Portfolio
CVaR     = E[Loss | Loss > VaR]
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| SPY Spot | $662.29 |
| ATM Call IV | 27.8% |
| ATM Put IV | 32.3% |
| Put-Call Skew | +4.5% |
| VaR 95% | ~$1,850 |
| ML Dir Accuracy | ~64% |
| Options Tracked | 985 |

---

## 🤖 AI/ML Components

| Component | Method | Purpose |
|-----------|--------|---------|
| Anomaly Detection | Isolation Forest | Flag abnormal IV |
| Spike Detection | Z-Score (2.5σ) | Real-time alerts |
| Vol Forecasting | XGBoost + LightGBM + RF Ensemble | Next-day IV |
| Surface Fitting | RBF Interpolation | Smooth surface |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Data | yfinance, pandas |
| Math | numpy, scipy |
| ML | scikit-learn, XGBoost, LightGBM |
| Visualization | Plotly, Dash |
| Storage | SQLite |
| Scheduling | APScheduler |

---

## 📚 References

- Black & Scholes (1973) — Option Pricing
- Hull (2022) — Options, Futures & Derivatives
- Gatheral (2006) — The Volatility Surface
- Brent (1973) — Root Finding Algorithms

