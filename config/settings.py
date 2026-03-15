# config/settings.py
# ─────────────────────────────────────────
# Central config — change values here only
# ─────────────────────────────────────────

# Ticker to build vol surface for
TICKER = "SPY"

# Data source: "yfinance" | "tradier" | "polygon"
DATA_SOURCE = "yfinance"

# Risk-free rate (use current approx 3-month T-bill rate)
RISK_FREE_RATE = 0.05   # 5% — update as needed

# How many expiries to include in surface
MAX_EXPIRIES = 8

# Minimum open interest filter (remove illiquid strikes)
MIN_OPEN_INTEREST = 100

# Minimum volume filter
MIN_VOLUME = 10

# Dashboard auto-refresh interval (seconds)
REFRESH_INTERVAL = 60

# API Keys (fill in later when you get them)
TRADIER_TOKEN = "YOUR_TOKEN_HERE"
POLYGON_API_KEY = "YOUR_KEY_HERE"
