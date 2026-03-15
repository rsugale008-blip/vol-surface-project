# data/fetcher_yfinance.py
# ─────────────────────────────────────────
# Fetches live options chain + spot price
# using yfinance (free, no API key needed)
# ─────────────────────────────────────────

import yfinance as yf
import pandas as pd
from datetime import datetime
import sys
import os

# So we can import from config/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TICKER, MAX_EXPIRIES, MIN_OPEN_INTEREST, MIN_VOLUME


def get_spot_price(ticker: str = TICKER) -> float:
    """Fetch current spot price of underlying."""
    stock = yf.Ticker(ticker)
    price = stock.fast_info["last_price"]
    print(f"[Fetcher] {ticker} Spot Price: ${price:.2f}")
    return price


def get_expiry_dates(ticker: str = TICKER) -> list:
    """Get all available option expiry dates."""
    stock = yf.Ticker(ticker)
    expiries = stock.options
    # Limit to MAX_EXPIRIES
    expiries = expiries[:MAX_EXPIRIES]
    print(f"[Fetcher] Found {len(expiries)} expiries: {expiries}")
    return expiries


def get_options_chain(ticker: str = TICKER) -> pd.DataFrame:
    """
    Fetch full options chain for all expiries.
    Returns a clean DataFrame with calls + puts.
    """
    stock = yf.Ticker(ticker)
    expiries = get_expiry_dates(ticker)
    spot = get_spot_price(ticker)

    all_options = []

    for expiry in expiries:
        try:
            chain = stock.option_chain(expiry)

            # --- Process CALLS ---
            calls = chain.calls.copy()
            calls["option_type"] = "call"
            calls["expiry"] = expiry

            # --- Process PUTS ---
            puts = chain.puts.copy()
            puts["option_type"] = "put"
            puts["expiry"] = expiry

            all_options.append(calls)
            all_options.append(puts)

        except Exception as e:
            print(f"[Fetcher] Error fetching {expiry}: {e}")
            continue

    # Combine all expiries
    df = pd.concat(all_options, ignore_index=True)

    # Add spot price column
    df["spot_price"] = spot

    # Add days to expiry (DTE)
    df["expiry_date"] = pd.to_datetime(df["expiry"])
    df["DTE"] = (df["expiry_date"] - pd.Timestamp.now()).dt.days

    # Add mid price (avg of bid + ask)
    df["mid_price"] = (df["bid"] + df["ask"]) / 2

    # Add moneyness (strike / spot)
    df["moneyness"] = df["strike"] / df["spot_price"]

    print(f"[Fetcher] Total options fetched: {len(df)}")
    return df


def filter_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove illiquid / bad data options.
    Filters by open interest, volume, price.
    """
    original = len(df)

    # Remove zero/NaN bid-ask
    df = df[df["bid"] > 0]
    df = df[df["ask"] > 0]

    # Remove wide spreads (ask > 3x bid — illiquid)
    df = df[df["ask"] <= df["bid"] * 3]

    # Open interest filter
    df = df[df["openInterest"] >= MIN_OPEN_INTEREST]

    # Volume filter (allow NaN volume — common in yfinance)
    df = df[(df["volume"] >= MIN_VOLUME) | (df["volume"].isna())]

    # Remove very short DTE (< 1 day) — expiry noise
    df = df[df["DTE"] >= 1]

    # Remove deep ITM / OTM (moneyness between 0.7 and 1.3)
    df = df[(df["moneyness"] >= 0.7) & (df["moneyness"] <= 1.3)]

    print(f"[Filter] {original} → {len(df)} options after filtering")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    # ── Quick test — run this file directly ──
    print("=" * 50)
    print(f"  Vol Surface Data Fetcher — {TICKER}")
    print("=" * 50)

    df_raw = get_options_chain()
    df_clean = filter_options(df_raw)

    print("\n── Sample Data ──")
    print(df_clean[["strike", "expiry", "option_type",
                     "bid", "ask", "mid_price",
                     "DTE", "moneyness", "openInterest"]].head(10))

    print(f"\n── Expiries in clean data ──")
    print(df_clean["expiry"].unique())

    print(f"\n── Shape: {df_clean.shape} ──")
    