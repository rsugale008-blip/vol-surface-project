# volatility/iv_solver.py
# ─────────────────────────────────────────
# Implied Volatility solver using
# Brent's method (robust + fast)
# ─────────────────────────────────────────

import numpy as np
from scipy.optimize import brentq
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from volatility.black_scholes import black_scholes_price
from config.settings import RISK_FREE_RATE


def calculate_iv(market_price, S, K, T, r=RISK_FREE_RATE,
                 option_type="call") -> float:
    """
    Calculate Implied Volatility using Brent's method.
    
    Parameters:
        market_price : Observed market price (use mid_price)
        S            : Spot price
        K            : Strike price
        T            : Time to expiry in years
        r            : Risk-free rate
        option_type  : "call" or "put"
    
    Returns:
        Implied volatility as decimal (e.g. 0.20 = 20%)
        Returns NaN if solver fails
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    # Intrinsic value check
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price < intrinsic:
        return np.nan

    try:
        # Brent's method — finds IV between 0.1% and 500%
        iv = brentq(
            lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type)
                          - market_price,
            a=0.001,   # min vol: 0.1%
            b=5.0,     # max vol: 500%
            xtol=1e-6,
            maxiter=1000
        )
        return round(iv, 6)

    except (ValueError, RuntimeError):
        return np.nan


def add_iv_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add implied volatility column to options dataframe.
    Applies IV solver row by row.
    """
    print("[IV Solver] Calculating implied volatilities...")

    ivs = []
    total = len(df)

    for i, row in df.iterrows():
        T = row["DTE"] / 365.0   # Convert days to years

        iv = calculate_iv(
            market_price=row["mid_price"],
            S=row["spot_price"],
            K=row["strike"],
            T=T,
            r=RISK_FREE_RATE,
            option_type=row["option_type"]
        )
        ivs.append(iv)

        # Progress indicator every 100 rows
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{total}...")

    df["iv"] = ivs

    # Drop rows where IV couldn't be solved
    before = len(df)
    df = df.dropna(subset=["iv"])
    df = df[df["iv"] > 0.01]   # Remove near-zero IVs
    df = df[df["iv"] < 3.0]    # Remove extreme IVs (>300%)
    after = len(df)

    print(f"[IV Solver] Done! {before} → {after} options with valid IV")
    print(f"[IV Solver] IV Range: {df['iv'].min():.2%} → {df['iv'].max():.2%}")
    print(f"[IV Solver] Avg IV  : {df['iv'].mean():.2%}")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    # ── Full pipeline test ──
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fetcher_yfinance import get_options_chain, filter_options

    print("=" * 50)
    print("  IV Solver Test — SPY")
    print("=" * 50)

    df = get_options_chain()
    df = filter_options(df)
    df = add_iv_to_dataframe(df)

    print("\n── Sample IV Data ──")
    print(df[["strike", "expiry", "option_type",
              "mid_price", "DTE", "iv"]].head(10))