# risk/var_calculator.py
# ─────────────────────────────────────────
# VaR / CVaR calculator from vol surface
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TICKER, RISK_FREE_RATE


def calculate_var_cvar(df: pd.DataFrame,
                       confidence: float = 0.95,
                       horizon_days: int = 1,
                       portfolio_value: float = 100000) -> dict:
    """
    Calculate Value at Risk and Conditional VaR
    using implied volatility from the surface.

    Parameters:
        df              : Options dataframe with IV
        confidence      : VaR confidence level (0.95 = 95%)
        horizon_days    : Time horizon in days
        portfolio_value : Portfolio size in USD

    Returns:
        dict with VaR, CVaR, and scenario analysis
    """
    spot    = df["spot_price"].iloc[0]
    avg_iv  = df["iv"].mean()
    atm_df  = df[df["strike"].between(spot * 0.99, spot * 1.01)]
    atm_iv  = atm_df["iv"].mean() if len(atm_df) > 0 else avg_iv

    # Daily vol scaling
    daily_vol = atm_iv / np.sqrt(252)
    horizon_vol = daily_vol * np.sqrt(horizon_days)

    # Monte Carlo simulation
    np.random.seed(42)
    n_simulations = 100000
    returns = np.random.normal(0, horizon_vol, n_simulations)

    # VaR and CVaR
    var_pct  = np.percentile(returns, (1 - confidence) * 100)
    cvar_pct = returns[returns <= var_pct].mean()

    var_usd  = abs(var_pct)  * portfolio_value
    cvar_usd = abs(cvar_pct) * portfolio_value

    # Scenario analysis
    scenarios = {
        "Base (ATM IV)":    atm_iv * 100,
        "Stress +25% Vol":  atm_iv * 1.25 * 100,
        "Stress +50% Vol":  atm_iv * 1.50 * 100,
        "Crisis +100% Vol": atm_iv * 2.00 * 100,
    }

    scenario_vars = {}
    for name, vol_pct in scenarios.items():
        s_vol     = (vol_pct / 100) / np.sqrt(252) * np.sqrt(horizon_days)
        s_returns = np.random.normal(0, s_vol, n_simulations)
        s_var     = abs(np.percentile(s_returns, (1 - confidence) * 100))
        scenario_vars[name] = round(s_var * portfolio_value, 2)

    results = {
        "ticker"          : TICKER,
        "spot"            : spot,
        "atm_iv_pct"      : round(atm_iv * 100, 2),
        "daily_vol_pct"   : round(daily_vol * 100, 2),
        "confidence"      : confidence,
        "horizon_days"    : horizon_days,
        "portfolio_value" : portfolio_value,
        "var_pct"         : round(abs(var_pct) * 100, 4),
        "cvar_pct"        : round(abs(cvar_pct) * 100, 4),
        "var_usd"         : round(var_usd, 2),
        "cvar_usd"        : round(cvar_usd, 2),
        "scenario_vars"   : scenario_vars
    }

    return results


def print_risk_report(results: dict):
    """Print formatted risk report."""
    print("\n" + "=" * 55)
    print(f"  RISK REPORT — {results['ticker']}")
    print("=" * 55)
    print(f"  Spot Price       : ${results['spot']:.2f}")
    print(f"  ATM IV           : {results['atm_iv_pct']:.2f}%")
    print(f"  Daily Vol        : {results['daily_vol_pct']:.2f}%")
    print(f"  Confidence Level : {results['confidence']*100:.0f}%")
    print(f"  Horizon          : {results['horizon_days']} day(s)")
    print(f"  Portfolio Value  : ${results['portfolio_value']:,.0f}")
    print("-" * 55)
    print(f"  VaR  ({results['confidence']*100:.0f}%)     : "
          f"${results['var_usd']:,.2f} "
          f"({results['var_pct']:.3f}%)")
    print(f"  CVaR ({results['confidence']*100:.0f}%)     : "
          f"${results['cvar_usd']:,.2f} "
          f"({results['cvar_pct']:.3f}%)")
    print("-" * 55)
    print("  SCENARIO ANALYSIS:")
    for name, var in results["scenario_vars"].items():
        print(f"  {name:<22}: ${var:>12,.2f}")
    print("=" * 55)


if __name__ == "__main__":
    from data.fetcher_yfinance import get_options_chain, filter_options
    from volatility.iv_solver import add_iv_to_dataframe

    df      = get_options_chain()
    df      = filter_options(df)
    df      = add_iv_to_dataframe(df)
    results = calculate_var_cvar(df, confidence=0.95,
                                 horizon_days=1,
                                 portfolio_value=100000)
    print_risk_report(results)