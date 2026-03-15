# volatility/black_scholes.py
# ─────────────────────────────────────────
# Black-Scholes pricer + Greeks calculator
# ─────────────────────────────────────────

import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type="call") -> float:
    """
    Calculate Black-Scholes option price.
    
    Parameters:
        S     : Spot price
        K     : Strike price
        T     : Time to expiry in years
        r     : Risk-free rate
        sigma : Volatility (decimal, e.g. 0.20 = 20%)
        option_type: "call" or "put"
    
    Returns:
        Option price (float)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def calculate_greeks(S, K, T, r, sigma, option_type="call") -> dict:
    """
    Calculate all Greeks for an option.
    
    Returns dict with: delta, gamma, vega, theta, rho
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Gamma and Vega are same for calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol move

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    else:
        delta = norm.cdf(d1) - 1
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "vega" : round(vega,  6),
        "theta": round(theta, 6),
        "rho"  : round(rho,   6)
    }


if __name__ == "__main__":
    # ── Quick test ──
    S, K, T, r, sigma = 662.29, 660.0, 1/365, 0.05, 0.20

    call_price = black_scholes_price(S, K, T, r, sigma, "call")
    put_price  = black_scholes_price(S, K, T, r, sigma, "put")
    greeks     = calculate_greeks(S, K, T, r, sigma, "call")

    print(f"SPY Call Price : ${call_price:.4f}")
    print(f"SPY Put  Price : ${put_price:.4f}")
    print(f"Greeks         : {greeks}")
    