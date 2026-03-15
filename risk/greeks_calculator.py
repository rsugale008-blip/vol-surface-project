# risk/greeks_calculator.py
# ─────────────────────────────────────────
# Black-Scholes Greeks Calculator
# Adds Delta, Gamma, Vega, Theta to dataframe
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy.stats import norm


# ══════════════════════════════════════════
# Black-Scholes Greeks
# ══════════════════════════════════════════
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):

    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    pdf = norm.pdf(d1)
    cdf = norm.cdf(d1)

    gamma = pdf / (S*sigma*np.sqrt(T))
    vega  = S * pdf * np.sqrt(T) / 100

    if option_type == "call":

        delta = cdf

        theta = (
            -(S*pdf*sigma)/(2*np.sqrt(T))
            - r*K*np.exp(-r*T)*norm.cdf(d2)
        ) / 365

    else:

        delta = cdf - 1

        theta = (
            -(S*pdf*sigma)/(2*np.sqrt(T))
            + r*K*np.exp(-r*T)*norm.cdf(-d2)
        ) / 365

    return delta, gamma, vega, theta


# ══════════════════════════════════════════
# Add Greeks to DataFrame
# ══════════════════════════════════════════
def add_greeks_to_dataframe(df: pd.DataFrame, r: float = 0.05):

    print("[Greeks] Calculating option greeks...")

    deltas, gammas, vegas, thetas = [], [], [], []

    for _, row in df.iterrows():

        try:
            S = row["spot_price"]
            K = row["strike"]
            T = row["DTE"] / 365
            sigma = row["iv"]
            option_type = row["option_type"]

            delta, gamma, vega, theta = black_scholes_greeks(
                S, K, T, r, sigma, option_type
            )

        except Exception:
            delta, gamma, vega, theta = 0, 0, 0, 0

        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)
        thetas.append(theta)

    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"]  = vegas
    df["theta"] = thetas

    print("[Greeks] Completed.")

    return df


# ══════════════════════════════════════════
# Summary
# ══════════════════════════════════════════
def print_greeks_summary(df):

    print("\nGreeks Summary")
    print("──────────────")

    print(f"Average Delta : {df['delta'].mean():.3f}")
    print(f"Average Gamma : {df['gamma'].mean():.4f}")
    print(f"Average Vega  : {df['vega'].mean():.4f}")
    print(f"Average Theta : {df['theta'].mean():.4f}")