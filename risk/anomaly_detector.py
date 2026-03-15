# risk/anomaly_detector.py
# ─────────────────────────────────────────
# AI-based vol spike and skew
# anomaly detection using Z-score
# + Isolation Forest (ML)
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TICKER


def detect_iv_anomalies(df: pd.DataFrame,
                        contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect anomalous IV points using Isolation Forest.
    Flags options where IV is abnormally high/low
    given their strike and DTE.

    contamination: expected fraction of anomalies (5%)
    """
    print("[Anomaly] Running Isolation Forest...")

    features = df[["strike", "DTE", "iv",
                   "moneyness", "mid_price"]].copy()
    features = features.dropna()

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Isolation Forest
    model    = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    preds    = model.fit_predict(X_scaled)
    scores   = model.score_samples(X_scaled)

    # Add results back
    features["anomaly"]       = preds == -1   # True = anomaly
    features["anomaly_score"] = scores

    anomalies = features[features["anomaly"] == True]
    print(f"[Anomaly] Found {len(anomalies)} anomalous options "
          f"out of {len(features)} ({len(anomalies)/len(features)*100:.1f}%)")

    return features


def detect_vol_spikes(df: pd.DataFrame,
                      z_threshold: float = 2.5) -> dict:
    """
    Detect vol spikes using Z-score method.
    Flags when IV deviates significantly from
    its expected level given moneyness.
    """
    print("[Spike] Running Z-score vol spike detection...")

    alerts = []

    for expiry, group in df.groupby("expiry"):
        if len(group) < 5:
            continue

        mean_iv = group["iv"].mean()
        std_iv  = group["iv"].std()

        if std_iv == 0:
            continue

        group   = group.copy()
        group["z_score"] = (group["iv"] - mean_iv) / std_iv

        spikes  = group[group["z_score"].abs() > z_threshold]

        for _, row in spikes.iterrows():
            alerts.append({
                "expiry"      : expiry,
                "strike"      : row["strike"],
                "option_type" : row["option_type"],
                "iv"          : round(row["iv"] * 100, 2),
                "z_score"     : round(row["z_score"], 2),
                "severity"    : "HIGH" if abs(row["z_score"]) > 4
                                else "MEDIUM"
            })

    alerts_df = pd.DataFrame(alerts)

    if len(alerts_df) > 0:
        print(f"[Spike] {len(alerts_df)} vol spike alerts!")
        print(alerts_df.to_string(index=False))
    else:
        print("[Spike] No significant vol spikes detected.")

    return {
        "alerts"      : alerts_df,
        "total_alerts": len(alerts_df),
        "high_alerts" : len(alerts_df[alerts_df["severity"] == "HIGH"])
                        if len(alerts_df) > 0 else 0
    }


def calculate_skew_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate vol skew metrics across the surface.
    25-delta skew, risk reversal, butterfly spread.
    """
    print("[Skew] Calculating skew metrics...")

    spot    = df["spot_price"].iloc[0]
    results = {}

    for expiry, group in df.groupby("expiry"):
        calls = group[group["option_type"] == "call"].sort_values("strike")

        if len(calls) < 5:
            continue

        # ATM IV
        atm_idx = (calls["strike"] - spot).abs().idxmin()
        atm_iv  = calls.loc[atm_idx, "iv"]

        # OTM put proxy (90% moneyness)
        otm_put = calls[calls["moneyness"] <= 0.92]
        otm_iv  = otm_put["iv"].mean() if len(otm_put) > 0 else atm_iv

        # OTM call (110% moneyness)
        otm_call = calls[calls["moneyness"] >= 1.08]
        otm_call_iv = otm_call["iv"].mean() if len(otm_call) > 0 else atm_iv

        # Skew metrics
        skew       = (otm_iv - atm_iv) * 100
        risk_rev   = (otm_call_iv - otm_iv) * 100
        butterfly  = (otm_iv + otm_call_iv) / 2 - atm_iv

        results[expiry] = {
            "atm_iv"     : round(atm_iv * 100,    2),
            "skew"       : round(skew,             2),
            "risk_rev"   : round(risk_rev,         2),
            "butterfly"  : round(butterfly * 100,  2)
        }

    skew_df = pd.DataFrame(results).T
    print("\n── Skew Metrics by Expiry ──")
    print(skew_df.to_string())
    return results


if __name__ == "__main__":
    from data.fetcher_yfinance import get_options_chain, filter_options
    from volatility.iv_solver import add_iv_to_dataframe

    print("=" * 55)
    print(f"  AI Anomaly Detector — {TICKER}")
    print("=" * 55)

    df = get_options_chain()
    df = filter_options(df)
    df = add_iv_to_dataframe(df)

    # Run all detectors
    anomalies  = detect_iv_anomalies(df)
    spikes     = detect_vol_spikes(df)
    skew       = calculate_skew_metrics(df)