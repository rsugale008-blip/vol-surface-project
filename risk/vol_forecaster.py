# risk/vol_forecaster.py
# ─────────────────────────────────────────
# ML Volatility Forecaster
# XGBoost + LightGBM + RandomForest
# With Plotly Dashboard
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import timedelta
import webbrowser
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TICKER


# ════════════════════════════════════════
# COLORS
# ════════════════════════════════════════
DARK_BG = "rgb(10,10,20)"
CARD_BG = "rgb(20,20,35)"
ACCENT  = "#00d4ff"
RED     = "#ff6b6b"
GREEN   = "#00ff88"
YELLOW  = "#ffd700"
PURPLE  = "#bb86fc"
WHITE   = "#ffffff"
GRAY    = "rgba(255,255,255,0.1)"


# ════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════
def fetch_historical_data(ticker=TICKER, period="2y"):
    print(f"\n[Forecaster] Fetching {ticker} history...")

    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    df    = df[["Open","High","Low","Close","Volume"]]
    df.columns = ["open","high","low","close","volume"]

    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]/df["close"].shift(1))

    for w in [5,10,21,63]:
        df[f"rv_{w}d"] = df["returns"].rolling(w).std()*np.sqrt(252)

    df["vol_of_vol"] = df["rv_21d"].rolling(10).std()

    for w in [5,10,21]:
        df[f"mom_{w}d"] = df["close"].pct_change(w)

    df["parkinson"] = np.sqrt(
        1/(4*np.log(2))*(np.log(df["high"]/df["low"])**2)
    )*np.sqrt(252)

    df["garman_klass"] = np.sqrt(
        0.5*np.log(df["high"]/df["low"])**2 -
        (2*np.log(2)-1)*np.log(df["close"]/df["open"])**2
    )*np.sqrt(252)

    df["vol_norm"]   = df["volume"]/df["volume"].rolling(21).mean()
    df["vol_change"] = df["volume"].pct_change()

    for lag in [1,2,3,5]:
        df[f"rv_lag{lag}"]  = df["rv_21d"].shift(lag)
        df[f"ret_lag{lag}"] = df["returns"].shift(lag)

    df["vol_zscore"] = (
        (df["rv_21d"] - df["rv_21d"].rolling(63).mean())
        / df["rv_21d"].rolling(63).std()
    )

    df["target"] = df["rv_21d"].shift(-1)
    df = df.dropna()

    print(f"[Forecaster] {len(df)} trading days loaded")
    print(f"[Forecaster] {df.index[0].date()} to "
          f"{df.index[-1].date()}")
    return df


def get_features():
    return [
        "rv_5d","rv_10d","rv_21d","rv_63d",
        "vol_of_vol","mom_5d","mom_10d","mom_21d",
        "parkinson","garman_klass",
        "vol_norm","vol_change","vol_zscore",
        "rv_lag1","rv_lag2","rv_lag3","rv_lag5",
        "ret_lag1","ret_lag2","ret_lag3"
    ]


def prepare_data(df, test_split=0.2):
    features = get_features()
    X        = df[features].values
    y        = df["target"].values
    dates    = df.index

    split    = int(len(X)*(1-test_split))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    d_train, d_test = dates[:split], dates[split:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[Forecaster] Train: {len(X_train)} | "
          f"Test: {len(X_test)} samples")

    return (X_train, X_test, y_train, y_test,
            d_train, d_test, scaler, features)


# ════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════
def train_models(X_train, y_train):
    models = {}

    print("\n[Forecaster] Training XGBoost...")
    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist",
        random_state=42, verbosity=0
    ).fit(X_train, y_train)

    print("[Forecaster] Training LightGBM...")
    models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        verbose=-1
    ).fit(X_train, y_train)

    print("[Forecaster] Training RandomForest...")
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200, max_depth=6,
        min_samples_leaf=5, random_state=42,
        n_jobs=-1
    ).fit(X_train, y_train)

    print("[Forecaster] All models trained!")
    return models


# ════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════
def evaluate_models(models, X_test, y_test, d_test):
    results   = {}
    all_preds = []

    print("\n" + "="*60)
    print("  MODEL PERFORMANCE")
    print("="*60)
    print(f"  {'Model':<15} {'RMSE':>8} "
          f"{'MAE':>8} {'DirAcc':>8}")
    print("-"*60)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        all_preds.append(y_pred)

        rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
        mae     = mean_absolute_error(y_test, y_pred)
        dir_acc = np.mean(
            (np.diff(y_test)>0) == (np.diff(y_pred)>0)
        )*100

        print(f"  {name:<15} {rmse*100:>7.3f}% "
              f"{mae*100:>7.3f}% {dir_acc:>7.1f}%")

        results[name] = {
            "y_pred" : y_pred,
            "y_test" : y_test,
            "dates"  : d_test,
            "rmse"   : rmse,
            "mae"    : mae,
            "dir_acc": dir_acc
        }

    # Ensemble
    ens_pred = np.mean(all_preds, axis=0)
    rmse     = np.sqrt(mean_squared_error(y_test, ens_pred))
    mae      = mean_absolute_error(y_test, ens_pred)
    dir_acc  = np.mean(
        (np.diff(y_test)>0) == (np.diff(ens_pred)>0)
    )*100

    print(f"  {'Ensemble':<15} {rmse*100:>7.3f}% "
          f"{mae*100:>7.3f}% {dir_acc:>7.1f}%")
    print("="*60)

    results["Ensemble"] = {
        "y_pred" : ens_pred,
        "y_test" : y_test,
        "dates"  : d_test,
        "rmse"   : rmse,
        "mae"    : mae,
        "dir_acc": dir_acc
    }

    return results


# ════════════════════════════════════════
# FORECASTING
# ════════════════════════════════════════
def forecast_next_days(models, df, scaler,
                        features, days=5):
    last_row = df[features].values[-1:]
    curr     = scaler.transform(last_row)
    rv_index = features.index("rv_21d")

    forecasts = {}
    for name, model in models.items():
        preds      = []
        curr_state = curr.copy()
        for _ in range(days):
            pred = model.predict(curr_state)[0]
            preds.append(pred)
            curr_state[0][rv_index] = pred
        forecasts[name] = preds

    forecasts["Ensemble"] = np.mean(
        list(forecasts.values()), axis=0
    ).tolist()

    last_date = df.index[-1]
    fut_dates = []
    d = last_date
    for _ in range(days):
        d = d + timedelta(days=1)
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        fut_dates.append(d)

    print(f"\n5-Day Vol Forecast")
    print(f"  {'Date':<12} {'XGB':>8} {'LGB':>8} "
          f"{'RF':>8} {'Ensemble':>10}")
    print("-"*52)

    for i, d in enumerate(fut_dates):
        xv = forecasts["XGBoost"][i]*100
        lv = forecasts["LightGBM"][i]*100
        rv = forecasts["RandomForest"][i]*100
        ev = forecasts["Ensemble"][i]*100
        print(f"  {str(d)[:10]:<12} {xv:>7.2f}% "
              f"{lv:>7.2f}% {rv:>7.2f}% {ev:>9.2f}%")

    return {"dates": fut_dates, "forecasts": forecasts}


# ════════════════════════════════════════
# FEATURE IMPORTANCE
# ════════════════════════════════════════
def get_feature_importance(models, features):
    model = models.get("XGBoost")
    if model is None:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature"   : features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Features (XGBoost)")
    print(fi.head(10).to_string(index=False))
    return fi


# ════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════
def plot_dashboard(eval_results, forecast, df, fi_df):

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Predicted vs Actual Volatility",
            "5-Day Volatility Forecast",
            "Feature Importance (XGBoost)",
            "Historical Volatility Regimes"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    colors = {
        "XGBoost"     : ACCENT,
        "LightGBM"    : YELLOW,
        "RandomForest": PURPLE,
        "Ensemble"    : RED
    }

    # ── Plot 1: Predicted vs Actual ──
    ens     = eval_results["Ensemble"]
    dates_s = [str(d)[:10] for d in ens["dates"]]

    fig.add_trace(go.Scatter(
        x=dates_s,
        y=ens["y_test"]*100,
        mode="lines",
        name="Actual",
        line=dict(color=GREEN, width=2),
        hovertemplate="Date:%{x}<br>Actual:%{y:.2f}%<extra></extra>"
    ), row=1, col=1)

    for name, res in eval_results.items():
        w = 3 if name == "Ensemble" else 1
        d = "dash" if name == "Ensemble" else "dot"
        fig.add_trace(go.Scatter(
            x=dates_s,
            y=res["y_pred"]*100,
            mode="lines",
            name=name,
            line=dict(
                color=colors.get(name, WHITE),
                width=w,
                dash=d
            ),
            hovertemplate=name + ":%{y:.2f}%<extra></extra>"
        ), row=1, col=1)

    # ── Plot 2: Forecast ──
    ctx_dates = [str(d)[:10] for d in df.index[-15:]]
    ctx_vols  = df["rv_21d"].values[-15:]*100

    fig.add_trace(go.Scatter(
        x=ctx_dates,
        y=ctx_vols,
        mode="lines+markers",
        name="Recent Vol",
        line=dict(color=GREEN, width=2),
        marker=dict(size=4)
    ), row=1, col=2)

    fut_dates = [str(d)[:10] for d in forecast["dates"]]
    ens_vols  = [v*100 for v in
                 forecast["forecasts"]["Ensemble"]]

    fig.add_trace(go.Scatter(
        x=fut_dates,
        y=ens_vols,
        mode="lines+markers",
        name="Forecast",
        line=dict(color=YELLOW, width=3, dash="dash"),
        marker=dict(size=10, symbol="star", color=YELLOW),
        hovertemplate="Date:%{x}<br>Forecast:%{y:.2f}%<extra></extra>"
    ), row=1, col=2)

    upper = [v*1.15 for v in ens_vols]
    lower = [v*0.85 for v in ens_vols]
    fig.add_trace(go.Scatter(
        x=fut_dates + fut_dates[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(255,215,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="15% Band"
    ), row=1, col=2)

    # ── Plot 3: Feature Importance ──
    if len(fi_df) > 0:
        top = fi_df.head(10)
        fig.add_trace(go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker=dict(
                color=top["importance"],
                colorscale="Viridis"
            ),
            name="Importance"
        ), row=2, col=1)

    # ── Plot 4: Historical Vol ──
    hist_d = [str(d)[:10] for d in df.index[-252:]]

    fig.add_trace(go.Scatter(
        x=hist_d,
        y=df["rv_21d"].values[-252:]*100,
        mode="lines",
        name="21d RV",
        line=dict(color=ACCENT, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.1)"
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=hist_d,
        y=df["rv_63d"].values[-252:]*100,
        mode="lines",
        name="63d RV",
        line=dict(color=RED, width=1.5)
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=hist_d,
        y=df["parkinson"].values[-252:]*100,
        mode="lines",
        name="Parkinson",
        line=dict(color=PURPLE, width=1, dash="dot")
    ), row=2, col=2)

    # ── Layout ──
    best      = min(eval_results.items(),
                    key=lambda x: x[1]["rmse"])
    best_name = best[0]
    best_acc  = best[1]["dir_acc"]
    best_rmse = best[1]["rmse"]

    fig.update_layout(
        height=850,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=WHITE, size=11),
        title=dict(
            text=(
                "<b>" + TICKER +
                " ML Volatility Forecaster</b> | "
                "Best: " + best_name +
                " | Dir Acc: " +
                str(round(best_acc, 1)) + "%" +
                " | RMSE: " +
                str(round(best_rmse*100, 3)) + "%"
            ),
            font=dict(color=WHITE, size=16),
            x=0.5
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="gray",
            font=dict(size=10)
        ),
        margin=dict(l=50, r=30, t=80, b=50)
    )

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(gridcolor=GRAY,
                             row=row, col=col)
            fig.update_yaxes(gridcolor=GRAY,
                             row=row, col=col)

    fig.update_yaxes(title_text="Vol (%)", row=1, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=1, col=2)
    fig.update_xaxes(title_text="Importance", row=2, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=2, col=2)

    return fig


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════
if __name__ == "__main__":

    print("="*55)
    print("  ML Vol Forecaster - " + TICKER)
    print("  XGBoost + LightGBM + RandomForest")
    print("="*55)

    # 1. Data
    df = fetch_historical_data(TICKER)

    # 2. Prepare
    (X_train, X_test,
     y_train, y_test,
     d_train, d_test,
     scaler, features) = prepare_data(df)

    # 3. Train
    models = train_models(X_train, y_train)

    # 4. Evaluate
    eval_results = evaluate_models(
        models, X_test, y_test, d_test
    )

    # 5. Feature importance
    fi_df = get_feature_importance(models, features)

    # 6. Forecast
    forecast = forecast_next_days(
        models, df, scaler, features, days=5
    )

    # 7. Plot and save HTML
    print("\n[Forecaster] Building dashboard...")
    fig = plot_dashboard(eval_results, forecast, df, fi_df)

    html_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "forecast_dashboard.html"
    )

    fig.write_html(html_path)
    print("[Forecaster] Saved to " + html_path)

    webbrowser.open("file:///" + html_path)
    print("[Forecaster] Opening in browser...")