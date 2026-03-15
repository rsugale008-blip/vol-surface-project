# visualization/dashboard.py
# ─────────────────────────────────────────
# Live Auto-Refresh Volatility Dashboard
# Built with Dash + Plotly
# ─────────────────────────────────────────

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetcher_yfinance import get_options_chain, filter_options
from volatility.iv_solver import add_iv_to_dataframe
from volatility.surface_builder import (build_smooth_surface,
                                         get_term_structure,
                                         get_vol_smile)
from config.settings import TICKER, REFRESH_INTERVAL
from risk.greeks_surface import add_greeks_to_dataframe, build_greek_surface

# ── App Init ──
app = dash.Dash(
    __name__,
    title=f"{TICKER} Vol Surface",
    update_title="Refreshing..."
)

# ── Color Theme ──
DARK_BG = "rgb(10,10,20)"
CARD_BG = "rgb(20,20,35)"
ACCENT  = "#00d4ff"
RED     = "#ff6b6b"
GREEN   = "#00ff88"
YELLOW  = "#ffd700"
WHITE   = "#ffffff"
GRAY    = "rgba(255,255,255,0.1)"

# ══════════════════════════════════════════
#  FETCH DATA ONCE AT STARTUP
# ══════════════════════════════════════════
def load_data():
    print("[Dashboard] Fetching live data...")
    df = get_options_chain()
    df = filter_options(df)
    df = add_iv_to_dataframe(df)
    df = add_greeks_to_dataframe(df)
    print("[Dashboard] Data ready!")
    return df

# Load once at startup
print("=" * 55)
print(f"  {TICKER} Live Vol Surface Dashboard")
print("=" * 55)
INITIAL_DF = load_data()

# ══════════════════════════════════════════
#  BUILD ALL FIGURES AT STARTUP
# ══════════════════════════════════════════
def build_surface_fig(df):
    surface = build_smooth_surface(df, "call", grid_points=50)
    iv_pct  = surface["iv_grid"] * 100
    spot    = df["spot_price"].iloc[0]
    raw     = surface["raw_df"]

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=surface["strikes_grid"],
        y=surface["dte_grid"],
        z=iv_pct,
        colorscale="RdYlGn_r",
        showscale=True,
        colorbar=dict(
            x=1.02,
            title=dict(text="IV %", side="right"),
            thickness=15
        ),
        hovertemplate=(
            "Strike: $%{x:.0f}<br>"
            "DTE: %{y:.0f}<br>"
            "IV: %{z:.1f}%<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=raw["strike"],
        y=raw["DTE"],
        z=raw["iv"] * 100,
        mode="markers",
        marker=dict(size=2, color="white", opacity=0.3),
        name="Market Data",
        hovertemplate=(
            "Strike: $%{x:.0f}<br>"
            "DTE: %{y:.0f}<br>"
            "IV: %{z:.1f}%<extra></extra>"
        )
    ))

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        scene=dict(
            xaxis=dict(
                title="Strike ($)",
                backgroundcolor=CARD_BG,
                gridcolor="gray",
                showbackground=True
            ),
            yaxis=dict(
                title="Days to Expiry",
                backgroundcolor=CARD_BG,
                gridcolor="gray",
                showbackground=True
            ),
            zaxis=dict(
                title="IV (%)",
                backgroundcolor=CARD_BG,
                gridcolor="gray",
                showbackground=True
            ),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
            bgcolor=DARK_BG
        ),
        title=dict(
            text=f"<b>{TICKER} Implied Volatility Surface</b>"
                 f" | Spot: ${spot:.2f}",
            font=dict(color=WHITE, size=16),
            x=0.5
        ),
        font=dict(color=WHITE),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.5)")
    )
    return fig


def build_term_fig(df):
    term_df = get_term_structure(df, "call")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=term_df["DTE"],
        y=term_df["atm_iv"] * 100,
        mode="lines+markers",
        line=dict(color=ACCENT, width=3),
        marker=dict(size=8, color=ACCENT),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.1)",
        hovertemplate="DTE: %{x}<br>ATM IV: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>ATM Vol Term Structure</b>",
            font=dict(color=WHITE, size=14),
            x=0.5
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=WHITE),
        xaxis=dict(title="Days to Expiry", gridcolor=GRAY),
        yaxis=dict(title="ATM IV (%)",     gridcolor=GRAY),
        margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig


def build_smile_fig(df):
    smile_df = get_vol_smile(df, expiry_index=0, option_type="call")
    spot     = df["spot_price"].iloc[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=smile_df["strike"],
        y=smile_df["iv"] * 100,
        mode="lines+markers",
        line=dict(color=RED, width=3),
        marker=dict(size=6, color=RED),
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.1)",
        name="Vol Smile",
        hovertemplate="Strike: $%{x:.0f}<br>IV: %{y:.1f}%<extra></extra>"
    ))

    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color=YELLOW,
        line_width=2,
        annotation_text=f"ATM ${spot:.0f}",
        annotation_font_color=YELLOW
    )

    fig.update_layout(
        title=dict(
            text="<b>Vol Smile — Nearest Expiry</b>",
            font=dict(color=WHITE, size=14),
            x=0.5
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=WHITE),
        xaxis=dict(title="Strike ($)", gridcolor=GRAY),
        yaxis=dict(title="IV (%)",     gridcolor=GRAY),
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False
    )
    return fig
def build_greeks_fig(df):
    spot   = df["spot_price"].iloc[0]
    atm_df = df[
        (df["option_type"] == "call") &
        (df["strike"].between(spot*0.995, spot*1.005))
    ].sort_values("DTE")

    fig = go.Figure()

    greeks = {
        "delta": ACCENT,
        "gamma": GREEN,
        "vega" : RED,
        "theta": YELLOW,
        "rho"  : "#bb86fc"
    }

    for greek, color in greeks.items():
        if greek in atm_df.columns:
            fig.add_trace(go.Scatter(
                x=atm_df["DTE"],
                y=atm_df[greek],
                mode="lines+markers",
                name=greek.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=(
                    "DTE: %{x}<br>" +
                    greek.capitalize() +
                    ": %{y:.4f}<extra></extra>"
                )
            ))

    fig.update_layout(
        title=dict(
            text="<b>ATM Greeks vs DTE</b>",
            font=dict(color=WHITE, size=14),
            x=0.5
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=WHITE),
        xaxis=dict(title="Days to Expiry",
                   gridcolor=GRAY),
        yaxis=dict(title="Greek Value",
                   gridcolor=GRAY),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig

# ── Greeks Row ──
html.Div(style={
    "backgroundColor": CARD_BG,
    "borderRadius": "12px",
    "padding": "15px",
    "marginBottom": "20px",
    "border": "1px solid rgba(0,212,255,0.2)"
}, children=[
    dcc.Graph(
        id="greeks-chart",
        figure=build_greeks_fig(INITIAL_DF),
        style={"height": "300px"},
        config={"displayModeBar": False}
    )
]),

def build_metric_cards(df):
    spot = df["spot_price"].iloc[0]

    atm_calls = df[
        (df["option_type"] == "call") &
        (df["strike"].between(spot * 0.99, spot * 1.01))
    ]
    atm_puts = df[
        (df["option_type"] == "put") &
        (df["strike"].between(spot * 0.99, spot * 1.01))
    ]

    avg_iv      = df["iv"].mean() * 100
    atm_call_iv = atm_calls["iv"].mean() * 100 if len(atm_calls) else 0
    atm_put_iv  = atm_puts["iv"].mean()  * 100 if len(atm_puts)  else 0
    skew        = atm_put_iv - atm_call_iv
    total_opts  = len(df)

    metrics = [
        ("Spot Price",    f"${spot:.2f}",        ACCENT),
        ("ATM Call IV",   f"{atm_call_iv:.1f}%", GREEN),
        ("ATM Put IV",    f"{atm_put_iv:.1f}%",  RED),
        ("Put-Call Skew", f"{skew:+.1f}%",       YELLOW),
        ("Options Live",  f"{total_opts}",        WHITE),
    ]

    cards = []
    for label, value, color in metrics:
        cards.append(html.Div(style={
            "backgroundColor": "rgb(30,30,50)",
            "borderRadius": "10px",
            "padding": "15px",
            "textAlign": "center",
            "border": f"1px solid {color}33"
        }, children=[
            html.P(label, style={
                "color": "rgba(255,255,255,0.5)",
                "margin": "0 0 6px 0",
                "fontSize": "12px",
                "textTransform": "uppercase",
                "letterSpacing": "1px"
            }),
            html.H3(value, style={
                "color": color,
                "margin": 0,
                "fontSize": "22px",
                "fontWeight": "bold"
            })
        ]))
    return cards


# ══════════════════════════════════════════
#  LAYOUT — uses pre-built figures
# ══════════════════════════════════════════
app.layout = html.Div(style={
    "backgroundColor": DARK_BG,
    "minHeight": "100vh",
    "fontFamily": "'Segoe UI', sans-serif",
    "padding": "20px"
}, children=[

    # ── Header ──
    html.Div(style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "borderBottom": f"1px solid {ACCENT}",
        "paddingBottom": "15px",
        "marginBottom": "20px"
    }, children=[
        html.Div([
            html.H1(
                f"📈 {TICKER} Live Volatility Surface",
                style={"color": WHITE, "margin": 0,
                       "fontSize": "28px", "fontWeight": "bold"}
            ),
            html.P(
                "Real-time Implied Volatility | Risk Dashboard",
                style={"color": ACCENT, "margin": "4px 0 0 0",
                       "fontSize": "14px"}
            )
        ]),
        html.Div([
            html.P(
                f"Last update: {datetime.now().strftime('%H:%M:%S')}",
                style={"color": "rgba(255,255,255,0.5)",
                       "fontSize": "13px", "textAlign": "right",
                       "margin": "0 0 4px 0"}
            ),
            html.H2(
                f"${INITIAL_DF['spot_price'].iloc[0]:.2f}",
                style={"color": GREEN, "margin": 0,
                       "fontSize": "24px", "textAlign": "right"}
            )
        ])
    ]),

    # ── Metric Cards ──
    html.Div(
        children=build_metric_cards(INITIAL_DF),
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(5, 1fr)",
            "gap": "12px",
            "marginBottom": "20px"
        }
    ),

    # ── 3D Surface ──
    html.Div(style={
        "backgroundColor": CARD_BG,
        "borderRadius": "12px",
        "padding": "15px",
        "marginBottom": "20px",
        "border": "1px solid rgba(0,212,255,0.2)"
    }, children=[
        dcc.Graph(
            id="vol-surface-3d",
            figure=build_surface_fig(INITIAL_DF),
            style={"height": "550px"},
            config={"displayModeBar": True, "scrollZoom": True}
        )
    ]),

    # ── Bottom Row ──
    html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "16px",
        "marginBottom": "20px"
    }, children=[
        html.Div(style={
            "backgroundColor": CARD_BG,
            "borderRadius": "12px",
            "padding": "15px",
            "border": "1px solid rgba(0,212,255,0.2)"
        }, children=[
            dcc.Graph(
                id="term-structure",
                figure=build_term_fig(INITIAL_DF),
                style={"height": "300px"},
                config={"displayModeBar": False}
            )
        ]),
        html.Div(style={
            "backgroundColor": CARD_BG,
            "borderRadius": "12px",
            "padding": "15px",
            "border": "1px solid rgba(0,212,255,0.2)"
        }, children=[
            dcc.Graph(
                id="vol-smile",
                figure=build_smile_fig(INITIAL_DF),
                style={"height": "300px"},
                config={"displayModeBar": False}
            )
        ])
    ]),

    # ── Auto Refresh ──
    dcc.Interval(
        id="interval",
        interval=REFRESH_INTERVAL * 1000,
        n_intervals=0
    )
])


# ══════════════════════════════════════════
#  CALLBACKS — Auto Refresh Every 60s
# ══════════════════════════════════════════
@app.callback(
    Output("vol-surface-3d", "figure"),
    Output("term-structure",  "figure"),
    Output("vol-smile",       "figure"),
    Output("greeks-chart",    "figure"),
    Input("interval",         "n_intervals")
)
def refresh_all(n):
    if n == 0:
        raise dash.exceptions.PreventUpdate
    print(f"[Dashboard] Auto-refresh #{n}...")
    df = load_data()
    return (
        build_surface_fig(df),
        build_term_fig(df),
        build_smile_fig(df),
        build_greeks_fig(df)
    )
# ══════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════
if __name__ == "__main__":
    print(f"  → Open browser: http://127.0.0.1:8050")
    print(f"  → Auto-refreshes every {REFRESH_INTERVAL}s")
    print(f"  → Press Ctrl+C to stop")
    print("=" * 55)
    app.run(debug=False, port=8050)