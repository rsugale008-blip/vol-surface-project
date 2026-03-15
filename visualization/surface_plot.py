# visualization/surface_plot.py — FIXED VERSION

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from volatility.surface_builder import build_smooth_surface, get_term_structure, get_vol_smile
from data.fetcher_yfinance import get_options_chain, filter_options
from volatility.iv_solver import add_iv_to_dataframe
from config.settings import TICKER


def plot_vol_dashboard(df: pd.DataFrame):

    # ── Build data ──
    call_surface = build_smooth_surface(df, "call", grid_points=50)
    term_df      = get_term_structure(df, "call")
    smile_df     = get_vol_smile(df, expiry_index=0, option_type="call")
    spot         = df["spot_price"].iloc[0]
    iv_pct       = call_surface["iv_grid"] * 100

    # ── Figure with subplots ──
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"},                  {"type": "xy"}]
        ],
        subplot_titles=[
            f"{TICKER} Vol Surface (Calls)",
            "ATM Term Structure",
            "Vol Smile — Nearest Expiry"
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08
    )

    # ── 3D Surface ──
    fig.add_trace(go.Surface(
        x=call_surface["strikes_grid"],
        y=call_surface["dte_grid"],
        z=iv_pct,
        colorscale="RdYlGn_r",
        showscale=True,
        colorbar=dict(
            x=0.05, y=0.75,
            len=0.45,
            title="IV %",
            thickness=15,),
        hovertemplate="Strike: $%{x:.0f}<br>DTE: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>"
    ), row=1, col=1)

    # ── Term Structure ──
    fig.add_trace(go.Scatter(
        x=term_df["DTE"],
        y=term_df["atm_iv"] * 100,
        mode="lines+markers",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=8, color="#00d4ff"),
        name="ATM IV",
        hovertemplate="DTE: %{x}<br>IV: %{y:.1f}%<extra></extra>"
    ), row=2, col=1)

    # ── Vol Smile ──
    fig.add_trace(go.Scatter(
        x=smile_df["strike"],
        y=smile_df["iv"] * 100,
        mode="lines+markers",
        line=dict(color="#ff6b6b", width=3),
        marker=dict(size=6, color="#ff6b6b"),
        name="Vol Smile",
        hovertemplate="Strike: $%{x:.0f}<br>IV: %{y:.1f}%<extra></extra>"
    ), row=2, col=2)

    # ── ATM vertical line on smile ──
    fig.add_trace(go.Scatter(
        x=[spot, spot],
        y=[smile_df["iv"].min() * 100, smile_df["iv"].max() * 100],
        mode="lines",
        line=dict(color="yellow", width=2, dash="dash"),
        name=f"ATM ${spot:.0f}"
    ), row=2, col=2)

    # ── Layout ──
    fig.update_layout(
        height=900,
        paper_bgcolor="rgb(10,10,20)",
        plot_bgcolor="rgb(20,20,35)",
        font=dict(color="white", size=12),
        title=dict(
            text=f"<b>{TICKER} Live Volatility Surface</b> | Spot: ${spot:.2f}",
            font=dict(size=20, color="white"),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="gray",
            font=dict(color="white")
        ),
        # ── 3D scene axes ──
        scene=dict(
            xaxis=dict(
                title="Strike ($)",
                backgroundcolor="rgb(20,20,30)",
                gridcolor="gray",
                showbackground=True,
                tickfont=dict(color="white")
            ),
            yaxis=dict(
                title="Days to Expiry",
                backgroundcolor="rgb(20,20,30)",
                gridcolor="gray",
                showbackground=True,
                tickfont=dict(color="white")
            ),
            zaxis=dict(
                title="IV (%)",
                backgroundcolor="rgb(20,20,30)",
                gridcolor="gray",
                showbackground=True,
                tickfont=dict(color="white")
            ),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
            bgcolor="rgb(10,10,20)"
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # ── 2D axis labels ──
    fig.update_xaxes(
        title_text="Days to Expiry",
        gridcolor="rgba(255,255,255,0.1)",
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="ATM IV (%)",
        gridcolor="rgba(255,255,255,0.1)",
        row=2, col=1
    )
    fig.update_xaxes(
        title_text="Strike ($)",
        gridcolor="rgba(255,255,255,0.1)",
        row=2, col=2
    )
    fig.update_yaxes(
        title_text="IV (%)",
        gridcolor="rgba(255,255,255,0.1)",
        row=2, col=2
    )

    fig.show()
    return fig


if __name__ == "__main__":
    print("=" * 55)
    print(f"  Vol Surface Dashboard — {TICKER}")
    print("=" * 55)

    df = get_options_chain()
    df = filter_options(df)
    df = add_iv_to_dataframe(df)

    print("\n[Viz] Launching dashboard in browser...")
    plot_vol_dashboard(df)
    print("✅ Done!")