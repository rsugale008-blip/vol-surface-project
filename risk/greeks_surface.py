# risk/greeks_surface.py
# ─────────────────────────────────────────
# Calculate + Visualize Greeks across the
# entire vol surface
# Delta, Gamma, Vega, Theta, Rho
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from volatility.black_scholes import calculate_greeks
from config.settings import TICKER, RISK_FREE_RATE


def add_greeks_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all Greeks for every option in dataframe.
    Adds columns: delta, gamma, vega, theta, rho
    """
    print("[Greeks] Calculating Greeks for all options...")

    greeks_list = []
    total = len(df)

    for i, row in df.iterrows():
        T = row["DTE"] / 365.0

        if T <= 0 or row["iv"] <= 0:
            greeks_list.append({
                "delta": np.nan, "gamma": np.nan,
                "vega" : np.nan, "theta": np.nan,
                "rho"  : np.nan
            })
            continue

        g = calculate_greeks(
            S=row["spot_price"],
            K=row["strike"],
            T=T,
            r=RISK_FREE_RATE,
            sigma=row["iv"],
            option_type=row["option_type"]
        )
        greeks_list.append(g)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{total}...")

    greeks_df = pd.DataFrame(greeks_list)
    df = pd.concat([df.reset_index(drop=True),
                    greeks_df.reset_index(drop=True)], axis=1)

    print(f"[Greeks] Done! Greeks added for {len(df)} options")
    return df


def build_greek_surface(df: pd.DataFrame,
                        greek: str = "delta",
                        option_type: str = "call",
                        grid_points: int = 50) -> dict:
    """
    Build smooth interpolated surface for a specific Greek.

    Parameters:
        greek       : 'delta','gamma','vega','theta','rho'
        option_type : 'call' or 'put'
        grid_points : smoothness of surface grid
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.ndimage import gaussian_filter

    df_type = df[df["option_type"] == option_type].copy()
    df_type = df_type.dropna(subset=[greek, "strike", "DTE"])

    strikes = df_type["strike"].values
    dtes    = df_type["DTE"].values
    values  = df_type[greek].values

    # Remove outliers
    q_low  = np.percentile(values, 2)
    q_high = np.percentile(values, 98)
    mask   = (values >= q_low) & (values <= q_high)

    strikes = strikes[mask]
    dtes    = dtes[mask]
    values  = values[mask]

    print(f"[Greeks] Building {greek} surface "
          f"({option_type}) with {len(strikes)} points...")

    # Grid
    strike_range = np.linspace(strikes.min(), strikes.max(), grid_points)
    dte_range    = np.linspace(dtes.min(),    dtes.max(),    grid_points)
    sg, dg       = np.meshgrid(strike_range, dte_range)

    # RBF Interpolation
    try:
        points   = np.column_stack([strikes, dtes])
        qpoints  = np.column_stack([sg.ravel(), dg.ravel()])
        rbf      = RBFInterpolator(points, values, smoothing=0.01)
        vg_flat  = rbf(qpoints)
        vg       = vg_flat.reshape(sg.shape)
        vg       = gaussian_filter(vg, sigma=1.0)
    except Exception as e:
        print(f"[Greeks] RBF failed: {e}, using linear fallback")
        from scipy.interpolate import LinearNDInterpolator
        lin = LinearNDInterpolator(
            list(zip(strikes, dtes)), values,
            fill_value=np.nanmean(values)
        )
        vg = lin(sg, dg)

    return {
        "strikes_grid" : sg,
        "dte_grid"     : dg,
        "values_grid"  : vg,
        "greek"        : greek,
        "option_type"  : option_type,
        "raw_df"       : df_type
    }


def plot_single_greek(surface: dict) -> go.Figure:
    """Plot a single Greek as 3D surface."""

    greek       = surface["greek"]
    option_type = surface["option_type"]
    sg          = surface["strikes_grid"]
    dg          = surface["dte_grid"]
    vg          = surface["values_grid"]
    spot        = surface["raw_df"]["spot_price"].iloc[0]

    # Color scheme per greek
    colorscales = {
        "delta": "RdYlGn",
        "gamma": "Viridis",
        "vega" : "Plasma",
        "theta": "RdBu",
        "rho"  : "Cividis"
    }

    colorscale = colorscales.get(greek, "Viridis")

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=sg, y=dg, z=vg,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            x=1.02,
            title=dict(text=greek.capitalize(), side="right"),
            thickness=15
        ),
        hovertemplate=(
            f"Strike: $%{{x:.0f}}<br>"
            f"DTE: %{{y:.0f}}<br>"
            f"{greek.capitalize()}: %{{z:.4f}}<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{TICKER} {greek.capitalize()} Surface "
                 f"({option_type.upper()}S)</b> | Spot: ${spot:.2f}",
            font=dict(color="white", size=18),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title="Strike ($)",
                       backgroundcolor="rgb(20,20,30)",
                       gridcolor="gray",
                       showbackground=True),
            yaxis=dict(title="Days to Expiry",
                       backgroundcolor="rgb(20,20,30)",
                       gridcolor="gray",
                       showbackground=True),
            zaxis=dict(title=greek.capitalize(),
                       backgroundcolor="rgb(20,20,30)",
                       gridcolor="gray",
                       showbackground=True),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
            bgcolor="rgb(10,10,20)"
        ),
        paper_bgcolor="rgb(10,10,20)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=60, b=0),
        height=600
    )

    fig.show()
    return fig


def plot_all_greeks_dashboard(df: pd.DataFrame) -> go.Figure:
    """
    Full Greeks dashboard — all 5 Greeks in one view.
    2x3 grid: Delta, Gamma, Vega, Theta, Rho + Summary
    """
    spot        = df["spot_price"].iloc[0]
    option_type = "call"

    greeks = ["delta", "gamma", "vega", "theta", "rho"]

    colorscales = {
        "delta": "RdYlGn",
        "gamma": "Viridis",
        "vega" : "Plasma",
        "theta": "RdBu_r",
        "rho"  : "Cividis"
    }

    descriptions = {
        "delta": "Price sensitivity to spot",
        "gamma": "Delta sensitivity to spot",
        "vega" : "Price sensitivity to vol",
        "theta": "Time decay (per day)",
        "rho"  : "Sensitivity to rates"
    }

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}, {"type": "xy"}]
        ],
        subplot_titles=[
            "Delta Surface", "Gamma Surface", "Vega Surface",
            "Theta Surface", "Rho Surface",   "ATM Greeks vs DTE"
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )

    scene_map = {
        (1,1): "scene",  (1,2): "scene2", (1,3): "scene3",
        (2,1): "scene4", (2,2): "scene5"
    }

    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

    for idx, greek in enumerate(greeks):
        row, col = positions[idx]

        try:
            surface = build_greek_surface(
                df, greek=greek,
                option_type=option_type,
                grid_points=40
            )
            sg = surface["strikes_grid"]
            dg = surface["dte_grid"]
            vg = surface["values_grid"]

            fig.add_trace(go.Surface(
                x=sg, y=dg, z=vg,
                colorscale=colorscales[greek],
                showscale=False,
                hovertemplate=(
                    f"Strike: $%{{x:.0f}}<br>"
                    f"DTE: %{{y:.0f}}<br>"
                    f"{greek.capitalize()}: %{{z:.4f}}"
                    f"<extra></extra>"
                ),
                name=greek.capitalize()
            ), row=row, col=col)

        except Exception as e:
            print(f"[Greeks] Could not build {greek} surface: {e}")

    # ── ATM Greeks vs DTE line chart ──
    atm_df = df[
        (df["option_type"] == option_type) &
        (df["strike"].between(spot * 0.995, spot * 1.005))
    ].sort_values("DTE")

    colors = {
        "delta": "#00d4ff",
        "gamma": "#00ff88",
        "vega" : "#ff6b6b",
        "theta": "#ffd700",
        "rho"  : "#ff88ff"
    }

    for greek in greeks:
        if greek in atm_df.columns:
            fig.add_trace(go.Scatter(
                x=atm_df["DTE"],
                y=atm_df[greek],
                mode="lines+markers",
                name=greek.capitalize(),
                line=dict(color=colors[greek], width=2),
                marker=dict(size=5),
                hovertemplate=(
                    f"DTE: %{{x}}<br>"
                    f"{greek.capitalize()}: %{{y:.4f}}"
                    f"<extra></extra>"
                )
            ), row=2, col=3)

    # ── Layout ──
    fig.update_layout(
        height=800,
        paper_bgcolor="rgb(10,10,20)",
        font=dict(color="white", size=11),
        title=dict(
            text=f"<b>{TICKER} Complete Greeks Surface Dashboard</b>"
                 f" | Spot: ${spot:.2f}",
            font=dict(color="white", size=18),
            x=0.5
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="gray",
            x=0.75, y=0.15
        ),
        margin=dict(l=20, r=20, t=80, b=20)
    )

    # Style all 3D scenes
    scene_style = dict(
        xaxis=dict(title="Strike",
                   backgroundcolor="rgb(20,20,30)",
                   gridcolor="gray",
                   showbackground=True,
                   showticklabels=False),
        yaxis=dict(title="DTE",
                   backgroundcolor="rgb(20,20,30)",
                   gridcolor="gray",
                   showbackground=True,
                   showticklabels=False),
        zaxis=dict(backgroundcolor="rgb(20,20,30)",
                   gridcolor="gray",
                   showbackground=True),
        bgcolor="rgb(10,10,20)",
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
    )

    for scene in ["scene", "scene2", "scene3", "scene4", "scene5"]:
        fig.update_layout(**{scene: scene_style})

    # Style ATM chart
    fig.update_xaxes(
        title_text="Days to Expiry",
        gridcolor="rgba(255,255,255,0.1)",
        row=2, col=3
    )
    fig.update_yaxes(
        title_text="Greek Value",
        gridcolor="rgba(255,255,255,0.1)",
        row=2, col=3
    )

    fig.show()
    return fig


def print_greeks_summary(df: pd.DataFrame):
    """Print ATM Greeks summary table."""
    spot    = df["spot_price"].iloc[0]
    atm_df  = df[df["strike"].between(spot * 0.99, spot * 1.01)]

    print("\n" + "=" * 60)
    print(f"  ATM GREEKS SUMMARY — {TICKER} @ ${spot:.2f}")
    print("=" * 60)
    print(f"  {'Expiry':<12} {'DTE':>4} {'Type':>5} "
          f"{'Delta':>8} {'Gamma':>8} "
          f"{'Vega':>8} {'Theta':>8}")
    print("-" * 60)

    for _, row in atm_df.sort_values(
            ["DTE", "option_type"]).iterrows():
        print(f"  {row['expiry']:<12} {row['DTE']:>4.0f} "
              f"{row['option_type']:>5} "
              f"{row.get('delta', 0):>8.4f} "
              f"{row.get('gamma', 0):>8.4f} "
              f"{row.get('vega',  0):>8.4f} "
              f"{row.get('theta', 0):>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    from data.fetcher_yfinance import get_options_chain, filter_options
    from volatility.iv_solver import add_iv_to_dataframe

    print("=" * 60)
    print(f"  Greeks Surface Builder — {TICKER}")
    print("=" * 60)

    # Full pipeline
    df = get_options_chain()
    df = filter_options(df)
    df = add_iv_to_dataframe(df)
    df = add_greeks_to_dataframe(df)

    # Print summary
    print_greeks_summary(df)

    # Plot all Greeks dashboard
    print("\n[Greeks] Launching full Greeks dashboard...")
    plot_all_greeks_dashboard(df)
    