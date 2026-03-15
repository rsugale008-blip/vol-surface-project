# volatility/surface_builder.py
# ─────────────────────────────────────────
# Builds the IV matrix (strike × expiry)
# and interpolates a smooth surface
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetcher_yfinance import get_options_chain, filter_options
from volatility.iv_solver import add_iv_to_dataframe
from config.settings import TICKER


def build_iv_matrix(df: pd.DataFrame,
                    option_type: str = "call") -> pd.DataFrame:
    """
    Build a 2D IV matrix: rows = strikes, columns = expiries.
    """
    df_type = df[df["option_type"] == option_type].copy()

    # Pivot table: strike vs expiry
    matrix = df_type.pivot_table(
        index="strike",
        columns="expiry",
        values="iv",
        aggfunc="mean"
    )

    print(f"[Surface] IV Matrix shape: {matrix.shape} "
          f"({matrix.shape[0]} strikes × {matrix.shape[1]} expiries)")
    return matrix


def build_smooth_surface(df: pd.DataFrame,
                         option_type: str = "call",
                         grid_points: int = 50) -> dict:
    """
    Build a smooth interpolated vol surface using RBF interpolation.

    Returns dict with:
        - strikes_grid  : 2D array of strike values
        - dte_grid      : 2D array of DTE values
        - iv_grid       : 2D array of interpolated IVs
        - raw_df        : filtered raw data used
    """
    df_type = df[df["option_type"] == option_type].copy()
    df_type = df_type.dropna(subset=["iv", "strike", "DTE"])

    # ── Input points ──
    strikes = df_type["strike"].values
    dtes    = df_type["DTE"].values
    ivs     = df_type["iv"].values

    print(f"[Surface] Building {option_type} surface with "
          f"{len(strikes)} data points...")

    # ── Create smooth grid ──
    strike_min, strike_max = strikes.min(), strikes.max()
    dte_min,    dte_max    = dtes.min(),    dtes.max()

    strike_range = np.linspace(strike_min, strike_max, grid_points)
    dte_range    = np.linspace(dte_min,    dte_max,    grid_points)

    strikes_grid, dte_grid = np.meshgrid(strike_range, dte_range)

    # ── RBF Interpolation ──
    try:
        points      = np.column_stack([strikes, dtes])
        grid_points_arr = np.column_stack([
            strikes_grid.ravel(),
            dte_grid.ravel()
        ])

        rbf         = RBFInterpolator(points, ivs, smoothing=0.001)
        iv_flat     = rbf(grid_points_arr)
        iv_grid     = iv_flat.reshape(strikes_grid.shape)

        # Smooth out any noise with gaussian filter
        iv_grid = gaussian_filter(iv_grid, sigma=1.0)

        # Clip to realistic vol range
        iv_grid = np.clip(iv_grid, 0.01, 2.0)

        print(f"[Surface] Surface built! "
              f"IV range: {iv_grid.min():.2%} → {iv_grid.max():.2%}")

    except Exception as e:
        print(f"[Surface] RBF failed ({e}), using linear fallback...")
        from scipy.interpolate import LinearNDInterpolator
        lin   = LinearNDInterpolator(
                    list(zip(strikes, dtes)), ivs, fill_value=np.nanmean(ivs))
        iv_grid = lin(strikes_grid, dte_grid)
        iv_grid = np.clip(iv_grid, 0.01, 2.0)

    return {
        "strikes_grid" : strikes_grid,
        "dte_grid"     : dte_grid,
        "iv_grid"      : iv_grid,
        "raw_df"       : df_type,
        "option_type"  : option_type
    }


def get_term_structure(df: pd.DataFrame,
                       option_type: str = "call") -> pd.DataFrame:
    """
    Extract ATM vol term structure (vol vs expiry at ATM strike).
    """
    df_type  = df[df["option_type"] == option_type].copy()
    spot     = df_type["spot_price"].iloc[0]

    # Find ATM strike per expiry (closest to spot)
    term = []
    for expiry, group in df_type.groupby("expiry"):
        idx    = (group["strike"] - spot).abs().idxmin()
        atm    = group.loc[idx]
        term.append({
            "expiry"  : expiry,
            "DTE"     : atm["DTE"],
            "atm_iv"  : atm["iv"],
            "strike"  : atm["strike"]
        })

    term_df = pd.DataFrame(term).sort_values("DTE")
    print(f"\n── ATM Vol Term Structure ({option_type.upper()}) ──")
    print(term_df.to_string(index=False))
    return term_df


def get_vol_smile(df: pd.DataFrame,
                  expiry_index: int = 0,
                  option_type: str = "call") -> pd.DataFrame:
    """
    Extract vol smile for a specific expiry (skew across strikes).
    """
    df_type  = df[df["option_type"] == option_type].copy()
    expiries = sorted(df_type["expiry"].unique())

    if expiry_index >= len(expiries):
        expiry_index = 0

    expiry   = expiries[expiry_index]
    smile_df = df_type[df_type["expiry"] == expiry].sort_values("strike")

    print(f"\n── Vol Smile for {expiry} ({option_type.upper()}) ──")
    print(smile_df[["strike", "moneyness", "iv"]].to_string(index=False))
    return smile_df


if __name__ == "__main__":
    print("=" * 55)
    print(f"  Vol Surface Builder — {TICKER}")
    print("=" * 55)

    # Full pipeline
    df       = get_options_chain()
    df       = filter_options(df)
    df       = add_iv_to_dataframe(df)

    # Build IV matrix
    matrix   = build_iv_matrix(df, option_type="call")

    # Build smooth surface
    surface  = build_smooth_surface(df, option_type="call", grid_points=50)

    # Term structure
    term_df  = get_term_structure(df, option_type="call")

    # Vol smile (nearest expiry)
    smile_df = get_vol_smile(df, expiry_index=0, option_type="call")

    print("\n✅ Surface ready for visualization!")