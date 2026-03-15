# data_store/storage.py
# ─────────────────────────────────────────
# SQLite historical storage
# Saves vol surface snapshots over time
# ─────────────────────────────────────────

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TICKER


DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vol_history.db"
)


# ══════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════
def init_database():
    """Create all tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # ── Surface snapshots ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS surface_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT,
            timestamp   TEXT,
            spot_price  REAL,
            atm_iv      REAL,
            avg_iv      REAL,
            skew        REAL,
            total_opts  INTEGER,
            iv_matrix   TEXT
        )
    """)

    # ── ATM term structure history ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS term_structure (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT,
            timestamp   TEXT,
            spot_price  REAL,
            expiry      TEXT,
            dte         INTEGER,
            atm_iv      REAL
        )
    """)

    # ── Greeks history ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS greeks_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT,
            timestamp   TEXT,
            spot_price  REAL,
            expiry      TEXT,
            dte         INTEGER,
            strike      REAL,
            option_type TEXT,
            delta       REAL,
            gamma       REAL,
            vega        REAL,
            theta       REAL,
            iv          REAL
        )
    """)

    # ── Risk metrics history ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS risk_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT,
            timestamp       TEXT,
            spot_price      REAL,
            atm_iv          REAL,
            var_95          REAL,
            cvar_95         REAL,
            var_99          REAL,
            cvar_99         REAL,
            skew            REAL,
            spike_alerts    INTEGER
        )
    """)

    # ── Vol spike alerts ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS spike_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT,
            timestamp   TEXT,
            expiry      TEXT,
            strike      REAL,
            option_type TEXT,
            iv          REAL,
            z_score     REAL,
            severity    TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


# ══════════════════════════════════════════
#  SAVE FUNCTIONS
# ══════════════════════════════════════════
def save_surface_snapshot(df: pd.DataFrame,
                          var_results: dict = None):
    """Save current vol surface snapshot to DB."""
    conn = sqlite3.connect(DB_PATH)
    now  = datetime.now().isoformat()
    spot = df["spot_price"].iloc[0]

    # ATM IV
    atm_df = df[df["strike"].between(spot * 0.99, spot * 1.01)]
    atm_iv = atm_df["iv"].mean() if len(atm_df) > 0 else df["iv"].mean()

    # Put-call skew
    calls  = df[(df["option_type"] == "call") &
                df["strike"].between(spot * 0.99, spot * 1.01)]
    puts   = df[(df["option_type"] == "put") &
                df["strike"].between(spot * 0.99, spot * 1.01)]
    skew   = (puts["iv"].mean() - calls["iv"].mean()) \
             if len(puts) > 0 and len(calls) > 0 else 0

    # IV matrix as JSON
    try:
        matrix = df.pivot_table(
            index="strike", columns="expiry",
            values="iv", aggfunc="mean"
        )
        iv_json = matrix.to_json()
    except Exception:
        iv_json = "{}"

    conn.execute("""
        INSERT INTO surface_snapshots
        (ticker, timestamp, spot_price, atm_iv,
         avg_iv, skew, total_opts, iv_matrix)
        VALUES (?,?,?,?,?,?,?,?)
    """, (TICKER, now, spot, round(atm_iv, 6),
          round(df["iv"].mean(), 6),
          round(skew, 6), len(df), iv_json))

    conn.commit()
    conn.close()
    print(f"[DB] Surface snapshot saved @ {now[:19]}")


def save_term_structure(df: pd.DataFrame):
    """Save ATM term structure to DB."""
    from volatility.surface_builder import get_term_structure

    conn     = sqlite3.connect(DB_PATH)
    now      = datetime.now().isoformat()
    spot     = df["spot_price"].iloc[0]
    term_df  = get_term_structure(df, "call")

    rows = []
    for _, row in term_df.iterrows():
        rows.append((
            TICKER, now, spot,
            row["expiry"],
            int(row["DTE"]),
            round(row["atm_iv"], 6)
        ))

    conn.executemany("""
        INSERT INTO term_structure
        (ticker, timestamp, spot_price,
         expiry, dte, atm_iv)
        VALUES (?,?,?,?,?,?)
    """, rows)

    conn.commit()
    conn.close()
    print(f"[DB] Term structure saved — {len(rows)} expiries")


def save_greeks(df: pd.DataFrame):
    """Save ATM Greeks to DB."""
    conn = sqlite3.connect(DB_PATH)
    now  = datetime.now().isoformat()
    spot = df["spot_price"].iloc[0]

    # Save ATM Greeks only (keep DB small)
    atm_df = df[df["strike"].between(spot * 0.98, spot * 1.02)]

    rows = []
    for _, row in atm_df.iterrows():
        rows.append((
            TICKER, now, spot,
            row["expiry"],
            int(row["DTE"]),
            row["strike"],
            row["option_type"],
            row.get("delta", None),
            row.get("gamma", None),
            row.get("vega",  None),
            row.get("theta", None),
            row["iv"]
        ))

    conn.executemany("""
        INSERT INTO greeks_history
        (ticker, timestamp, spot_price, expiry,
         dte, strike, option_type,
         delta, gamma, vega, theta, iv)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    conn.commit()
    conn.close()
    print(f"[DB] Greeks saved — {len(rows)} ATM options")


def save_risk_metrics(df: pd.DataFrame,
                      var_results: dict,
                      spike_count: int = 0):
    """Save risk metrics snapshot to DB."""
    conn = sqlite3.connect(DB_PATH)
    now  = datetime.now().isoformat()
    spot = df["spot_price"].iloc[0]

    # Also calculate 99% VaR
    from risk.var_calculator import calculate_var_cvar
    var_99 = calculate_var_cvar(df, confidence=0.99,
                                horizon_days=1,
                                portfolio_value=100000)

    atm_df = df[df["strike"].between(spot * 0.99, spot * 1.01)]
    calls  = atm_df[atm_df["option_type"] == "call"]
    puts   = atm_df[atm_df["option_type"] == "put"]
    skew   = (puts["iv"].mean() - calls["iv"].mean()) \
             if len(puts) > 0 and len(calls) > 0 else 0

    conn.execute("""
        INSERT INTO risk_history
        (ticker, timestamp, spot_price, atm_iv,
         var_95, cvar_95, var_99, cvar_99,
         skew, spike_alerts)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        TICKER, now, spot,
        round(var_results["atm_iv_pct"], 4),
        round(var_results["var_usd"],    2),
        round(var_results["cvar_usd"],   2),
        round(var_99["var_usd"],         2),
        round(var_99["cvar_usd"],        2),
        round(skew, 6),
        spike_count
    ))

    conn.commit()
    conn.close()
    print(f"[DB] Risk metrics saved")


def save_spike_alerts(alerts_df: pd.DataFrame):
    """Save vol spike alerts to DB."""
    if len(alerts_df) == 0:
        return

    conn = sqlite3.connect(DB_PATH)
    now  = datetime.now().isoformat()

    rows = []
    for _, row in alerts_df.iterrows():
        rows.append((
            TICKER, now,
            row["expiry"],
            row["strike"],
            row["option_type"],
            row["iv"],
            row["z_score"],
            row["severity"]
        ))

    conn.executemany("""
        INSERT INTO spike_alerts
        (ticker, timestamp, expiry, strike,
         option_type, iv, z_score, severity)
        VALUES (?,?,?,?,?,?,?,?)
    """, rows)

    conn.commit()
    conn.close()
    print(f"[DB] {len(rows)} spike alerts saved")


# ══════════════════════════════════════════
#  QUERY FUNCTIONS
# ══════════════════════════════════════════
def get_iv_history(days: int = 7) -> pd.DataFrame:
    """Get ATM IV history for last N days."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
        SELECT timestamp, spot_price, atm_iv,
               avg_iv, skew, total_opts
        FROM surface_snapshots
        WHERE ticker = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, conn, params=(TICKER, days * 100))
    conn.close()
    return df


def get_risk_history(days: int = 7) -> pd.DataFrame:
    """Get risk metrics history."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
        SELECT timestamp, spot_price, atm_iv,
               var_95, cvar_95, skew, spike_alerts
        FROM risk_history
        WHERE ticker = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, conn, params=(TICKER, days * 100))
    conn.close()
    return df


def get_spike_history() -> pd.DataFrame:
    """Get all spike alerts."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
        SELECT timestamp, expiry, strike,
               option_type, iv, z_score, severity
        FROM spike_alerts
        WHERE ticker = ?
        ORDER BY timestamp DESC
        LIMIT 100
    """, conn, params=(TICKER,))
    conn.close()
    return df


def print_db_summary():
    """Print database statistics."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    tables = [
        "surface_snapshots",
        "term_structure",
        "greeks_history",
        "risk_history",
        "spike_alerts"
    ]

    print("\n" + "=" * 50)
    print(f"  DATABASE SUMMARY — {TICKER}")
    print("=" * 50)

    for table in tables:
        try:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            count = c.fetchone()[0]
            print(f"  {table:<25}: {count:>6} rows")
        except Exception:
            print(f"  {table:<25}: not found")

    # Latest snapshot
    try:
        c.execute("""
            SELECT timestamp, spot_price, atm_iv
            FROM surface_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """)
        row = c.fetchone()
        if row:
            print(f"\n  Latest Snapshot  : {row[0][:19]}")
            print(f"  Spot             : ${row[1]:.2f}")
            print(f"  ATM IV           : {row[2]*100:.2f}%")
    except Exception:
        pass

    print("=" * 50)
    conn.close()


# ══════════════════════════════════════════
#  FULL SAVE PIPELINE
# ══════════════════════════════════════════
def save_full_snapshot(df: pd.DataFrame,
                       var_results: dict,
                       spikes: dict,
                       has_greeks: bool = False):
    """Save everything in one call."""
    print("[DB] Saving full snapshot...")

    save_surface_snapshot(df, var_results)
    save_term_structure(df)
    save_risk_metrics(df, var_results,
                      spikes.get("total_alerts", 0))

    if has_greeks:
        save_greeks(df)

    if spikes.get("total_alerts", 0) > 0:
        save_spike_alerts(spikes["alerts"])

    print_db_summary()


if __name__ == "__main__":
    from data.fetcher_yfinance import get_options_chain, filter_options
    from volatility.iv_solver import add_iv_to_dataframe
    from risk.var_calculator import calculate_var_cvar
    from risk.anomaly_detector import detect_vol_spikes

    print("=" * 50)
    print(f"  Storage Test — {TICKER}")
    print("=" * 50)

    # Init DB
    init_database()

    # Full pipeline
    df          = get_options_chain()
    df          = filter_options(df)
    df          = add_iv_to_dataframe(df)
    var_results = calculate_var_cvar(df)
    spikes      = detect_vol_spikes(df)

    # Save everything
    save_full_snapshot(df, var_results, spikes)

    # Query back
    print("\n── IV History ──")
    history = get_iv_history(days=1)
    print(history.head())

    print("\n── Risk History ──")
    risk = get_risk_history(days=1)
    print(risk.head())