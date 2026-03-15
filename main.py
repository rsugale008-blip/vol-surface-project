# main.py
# ─────────────────────────────────────────
# Master entry point
# Runs the full vol surface pipeline
# ─────────────────────────────────────────

import sys, os
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher_yfinance      import get_options_chain, filter_options
from volatility.iv_solver       import add_iv_to_dataframe
from volatility.surface_builder import (build_smooth_surface,
                                         get_term_structure)
from risk.var_calculator        import calculate_var_cvar, print_risk_report
from risk.anomaly_detector      import (detect_vol_spikes,
                                         calculate_skew_metrics,
                                         detect_iv_anomalies)
from config.settings            import TICKER, REFRESH_INTERVAL


# ══════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════
def print_banner():
    print("\n")
    print("█" * 55)
    print("█                                                     █")
    print("█     SPY LIVE VOLATILITY SURFACE SYSTEM              █")
    print("█     Quant Finance | Risk Management | AI            █")
    print("█                                                     █")
    print("█" * 55)
    print(f"  Ticker          : {TICKER}")
    print(f"  Refresh Rate    : Every {REFRESH_INTERVAL}s")
    print(f"  Started         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 55)
    print()


# ══════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════
def run_pipeline(verbose: bool = True) -> dict:
    """
    Run the complete vol surface pipeline.
    Returns all results as a dict.
    """
    start = time.time()
    now   = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] ── Running Pipeline ──")

    # ── Step 1: Fetch Data ──
    print("[1/5] Fetching live options chain...")
    df = get_options_chain()
    df = filter_options(df)

    # ── Step 2: Calculate IV ──
    print("[2/5] Calculating implied volatilities...")
    df = add_iv_to_dataframe(df)

    # ── Step 3: Build Surface ──
    print("[3/5] Building volatility surface...")
    surface  = build_smooth_surface(df, "call", grid_points=50)
    term_df  = get_term_structure(df, "call")

    # ── Step 4: Risk Metrics ──
    print("[4/5] Calculating risk metrics...")
    var_results = calculate_var_cvar(
        df,
        confidence=0.95,
        horizon_days=1,
        portfolio_value=100000
    )

    # ── Step 5: AI Anomaly Detection ──
    print("[5/5] Running AI anomaly detection...")
    spikes    = detect_vol_spikes(df)
    skew      = calculate_skew_metrics(df)
    anomalies = detect_iv_anomalies(df)

    elapsed = time.time() - start

    # ── Summary ──
    spot = df["spot_price"].iloc[0]
    print(f"\n{'─'*55}")
    print(f"  ✅ Pipeline Complete in {elapsed:.1f}s")
    print(f"  📊 {TICKER} Spot     : ${spot:.2f}")
    print(f"  📈 ATM IV        : {var_results['atm_iv_pct']:.2f}%")
    print(f"  ⚠️  VaR (95%)    : ${var_results['var_usd']:,.0f}")
    print(f"  🚨 Spike Alerts  : {spikes['total_alerts']}")
    print(f"  🤖 Anomalies     : {anomalies['anomaly'].sum()}")
    print(f"{'─'*55}")

    if verbose:
        print_risk_report(var_results)

    return {
        "df"         : df,
        "surface"    : surface,
        "term_df"    : term_df,
        "var"        : var_results,
        "spikes"     : spikes,
        "skew"       : skew,
        "anomalies"  : anomalies
    }


# ══════════════════════════════════════════
#  SCHEDULER — Auto Refresh
# ══════════════════════════════════════════
def start_scheduler():
    """Run pipeline on schedule + keep running."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=lambda: run_pipeline(verbose=False),
        trigger="interval",
        seconds=REFRESH_INTERVAL,
        id="vol_pipeline",
        max_instances=1
    )
    scheduler.start()
    print(f"\n[Scheduler] Running every {REFRESH_INTERVAL}s")
    print("[Scheduler] Press Ctrl+C to stop\n")
    return scheduler


# ══════════════════════════════════════════
#  MAIN MENU
# ══════════════════════════════════════════
def main():
    print_banner()

    print("Select mode:")
    print("  [1] Run pipeline once")
    print("  [2] Run pipeline + auto-refresh (scheduler)")
    print("  [3] Launch live dashboard (browser)")
    print("  [4] Full system (scheduler + dashboard)")
    print()

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        # ── Single run ──
        run_pipeline(verbose=True)

    elif choice == "2":
        # ── Scheduler only ──
        run_pipeline(verbose=True)
        scheduler = start_scheduler()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.shutdown()
            print("\n[Scheduler] Stopped.")

    elif choice == "3":
        # ── Dashboard only ──
        print("\n[Dashboard] Launching...")
        print("→ Open browser at: http://localhost:8050\n")
        from visualization.dashboard import app
        app.run(debug=False, port=8050)

    elif choice == "4":
        # ── Full system ──
        run_pipeline(verbose=True)
        scheduler = start_scheduler()
        print("\n[System] Launching dashboard...")
        print("→ Open browser at: http://localhost:8050\n")
        from visualization.dashboard import app
        app.run(debug=False, port=8050)

    else:
        print("Invalid choice. Running pipeline once...")
        run_pipeline(verbose=True)


if __name__ == "__main__":
    main()