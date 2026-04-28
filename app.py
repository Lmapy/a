"""Streamlit UI for the Prop Firm Backtesting Engine."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prop Firm Backtest Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (lazy so Streamlit renders fast) ──────────────────────────────────
from prop_backtest.contracts.specs import CONTRACT_REGISTRY, get_contract
from prop_backtest.data.loader import DataLoader
from prop_backtest.engine.backtest import BacktestEngine
from prop_backtest.firms import FIRM_REGISTRY, get_firm
from prop_backtest.reporting.results import BacktestResult

# ── Helpers ───────────────────────────────────────────────────────────────────

FIRM_KEYS = sorted({v.firm_name: k for k, v in FIRM_REGISTRY.items()}.values())
FIRM_NAMES = {k: FIRM_REGISTRY[k].firm_name for k in FIRM_KEYS}

STRATEGY_OPTIONS = {
    "SMA Crossover":       "prop_backtest.strategy.examples.sma_crossover.SMACrossover",
    "RSI Mean Reversion":  "prop_backtest.strategy.examples.rsi_mean_reversion.RSIMeanReversion",
    "ATR Trend":           "prop_backtest.strategy.examples.atr_trend.ATRTrend",
    "Hold (no trades)":    None,
}

CONTRACT_NAMES = list(CONTRACT_REGISTRY.keys())


def _load_strategy(name: str, params: dict):
    if name == "SMA Crossover":
        from prop_backtest.strategy.examples.sma_crossover import SMACrossover
        return SMACrossover(fast=params["fast"], slow=params["slow"])
    if name == "RSI Mean Reversion":
        from prop_backtest.strategy.examples.rsi_mean_reversion import RSIMeanReversion
        return RSIMeanReversion(rsi_period=params["rsi_period"],
                                oversold=params["oversold"],
                                overbought=params["overbought"])
    if name == "ATR Trend":
        from prop_backtest.strategy.examples.atr_trend import ATRTrend
        return ATRTrend(fast=params["fast"], slow=params["slow"],
                        risk_dollars=params["risk_dollars"])
    from prop_backtest.strategy.base import FunctionStrategy
    from prop_backtest.engine.broker import Signal
    return FunctionStrategy(lambda bar, hist, acc: Signal(action="hold"), name="Hold")


def _equity_fig(result: BacktestResult) -> go.Figure:
    ts   = [t for t, _ in result.equity_curve]
    eq   = [e for _, e in result.equity_curve]
    color = "#00c853" if result.passed else "#ff1744"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=eq, mode="lines", name="Equity",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in color else color + "14",
    ))
    fig.add_hline(y=result.starting_balance, line_dash="dash",
                  line_color="gray", annotation_text="Starting Balance")

    for rc in result.rule_checks:
        if rc.rule_name == "profit_target":
            fig.add_hline(
                y=result.starting_balance + rc.threshold,
                line_dash="dot", line_color="#2979ff",
                annotation_text=f"Target ${result.starting_balance + rc.threshold:,.0f}",
            )

    verdict = "PASSED ✓" if result.passed else f"FAILED ✗ ({result.failure_reason})"
    fig.update_layout(
        title=dict(text=f"{result.firm_name} {result.tier_name} — {result.contract_symbol} — {verdict}",
                   font=dict(size=15)),
        xaxis_title="Date", yaxis_title="Equity ($)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h"),
        hovermode="x unified",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
    )
    return fig


def _drawdown_fig(result: BacktestResult) -> go.Figure:
    ts  = [t for t, _ in result.equity_curve]
    eq  = [e for _, e in result.equity_curve]
    peak = eq[0]
    dd = []
    for e in eq:
        if e > peak:
            peak = e
        dd.append(-(peak - e))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=dd, mode="lines", name="Drawdown",
        line=dict(color="#ff6d00", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,109,0,0.15)",
    ))
    fig.update_layout(
        title="Drawdown ($)", height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Prop Firm Engine")
    st.divider()

    st.subheader("Firm & Account")
    firm_key = st.selectbox("Prop Firm", FIRM_KEYS,
                             format_func=lambda k: FIRM_REGISTRY[k].firm_name)
    firm_rules = get_firm(firm_key)
    tier_names = [t.name for t in firm_rules.tiers]
    tier_name  = st.selectbox("Account Tier", tier_names)
    tier       = firm_rules.get_tier(tier_name)

    st.divider()
    st.subheader("Contract & Data")
    contract_sym = st.selectbox("Contract", CONTRACT_NAMES)
    contract     = get_contract(contract_sym)

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", value=pd.Timestamp("2022-01-01"))
    end_date   = col2.date_input("End",   value=pd.Timestamp("2024-12-31"))

    source = st.radio("Data Source", ["yfinance", "barchart", "CSV upload"],
                      horizontal=True)

    uploaded_csv = None
    barchart_key = None
    if source == "CSV upload":
        uploaded_csv = st.file_uploader("Upload OHLCV CSV", type=["csv"])
    elif source == "barchart":
        barchart_key = st.text_input("Barchart API Key (optional)", type="password")

    interval = st.selectbox("Bar Interval", ["1d", "1h", "30m", "15m", "5m"],
                             help="Note: intraday history is limited by source")

    st.divider()
    st.subheader("Strategy")
    strat_name = st.selectbox("Strategy", list(STRATEGY_OPTIONS.keys()))

    strat_params = {}
    if strat_name == "SMA Crossover":
        c1, c2 = st.columns(2)
        strat_params["fast"] = c1.number_input("Fast period", 2, 50, 9)
        strat_params["slow"] = c2.number_input("Slow period", 3, 200, 21)
    elif strat_name == "RSI Mean Reversion":
        strat_params["rsi_period"]  = st.number_input("RSI period", 2, 50, 14)
        c1, c2 = st.columns(2)
        strat_params["oversold"]    = c1.number_input("Oversold", 1, 49, 30)
        strat_params["overbought"]  = c2.number_input("Overbought", 51, 99, 70)
    elif strat_name == "ATR Trend":
        c1, c2 = st.columns(2)
        strat_params["fast"]         = c1.number_input("Fast EMA", 2, 50, 9)
        strat_params["slow"]         = c2.number_input("Slow EMA", 3, 200, 21)
        strat_params["risk_dollars"] = st.number_input("Risk per trade ($)", 50, 5000, 200)

    st.divider()
    st.subheader("Execution")
    commission = st.number_input("Commission / RT ($)", 0.0, 50.0, 4.50, step=0.50)
    slippage   = st.number_input("Slippage (ticks)",    0,   10,   0)

    st.divider()
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

    run_mc  = st.checkbox("Monte Carlo (1 000 sims)", value=False)
    run_wfo = st.checkbox("Walk-Forward Optimisation", value=False)


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Prop Firm Backtest Engine")

if not run_btn:
    # Landing screen
    st.markdown("""
    **Configure your backtest in the sidebar, then click ▶ Run Backtest.**

    | What | Details |
    |---|---|
    | **Firms** | TopStep · MyFundedFutures · Lucid |
    | **Contracts** | ES · NQ · CL · GC · RTY · MES · MNQ |
    | **Strategies** | SMA Crossover · RSI Mean Reversion · ATR Trend |
    | **Data** | yfinance (free) · Barchart · your own CSV |
    | **Analysis** | Monte Carlo pass probability · Walk-Forward optimiser |
    """)

    st.info("💡 On your machine yfinance pulls real CME futures data automatically — no API key needed.")
    st.stop()


# ── Run ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    try:
        loader = DataLoader(contract)

        if source == "CSV upload":
            if uploaded_csv is None:
                st.error("Please upload a CSV file.")
                st.stop()
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(uploaded_csv.read())
                tmp_path = tmp.name
            bars = loader.load(str(start_date), str(end_date), local_csv_path=tmp_path)
            os.unlink(tmp_path)
        elif source == "barchart":
            bars = loader.load(str(start_date), str(end_date), interval=interval,
                               source="barchart", barchart_api_key=barchart_key or None)
        else:
            bars = loader.load(str(start_date), str(end_date), interval=interval)

    except Exception as e:
        st.error(f"**Data load failed:** {e}")
        st.stop()

st.caption(f"Loaded **{len(bars):,} bars** — {bars[0].timestamp.date()} → {bars[-1].timestamp.date()}")

with st.spinner("Running backtest..."):
    try:
        strategy = _load_strategy(strat_name, strat_params)
        engine   = BacktestEngine(strategy, firm_rules, tier_name, contract,
                                  commission_per_rt=commission,
                                  slippage_ticks=slippage)
        result   = engine.run(bars)
    except Exception as e:
        st.error(f"**Backtest failed:** {e}")
        st.stop()

# ── Verdict banner ────────────────────────────────────────────────────────────
if result.passed:
    st.success(f"### ✅ PASSED — {result.firm_name} {result.tier_name}")
else:
    st.error(f"### ❌ FAILED — {result.failure_reason.replace('_', ' ').title() if result.failure_reason else 'Unknown'}")

# ── KPI row ───────────────────────────────────────────────────────────────────
s = result.stats
net_pnl = result.final_realized_balance - result.starting_balance
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Net PnL",       f"${net_pnl:+,.0f}")
k2.metric("Trades",        s.get("total_trades", 0))
k3.metric("Win Rate",      f"{s.get('win_rate', 0)*100:.1f}%")
k4.metric("Sharpe",        f"{s.get('sharpe_ratio', 0):.2f}")
k5.metric("Calmar",        f"{s.get('calmar_ratio', 0):.2f}" if s.get('calmar_ratio') != float('inf') else "∞")
k6.metric("Max Drawdown",  f"${result.max_drawdown_dollars:,.0f}")

st.divider()

# ── Equity curve + drawdown ───────────────────────────────────────────────────
if result.equity_curve:
    st.plotly_chart(_equity_fig(result), use_container_width=True)
    st.plotly_chart(_drawdown_fig(result), use_container_width=True)

# ── Rule checks + stats side by side ─────────────────────────────────────────
col_rules, col_stats = st.columns(2)

with col_rules:
    st.subheader("Challenge Rules")
    rules_data = []
    for rc in result.rule_checks:
        rules_data.append({
            "Rule":     rc.rule_name.replace("_", " ").title(),
            "Required": (f"${rc.threshold:,.2f}" if rc.rule_name == "profit_target"
                         else ("Not breached" if rc.rule_name in ("trailing_drawdown", "daily_loss_limit")
                               else str(int(rc.threshold)))),
            "Actual":   (f"${rc.value:,.2f}" if rc.rule_name == "profit_target"
                         else ("OK" if rc.passed else "BREACHED") if rc.rule_name in ("trailing_drawdown", "daily_loss_limit")
                         else str(int(rc.value))),
            "Status":   "✅ PASS" if rc.passed else "❌ FAIL",
        })
    st.dataframe(pd.DataFrame(rules_data), hide_index=True, use_container_width=True)

with col_stats:
    st.subheader("Performance Metrics")
    metrics = {
        "Expectancy":        f"${s.get('expectancy', 0):,.2f}",
        "Profit Factor":     f"{s.get('profit_factor', 0):.2f}",
        "Avg Win":           f"${s.get('avg_win', 0):,.2f}",
        "Avg Loss":          f"${s.get('avg_loss', 0):,.2f}",
        "Largest Win":       f"${s.get('largest_win', 0):,.2f}",
        "Largest Loss":      f"${s.get('largest_loss', 0):,.2f}",
        "Sortino Ratio":     f"{s.get('sortino_ratio', 0):.2f}",
        "Ulcer Index":       f"{s.get('ulcer_index', 0):.2f}",
        "Max Consec Wins":   str(s.get("max_consec_wins", 0)),
        "Max Consec Losses": str(s.get("max_consec_losses", 0)),
        "Avg Drawdown":      f"${s.get('avg_drawdown_dollars', 0):,.2f}",
        "Total Commission":  f"${s.get('total_commission', 0):,.2f}",
    }
    st.dataframe(
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"]),
        hide_index=True, use_container_width=True,
    )

# ── Trade log ─────────────────────────────────────────────────────────────────
if result.trades:
    st.divider()
    st.subheader(f"Trade Log ({len(result.trades)} trades)")
    trades_df = pd.DataFrame([{
        "#":          t.trade_id,
        "Entry":      t.entry_time.strftime("%Y-%m-%d"),
        "Exit":       t.exit_time.strftime("%Y-%m-%d"),
        "Dir":        t.direction.upper(),
        "Contracts":  t.contracts,
        "Entry $":    f"{t.entry_price:.2f}",
        "Exit $":     f"{t.exit_price:.2f}",
        "Gross PnL":  f"${t.gross_pnl:,.2f}",
        "Commission": f"${t.commission:,.2f}",
        "Net PnL":    f"${t.net_pnl:,.2f}",
        "MAE":        f"${t.mae:,.2f}",
        "MFE":        f"${t.mfe:,.2f}",
    } for t in result.trades])
    st.dataframe(trades_df, hide_index=True, use_container_width=True)

    # Download button
    csv_bytes = trades_df.to_csv(index=False).encode()
    st.download_button("⬇ Download Trade Log CSV", csv_bytes,
                       file_name="trades.csv", mime="text/csv")

# ── Monte Carlo ───────────────────────────────────────────────────────────────
if run_mc and result.trades:
    st.divider()
    st.subheader("Monte Carlo Pass Probability")
    with st.spinner("Running 1,000 simulations..."):
        from prop_backtest.analysis.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(result, firm_rules, tier_name, n_simulations=1000)

    pass_pct = mc.pass_rate * 100
    colour   = "normal" if pass_pct >= 60 else "off" if pass_pct >= 30 else "inverse"
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Pass Rate",          f"{pass_pct:.1f}%")
    mc2.metric("Median Final Equity",f"${mc.median_final_equity:,.0f}")
    mc3.metric("5th Pct Equity",     f"${mc.p5_final_equity:,.0f}")
    mc4.metric("95th Pct Equity",    f"${mc.p95_final_equity:,.0f}")

    # Distribution histogram
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=mc.final_equities, nbinsx=50, name="Final Equity",
        marker_color="#2979ff", opacity=0.75,
    ))
    fig_mc.add_vline(x=result.starting_balance + tier.profit_target,
                     line_dash="dot", line_color="#00c853",
                     annotation_text="Profit Target")
    fig_mc.update_layout(
        title="Distribution of Final Equity (1,000 sims)",
        xaxis_title="Final Equity ($)", yaxis_title="Count",
        height=300, margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    if pass_pct < 30:
        st.error("Strategy has <30% pass probability — edge is insufficient.")
    elif pass_pct < 60:
        st.warning("Marginal edge — consider tighter risk management.")
    else:
        st.success("Strong pass probability — strategy has consistent edge.")

elif run_mc and not result.trades:
    st.warning("Monte Carlo skipped — no trades were made.")

# ── Walk-Forward ──────────────────────────────────────────────────────────────
if run_wfo:
    st.divider()
    st.subheader("Walk-Forward Optimisation")

    wfo_col1, wfo_col2, wfo_col3 = st.columns(3)
    is_window  = wfo_col1.number_input("In-sample bars",     50, 500, 120)
    oos_window = wfo_col2.number_input("Out-of-sample bars", 10, 200,  30)
    step       = wfo_col3.number_input("Step bars",          10, 200,  30)

    if strat_name not in ("SMA Crossover",):
        st.info("Walk-forward parameter search is currently available for SMA Crossover. Other strategies use fixed params.")

    if st.button("▶  Run Walk-Forward"):
        with st.spinner("Optimising across folds..."):
            try:
                from prop_backtest.analysis.walk_forward import WalkForwardOptimizer
                from prop_backtest.strategy.examples.sma_crossover import SMACrossover

                param_grid = [
                    {"fast": f, "slow": s}
                    for f in [3, 5, 9, 15]
                    for s in [10, 15, 21, 30]
                    if f < s
                ]
                wfo = WalkForwardOptimizer(
                    firm_rules=firm_rules, tier_name=tier_name, contract=contract,
                    strategy_factory=lambda p: SMACrossover(fast=p["fast"], slow=p["slow"]),
                    param_grid=param_grid,
                    is_window=is_window, oos_window=oos_window, step=step,
                    commission_per_rt=commission, slippage_ticks=slippage,
                )
                wfo_result = wfo.run(bars)

                # Summary metrics
                w1, w2, w3 = st.columns(3)
                w1.metric("Total OOS PnL",  f"${wfo_result.total_oos_pnl:,.2f}")
                w2.metric("Avg OOS Sharpe", f"{wfo_result.avg_oos_sharpe:.2f}")
                w3.metric("OOS Pass Rate",  f"{wfo_result.pass_rate*100:.0f}%")

                # Per-fold table
                fold_data = [{
                    "Fold":       f.fold_index + 1,
                    "IS Score":   f"{f.is_score:.2f}",
                    "Best Params": ", ".join(f"{k}={v}" for k, v in f.best_params.items()),
                    "OOS PnL":    f"${f.oos_net_pnl:,.2f}",
                    "OOS Sharpe": f"{f.oos_sharpe:.2f}",
                    "Trades":     len(f.oos_result.trades),
                    "Pass":       "✅" if f.oos_passed else "❌",
                } for f in wfo_result.folds]
                st.dataframe(pd.DataFrame(fold_data), hide_index=True, use_container_width=True)

                # OOS equity curve
                if wfo_result.oos_equity_curve:
                    oos_ts = [t for t, _ in wfo_result.oos_equity_curve]
                    oos_eq = [e for _, e in wfo_result.oos_equity_curve]
                    fig_wfo = go.Figure()
                    fig_wfo.add_trace(go.Scatter(
                        x=oos_ts, y=oos_eq, mode="lines", name="OOS Equity",
                        line=dict(color="#aa00ff", width=2),
                    ))
                    fig_wfo.update_layout(
                        title="Stitched Out-of-Sample Equity Curve",
                        height=300, margin=dict(l=10, r=10, t=50, b=10),
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font=dict(color="#fafafa"),
                        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
                    )
                    st.plotly_chart(fig_wfo, use_container_width=True)

                # Parameter stability
                st.markdown("**Parameter Stability**")
                for param, counts in wfo_result.param_stability.items():
                    total_folds = sum(counts.values())
                    cols = st.columns(len(counts))
                    for i, (val, cnt) in enumerate(sorted(counts.items())):
                        cols[i].metric(f"{param}={val}", f"{cnt/total_folds*100:.0f}%")

            except Exception as e:
                st.error(f"Walk-forward failed: {e}")
