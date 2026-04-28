"""Report rendering — terminal output, CSV export, and equity curve plot."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from .results import BacktestResult


def print_report(result: BacktestResult) -> None:
    """Print a formatted report to the terminal using Rich."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        from rich.text import Text
    except ImportError:
        _print_plain_report(result)
        return

    console = Console()
    verdict_color = "bold green" if result.passed else "bold red"
    verdict_text = "PASSED ✓" if result.passed else f"FAILED ✗  ({result.failure_reason})"

    console.print()
    console.print(Panel(
        f"[bold]Prop Firm Backtest Report[/bold]\n"
        f"Firm: [cyan]{result.firm_name}[/cyan]  |  "
        f"Tier: [cyan]{result.tier_name}[/cyan]  |  "
        f"Contract: [cyan]{result.contract_symbol}[/cyan]\n"
        f"Period: {result.start_date} → {result.end_date}",
        title="[bold blue]PROP FIRM ENGINE[/bold blue]",
        border_style="blue",
    ))

    console.print(f"\n  Verdict: [{verdict_color}]{verdict_text}[/{verdict_color}]\n")

    # ── Rule checks ───────────────────────────────────────────────────────
    rule_table = Table(title="Challenge Rules", box=box.ROUNDED, show_header=True)
    rule_table.add_column("Rule", style="bold")
    rule_table.add_column("Required", justify="right")
    rule_table.add_column("Actual", justify="right")
    rule_table.add_column("Status", justify="center")

    for rc in result.rule_checks:
        status = "[green]PASS[/green]" if rc.passed else "[red]FAIL[/red]"
        if rc.rule_name in ("profit_target",):
            rule_table.add_row(
                rc.rule_name.replace("_", " ").title(),
                f"${rc.threshold:,.2f}",
                f"${rc.value:,.2f}",
                status,
            )
        elif rc.rule_name in ("trailing_drawdown", "daily_loss_limit"):
            rule_table.add_row(
                rc.rule_name.replace("_", " ").title(),
                "Not breached",
                "OK" if rc.passed else "BREACHED",
                status,
            )
        else:
            rule_table.add_row(
                rc.rule_name.replace("_", " ").title(),
                str(int(rc.threshold)),
                str(int(rc.value)),
                status,
            )
    console.print(rule_table)

    # ── Financial summary ─────────────────────────────────────────────────
    fin_table = Table(title="Financial Summary", box=box.ROUNDED)
    fin_table.add_column("Metric", style="bold")
    fin_table.add_column("Value", justify="right")
    net_pnl = result.final_realized_balance - result.starting_balance
    color = "green" if net_pnl >= 0 else "red"
    fin_table.add_row("Starting Balance", f"${result.starting_balance:,.2f}")
    fin_table.add_row("Final Balance", f"${result.final_realized_balance:,.2f}")
    fin_table.add_row("Net PnL", f"[{color}]${net_pnl:+,.2f}[/{color}]")
    fin_table.add_row("Peak Equity", f"${result.peak_equity:,.2f}")
    fin_table.add_row("Max Drawdown", f"${result.max_drawdown_dollars:,.2f}")
    console.print(fin_table)

    # ── Trade statistics ──────────────────────────────────────────────────
    s = result.stats
    if s.get("total_trades", 0) > 0:
        trade_table = Table(title="Trade Statistics", box=box.ROUNDED)
        trade_table.add_column("Metric", style="bold")
        trade_table.add_column("Value", justify="right")
        wr = s.get("win_rate", 0) * 100
        trade_table.add_row("Total Trades",     str(s.get("total_trades", 0)))
        trade_table.add_row("Win Rate",          f"{wr:.1f}%")
        trade_table.add_row("Avg Win",           f"${s.get('avg_win', 0):,.2f}")
        trade_table.add_row("Avg Loss",          f"${s.get('avg_loss', 0):,.2f}")
        trade_table.add_row("Largest Win",       f"${s.get('largest_win', 0):,.2f}")
        trade_table.add_row("Largest Loss",      f"${s.get('largest_loss', 0):,.2f}")
        trade_table.add_row("Expectancy",        f"${s.get('expectancy', 0):,.2f}")
        trade_table.add_row("Profit Factor",     f"{s.get('profit_factor', 0):.2f}")
        trade_table.add_row("Sharpe Ratio",      f"{s.get('sharpe_ratio', 0):.2f}")
        trade_table.add_row("Sortino Ratio",     f"{s.get('sortino_ratio', 0):.2f}")
        calmar = s.get("calmar_ratio", 0)
        calmar_str = f"{calmar:.2f}" if calmar != float("inf") else "inf"
        trade_table.add_row("Calmar Ratio",      calmar_str)
        trade_table.add_row("Ulcer Index",       f"{s.get('ulcer_index', 0):.2f}")
        trade_table.add_row("Max Consec Wins",   str(s.get("max_consec_wins", 0)))
        trade_table.add_row("Max Consec Losses", str(s.get("max_consec_losses", 0)))
        trade_table.add_row("Total Commission",  f"${s.get('total_commission', 0):,.2f}")
        trade_table.add_row("Trading Days",      str(s.get("trading_days", 0)))
        console.print(trade_table)

    console.print()


def save_report_csv(result: BacktestResult, output_dir: str | Path) -> None:
    """Save trades and equity curve as CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Trades CSV
    trades_path = out / "trades.csv"
    if result.trades:
        with open(trades_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "trade_id", "entry_time", "exit_time", "direction",
                "contracts", "entry_price", "exit_price",
                "gross_pnl", "commission", "net_pnl", "mae", "mfe",
            ])
            writer.writeheader()
            for t in result.trades:
                writer.writerow({
                    "trade_id": t.trade_id,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "direction": t.direction,
                    "contracts": t.contracts,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "gross_pnl": round(t.gross_pnl, 2),
                    "commission": round(t.commission, 2),
                    "net_pnl": round(t.net_pnl, 2),
                    "mae": round(t.mae, 2),
                    "mfe": round(t.mfe, 2),
                })

    # Equity curve CSV
    eq_path = out / "equity_curve.csv"
    with open(eq_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "equity"])
        for ts, eq in result.equity_curve:
            writer.writerow([ts.isoformat(), round(eq, 2)])

    print(f"Saved trades → {trades_path}")
    print(f"Saved equity curve → {eq_path}")


def plot_equity_curve(
    result: BacktestResult,
    output_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Plot the equity curve with the drawdown floor."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    if not result.equity_curve:
        print("No equity curve data to plot.")
        return

    timestamps = [ts for ts, _ in result.equity_curve]
    equities = [eq for _, eq in result.equity_curve]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Equity curve
    color = "green" if result.passed else "red"
    ax.plot(timestamps, equities, color=color, linewidth=1.5, label="Equity", zorder=3)

    # Starting balance reference line
    ax.axhline(
        y=result.starting_balance,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"Starting Balance ${result.starting_balance:,.0f}",
    )

    # Profit target line
    for rc in result.rule_checks:
        if rc.rule_name == "profit_target":
            target_line = result.starting_balance + rc.threshold
            ax.axhline(
                y=target_line,
                color="blue",
                linestyle=":",
                linewidth=1.0,
                label=f"Profit Target ${target_line:,.0f}",
            )
            break

    ax.fill_between(timestamps, equities, result.starting_balance, alpha=0.1, color=color)

    verdict = "PASSED" if result.passed else f"FAILED ({result.failure_reason})"
    ax.set_title(
        f"{result.firm_name} {result.tier_name} — {result.contract_symbol} — {verdict}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved equity curve plot → {path}")

    if show:
        plt.show()

    plt.close(fig)


def _print_plain_report(result: BacktestResult) -> None:
    """Fallback plain-text report when rich is not available."""
    print("\n" + "=" * 60)
    print("  PROP FIRM BACKTEST REPORT")
    print(f"  {result.firm_name} | {result.tier_name} | {result.contract_symbol}")
    print(f"  {result.start_date} → {result.end_date}")
    print("=" * 60)
    verdict = "PASSED" if result.passed else f"FAILED ({result.failure_reason})"
    print(f"  Verdict: {verdict}")
    print()
    for rc in result.rule_checks:
        status = "PASS" if rc.passed else "FAIL"
        print(f"  [{status}] {rc.rule_name}: {rc.detail}")
    print()
    net = result.final_realized_balance - result.starting_balance
    print(f"  Net PnL: ${net:+,.2f}")
    print(f"  Total Trades: {result.stats.get('total_trades', 0)}")
    print(f"  Win Rate: {result.stats.get('win_rate', 0)*100:.1f}%")
    print("=" * 60 + "\n")
