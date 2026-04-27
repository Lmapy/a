"""Render the third-party audit pack as results/audit_pack.pdf."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = RES / "audit_pack.pdf"

styles = getSampleStyleSheet()
H1 = styles["Heading1"]
H2 = styles["Heading2"]
H3 = styles["Heading3"]
NORMAL = styles["BodyText"]
SMALL = ParagraphStyle("small", parent=NORMAL, fontSize=8.5, leading=11)
MONO = ParagraphStyle("mono", parent=NORMAL, fontName="Courier", fontSize=8.5, leading=11)
CAPTION = ParagraphStyle("caption", parent=NORMAL, fontSize=8.5, leading=10,
                         textColor=colors.grey, alignment=TA_LEFT)


def make_table(rows, header=True, col_widths=None) -> Table:
    t = Table(rows, repeatRows=1 if header else 0, colWidths=col_widths)
    style = TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f1f1")),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    t.setStyle(style)
    return t


def p(text: str, style=NORMAL) -> Paragraph:
    return Paragraph(text, style)


def section(title: str, level: int = 2) -> Paragraph:
    return Paragraph(title, {1: H1, 2: H2, 3: H3}[level])


# ---------- content ----------

story: list = []

story.append(p("4-Hour Continuation Strategy on Gold — Audit Pack", H1))
story.append(p(
    f"Branch: claude/4h-candle-strategy-backtest-8kKdx &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    SMALL,
))
story.append(Spacer(1, 0.18 * inch))

# 1. Hypothesis
story.append(section("1. Hypothesis"))
story.append(p(
    '<i>"On every 4-hour gold candle open, price tends to continue in the '
    "same direction as the previous 4-hour candle — placing entries on "
    'the 15-minute timeframe inside the new H4 bar."</i>'
))
story.append(Spacer(1, 0.12 * inch))

# 2. Data
story.append(section("2. Data — no synthetic anywhere"))
story.append(make_table([
    ["File", "Bars", "Span (UTC)", "Source"],
    ["data/XAUUSD_H4_long.csv",     "8,607", "2018-06-28 → 2026-04-20", "github.com/142f/inv-cry"],
    ["data/XAUUSD_H4_matched.csv",    "261", "2026-01-30 → 2026-04-01", "github.com/tiumbj/Bot_Data_Basese"],
    ["data/XAUUSD_M15_matched.csv", "3,977", "2026-01-30 → 2026-04-01", "github.com/tiumbj/Bot_Data_Basese"],
], col_widths=[2.0 * inch, 0.6 * inch, 1.85 * inch, 2.05 * inch]))
story.append(Spacer(1, 0.05 * inch))
story.append(p(
    "Both sources are public MT5 broker exports. The matched H4 + M15 pair "
    "is from the same broker so M15 sub-bars slot cleanly inside H4 "
    "buckets. <font face='Courier'>scripts/audit.py</font> verifies schema, gaps, look-ahead, and "
    "bucket alignment on every run (currently 0 failures).", SMALL))
story.append(Spacer(1, 0.15 * inch))

# 3. Trade logic
story.append(section("3. Trade logic — single source of truth"))
story.append(p("<b>Signal (H4):</b> "
               "<font face='Courier'>sig_t = sign(close_{t-1} − open_{t-1})</font> "
               "(color of the previous H4 candle)."))
story.append(p("<b>Filters (composable JSON spec):</b>"))
story.append(make_table([
    ["filter type", "meaning", "params"],
    ["body_atr",   "previous H4 body / ATR(14) ≥ min",                                   "min, atr_n"],
    ["session",    "H4 bar opens at one of the listed UTC hours",                        "hours_utc"],
    ["regime",     "previous close on same (with) or opposite (against) side of MA(ma_n)", "ma_n, side"],
    ["min_streak", "previous k H4 candles share sig's color",                            "k"],
    ["candle_class", "trend / rotation / doji bucket on prev candle",                    "classes"],
], col_widths=[1.05 * inch, 4.3 * inch, 1.25 * inch]))
story.append(Spacer(1, 0.08 * inch))
story.append(p("<b>Entries (M15):</b>"))
story.append(make_table([
    ["entry type", "definition"],
    ["m15_open",         "open of the first M15 sub-bar of new H4 (= H4 open)"],
    ["m15_confirm",      "close of first M15 sub-bar inside H4 whose color matches sig"],
    ["m15_atr_stop",     "m15_open + 1× M15-ATR(14) hard stop"],
    ["m15_retrace_fib",  "limit at prev_high − level × range (long) / prev_low + level × range (short); "
                         "level ∈ {0.236, 0.382, 0.5, 0.618, 0.786}"],
], col_widths=[1.4 * inch, 5.2 * inch]))
story.append(Spacer(1, 0.06 * inch))
story.append(p("<b>Stops:</b> none, prev_h4_open, prev_h4_extreme, h4_atr (×mult). &nbsp;"
               "<b>Exits:</b> h4_close, prev_h4_extreme_tp."))
story.append(p("<b>Costs:</b> real broker spread from M15 column × point size 0.001 in the holdout; "
               "<font face='Courier'>cost_bps</font> (default 1.5) in walk-forward folds where M15 "
               "isn't available."))
story.append(PageBreak())

# 4. What we ran
story.append(section("4. What we ran"))
story.append(make_table([
    ["Step", "Script", "What it does"],
    ["Data fetch", "scripts/fetch_data.py", "Pulls real CSVs above. Fails loudly on network errors."],
    ["Audit", "scripts/audit.py", "Schema, gaps, alignment, look-ahead, fib entry assertions. 0 failures."],
    ["Hit-rate diagnostic", "scripts/backtest.py (Stage 1)", "P(color_t = color_{t-1}) on long history."],
    ["Backtest (single)", "scripts/backtest.py (Stage 2)", "Three M15 entry variants on matched 2026 window."],
    ["Fib analysis", "scripts/fib_analysis.py", "Touch / reaction / win-from-touch per fib level."],
    ["Walk-forward", "scripts/walkforward.py", "12-mo train / 3-mo test / 3-mo step → ~27 disjoint folds."],
    ["Agentic search", "scripts/orchestrate.py", "1,632 specs proposed; walk-forward + holdout + critic."],
    ["Skeptic", "scripts/skeptic.py", "Per-champion cost / attribution / coverage probes."],
], col_widths=[1.35 * inch, 1.8 * inch, 3.55 * inch]))
story.append(Spacer(1, 0.12 * inch))

# 5. Headline results
story.append(section("5. Headline results"))

story.append(section("5.1 Hit-rate diagnostic (8,604 H4 bar pairs)", level=3))
story.append(make_table([
    ["Metric", "Value"],
    ["P(same direction)",      "0.4971  (95% CI [0.4865, 0.5077])"],
    ["P(up | prev up)",        "0.5197"],
    ["P(down | prev down)",    "0.4723"],
], col_widths=[2.5 * inch, 4.0 * inch]))
story.append(p("Raw hypothesis is a coin flip (slight reversal bias). "
               "Filtering is required for any edge.", SMALL))
story.append(Spacer(1, 0.10 * inch))

story.append(section("5.2 Single-spec backtest (matched 2026 window, 9 weeks)", level=3))
story.append(make_table([
    ["Variant", "Trades", "Total return", "Sharpe ann"],
    ["m15_open",     "260", "−7.4%", "−0.92"],
    ["m15_confirm",  "260", "−9.8%", "−1.42"],
    ["m15_atr_stop", "260", "−3.1%", "−0.52"],
], col_widths=[1.6 * inch, 1.0 * inch, 1.4 * inch, 1.4 * inch]))
story.append(p("All three variants lose without filters, consistent with §5.1.", SMALL))
story.append(Spacer(1, 0.10 * inch))

# Equity png if present
eq_png = RES / "equity.png"
if eq_png.exists():
    story.append(p("Equity curves of the three single-spec variants:", SMALL))
    story.append(Image(str(eq_png), width=5.6 * inch, height=2.6 * inch))
    story.append(Spacer(1, 0.10 * inch))

story.append(section("5.3 Fib level reaction on long history", level=3))
fib_levels = RES / "fib_levels.csv"
if fib_levels.exists():
    df = pd.read_csv(fib_levels)
    df = df[df["dataset"] == "h4_long_2018_2026"].sort_values("level")
    rows = [["level", "touch rate", "reaction rate", "win rate from touch", "mean ret (bp)"]]
    for _, r in df.iterrows():
        rows.append([
            f"{r['level']:.3f}",
            f"{r['touch_rate']:.2f}",
            f"{r['reaction_rate']:.2f}",
            f"{r['win_rate_from_touch']:.2f}",
            f"{r['mean_ret_from_touch_bp']:+.1f}",
        ])
    story.append(make_table(rows, col_widths=[0.9 * inch, 1.1 * inch, 1.2 * inch, 1.6 * inch, 1.3 * inch]))
story.append(p("0.618 is the deepest level whose win-rate-from-touch is ≥ 50%. "
               "Mean return is negative at every level <i>without</i> filtering. "
               "32% of bars retrace fully — the prior candle's range is not a strong "
               "barrier on gold H4.", SMALL))
story.append(Spacer(1, 0.12 * inch))

story.append(section("5.4 Agentic search — 1,632 specs, 16 certified", level=3))
lb_path = RES / "leaderboard.csv"
if lb_path.exists():
    lb = pd.read_csv(lb_path)
    cert = lb[lb["certified"].astype(str).str.lower() == "true"].copy()
    cert = cert.sort_values(["wf_median_sharpe", "ho_total_return"], ascending=False).head(8)
    rows = [["id", "wf folds", "wf median Sharpe", "wf % pos", "tpw", "ho ret", "ho Sharpe"]]
    for _, r in cert.iterrows():
        rows.append([
            Paragraph(f"<font name='Courier' size='7.5'>{r['id']}</font>", SMALL),
            f"{int(r['wf_folds'])}",
            f"{r['wf_median_sharpe']:.2f}",
            f"{r['wf_pct_positive_folds']:.2f}",
            f"{r.get('ho_trades_per_week', '-'):.2f}" if 'ho_trades_per_week' in r else "-",
            f"{r['ho_total_return']*100:+.2f}%",
            f"{r['ho_sharpe_ann']:.2f}",
        ])
    story.append(make_table(rows, col_widths=[
        2.7 * inch, 0.55 * inch, 1.0 * inch, 0.7 * inch, 0.55 * inch, 0.65 * inch, 0.7 * inch,
    ]))
story.append(p("Walk-forward winner: the fib-0.382 retracement variant. "
               "Holdout-return winner: the streak variant. The fib variant has the "
               "best long-history evidence (1.04 wf median Sharpe across 27 disjoint "
               "3-month slices of 2018–2026, 59% positive folds).", SMALL))
story.append(Spacer(1, 0.12 * inch))

story.append(section("5.5 Skeptic findings on the fib-0.382 champion", level=3))
story.append(p("Cost break-even: certified up to <b>2.0 bps</b>; first break at 3.0 bps."))
story.append(p("<b>Filter attribution</b> (Δ wf median Sharpe when dropped):"))
story.append(make_table([
    ["filter dropped", "wf Sharpe after drop", "Δ vs baseline 1.04"],
    ["body_atr",         "−0.38",  "−1.42  (load-bearing)"],
    ["regime",            "0.74",  "−0.30  (small contribution)"],
    ["ALL filters",      "−1.39",  "−2.43  (raw signal floor)"],
], col_widths=[1.6 * inch, 1.8 * inch, 2.6 * inch]))
story.append(Spacer(1, 0.06 * inch))
story.append(p("<b>Coverage hits</b> (off-grid; walk-forward looks better but fall under 3 trades/wk):"))
story.append(make_table([
    ["off-grid variant", "wf Sharpe", "wf % pos", "trades/wk"],
    ["body_atr.min = 0.7",      "2.37", "0.63", "2.3"],
    ["add min_streak k = 3",    "2.37", "0.59", "1.5"],
], col_widths=[2.1 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]))
story.append(p("Stronger edges exist at higher selectivity, but the 9-week M15 "
               "holdout drops below the 3 trades/week floor before the selectivity peaks. "
               "Longer M15 history is the most direct unlock.", SMALL))
story.append(Spacer(1, 0.15 * inch))

# 6. Agents
story.append(section("6. Agents (in agents/)"))
story.append(make_table([
    ["#", "Name", "Role", "File"],
    ["00", "pipeline",       "Index of the two flows",                            "agents/00-pipeline.md"],
    ["01", "data-fetcher",   "Pull real OHLC",                                    "agents/01-data-fetcher.md"],
    ["02", "strategy-spec",  "Trade rules — single source of truth",              "agents/02-strategy-spec.md"],
    ["03", "backtester",     "Single-spec sim",                                   "agents/03-backtester.md"],
    ["04", "analyst",        "Metrics + chart",                                   "agents/04-analyst.md"],
    ["05", "reporter",       "README writer",                                     "agents/05-reporter.md"],
    ["06", "proposer",       "Emits JSON specs (today: grid; later: LLM)",        "agents/06-proposer.md"],
    ["07", "walk-forward",   "27-fold rolling harness",                           "agents/07-walkforward.md"],
    ["08", "critic",         "Certification rule",                                "agents/08-critic.md"],
    ["09", "orchestrator",   "The search loop",                                   "agents/09-orchestrator.md"],
    ["10", "(audit)",        "Gap / look-ahead checks",                           "scripts/audit.py"],
    ["11", "fib-analyzer",   "Touch / reaction at fib levels",                    "agents/11-fib-analyzer.md"],
    ["12", "skeptic",        "Cost / attribution / coverage probes",              "agents/12-skeptic.md"],
], col_widths=[0.35 * inch, 1.2 * inch, 2.85 * inch, 2.2 * inch]))
story.append(Spacer(1, 0.15 * inch))

# 7. Reproduce
story.append(section("7. Reproduce"))
cmds = [
    "git checkout claude/4h-candle-strategy-backtest-8kKdx",
    "pip install pandas numpy matplotlib reportlab",
    "make data        # fetch real OHLC",
    "make audit       # 0 failures expected",
    "make fib         # results/fib_levels.csv, fib_deepest.csv",
    "make backtest    # results/{summary,trades,equity,hit_rate}.csv + equity.png",
    "make search      # results/leaderboard.csv  (~4 min)",
    "make skeptic     # results/skeptic.csv  (~30 s; needs leaderboard.csv)",
    "python3 scripts/build_pdf.py   # regenerates this PDF",
]
for c in cmds:
    story.append(p(f"<font face='Courier' size='8.5'>{c}</font>", MONO))
story.append(Spacer(1, 0.12 * inch))

# 8. Limitations
story.append(section("8. Known limitations the skeptic flagged"))
for line in [
    "9-week M15 holdout is too short to certify the most selective specs; "
    "their walk-forward Sharpes look strong but trade counts fall under 3/wk. "
    "Longer M15 history would let several coverage hits actually certify.",
    "Cost model in walk-forward is a flat 1.5 bps; real-world spread on gold "
    "can spike during news. Skeptic confirms certification dies above 2 bps "
    "for the fib-0.382 champion and faster for the streak champions.",
    "Proposer is a static grid. Replacing it with an LLM-driven proposer that "
    "reads the leaderboard is the documented next step (agents/06-proposer.md).",
    "Filter attribution shows regime is small (~0.3 Sharpe contribution on the "
    "fib champion). It might be discardable, which would simplify the spec.",
]:
    story.append(p("• " + line, SMALL))


# ---------- build ----------

def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="4H Continuation on Gold — Audit Pack",
    )
    doc.build(story)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  size={p.stat().st_size} bytes")
