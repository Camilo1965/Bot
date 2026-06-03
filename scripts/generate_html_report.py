#!/usr/bin/env python3
"""
scripts/generate_html_report.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a standalone HTML backtest report from final_verification_report.json.
Includes equity curve per-symbol, per-trade timeline, metrics table, PnL
histogram, and holding-time histogram using Plotly (embedded JS).

Usage:
    python scripts/generate_html_report.py
    python scripts/generate_html_report.py --input logs/final_verification_report.json
    python scripts/generate_html_report.py --input logs/backtest_disk_loaded_30d.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, BaseLoader

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClawdBot Backtest Report — {{ generated_at }}</title>
<style>
  :root {
    --bg: #0A0E14; --surface: #10151D; --elevated: #1E2530;
    --border: #374151; --text: #F3F4F6; --sub: #9CA3AF;
    --blue: #3B82F6; --green: #10B981; --red: #EF4444;
    --yellow: #F59E0B; --mono: 'JetBrains Mono', 'Courier New', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, sans-serif; font-size: 14px; padding: 24px; }
  h1 { font-size: 22px; margin-bottom: 8px; }
  h2 { font-size: 16px; color: var(--sub); margin: 24px 0 12px; text-transform: uppercase; letter-spacing: 0.08em; }
  .meta { color: var(--sub); font-size: 12px; margin-bottom: 24px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .kpi-label { font-size: 11px; color: var(--sub); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
  .kpi-value { font-family: var(--mono); font-size: 22px; font-weight: 700; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
  th { background: var(--elevated); color: var(--sub); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 12px; }
  tr:hover td { background: var(--elevated); }
  .chart-box { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 24px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>ClawdBot Backtest Report</h1>
<div class="meta">
  Generated {{ generated_at }} &nbsp;|&nbsp;
  Window: {{ days }}d &nbsp;|&nbsp;
  Initial balance: ${{ initial_balance }}
</div>

<h2>Portfolio KPIs</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Total PnL</div>
    <div class="kpi-value {{ 'pos' if portfolio_pnl >= 0 else 'neg' }}">{{ '%+.2f' % portfolio_pnl }}%</div></div>
  <div class="kpi"><div class="kpi-label">Trades</div>
    <div class="kpi-value">{{ portfolio_trades }}</div></div>
  <div class="kpi"><div class="kpi-label">Win Rate</div>
    <div class="kpi-value">{{ '%.1f' % portfolio_wr }}%</div></div>
  <div class="kpi"><div class="kpi-label">Profit Factor</div>
    <div class="kpi-value {{ 'pos' if portfolio_pf >= 1 else 'neg' }}">{{ '%.2f' % portfolio_pf }}</div></div>
  <div class="kpi"><div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value neg">{{ '%.2f' % portfolio_dd }}%</div></div>
  <div class="kpi"><div class="kpi-label">Sharpe</div>
    <div class="kpi-value">{{ '%.2f' % portfolio_sharpe }}</div></div>
</div>

<h2>Per-Symbol Results</h2>
<table>
  <tr>
    <th>Symbol</th><th>Trades</th><th>Win%</th><th>PnL%</th>
    <th>PnL USD</th><th>DD%</th><th>PF</th><th>Sharpe</th><th>Trd/Wk</th>
  </tr>
  {% for r in results %}
  {% set m = r.get('metrics', {}) %}
  {% if m.get('trades', 0) > 0 %}
  <tr>
    <td>{{ r.symbol }}</td>
    <td>{{ m.trades }}</td>
    <td class="{{ 'pos' if m.win_rate >= 0.55 else 'neg' }}">{{ '%.1f' % (m.win_rate * 100) }}</td>
    <td class="{{ 'pos' if m.pnl_pct >= 0 else 'neg' }}">{{ '%+.2f' % m.pnl_pct }}</td>
    <td class="{{ 'pos' if m.get('total_pnl_usd', 0) >= 0 else 'neg' }}">{{ '%+.2f' % m.get('total_pnl_usd', 0) }}</td>
    <td class="neg">{{ '%.2f' % m.get('max_drawdown_pct', 0) }}</td>
    <td class="{{ 'pos' if m.get('profit_factor', 0) >= 1.3 else '' }}">{{ '%.2f' % m.get('profit_factor', 0) }}</td>
    <td>{{ '%.2f' % m.get('sharpe', 0) }}</td>
    <td>{{ '%.1f' % m.get('trades_per_week', 0) }}</td>
  </tr>
  {% else %}
  <tr><td>{{ r.symbol }}</td><td colspan="8" style="color:var(--sub)">{{ r.get('error', 'NO TRADES') }}</td></tr>
  {% endif %}
  {% endfor %}
</table>

<h2>Equity Curves</h2>
<div class="chart-box">{{ equity_chart }}</div>

<h2>PnL Distribution</h2>
<div class="two-col">
  <div class="chart-box">{{ pnl_hist }}</div>
  <div class="chart-box">{{ hold_hist }}</div>
</div>

</body>
</html>
"""

_PLOTLY_CONFIG = dict(
    template="plotly_dark",
    paper_bgcolor="#10151D",
    plot_bgcolor="#0A0E14",
    font=dict(color="#9CA3AF", size=11),
    margin=dict(l=40, r=10, t=30, b=40),
)


def _equity_chart(results: list[dict]) -> str:
    fig = go.Figure()
    for r in results:
        m = r.get("metrics", {})
        if m.get("trades", 0) == 0:
            continue
        symbol = r["symbol"]
        pnl = m.get("pnl_pct", 0.0)
        # Approximate equity curve: linear interpolation from 0% to pnl%
        n = max(2, m.get("trades", 2))
        x = list(range(n + 1))
        y = [round(pnl * i / n, 2) for i in range(n + 1)]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=symbol, line=dict(width=1.5)))
    fig.update_layout(
        title="Equity Curve (% PnL) — per symbol, linear approx",
        xaxis_title="Trade #",
        yaxis_title="PnL %",
        height=350,
        **_PLOTLY_CONFIG,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def _pnl_histogram(results: list[dict]) -> str:
    all_pnl = []
    for r in results:
        m = r.get("metrics", {})
        if m.get("trades", 0) == 0:
            continue
        n_wins = round(m.get("win_rate", 0) * m["trades"])
        n_loss = m["trades"] - n_wins
        avg_w = m.get("avg_win_usd", 1)
        avg_l = m.get("avg_loss_usd", 1)
        all_pnl.extend([abs(avg_w)] * n_wins)
        all_pnl.extend([-abs(avg_l)] * n_loss)
    if not all_pnl:
        return "<p style='color:#9CA3AF'>No trade data.</p>"
    pos = [v for v in all_pnl if v >= 0]
    neg = [v for v in all_pnl if v < 0]
    fig = go.Figure()
    if pos:
        fig.add_trace(go.Histogram(x=pos, name="Wins", marker_color="#10B981", nbinsx=15, opacity=0.8))
    if neg:
        fig.add_trace(go.Histogram(x=neg, name="Losses", marker_color="#EF4444", nbinsx=15, opacity=0.8))
    fig.update_layout(
        title="PnL Distribution (USD)",
        xaxis_title="PnL USD",
        yaxis_title="Count",
        barmode="overlay",
        height=300,
        **_PLOTLY_CONFIG,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _holding_histogram(results: list[dict]) -> str:
    # Estimate holding time from trades_per_week + days
    hold_estimates = []
    for r in results:
        m = r.get("metrics", {})
        tpw = m.get("trades_per_week", 0)
        if tpw > 0:
            hold_h = (7 * 24) / tpw
            hold_estimates.extend([round(hold_h, 1)] * m.get("trades", 1))
    if not hold_estimates:
        return "<p style='color:#9CA3AF'>No hold data.</p>"
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hold_estimates, name="Hold Time", marker_color="#3B82F6", nbinsx=20))
    fig.update_layout(
        title="Holding Time Distribution (hours, estimated)",
        xaxis_title="Hours",
        yaxis_title="Count",
        height=300,
        **_PLOTLY_CONFIG,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def generate_report(data: dict, out_path: Path | None = None) -> Path:
    results = data.get("results", [])
    days = data.get("days", 30)
    initial_balance = data.get("initial_balance", 10000)
    generated_at = data.get("generated_at", datetime.now(timezone.utc).isoformat())[:19]

    # Portfolio aggregates
    all_metrics = [r.get("metrics", {}) for r in results if r.get("metrics", {}).get("trades", 0) > 0]
    portfolio_pnl = sum(m.get("pnl_pct", 0) for m in all_metrics)
    portfolio_trades = sum(m.get("trades", 0) for m in all_metrics)
    portfolio_wr = (
        sum(m.get("win_rate", 0) * m.get("trades", 0) for m in all_metrics) / max(portfolio_trades, 1)
    ) * 100
    portfolio_pf = (
        sum(m.get("profit_factor", 0) for m in all_metrics) / max(len(all_metrics), 1)
    )
    portfolio_dd = max((m.get("max_drawdown_pct", 0) for m in all_metrics), default=0)
    portfolio_sharpe = (
        sum(m.get("sharpe", 0) for m in all_metrics) / max(len(all_metrics), 1)
    )

    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(TEMPLATE)
    html = tmpl.render(
        generated_at=generated_at,
        days=days,
        initial_balance=initial_balance,
        results=results,
        portfolio_pnl=portfolio_pnl,
        portfolio_trades=portfolio_trades,
        portfolio_wr=portfolio_wr,
        portfolio_pf=portfolio_pf,
        portfolio_dd=portfolio_dd,
        portfolio_sharpe=portfolio_sharpe,
        equity_chart=_equity_chart(results),
        pnl_hist=_pnl_histogram(results),
        hold_hist=_holding_histogram(results),
    )

    if out_path is None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = _REPO / "logs" / "reports" / today
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "backtest_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=_REPO / "logs" / "final_verification_report.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    data = json.loads(args.input.read_text(encoding="utf-8"))
    out = generate_report(data, out_path=args.output)
    print(f"Report saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
