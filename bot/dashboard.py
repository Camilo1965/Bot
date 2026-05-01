"""Rich mega-dashboard layout (market table, risk panel, events)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box as rich_box

from bot import state as dash_state
from bot.dashboard_helpers import compute_rsi, mt5_dashboard_mark
from execution.paper_executor import PaperExecutor
from risk.risk_manager import LEVERAGE as _RISK_LEVERAGE, RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD, BUY_SENTIMENT_THRESHOLD


def generate_dashboard(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    watchlist: list[str],
) -> Group:
    """Build and return a Rich mega-dashboard layout with system health,
    market data, and risk/events panels.

    The returned :class:`~rich.console.Group` stacks three sections:

    1. **Header panel** – uptime, API latency, model names, sentiment.
    2. **Market table** – per-symbol price, 24 h %, ATR volatility, RSI,
       HTF trend, open position, unrealised PnL, and AI action.
       Rows for open positions are highlighted with a neon-green background.
    3. **Footer columns** – risk & wallet panel (balance, available margin,
       session PnL, max drawdown) beside a live-events log.
    """
    now_utc = datetime.now(tz=timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    sentiment: float | None = state.get("sentiment")
    ml_probs: dict[str, float] = state.get("ml_probs", {})
    news_hold_until: datetime | None = state.get("news_hold_until")
    global_hold = (
        (sentiment is not None and sentiment < BUY_SENTIMENT_THRESHOLD)
        or (news_hold_until is not None and now_utc < news_hold_until)
    )

    # ── Header Panel: System Health ──────────────────────────────────────────
    if dash_state.bot_start_time is not None:
        elapsed = now_utc - dash_state.bot_start_time
        total_secs = int(elapsed.total_seconds())
        h, rem = divmod(total_secs, 3600)
        m, s = divmod(rem, 60)
        uptime_str = f"{h:02d}h {m:02d}m {s:02d}s"
    else:
        uptime_str = "—"

    latency_ms: float = state.get("api_latency_ms", 0.0)
    lat_color = "bright_green" if latency_ms < 100 else "yellow" if latency_ms < 500 else "bright_red"
    lat_str = f"{latency_ms:.0f} ms" if latency_ms > 0 else "—"

    sentiment_val_str = f"{sentiment:.4f}" if sentiment is not None else "—"
    s_color = "bright_green" if (sentiment or 0) >= 0.55 else "yellow" if (sentiment or 0) >= 0.35 else "bright_red"
    news_str = "[bright_red]⛔ HOLD[/bright_red]" if global_hold else "[bright_green]✅ OK[/bright_green]"

    header_text = Text(justify="left")
    header_text.append("🤖  ClawdBot  –  Mega Dashboard  |  ", style="bold cyan")
    header_text.append(now_str, style="white")
    header_text.append("\n")
    header_text.append("⏱ Uptime: ", style="dim")
    header_text.append(uptime_str, style="bold white")
    header_text.append("   │   🌐 Latencia API: ", style="dim")
    header_text.append(lat_str, style=f"bold {lat_color}")
    header_text.append("   │   🧠 Modelo: ", style="dim")
    header_text.append("XGBoost_v3 + Gemini-2.5-Flash-Lite", style="bold magenta")
    header_text.append("   │   🔮 Sentimiento IA: ", style="dim")
    header_text.append(sentiment_val_str, style=f"bold {s_color}")
    header_text.append("   │   Noticias: ", style="dim")
    header_text.append_text(Text.from_markup(news_str))

    header_panel = Panel(
        header_text,
        title="[bold cyan]⚡ SYSTEM HEALTH[/bold cyan]",
        border_style="cyan",
        box=rich_box.ROUNDED,
    )

    # ── Market Data Table ────────────────────────────────────────────────────
    table = Table(
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        show_footer=False,
        border_style="bright_blue",
        title="[bold cyan]📊 MARKET DATA[/bold cyan]",
        title_style="bold cyan",
    )
    table.add_column("Símbolo", style="bold white", no_wrap=True)
    table.add_column("Precio", justify="right")
    table.add_column("24h %", justify="right")
    table.add_column("Vol (ATR)", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("ML Conf.", justify="right")
    table.add_column("Tendencia", justify="center")
    table.add_column("Posición", justify="center")
    table.add_column("PnL Pos.", justify="right")
    table.add_column("Acción IA", justify="center")

    _trend_color = {"bullish": "bright_green", "bearish": "bright_red", "neutral": "yellow"}

    for sym in watchlist:
        prices = list(state["prices"].get(sym, []))
        candle_last: float | None = prices[-1] if prices else None
        price: float | None = mt5_dashboard_mark(state, sym, candle_last)
        price_str = f"{price:,.2f}" if price is not None else "[dim]N/A[/dim]"

        # 24 h % change – 96 bars × 15 m = 24 h
        change_24h_str = "[dim]—[/dim]"
        if price is not None and len(prices) >= 96:
            price_24h = prices[-96]
            if price_24h > 0:
                pct = (price - price_24h) / price_24h * 100
                ch_color = "bright_green" if pct >= 0 else "bright_red"
                sign = "+" if pct >= 0 else ""
                change_24h_str = f"[{ch_color}]{sign}{pct:.2f}%[/{ch_color}]"

        # Vol (ATR)
        atr_val: float | None = state.get("atrs", {}).get(sym)
        atr_str = f"{atr_val:.2f}" if atr_val is not None else "[dim]—[/dim]"

        # RSI
        rsi_val = compute_rsi(prices) if len(prices) >= 15 else None
        if rsi_val is not None:
            rsi_color = "bright_red" if rsi_val >= 70 else "bright_green" if rsi_val <= 30 else "white"
            rsi_str = f"[{rsi_color}]{rsi_val:.1f}[/{rsi_color}]"
        else:
            rsi_str = "[dim]—[/dim]"

        # ML probability (kept for Acción IA and ML Conf. column)
        prob = ml_probs.get(sym, 0.0)

        # ML Conf. display
        prob_pct = prob * 100
        if prob_pct > 55:
            ml_conf_str = f"[bold green]{prob_pct:.1f}%[/bold green]"
        elif prob_pct > 50:
            ml_conf_str = f"[bold yellow]{prob_pct:.1f}%[/bold yellow]"
        else:
            ml_conf_str = f"[dim]{prob_pct:.1f}%[/dim]"

        # HTF trend
        htf_trend = state.get("htf_trend", {}).get(sym, {})
        t15 = htf_trend.get("15m", "neutral")
        t1h = htf_trend.get("1h", "neutral")
        t4h = htf_trend.get("4h", "neutral")
        trend_str = (
            f"[{_trend_color.get(t15, 'white')}]{t15[:4]}[/] / "
            f"[{_trend_color.get(t1h, 'white')}]{t1h[:4]}[/] / "
            f"[{_trend_color.get(t4h, 'white')}]{t4h[:4]}[/]"
        )

        # Position & unrealised PnL
        pos = paper_executor.open_positions.get(sym)
        row_style = ""
        if pos and price is not None:
            qty = pos.position_size / pos.entry_price
            unrealized_pnl = (price - pos.entry_price) * qty
            pnl_color = "bright_green" if unrealized_pnl >= 0 else "bright_red"
            pos_str = f"[bright_cyan]LONG @{pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str = f"[{pnl_color}]{unrealized_pnl:+.2f}[/{pnl_color}]"
            row_style = "on dark_green"
        elif pos:
            pos_str = f"[bright_cyan]LONG @{pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str = "[dim]N/A[/dim]"
            row_style = "on dark_green"
        else:
            pos_str = "[dim]—[/dim]"
            pnl_str = "[dim]—[/dim]"

        # Acción IA
        if pos:
            accion_str = "[bright_blue]TRADING 🔵[/bright_blue]"
        elif prob >= BUY_PROB_THRESHOLD:
            accion_str = "[bright_green]BUY! 🟢[/bright_green]"
        else:
            accion_str = "[bright_yellow]HOLD 🟡[/bright_yellow]"

        table.add_row(
            sym.split("/")[0],
            price_str,
            change_24h_str,
            atr_str,
            rsi_str,
            ml_conf_str,
            trend_str,
            pos_str,
            pnl_str,
            accion_str,
            style=row_style,
        )

    # ── Footer: Risk & Wallet + Events Log ──────────────────────────────────
    n_pos = len(paper_executor.open_positions)
    pending_closes = sum(
        1 for p in paper_executor.open_positions.values() if getattr(p, "close_pending", False)
    )
    total_pnl = paper_executor.total_pnl
    max_drawdown: float = state.get("max_drawdown", 0.0)

    mt5_wallet = state.get("mt5_wallet")
    use_mt5_wallet = (
        isinstance(mt5_wallet, dict)
        and all(k in mt5_wallet for k in ("balance", "equity", "margin_free"))
    )
    acct_unit = "USDT"
    if use_mt5_wallet:
        c = mt5_wallet.get("currency")
        if isinstance(c, str) and c.strip():
            acct_unit = c.strip()

    if use_mt5_wallet:
        balance = float(mt5_wallet["balance"])
        equity = float(mt5_wallet["equity"])
        available_margin = float(mt5_wallet["margin_free"])
        margin_used = float(mt5_wallet.get("margin", 0.0))
        ref_liquidity = equity if equity > 1e-9 else balance
        margin_color = (
            "bright_green" if available_margin > ref_liquidity * 0.2
            else "yellow" if available_margin > 0
            else "bright_red"
        )
        dd_color = (
            "bright_red" if max_drawdown < -(ref_liquidity * 0.05)
            else "yellow" if max_drawdown < 0
            else "bright_green"
        )
    else:
        balance = risk_manager.balance
        equity = balance
        used_margin = sum(
            pos.position_size / _RISK_LEVERAGE for pos in paper_executor.open_positions.values()
        )
        available_margin = balance - used_margin
        margin_color = (
            "bright_green" if available_margin > balance * 0.2
            else "yellow" if available_margin > 0
            else "bright_red"
        )
        dd_color = (
            "bright_red" if max_drawdown < -(balance * 0.05)
            else "yellow" if max_drawdown < 0
            else "bright_green"
        )
        margin_used = 0.0

    pnl_color_r = "bright_green" if total_pnl >= 0 else "bright_red"

    risk_text = Text()
    if use_mt5_wallet:
        risk_text.append("💰 Balance (MT5):     ", style="dim")
        risk_text.append(f"{balance:>12,.2f} {acct_unit}\n", style="bold white")
        risk_text.append("🏛 Patrimonio:        ", style="dim")
        risk_text.append(f"{equity:>12,.2f} {acct_unit}\n", style="bold white")
        risk_text.append("📊 Margen libre:     ", style="dim")
        risk_text.append(f"{available_margin:>12,.2f} {acct_unit}\n", style=f"bold {margin_color}")
        risk_text.append("📐 Margen usado:     ", style="dim")
        risk_text.append(f"{margin_used:>12,.2f} {acct_unit}\n", style="dim")
    else:
        risk_text.append("💰 Balance:           ", style="dim")
        risk_text.append(f"{balance:>12,.2f} USDT\n", style="bold white")
        risk_text.append("📊 Margen Disponible: ", style="dim")
        risk_text.append(f"{available_margin:>12,.2f} USDT\n", style=f"bold {margin_color}")
    risk_text.append("📈 PnL Sesión:        ", style="dim")
    risk_text.append(f"{total_pnl:>+12.4f} USDT\n", style=f"bold {pnl_color_r}")
    risk_text.append("📉 Max Drawdown:      ", style="dim")
    risk_text.append(f"{max_drawdown:>+12.4f} USDT\n", style=f"bold {dd_color}")
    risk_text.append("🔄 Posiciones:        ", style="dim")
    risk_text.append(f"{n_pos}/{risk_manager.max_positions}\n", style="bold white")
    risk_text.append("⏳ Cierres pendientes:", style="dim")
    pending_style = "bold bright_red" if pending_closes > 0 else "bold bright_green"
    risk_text.append(f"{pending_closes}", style=pending_style)

    risk_panel = Panel(
        risk_text,
        title="[bold yellow]💼 RIESGO & WALLET[/bold yellow]",
        border_style="yellow",
        box=rich_box.ROUNDED,
    )

    # Events log panel
    events_text = Text()
    events = list(dash_state.dashboard_events)
    if events:
        for evt in events[-3:]:
            events_text.append_text(Text.from_markup(evt + "\n"))
    else:
        events_text.append_text(Text.from_markup("[dim]Sin eventos recientes…[/dim]"))

    events_panel = Panel(
        events_text,
        title="[bold green]📋 ÚLTIMOS EVENTOS[/bold green]",
        border_style="green",
        box=rich_box.ROUNDED,
    )

    # Operations panel (live positions and pending-close status)
    ops_text = Text()
    if paper_executor.open_positions:
        for sym, pos in list(paper_executor.open_positions.items())[:4]:
            px_buf = state.get("prices", {}).get(sym, [])
            candle_m = float(px_buf[-1]) if px_buf else pos.entry_price
            mark = mt5_dashboard_mark(state, sym, candle_m) or pos.entry_price
            qty = pos.position_size / pos.entry_price if pos.entry_price > 0 else 0.0
            unrl = (mark - pos.entry_price) * qty
            pnl_color = "bright_green" if unrl >= 0 else "bright_red"
            pending = getattr(pos, "close_pending", False)
            status = "[red]PENDING_CLOSE[/red]" if pending else "[green]OPEN[/green]"
            ops_text.append_text(
                Text.from_markup(
                    f"{sym:<10} {status}  PnL [{pnl_color}]{unrl:+.2f}[/{pnl_color}]\n"
                )
            )
    else:
        ops_text.append_text(Text.from_markup("[dim]Sin posiciones abiertas[/dim]"))

    ops_panel = Panel(
        ops_text,
        title="[bold magenta]🧾 OPERACIONES EN VIVO[/bold magenta]",
        border_style="magenta",
        box=rich_box.ROUNDED,
    )

    footer = Columns([risk_panel, ops_panel, events_panel], equal=True)

    return Group(header_panel, table, footer)
