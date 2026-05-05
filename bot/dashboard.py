"""Rich mega-dashboard layout (market table, risk panel, positions, events)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich import box as rich_box
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bot import state as dash_state
from bot.dashboard_helpers import compute_rsi, mt5_dashboard_mark
from execution.paper_executor import PaperExecutor, compute_dynamic_tp_hint
from risk.risk_manager import LEVERAGE as _RISK_LEVERAGE
from risk.risk_manager import RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD, BUY_SENTIMENT_THRESHOLD


def _bar(value: float, max_val: float, width: int = 12, color: str = "cyan") -> str:
    """Render a mini progress bar as Rich markup."""
    if max_val <= 0:
        return "[dim]" + "░" * width + "[/dim]"
    ratio = max(0.0, min(1.0, value / max_val))
    filled = int(ratio * width)
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"


def _format_duration(total_seconds: float) -> str:
    secs = int(abs(total_seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _pnl_str(val: float) -> tuple[str, str]:
    """Return (formatted_value, color) for a PnL number."""
    color = "bright_green" if val >= 0 else "bright_red"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.4f}", color


def generate_dashboard(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    watchlist: list[str],
) -> Group:
    """Build and return a Rich mega-dashboard."""
    now_utc = datetime.now(tz=timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    sentiment: float | None = state.get("sentiment")
    ml_probs: dict[str, float] = state.get("ml_probs", {})
    news_hold_until: datetime | None = state.get("news_hold_until")
    global_hold = (
        (sentiment is not None and sentiment < BUY_SENTIMENT_THRESHOLD)
        or (news_hold_until is not None and now_utc < news_hold_until)
    )

    # ── Header Panel ──────────────────────────────────────────────────────────
    if dash_state.bot_start_time is not None:
        elapsed = now_utc - dash_state.bot_start_time
        uptime_str = _format_duration(elapsed.total_seconds())
    else:
        uptime_str = "—"

    latency_ms: float = state.get("api_latency_ms", 0.0)
    lat_color = "bright_green" if latency_ms < 100 else "yellow" if latency_ms < 500 else "bright_red"
    lat_str = f"{latency_ms:.0f}ms" if latency_ms > 0 else "—"

    sentiment_val = sentiment if sentiment is not None else 0.0
    sentiment_str = f"{sentiment_val:.4f}" if sentiment is not None else "—"
    s_color = "bright_green" if sentiment_val >= 0.55 else "yellow" if sentiment_val >= 0.35 else "bright_red"

    n_pos = len(paper_executor.open_positions)
    total_pnl = paper_executor.total_pnl
    wins = dash_state.session_wins
    losses = dash_state.session_losses
    total_trades = wins + losses
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    # Status indicator
    if global_hold:
        status_icon = "[bright_red]⛔ PAUSA[/bright_red]"
    elif n_pos > 0:
        status_icon = "[bright_cyan]⚡ OPERANDO[/bright_cyan]"
    else:
        status_icon = "[bright_green]✅ LISTO[/bright_green]"

    header = Table(box=None, show_header=False, expand=True, padding=(0, 1))
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_column(ratio=1)

    col1 = Text()
    col1.append("🤖 ClawdBot Pro\n", style="bold cyan")
    col1.append(f"   {now_str}\n", style="dim")
    col1.append("   ⏱ ", style="dim")
    col1.append(uptime_str, style="bold white")
    col1.append("  🌐 ", style="dim")
    col1.append(lat_str, style=f"bold {lat_color}")

    col2 = Text()
    col2.append("🔮 Sentimiento: ", style="dim")
    col2.append(sentiment_str, style=f"bold {s_color}")
    col2.append("\n")
    col2.append("🧠 Modelo: ", style="dim")
    col2.append("XGBoost + Gemini\n", style="bold magenta")
    col2.append("📡 Estado: ", style="dim")
    col2.append_text(Text.from_markup(status_icon))

    col3 = Text()
    pnl_s, pnl_c = _pnl_str(total_pnl)
    col3.append("📈 PnL Sesión: ", style="dim")
    col3.append(pnl_s, style=f"bold {pnl_c}")
    col3.append("\n")
    col3.append(f"🏆 W/L: {wins}/{losses}", style="dim")
    if total_trades > 0:
        wr_color = "bright_green" if winrate >= 55 else "yellow" if winrate >= 45 else "bright_red"
        col3.append(f"  ({winrate:.0f}%)", style=f"bold {wr_color}")
    col3.append("\n")
    col3.append(f"📊 Posiciones: {n_pos}/{risk_manager.max_positions}", style="dim")

    header.add_row(col1, col2, col3)

    header_panel = Panel(
        header,
        title="[bold cyan]⚡ SYSTEM HEALTH[/bold cyan]",
        border_style="cyan",
        box=rich_box.HEAVY,
    )

    # ── Market Data Table ─────────────────────────────────────────────────────
    table = Table(
        box=rich_box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_blue",
        expand=True,
        border_style="blue",
        title="[bold bright_blue]📊 MERCADO EN VIVO[/bold bright_blue]",
        title_style="bold bright_blue",
        padding=(0, 1),
    )
    table.add_column("", width=2)
    table.add_column("Símbolo", style="bold white", no_wrap=True)
    table.add_column("Precio", justify="right")
    table.add_column("24h", justify="right")
    table.add_column("ATR", justify="right")
    table.add_column("RSI", justify="center", width=16)
    table.add_column("Prob.", justify="center", width=16)
    table.add_column("15m/1H/4H", justify="center")
    table.add_column("Posición", justify="center")
    table.add_column("PnL", justify="right")
    table.add_column("Ejec.", justify="center")

    _trend_token = {
        "bullish": "[bright_green]B[/bright_green]",
        "bearish": "[bright_red]S[/bright_red]",
        "neutral": "[yellow]N[/yellow]",
    }

    ml_signals_map: dict[str, str] = state.get("ml_signals", {})

    for sym in watchlist:
        prices = list(state["prices"].get(sym, []))
        candle_last: float | None = prices[-1] if prices else None
        price: float | None = mt5_dashboard_mark(state, sym, candle_last)
        price_str = f"{price:,.2f}" if price is not None else "[dim]N/A[/dim]"

        # 24h % change
        change_str = "[dim]—[/dim]"
        if price is not None and len(prices) >= 96:
            price_24h = prices[-96]
            if price_24h > 0:
                pct = (price - price_24h) / price_24h * 100
                ch_color = "bright_green" if pct >= 0 else "bright_red"
                sign = "+" if pct >= 0 else ""
                change_str = f"[{ch_color}]{sign}{pct:.1f}%[/{ch_color}]"

        # ATR
        atr_val: float | None = state.get("atrs", {}).get(sym)
        atr_str = f"{atr_val:.1f}" if atr_val is not None else "[dim]—[/dim]"

        # RSI with bar
        rsi_val = compute_rsi(prices) if len(prices) >= 15 else None
        if rsi_val is not None:
            rsi_color = "bright_red" if rsi_val >= 70 else "bright_green" if rsi_val <= 30 else "white"
            rsi_bar_color = "red" if rsi_val >= 70 else "green" if rsi_val <= 30 else "cyan"
            rsi_str = f"[{rsi_color}]{rsi_val:.0f}[/{rsi_color}] {_bar(rsi_val, 100, 8, rsi_bar_color)}"
        else:
            rsi_str = "[dim]—[/dim]"

        # ML confidence with bar
        prob = ml_probs.get(sym, 0.0)
        prob_pct = prob * 100
        if prob_pct >= 62:
            ml_color = "bright_green"
        elif prob_pct >= 50:
            ml_color = "yellow"
        else:
            ml_color = "dim"
        ml_str = f"[{ml_color}]{prob_pct:.0f}%[/{ml_color}] {_bar(prob_pct, 100, 8, 'green' if prob_pct >= 62 else 'yellow' if prob_pct >= 50 else 'white')}"

        # HTF trend tags (15m/1h/4h): B=bullish, S=bearish, N=neutral.
        htf = state.get("htf_trend", {}).get(sym, {})
        t15 = htf.get("15m", "neutral")
        t1h = htf.get("1h", "neutral")
        t4h = htf.get("4h", "neutral")
        trend_str = (
            f"{_trend_token.get(t15, '[dim]?[/dim]')}/"
            f"{_trend_token.get(t1h, '[dim]?[/dim]')}/"
            f"{_trend_token.get(t4h, '[dim]?[/dim]')}"
        )

        # Position
        pos = paper_executor.open_positions.get(sym)
        row_style = ""
        status_dot = "[dim]○[/dim]"
        if pos and price is not None:
            qty = pos.position_size / pos.entry_price
            unrealized = (price - pos.entry_price) * qty
            pnl_color = "bright_green" if unrealized >= 0 else "bright_red"
            pos_str = f"[bright_cyan]LONG {pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str_r = f"[{pnl_color}]{unrealized:+.2f}[/{pnl_color}]"
            row_style = "on #0a2a1a" if unrealized >= 0 else "on #2a0a0a"
            status_dot = "[bright_cyan]●[/bright_cyan]"
        elif pos:
            pos_str = f"[bright_cyan]LONG {pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str_r = "[dim]—[/dim]"
            row_style = "on #0a1a2a"
            status_dot = "[bright_cyan]●[/bright_cyan]"
        else:
            pos_str = "[dim]—[/dim]"
            pnl_str_r = "[dim]—[/dim]"

        # Executable signal: same gates as try_open_trade path (strategy + global pause)
        raw_sig = ml_signals_map.get(sym, "HOLD")
        can_enter = (
            not global_hold
            and prob >= BUY_PROB_THRESHOLD
            and raw_sig == "BUY"
        )
        if pos:
            pending = getattr(pos, "close_pending", False)
            if pending:
                signal_str = "[bold bright_red]⚠ CLOSE[/bold bright_red]"
            elif getattr(pos, "trailing_stop_active", False):
                signal_str = "[bold bright_yellow]🔒 TRAIL[/bold bright_yellow]"
            else:
                signal_str = "[bold bright_cyan]📊 HOLD[/bold bright_cyan]"
        elif global_hold:
            signal_str = "[yellow]⏸ WAIT[/yellow]"
        elif can_enter:
            signal_str = "[bold bright_green]🚀 BUY[/bold bright_green]"
        elif raw_sig == "SELL":
            signal_str = "[bold bright_red]▼ SELL[/bold bright_red]"
        else:
            signal_str = "[dim]WAIT[/dim]"

        table.add_row(
            status_dot, sym.split("/")[0],
            price_str, change_str, atr_str, rsi_str,
            ml_str, trend_str, pos_str, pnl_str_r, signal_str,
            style=row_style,
        )

    # ── Open Positions Detail Panel ───────────────────────────────────────────
    ops_table = Table(box=rich_box.SIMPLE, show_header=True, header_style="bold magenta", expand=True, padding=(0, 1))
    ops_table.add_column("Símbolo", style="bold")
    ops_table.add_column("Entrada", justify="right")
    ops_table.add_column("Actual", justify="right")
    ops_table.add_column("SL Activo", justify="right")
    ops_table.add_column("TP Hint", justify="right")
    ops_table.add_column("Pico", justify="right")
    ops_table.add_column("PnL %", justify="right")
    ops_table.add_column("Trailing", justify="center")
    ops_table.add_column("Duración", justify="right")

    if paper_executor.open_positions:
        for sym, pos in list(paper_executor.open_positions.items())[:6]:
            px_buf = state.get("prices", {}).get(sym, [])
            candle_m = float(px_buf[-1]) if px_buf else pos.entry_price
            mark = mt5_dashboard_mark(state, sym, candle_m) or pos.entry_price
            pnl_pct = (mark - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
            pnl_color = "bright_green" if pnl_pct >= 0 else "bright_red"

            tp_hint = compute_dynamic_tp_hint(pos)
            tp_str = f"{tp_hint:,.2f}" if tp_hint else "[dim]—[/dim]"
            trail = "[bright_yellow]🔒 ON[/bright_yellow]" if pos.trailing_stop_active else "[dim]OFF[/dim]"
            dur = _format_duration((now_utc - pos.entry_time).total_seconds())

            ops_table.add_row(
                sym.split("/")[0],
                f"{pos.entry_price:,.2f}",
                f"{mark:,.2f}",
                f"[yellow]{pos.current_stop_loss:,.2f}[/yellow]",
                tp_str,
                f"[bright_cyan]{pos.peak_price:,.2f}[/bright_cyan]",
                f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]",
                trail,
                f"[dim]{dur}[/dim]",
            )
    else:
        ops_table.add_row("[dim]Sin posiciones abiertas[/dim]", *[""] * 8)

    ops_panel = Panel(
        ops_table,
        title="[bold magenta]🧾 POSICIONES ACTIVAS[/bold magenta]",
        border_style="magenta",
        box=rich_box.ROUNDED,
    )

    # ── Risk & Wallet Panel ───────────────────────────────────────────────────
    max_drawdown: float = state.get("max_drawdown", 0.0)
    pending_closes = sum(
        1 for p in paper_executor.open_positions.values() if getattr(p, "close_pending", False)
    )

    mt5_wallet = state.get("mt5_wallet")
    use_mt5 = isinstance(mt5_wallet, dict) and all(k in mt5_wallet for k in ("balance", "equity", "margin_free"))

    if use_mt5:
        balance = float(mt5_wallet["balance"])
        equity = float(mt5_wallet["equity"])
        margin_free = float(mt5_wallet["margin_free"])
        margin_used = float(mt5_wallet.get("margin", 0.0))
        unit = str(mt5_wallet.get("currency", "USD")).strip() or "USD"
    else:
        balance = risk_manager.balance
        equity = balance
        margin_used = sum(p.position_size / _RISK_LEVERAGE for p in paper_executor.open_positions.values())
        margin_free = balance - margin_used
        unit = "USDT"

    margin_color = "bright_green" if margin_free > equity * 0.3 else "yellow" if margin_free > 0 else "bright_red"
    dd_color = "bright_red" if max_drawdown < -(equity * 0.05) else "yellow" if max_drawdown < 0 else "bright_green"
    pnl_val, pnl_c = _pnl_str(total_pnl)

    risk_text = Text()
    risk_text.append("💰 Balance:        ", style="dim")
    risk_text.append(f"{balance:>10,.2f} {unit}\n", style="bold white")
    if use_mt5:
        risk_text.append("🏛 Patrimonio:     ", style="dim")
        risk_text.append(f"{equity:>10,.2f} {unit}\n", style="bold white")
    risk_text.append("📊 Margen libre:   ", style="dim")
    risk_text.append(f"{margin_free:>10,.2f} {unit}\n", style=f"bold {margin_color}")
    risk_text.append("📐 Margen usado:   ", style="dim")
    risk_text.append(f"{margin_used:>10,.2f} {unit}\n", style="dim")
    risk_text.append("─" * 32 + "\n", style="dim")
    risk_text.append("📈 PnL Sesión:     ", style="dim")
    risk_text.append(f"{pnl_val:>10} {unit}\n", style=f"bold {pnl_c}")
    risk_text.append("📉 Max Drawdown:   ", style="dim")
    dd_s, _ = _pnl_str(max_drawdown)
    risk_text.append(f"{dd_s:>10} {unit}\n", style=f"bold {dd_color}")
    risk_text.append("─" * 32 + "\n", style="dim")
    risk_text.append("🔄 Posiciones:     ", style="dim")
    pos_color = "bright_red" if n_pos >= risk_manager.max_positions else "bold white"
    risk_text.append(f"{n_pos}/{risk_manager.max_positions}\n", style=pos_color)
    risk_text.append("🏆 Win/Loss:       ", style="dim")
    risk_text.append(f"{wins}W {losses}L", style="bold white")
    if total_trades > 0:
        wr_color = "bright_green" if winrate >= 55 else "yellow" if winrate >= 45 else "bright_red"
        risk_text.append(f" ({winrate:.0f}%)", style=f"bold {wr_color}")
    risk_text.append("\n")
    if pending_closes > 0:
        risk_text.append("⚠ Cierres pend.:   ", style="dim")
        risk_text.append(f"{pending_closes}", style="bold bright_red")

    risk_panel = Panel(
        risk_text,
        title="[bold yellow]💼 RIESGO & WALLET[/bold yellow]",
        border_style="yellow",
        box=rich_box.ROUNDED,
    )

    # ── Events Log Panel ──────────────────────────────────────────────────────
    events_text = Text()
    events = list(dash_state.dashboard_events)
    if events:
        for evt in events[-6:]:
            events_text.append_text(Text.from_markup(evt + "\n"))
    else:
        events_text.append_text(Text.from_markup("[dim]Sin eventos recientes…[/dim]"))

    events_panel = Panel(
        events_text,
        title="[bold green]📋 ÚLTIMOS EVENTOS[/bold green]",
        border_style="green",
        box=rich_box.ROUNDED,
    )

    footer = Columns([risk_panel, events_panel], equal=True)

    return Group(header_panel, table, ops_panel, footer)
