"""Rich mega-dashboard — professional crypto trading terminal aesthetic.

Dense, information-rich layout inspired by Bloomberg/TradingView terminals.
Color language: cyan=system, green=positive, red=negative, yellow=caution,
magenta=AI/vibe, white=data, dim=secondary.
"""

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
from bot.dashboard_helpers import (
    compute_rsi,
    display_timezone,
    mt5_dashboard_mark,
    pct_change_24h_vs_h1,
)
from execution.paper_executor import PaperExecutor, compute_dynamic_tp_hint
from risk.risk_manager import LEVERAGE as _RISK_LEVERAGE
from risk.risk_manager import RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD


# ── Visual helpers ──────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 12, color: str = "cyan") -> str:
    if max_val <= 0:
        return "[dim]" + "\u2591" * width + "[/dim]"
    ratio = max(0.0, min(1.0, value / max_val))
    filled = int(ratio * width)
    return f"[{color}]{'\u2588' * filled}[/{color}][dim]{'\u2591' * (width - filled)}[/dim]"


def _format_duration(total_seconds: float) -> str:
    secs = int(abs(total_seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _pnl_str(val: float) -> tuple[str, str]:
    color = "bright_green" if val >= 0 else "bright_red"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.4f}", color


def _extract_vibe_text(data: Any) -> str:
    try:
        if isinstance(data, dict) and "content" in data:
            parts = []
            for item in data["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
            return " ".join(parts).strip()
        return str(data)
    except Exception:
        return str(data)


_SIG_STYLE = {
    "BUY": "[bold bright_green]\u25b2 BUY[/bold bright_green]",
    "SELL": "[bold bright_red]\u25bc SELL[/bold bright_red]",
    "HOLD": "[dim]\u25cf HOLD[/dim]",
    "WAIT": "[yellow]\u23f8 WAIT[/yellow]",
    "TRAIL": "[bold bright_yellow]\U0001f512 TRAIL[/bold bright_yellow]",
    "CLOSE": "[bold bright_red]\u26a0 CLOSE[/bold bright_red]",
}


def _signal_cell(pos: Any, sym: str, ml_signals_map: dict, ml_probs: dict,
                 global_hold: bool) -> str:
    raw_sig = ml_signals_map.get(sym, "HOLD")
    prob = ml_probs.get(sym, 0.0)
    if pos:
        pending = getattr(pos, "close_pending", False)
        if pending:
            return _SIG_STYLE["CLOSE"]
        if getattr(pos, "trailing_stop_active", False):
            return _SIG_STYLE["TRAIL"]
        return _SIG_STYLE["HOLD"]
    if global_hold:
        return _SIG_STYLE["WAIT"]
    if prob >= BUY_PROB_THRESHOLD and raw_sig == "BUY":
        return _SIG_STYLE["BUY"]
    if raw_sig == "SELL":
        return _SIG_STYLE["SELL"]
    return "[dim]\u25cf[/dim]"


_TREND_TOKEN = {
    "bullish": "[bright_green]B[/bright_green]",
    "bearish": "[bright_red]S[/bright_red]",
    "neutral": "[yellow]N[/yellow]",
}


# ── Dashboard composition ───────────────────────────────────────────────────

def generate_dashboard(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    watchlist: list[str],
) -> Group:
    now_utc = datetime.now(tz=timezone.utc)
    _tz, tz_name = display_timezone()
    now_str = f"{now_utc.astimezone(_tz).strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"
    ml_probs: dict[str, float] = state.get("ml_probs", {})
    ml_signals_map: dict[str, str] = state.get("ml_signals", {})
    global_hold = False

    # ── Gather shared metrics ───────────────────────────────────────────────
    if dash_state.bot_start_time is not None:
        elapsed = now_utc - dash_state.bot_start_time
        uptime_str = _format_duration(elapsed.total_seconds())
    else:
        uptime_str = "\u2014"

    latency_ms: float = state.get("api_latency_ms", 0.0)
    lat_color = "bright_green" if latency_ms < 100 else "yellow" if latency_ms < 500 else "bright_red"
    lat_str = f"{latency_ms:.0f}ms" if latency_ms > 0 else "\u2014"

    broker_positions_any: list[dict[str, Any]] = []
    broker_open_symbols: set[str] = set()
    get_positions = getattr(paper_executor, "get_open_positions", None)
    if callable(get_positions):
        try:
            broker_positions_any = get_positions(include_foreign=True)
            if isinstance(broker_positions_any, list):
                for p in broker_positions_any:
                    bsym = str(p.get("symbol", ""))
                    local_sym = (
                        paper_executor._local_symbol_from_broker(bsym)
                        if hasattr(paper_executor, "_local_symbol_from_broker")
                        else bsym
                    )
                    if local_sym:
                        broker_open_symbols.add(local_sym)
        except Exception:
            broker_positions_any = []
            broker_open_symbols = set()
    n_pos = len(broker_open_symbols) if broker_open_symbols else len(paper_executor.open_positions)
    total_pnl = paper_executor.total_pnl
    wins = dash_state.session_wins
    losses = dash_state.session_losses
    total_trades = wins + losses
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    max_drawdown: float = state.get("max_drawdown", 0.0)
    exp_pct = risk_manager.get_current_risk_exposure() * 100

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

    pending_closes = sum(
        1 for p in paper_executor.open_positions.values() if getattr(p, "close_pending", False)
    )

    if global_hold:
        status_icon = "[bright_red]\u26d4 PAUSA[/bright_red]"
    elif n_pos > 0:
        status_icon = "[bright_cyan]\u26a1 OPERANDO[/bright_cyan]"
    else:
        status_icon = "[bright_green]\u2705 LISTO[/bright_green]"

    pnl_val, pnl_c = _pnl_str(total_pnl)
    dd_val, dd_c = _pnl_str(max_drawdown)
    wp_raw = risk_manager.balance - risk_manager._weekly_start_balance
    wp_val, wp_c = _pnl_str(wp_raw)

    # ── 1. Header status bar ─────────────────────────────────────────────────
    hdr = Table(box=None, show_header=False, expand=True, padding=(0, 2))
    hdr.add_column(ratio=2)
    hdr.add_column(ratio=2)
    hdr.add_column(ratio=2)

    h_left = Text()
    h_left.append("\U0001f916 ClawdBot Pro  ", style="bold cyan")
    h_left.append(f"{now_str}  ", style="dim")
    h_left.append(f"\u23f1 {uptime_str}  ", style="bold white")
    h_left.append(f"\U0001f310 {lat_str}", style=f"bold {lat_color}")

    h_mid = Text()
    h_mid.append(f"\U0001f9e0 XGBoost  ", style="dim")
    h_mid.append(f"{watchlist[0] if watchlist else '\u2014'}  ", style="bold magenta")
    h_mid.append(f"\U0001f4e1 ", style="dim")
    h_mid.append_text(Text.from_markup(status_icon))

    h_right = Text()
    h_right.append(f"\U0001f4c8 PnL ", style="dim")
    h_right.append(f"{pnl_val} {unit}  ", style=f"bold {pnl_c}")
    h_right.append(f"\U0001f3c6 {wins}W/{losses}L", style="dim")
    if total_trades > 0:
        wr_c = "bright_green" if winrate >= 55 else "yellow" if winrate >= 45 else "bright_red"
        h_right.append(f" {winrate:.0f}%", style=f"bold {wr_c}")
    h_right.append(f"  \U0001f4ca {n_pos}/{risk_manager.max_positions}", style="dim")

    hdr.add_row(h_left, h_mid, h_right)
    header_panel = Panel(
        hdr,
        title="[bold cyan]\u26a1 SYSTEM HEALTH[/bold cyan]",
        border_style="cyan",
        box=rich_box.HEAVY,
    )

    # ── 2. Market table ───────────────────────────────────────────────────────
    mkt = Table(
        box=rich_box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_blue",
        expand=True,
        border_style="blue",
        title="[bold bright_blue]\U0001f4ca MERCADO EN VIVO[/bold bright_blue]",
        title_style="bold bright_blue",
        padding=(0, 1),
    )
    mkt.add_column("", width=2)
    mkt.add_column("S\u00edmbolo", style="bold white", no_wrap=True)
    mkt.add_column("Precio", justify="right")
    mkt.add_column("24h", justify="right")
    mkt.add_column("ATR", justify="right")
    mkt.add_column("RSI", justify="center", width=16)
    mkt.add_column("Prob.", justify="center", width=16)
    mkt.add_column("15m/1H/4H", justify="center")
    mkt.add_column("Posici\u00f3n", justify="center")
    mkt.add_column("PnL", justify="right")
    mkt.add_column("Se\u00f1al", justify="center")

    for sym in watchlist:
        prices = list(state["prices"].get(sym, []))
        candle_last: float | None = prices[-1] if prices else None
        price: float | None = mt5_dashboard_mark(state, sym, candle_last)
        price_str = f"{price:,.2f}" if price is not None else "[dim]N/A[/dim]"

        change_str = "[dim]\u2014[/dim]"
        h1_hist = list(state.get("htf_closes", {}).get(sym, {}).get("1h", []) or [])
        if price is not None:
            pct24 = pct_change_24h_vs_h1(h1_hist, float(price))
            if pct24 is not None:
                ch_c = "bright_green" if pct24 >= 0 else "bright_red"
                s = "+" if pct24 >= 0 else ""
                change_str = f"[{ch_c}]{s}{pct24:.1f}%[/{ch_c}]"

        atr_val: float | None = state.get("atrs", {}).get(sym)
        atr_str = f"{atr_val:.1f}" if atr_val is not None else "[dim]\u2014[/dim]"

        rsi_series = list(state.get("dashboard_rsi_closes", {}).get(sym, []) or [])
        rsi_val = compute_rsi(rsi_series) if len(rsi_series) >= 15 else None
        if rsi_val is not None:
            r_c = "bright_red" if rsi_val >= 70 else "bright_green" if rsi_val <= 30 else "white"
            r_bc = "red" if rsi_val >= 70 else "green" if rsi_val <= 30 else "cyan"
            rsi_str = f"[{r_c}]{rsi_val:.0f}[/{r_c}] {_bar(rsi_val, 100, 8, r_bc)}"
        else:
            rsi_str = "[dim]\u2014[/dim]"

        prob = ml_probs.get(sym, 0.0)
        prob_pct = prob * 100
        th_pct = BUY_PROB_THRESHOLD * 100
        ml_color = "bright_green" if prob_pct >= th_pct else "yellow" if prob_pct >= th_pct - 10 else "dim"
        bar_c = "green" if prob_pct >= th_pct else "yellow" if prob_pct >= th_pct - 10 else "white"
        ml_str = f"[{ml_color}]{prob_pct:.0f}%[/{ml_color}] {_bar(prob_pct, 100, 8, bar_c)}"

        htf = state.get("htf_trend", {}).get(sym, {})
        trend_str = "/".join(
            _TREND_TOKEN.get(htf.get(k, "neutral"), "[dim]?[/dim]")
            for k in ("15m", "1h", "4h")
        )

        pos = paper_executor.open_positions.get(sym)
        row_style = ""
        status_dot = "[dim]\u25cb[/dim]"
        if pos and price is not None:
            qty = pos.position_size / pos.entry_price
            unrealized = (price - pos.entry_price) * qty
            pnl_c2 = "bright_green" if unrealized >= 0 else "bright_red"
            pos_str = f"[bright_cyan]LONG {pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str_r = f"[{pnl_c2}]{unrealized:+.2f}[/{pnl_c2}]"
            row_style = "on #0a2a1a" if unrealized >= 0 else "on #2a0a0a"
            status_dot = "[bright_cyan]\u25cf[/bright_cyan]"
        elif pos:
            pos_str = f"[bright_cyan]LONG {pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str_r = "[dim]\u2014[/dim]"
            row_style = "on #0a1a2a"
            status_dot = "[bright_cyan]\u25cf[/bright_cyan]"
        else:
            pos_str = "[dim]\u2014[/dim]"
            pnl_str_r = "[dim]\u2014[/dim]"

        sig_str = _signal_cell(pos, sym, ml_signals_map, ml_probs, global_hold)

        mkt.add_row(
            status_dot, sym.split("/")[0],
            price_str, change_str, atr_str, rsi_str,
            ml_str, trend_str, pos_str, pnl_str_r, sig_str,
            style=row_style,
        )

    # ── 3. Positions detail ──────────────────────────────────────────────────
    ops = Table(
        box=rich_box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
        expand=True,
        padding=(0, 1),
    )
    ops.add_column("S\u00edmbolo", style="bold")
    ops.add_column("Entrada", justify="right")
    ops.add_column("Actual", justify="right")
    ops.add_column("SL Activo", justify="right")
    ops.add_column("TP Hint", justify="right")
    ops.add_column("Pico", justify="right")
    ops.add_column("PnL %", justify="right")
    ops.add_column("Trailing", justify="center")
    ops.add_column("Duraci\u00f3n", justify="right")

    if paper_executor.open_positions or broker_positions_any:
        shown = 0
        for sym, pos in list(paper_executor.open_positions.items()):
            if shown >= 6:
                break
            px_buf = state.get("prices", {}).get(sym, [])
            candle_m = float(px_buf[-1]) if px_buf else pos.entry_price
            mark = mt5_dashboard_mark(state, sym, candle_m) or pos.entry_price
            pnl_pct = (mark - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0.0
            pnl_color = "bright_green" if pnl_pct >= 0 else "bright_red"

            tp_hint = compute_dynamic_tp_hint(pos)
            tp_str = f"{tp_hint:,.2f}" if tp_hint else "[dim]\u2014[/dim]"
            trail = "[bright_yellow]\U0001f512 ON[/bright_yellow]" if pos.trailing_stop_active else "[dim]OFF[/dim]"
            dur = _format_duration((now_utc - pos.entry_time).total_seconds())

            ops.add_row(
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
            shown += 1
        if shown < 6:
            for p in broker_positions_any:
                if shown >= 6:
                    break
                bsym = str(p.get("symbol", ""))
                local_sym = (
                    paper_executor._local_symbol_from_broker(bsym)
                    if hasattr(paper_executor, "_local_symbol_from_broker")
                    else bsym
                )
                if local_sym in paper_executor.open_positions:
                    continue
                entry = float(p.get("price_open", 0.0) or 0.0)
                current = float(p.get("price_current", 0.0) or 0.0)
                profit = float(p.get("profit", 0.0) or 0.0)
                slb = float(p.get("sl", 0.0) or 0.0)
                tpb = float(p.get("tp", 0.0) or 0.0)
                pnl_color = "bright_green" if profit >= 0 else "bright_red"
                ops.add_row(
                    str(local_sym).split("/")[0],
                    f"{entry:,.2f}" if entry > 0 else "\u2014",
                    f"{current:,.2f}" if current > 0 else "\u2014",
                    f"{slb:,.2f}" if slb > 0 else "\u2014",
                    f"{tpb:,.2f}" if tpb > 0 else "\u2014",
                    "[dim]\u2014[/dim]",
                    f"[{pnl_color}]{profit:+.2f}[/{pnl_color}]",
                    "[dim]MANUAL[/dim]",
                    "[dim]broker[/dim]",
                )
                shown += 1
    else:
        ops.add_row("[dim]Sin posiciones abiertas[/dim]", *[""] * 8)

    ops_panel = Panel(
        ops,
        title="[bold magenta]\U0001f4be POSICIONES ACTIVAS[/bold magenta]",
        border_style="magenta",
        box=rich_box.ROUNDED,
    )

    # ── 4. Vibe-Trading AI panel ──────────────────────────────────────────────
    vibe_client = state.get("vibe_client")
    vibe_ok = vibe_client is not None and vibe_client.available

    vibe_rows = Table(box=rich_box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
    vibe_rows.add_column(width=16, style="bold")
    vibe_rows.add_column()

    # Status row
    if vibe_ok:
        vibe_rows.add_row("\U0001f7e2 MCP", "[bold bright_green]Conectado \u2713[/bold bright_green]")
    else:
        err = getattr(vibe_client, "last_error", None) if vibe_client else None
        err_short = (err or "").split("\n")[0][:40] if err else ""
        vibe_rows.add_row("\U0001f534 MCP", f"[bright_red]Desconectado[/bright_red] [dim]{err_short}[/dim]")

    # Patterns
    vibe_patterns: dict = state.get("vibe_patterns", {})
    if vibe_patterns:
        for sym, pdata in vibe_patterns.items():
            txt = _extract_vibe_text(pdata)
            short = txt[:100].replace("\n", " ").strip()
            vibe_rows.add_row(f"\U0001f50d {sym}", f"[white]{short}[/white]")
    else:
        vibe_rows.add_row("\U0001f50d Patterns", "[dim]Esperando datos\u2026[/dim]")

    # Journal
    vibe_journal = state.get("vibe_journal_analysis")
    if vibe_journal:
        txt = _extract_vibe_text(vibe_journal)
        short = txt[:100].replace("\n", " ").strip()
        vibe_rows.add_row("\U0001f4cb Journal", f"[white]{short}[/white]")
    else:
        vibe_rows.add_row("\U0001f4cb Journal", "[dim]Diario sin analizar[/dim]")

    # Backtest
    vibe_backtest: dict = state.get("vibe_backtest", {})
    if vibe_backtest:
        for sym, bdata in vibe_backtest.items():
            txt = _extract_vibe_text(bdata)
            short = txt[:100].replace("\n", " ").strip()
            vibe_rows.add_row(f"\U0001f4ca Backtest {sym}", f"[white]{short}[/white]")
    else:
        vibe_rows.add_row("\U0001f4ca Backtest", "[dim]Semanal\u2026[/dim]")

    # Factors
    vibe_factors: dict = state.get("vibe_factors", {})
    if vibe_factors:
        for sym, fdata in vibe_factors.items():
            txt = _extract_vibe_text(fdata)
            short = txt[:100].replace("\n", " ").strip()
            vibe_rows.add_row(f"\U0001f9ec Factors {sym}", f"[white]{short}[/white]")
    else:
        vibe_rows.add_row("\U0001f9ec Factors", "[dim]Mensual\u2026[/dim]")

    # Shadow account
    vibe_shadow = state.get("vibe_shadow_report")
    if vibe_shadow:
        txt = _extract_vibe_text(vibe_shadow)
        short = txt[:100].replace("\n", " ").strip()
        vibe_rows.add_row("\U0001f319 Shadow", f"[white]{short}[/white]")
    else:
        vibe_rows.add_row("\U0001f319 Shadow", "[dim]Semanal\u2026[/dim]")

    vibe_panel = Panel(
        vibe_rows,
        title="[bold magenta]\U0001f52c VIBE-TRADING AI[/bold magenta]",
        border_style="magenta",
        box=rich_box.ROUNDED,
    )

    # ── 5. Risk & Wallet panel ────────────────────────────────────────────────
    margin_color = "bright_green" if margin_free > equity * 0.3 else "yellow" if margin_free > 0 else "bright_red"
    exp_color = "bright_red" if exp_pct > 8 else "yellow" if exp_pct > 5 else "bright_green"

    risk = Text()
    risk.append(f"\U0001f4c5 PnL Semanal:  ", style="dim")
    risk.append(f"{wp_val} {unit}\n", style=f"bold {wp_c}")
    risk.append(f"\U0001f6e1 Riesgo Port.:  ", style="dim")
    risk.append(f"{exp_pct:.1f}%\n", style=f"bold {exp_color}")
    risk.append("\u2500" * 34 + "\n", style="dim")
    risk.append(f"\U0001f4b0 Balance:       ", style="dim")
    risk.append(f"{balance:>10,.2f} {unit}\n", style="bold white")
    if use_mt5:
        risk.append(f"\U0001f3db Patrimonio:   ", style="dim")
        risk.append(f"{equity:>10,.2f} {unit}\n", style="bold white")
    risk.append(f"\U0001f4ca Margen libre:  ", style="dim")
    risk.append(f"{margin_free:>10,.2f} {unit}\n", style=f"bold {margin_color}")
    risk.append(f"\U0001f4d0 Margen usado:   ", style="dim")
    risk.append(f"{margin_used:>10,.2f} {unit}\n", style="dim")
    risk.append("\u2500" * 34 + "\n", style="dim")
    risk.append(f"\U0001f4c8 PnL Sesi\u00f3n:   ", style="dim")
    risk.append(f"{pnl_val} {unit}\n", style=f"bold {pnl_c}")
    risk.append(f"\U0001f4c9 Max Drawdown:  ", style="dim")
    risk.append(f"{dd_val} {unit}\n", style=f"bold {dd_c}")
    risk.append("\u2500" * 34 + "\n", style="dim")
    risk.append(f"\U0001f504 Posiciones:    ", style="dim")
    pos_color = "bright_red" if n_pos >= risk_manager.max_positions else "bold white"
    risk.append(f"{n_pos}/{risk_manager.max_positions}\n", style=pos_color)
    risk.append(f"\U0001f3c6 Win/Loss:      ", style="dim")
    risk.append(f"{wins}W {losses}L", style="bold white")
    if total_trades > 0:
        wr_color = "bright_green" if winrate >= 55 else "yellow" if winrate >= 45 else "bright_red"
        risk.append(f" ({winrate:.0f}%)", style=f"bold {wr_color}")
    risk.append("\n")
    if pending_closes > 0:
        risk.append(f"\u26a0 Cierres pend.:   ", style="dim")
        risk.append(f"{pending_closes}", style="bold bright_red")

    risk_panel = Panel(
        risk,
        title="[bold yellow]\U0001f4bc RIESGO & WALLET[/bold yellow]",
        border_style="yellow",
        box=rich_box.ROUNDED,
    )

    # ── 6. Events log ─────────────────────────────────────────────────────────
    events_text = Text()
    events = list(dash_state.dashboard_events)
    if events:
        for evt in events[-8:]:
            events_text.append_text(Text.from_markup(evt + "\n"))
    else:
        events_text.append_text(Text.from_markup("[dim]Sin eventos recientes\u2026[/dim]"))

    events_panel = Panel(
        events_text,
        title="[bold green]\U0001f4cb \u00daLTIMOS EVENTOS[/bold green]",
        border_style="green",
        box=rich_box.ROUNDED,
    )

    # ── Compose ───────────────────────────────────────────────────────────────
    footer = Columns([risk_panel, events_panel], equal=True)

    return Group(header_panel, mkt, ops_panel, vibe_panel, footer)