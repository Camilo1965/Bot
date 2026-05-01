"""Build a single Markdown snapshot for AI/human review (SL timeline, errors, logs)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None  # type: ignore[assignment]


def _tail_text(path: Path, max_lines: int, max_bytes: int = 400_000) -> str:
    if not path.is_file():
        return f"(no existe: {path})\n"
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raw = raw[-max_bytes:]
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _sanitize_env_lines() -> list[str]:
    keys = [
        "EXECUTION_MODE",
        "MT5_SERVER",
        "MT5_LOGIN",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "RUNTIME_METRICS_INTERVAL_S",
        "DIAGNOSTIC_BUNDLE_INTERVAL_S",
        "TELEGRAM_LOG_ALERTS",
        "TELEGRAM_LOG_MIN_LEVEL",
        "BUY_PROB_THRESHOLD",
    ]
    lines: list[str] = []
    for k in keys:
        v = os.environ.get(k, "").strip()
        if not v:
            lines.append(f"- `{k}`: _(vacío)_")
        elif k == "MT5_LOGIN":
            lines.append(f"- `{k}`: `{v[:4]}…` (truncado)")
        else:
            lines.append(f"- `{k}`: `{v}`")
    lines.append("- Secretos (`MT5_PASSWORD`, `DB_PASSWORD`, tokens): **omitidos**")
    return lines


def _parse_runtime_jsonl(path: Path, max_lines: int = 250) -> tuple[str, dict]:
    stats: dict[str, object] = {
        "file_exists": path.is_file(),
        "lines_total": 0,
        "first_ts": None,
        "last_ts": None,
        "symbols_ever_open": set(),
    }
    if not path.is_file():
        return "(no existe runtime_metrics.jsonl — arranca el bot con métricas activas)\n", stats

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stats["lines_total"] = len(lines)
    parsed_tail: list[dict] = []
    for raw in lines[-max_lines:]:
        try:
            parsed_tail.append(json.loads(raw))
        except json.JSONDecodeError:
            parsed_tail.append({"_raw": raw[:200]})

    if parsed_tail and isinstance(parsed_tail[0], dict):
        stats["first_ts"] = parsed_tail[0].get("ts")
    if parsed_tail and isinstance(parsed_tail[-1], dict):
        stats["last_ts"] = parsed_tail[-1].get("ts")

    sym_set: set[str] = set()
    for row in parsed_tail:
        if not isinstance(row, dict):
            continue
        for s in row.get("symbols_open") or []:
            sym_set.add(str(s))
        for p in row.get("positions") or []:
            if isinstance(p, dict) and p.get("symbol"):
                sym_set.add(str(p["symbol"]))
    stats["symbols_ever_open"] = sym_set

    tail_out = "\n".join(lines[-max_lines:]) + "\n"
    return tail_out, stats


def _sl_tp_timeline_from_jsonl(path: Path) -> str:
    if not path.is_file():
        return "_Sin archivo JSONL._\n"
    rows_out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]:
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = row.get("ts", "?")
        for p in row.get("positions") or []:
            if not isinstance(p, dict):
                continue
            sym = p.get("symbol", "?")
            rows_out.append(
                f"| {ts} | {sym} | entry={p.get('entry_price')} | "
                f"peak={p.get('peak_price')} | SL_now={p.get('current_stop_loss')} | "
                f"SL_ini={p.get('initial_stop_price')} | trail={p.get('trailing_stop_active')} | "
                f"tp_hint={p.get('dynamic_tp_hint')} |"
            )
    if not rows_out:
        return "_Sin posiciones en el tramo final del JSONL (o vacío)._\n"
    return "```text\n" + "\n".join(rows_out) + "\n```\n"


def _filter_bot_debug_warnings(path: Path, max_lines: int = 120) -> str:
    if not path.is_file():
        return f"(no existe {path})\n"
    keep: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"level": "ERROR"' in raw or '"level": "WARNING"' in raw:
            keep.append(raw)
        elif "| ERROR " in raw or "| WARNING " in raw:
            keep.append(raw)
    if not keep:
        return "_Sin líneas ERROR/WARNING recientes en formato conocido._\n"
    return "\n".join(keep[-max_lines:]) + "\n"


def write_diagnostic_bundle(
    *,
    repo_root: Path,
    output: Path | None = None,
    load_dotenv_file: bool = True,
) -> Path:
    """Write consolidated Markdown. Default ``output`` = ``repo_root/DIAGNOSTIC_FOR_REVIEW.md``."""
    if load_dotenv_file and _load_dotenv is not None:
        _load_dotenv(repo_root / ".env")

    logs_dir = repo_root / "logs"
    out = output if output is not None else repo_root / "DIAGNOSTIC_FOR_REVIEW.md"
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snap = _read_json(logs_dir / "bot_startup_snapshot.json")

    metrics_tail, mstats = _parse_runtime_jsonl(logs_dir / "runtime_metrics.jsonl", max_lines=250)

    parts: list[str] = []
    parts.append("# ClawdBot — paquete único para revisión (IA / humano)\n")
    parts.append(f"_Generado: `{now}` — repo: `{repo_root.resolve()}`_\n")
    parts.append(
        "## Cómo usar\n"
        "Adjunta o pega **este archivo completo** al asistente.\n\n"
        "**Automático:** el bot puede regenerar este archivo en la raíz cada X minutos "
        "(`DIAGNOSTIC_BUNDLE_INTERVAL_S`). También podés ejecutar "
        "`python scripts/export_diagnostic_bundle.py`.\n\n"
    )

    parts.append("## 1) Resumen rápido\n")
    parts.append(f"- Líneas en `runtime_metrics.jsonl`: **{mstats.get('lines_total', 0)}**\n")
    parts.append(f"- Primera muestra (último archivo): `{mstats.get('first_ts')}`\n")
    parts.append(f"- Última muestra: `{mstats.get('last_ts')}`\n")
    syms = mstats.get("symbols_ever_open")
    if isinstance(syms, set) and syms:
        parts.append(f"- Símbolos con posición en cola final JSONL: `{', '.join(sorted(syms))}`\n")
    parts.append("\n")

    parts.append("## 2) Variables entorno (sin secretos)\n")
    parts.extend([x + "\n" for x in _sanitize_env_lines()])
    parts.append("\n")

    parts.append("## 3) Snapshot de arranque (`logs/bot_startup_snapshot.json`)\n")
    if snap:
        parts.append("```json\n" + json.dumps(snap, indent=2, ensure_ascii=False) + "\n```\n\n")
    else:
        parts.append("_Archivo ausente._\n\n")

    parts.append("## 4) Cronología SL / pico / trailing (desde JSONL)\n")
    parts.append(
        "Si **SL_now** sube cuando **peak** sube y **trailing** pasa a `true`, el ratchet funcionaba.\n\n"
    )
    parts.append(_sl_tp_timeline_from_jsonl(logs_dir / "runtime_metrics.jsonl"))

    parts.append("\n## 5) Últimas líneas de `runtime_metrics.jsonl` (crudo)\n")
    parts.append("```text\n" + metrics_tail + "```\n")

    parts.append("\n## 6) `logs/last_session.log` (últimas 200 líneas)\n")
    parts.append("```text\n" + _tail_text(logs_dir / "last_session.log", 200) + "```\n")

    parts.append("\n## 7) ERROR / WARNING en `bot_debug.log` (filtrado)\n")
    parts.append("```text\n" + _filter_bot_debug_warnings(repo_root / "bot_debug.log") + "```\n")

    parts.append("\n## 8) `audit.log` (últimas 120 líneas)\n")
    parts.append("```text\n" + _tail_text(repo_root / "audit.log", 120) + "```\n")

    parts.append("\n## 9) `logs/trade_journal.csv` (últimas 80 líneas)\n")
    parts.append("```text\n" + _tail_text(logs_dir / "trade_journal.csv", 80) + "```\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out
