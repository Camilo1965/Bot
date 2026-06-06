"""
GET /api/models                          — list all models in models/ dir
GET /api/models/{sym}/feature-importance — from {SYM}_v2.meta.json
GET /api/models/{sym}/drift-test         — from {SYM}_prob_ref.json
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_MODELS_DIR = _REPO / "models"

# Filename helpers — order matters: prefer v3 over v2.
_LONG_MODEL_VARIANTS = ["v3", "v2"]
_SHORT_MODEL_VARIANTS = ["short_v3", "short_v2"]


def _read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _active_symbol_set() -> set[str]:
    """Symbols currently enabled in SYMBOL_CONFIG (matches Signals page)."""
    try:
        sys.path.insert(0, str(_REPO))
        from scripts.backtest_full_bot import SYMBOL_CONFIG
        return {
            sym.replace("/", "_").upper()
            for sym, cfg in SYMBOL_CONFIG.items()
            if cfg.get("prob_threshold", 1.0) < 0.95
        }
    except Exception:
        return set()


_MODEL_FILE_RE = re.compile(
    r"^(?P<sym>[A-Z0-9]+(?:_[A-Z0-9]+)+?)_"
    r"(?P<side>short_)?"
    r"(?P<ver>v[23])"
    r"(?P<meta>\.meta)?\.json$"
)


def _list_symbol_dirs() -> dict[str, dict]:
    """Group on-disk model files by underlying symbol code (e.g. BTC_USDT).

    Only canonical files like {SYM}_v2.json, {SYM}_v3.meta.json,
    {SYM}_short_v2.json, {SYM}_short_v3.meta.json are picked up.
    Ensemble/regime/calibration/prob_ref artefacts are intentionally skipped.
    """
    if not _MODELS_DIR.is_dir():
        return {}
    by_sym: dict[str, dict] = {}
    for f in _MODELS_DIR.iterdir():
        if not f.is_file() or f.name.endswith(".bak"):
            continue
        m = _MODEL_FILE_RE.match(f.name)
        if not m:
            continue
        sym = m.group("sym")
        side_prefix = "short_" if m.group("side") else ""
        key = f"{side_prefix}{m.group('ver')}" + (".meta" if m.group("meta") else "")
        by_sym.setdefault(sym, {})[key] = f
    return by_sym


def _pick_meta_for(files: dict[str, Path], variants: list[str]) -> tuple[Optional[Path], Optional[dict]]:
    for v in variants:
        meta_key = f"{v}.meta"
        if meta_key in files:
            return files[meta_key], _read_meta(files[meta_key])
    # Fallback: model exists but meta missing
    for v in variants:
        if v in files:
            return files[v], None
    return None, None


def _list_models_sync(active_only: bool = False) -> list[dict]:
    by_sym = _list_symbol_dirs()
    active_set = _active_symbol_set() if active_only else set()

    result: list[dict] = []
    for sym in sorted(by_sym.keys()):
        if active_only and active_set and sym not in active_set:
            continue
        files = by_sym[sym]

        long_path, long_meta = _pick_meta_for(files, _LONG_MODEL_VARIANTS)
        short_path, short_meta = _pick_meta_for(files, _SHORT_MODEL_VARIANTS)

        has_long = long_path is not None
        has_short = short_path is not None

        # Decorate meta with trained_at fallback (file mtime) and feature count
        def _decorate(meta: Optional[dict], path: Optional[Path]) -> Optional[dict]:
            if meta is None and path is None:
                return None
            m = dict(meta or {})
            if path is not None and not m.get("trained_at"):
                from datetime import datetime, timezone
                m["trained_at"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            feats = m.get("features")
            if isinstance(feats, list) and "feature_count" not in m:
                m["feature_count"] = len(feats)
            metrics = m.get("metrics")
            if not isinstance(metrics, dict):
                m["metrics"] = {}
            return m

        result.append({
            "symbol": sym,
            "has_long_model": has_long,
            "has_short_model": has_short,
            "long_meta": _decorate(long_meta, long_path) if has_long else None,
            "short_meta": _decorate(short_meta, short_path) if has_short else None,
            "is_active": sym in (active_set if active_set else _active_symbol_set()),
        })
    return result


def _feature_importance_sync(sym: str) -> dict:
    meta_path = _MODELS_DIR / f"{sym}_v2.meta.json"
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("feature_importance", meta)
    except Exception:
        return {}


def _drift_test_sync(sym: str) -> dict:
    ref_path = _MODELS_DIR / f"{sym}_prob_ref.json"
    if not ref_path.is_file():
        return {"available": False}
    try:
        data = json.loads(ref_path.read_text(encoding="utf-8"))
        return {
            "available": True,
            "count": data.get("count", 0),
            "last_updated": data.get("last_updated"),
            "data": data,
        }
    except Exception:
        return {"available": False}


@router.get("/api/models")
async def list_models(active_only: bool = Query(default=False)) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _list_models_sync, active_only)


@router.get("/api/models/{sym}/feature-importance")
async def get_feature_importance(sym: str) -> dict:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _feature_importance_sync, sym.upper())
    if not result:
        raise HTTPException(status_code=404, detail=f"No meta.json found for {sym}")
    return result


@router.get("/api/models/{sym}/drift-test")
async def get_drift_test(sym: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _drift_test_sync, sym.upper())
