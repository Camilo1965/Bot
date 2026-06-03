"""
GET /api/models                          — list all models in models/ dir
GET /api/models/{sym}/feature-importance — from {SYM}_v2.meta.json
GET /api/models/{sym}/drift-test         — from {SYM}_prob_ref.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_MODELS_DIR = _REPO / "models"


def _list_models_sync() -> list[dict]:
    if not _MODELS_DIR.is_dir():
        return []

    symbols: dict[str, dict] = {}
    for f in _MODELS_DIR.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if name.endswith("_v2.json") and not name.endswith(".bak"):
            sym = name[: -len("_v2.json")]
            symbols.setdefault(sym, {})["has_long_model"] = True
        if name.endswith("_v2.meta.json"):
            sym = name[: -len("_v2.meta.json")]
            entry = symbols.setdefault(sym, {})
            try:
                entry["meta"] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                entry["meta"] = {}
        # Treat regime model as short model
        if name.endswith("_regime.json"):
            sym = name[: -len("_regime.json")]
            symbols.setdefault(sym, {})["has_short_model"] = True

    result = []
    for sym, data in sorted(symbols.items()):
        result.append({
            "symbol": sym,
            "has_long_model": data.get("has_long_model", False),
            "has_short_model": data.get("has_short_model", False),
            "meta": data.get("meta", {}),
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
async def list_models() -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _list_models_sync)


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
