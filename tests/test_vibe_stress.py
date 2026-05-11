"""Test the VIBE MCP server for timeouts, validation errors, and stress."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from vibe_mcp.server import create_app

@pytest_asyncio.fixture
async def server():
    """Start the VIBE MCP server in background."""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    port = random.randint(10000, 20000)
    site = web.TCPSite(runner, '127.0.0.1', port)
    await site.start()
    
    yield f"http://127.0.0.1:{port}"
    
    await runner.cleanup()


@pytest.mark.asyncio
async def test_predict_stress_16_features(server):
    """
    Simulates sending a 16-feature vector and 500 OHLCV rows.
    Validates that Pydantic doesn't crash on extra data and that
    the timeout is respected (< 30s) returning a valid JSON.
    """
    server_url = server
    
    # Generate 16 random features
    features = [random.random() for _ in range(16)]

    # Generate 500 rows of garbage OHLCV data
    ohlcv = []
    base_price = 100.0
    for _ in range(500):
        o = base_price + random.uniform(-1, 1)
        h = o + random.uniform(0, 2)
        l = o - random.uniform(0, 2)
        c = (h + l) / 2 + random.uniform(-0.5, 0.5)
        v = random.uniform(100, 1000)
        ohlcv.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
        base_price = c

    payload = {
        "features": features,
        "ohlcv": ohlcv,
        "symbol": "BTC/USDT",
        # Extra garbage to test ConfigDict(extra="allow")
        "some_garbage_field": "should be ignored",
        "nested_garbage": {"a": 1, "b": [1, 2, 3]},
    }

    start = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{server_url}/predict", json=payload) as resp:
            elapsed = time.time() - start
            assert resp.status == 200, await resp.text()
            data = await resp.json()

    assert "decision" in data
    assert "confidence" in data
    
    # Check that decision is one of the expected values
    assert data["decision"] in ("BUY", "SELL", "NEUTRAL")
    assert 0.0 <= float(data["confidence"]) <= 1.0

    # Ensure it responds in < 30 seconds (it should be much faster or hit timeout fallback)
    assert elapsed < 30.0, f"Request took too long: {elapsed}s"


@pytest.mark.asyncio
async def test_predict_invalid_data_returns_neutral(server):
    """Test that completely malformed data doesn't crash the server."""
    server_url = server
    
    async with aiohttp.ClientSession() as session:
        # 1. Invalid JSON
        async with session.post(f"{server_url}/predict", data="Not a JSON") as resp:
            assert resp.status == 400
            data = await resp.json()
            assert data["decision"] == "NEUTRAL"

        # 2. Invalid types (this will hit validation error)
        async with session.post(f"{server_url}/predict", json={"features": "not a list", "symbol": 123}) as resp:
            assert resp.status == 422
            data = await resp.json()
            assert data["decision"] == "NEUTRAL"

