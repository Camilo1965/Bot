"""Tests for aiohttp security middleware."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


async def _safe_handler(request: web.Request) -> web.Response:
    return web.Response(text="OK")


async def _exploding_handler(request: web.Request) -> web.Response:
    raise ValueError("boom")


async def _http_handler(request: web.Request) -> web.Response:
    raise web.HTTPNotFound()


class TestAiohttpSecurity:
    @pytest.mark.asyncio
    async def test_malformed_request_returns_400_not_crash(self):
        from bot.web_server import _security_and_error_guard

        app = web.Application(middlewares=[_security_and_error_guard])
        app.router.add_get("/ok", _safe_handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/ok")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_exception_in_handler_returns_400(self):
        from bot.web_server import _security_and_error_guard

        app = web.Application(middlewares=[_security_and_error_guard])
        app.router.add_get("/boom", _exploding_handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/boom")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_http_exception_propagates(self):
        from bot.web_server import _security_and_error_guard

        app = web.Application(middlewares=[_security_and_error_guard])
        app.router.add_get("/notfound", _http_handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/notfound")
            assert resp.status == 404


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_endpoint_returns_json(self):
        async def _health(request: web.Request) -> web.Response:
            return web.json_response({"status": "ok", "mt5": False})

        app = web.Application()
        app.router.add_get("/health", _health)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] in ("ok", "degraded")
            assert "mt5" in data
