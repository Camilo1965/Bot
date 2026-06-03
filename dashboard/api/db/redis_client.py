"""Redis client with optional in-memory backend for Docker-less VPS.

REDIS_BACKEND env (default "redis"):
  redis  → connect to REDIS_URL (real Redis server)
  memory → use fakeredis in-process (zero deps, single-process only)

The in-memory backend supports the subset we use: GET/SET/EX, MGET,
KEYS, SADD/SMEMBERS, PUBLISH/SUBSCRIBE. Survives only within the
process lifetime — fine for self-contained VPS install where bot + API
run in one supervisor.
"""

from __future__ import annotations

import os

_redis = None


async def get_redis():
    global _redis
    if _redis is not None:
        return _redis

    backend = os.environ.get("REDIS_BACKEND", "redis").strip().lower()
    if backend == "memory":
        import fakeredis.aioredis as fakeredis_aio
        _redis = fakeredis_aio.FakeRedis()
    else:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    return _redis
