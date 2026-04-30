# ClawdBot — backlog (post-remediation)

Items from the external review that are **intentionally deferred** until product
priority warrants the extra complexity (ops surface, deployment, or strategy
work).  They are not blockers for the core paper/MT5 loop documented in
`README.md`.

| Theme | Direction when picked up |
|-------|---------------------------|
| **Event bus** | Thin async wrapper over the existing `asyncio` queue(s), or a small pub/sub lib; migrate call sites gradually. |
| **Redis** | Optional cache for order-book snapshots or GUI-facing state—only after measuring real need. |
| **GUI realtime** | WebSocket pub/sub, or mmap + lighter polling, after stable event contract. |
| **Short selling** | New position model plus `MT5Executor` sell-to-open semantics; separate project from long-only exits. |
| **Live Sharpe / profit factor** | Session accumulation in `PaperExecutor` / DB-backed metrics. |
| **LightGBM / walk-forward** | Offline ML research track; keep separate from runtime thresholds until validated. |
| **Telegram bidirectional** | HTTP polling or webhook bot commands alongside alerts. |

Prioritise these against business goals (latency, AUM, exchange mix) before
scheduling implementation.
