"""
bot.task_supervisor
~~~~~~~~~~~~~~~~~~~

Supervisor de tareas async robusto para ClawdBot.

Problemas que resuelve:
1. Una tarea no crítica (Vibe, reports) que crashea no debe matar todo el bot.
2. Las tareas fire-and-forget sin referencia pierden excepciones.
3. No hay visibilidad de qué tareas están vivas, muertas, o reiniciándose.

Uso:
    supervisor = TaskSupervisor()
    supervisor.spawn("signal_emitter", signal_emitter(...), critical=True)
    supervisor.spawn("vibe_swarm", crypto_desk_swarm_loop(...), critical=False)
    await supervisor.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("clawdbot.supervisor")


@dataclass
class TaskInfo:
    name: str
    coro_factory: Callable[[], Awaitable[Any]]
    critical: bool = False
    restart: bool = True
    restart_delay_s: float = 5.0
    max_restarts: int = 10
    _task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _restart_count: int = field(default=0, repr=False)
    _last_error: str | None = field(default=None, repr=False)
    _started_at: float = field(default_factory=time.time, repr=False)


class TaskSupervisor:
    """Mantiene tareas async vivas, reinicia las no críticas, y expone métricas."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskInfo] = {}
        self._shutdown: bool = False
        self._metrics: dict[str, dict[str, Any]] = {}

    def spawn(
        self,
        name: str,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        critical: bool = False,
        restart: bool = True,
        restart_delay_s: float = 5.0,
        max_restarts: int = 10,
    ) -> None:
        """Registra una tarea para supervisión.

        *coro_factory* debe ser una función sin argumentos que retorne una corutina.
        Se usa factory en lugar de corutina cruda para poder reiniciar.
        """
        if name in self._tasks:
            raise ValueError(f"Task {name} already registered")
        self._tasks[name] = TaskInfo(
            name=name,
            coro_factory=coro_factory,
            critical=critical,
            restart=restart and not critical,
            restart_delay_s=restart_delay_s,
            max_restarts=max_restarts,
        )
        logger.info("[SUPERVISOR] Registered task %s (critical=%s)", name, critical)

    def _start(self, info: TaskInfo) -> asyncio.Task[Any]:
        """Crea y arranca el asyncio.Task para *info*."""

        async def _wrapped() -> Any:
            info._started_at = time.time()
            try:
                return await info.coro_factory()
            except asyncio.CancelledError:
                raise  # No swallow cancellations
            except Exception as exc:
                info._last_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "[SUPERVISOR] Task %s crashed: %s",
                    info.name,
                    info._last_error,
                )
                raise

        task = asyncio.create_task(_wrapped(), name=info.name)
        info._task = task
        return task

    async def run_forever(self) -> None:
        """Arranca todas las tareas registradas y entra en loop de supervisión.

        Si una tarea *crítica* termina o crashea, se cancela todo y se propaga
        la excepción. Si es *no crítica* y tiene *restart=True*, se reinicia
        después de *restart_delay_s* (hasta *max_restarts*).
        """
        # Start all tasks
        pending: set[asyncio.Task[Any]] = set()
        for info in self._tasks.values():
            pending.add(self._start(info))

        while pending and not self._shutdown:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                name = task.get_name()
                info = self._tasks.get(name)
                if info is None:
                    continue

                exc = task.exception()
                if exc is not None and isinstance(exc, asyncio.CancelledError):
                    # Normal shutdown path
                    continue

                # Update metrics
                self._metrics[name] = {
                    "status": "crashed" if exc else "finished",
                    "last_error": info._last_error,
                    "uptime_s": time.time() - info._started_at,
                    "restarts": info._restart_count,
                }

                if info.critical:
                    logger.critical(
                        "[SUPERVISOR] Critical task %s died. Shutting down.",
                        name,
                    )
                    if exc:
                        raise exc
                    raise RuntimeError(f"Critical task {name} finished unexpectedly")

                if info.restart and info._restart_count < info.max_restarts:
                    info._restart_count += 1
                    logger.warning(
                        "[SUPERVISOR] Restarting task %s in %.0fs (attempt %d/%d)",
                        name,
                        info.restart_delay_s,
                        info._restart_count,
                        info.max_restarts,
                    )
                    await asyncio.sleep(info.restart_delay_s)
                    pending.add(self._start(info))
                else:
                    logger.error(
                        "[SUPERVISOR] Task %s terminated and will NOT be restarted.",
                        name,
                    )

        # Normal exit: cancel anything still pending
        for task in pending:
            if not task.done():
                task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def shutdown(self) -> None:
        """Cancela todas las tareas supervisadas de forma graceful."""
        self._shutdown = True
        for info in self._tasks.values():
            if info._task and not info._task.done():
                info._task.cancel()
        # Wait a bit for graceful cancellation
        await asyncio.sleep(0.5)
        # Gather any stragglers
        pending = [
            info._task for info in self._tasks.values()
            if info._task and not info._task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def health(self) -> dict[str, dict[str, Any]]:
        """Retorna métricas de salud de cada tarea."""
        result: dict[str, dict[str, Any]] = {}
        for name, info in self._tasks.items():
            task = info._task
            result[name] = {
                "alive": task is not None and not task.done(),
                "critical": info.critical,
                "restarts": info._restart_count,
                "last_error": info._last_error,
                "uptime_s": (
                    time.time() - info._started_at
                    if task and not task.done()
                    else self._metrics.get(name, {}).get("uptime_s", 0.0)
                ),
            }
        return result
