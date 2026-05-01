import asyncio
import logging
from typing import Any

from aiohttp import web
from rich.console import Console

from bot.dashboard import generate_dashboard
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

async def start_web_dashboard(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    watchlist: list[str],
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """Run an aiohttp web server to export the Rich dashboard as HTML."""
    
    async def handle_dashboard(request: web.Request) -> web.Response:
        try:
            # We create a dummy console that forces terminal formatting to record HTML
            console = Console(record=True, width=120, force_terminal=True)
            dashboard = generate_dashboard(state, paper_executor, risk_manager, watchlist)
            console.print(dashboard)
            
            # Export to HTML with custom theme
            html = console.export_html(
                inline_styles=True,
                code_format="<pre style=\"font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace; background-color:#111; color:#eee; padding:10px\">{code}</pre>"
            )
            
            # Auto-refresh every 2 seconds
            refresh_script = "<script>setTimeout(function(){location.reload()}, 2000);</script>"
            final_html = html.replace("</body>", f"{refresh_script}</body>")
            
            return web.Response(text=final_html, content_type="text/html")
        except Exception as e:
            logger.error("Web Dashboard Error: %s", e)
            return web.Response(text="Internal Server Error", status=500)

    app = web.Application()
    app.add_routes([web.get("/", handle_dashboard)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    logger.info("🌐 Web dashboard server listening on http://%s:%d", host, port)
    await site.start()
    
    # Keeps the task alive
    while True:
        await asyncio.sleep(3600)
