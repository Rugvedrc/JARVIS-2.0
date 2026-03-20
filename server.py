import asyncio
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from config import SERVER_HOST, SERVER_PORT
from core.environment import discover_environment
from core.orchestrator import MultiAgentOrchestrator

app = FastAPI(title="Agent Nexus")

# ── WebSocket client registry ─────────────────────────────────────────────────
_clients: list[asyncio.Queue] = []
_clients_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


def broadcast(event: dict):
    if _loop and _loop.is_running():
        with _clients_lock:
            snapshot = list(_clients)
        for q in snapshot:
            try:
                _loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass


# ── Orchestrator ──────────────────────────────────────────────────────────────
orchestrator = MultiAgentOrchestrator(event_callback=broadcast)
_run_thread: threading.Thread | None = None
_env_snapshot: str = ""


@app.on_event("startup")
async def _startup():
    global _loop, _env_snapshot
    _loop = asyncio.get_event_loop()
    _env_snapshot = discover_environment()
    print(f"\n[Agent Nexus] Ready → http://localhost:{SERVER_PORT}\n")


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path("ui/index.html")
    return p.read_text() if p.exists() else "<h1>UI not found</h1>"


@app.post("/api/run")
async def run(body: dict):
    global _run_thread
    goal = body.get("goal", "").strip()
    if not goal:
        return JSONResponse({"error": "goal required"}, status_code=400)
    if _run_thread and _run_thread.is_alive():
        return JSONResponse({"error": "already running"}, status_code=409)

    def _go():
        orchestrator.run(goal, _env_snapshot)

    _run_thread = threading.Thread(target=_go, daemon=True)
    _run_thread.start()
    return {"status": "started", "goal": goal}


@app.post("/api/stop")
async def stop():
    orchestrator.stop()
    return {"status": "stopped"}


@app.get("/api/status")
async def status():
    running = bool(_run_thread and _run_thread.is_alive())
    with orchestrator.lock:
        agents = {
            n: {"status": a.status, "action_count": a.action_count, "current_action": a.current_action}
            for n, a in orchestrator.agents.items()
        }
    return {
        "running": running,
        "stats": orchestrator.stats,
        "agents": agents,
        "learnings": orchestrator.learnings,
    }


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue()
    with _clients_lock:
        _clients.append(q)

    # Send current state on connect
    running = bool(_run_thread and _run_thread.is_alive())
    with orchestrator.lock:
        agents_snapshot = {
            n: {"status": a.status, "action_count": a.action_count}
            for n, a in orchestrator.agents.items()
        }
    await ws.send_json({
        "type": "connected",
        "running": running,
        "agents": agents_snapshot,
        "learnings": list(orchestrator.learnings),
    })

    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=25.0)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        with _clients_lock:
            if q in _clients:
                _clients.remove(q)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    no_browser = "--no-browser" in sys.argv

    if not no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{SERVER_PORT}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
