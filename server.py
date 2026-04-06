import asyncio
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import config
from core.environment import discover_environment
from core.orchestrator import MultiAgentOrchestrator

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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _loop, _env_snapshot
    _loop = asyncio.get_event_loop()
    _env_snapshot = discover_environment()
    print(f"\n[{config.AI_NAME}] Ready on http://localhost:{config.SERVER_PORT}\n")
    yield


app = FastAPI(title="Agent Nexus", lifespan=lifespan)


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path("ui/index.html")
    return p.read_text(encoding="utf-8") if p.exists() else "<h1>UI not found</h1>"


@app.get("/api/config")
async def get_config():
    return {
        "ai_name": config.AI_NAME,
        "has_api_key": bool(config.NVIDIA_API_KEY),
        "model": config.MODEL,
        "max_iterations": config.MAX_ITERATIONS,
    }


@app.post("/api/setup")
async def setup(body: dict):
    ai_name = body.get("ai_name", "JARVIS").strip() or "JARVIS"
    api_key = body.get("api_key", "").strip()
    model = body.get("model", "mistralai/mistral-small-4-119b-2603").strip()

    if not api_key:
        return JSONResponse({"error": "api_key required"}, status_code=400)

    # Read existing .env values to preserve other settings
    env_path = Path(".env")
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, v = stripped.split("=", 1)
                existing[k.strip()] = v.strip()

    # Update / add the keys we care about
    existing["AI_NAME"] = ai_name
    existing["NVIDIA_API_KEY"] = api_key
    existing["MODEL"] = model

    # Ensure defaults exist
    for k, v in {
        "BASE_URL": "https://integrate.api.nvidia.com/v1/chat/completions",
        "MAX_ITERATIONS": "30",
        "MAX_TOKENS": "4096",
        "LLM_TEMPERATURE": "0.2",
        "SERVER_HOST": "0.0.0.0",
        "SERVER_PORT": str(config.SERVER_PORT),
    }.items():
        existing.setdefault(k, v)

    env_text = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    env_path.write_text(env_text, encoding="utf-8")

    # Reload module-level config
    config.reload_config()

    return {"status": "ok", "ai_name": config.AI_NAME, "model": config.MODEL}


@app.post("/api/run")
async def run(body: dict):
    global _run_thread
    if not config.NVIDIA_API_KEY:
        return JSONResponse({"error": "API key not configured — complete setup first"}, status_code=400)
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
            webbrowser.open(f"http://localhost:{config.SERVER_PORT}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level="warning")
