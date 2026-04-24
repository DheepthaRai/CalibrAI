import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from ..services.calibration_service import start_calibration, get_run_state, run_queues
from ..services.llm_service import check_ollama_status

router = APIRouter(tags=["calibration"])


class CalibrationRequest(BaseModel):
    industry: str = Field(..., examples=["Banking"])
    wave_size: int = Field(50, ge=10, le=200)
    cost_per_violation: float = Field(..., gt=0)
    weekly_volume: int = Field(..., gt=0)
    base_url: str = Field("http://localhost:11434")
    llama_model: str = Field("llama3.1:latest")
    deepseek_model: str = Field("deepseek-r1:32b")


@router.post("/api/calibration/start")
async def start(req: CalibrationRequest) -> dict:
    """Start a calibration wave. Returns a run_id to connect the WebSocket with."""
    run_id = await start_calibration(
        industry=req.industry,
        wave_size=req.wave_size,
        cost_per_violation=req.cost_per_violation,
        weekly_volume=req.weekly_volume,
        base_url=req.base_url,
        llama_model=req.llama_model,
        deepseek_model=req.deepseek_model,
    )
    return {
        "run_id": run_id,
        "status": "started",
        "total_tests": req.wave_size * 5,
    }


@router.get("/api/calibration/{run_id}/summary")
async def summary(run_id: str) -> dict:
    """Return the current or final state of a calibration run."""
    state = get_run_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    return state


@router.get("/api/status")
async def model_status(base_url: str = "http://localhost:11434") -> dict:
    """Check Ollama and model availability."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        result = await loop.run_in_executor(ex, check_ollama_status, base_url)
    return result


@router.websocket("/ws/calibration/{run_id}")
async def ws_calibration(websocket: WebSocket, run_id: str):
    """
    Stream calibration progress events as JSON messages.

    Event types:
      status        — human-readable status string
      queries_ready — all queries generated, testing starting
      query_result  — one query tested across all 5 levels
      complete      — calibration finished with full summary
      error         — something went wrong
    """
    await websocket.accept()

    # Wait up to 8 s for the run to be registered (handles slight start-up lag)
    for _ in range(80):
        if run_id in run_queues:
            break
        await asyncio.sleep(0.1)

    queue = run_queues.get(run_id)
    if queue is None:
        await websocket.send_json({"type": "error", "message": f"Run {run_id} not found"})
        await websocket.close()
        return

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=600)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Timeout waiting for calibration results",
                })
                break

            # Heartbeat pings keep the connection alive — don't forward to client
            if msg.get("type") == "heartbeat":
                await websocket.send_json({"type": "heartbeat"})
                continue

            await websocket.send_json(msg)

            if msg.get("type") in ("complete", "error"):
                break

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        run_queues.pop(run_id, None)
