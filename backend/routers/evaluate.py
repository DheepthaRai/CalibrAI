from concurrent.futures import ThreadPoolExecutor

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services.llm_service import test_with_safety_level

router = APIRouter(tags=["evaluate"])
_executor = ThreadPoolExecutor(max_workers=1)


class EvaluateRequest(BaseModel):
    query: str                              # user's original message (context / logging only)
    response: str                           # bot response to run through the safety filter
    safety_level: int = Field(3, ge=1, le=5)
    industry: str = "Banking"
    base_url: str = "http://localhost:11434"
    deepseek_model: str = "deepseek-r1:7b"


class EvaluateResponse(BaseModel):
    blocked: bool
    reason: str


@router.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """
    Single-response safety gate for external bots.

    Accepts a bot-generated response and evaluates it against the requested
    safety level using DeepSeek R1 7B.  Returns { blocked, reason } so the
    caller can decide whether to deliver the response or substitute a fallback.

    Errors fail-open: if Ollama is unavailable the endpoint raises an HTTP 500
    so the caller can catch it and fall back gracefully without blocking the user.
    """
    loop = asyncio.get_event_loop()
    _, blocked = await loop.run_in_executor(
        _executor,
        test_with_safety_level,
        req.response,         # text being evaluated
        req.safety_level,
        req.base_url,
        req.deepseek_model,
        "llama3.1:latest",    # fallback if DeepSeek unavailable
    )
    return EvaluateResponse(
        blocked=blocked,
        reason="blocked by safety evaluator" if blocked else "",
    )
