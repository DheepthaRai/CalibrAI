from fastapi import APIRouter, HTTPException
from ..services.calibration_service import active_runs

router = APIRouter(prefix="/api/inspector", tags=["inspector"])


@router.get("/{run_id}")
async def list_queries(run_id: str) -> dict:
    """List all queries in a completed run (for the query selector dropdown)."""
    state = active_runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    queries = state.get("queries", {})
    return {
        "run_id": run_id,
        "queries": [
            {"qid": int(qid), "query": d["query"], "is_attack": d["is_attack"]}
            for qid, d in queries.items()
        ],
    }


@router.get("/{run_id}/{qid}")
async def inspect_query(run_id: str, qid: int) -> dict:
    """
    Return how each of the 5 safety levels handled a specific query.
    Used by the Live Inspector panel.
    """
    state = active_runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")

    # Pull detailed results from in-memory task results (stored on task completion)
    # This requires the background task to expose per-query detail.
    # We store a compact form in active_runs["query_detail"].
    detail = state.get("query_detail", {}).get(str(qid))
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Query {qid} not found")

    return detail
