from fastapi import APIRouter
from pydantic import BaseModel, Field
from ..services.policy_service import recommend_threshold

router = APIRouter(prefix="/api/policy", tags=["policy"])


class PolicyRequest(BaseModel):
    industry: str = Field(..., examples=["Banking"])
    cost_per_violation: float = Field(..., gt=0, examples=[5000.0])
    weekly_volume: int = Field(..., gt=0, examples=[10000])


@router.post("")
async def get_policy_recommendation(req: PolicyRequest) -> dict:
    """Derive a recommended safety threshold from the three policy inputs."""
    return recommend_threshold(req.industry, req.cost_per_violation, req.weekly_volume)
