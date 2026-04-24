import csv
import io
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from ..db.database import get_db
from ..db.models import AuditLog

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("")
async def get_audit_log(
    run_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return paginated audit log rows, optionally filtered by run_id."""
    q = select(AuditLog).order_by(AuditLog.timestamp.desc())
    count_q = select(func.count()).select_from(AuditLog)

    if run_id:
        q = q.where(AuditLog.run_id == run_id)
        count_q = count_q.where(AuditLog.run_id == run_id)

    total_result = await db.execute(count_q)
    total = total_result.scalar_one()

    q = q.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(q)
    rows = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": r.id,
                "run_id": r.run_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "query_text": r.query_text,
                "is_attack": r.is_attack,
                "risk_score": r.risk_score,
                "safety_level": r.safety_level,
                "outcome": r.outcome,
                "cost_justification": r.cost_justification,
                "industry": r.industry,
            }
            for r in rows
        ],
    }


@router.get("/export")
async def export_csv(
    run_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export the audit log as a CSV file."""
    q = select(AuditLog).order_by(AuditLog.timestamp.asc())
    if run_id:
        q = q.where(AuditLog.run_id == run_id)

    result = await db.execute(q)
    rows = result.scalars().all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "run_id", "timestamp", "query_text", "is_attack",
        "risk_score", "safety_level", "outcome", "cost_justification", "industry",
    ])
    for r in rows:
        writer.writerow([
            r.id, r.run_id,
            r.timestamp.isoformat() if r.timestamp else "",
            r.query_text, r.is_attack,
            r.risk_score, r.safety_level,
            r.outcome, r.cost_justification, r.industry,
        ])

    buf.seek(0)
    filename = f"calibrai_audit_{run_id or 'all'}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
