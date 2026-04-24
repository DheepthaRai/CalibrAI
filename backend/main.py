from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.database import init_db
from .routers import policy, calibration, audit, inspector, evaluate


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="CalibrAI API",
    description="Governance-aligned LLM safety calibration backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(policy.router)
app.include_router(calibration.router)
app.include_router(audit.router)
app.include_router(inspector.router)
app.include_router(evaluate.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CalibrAI API v2"}
