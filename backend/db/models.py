from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    query_text = Column(Text, nullable=False)
    is_attack = Column(Boolean, nullable=False)
    risk_score = Column(Float, nullable=False)      # 0.0–1.0: fraction of levels that blocked
    safety_level = Column(Integer, nullable=False)  # 1–5
    outcome = Column(String, nullable=False)        # "ALLOWED" or "BLOCKED"
    response_text = Column(Text)
    cost_justification = Column(Text)
    industry = Column(String)


class CalibrationRun(Base):
    __tablename__ = "calibration_runs"

    id = Column(String, primary_key=True)           # UUID
    timestamp = Column(DateTime, default=datetime.utcnow)
    industry = Column(String, nullable=False)
    wave_size = Column(Integer, nullable=False)
    cost_per_violation = Column(Float, nullable=False)
    weekly_volume = Column(Integer, nullable=False)
    recommended_level = Column(Integer, nullable=True)
    status = Column(String, default="pending")      # pending | running | complete | error
