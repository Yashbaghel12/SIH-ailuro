"""
FastAPI wrapper exposing the RAG distress engine as an HTTP API.

Run from the REPO ROOT (not from inside api/), so the engine
package resolves correctly:
    pip install -r requirements.txt
    uvicorn api.main:app --reload --port 8000

Then POST to http://localhost:8000/checkin
"""

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.rag_distress_engine import RagDistressEngine
from engine.distress_engine import CheckIn
from engine.alerting import CaseWorkerNotifier

app = FastAPI(title="Distress Monitoring API")

# Allows the frontend (different port/domain) to call this API.
# Lock allow_origins down to your real frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RagDistressEngine()
notifier = CaseWorkerNotifier()


class CheckInRequest(BaseModel):
    mood_score: int = Field(..., ge=1, le=5, description="1 (very distressed) - 5 (very good)")
    text: str = ""
    days_since_complaint: int = 0
    missed_checkins: int = 0
    threat_reported: bool = False
    case_id: str | None = None  # optional; auto-generated if not provided


class CheckInResponse(BaseModel):
    case_id: str
    score: float
    risk_level: str
    alert: bool
    reasons: list[str]
    llm_indicators: list[str]
    retrieved_context: list[str]
    text_engine: str
    authority_notified: bool


@app.post("/checkin", response_model=CheckInResponse)
def submit_checkin(payload: CheckInRequest):
    case_id = payload.case_id or f"checkin-{uuid.uuid4().hex[:8]}"

    checkin = CheckIn(
        mood_score=payload.mood_score,
        text=payload.text,
        days_since_complaint=payload.days_since_complaint,
        missed_checkins=payload.missed_checkins,
        threat_reported=payload.threat_reported,
    )

    try:
        result = engine.score(checkin)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    authority_notified = False
    if result.alert:
        # High risk -> hand off to a human case worker/authority.
        # Never an autonomous AI decision -- see engine/alerting.py.
        notify_result = notifier.notify(
            case_id=case_id,
            score=result.score,
            risk_level=result.risk_level,
            reasons=result.reasons,
            threat_reported=payload.threat_reported,
        )
        authority_notified = notify_result["delivered"]

    return CheckInResponse(
        case_id=case_id,
        score=result.score,
        risk_level=result.risk_level,
        alert=result.alert,
        reasons=result.reasons,
        llm_indicators=result.llm_indicators,
        retrieved_context=result.retrieved_context,
        text_engine=result.text_engine,
        authority_notified=authority_notified,
    )


@app.get("/health")
def health():
    return {"status": "ok"}