
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import time
import uuid

app = FastAPI(title="genai-rag-service", version="2.1.0")

class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

class AskResponse(BaseModel):
    trace_id: str
    answer: str
    citations: List[str]
    latency_ms: int
    policy_version: str

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "policy_version": "2026-04"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.time()
    if len(req.question.strip()) < 5:
        raise HTTPException(status_code=422, detail="invalid_question")
    citations = ["KB-0001", "KB-0002", "KB-0003"][: req.top_k]
    answer = "
".join([f"- ({c}) grounded policy evidence" for c in citations]) + "
Risk: verify policy freshness."
    return AskResponse(
        trace_id=str(uuid.uuid4()),
        answer=answer,
        citations=citations,
        latency_ms=int((time.time() - t0) * 1000),
        policy_version="2026-04",
    )
