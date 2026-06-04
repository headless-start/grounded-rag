"""FastAPI surface: POST /ask, GET /health.

The answer service (and its embedder/store/LLM) is built once at startup and reused.
Run with: ``uvicorn grounded_rag.api:app`` (see the Makefile ``run`` target).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .answer import AnswerService, get_answer_service
from .log import get_logger
from .models import Answer

log = get_logger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    log.info("startup_loading_service")
    _state["service"] = get_answer_service()
    log.info("startup_ready")
    yield
    _state.clear()


app = FastAPI(title="grounded-rag", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service_ready": str("service" in _state)}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    service: AnswerService = _state["service"]
    return service.ask(req.question)
