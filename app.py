"""
FastAPI entrypoint.

Run with:  uvicorn app:app --reload
Docs at:   http://localhost:8000/docs
"""
import logging
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import build_agent, run_turn
from src.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Assistant",
    description="A tool-using LLM agent with weather, search, document RAG, "
    "calculator, and word-count tools.",
    version="1.0.0",
)

# Built once at startup and reused across requests; the agent object
# itself is stateless, per-conversation state lives in the checkpointer.
_agent = None


@app.on_event("startup")
def _startup() -> None:
    global _agent
    _agent = build_agent()
    logger.info("Agent initialized.")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message.")
    thread_id: str | None = Field(
        default=None,
        description="Conversation/session id. Omit to start a new conversation.",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet.")

    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        reply = run_turn(_agent, req.message, thread_id=thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent turn failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(reply=reply, thread_id=thread_id)
