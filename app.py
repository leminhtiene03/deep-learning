# app.py

import os
import asyncio
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.retriever import RRFHybridRetriever
from core.answer import load_lora_model, ADAPTER_DIR
from core.pipeline import answer_question_controlled
from pathlib import Path
from fastapi.responses import RedirectResponse, FileResponse
from knowledge.db import init_db
from knowledge.store import (
    list_pending_questions,
    list_corrections,
    mark_correction_notified,
)

from workflow.teacher_queue import answer_pending_question


# ============================================================
# Config
# ============================================================

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "local-admin-token")

MODEL_ADAPTER_DIR = os.environ.get(
    "BARTPHO_LORA_ADAPTER",
    ADAPTER_DIR
)

MAX_CONCURRENT_GENERATIONS = int(
    os.environ.get("MAX_CONCURRENT_GENERATIONS", "1")
)

generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Vietnamese RAG Chatbot",
    version="0.1.0"
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================
# Global objects: load một lần
# ============================================================

retriever = None
tokenizer = None
model = None
device = None


@app.on_event("startup")
def startup_event():
    global retriever, tokenizer, model, device

    print("Initializing SQLite database...")
    init_db()

    print("Loading retriever...")
    retriever = RRFHybridRetriever(
        bm25_weight=0.2,
        faiss_weight=0.8,
        rrf_k=60
    )

    print("Loading BARTpho-LoRA model...")
    tokenizer, model, device = load_lora_model(MODEL_ADAPTER_DIR)

    print("Server ready.")
    print("Device:", device)


# ============================================================
# Request / Response models
# ============================================================

class AskRequest(BaseModel):
    question: str
    user_id: Optional[str] = "web_user"
    conversation_id: Optional[str] = None
    candidate_k: Optional[int] = 30
    top_k: Optional[int] = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    decision: str
    confidence: str
    topic: Optional[str] = None
    pending_id: Optional[int] = None
    answer_log_id: Optional[int] = None
    source_type: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None


class AnswerPendingRequest(BaseModel):
    pending_id: int
    answer: str
    verified_by: Optional[str] = "admin"
    source: Optional[str] = "teacher"


class MarkCorrectionRequest(BaseModel):
    correction_id: int


# ============================================================
# Helpers
# ============================================================

def require_admin(x_admin_token: Optional[str]):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


def compact_source(chunk: Optional[Dict[str, Any]], chunks=None):
    if chunks:
        return {
            "top_chunk": {
                "chunk_id": chunk.get("chunk_id") if chunk else None,
                "page": chunk.get("page") if chunk else None,
                "title": chunk.get("title") if chunk else None,
            },
            "chunks": [
                {
                    "rank": c.get("rank"),
                    "chunk_id": c.get("chunk_id"),
                    "page": c.get("page"),
                    "title": c.get("title"),
                    "rrf_score": c.get("rrf_score"),
                    "bm25_rank": c.get("bm25_rank"),
                    "faiss_rank": c.get("faiss_rank"),
                }
                for c in chunks
            ]
        }

    if not chunk:
        return None

    return {
        "chunk_id": chunk.get("chunk_id"),
        "page": chunk.get("page"),
        "title": chunk.get("title"),
        "rrf_score": chunk.get("rrf_score"),
        "bm25_rank": chunk.get("bm25_rank"),
        "faiss_rank": chunk.get("faiss_rank"),
    }


# ============================================================
# Pages
# ============================================================

@app.get("/")
def root():
    return RedirectResponse(url="/chat")


@app.get("/chat")
def chat_page():
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


# ============================================================
# Chat API
# ============================================================

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")

    if retriever is None or tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    result = await asyncio.to_thread(
        answer_question_controlled,
        req.question,
        retriever,
        tokenizer,
        model,
        device,
        req.user_id,
        req.candidate_k,
        req.top_k,
        req.conversation_id
    )

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        decision=result["decision"],
        confidence=result["confidence"],
        topic=result.get("topic"),
        pending_id=result.get("pending_id"),
        answer_log_id=result.get("answer_log_id"),
        source_type=result.get("source_type"),
        source=compact_source(result.get("chunk"), result.get("chunks")),
        debug={
            "context_check": result.get("context_check"),
            "answer_check": result.get("answer_check"),
            "gate": result.get("gate"),
            "raw_model_answer": result.get("raw_model_answer"),
        }
    )


# ============================================================
# Admin API
# ============================================================

@app.get("/api/admin/pending")
def api_list_pending(
    status: str = "pending",
    limit: int = 50,
    x_admin_token: Optional[str] = Header(default=None)
):
    require_admin(x_admin_token)

    items = list_pending_questions(
        status=status,
        limit=limit
    )

    return {
        "status": status,
        "count": len(items),
        "items": items
    }


@app.post("/api/admin/answer-pending")
def api_answer_pending(
    req: AnswerPendingRequest,
    x_admin_token: Optional[str] = Header(default=None)
):
    require_admin(x_admin_token)

    if not req.answer.strip():
        raise HTTPException(status_code=400, detail="Answer is empty")

    result = answer_pending_question(
        pending_id=req.pending_id,
        answer=req.answer,
        verified_by=req.verified_by,
        source=req.source,
        create_corrections=True
    )

    return result


@app.get("/api/admin/corrections")
def api_list_corrections(
    status: str = "pending",
    limit: int = 50,
    x_admin_token: Optional[str] = Header(default=None)
):
    require_admin(x_admin_token)

    items = list_corrections(
        status=status,
        limit=limit
    )

    return {
        "status": status,
        "count": len(items),
        "items": items
    }


@app.post("/api/admin/corrections/mark-notified")
def api_mark_correction_notified(
    req: MarkCorrectionRequest,
    x_admin_token: Optional[str] = Header(default=None)
):
    require_admin(x_admin_token)

    mark_correction_notified(req.correction_id)

    return {
        "ok": True,
        "correction_id": req.correction_id,
        "status": "notified"
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "device": str(device),
        "model_ready": model is not None,
        "retriever_ready": retriever is not None
    }