"""
TicketPilot - Sprint 5 Day 21: Production FastAPI Backend
---------------------------------------------------------
Endpoints:
1. POST /triage: Fast queue and priority classification with confidence score.
2. POST /draft-reply: Grounded RAG reply drafting with citations and routing policy.
3. GET /health: Health check, model status, and indexed vector count.
4. GET /: API metadata and documentation links.

Architecture:
- Models loaded ONCE at startup via FastAPI lifespan manager into app.state.pipeline.
- Pydantic models for strict type validation and automatic OpenAPI Swagger docs (/docs).
- CORS middleware enabled for seamless frontend communication.
"""

import os
import time
import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import pipeline & routing policy
from day15_end_to_end_pipeline import TicketPilotPipeline
from routing_policy import evaluate_routing_policy, CONFIDENCE_THRESHOLD, RERANK_RELEVANCE_THRESHOLD


# ---------------------------------------------------------------------------
# 1. Pydantic Schemas
# ---------------------------------------------------------------------------

class TriageRequest(BaseModel):
    ticket_id: Optional[str] = Field(default="TCK-SAMPLE", description="Unique customer ticket identifier")
    subject: str = Field(..., description="Subject line of the support ticket", example="Unable to connect to PostgreSQL database on port 5432")
    body: str = Field(..., description="Full text body of the support ticket", example="Connections are timing out after 30 seconds from our application cluster. Please assist.")

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TCK-8012",
                "subject": "PostgreSQL database connection timeout error 5432",
                "body": "Our application servers are unable to connect to the PostgreSQL cluster on port 5432. All connections timeout after 30s. How do we fix this?"
            }
        }


class TriageResponse(BaseModel):
    ticket_id: str
    predicted_queue: str
    confidence: float
    predicted_priority: str
    latency_ms: float
    timestamp: str


class DraftReplyRequest(BaseModel):
    ticket_id: Optional[str] = Field(default="TCK-SAMPLE", description="Unique ticket identifier")
    subject: str = Field(..., description="Subject line of the customer inquiry")
    body: str = Field(..., description="Full message body of the customer inquiry")
    override_queue: Optional[str] = Field(default=None, description="Optional manual override for queue metadata filter")

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "TCK-8013",
                "subject": "Duplicate subscription renewal charge on credit card",
                "body": "I was charged twice on my Visa card for this month's renewal invoice INV-904. I need a refund for the duplicate transaction."
            }
        }


class KBSourceSnippet(BaseModel):
    kb_id: str
    queue: str
    rerank_score: float
    problem: str
    resolution: str


class DraftReplyResponse(BaseModel):
    ticket_id: str
    predicted_queue: str
    confidence: float
    predicted_priority: str
    routing_tier: str
    routing_tier_label: str
    action: str
    auto_send_eligible: bool
    requires_human_approval: bool
    draft_text: str
    top_sources: List[KBSourceSnippet]
    citations: List[str]
    grounding_audit_passed: bool
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    chromadb_vector_count: int
    kb_article_count: int
    confidence_threshold: float
    rerank_threshold: float
    uptime_seconds: float


# ---------------------------------------------------------------------------
# 2. Lifespan Manager: Load Models Once at Startup
# ---------------------------------------------------------------------------

START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all triage models, ChromaDB index, BM25 index, and Cross-Encoder
    ONCE at application startup.
    """
    print("\n[FastAPI Startup] Initializing TicketPilot Pipeline...")
    t0 = time.time()
    
    # Initialize pipeline
    app.state.pipeline = TicketPilotPipeline(
        models_dir="models",
        kb_path="data/knowledge_base_clean.csv",
        chroma_path="chroma_db",
        relevance_threshold=RERANK_RELEVANCE_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    load_time = time.time() - t0
    print(f"[FastAPI Startup] Pipeline loaded successfully in {load_time:.2f} seconds.")
    print("[FastAPI Startup] TicketPilot REST API is ready to accept requests at /docs.\n")
    
    yield
    
    print("\n[FastAPI Shutdown] Tearing down TicketPilot API resources.")


# ---------------------------------------------------------------------------
# 3. Application Initialization & Middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TicketPilot API",
    description=(
        "Production-grade Customer Support Ticket Triage and Grounded RAG Assistant.\n\n"
        "**Core Capabilities**:\n"
        "- `/triage`: High-speed queue & priority classification with confidence score.\n"
        "- `/draft-reply`: Grounded RAG reply drafting with citations, guardrail refusal, and compound routing tiers.\n"
        "- **Models Loaded Once at Startup** for sub-second real-time responsiveness."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit / React / Frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 4. Route Handlers
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    """Root info endpoint."""
    return {
        "service": "TicketPilot REST API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "description": "Customer Support Ticket Triage & Grounded RAG Assistant"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health(request: Request):
    """Health check endpoint reporting pipeline readiness and vector counts."""
    pipeline: TicketPilotPipeline = request.app.state.pipeline
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        chromadb_vector_count=pipeline.collection.count(),
        kb_article_count=len(pipeline.kb_df),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        rerank_threshold=RERANK_RELEVANCE_THRESHOLD,
        uptime_seconds=round(time.time() - START_TIME, 2)
    )


@app.post("/triage", response_model=TriageResponse, tags=["Triage & Classification"])
async def triage_ticket(payload: TriageRequest, request: Request):
    """
    Step 1: Triage Ticket.
    Classifies incoming ticket into one of 10 department queues and 3 priority levels
    with a softmax confidence score in ~1 ms.
    """
    t0 = time.perf_counter()
    pipeline: TicketPilotPipeline = request.app.state.pipeline
    full_text = f"{payload.subject} {payload.body}"

    try:
        triage_res = pipeline.classify_ticket(full_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

    latency_ms = (time.perf_counter() - t0) * 1000

    return TriageResponse(
        ticket_id=payload.ticket_id or "TCK-SAMPLE",
        predicted_queue=triage_res["pred_queue"],
        confidence=round(triage_res["confidence"], 4),
        predicted_priority=triage_res["pred_priority"],
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z"
    )


@app.post("/draft-reply", response_model=DraftReplyResponse, tags=["RAG & Grounded Drafting"])
async def draft_reply(payload: DraftReplyRequest, request: Request):
    """
    Step 2: End-to-End Grounded Reply Drafting.
    Executes full pipeline:
    Triage -> Scoped Hybrid Retrieval -> Cross-Encoder Reranker -> Routing Policy -> Grounded Drafting with Citations.
    """
    t0 = time.perf_counter()
    pipeline: TicketPilotPipeline = request.app.state.pipeline
    full_text = f"{payload.subject} {payload.body}"

    ticket_dict = {
        "ticket_id": payload.ticket_id or "TCK-SAMPLE",
        "subject": payload.subject,
        "body": payload.body,
        "text": full_text
    }

    try:
        # Run pipeline
        res = pipeline.run_ticket(ticket_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")

    # Evaluate production compound routing policy
    routing = evaluate_routing_policy(
        pred_queue=res["pred_queue"],
        confidence=res["confidence"],
        pred_priority=res["pred_priority"],
        top_rerank_score=res["top_rerank_score"]
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Format top sources
    top_sources = []
    if res.get("top_kb_id") and res["top_kb_id"] != "None":
        idx = pipeline.kb_id_to_idx.get(res["top_kb_id"])
        if idx is not None:
            row = pipeline.kb_df.iloc[idx]
            top_sources.append(KBSourceSnippet(
                kb_id=res["top_kb_id"],
                queue=row["queue"],
                rerank_score=round(res["top_rerank_score"], 2),
                problem=row["problem"],
                resolution=row["solution"]
            ))

    citations_list = [c.strip() for c in res["found_citations"].split(",") if c.strip() and c.strip() != "None"]

    return DraftReplyResponse(
        ticket_id=res["ticket_id"],
        predicted_queue=res["pred_queue"],
        confidence=round(res["confidence"], 4),
        predicted_priority=res["pred_priority"],
        routing_tier=routing["tier"],
        routing_tier_label=routing["tier_label"],
        action=routing["action"],
        auto_send_eligible=routing["auto_send_eligible"],
        requires_human_approval=routing["requires_human_approval"],
        draft_text=res["draft_text"],
        top_sources=top_sources,
        citations=citations_list,
        grounding_audit_passed=res["audit_pass"],
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z"
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting TicketPilot API on http://127.0.0.1:8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
