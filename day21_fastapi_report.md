# Day 21: Production FastAPI Backend & REST Service Report

**Project**: TicketPilot  
**Sprint**: 5 (Productization — Days 21 to 25)  
**Deliverables**:
- Production FastAPI Application: [`api.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/api.py)
- Automated Test Suite: [`test_day21_api.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day21_api.py)
- Verified Test Payloads & Output: [`day21_api_test_results.json`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day21_api_test_results.json)

---

## 1. Executive Summary & Architecture

Day 21 marks the transition of TicketPilot from an offline machine learning pipeline into an enterprise-ready, high-throughput REST API backend using **FastAPI** and **Uvicorn**.

The architecture adheres to three critical production design principles:

1. **Lifespan Context Management (Load Once at Startup)**:
   * Rather than reloading multi-megabyte machine learning weights and vector database connections on every incoming HTTP request, the entire pipeline (TF-IDF, Linear SVM, ChromaDB, BM25, and Cross-Encoder) is initialized **strictly once** at application startup via FastAPI's `@asynccontextmanager lifespan`.
   * Startup duration: **`2.01 seconds`**. Subsequent inference requests execute in sub-second real time.
2. **Strict Pydantic Type Contracts & Auto-Documentation**:
   * All request payloads and response bodies are strictly typed and validated using Pydantic schemas.
   * Auto-generates interactive Swagger documentation at `/docs` and ReDoc at `/redoc`.
3. **Decoupled Architecture with CORS Enabled**:
   * Built-in `CORSMiddleware` allows frontend clients (Streamlit, React, Next.js, or curl) to communicate with the backend.

---

## 2. API Endpoints Specification

### A. `POST /triage` (Ultra-Fast Ticket Classification)
* **Purpose**: Instant queue routing and priority triage before human assignment or automated drafting.
* **Latency**: **`~9.2 ms`** per request.
* **Request Schema**:
  ```json
  {
    "ticket_id": "TEST-TRIAGE-01",
    "subject": "PostgreSQL database connection timeout error 5432",
    "body": "Our application servers are unable to connect to the PostgreSQL cluster on port 5432. All connections timeout after 30 seconds."
  }
  ```
* **Response Schema (200 OK)**:
  ```json
  {
    "ticket_id": "TEST-TRIAGE-01",
    "predicted_queue": "Technical Support",
    "confidence": 0.4494,
    "predicted_priority": "high",
    "latency_ms": 8.21,
    "timestamp": "2026-11-21T15:05:55.405622Z"
  }
  ```

---

### B. `POST /draft-reply` (Grounded RAG Drafting & Routing Tiers)
* **Purpose**: Full end-to-end RAG assistant: triage $\rightarrow$ hybrid search $\rightarrow$ cross-encoder rerank $\rightarrow$ compound routing tier decision $\rightarrow$ grounded email drafting with `[KB-XXXXX]` citations.
* **Request Schema**:
  ```json
  {
    "ticket_id": "TEST-DRAFT-01",
    "subject": "Duplicate subscription renewal charge on credit card",
    "body": "I was charged twice on my Visa card for this month's renewal invoice INV-904. I need a refund for the duplicate transaction."
  }
  ```
* **Response Schema (200 OK - Tier 1 Auto-Draft)**:
  ```json
  {
    "ticket_id": "TEST-DRAFT-01",
    "predicted_queue": "Billing and Payments",
    "confidence": 0.7801,
    "predicted_priority": "medium",
    "routing_tier": "TIER_1_AUTO_DRAFT",
    "routing_tier_label": "Tier 1: Automated Auto-Draft",
    "action": "Auto-Draft Cited Reply",
    "auto_send_eligible": true,
    "requires_human_approval": false,
    "draft_text": "Hello,\n\nThank you for contacting customer support. We are happy to assist you with your inquiry.\n\n[Customer Name], we apologize for the issue with your subscription renewal payment [KB-14920]...",
    "top_sources": [
      {
        "kb_id": "KB-14920",
        "queue": "Billing and Payments",
        "rerank_score": 5.95,
        "problem": "Problem with Subscription Renewal Payment...",
        "resolution": "[Customer Name], we apologize for the issue with your subscription renewal payment..."
      }
    ],
    "citations": ["[KB-14920]"],
    "grounding_audit_passed": true,
    "latency_ms": 1301.01,
    "timestamp": "2026-11-21T15:05:56.709317Z"
  }
  ```

---

### C. `GET /health` & `GET /` (Telemetry & Status)
* Reports API operational status, uptime, ChromaDB vector collection count (`17,469`), and active decision thresholds.

---

## 3. Automated Verification Test Suite (6 / 6 Passed)

We verified the live endpoints via [`test_day21_api.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day21_api.py):

| Test Case | Target Endpoint | Scenario Tested | Status | Latency | Key Output Verification |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `GET /` | Root info & Swagger URLs | **200 OK** | 1.7 ms | Verified service online and `/docs` link active |
| **2** | `GET /health` | System health & ChromaDB count | **200 OK** | 3.7 ms | 17,469 indexed vectors reported |
| **3** | `POST /triage` | Fast triage classification | **200 OK** | 9.3 ms | Classified to Technical Support (44.9% conf) |
| **4** | `POST /draft-reply` | High-confidence billing query | **200 OK** | 1,313 ms | Correctly mapped to **Tier 1: Auto-Draft** with valid citations |
| **5** | `POST /draft-reply` | Critical cloud outage query | **200 OK** | 865 ms | Correctly mapped to **Tier 2: Expedited Review** (`requires_human_approval=True`) |
| **6** | `POST /draft-reply` | Ambiguous low-confidence query | **200 OK** | 737 ms | Correctly mapped to **Tier 3: Human Assist** with safe refusal guardrail |

---

## 4. How to Launch the API Server Locally

```bash
cd /Users/poojarivenkataramana/Downloads/ticket_complaint
myenv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Interactive documentation is instantly available in any web browser at:
* **Swagger UI**: `http://127.0.0.1:8000/docs`
* **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## "Done When" Verification Check
- [x] FastAPI REST service created in [`api.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/api.py).
- [x] Models loaded strictly once at application startup using modern lifespan manager.
- [x] `/triage` and `/draft-reply` endpoints implemented with strict Pydantic schemas.
- [x] Comprehensive test suite [`test_day21_api.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day21_api.py) verified 6/6 tests passing.
- [x] Test results logged in [`day21_api_test_results.json`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day21_api_test_results.json).
- [x] **Both endpoints work and are fully documented in the interactive API docs page.**
