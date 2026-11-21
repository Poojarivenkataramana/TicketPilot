# Day 22: Production Decision Logging & Audit Governance Report

**Project**: TicketPilot  
**Sprint**: 5 (Productization — Days 21 to 25)  
**Deliverables**:
- Decision Logger Module: [`decision_logger.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/decision_logger.py)
- Updated FastAPI Backend: [`api.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/api.py)
- Automated Verification Suite: [`test_day22_logging.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day22_logging.py)
- Live JSON Lines Stream Log: [`logs/routing_decisions.jsonl`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/logs/routing_decisions.jsonl)
- Analytical CSV Decision Log: [`logs/routing_decisions.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/logs/routing_decisions.csv)
- Test Verification Summary: [`day22_logging_verification_results.json`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day22_logging_verification_results.json)

---

## 1. Executive Summary & Objective

In enterprise AI deployment, transparency and auditability are non-negotiable. Customer support platforms (e.g. Zendesk, Salesforce Service Cloud) require an immutable audit trail for every automated decision to evaluate compliance, measure human-in-the-loop escalation rates, and identify model drift.

The objective for Day 22 is to implement a thread-safe **Routing Decision Logger** that audits every triage and reply drafting request, categorizes decisions into **"auto-draft"** vs. **"human review"**, and exposes real-time audit queries via the FastAPI backend.

---

## 2. Decision Logging Architecture

```
                    Incoming Ticket Request (POST /draft-reply)
                                       │
                                       ▼
                     TicketPilot Pipeline & Routing Policy
                                       │
                                       ▼
                        Thread-Safe Decision Logger
                      (Mutex Lock for Concurrency Safety)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
    Stream Ingestion Log (JSONL)                  Analytical Snapshot (CSV)
   logs/routing_decisions.jsonl                  logs/routing_decisions.csv
  (Append-only, high throughput)               (Relational analytics & BI)
```

### Log Record Schema

Every logged decision contains 17 structured attributes:
* `log_id`: Unique identifier (e.g., `LOG-A1B2C3D4`).
* `timestamp`: ISO 8601 UTC timestamp.
* `ticket_id`: Customer ticket identifier.
* `subject` & `body_preview`: Input customer inquiry.
* `predicted_queue`: Department queue assigned by the classifier.
* `confidence`: Softmax probability score.
* `predicted_priority`: Priority level (`low`, `medium`, `high`).
* `top_kb_id`: Retrieved source article ID.
* `rerank_score`: Cross-Encoder contextual relevance score.
* `routing_tier`: Operational tier (`TIER_1_AUTO_DRAFT`, `TIER_2_EXPEDITED_REVIEW`, `TIER_3_HUMAN_ASSIST`).
* `decision_category`: High-level operational category (`"auto-draft"` vs. `"human review"`).
* `auto_send_eligible`: Boolean flag for direct automated customer dispatch.
* `requires_human_approval`: Boolean flag for supervisor sign-off.
* `citations`: Bracketed citation IDs (`[KB-XXXXX]`).
* `draft_preview`: Preview of the generated customer email or refusal string.
* `latency_ms`: Total execution time in milliseconds.

---

## 3. Decision Categorization Logic

Decisions are classified using the calibrated compound threshold policy established on Day 20:

$$\begin{aligned}
\text{auto-draft} &\iff \text{Confidence} \ge 0.40 \quad \land \quad \text{Rerank Score} \ge +4.0 \quad \land \quad \text{Priority} \ne \text{"high"} \\
\text{human review} &\iff \text{Priority} == \text{"high"} \quad \lor \quad \text{Confidence} < 0.40 \quad \lor \quad \text{Rerank Score} < +4.0
\end{aligned}$$

* **Tier 1 (`auto-draft`)**: Safe for automated dispatch or 1-click agent dispatch.
* **Tier 2 (`human review`)**: High-priority tickets flagged for mandatory supervisor sign-off.
* **Tier 3 (`human review`)**: Ambiguous or low-confidence tickets safely escalated with top KB snippets pre-attached.

---

## 4. New Audit Governance API Endpoints

We integrated two new governance endpoints into [`api.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/api.py):

### A. `GET /decisions` (Query Recent Decision Logs)
* Supports filtering by `limit` (1 to 100), `tier` (e.g. `TIER_1_AUTO_DRAFT`), and `queue`.
* Example: `GET /decisions?tier=TIER_1_AUTO_DRAFT&limit=10`

### B. `GET /decisions/stats` (Real-Time Telemetry)
* Aggregates live system KPIs:
  ```json
  {
    "total_logged": 20,
    "auto_draft_count": 2,
    "human_review_count": 18,
    "auto_draft_rate_pct": 10.0,
    "human_review_rate_pct": 90.0,
    "avg_confidence_pct": 39.3,
    "avg_latency_ms": 760.13
  }
  ```

---

## 5. Verification Test Suite Results (100% Passed)

We verified the logging architecture via [`test_day22_logging.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day22_logging.py) by running a batch of 20 evaluation tickets through `POST /draft-reply`:

1. **JSONL Log**: Exactly **20 records** created in `logs/routing_decisions.jsonl` with 100% schema compliance.
2. **CSV Log**: Exactly **20 rows** created in `logs/routing_decisions.csv` with persistent headers.
3. **Endpoint Validation**: `GET /decisions` and `GET /decisions/stats` passed all query filter assertions.
4. **Execution Summary**: Saved to [`day22_logging_verification_results.json`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day22_logging_verification_results.json).

---

## "Done When" Verification Check
- [x] Production decision logger created in [`decision_logger.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/decision_logger.py).
- [x] Confidence and priority decide "auto-draft" vs. "human review".
- [x] Thread-safe dual storage implemented (`logs/routing_decisions.jsonl` and `.csv`).
- [x] API endpoints `/decisions` and `/decisions/stats` operational.
- [x] Verification test suite [`test_day22_logging.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day22_logging.py) passes with 100% success.
- [x] **Decisions are logged and audited across incoming requests.**
