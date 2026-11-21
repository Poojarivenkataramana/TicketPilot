# Day 20: TicketPilot One-Page Executive Evaluation Report & Production Thresholds

**Project**: TicketPilot — Enterprise Support Triage & RAG Assistant  
**Author**: Venkataramana Poojari  
**Sprint**: 4 Capstone (Evaluation — Days 16 to 20)  
**Deliverables**:
- Production Routing Policy Module: [`routing_policy.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/routing_policy.py)
- Executive Visual Dashboard: [`charts/chart20_sprint4_executive_summary.html`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart20_sprint4_executive_summary.html)
- Official Scored Sheet (N=30): [`data/day18_scored_sheet_30.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/data/day18_scored_sheet_30.csv)
- Latency Benchmark (N=60): [`day19_latency_summary.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day19_latency_summary.csv)
- Master Results Table: [`results_table.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/results_table.csv)

---

## 1. Executive Summary

TicketPilot is an end-to-end customer support assistant that predicts ticket queue and priority, routes incoming requests, and drafts cited email resolutions using a grounded Knowledge Base. Sprint 4 evaluated the system against a **locked, uncorrupted benchmark of 60 held-out test tickets** stratified across all 10 customer service queues and 3 priority levels. 

Across quantitative information retrieval, human-centric qualitative audits, and nanosecond latency profiling, TicketPilot proved ready for enterprise production deployment.

```
       ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
       │   Top-1 Hit Rate     │     │   Composite Quality  │     │    Median Latency    │
       │       90.00%         │     │     4.81 / 5.00      │     │      609.16 ms       │
       │ (+15.0% over vector) │     │  (0.0% Hallucination)│     │  (Sub-Second Realtime│
       └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

---

## 2. Quantitative Retrieval Benchmark (N=60 Tickets)

Evaluating across all 60 held-out tickets demonstrated that combining dense semantic search with sparse BM25 and Cross-Encoder reranking significantly outperforms standard vector search:

| Retrieval Architecture | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | Mean Reciprocal Rank (MRR) | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Method 1: Plain Vector Search** | 75.00% | 90.00% | 90.00% | 0.8139 | **19.83 ms** |
| **Method 2: Hybrid Search (Dense + BM25)** | 83.33% | 90.00% | 90.00% | 0.8639 | 272.01 ms |
| **Method 3: Hybrid + Cross-Encoder Rerank 🏆** | **90.00%** | **90.00%** | **91.67%** | **0.9033** | 591.90 ms |
| **Net Gain (Method 3 vs. Method 1)** | **+15.00%** | **—** | **+1.67%** | **+0.0894** | (+572 ms) |

* **Why It Works**: BM25 catches exact technical keywords ("PostgreSQL", "RMA", "Keras") that vector embeddings dilute, while the Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) performs token-level bidirectional cross-attention, boosting MRR to **0.9033**.

---

## 3. Human Qualitative Quality Audit (N=30 Drafts)

Scoring 30 stratified drafts across 4 dimensions on a 1-to-5 Likert scale confirmed strict enterprise reliability:

* **Accuracy (`4.90 / 5.00`)**: Correct troubleshooting steps, direct navigation paths, and valid policies.
* **Grounding (`4.97 / 5.00`)**: 100% citation compliance (`[KB-XXXXX]`) with **zero hallucinated facts**.
* **Tone (`4.40 / 5.00`)**: Professional customer email formatting with warm greetings and clear closings.
* **Safety (`4.97 / 5.00`)**: Zero PII leaks, zero false refund commitments, and reliable refusal on ambiguity.
* **Overall Composite Score**: **`4.81 / 5.00`**.

---

## 4. Latency & Throughput Profile

* **End-to-End Latency**: Mean: **656.55 ms** | Median ($p_{50}$): **609.16 ms** | $p_{90}$: **940.19 ms** | $p_{99}$: **1,387.18 ms**.
* **Stage Distribution**: Cross-Encoder (49.0%) and Python BM25 (42.0%) account for 91% of execution time. Triage classification (0.97 ms) and Grounded Drafting (0.08 ms) are virtually instantaneous.
* **Throughput**: **`1.52 tickets/second`** on single local worker (~**91 tickets/minute**).

---

## 5. Formal Production Thresholds: Auto-Draft vs. Human Review

To eliminate cascade misrouting and protect customer trust, TicketPilot implements a **Compound Three-Tier Routing Architecture** encoded in [`routing_policy.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/routing_policy.py):

```
                                  Incoming Customer Ticket
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             High Risk or Outage?                          Standard Ticket
            (Priority = High OR Outage)                            │
                       │                                           ▼
                       │                             Confidence ≥ 0.40 AND
                       │                             Rerank Score ≥ +4.0?
                       │                                    │             │
                       │                               Yes  │             │ No
                       ▼                                    ▼             ▼
             ┌───────────────────┐               ┌────────────────┐  ┌──────────────────┐
             │      TIER 2       │               │     TIER 1     │  │      TIER 3      │
             │ Expedited Review  │               │   Auto-Draft   │  │   Human Assist   │
             │ (Human Sign-Off)  │               │ (Direct Send)  │  │(Refusal + Top KB)│
             └───────────────────┘               └────────────────┘  └──────────────────┘
```

1. **Tier 1: Fully Automated Auto-Draft (High Confidence, High Relevance)**
   - **Condition**: $\text{Confidence} \ge 0.40 \quad \mathbf{AND} \quad \text{Cross-Encoder Score} \ge +4.0 \quad \mathbf{AND} \quad \text{Priority} \ne \text{"high"}$
   - **Action**: Auto-draft cited email reply with `[KB-XXXXX]` tags. Eligible for direct automated delivery or 1-click human dispatch (~28% of total volume).
2. **Tier 2: Expedited Human Review (High-Risk Tickets)**
   - **Condition**: $\text{Priority} == \text{"high"} \quad \mathbf{OR} \quad \text{Queue} == \text{"Service Outages and Maintenance"}$
   - **Action**: Draft cited reply immediately, but flag as high-priority and require supervisor sign-off before sending (~22% of total volume).
3. **Tier 3: Human Agent Review & Assist (Ambiguous / Low-Confidence Tickets)**
   - **Condition**: $\text{Confidence} < 0.40 \quad \mathbf{OR} \quad \text{Cross-Encoder Score} < +4.0$
   - **Action**: Refusal guardrail triggers (*"Not enough information, escalating to a human agent."*); route raw ticket to specialized human queue with top-3 retrieved KB snippets pre-attached for human agent context (~50% of volume).

---

## 6. Key ML Interview Talking Points

1. **Anti-Leakage Rigor**: Built the 17,469-article Knowledge Base strictly from the training split. Confirmed 0 test questions and 0 test solutions exist in the KB.
2. **Overcoming Cascade Errors**: Discovered on Day 15 that rigid metadata filtering misroutes borderline tickets. Solved on Day 19–20 by establishing compound gating ($\tau_{conf} \ge 0.40 \land \tau_{rerank} \ge +4.0$), slashing misrouted drafts from 40% to 0%.
3. **Information Retrieval Tradeoff**: Proved that Bi-Encoders are fast (19 ms) but suffer from token dilution. Blending BM25 with Cross-Encoder reranking boosted Hit Rate@1 by **+15.0%** (to 90.0%) and MRR to **0.9033** within acceptable sub-second latency (609 ms).

---

## "Done When" Verification Check
- [x] Quantitative retrieval results consolidated (Hit Rate@1, Hit Rate@5, MRR, Latency).
- [x] Human qualitative audit scores synthesized across Accuracy, Grounding, Tone, and Safety.
- [x] Production thresholds for Auto-Draft vs. Human Review mathematically defined and implemented in [`routing_policy.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/routing_policy.py).
- [x] Executive visual dashboard created at [`charts/chart20_sprint4_executive_summary.html`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart20_sprint4_executive_summary.html).
- [x] **Official one-page evaluation report finished and committed to Git.**
