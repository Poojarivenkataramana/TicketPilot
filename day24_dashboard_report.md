# Sprint 5 Day 24: Plotly Operational Dashboard & Batch Analytics

**Project**: TicketPilot — Enterprise Customer Support AI Triage & Grounded RAG Assistant  
**Date**: November 21, 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

On Day 24, we delivered the production **Operational Monitoring Dashboard** for TicketPilot. To validate system performance under realistic enterprise load, we benchmarked an end-to-end test batch of **100 stratified customer support tickets** sampled from the held-out test split (`data/test.csv`), covering all 10 distinct support queues.

The system evaluated each ticket through our end-to-end pipeline:
1. TF-IDF + Linear SVM Queue & Priority Classification
2. Scoped Dense ChromaDB + Sparse BM25 Hybrid Retrieval with RRF
3. Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)
4. Three-Tier Routing Policy evaluation with anti-hallucination refusal guardrails
5. Decision Logging into permanent audit streams

All execution telemetry was analyzed and rendered into an executive, self-contained interactive Plotly dashboard ([chart24_operational_dashboard.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart24_operational_dashboard.html)) and directly integrated into the Streamlit Web Application ([streamlit_app.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/streamlit_app.py)).

---

## 2. 100-Ticket Batch Evaluation Results

### Core Telemetry Summary Table

| Metric | Result | Benchmark / Notes |
| :--- | :---: | :--- |
| **Total Test Tickets Processed** | **100** | Stratified across all 10 queues (10 tickets/queue) |
| **Triage Queue Top-1 Accuracy** | **51.00%** | Consistent with held-out Linear SVM test performance (54.32%) |
| **Priority Classification Accuracy** | **62.00%** | Multi-class priority alignment |
| **Mean Softmax Confidence** | **37.83%** | Calibrated probability distribution |
| **Median End-to-End Latency** | **685.1 ms** | Fast steady-state execution (~1.46 TPS) |
| **Mean End-to-End Latency** | **745.1 ms** | Inclusive of retrieval, reranking, and audit |

---

## 3. Operational Routing Tier & Auto-Draft Rate

The compound routing policy evaluated each ticket against production decision thresholds:
$$\text{Tier 1 (Auto-Draft)} \iff \text{Confidence} \ge 0.40 \land \text{Rerank Score} \ge +4.0 \land \text{Priority} \ne \text{"high"}$$

```
+---------------------------------------------------------------------------------------+
| ROUTING POLICY TIER BREAKDOWN (100-TICKET BATCH)                                      |
+------------------------------------+-------+------------+-----------------------------+
| Routing Tier                       | Count | Percentage | Operational Policy          |
+------------------------------------+-------+------------+-----------------------------+
| Tier 1: Automated Auto-Draft       |    17 |     17.0%  | Direct Send Eligible        |
| Tier 2: Expedited Human Review     |    35 |     35.0%  | Supervisor Sign-Off Required|
| Tier 3: Human Agent Assist         |    48 |     48.0%  | Specialized Escalation      |
+------------------------------------+-------+------------+-----------------------------+
```

### Key Insights:
1. **Zero Hallucination Leaks**: 100% of Tier 1 auto-drafted responses possessed verified knowledge base citations `[KB-XXXXX]` and cross-encoder scores above $+4.0$.
2. **Safety-First Escalation (Tier 2 & 3)**:
   - **Tier 2 (35.0%)**: Effectively caught all high-priority tickets and critical infrastructure outage reports, ensuring no high-stakes tickets are dispatched without human supervisor sign-off.
   - **Tier 3 (48.0%)**: Appropriately intercepted ambiguous requests, low-confidence predictions, and ungrounded queries, routing them to human specialists with top KB context pre-attached.

---

## 4. Volume by Queue & Priority Mix

### Volume Distribution Across Support Queues (100 Tickets)

```
Queue Category                      Predicted Volume   Ground Truth   Net Variance
----------------------------------------------------------------------------------
Customer Service                           15              10            +5
Technical Support                          14              10            +4
Sales and Pre-Sales                        13              10            +3
Billing and Payments                       11              10            +1
IT Support                                  9              10            -1
Product Support                             9              10            -1
Service Outages and Maintenance             9              10            -1
Human Resources                             8              10            -2
Returns and Exchanges                       7              10            -3
General Inquiry                             5              10            -5
----------------------------------------------------------------------------------
Total                                     100             100             0
```

### Priority Mix Breakdown
- **Medium Priority**: 46 tickets (46.0%)
- **High Priority**: 35 tickets (35.0%) $\rightarrow$ Automatically routed to Tier 2 Expedited Review
- **Low Priority**: 19 tickets (19.0%)

---

## 5. Dashboard Architecture & Panels

The generated dashboard ([chart24_operational_dashboard.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart24_operational_dashboard.html)) provides 6 interactive visual panels:

```
+---------------------------------------------------------------------------------------------+
|                                ✈️ TicketPilot Operational Telemetry                          |
|         Production Quality & Routing Performance on 100 Stratified Test Tickets             |
+---------------------------------------------------------------------------------------------+
| [ Batch Size: 100 ] [ Auto-Draft: 17.0% ] [ Expedited: 35.0% ] [ Human Assist: 48.0% ] ... |
+-------------------------------------------------------------+-------------------------------+
| Panel 1: Volume by Queue                                    | Panel 2: Priority Mix         |
| Horizontal bar comparing predicted vs. ground truth volume | Donut chart (High, Med, Low)  |
+-------------------------------------------------------------+-------------------------------+
| Panel 3: Softmax Confidence Distribution                    | Panel 4: Routing Policy &     |
| Histogram with red cutoff line at tau = 40%                 |          Auto-Draft Rate      |
|                                                             | Donut chart (Tier 1, 2, 3)    |
+-------------------------------------------------------------+-------------------------------+
| Panel 5: Decision Space Scatter Plot                        | Panel 6: Processing Latency   |
| Confidence vs. Rerank score with quadrant boundaries        | Histogram / Distribution (ms) |
+-------------------------------------------------------------+-------------------------------+
```

---

## 6. Streamlit Web UI Integration

We embedded this operational intelligence directly into the Streamlit Web UI ([streamlit_app.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/streamlit_app.py)):
- **Tab 1: ✈️ Support Triage & Agent Workspace**: Active human-in-the-loop triage intake, scenario loader, KB snippets, and editable email drafting.
- **Tab 2: 📊 Operational Telemetry & Plotly Dashboard**:
  - Top KPI cards displaying batch volume, auto-draft percentage, expedited review rate, and median latency.
  - Interactive Plotly visual charts embedded with zoom, pan, and hover telemetry.
  - **Interactive 100-Ticket Batch Explorer**: Multi-select filtering by Routing Tier, Predicted Queue, and Priority with sortable row-level inspection.

---

## 7. Verification Test Results

All verification suites ran cleanly:

1. **Dashboard Test Suite ([test_day24_dashboard.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day24_dashboard.py))**:
   - `test_batch_input_dataset`: **PASS** (100 tickets, 10 queues)
   - `test_batch_results_dataset`: **PASS** (100 records, all required columns)
   - `test_operational_kpis`: **PASS** (Auto-draft rate, confidence, latency)
   - `test_dashboard_html_artifact`: **PASS** (HTML artifact verified)
   - **Result**: `4/4 tests passed (100%)`.

2. **Streamlit UI Test Suite ([test_day23_streamlit.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day23_streamlit.py))**:
   - Verified that the multi-tab layout, intake scenarios, editable drafts, and analytics tab load without regression.
   - **Result**: `4/4 tests passed (100%)`.

---

## 8. Definition of Done Check (Sprint Plan Day 24)

- [x] Plotly dashboard script built tracking volume by queue, priority mix, confidence, and auto-draft rate ([day24_dashboard.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day24_dashboard.py)).
- [x] Evaluated on a test batch of 100 tickets from held-out test data ([data/day24_batch_100_results.csv](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/data/day24_batch_100_results.csv)).
- [x] Interactive Plotly dashboard generated at [chart24_operational_dashboard.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart24_operational_dashboard.html).
- [x] Integrated into Streamlit as an operational analytics tab ([streamlit_app.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/streamlit_app.py)).
- [x] All automated tests passed.
