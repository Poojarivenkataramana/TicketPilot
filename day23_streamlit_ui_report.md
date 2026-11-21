# Sprint 5 Day 23: Interactive Streamlit Web UI & Human-in-the-Loop Workspace

**Project**: TicketPilot — Enterprise Customer Support AI Triage & Grounded RAG Assistant  
**Date**: November 21, 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

On Day 23, we delivered the primary interactive front-end application for TicketPilot: a enterprise-grade Streamlit web application ([streamlit_app.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/streamlit_app.py)).

The UI bridges autonomous machine learning with human-in-the-loop customer support operations. Support agents and supervisors can paste customer inquiries or select pre-loaded scenarios, view sub-second AI triage classifications, inspect grounded Knowledge Base citations with relevance scores, edit AI-drafted responses inline, and trigger 1-click dispatch or escalation—all synchronized with the production Decision Logger.

All tests passed with 100% pass rate in the headless verification suite ([test_day23_streamlit.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day23_streamlit.py)).

---

## 2. Web UI Architecture & Layout

The UI is built with a dual-pane workflow optimized for high-throughput support agents:

```
+---------------------------------------------------------------------------------------------------------+
|                                ✈️ TicketPilot Support Workspace                                          |
|         Autonomous customer ticket triage, hybrid vector retrieval, and grounded reply generation       |
+---------------------------------------------------+-----------------------------------------------------+
| 📥 LEFT PANE: Customer Ticket Intake              | 🎯 RIGHT PANE: Triage & Grounded Resolution         |
|                                                   |                                                     |
| [Select Scenario Dropdown                      ▼] | [ Predicted Queue ] [ Priority ] [ Confidence ]    |
|   - Scenario 1: Billing Duplicate Charge          |   Billing & Payments     HIGH             88.4%     |
|   - Scenario 2: Critical Cloud Outage             |                                                     |
|   - Scenario 3: Defective Hardware RMA            | [====================================] Progress bar |
|   - Scenario 4: Ambiguous Custom API Webhook      |                                                     |
|   - Scenario 5: Employee HR Portal Onboarding     | [✅ TIER 1 AUTO-DRAFT: Direct Send Eligible       ] |
|                                                   | Operational Action: Direct send eligible            |
| Ticket ID: [ TCK-8492                           ] |                                                     |
| Subject:   [ Duplicate charge on credit card    ] | 📚 Grounded Knowledge Base Sources (Expander)       |
| Message:   [ I was charged twice on my card...  ] |   ▼ [KB-10492] Billing and Payments (Relevance: +9.2)|
|                                                   |     • Problem snippet & verified solution text      |
| [🚀 Process & Generate Cited Reply              ] |                                                     |
+---------------------------------------------------+-----------------------------------------------------+
| ✍️ FULL WIDTH: Agent Response Workspace (Human-in-the-Loop)                                             |
| +-----------------------------------------------------------------------------------------------------+ |
| | Dear Customer,                                                                                      | |
| | Regarding duplicate subscription renewal charge on credit card:                                     | |
| | We have verified the transaction and initiated a refund for the duplicate charge [KB-10492].        | |
| | The funds will reflect in your account within 3 to 5 business days.                                 | |
| | Support Agent Note: Customer verified email address.                                                | |
| +-----------------------------------------------------------------------------------------------------+ |
| [✉️ Approve & Send to Customer]    [👥 Escalate to Specialist]    🔒 Permanent audit log synchronized  |
+---------------------------------------------------------------------------------------------------------+
| 📊 SIDEBAR: Active Thresholds (Tau=40%, Cutoff=+4.0) | Real-Time Telemetry (Logged, Auto-Draft %, Latency)|
+---------------------------------------------------------------------------------------------------------+
```

---

## 3. Core Features Delivered

### A. Pre-Loaded Test Scenarios
To enable instant executive demonstrations and regression testing without manual typing, the application features 5 pre-loaded scenarios covering distinct routing behaviors:
1. **Scenario 1 (Billing Duplicate Charge)**: High confidence + high relevance score $\rightarrow$ **Tier 1 Auto-Draft** eligible for direct dispatch.
2. **Scenario 2 (Critical Cloud Outage)**: Production database cluster timeout $\rightarrow$ **Tier 2 Expedited Review** requiring mandatory supervisor sign-off.
3. **Scenario 3 (Defective Hardware RMA)**: Hardware warranty replacement request $\rightarrow$ **Tier 1 Auto-Draft** with hardware KB citations.
4. **Scenario 4 (Ambiguous Integration)**: Proprietary ERP webhook question not in knowledge base $\rightarrow$ **Tier 3 Human Assist** triggering safe refusal guardrail.
5. **Scenario 5 (Employee HR Portal)**: Payroll sync question $\rightarrow$ **Tier 1 Auto-Draft** routed to HR queue.

### B. High-Performance Dual-Mode Backend
The UI supports seamless switching via the sidebar radio control:
- **Direct In-Process**: Uses `@st.cache_resource` to cache the neural models (`LinearSVC`, ChromaDB, Sentence Transformers, Cross-Encoder) directly in memory, yielding sub-second responses without network overhead.
- **FastAPI REST API**: Connects over HTTP to `http://127.0.0.1:8000/draft-reply` for decoupled microservice deployments. If the REST API is offline, the UI automatically falls back to in-process mode with zero user disruption.

### C. Grounded Sources Explorer
Top-matching articles from the 17,469-article indexed Knowledge Base are displayed in collapsible accordion expanders:
- **Article ID**: Verified training split identifier `[KB-XXXXX]`.
- **Cross-Encoder Score**: Color-coded relevance rating (Green $\ge +4.0$, Orange $0.0 \dots 4.0$, Red $< 0.0$).
- **Resolution Snippet**: Authoritative resolution text extracted directly from training data with test-leakage isolation.

### D. Human-in-the-Loop Editable Draft Workspace
- AI drafts are pre-populated into an editable multi-line text area with exact bracket citations (`[KB-XXXXX]`).
- Agents can modify wording, add customer-specific account notes, or adjust tone.
- **Approve & Send**: Logs the dispatch event to the permanent decision stream.
- **Escalate to Human Specialist**: Routes ambiguous or critical tickets to specialist teams with full context pre-attached.

---

## 4. Test Verification & Results

We validated the application with Streamlit's official headless test framework ([test_day23_streamlit.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day23_streamlit.py)):

| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_streamlit_initial_load` | Validates sidebar, inputs, layout, and absence of runtime exceptions | **PASS** (0 exceptions) |
| `test_streamlit_process_billing_scenario` | End-to-end triage of Scenario 1, verifies queue prediction, metrics, and draft generation | **PASS** (Billing queue, draft generated) |
| `test_streamlit_process_outage_scenario` | Validates Scenario 2 critical outage detection and routing tier classification | **PASS** (High priority identified) |
| `test_streamlit_human_in_the_loop_actions` | Validates custom ticket intake, draft modification, and action buttons | **PASS** (Approved & logged) |

**Overall Verification**: **4/4 Tests Passed (100%)**.

---

## 5. How to Run the Application

To launch the interactive web application locally:

```bash
# Activate virtual environment
source myenv/bin/activate

# Launch Streamlit web server
streamlit run streamlit_app.py
```

The application will be accessible at:
- **Local URL**: `http://localhost:8501`
- **Network URL**: `http://192.168.x.x:8501`

---

## 6. Definition of Done Check (Sprint Plan Day 23)

- [x] Streamlit UI built with ticket paste intake, predicted queue category, priority, confidence meter, KB sources expander, and editable response draft.
- [x] Pre-loaded test scenarios for immediate demonstration.
- [x] Integration with 3-tier routing policy and thread-safe Decision Logger.
- [x] End-to-end demo verified and tested cleanly in [test_day23_streamlit.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day23_streamlit.py).
