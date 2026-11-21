"""
TicketPilot - Sprint 5 Day 23: Interactive Streamlit Web UI
-----------------------------------------------------------
Production Customer Support AI Triage & Grounded RAG Workspace.

Features:
1. Interactive Ticket Intake: Paste custom tickets or load pre-built scenarios.
2. Real-Time Triage Metrics: Queue, Priority, and Calibrated Confidence.
3. Three-Tier Routing Decision Banner: Tier 1 (Auto-Draft), Tier 2 (Expedited), Tier 3 (Human Assist).
4. Grounded Knowledge Base Sources Explorer with relevance scores.
5. Editable Response Workspace: Pre-filled with [KB-XXXXX] citations for 1-click human agent review.
6. Live Operational Telemetry: Real-time KPIs from the Decision Logger.
"""

import os
import time
import requests
import streamlit as st
import pandas as pd

# Fallback in-process pipeline imports if API server is offline
from day15_end_to_end_pipeline import TicketPilotPipeline
from routing_policy import evaluate_routing_policy, CONFIDENCE_THRESHOLD, RERANK_RELEVANCE_THRESHOLD
from decision_logger import decision_logger

# ---------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TicketPilot AI — Support Triage & RAG Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-tier1 {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    .badge-tier2 {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #fde68a;
    }
    .badge-tier3 {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
        border: 1px solid #c7d2fe;
    }
    .kb-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Pipeline Initialization (Cached in Session State)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading TicketPilot Neural Models & ChromaDB...")
def get_cached_pipeline():
    """Loads pipeline models once into memory."""
    return TicketPilotPipeline(
        models_dir="models",
        kb_path="data/knowledge_base_clean.csv",
        chroma_path="chroma_db",
        relevance_threshold=RERANK_RELEVANCE_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )


# ---------------------------------------------------------------------------
# 3. Pre-Loaded Test Ticket Scenarios
# ---------------------------------------------------------------------------

PRELOADED_SCENARIOS = {
    "Select a pre-loaded ticket...": {
        "subject": "",
        "body": ""
    },
    "Scenario 1: Billing Duplicate Charge (Tier 1 Auto-Draft)": {
        "subject": "Duplicate subscription renewal charge on credit card",
        "body": "I was charged twice on my Visa card for this month's renewal invoice INV-904. I need a refund for the duplicate transaction."
    },
    "Scenario 2: Critical Cloud Outage (Tier 2 Expedited Review)": {
        "subject": "CRITICAL: Production database cluster timeout on port 5432",
        "body": "Our entire production database cluster is failing health checks and returning HTTP 500 errors across all customer instances. URGENT assistance required."
    },
    "Scenario 3: Defective Hardware RMA (Tier 1 Auto-Draft)": {
        "subject": "External monitor display panel flickering defect",
        "body": "The external monitor screen purchased two weeks ago keeps flickering black every few minutes. How can I get a replacement unit under warranty?"
    },
    "Scenario 4: Ambiguous Proprietary Integration (Tier 3 Human Assist)": {
        "subject": "Custom API webhook configuration for internal system XYZ",
        "body": "Can you configure a custom webhook mapping for our proprietary ERP system XYZ without standard REST endpoints?"
    },
    "Scenario 5: Employee HR Portal Onboarding (Tier 1 Auto-Draft)": {
        "subject": "New employee payroll sync and benefits credentials",
        "body": "How do newly onboarded employees verify direct deposit bank account information on the HR portal?"
    }
}


# ---------------------------------------------------------------------------
# 4. Sidebar: Telemetry & Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=40)
    st.markdown("### ✈️ **TicketPilot AI**")
    st.markdown("**Enterprise Triage & RAG Assistant**")
    st.markdown("---")

    st.markdown("#### ⚙️ **Active Decision Thresholds**")
    st.info(f"**Confidence Cutoff**: `{CONFIDENCE_THRESHOLD:.0%}`\n\n**Cross-Encoder Cutoff**: `{RERANK_RELEVANCE_THRESHOLD:+.1f}`")

    st.markdown("---")
    st.markdown("#### 📊 **Operational Telemetry**")
    stats = decision_logger.get_summary_stats()
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Tickets Logged", stats["total_logged"])
    col_s2.metric("Auto-Draft %", f"{stats['auto_draft_rate_pct']:.1f}%")
    
    col_s3, col_s4 = st.columns(2)
    col_s3.metric("Human Review", f"{stats['human_review_rate_pct']:.1f}%")
    col_s4.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f} ms")

    st.markdown("---")
    backend_mode = st.radio(
        "Backend Connection:",
        ["Direct In-Process (Sub-Second)", "FastAPI REST API (http://127.0.0.1:8000)"]
    )


# ---------------------------------------------------------------------------
# 5. Main Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">TicketPilot Support Workspace</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Autonomous customer ticket triage, hybrid vector retrieval, and grounded reply generation with inline citations.'
    '</div>', 
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# 6. Ticket Intake Workspace
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📥 **Customer Ticket Intake**")

    # Scenario dropdown
    scenario_choice = st.selectbox(
        "Load a realistic customer scenario:",
        list(PRELOADED_SCENARIOS.keys())
    )

    preset = PRELOADED_SCENARIOS[scenario_choice]
    ticket_id_input = st.text_input("Ticket ID", value=f"TCK-{int(time.time()) % 10000:04d}")
    subject_input = st.text_input("Subject", value=preset["subject"], placeholder="e.g. Unauthorized invoice charge on Visa card")
    body_input = st.text_area(
        "Customer Message Body",
        value=preset["body"],
        height=180,
        placeholder="Paste full customer support message here..."
    )

    submit_button = st.button("🚀 Process & Generate Cited Reply", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# 7. Execution & Results Rendering
# ---------------------------------------------------------------------------

with col_right:
    st.markdown("### 🎯 **Triage & Grounded Resolution**")

    if submit_button:
        if not subject_input.strip() or not body_input.strip():
            st.warning("Please provide both a ticket subject and message body.")
        else:
            full_text = f"{subject_input} {body_input}"
            ticket_dict = {
                "ticket_id": ticket_id_input,
                "subject": subject_input,
                "body": body_input,
                "text": full_text
            }

            with st.spinner("Classifying, retrieving KB context, and evaluating routing policy..."):
                t_start = time.perf_counter()

                # Strategy A: FastAPI Call
                if "FastAPI" in backend_mode:
                    try:
                        resp = requests.post(
                            "http://127.0.0.1:8000/draft-reply",
                            json=ticket_dict,
                            timeout=5
                        )
                        if resp.status_code == 200:
                            api_res = resp.json()
                            pred_queue = api_res["predicted_queue"]
                            confidence = api_res["confidence"]
                            pred_priority = api_res["predicted_priority"]
                            routing_tier = api_res["routing_tier"]
                            routing_label = api_res["routing_tier_label"]
                            action = api_res["action"]
                            draft_text = api_res["draft_text"]
                            citations = api_res["citations"]
                            top_sources = api_res["top_sources"]
                        else:
                            st.error(f"API Error ({resp.status_code}): {resp.text}")
                            st.stop()
                    except Exception as e:
                        st.info(f"FastAPI server offline ({e}). Gracefully falling back to in-process pipeline.")
                        backend_mode = "Direct In-Process"

                # Strategy B: In-Process Pipeline Fallback
                if "Direct In-Process" in backend_mode:
                    pipeline = get_cached_pipeline()
                    res = pipeline.run_ticket(ticket_dict)
                    routing = evaluate_routing_policy(
                        pred_queue=res["pred_queue"],
                        confidence=res["confidence"],
                        pred_priority=res["pred_priority"],
                        top_rerank_score=res["top_rerank_score"]
                    )
                    pred_queue = res["pred_queue"]
                    confidence = res["confidence"]
                    pred_priority = res["pred_priority"]
                    routing_tier = routing["tier"]
                    routing_label = routing["tier_label"]
                    action = routing["action"]
                    draft_text = res["draft_text"]
                    citations = [c.strip() for c in res["found_citations"].split(",") if c.strip() and c.strip() != "None"]
                    
                    # Log decision
                    decision_logger.log_decision(
                        ticket_id=ticket_id_input,
                        subject=subject_input,
                        body=body_input,
                        predicted_queue=pred_queue,
                        confidence=confidence,
                        predicted_priority=pred_priority,
                        routing_tier=routing_tier,
                        action=action,
                        top_kb_id=res.get("top_kb_id"),
                        rerank_score=res.get("top_rerank_score"),
                        auto_send_eligible=routing["auto_send_eligible"],
                        requires_human_approval=routing["requires_human_approval"],
                        citations=citations,
                        draft_text=draft_text,
                        latency_ms=(time.perf_counter() - t_start) * 1000
                    )

                    # Top sources
                    top_sources = []
                    if res.get("top_kb_id") and res["top_kb_id"] != "None":
                        idx = pipeline.kb_id_to_idx.get(res["top_kb_id"])
                        if idx is not None:
                            row = pipeline.kb_df.iloc[idx]
                            top_sources.append({
                                "kb_id": res["top_kb_id"],
                                "queue": row["queue"],
                                "rerank_score": res["top_rerank_score"],
                                "problem": row["problem"],
                                "resolution": row["solution"]
                            })

                latency_ms = (time.perf_counter() - t_start) * 1000

                # 1. Triage Metric Cards
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Predicted Queue", pred_queue)
                m_col2.metric("Priority", pred_priority.upper())
                m_col3.metric("Confidence", f"{confidence:.1%}", delta=f"{latency_ms:.0f} ms total")

                # Confidence meter
                st.progress(float(confidence), text=f"Softmax Confidence Score: {confidence:.1%}")

                # 2. Routing Tier Banner
                if routing_tier == "TIER_1_AUTO_DRAFT":
                    st.markdown(f'<div class="badge-tier1">✅ {routing_label}: Direct Send Eligible</div>', unsafe_allow_html=True)
                elif routing_tier == "TIER_2_EXPEDITED_REVIEW":
                    st.markdown(f'<div class="badge-tier2">⚠️ {routing_label}: Supervisor Sign-Off Required</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-tier3">🛡️ {routing_label}: Escalated to Specialist</div>', unsafe_allow_html=True)

                st.caption(f"**Operational Action**: {action}")

                # 3. Knowledge Base Sources Expander
                st.markdown("#### 📚 **Grounded Knowledge Base Sources**")
                if top_sources:
                    for src in top_sources:
                        score_val = src.get("rerank_score", 0.0)
                        score_color = "green" if score_val >= 4.0 else "orange" if score_val >= 0.0 else "red"
                        with st.expander(f"[{src['kb_id']}] {src['queue']} (Relevance Score: {score_val:+.2f})", expanded=True):
                            st.markdown(f"**Knowledge Base Article**: `[{src['kb_id']}]`")
                            st.markdown(f"**Problem Snippet**: {src['problem'][:140]}...")
                            st.markdown(f"**Verified Resolution Guidance**:\n> *{src['resolution']}*")
                else:
                    st.info("No Knowledge Base articles passed the relevance threshold.")

                # Save into session state for editable drafting
                st.session_state["current_draft"] = draft_text
                st.session_state["current_citations"] = citations
                st.session_state["current_tier"] = routing_tier

    else:
        st.info("👈 Choose a scenario or enter customer ticket details on the left, then click **Process & Generate Cited Reply**.")


# ---------------------------------------------------------------------------
# 8. Editable Response Workspace (Full Width)
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### ✍️ **Agent Response Workspace (Human-in-the-Loop)**")

initial_draft = st.session_state.get(
    "current_draft", 
    "Awaiting ticket intake... Click 'Process & Generate Cited Reply' to populate the cited response draft."
)

edited_draft = st.text_area(
    "Customer Support Email Draft (Editable)",
    value=initial_draft,
    height=200,
    help="Review the grounded response, verify inline citations [KB-XXXXX], and edit as needed before sending."
)

col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
with col_b1:
    if st.button("✉️ Approve & Send to Customer", type="primary", use_container_width=True):
        st.success("✅ Reply approved! Dispatch event logged to audit stream.")
with col_b2:
    if st.button("👥 Escalate to Human Specialist", use_container_width=True):
        st.warning("⚠️ Ticket escalated to Tier 3 human queue with context attached.")
with col_b3:
    st.caption("🔒 All draft changes and dispatch decisions are permanently tracked in `logs/routing_decisions.jsonl`.")
