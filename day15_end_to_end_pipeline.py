"""
TicketPilot - Sprint 3 Day 15: End-to-End Support Ticket Pipeline
-----------------------------------------------------------------
Pipeline Architecture:
1. Raw Customer Ticket Ingestion (Subject + Body)
2. Triage Classification: TF-IDF + Classifier -> Queue & Priority + Confidence
3. Metadata-Filtered Hybrid Retrieval: ChromaDB (Dense) + BM25 (Sparse) with RRF
4. Deep Cross-Encoder Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2
5. Guardrail Decision Engine: Threshold-based Auto-Draft vs. Safe Escalation
6. Grounded Reply Drafting: Professional support email with [KB-XXXXX] citations
7. Grounding & Citation Audit: Automated anti-hallucination verification
8. 20-Ticket Test Suite: Detailed analysis of what works and what fails
"""

import os
import re
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import prompt & guardrail components from Day 14
from day14_reply_prompt import (
    ESCALATION_PHRASE,
    GroundingAuditor,
    TicketPilotGroundedGenerator
)


class TicketPilotPipeline:
    """
    Production End-to-End Ticket Triage and Grounded RAG Assistant.
    """

    def __init__(
        self,
        models_dir: str = "models",
        kb_path: str = "data/knowledge_base_clean.csv",
        chroma_path: str = "chroma_db",
        relevance_threshold: float = -1.0,
        confidence_threshold: float = 0.35
    ):
        print("=" * 70)
        print("INITIALIZING TICKETPILOT END-TO-END PIPELINE")
        print("=" * 70)

        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold

        # 1. Load Classification Models
        print("1/5 Loading Triage Models (TF-IDF + Classifiers)...")
        self.tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        self.queue_model = joblib.load(os.path.join(models_dir, "baseline_queue_model.pkl"))
        self.priority_model = joblib.load(os.path.join(models_dir, "baseline_priority_model.pkl"))
        self.svm_queue = joblib.load(os.path.join(models_dir, "best_classical_linear_svm_queue.pkl"))
        print(f"   Loaded TF-IDF ({len(self.tfidf.get_feature_names_out()):,} features).")
        print(f"   Queue classes: {len(self.queue_model.classes_)} | Priority classes: {len(self.priority_model.classes_)}")

        # 2. Connect to ChromaDB
        print("2/5 Connecting to ChromaDB Vector Store...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection("ticketpilot_knowledge_base")
        print(f"   ChromaDB Collection active: {self.collection.count():,} indexed vectors.")

        # 3. Dense Embedding Model
        print("3/5 Loading Dense Sentence Embedder (all-MiniLM-L6-v2)...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 4. Sparse BM25 Index
        print("4/5 Loading Knowledge Base & Building BM25 Inverted Index...")
        self.kb_df = pd.read_csv(kb_path)
        self.corpus = [
            f"{str(r['problem'])} {str(r['solution'])}".lower().split()
            for _, r in self.kb_df.iterrows()
        ]
        self.bm25 = BM25Okapi(self.corpus)
        self.kb_id_to_idx = {r["kb_id"]: i for i, r in self.kb_df.iterrows()}
        print(f"   BM25 Index ready: {len(self.kb_df):,} articles indexed.")

        # 5. Cross-Encoder & Generator
        print("5/5 Loading Cross-Encoder Reranker & Grounded Generator...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.generator = TicketPilotGroundedGenerator(relevance_threshold=self.relevance_threshold)
        print("Pipeline initialization complete!\n")

    def classify_ticket(self, text: str) -> Dict[str, Any]:
        """
        Step 1: Predict Queue and Priority using TF-IDF and Classifiers.
        """
        X = self.tfidf.transform([text])
        
        # Queue prediction with probability confidence
        queue_probs = self.queue_model.predict_proba(X)[0]
        max_idx = np.argmax(queue_probs)
        pred_queue = self.queue_model.classes_[max_idx]
        confidence = float(queue_probs[max_idx])

        # Priority prediction
        prio_probs = self.priority_model.predict_proba(X)[0]
        pred_priority = self.priority_model.classes_[np.argmax(prio_probs)]

        return {
            "pred_queue": pred_queue,
            "confidence": confidence,
            "pred_priority": pred_priority
        }

    def hybrid_retrieve(self, query: str, target_queue: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Step 2: Scoped Hybrid Retrieval (Dense ChromaDB + Sparse BM25 + RRF).
        Filtered strictly to predicted target_queue.
        """
        # Dense Retrieval
        dense_vec = self.dense_model.encode(query).tolist()
        where_filter = {"queue": target_queue} if target_queue else None

        chroma_res = self.collection.query(
            query_embeddings=[dense_vec],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        dense_ranks = {}
        if chroma_res["ids"] and len(chroma_res["ids"][0]) > 0:
            for rank, kb_id in enumerate(chroma_res["ids"][0]):
                dense_ranks[kb_id] = rank + 1

        # Sparse BM25 Retrieval
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        
        # Filter BM25 candidates by target_queue
        kb_matches = self.kb_df[self.kb_df["queue"] == target_queue]
        matched_indices = kb_matches.index.tolist()
        
        scored_bm25 = [(self.kb_df.iloc[idx]["kb_id"], bm25_scores[idx]) for idx in matched_indices]
        scored_bm25.sort(key=lambda x: x[1], reverse=True)
        top_bm25 = scored_bm25[:top_k]

        sparse_ranks = {kb_id: rank + 1 for rank, (kb_id, _) in enumerate(top_bm25)}

        # Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        rrf_scores = {}
        k_rrf = 60
        for kb_id in all_candidate_ids:
            score = 0.0
            if kb_id in dense_ranks:
                score += 1.0 / (k_rrf + dense_ranks[kb_id])
            if kb_id in sparse_ranks:
                score += 1.0 / (k_rrf + sparse_ranks[kb_id])
            rrf_scores[kb_id] = score

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Assemble Candidate Pool
        candidate_docs = []
        for kb_id, rrf_score in sorted_candidates:
            idx = self.kb_id_to_idx.get(kb_id)
            if idx is not None:
                row = self.kb_df.iloc[idx]
                candidate_docs.append({
                    "kb_id": kb_id,
                    "queue": row["queue"],
                    "priority": row["priority"],
                    "problem": row["problem"],
                    "resolution": row["solution"],
                    "rrf_score": rrf_score
                })

        return candidate_docs

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Step 3: Cross-Encoder Reranking with ms-marco-MiniLM-L-6-v2.
        """
        if not candidates:
            return []

        pairs = [[query, f"{c['problem']} {c['resolution']}"] for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

    def run_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end processing for a single ticket.
        """
        t_start = time.time()
        text = str(ticket.get("text", ticket.get("body", "")))

        # 1. Triage Classification
        t0 = time.time()
        triage = self.classify_ticket(text)
        t_classify = (time.time() - t0) * 1000

        pred_queue = triage["pred_queue"]
        confidence = triage["confidence"]
        pred_priority = triage["pred_priority"]

        # 2. Hybrid Retrieval (Scoped by Predicted Queue)
        t0 = time.time()
        candidates = self.hybrid_retrieve(query=text, target_queue=pred_queue, top_k=15)
        t_retrieve = (time.time() - t0) * 1000

        # 3. Cross-Encoder Reranking
        t0 = time.time()
        reranked_docs = self.rerank_candidates(query=text, candidates=candidates, top_n=3)
        t_rerank = (time.time() - t0) * 1000

        top_rerank_score = reranked_docs[0]["rerank_score"] if reranked_docs else -99.0
        top_kb_id = reranked_docs[0]["kb_id"] if reranked_docs else "None"

        # 4. Guardrail Decision Engine
        # Escalate if confidence < 0.20 OR rerank score < relevance_threshold
        t0 = time.time()
        should_escalate = (confidence < self.confidence_threshold) or (top_rerank_score < self.relevance_threshold)
        
        if should_escalate:
            draft = ESCALATION_PHRASE
            decision = "Escalated to Human Agent"
        else:
            draft = self.generator.generate(ticket, reranked_docs)
            decision = "Auto-Drafted Cited Reply"
        t_draft = (time.time() - t0) * 1000

        total_latency = (time.time() - t_start) * 1000

        # 5. Grounding & Citation Audit
        audit = GroundingAuditor.audit_draft(
            draft=draft,
            retrieved_docs=reranked_docs,
            ticket_text=text,
            expected_refusal=should_escalate
        )

        # 6. Evaluation against ground truth
        true_queue = ticket.get("queue", "Unknown")
        true_priority = ticket.get("priority", "Unknown")
        queue_correct = (pred_queue == true_queue)
        priority_correct = (pred_priority == true_priority)

        # 7. Diagnostic Assessment of What Works / What Fails
        if queue_correct and not should_escalate and audit["overall_pass"]:
            diagnosis = "SUCCESS: Optimal triage, high-relevance retrieval, grounded reply drafted."
            status = "SUCCESS"
        elif queue_correct and should_escalate:
            diagnosis = "SAFE ESCALATION: Correct queue, but low retrieval relevance safely escalated to human."
            status = "SAFE_ESCALATION"
        elif not queue_correct and should_escalate:
            diagnosis = "SAFE ESCALATION: Queue misclassified, but low relevance safely escalated without hallucinating."
            status = "SAFE_ESCALATION"
        elif not queue_correct and not should_escalate:
            diagnosis = "CASCADE ERROR: Queue misclassified, driving retrieval into wrong department pool."
            status = "CASCADE_ERROR"
        else:
            diagnosis = "SUBOPTIMAL: Low grounding or formatting mismatch."
            status = "SUBOPTIMAL"

        return {
            "ticket_id": ticket.get("ticket_id", "TICKET"),
            "subject": ticket.get("subject", ""),
            "true_queue": true_queue,
            "pred_queue": pred_queue,
            "queue_correct": queue_correct,
            "confidence": confidence,
            "true_priority": true_priority,
            "pred_priority": pred_priority,
            "priority_correct": priority_correct,
            "top_kb_id": top_kb_id,
            "top_rerank_score": top_rerank_score,
            "decision": decision,
            "draft_text": draft,
            "is_escalated": audit["is_escalated"],
            "found_citations": ", ".join(audit["found_citations"]) if audit["found_citations"] else "None",
            "citation_validity": audit["citation_validity"],
            "grounding_ratio": audit["grounding_ratio"],
            "audit_pass": audit["overall_pass"],
            "status": status,
            "diagnosis": diagnosis,
            "t_classify_ms": t_classify,
            "t_retrieve_ms": t_retrieve,
            "t_rerank_ms": t_rerank,
            "t_draft_ms": t_draft,
            "total_latency_ms": total_latency
        }


def run_day15_benchmark(pipeline: TicketPilotPipeline, n_tickets: int = 20) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs the full pipeline on 20 stratified test tickets from data/test.csv.
    """
    test_df = pd.read_csv("data/test.csv")
    
    # Stratified selection: 2 tickets from each of the 10 queues
    sample_df = (
        test_df[test_df["subject"].notna()]
        .groupby("queue", as_index=False)
        .head(2)
        .reset_index(drop=True)
    )

    print(f"\nRunning Day 15 End-to-End Pipeline on {len(sample_df)} test tickets...")
    results = []

    for i, row in sample_df.iterrows():
        ticket_id = f"TEST-{i+1:04d}"
        ticket_data = {
            "ticket_id": ticket_id,
            "subject": row["subject"],
            "body": row["body"],
            "text": row["text"],
            "queue": row["queue"],
            "priority": row["priority"]
        }

        res = pipeline.run_ticket(ticket_data)
        results.append(res)

        print(f"[{ticket_id}] True Queue: {res['true_queue']} | Pred: {res['pred_queue']} (Conf: {res['confidence']:.1%})")
        print(f"   -> Top KB: {res['top_kb_id']} (Score: {res['top_rerank_score']:+.2f}) | Decision: {res['decision']}")
        print(f"   -> Status: {res['status']} | Latency: {res['total_latency_ms']:.1f} ms\n")

    results_df = pd.DataFrame(results)

    # Compute Summary Statistics
    total = len(results_df)
    queue_acc = (results_df["queue_correct"].sum() / total) * 100
    prio_acc = (results_df["priority_correct"].sum() / total) * 100
    auto_draft_rate = ((results_df["decision"] == "Auto-Drafted Cited Reply").sum() / total) * 100
    escalation_rate = ((results_df["decision"] == "Escalated to Human Agent").sum() / total) * 100
    success_rate = ((results_df["status"] == "SUCCESS").sum() / total) * 100
    safe_escalation_rate = ((results_df["status"] == "SAFE_ESCALATION").sum() / total) * 100
    cascade_error_rate = ((results_df["status"] == "CASCADE_ERROR").sum() / total) * 100
    citation_acc = (results_df["citation_validity"].sum() / total) * 100
    avg_latency = results_df["total_latency_ms"].mean()

    summary = {
        "total_tickets": total,
        "queue_accuracy_pct": queue_acc,
        "priority_accuracy_pct": prio_acc,
        "auto_draft_rate_pct": auto_draft_rate,
        "escalation_rate_pct": escalation_rate,
        "success_rate_pct": success_rate,
        "safe_escalation_rate_pct": safe_escalation_rate,
        "cascade_error_rate_pct": cascade_error_rate,
        "citation_accuracy_pct": citation_acc,
        "avg_latency_ms": avg_latency
    }

    print("=" * 70)
    print("DAY 15 END-TO-END PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total Test Tickets Processed:  {total}")
    print(f"Queue Triage Accuracy:         {queue_acc:.1f}%")
    print(f"Priority Triage Accuracy:      {prio_acc:.1f}%")
    print(f"Auto-Drafted Cited Replies:    {auto_draft_rate:.1f}%")
    print(f"Escalated to Human Agents:     {escalation_rate:.1f}%")
    print(f"Clean End-to-End Success Rate: {success_rate:.1f}%")
    print(f"Safe Escalations (Prevented):  {safe_escalation_rate:.1f}%")
    print(f"Cascade Errors (Wrong Queue):  {cascade_error_rate:.1f}%")
    print(f"Citation Integrity:            {citation_acc:.1f}%")
    print(f"Average Pipeline Latency:      {avg_latency:.1f} ms")
    print("=" * 70)

    return results_df, summary


def generate_pipeline_chart(results_df: pd.DataFrame, output_path: str = "charts/chart15_pipeline_performance.html"):
    """
    Generates an interactive Plotly dashboard visualizing pipeline outcomes,
    status distribution, and latency breakdown.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "End-to-End Decision & Operational Status Breakdown",
            "Latency Breakdown per Processing Stage (ms)"
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        horizontal_spacing=0.15
    )

    # 1. Status Breakdown (Donut Chart)
    status_counts = results_df["status"].value_counts()
    status_labels = [s.replace("_", " ").title() for s in status_counts.index]
    color_map = {
        "SUCCESS": "#10b981",          # Green
        "SAFE_ESCALATION": "#6366f1",  # Indigo
        "CASCADE_ERROR": "#ef4444",    # Red
        "SUBOPTIMAL": "#f59e0b"        # Amber
    }
    chart_colors = [color_map.get(s, "#94a3b8") for s in status_counts.index]

    fig.add_trace(
        go.Pie(
            labels=status_labels,
            values=status_counts.values,
            hole=0.45,
            marker=dict(colors=chart_colors, line=dict(color="#0f172a", width=1.5)),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        ),
        row=1, col=1
    )

    # 2. Latency Breakdown per Stage
    stages = ["Classification", "Hybrid Retrieval", "Cross-Encoder Rerank", "Grounded Drafting"]
    latencies = [
        results_df["t_classify_ms"].mean(),
        results_df["t_retrieve_ms"].mean(),
        results_df["t_rerank_ms"].mean(),
        results_df["t_draft_ms"].mean()
    ]
    bar_colors = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981"]

    fig.add_trace(
        go.Bar(
            x=stages,
            y=latencies,
            text=[f"{l:.1f} ms" for l in latencies],
            textposition="outside",
            marker=dict(color=bar_colors, line=dict(color="#0f172a", width=1.5)),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="<b>TicketPilot Day 15: End-to-End Pipeline Performance & Latency Audit</b><br><sup>Evaluating Triage, Hybrid Retrieval, Reranking, and Grounded Generation across 20 Test Tickets</sup>",
        template="plotly_white",
        height=520,
        font=dict(family="Inter, sans-serif", size=13),
        yaxis2=dict(title="Average Latency (ms)", range=[0, max(latencies) * 1.25])
    )

    fig.write_html(output_path)
    print(f"Chart generated at {output_path}")


if __name__ == "__main__":
    pipeline = TicketPilotPipeline(
        models_dir="models",
        kb_path="data/knowledge_base_clean.csv",
        chroma_path="chroma_db",
        relevance_threshold=-1.0,
        confidence_threshold=0.35
    )

    results_df, summary = run_day15_benchmark(pipeline, n_tickets=20)
    
    # Save CSV
    results_df.to_csv("day15_pipeline_run_results.csv", index=False)
    print("Run results saved to day15_pipeline_run_results.csv")

    # Generate Chart
    generate_pipeline_chart(results_df, output_path="charts/chart15_pipeline_performance.html")
