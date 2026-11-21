"""
TicketPilot - Sprint 4 Day 19: Granular Latency Profiling & Failure Analysis
---------------------------------------------------------------------------
Objective:
1. Benchmark end-to-end and stage-by-stage latency across all 60 evaluation tickets
   in data/eval_test_set_60.csv under steady-state conditions (with warm-up).
2. Measure high-resolution nanosecond timing for:
   - Triage Classification (TF-IDF + Linear SVM + Probability Confidence)
   - Dense Vector Retrieval (SentenceTransformer + ChromaDB)
   - Sparse Lexical Retrieval (BM25 Inverted Index)
   - Reciprocal Rank Fusion (RRF Candidate Assembly)
   - Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2 on top-15 candidates)
   - Grounded Synthesis & Safety Audit
   - Total End-to-End Pipeline Latency
3. Compute statistical percentiles: Mean, Std, Median (p50), p90, p95, p99, Min, Max.
4. Document failure cases, throughput, and hardware bottlenecks.
5. Export day19_latency_summary.csv, day19_latency_per_ticket.csv,
   and interactive Plotly visualization (charts/chart19_latency_breakdown.html).
"""

import os
import re
import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import pipeline components
from day15_end_to_end_pipeline import TicketPilotPipeline
from day14_reply_prompt import GroundingAuditor, ESCALATION_PHRASE


class LatencyProfiler:
    """
    High-resolution, steady-state latency profiler for TicketPilot.
    """

    def __init__(self, pipeline: TicketPilotPipeline):
        self.pipeline = pipeline

    def profile_ticket(self, ticket: dict) -> dict:
        """
        Profiles a single ticket with nanosecond-resolution timers across every sub-stage.
        """
        text = str(ticket.get("text", ticket.get("body", "")))
        t_global_start = time.perf_counter_ns()

        # 1. Triage Classification
        t0 = time.perf_counter_ns()
        X = self.pipeline.tfidf.transform([text])
        queue_probs = self.pipeline.queue_model.predict_proba(X)[0]
        max_idx = np.argmax(queue_probs)
        pred_queue = self.pipeline.queue_model.classes_[max_idx]
        confidence = float(queue_probs[max_idx])
        prio_probs = self.pipeline.priority_model.predict_proba(X)[0]
        pred_priority = self.pipeline.priority_model.classes_[np.argmax(prio_probs)]
        t_classify_ns = time.perf_counter_ns() - t0

        # 2. Dense Vector Retrieval (Embedding + ChromaDB Query)
        t0 = time.perf_counter_ns()
        dense_vec = self.pipeline.dense_model.encode(text).tolist()
        chroma_res = self.pipeline.collection.query(
            query_embeddings=[dense_vec],
            n_results=15,
            where={"queue": pred_queue} if pred_queue else None,
            include=["documents", "metadatas", "distances"]
        )
        dense_ranks = {}
        if chroma_res["ids"] and len(chroma_res["ids"][0]) > 0:
            for rank, kb_id in enumerate(chroma_res["ids"][0]):
                dense_ranks[kb_id] = rank + 1
        t_dense_ns = time.perf_counter_ns() - t0

        # 3. Sparse BM25 Retrieval
        t0 = time.perf_counter_ns()
        tokens = text.lower().split()
        bm25_scores = self.pipeline.bm25.get_scores(tokens)
        kb_matches = self.pipeline.kb_df[self.pipeline.kb_df["queue"] == pred_queue]
        matched_indices = kb_matches.index.tolist()
        scored_bm25 = [(self.pipeline.kb_df.iloc[idx]["kb_id"], bm25_scores[idx]) for idx in matched_indices]
        scored_bm25.sort(key=lambda x: x[1], reverse=True)
        top_bm25 = scored_bm25[:15]
        sparse_ranks = {kb_id: rank + 1 for rank, (kb_id, _) in enumerate(top_bm25)}
        t_bm25_ns = time.perf_counter_ns() - t0

        # 4. Reciprocal Rank Fusion (RRF)
        t0 = time.perf_counter_ns()
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
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:15]

        candidate_docs = []
        for kb_id, rrf_score in sorted_candidates:
            idx = self.pipeline.kb_id_to_idx.get(kb_id)
            if idx is not None:
                row = self.pipeline.kb_df.iloc[idx]
                candidate_docs.append({
                    "kb_id": kb_id,
                    "queue": row["queue"],
                    "priority": row["priority"],
                    "problem": row["problem"],
                    "resolution": row["solution"],
                    "rrf_score": rrf_score
                })
        t_rrf_ns = time.perf_counter_ns() - t0

        # 5. Cross-Encoder Reranking
        t0 = time.perf_counter_ns()
        if candidate_docs:
            pairs = [[text, f"{c['problem']} {c['resolution']}"] for c in candidate_docs]
            scores = self.pipeline.reranker.predict(pairs)
            for i, score in enumerate(scores):
                candidate_docs[i]["rerank_score"] = float(score)
            candidate_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            reranked_docs = candidate_docs[:3]
        else:
            reranked_docs = []
        t_rerank_ns = time.perf_counter_ns() - t0

        top_rerank_score = reranked_docs[0]["rerank_score"] if reranked_docs else -99.0
        top_kb_id = reranked_docs[0]["kb_id"] if reranked_docs else "None"

        # 6. Decision Gating, Grounded Drafting & Audit
        t0 = time.perf_counter_ns()
        should_escalate = (confidence < self.pipeline.confidence_threshold) or (top_rerank_score < self.pipeline.relevance_threshold)
        if should_escalate:
            draft = ESCALATION_PHRASE
            decision = "Escalated to Human Agent"
        else:
            draft = self.pipeline.generator.generate(ticket, reranked_docs)
            decision = "Auto-Drafted Cited Reply"

        audit = GroundingAuditor.audit_draft(
            draft=draft,
            retrieved_docs=reranked_docs,
            ticket_text=text,
            expected_refusal=should_escalate
        )
        t_draft_ns = time.perf_counter_ns() - t0

        t_total_ns = time.perf_counter_ns() - t_global_start

        # Check for failure cases
        is_failure = False
        failure_type = "None"
        if not should_escalate and pred_queue != ticket.get("queue"):
            is_failure = True
            failure_type = "Cascade Misrouting"
        elif not should_escalate and top_rerank_score < 2.0:
            is_failure = True
            failure_type = "Low Retrieval Relevance"
        elif not audit["overall_pass"]:
            is_failure = True
            failure_type = "Audit Non-Compliance"

        return {
            "eval_id": ticket.get("eval_id", "TICKET"),
            "queue": ticket.get("queue", "Unknown"),
            "pred_queue": pred_queue,
            "confidence": confidence,
            "decision": decision,
            "top_kb_id": top_kb_id,
            "rerank_score": top_rerank_score,
            "is_failure": is_failure,
            "failure_type": failure_type,
            # Millisecond conversions
            "t_classify_ms": t_classify_ns / 1e6,
            "t_dense_ms": t_dense_ns / 1e6,
            "t_bm25_ms": t_bm25_ns / 1e6,
            "t_rrf_ms": t_rrf_ns / 1e6,
            "t_rerank_ms": t_rerank_ns / 1e6,
            "t_draft_ms": t_draft_ns / 1e6,
            "t_total_ms": t_total_ns / 1e6
        }


def run_latency_profiling_benchmark():
    print("=" * 70)
    print("DAY 19: END-TO-END LATENCY BENCHMARK & FAILURE PROFILING (N=60)")
    print("=" * 70)

    # 1. Initialize Pipeline
    pipeline = TicketPilotPipeline(
        models_dir="models",
        kb_path="data/knowledge_base_clean.csv",
        chroma_path="chroma_db",
        relevance_threshold=-1.0,
        confidence_threshold=0.35
    )
    profiler = LatencyProfiler(pipeline)

    # 2. Load Evaluation Dataset
    eval_df = pd.read_csv("data/eval_test_set_60.csv")
    print(f"Loaded {len(eval_df)} tickets from data/eval_test_set_60.csv.")

    # 3. Warm-up runs (eliminate cold-start JIT & PyTorch loading artifacts)
    print("\nExecuting 3 warm-up runs to ensure steady-state measurement...")
    for w in range(3):
        sample_ticket = eval_df.iloc[w].to_dict()
        _ = profiler.profile_ticket(sample_ticket)
    print("Warm-up complete. Steady-state engaged.\n")

    # 4. Profile All 60 Tickets
    print("Profiling 60 test tickets...")
    records = []
    for idx, row in eval_df.iterrows():
        res = profiler.profile_ticket(row.to_dict())
        records.append(res)
        if (idx + 1) % 15 == 0 or (idx + 1) == len(eval_df):
            print(f"  Processed {idx + 1}/60 tickets... (Current Total Latency: {res['t_total_ms']:.1f} ms)")

    df_records = pd.DataFrame(records)

    # 5. Calculate Granular Statistical Percentiles
    stage_columns = [
        ("Triage Classification", "t_classify_ms"),
        ("Dense Vector Retrieval", "t_dense_ms"),
        ("Sparse BM25 Search", "t_bm25_ms"),
        ("Reciprocal Rank Fusion (RRF)", "t_rrf_ms"),
        ("Cross-Encoder Reranking", "t_rerank_ms"),
        ("Grounded Drafting & Audit", "t_draft_ms"),
        ("Total End-to-End Pipeline", "t_total_ms")
    ]

    summary_rows = []
    print("\n" + "=" * 90)
    print("DAY 19 OFFICIAL LATENCY PERCENTILES TABLE (N=60 TICKETS)")
    print("=" * 90)
    print(f"{'Pipeline Processing Stage':<30} | {'Mean (ms)':<10} | {'p50 (Med)':<10} | {'p90 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Share %':<8}")
    print("-" * 90)

    total_mean = df_records["t_total_ms"].mean()

    for stage_name, col_name in stage_columns:
        vals = df_records[col_name].values
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        min_val = np.min(vals)
        max_val = np.max(vals)
        share_pct = (mean_val / total_mean) * 100 if stage_name != "Total End-to-End Pipeline" else 100.0

        summary_rows.append({
            "Stage": stage_name,
            "Mean_ms": round(mean_val, 2),
            "Std_ms": round(std_val, 2),
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "Min_ms": round(min_val, 2),
            "Max_ms": round(max_val, 2),
            "Share_pct": round(share_pct, 1)
        })

        print(f"{stage_name:<30} | {mean_val:>8.2f} ms | {p50:>8.2f} ms | {p90:>8.2f} ms | {p95:>8.2f} ms | {p99:>8.2f} ms | {share_pct:>6.1f}%")

    print("=" * 90)

    # 6. Failure Case Analysis
    failures = df_records[df_records["is_failure"]]
    failure_count = len(failures)
    failure_rate = (failure_count / len(df_records)) * 100
    print(f"\nFAILURE CASE AUDIT: {failure_count}/60 tickets flagged ({failure_rate:.1f}%)")
    if failure_count > 0:
        print("Failure breakdown by type:")
        print(failures["failure_type"].value_counts())
        print("\nFlagged tickets:")
        for _, f in failures.iterrows():
            print(f"  - [{f['eval_id']}] {f['queue']} -> Pred: {f['pred_queue']} (Conf: {f['confidence']:.1%}) | Type: {f['failure_type']}")
    else:
        print("Zero failure cases encountered under calibrated routing!")

    throughput_tps = 1000.0 / total_mean
    print(f"\nPipeline Throughput: {throughput_tps:.2f} tickets/sec (single CPU core / Metal GPU)")

    # 7. Export CSV files
    df_records.to_csv("day19_latency_per_ticket.csv", index=False)
    print("Saved per-ticket latency records to day19_latency_per_ticket.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("day19_latency_summary.csv", index=False)
    print("Saved official latency summary table to day19_latency_summary.csv")

    # 8. Generate Visual Dashboard
    generate_latency_chart(summary_df, df_records, output_path="charts/chart19_latency_breakdown.html")

    return summary_df, df_records


def generate_latency_chart(summary_df: pd.DataFrame, per_ticket_df: pd.DataFrame, output_path: str = "charts/chart19_latency_breakdown.html"):
    """
    Generates an interactive Plotly dashboard visualizing latency percentiles and stage runtime shares.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Filter out total from stage breakdown
    sub_stages = summary_df[summary_df["Stage"] != "Total End-to-End Pipeline"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Latency Breakdown by Stage (% of Total Runtime)",
            "Percentile Latency Distribution (p50, p90, p95, p99)"
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        horizontal_spacing=0.15
    )

    # 1. Pie / Donut Chart of Latency Shares
    colors = ["#3b82f6", "#06b6d4", "#f59e0b", "#8b5cf6", "#ef4444", "#10b981"]
    fig.add_trace(
        go.Pie(
            labels=sub_stages["Stage"],
            values=sub_stages["Mean_ms"],
            hole=0.45,
            marker=dict(colors=colors, line=dict(color="#0f172a", width=1.5)),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        ),
        row=1, col=1
    )

    # 2. Grouped Bar Chart of Percentiles
    percentiles = ["p50_ms", "p90_ms", "p95_ms", "p99_ms"]
    p_names = ["p50 (Median)", "p90", "p95", "p99"]
    p_colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    for i, p_col in enumerate(percentiles):
        fig.add_trace(
            go.Bar(
                name=p_names[i],
                x=sub_stages["Stage"],
                y=sub_stages[p_col],
                text=[f"{v:.1f}ms" for v in sub_stages[p_col]],
                textposition="outside",
                marker_color=p_colors[i]
            ),
            row=1, col=2
        )

    fig.update_layout(
        title="<b>TicketPilot Day 19: Granular Latency Profiling Benchmark (N=60 Tickets)</b><br><sup>Steady-State Timing Breakdown across Triage, Hybrid Retrieval, Reranking, and Drafting</sup>",
        template="plotly_white",
        height=540,
        font=dict(family="Inter, sans-serif", size=12),
        barmode="group",
        yaxis2=dict(title="Latency (ms)", range=[0, max(sub_stages["p99_ms"]) * 1.3])
    )

    fig.write_html(output_path)
    print(f"Chart generated at {output_path}")


if __name__ == "__main__":
    run_latency_profiling_benchmark()
