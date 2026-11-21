"""
TicketPilot - Sprint 4 Day 17: Retrieval Benchmark (Vector vs. Hybrid vs. Hybrid+Rerank)
----------------------------------------------------------------------------------------
Objective:
1. Run retrieval across all 60 evaluation tickets in data/eval_test_set_60.csv.
2. Benchmark three retrieval architectures:
   - Method 1: Plain Vector Search (all-MiniLM-L6-v2 in ChromaDB)
   - Method 2: Hybrid Search (ChromaDB Dense + BM25 Sparse with RRF)
   - Method 3: Hybrid Search + Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
3. Evaluate binary relevance (Grade >= 2) using Day 16 ground-truth relevance rules.
4. Calculate Hit Rate@1, Hit Rate@3, Hit Rate@5, MRR, and Latency for all 3 methods.
5. Export day17_retrieval_benchmark_results.csv, day17_retrieval_summary_table.csv,
   and interactive Plotly comparison chart (charts/chart17_retrieval_benchmark.html).
"""

import os
import re
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class RetrievalBenchmarkEngine:
    """
    Evaluates Plain Vector Search, Hybrid Search, and Hybrid + Reranking
    against the 60-ticket evaluation benchmark.
    """

    def __init__(
        self,
        kb_path: str = "data/knowledge_base_clean.csv",
        chroma_path: str = "chroma_db"
    ):
        print("=" * 70)
        print("INITIALIZING RETRIEVAL BENCHMARK ENGINE")
        print("=" * 70)

        # 1. Load Knowledge Base
        print("1/4 Loading Knowledge Base...")
        self.kb_df = pd.read_csv(kb_path)
        print(f"   Loaded {len(self.kb_df):,} articles.")

        # 2. Dense Vector Model & ChromaDB
        print("2/4 Loading Dense Embedder & ChromaDB...")
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection("ticketpilot_knowledge_base")
        print(f"   ChromaDB connected: {self.collection.count():,} vectors.")

        # 3. Sparse BM25 Index
        print("3/4 Building BM25 Inverted Index...")
        self.corpus = [
            f"{str(r['problem'])} {str(r['solution'])}".lower().split()
            for _, r in self.kb_df.iterrows()
        ]
        self.bm25 = BM25Okapi(self.corpus)
        self.kb_id_to_idx = {r["kb_id"]: i for i, r in self.kb_df.iterrows()}
        print(f"   BM25 index ready ({len(self.corpus):,} documents).")

        # 4. Cross-Encoder Reranker
        print("4/4 Loading Cross-Encoder: 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Engine Ready!\n")

    def search_plain_vector(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
        """
        Method 1: Plain Vector Search using ChromaDB (Dense Cosine Similarity).
        """
        t0 = time.time()
        dense_vec = self.dense_model.encode(query).tolist()
        res = self.collection.query(
            query_embeddings=[dense_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        latency = (time.time() - t0) * 1000

        results = []
        if res["ids"] and len(res["ids"][0]) > 0:
            for i, kb_id in enumerate(res["ids"][0]):
                idx = self.kb_id_to_idx.get(kb_id)
                if idx is not None:
                    row = self.kb_df.iloc[idx]
                    dist = res["distances"][0][i]
                    # Cosine distance to similarity score
                    sim = 1.0 - dist
                    results.append({
                        "kb_id": kb_id,
                        "queue": row["queue"],
                        "priority": row["priority"],
                        "problem": row["problem"],
                        "resolution": row["solution"],
                        "score": sim,
                        "rank": i + 1
                    })
        return results, latency

    def search_hybrid(self, query: str, top_k: int = 5, candidate_pool: int = 15) -> Tuple[List[Dict[str, Any]], float]:
        """
        Method 2: Hybrid Search (ChromaDB Dense + BM25 Sparse with RRF).
        """
        t0 = time.time()

        # Dense candidates
        dense_vec = self.dense_model.encode(query).tolist()
        chroma_res = self.collection.query(
            query_embeddings=[dense_vec],
            n_results=candidate_pool,
            include=["documents", "metadatas", "distances"]
        )
        dense_ranks = {}
        if chroma_res["ids"] and len(chroma_res["ids"][0]) > 0:
            for rank, kb_id in enumerate(chroma_res["ids"][0]):
                dense_ranks[kb_id] = rank + 1

        # Sparse candidates
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]
        sparse_ranks = {
            self.kb_df.iloc[idx]["kb_id"]: rank + 1
            for rank, idx in enumerate(top_bm25_indices)
        }

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
        latency = (time.time() - t0) * 1000

        results = []
        for rank, (kb_id, rrf_score) in enumerate(sorted_candidates, start=1):
            idx = self.kb_id_to_idx.get(kb_id)
            if idx is not None:
                row = self.kb_df.iloc[idx]
                results.append({
                    "kb_id": kb_id,
                    "queue": row["queue"],
                    "priority": row["priority"],
                    "problem": row["problem"],
                    "resolution": row["solution"],
                    "score": rrf_score,
                    "rank": rank
                })
        return results, latency

    def search_hybrid_rerank(self, query: str, top_k: int = 5, candidate_pool: int = 15) -> Tuple[List[Dict[str, Any]], float]:
        """
        Method 3: Hybrid Search + Cross-Encoder Reranking.
        """
        t0 = time.time()
        # Retrieve candidate pool from Hybrid RRF
        hybrid_candidates, _ = self.search_hybrid(query=query, top_k=candidate_pool, candidate_pool=candidate_pool)

        if not hybrid_candidates:
            return [], (time.time() - t0) * 1000

        # Rerank with Cross-Encoder
        pairs = [[query, f"{c['problem']} {c['resolution']}"] for c in hybrid_candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            hybrid_candidates[i]["rerank_score"] = float(score)

        hybrid_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_results = hybrid_candidates[:top_k]

        for rank, doc in enumerate(top_results, start=1):
            doc["rank"] = rank
            doc["score"] = doc["rerank_score"]

        latency = (time.time() - t0) * 1000
        return top_results, latency

    @staticmethod
    def assess_relevance(ticket: Dict[str, Any], doc: Dict[str, Any], ce_model: CrossEncoder) -> Tuple[int, int]:
        """
        Evaluates a retrieved document against the Day 16 ground-truth relevance charter.
        Returns:
            grade: 0 (Irrelevant), 1 (Marginal), 2 (Partially Relevant), 3 (Fully Relevant)
            is_hit: 1 if grade >= 2, else 0 (Binary Relevance)
        """
        true_queue = str(ticket["queue"]).lower()
        doc_queue = str(doc["queue"]).lower()
        queue_match = (true_queue == doc_queue)

        # Keyword overlap check
        kw_str = str(ticket.get("relevance_keywords", ""))
        keywords = [k.strip().lower() for k in kw_str.split(",") if len(k.strip()) >= 3]
        doc_text = f"{doc.get('problem', '')} {doc.get('resolution', '')}".lower()

        matched_kw_count = sum(1 for kw in keywords if kw in doc_text)
        kw_ratio = matched_kw_count / max(1, len(keywords))

        # Semantic relevance check using cross-encoder
        score = doc.get("rerank_score")
        if score is None:
            # Score on-the-fly if not already computed
            score = float(ce_model.predict([(ticket["text"], f"{doc['problem']} {doc['resolution']}")])[0])

        # Grading logic:
        # Grade 3: Exact queue match + high keyword overlap (>= 0.40) + strong cross-encoder score (>= 4.0)
        # Grade 2: Correct queue + moderate overlap/score OR cross-queue equivalent with score >= 5.0
        # Grade 1: Broad queue match but weak score (< 0.0) or low keyword overlap
        # Grade 0: Completely off-topic / negative score (< -2.0)
        if queue_match and score >= 4.0 and kw_ratio >= 0.33:
            grade = 3
        elif (queue_match and score >= 1.0) or (score >= 5.5 and kw_ratio >= 0.50):
            grade = 2
        elif queue_match or score >= -0.5:
            grade = 1
        else:
            grade = 0

        is_hit = 1 if grade >= 2 else 0
        return grade, is_hit


def run_benchmark(eval_csv_path: str = "data/eval_test_set_60.csv"):
    """
    Executes the 60-ticket retrieval benchmark across all 3 architectures.
    """
    eval_df = pd.read_csv(eval_csv_path)
    engine = RetrievalBenchmarkEngine()

    print(f"\nBenchmarking Retrieval on {len(eval_df)} tickets across 3 architectures...")

    results_records = []
    
    # Tracking metrics per method
    metrics = {
        "Plain Vector": {"hits@1": 0, "hits@3": 0, "hits@5": 0, "reciprocal_ranks": [], "latencies": []},
        "Hybrid Search": {"hits@1": 0, "hits@3": 0, "hits@5": 0, "reciprocal_ranks": [], "latencies": []},
        "Hybrid + Rerank": {"hits@1": 0, "hits@3": 0, "hits@5": 0, "reciprocal_ranks": [], "latencies": []}
    }

    for idx, row in eval_df.iterrows():
        eval_id = row["eval_id"]
        ticket_data = {
            "eval_id": eval_id,
            "queue": row["queue"],
            "priority": row["priority"],
            "subject": row["subject"],
            "text": row["text"],
            "relevance_keywords": row["relevance_keywords"]
        }
        query = str(row["text"])

        # 1. Plain Vector Search
        res_v, lat_v = engine.search_plain_vector(query, top_k=5)
        # 2. Hybrid Search
        res_h, lat_h = engine.search_hybrid(query, top_k=5, candidate_pool=15)
        # 3. Hybrid + Rerank
        res_hr, lat_hr = engine.search_hybrid_rerank(query, top_k=5, candidate_pool=15)

        # Evaluate each method
        for method_name, res_list, latency in [
            ("Plain Vector", res_v, lat_v),
            ("Hybrid Search", res_h, lat_h),
            ("Hybrid + Rerank", res_hr, lat_hr)
        ]:
            metrics[method_name]["latencies"].append(latency)

            # Check hits at ranks 1, 2, 3, 4, 5
            first_hit_rank = None
            hits_at_k = {1: 0, 3: 0, 5: 0}

            for doc in res_list:
                rank = doc["rank"]
                grade, is_hit = engine.assess_relevance(ticket_data, doc, engine.reranker)
                doc["grade"] = grade
                doc["is_hit"] = is_hit

                if is_hit == 1 and first_hit_rank is None:
                    first_hit_rank = rank

                if is_hit == 1:
                    if rank <= 1:
                        hits_at_k[1] = 1
                    if rank <= 3:
                        hits_at_k[3] = 1
                    if rank <= 5:
                        hits_at_k[5] = 1

            # Update metrics
            if hits_at_k[1]: metrics[method_name]["hits@1"] += 1
            if hits_at_k[3]: metrics[method_name]["hits@3"] += 1
            if hits_at_k[5]: metrics[method_name]["hits@5"] += 1

            rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
            metrics[method_name]["reciprocal_ranks"].append(rr)

        # Store detailed row record
        top_v = res_v[0] if res_v else {}
        top_h = res_h[0] if res_h else {}
        top_hr = res_hr[0] if res_hr else {}

        results_records.append({
            "eval_id": eval_id,
            "queue": row["queue"],
            "subject": row["subject"],
            "vector_top1_kb": top_v.get("kb_id", "None"),
            "vector_top1_hit": top_v.get("is_hit", 0),
            "hybrid_top1_kb": top_h.get("kb_id", "None"),
            "hybrid_top1_hit": top_h.get("is_hit", 0),
            "rerank_top1_kb": top_hr.get("kb_id", "None"),
            "rerank_top1_score": top_hr.get("rerank_score", 0.0),
            "rerank_top1_hit": top_hr.get("is_hit", 0)
        })

        if (idx + 1) % 15 == 0 or (idx + 1) == len(eval_df):
            print(f"Evaluated {idx + 1}/{len(eval_df)} tickets...")

    n_total = len(eval_df)
    summary_rows = []

    print("\n" + "=" * 75)
    print("DAY 17 OFFICIAL RETRIEVAL BENCHMARK RESULTS (N=60)")
    print("=" * 75)
    print(f"{'Retrieval Architecture':<32} | {'Hit Rate@1':<10} | {'Hit Rate@3':<10} | {'Hit Rate@5':<10} | {'MRR':<8} | {'Avg Latency':<12}")
    print("-" * 92)

    for method in ["Plain Vector", "Hybrid Search", "Hybrid + Rerank"]:
        hr1 = (metrics[method]["hits@1"] / n_total) * 100
        hr3 = (metrics[method]["hits@3"] / n_total) * 100
        hr5 = (metrics[method]["hits@5"] / n_total) * 100
        mrr = np.mean(metrics[method]["reciprocal_ranks"])
        avg_lat = np.mean(metrics[method]["latencies"])

        summary_rows.append({
            "Retrieval Architecture": method,
            "Hit Rate@1": f"{hr1:.2f}%",
            "Hit Rate@3": f"{hr3:.2f}%",
            "Hit Rate@5": f"{hr5:.2f}%",
            "MRR": f"{mrr:.4f}",
            "Avg Latency (ms)": f"{avg_lat:.2f} ms",
            "raw_hr1": hr1,
            "raw_hr3": hr3,
            "raw_hr5": hr5,
            "raw_mrr": mrr,
            "raw_latency": avg_lat
        })
        print(f"{method:<32} | {hr1:>9.2f}% | {hr3:>9.2f}% | {hr5:>9.2f}% | {mrr:>8.4f} | {avg_lat:>9.2f} ms")
    print("=" * 75)

    detailed_df = pd.DataFrame(results_records)
    summary_df = pd.DataFrame(summary_rows)

    return detailed_df, summary_df


def generate_benchmark_chart(summary_df: pd.DataFrame, output_path: str = "charts/chart17_retrieval_benchmark.html"):
    """
    Generates interactive Plotly visualization comparing Hit Rate@k, MRR, and Latency.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Retrieval Accuracy Comparison (Hit Rate@k & MRR)",
            "Query Latency vs. Accuracy Tradeoff"
        ),
        horizontal_spacing=0.15
    )

    methods = summary_df["Retrieval Architecture"].tolist()
    colors = ["#3b82f6", "#06b6d4", "#10b981"]

    # 1. Grouped Bar Chart of Hit Rate@1, Hit Rate@5, MRR
    fig.add_trace(
        go.Bar(
            name="Hit Rate@1 (%)",
            x=methods,
            y=summary_df["raw_hr1"],
            text=[f"{v:.1f}%" for v in summary_df["raw_hr1"]],
            textposition="outside",
            marker_color="#3b82f6"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name="Hit Rate@5 (%)",
            x=methods,
            y=summary_df["raw_hr5"],
            text=[f"{v:.1f}%" for v in summary_df["raw_hr5"]],
            textposition="outside",
            marker_color="#06b6d4"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name="MRR (x100)",
            x=methods,
            y=summary_df["raw_mrr"] * 100,
            text=[f"{v:.3f}" for v in summary_df["raw_mrr"]],
            textposition="outside",
            marker_color="#10b981"
        ),
        row=1, col=1
    )

    # 2. Latency Bar Chart
    fig.add_trace(
        go.Bar(
            name="Latency (ms)",
            x=methods,
            y=summary_df["raw_latency"],
            text=[f"{v:.1f} ms" for v in summary_df["raw_latency"]],
            textposition="outside",
            marker=dict(color=["#94a3b8", "#64748b", "#6366f1"])
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="<b>TicketPilot Day 17: Retrieval Architecture Benchmark (N=60)</b><br><sup>Comparing Plain Vector Search vs. BM25 Hybrid Fusion vs. Cross-Encoder Reranking</sup>",
        template="plotly_white",
        height=520,
        barmode="group",
        font=dict(family="Inter, sans-serif", size=13),
        yaxis=dict(title="Score (%) / Normalized MRR", range=[0, 115]),
        yaxis2=dict(title="Average Search Latency (ms)", range=[0, max(summary_df["raw_latency"]) * 1.3]),
        showlegend=True
    )

    fig.write_html(output_path)
    print(f"Chart generated at {output_path}")


if __name__ == "__main__":
    detailed_df, summary_df = run_benchmark("data/eval_test_set_60.csv")

    # Save detailed per-ticket CSV
    detailed_df.to_csv("day17_retrieval_benchmark_results.csv", index=False)
    print("Detailed results saved to day17_retrieval_benchmark_results.csv")

    # Save summary table CSV
    clean_summary = summary_df[[
        "Retrieval Architecture", "Hit Rate@1", "Hit Rate@3", "Hit Rate@5", "MRR", "Avg Latency (ms)"
    ]]
    clean_summary.to_csv("day17_retrieval_summary_table.csv", index=False)
    print("Summary table saved to day17_retrieval_summary_table.csv")

    # Generate Chart
    generate_benchmark_chart(summary_df, output_path="charts/chart17_retrieval_benchmark.html")
