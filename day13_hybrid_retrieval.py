"""
TicketPilot - Sprint 3, Day 13: Hybrid Search, Metadata Filtering & Reranking
-----------------------------------------------------------------------------
Pipeline Architecture:
1. Dense Retrieval: ChromaDB Vector Search (all-MiniLM-L6-v2).
2. Sparse Retrieval: BM25 Lexical Keyword Search (rank_bm25.BM25Okapi).
3. Metadata Filtering: Scope search strictly to predicted/target department queue.
4. Reciprocal Rank Fusion (RRF): Combines dense and sparse rank candidates.
5. Cross-Encoder Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 computes deep cross-attention.
6. 10-Example Benchmark: Head-to-head comparison against Plain Vector Search.
7. Export interactive comparison chart (charts/chart13_retrieval_comparison.html).
"""

import os
import time
import pandas as pd
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import plotly.graph_objects as go

class HybridRetriever:
    def __init__(self, kb_path="data/knowledge_base_clean.csv", db_path="chroma_db"):
        print("Initializing Hybrid Retrieval & Reranking Engine...")
        self.kb_df = pd.read_csv(kb_path)
        print(f"Loaded {len(self.kb_df):,} Knowledge Base articles.")

        # 1. Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection("ticketpilot_knowledge_base")
        print(f"Connected to ChromaDB collection: {self.collection.count():,} vectors.")

        # 2. Dense Embedder
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 3. Sparse BM25 Index
        print("Building BM25 Sparse Inverted Index...")
        self.corpus = [
            f"{r['problem']} {r['solution']}".lower().split()
            for _, r in self.kb_df.iterrows()
        ]
        self.bm25 = BM25Okapi(self.corpus)
        self.kb_id_to_idx = {r["kb_id"]: i for i, r in self.kb_df.iterrows()}

        # 4. Cross-Encoder Reranker
        print("Loading Cross-Encoder Reranker: 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Engine Ready.\n")

    def plain_vector_search(self, query: str, top_k: int = 3):
        """Method A: Plain dense vector search without filtering or reranking."""
        q_emb = self.dense_model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        hits = []
        for rank in range(len(results["ids"][0])):
            meta = results["metadatas"][0][rank]
            dist = results["distances"][0][rank]
            sim = (1.0 - dist)
            hits.append({
                "kb_id": meta["kb_id"],
                "queue": meta["queue"],
                "priority": meta["priority"],
                "solution": meta["solution"],
                "score": sim
            })
        return hits

    def hybrid_rerank_search(self, query: str, target_queue: str = None, top_k: int = 3):
        """Method B: Metadata Filtering + Dense/Sparse Hybrid (RRF) + Cross-Encoder Reranking."""
        q_emb = self.dense_model.encode([query], normalize_embeddings=True).tolist()

        # Step 1: Filtered Dense Search (fetch top 15)
        where_filter = {"queue": target_queue} if target_queue else None
        dense_res = self.collection.query(
            query_embeddings=q_emb,
            n_results=15,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )

        dense_ranked_ids = dense_res["ids"][0]

        # Step 2: Sparse BM25 Search
        q_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(q_tokens)

        # Filter by department if target_queue provided
        if target_queue:
            mask = (self.kb_df["queue"] == target_queue).values
            bm25_scores = np.where(mask, bm25_scores, -1e9)

        top_bm25_indices = np.argsort(bm25_scores)[::-1][:15]
        sparse_ranked_ids = [self.kb_df.iloc[idx]["kb_id"] for idx in top_bm25_indices]

        # Step 3: Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank))
        for rank, doc_id in enumerate(sparse_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60 + rank))

        top_candidate_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]

        # Step 4: Cross-Encoder Reranking
        candidate_rows = self.kb_df[self.kb_df["kb_id"].isin(top_candidate_ids)]
        pairs = [
            (query, f"Problem: {r['problem']}\nResolution: {r['solution']}")
            for _, r in candidate_rows.iterrows()
        ]
        cross_scores = self.reranker.predict(pairs)

        ranked_indices = np.argsort(cross_scores)[::-1][:top_k]
        
        final_hits = []
        for idx in ranked_indices:
            row = candidate_rows.iloc[idx]
            final_hits.append({
                "kb_id": row["kb_id"],
                "queue": row["queue"],
                "priority": row["priority"],
                "solution": row["solution"],
                "rerank_score": float(cross_scores[idx])
            })
        return final_hits

def run_day13_evaluation():
    print("=" * 70)
    print("TicketPilot - Sprint 3, Day 13: 10-Example Head-to-Head Retrieval Benchmark")
    print("=" * 70)

    retriever = HybridRetriever()

    # 10 Diverse, Challenging Test Queries
    benchmark_queries = [
        {
            "id": 1,
            "category": "Billing Discrepancy",
            "query": "Unexpected unauthorized subscription fee on June credit card invoice, need refund and cycle clarification.",
            "target_queue": "Billing and Payments",
            "keywords": ["unauthorized", "invoice", "refund"]
        },
        {
            "id": 2,
            "category": "Cloud Infrastructure Outage",
            "query": "AWS EC2 instance failure causing database connection timeout and severe disruption during business hours.",
            "target_queue": "Service Outages and Maintenance",
            "keywords": ["aws", "ec2", "outage"]
        },
        {
            "id": 3,
            "category": "Enterprise IT Security",
            "query": "Employee locked out of internal corporate network, requesting LDAP password reset and VPN profile setup.",
            "target_queue": "IT Support",
            "keywords": ["ldap", "vpn", "password"]
        },
        {
            "id": 4,
            "category": "Return Merchandise Authorization",
            "query": "Defective item received, need RMA return merchandise label and policy details on return shipping window.",
            "target_queue": "Returns and Exchanges",
            "keywords": ["rma", "return", "refund"]
        },
        {
            "id": 5,
            "category": "Hardware Laptop Issue",
            "query": "Dell XPS 13 laptop display screen flickering repeatedly after firmware driver installation.",
            "target_queue": "Product Support",
            "keywords": ["dell xps", "screen", "display"]
        },
        {
            "id": 6,
            "category": "Software Framework Bug",
            "query": "Laravel 8 unexpected memory allocation crash on MacBook Pro when running batch data export.",
            "target_queue": "Technical Support",
            "keywords": ["laravel", "crash", "memory"]
        },
        {
            "id": 7,
            "category": "Critical Security Breach",
            "query": "Report of serious security breach involving unauthorized access to hospital patient medical records.",
            "target_queue": "Technical Support",
            "keywords": ["breach", "security", "medical"]
        },
        {
            "id": 8,
            "category": "HR Portal Onboarding",
            "query": "New employee unable to sign into payroll onboarding system, need updated credentials.",
            "target_queue": "Human Resources",
            "keywords": ["onboarding", "payroll", "employee"]
        },
        {
            "id": 9,
            "category": "Sales Enterprise Pricing",
            "query": "Interested in purchasing enterprise volume subscription, need pricing quote and discount tiers for CRM.",
            "target_queue": "Sales and Pre-Sales",
            "keywords": ["pricing", "enterprise", "discount"]
        },
        {
            "id": 10,
            "category": "General Company Information",
            "query": "Requesting official business hours and physical headquarters office location for corporate visit.",
            "target_queue": "General Inquiry",
            "keywords": ["hours", "headquarters", "office"]
        }
    ]

    benchmark_rows = []

    print("\n--- Running Head-to-Head Comparison on 10 Queries ---")
    for item in benchmark_queries:
        qid = item["id"]
        cat = item["category"]
        q = item["query"]
        t_queue = item["target_queue"]

        # Method A: Plain Vector Search
        hits_plain = retriever.plain_vector_search(q, top_k=3)
        top_plain = hits_plain[0]
        # Cross-encoder score of top plain hit for fair quality comparison
        plain_ce_score = retriever.reranker.predict([(q, top_plain["solution"])])[0]

        # Method B: Hybrid + Metadata Filter + Reranking
        hits_hybrid = retriever.hybrid_rerank_search(q, target_queue=t_queue, top_k=3)
        top_hybrid = hits_hybrid[0]
        hybrid_ce_score = top_hybrid["rerank_score"]

        # Department alignment check
        plain_correct_dept = (top_plain["queue"] == t_queue)
        hybrid_correct_dept = (top_hybrid["queue"] == t_queue)

        # Keyword match check
        q_lower = q.lower()
        hybrid_kw_matches = sum(1 for kw in item["keywords"] if kw in top_hybrid["solution"].lower())
        plain_kw_matches = sum(1 for kw in item["keywords"] if kw in top_plain["solution"].lower())

        winner = "Hybrid + Rerank 🏆" if hybrid_ce_score > plain_ce_score else "Tie/Plain"

        print(f"\n[Query #{qid}: {cat}]")
        print(f"  Query: \"{q}\"")
        print(f"  Plain Vector Top-1:   [{top_plain['kb_id']}] ({top_plain['queue']}) | Cross-Encoder Score: {plain_ce_score:.2f}")
        print(f"  Hybrid+Rerank Top-1:  [{top_hybrid['kb_id']}] ({top_hybrid['queue']}) | Cross-Encoder Score: {hybrid_ce_score:.2f}")
        print(f"  Outcome: {winner} (Delta: +{hybrid_ce_score - plain_ce_score:.2f} pts)")

        benchmark_rows.append({
            "Query #": qid,
            "Category": cat,
            "Target Department": t_queue,
            "Plain Vector Top-1": f"{top_plain['kb_id']} ({top_plain['queue']})",
            "Plain CE Score": round(float(plain_ce_score), 2),
            "Hybrid+Rerank Top-1": f"{top_hybrid['kb_id']} ({top_hybrid['queue']})",
            "Hybrid CE Score": round(float(hybrid_ce_score), 2),
            "Relevance Gain (Delta)": round(float(hybrid_ce_score - plain_ce_score), 2),
            "Winner": winner
        })

    results_df = pd.DataFrame(benchmark_rows)

    print("\n" + "=" * 80)
    print("DAY 13 OFFICIAL 10-EXAMPLE RETRIEVAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(results_df[["Query #", "Category", "Plain CE Score", "Hybrid CE Score", "Relevance Gain (Delta)", "Winner"]].to_string(index=False))

    mean_plain = results_df["Plain CE Score"].mean()
    mean_hybrid = results_df["Hybrid CE Score"].mean()
    wins = (results_df["Hybrid CE Score"] > results_df["Plain CE Score"]).sum()

    print(f"\nSummary Metrics across 10 Examples:")
    print(f"  Plain Vector Average Cross-Encoder Score:  {mean_plain:.2f}")
    print(f"  Hybrid+Rerank Average Cross-Encoder Score: {mean_hybrid:.2f} (+{mean_hybrid - mean_plain:.2f} gain)")
    print(f"  Hybrid Retrieval Win Rate:                 {wins} / 10 ({wins*10}%)")

    # Generate Plotly Comparison Bar Chart
    fig = go.Figure()
    categories = [f"Q{r['Query #']}: {r['Category']}" for _, r in results_df.iterrows()]
    fig.add_trace(go.Bar(
        name="Plain Vector Search",
        x=categories,
        y=results_df["Plain CE Score"],
        marker_color="#FFA15A",
        text=results_df["Plain CE Score"],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="Hybrid Search + Reranker (Method B)",
        x=categories,
        y=results_df["Hybrid CE Score"],
        marker_color="#00CC96",
        text=results_df["Hybrid CE Score"],
        textposition="auto"
    ))

    fig.update_layout(
        title="Day 13: Retrieval Relevance Comparison on 10 Examples (Plain Vector vs. Hybrid+Rerank)",
        yaxis_title="Cross-Encoder Relevance Score",
        barmode="group",
        template="plotly_white",
        height=600,
        width=1000
    )
    chart13_path = "charts/chart13_retrieval_comparison.html"
    fig.write_html(chart13_path)
    print(f"\nSaved interactive benchmark chart to: {chart13_path}")

    return results_df

if __name__ == "__main__":
    run_day13_evaluation()
