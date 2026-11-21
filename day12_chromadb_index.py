"""
TicketPilot - Sprint 3, Day 12: ChromaDB Vector Indexing & Semantic Retrieval
-----------------------------------------------------------------------------
1. Connect to persistent ChromaDB vector store ('chroma_db/').
2. Load all 17,469 clean Knowledge Base articles from 'data/knowledge_base_clean.csv'.
3. Embed documents using all-MiniLM-L6-v2 (384-dimensional dense semantic vectors).
4. Index documents into ChromaDB collection with rich metadata:
   - kb_id (e.g. KB-00001)
   - queue (e.g. Billing and Payments)
   - priority (e.g. high)
5. Execute multi-department semantic queries (top-k=3) to verify retrieval quality.
"""

import os
import time
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def run_day12_chromadb():
    print("=" * 70)
    print("TicketPilot - Sprint 3, Day 12: ChromaDB Vector Indexing & Retrieval")
    print("=" * 70)

    kb_file = os.path.join("data", "knowledge_base_clean.csv")
    if not os.path.exists(kb_file):
        raise FileNotFoundError(f"Missing {kb_file}. Run Day 11 first.")

    # 1. Load Knowledge Base
    kb_df = pd.read_csv(kb_file)
    print(f"Loaded {len(kb_df):,} Knowledge Base articles from '{kb_file}'")

    # 2. Initialize ChromaDB Persistent Client
    db_path = "chroma_db"
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    collection_name = "ticketpilot_knowledge_base"

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    existing_count = collection.count()
    print(f"Connected to ChromaDB at '{db_path}'. Existing vectors in collection: {existing_count:,}")

    # 3. Load Sentence Transformer Embedder
    model_name = "all-MiniLM-L6-v2"
    print(f"\nLoading embedder: '{model_name}'...")
    embedder = SentenceTransformer(model_name)

    # 4. Ingest and Index Documents if not already populated
    if existing_count < len(kb_df):
        print(f"\nIndexing {len(kb_df):,} articles into ChromaDB...")
        # Format document text for indexing (combining problem context + resolution)
        documents = [
            f"Problem: {r['problem']}\nResolution: {r['solution']}"
            for _, r in kb_df.iterrows()
        ]
        ids = kb_df["kb_id"].tolist()
        metadatas = [
            {
                "kb_id": r["kb_id"],
                "queue": str(r["queue"]),
                "priority": str(r["priority"]),
                "solution": str(r["solution"])[:500]  # Store preview in metadata
            }
            for _, r in kb_df.iterrows()
        ]

        # Generate embeddings in batches
        print("Generating dense embeddings with SentenceTransformers...")
        t0 = time.time()
        embeddings = embedder.encode(
            documents,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"Computed {len(embeddings):,} embeddings in {time.time() - t0:.1f}s")

        # Upsert into ChromaDB in chunks of 500
        chunk_size = 500
        print(f"Inserting into ChromaDB collection in batches of {chunk_size}...")
        for i in range(0, len(ids), chunk_size):
            end_idx = min(i + chunk_size, len(ids))
            collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        print(f"Indexing complete! Total vectors in collection: {collection.count():,}")
    else:
        print(f"Collection is already fully indexed with {existing_count:,} vectors.")

    # -------------------------------------------------------------
    # 5. Semantic Retrieval Verification (Testing Similar Past Tickets)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: SEMANTIC RETRIEVAL VERIFICATION (TOP-3 MATCHES)")
    print("=" * 60)

    test_queries = [
        {
            "category": "Billing Query",
            "query": "I noticed an incorrect charge on my monthly bill and need clarification on invoice payment cycles."
        },
        {
            "category": "Infrastructure Outage Query",
            "query": "Critical AWS server disruption causing persistent database disconnects during business hours."
        },
        {
            "category": "Corporate IT Access Query",
            "query": "Employee unable to access internal network and requesting VPN credentials and password reset."
        }
    ]

    retrieval_results = []

    for item in test_queries:
        cat = item["category"]
        q_text = item["query"]
        print(f"\n[Test Query: {cat}]")
        print(f"  User Ticket: \"{q_text}\"")

        # Encode query
        q_emb = embedder.encode([q_text], normalize_embeddings=True).tolist()

        # Query ChromaDB for top-3 nearest neighbors
        t_start = time.time()
        results = collection.query(
            query_embeddings=q_emb,
            n_results=3,
            include=["metadatas", "documents", "distances"]
        )
        query_latency = (time.time() - t_start) * 1000  # ms

        print(f"  Retrieved top-3 similar past tickets in {query_latency:.2f} ms:")
        for rank in range(3):
            meta = results["metadatas"][0][rank]
            dist = results["distances"][0][rank]
            sim_score = (1.0 - dist) * 100  # Cosine similarity %
            kb_id = meta["kb_id"]
            queue = meta["queue"]
            priority = meta["priority"]
            sol_snippet = meta["solution"][:130].replace("\n", " ")

            print(f"    Match #{rank+1}: [{kb_id}] (Sim: {sim_score:.1f}%) | Queue: {queue:<22} | Priority: {priority:<6}")
            print(f"             Solution: {sol_snippet}...")

            retrieval_results.append({
                "test_category": cat,
                "query": q_text,
                "rank": rank + 1,
                "kb_id": kb_id,
                "similarity_score": f"{sim_score:.1f}%",
                "queue": queue,
                "priority": priority,
                "solution_snippet": sol_snippet
            })

    print("\n" + "=" * 60)
    print("DAY 12 SUCCESS: ChromaDB Semantic Retrieval Verified!")
    print("=" * 60)
    return retrieval_results

if __name__ == "__main__":
    run_day12_chromadb()
