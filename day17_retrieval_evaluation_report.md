# Day 17: Information Retrieval Benchmark & Head-to-Head Comparison Report

**Project**: TicketPilot  
**Sprint**: 4 (Rigorous Evaluation — Days 16 to 20)  
**Deliverables**:
- Retrieval Benchmark Engine: [`day17_retrieval_benchmark.py`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day17_retrieval_benchmark.py)
- Official 60-Ticket Summary Table: [`day17_retrieval_summary_table.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day17_retrieval_summary_table.csv)
- Master Results Table Updated: [`results_table.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/results_table.csv)
- Detailed Per-Ticket Log: [`day17_retrieval_benchmark_results.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day17_retrieval_benchmark_results.csv)
- Interactive Visual Dashboard: [`charts/chart17_retrieval_benchmark.html`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart17_retrieval_benchmark.html)

---

## 1. Executive Summary & Experimental Design

The primary mission for Day 17 is to empirically quantify the retrieval accuracy gains across three distinct Information Retrieval architectures using the **60 stratified evaluation tickets** established on Day 16:

1. **Method 1: Plain Vector Search**
   * Dense semantic embeddings via `all-MiniLM-L6-v2` in ChromaDB (cosine similarity).
2. **Method 2: Hybrid Search (Dense + Sparse)**
   * Dense vector search combined with Sparse BM25 keyword matching (`rank_bm25`) using **Reciprocal Rank Fusion (RRF, $k=60$)**.
3. **Method 3: Hybrid Search + Cross-Encoder Reranking 🏆**
   * Top-15 candidates from Hybrid RRF reranked using deep transformer cross-attention (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

Relevance was evaluated strictly against the Day 16 ground-truth charter ($R \ge 2$: Partially or Fully Relevant).

---

## 2. Official Quantitative Results Table (N=60)

| Retrieval Architecture | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | Mean Reciprocal Rank (MRR) | Average Search Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Method 1: Plain Vector Search** | 75.00% | 90.00% | 90.00% | 0.8139 | **19.83 ms** |
| **Method 2: Hybrid Search (Dense + BM25)** | 83.33% | 90.00% | 90.00% | 0.8639 | 272.01 ms |
| **Method 3: Hybrid + Cross-Encoder Rerank 🏆** | **90.00%** | **90.00%** | **91.67%** | **0.9033** | 591.90 ms |
| **Net Gain (Method 3 vs. Method 1)** | **+15.00%** | **—** | **+1.67%** | **+0.0894** | (+572 ms) |

*Interactive Visual Benchmark*: [`charts/chart17_retrieval_benchmark.html`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart17_retrieval_benchmark.html)

---

## 3. Deep-Dive Performance Insights

### A. Why Hybrid Search Beats Plain Vector (+8.33% at Top-1)
* **Keyword Exactness**: Dense bi-encoders compress multi-token technical strings (e.g., invoice numbers, specific SaaS tools, server error codes) into broad semantic spaces, leading to semantic dilution.
* **The BM25 Inverted Index** guarantees that articles containing exact technical vocabulary ("Keras", "PostgreSQL", "RMA", "INV-904") are surfaced into the candidate pool even if their generic embedding is distant.
* **Result**: Combining dense semantic awareness with BM25 keyword precision boosted Top-1 Hit Rate from **75.00% to 83.33%**, and MRR from **0.8139 to 0.8639**.

### B. Why Cross-Encoder Reranking Crowned the Benchmark (90.00% Hit Rate@1)
* **Bi-Encoder Blindness**: Bi-encoders encode queries and documents independently into fixed-length 384-dimensional vectors, computing similarity via simple dot products without cross-token interaction.
* **Cross-Encoder Attention**: The Cross-Encoder feeds the query and candidate document together into the transformer layers:
  $$\text{Input} = \text{[CLS]} \circ \text{Ticket Query} \circ \text{[SEP]} \circ \text{KB Article} \circ \text{[SEP]}$$
  Every query token attends directly to every resolution token via bidirectional multi-head self-attention.
* **Result**: Ambiguous candidates were cleanly filtered out, elevating the true gold-standard resolution to Rank 1 in **90.00% of all test tickets** and raising MRR to **0.9033**.

---

## 4. The Speed vs. Accuracy Tradeoff

* **Plain Vector Search**: Extremely fast (**19.83 ms**), but misses the correct answer at Rank 1 in **1 out of every 4 customer queries** (25.0% error rate).
* **Hybrid Search + Cross-Encoder**: Takes **591.90 ms**, but reduces Rank-1 error to only **10.0%** (a 60% relative error reduction).
* **Production Decision**: In customer support automation, an extra 570 ms is completely imperceptible to a customer (human reading delay is $> 3$ seconds), whereas a wrong or irrelevant answer destroys customer trust. **Hybrid + Cross-Encoder Reranking is unequivocally the champion architecture for production deployment.**

---

## "Done When" Verification Check
- [x] Evaluated across all 60 tickets from `data/eval_test_set_60.csv`.
- [x] Hit Rate@1, Hit Rate@3, Hit Rate@5, MRR, and Latency computed across all 3 architectures.
- [x] Official summary results table generated in [`day17_retrieval_summary_table.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day17_retrieval_summary_table.csv).
- [x] Master [`results_table.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/results_table.csv) updated with retrieval milestones.
- [x] Interactive Plotly dashboard generated at [`charts/chart17_retrieval_benchmark.html`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart17_retrieval_benchmark.html).
- [x] **Official results table committed to Git.**
