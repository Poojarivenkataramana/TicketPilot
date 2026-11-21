# Day 7: Sentence Embeddings Feature Extraction & Classification Report

**Project**: TicketPilot  
**Sprint**: 2 (Better Models & Error Analysis)  
**Deliverable**: Dense semantic embeddings extraction (`all-MiniLM-L6-v2`), downstream classification head evaluation, and results table update (`day7_sentence_embeddings.py`).

---

## 1. Methodology: Transitioning to Dense Semantic Vectors

In Days 4–6, we utilized **sparse, high-dimensional bag-of-words features** (TF-IDF with 15,000 n-grams).  
On Day 7, we tested **dense semantic representations** via transfer learning:

* **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
* **Embedding Dimension**: 384 continuous dimensions
* **Tickets Encoded & Cached**:
  - Training Set: `17,591` vectors $\rightarrow$ `data/train_emb_minilm.npy` (26 MB)
  - Validation Set: `3,770` vectors $\rightarrow$ `data/val_emb_minilm.npy` (5.5 MB)
* **Downstream Classifier**: Linear SVM with balanced class weights (`LinearSVC(class_weight='balanced')`).

---

## 2. Benchmark Comparison on Queue Classification

| Model Representation | Architecture | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Sparse Bag-of-Words** | **TF-IDF + Linear SVM** | **55.09%** | **51.63%** | **59.02%** | **54.50%** | **Current Leader** |
| **Sparse Bag-of-Words** | TF-IDF + Logistic Regression | 47.00% | 43.42% | 53.37% | 46.50% | Sprint 1 Baseline |
| **Dense Embeddings (Frozen)** | **MiniLM (384-d) + Linear SVM** | **33.55%** | **30.03%** | **40.37%** | **30.25%** | **Day 7 Model** |

*Official Results Table*: [results_table.csv](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/results_table.csv)

---

## 3. Deep Analysis: Why Did Frozen Embeddings Underperform TF-IDF?

This experiment highlights a fundamental, real-world lesson in applied Natural Language Processing:

1. **Domain Mismatch in Off-the-Shelf Embeddings**:
   - `all-MiniLM-L6-v2` was pre-trained on generic web corpora (Wikipedia, Reddit, Yahoo Answers, SNLI) to measure general sentence similarity.
   - It projects sentences into an embedding space optimized for broad conversational meaning, **not technical support ticket routing**.
2. **Dilution of High-Value Domain Keywords**:
   - In customer support, a single keyword like *"invoice"*, *"refund"*, *"outage"*, or *"password"* is a 100% decisive cue for the department.
   - In sparse TF-IDF, these decisive keywords receive direct, isolated weights.
   - In frozen 384-dimensional dense sentence embeddings, the signal of a critical keyword is pooled and diluted across the entire sentence representation.
3. **The Empirical Justification for Day 8 (DistilBERT Fine-Tuning)**:
   - This failure empirically demonstrates that simply using a **frozen, pre-trained transformer as a feature extractor** is insufficient.
   - To unlock the true power of transformers for support ticket triage, we must **fine-tune the attention layers end-to-end** so the model learns which specific words and syntactic structures matter for enterprise support queues.

---

## 4. Artifacts Persisted
* Pipeline Script: `day7_sentence_embeddings.py`
* Dense Vectors: `data/train_emb_minilm.npy`, `data/val_emb_minilm.npy`
* Classifier: `models/embedding_classifier_queue.pkl`
* Updated Benchmark: `results_table.csv`

## "Done When" Verification Check
- [x] Generated dense sentence embeddings for train and validation splits.
- [x] Trained classification heads on dense embeddings.
- [x] Evaluated on held-out validation set.
- [x] Embedding model scores added to `results_table.csv`.
- [x] Technical root-cause analysis documented.
