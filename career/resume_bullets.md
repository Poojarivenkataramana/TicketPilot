# TicketPilot: Production AI Resume Bullet Points & Portfolio Descriptions

> **Candidate**: Venkataramana Poojari  
> **Project**: TicketPilot — Enterprise Customer Support AI Triage & Grounded RAG Assistant  
> **GitHub**: [https://github.com/Poojarivenkataramana/TicketPilot](https://github.com/Poojarivenkataramana/TicketPilot)  

---

## 1. Resume Bullet Points (Tailored by Target Role)

### Target Role A: Machine Learning Engineer / Applied AI Scientist
* **Architected an end-to-end, anti-hallucination customer support assistant** across 25,131 tickets, combining sub-second classical triage, metadata-scoped hybrid retrieval (ChromaDB + BM25), and cross-encoder neural reranking to achieve a **90.00% Hit Rate@1** and **0.9033 MRR** over 17,469 indexed articles.
* **Engineered a compound three-tier routing policy with refusal guardrails** that calibrated softmax confidence thresholds ($\tau = 0.40$) and cross-encoder relevance gates ($+4.0$), eliminating cascade errors from 40% to 5% and achieving a **0.0% hallucination rate** with 100% cited answers (`[KB-XXXXX]`).
* **Conducted rigorous benchmarking across 4 classification paradigms** (Logistic Regression, frozen `all-MiniLM-L6-v2` embeddings, fine-tuned DistilBERT, and Linear SVM), establishing a production champion delivering **54.32% test accuracy, 54.15% Macro-F1 at 0.001 ms inference latency** while cutting memory footprint by 97% (12 MB vs. 420 MB).
* **Led qualitative human evaluation of 30 production drafts** across accuracy, grounding, tone, and safety, attaining a **4.81 / 5.00 composite score (98.3% pass rate)**.

---

### Target Role B: Senior Backend / Python Software Engineer
* **Developed a high-throughput, low-latency microservice architecture in FastAPI (Python 3.12)** utilizing an async lifespan manager to pre-load ChromaDB vectors and neural models once at startup, ensuring sub-second response times (**609 ms median pipeline latency, 1.52 TPS**).
* **Implemented a thread-safe, high-concurrency Decision Logger** streaming routing telemetry and audit records simultaneously to immutable JSONL streams and tabular CSV logs with zero I/O deadlocks under batch evaluation.
* **Built an interactive full-stack support workspace in Streamlit** featuring dual-pane ticket intake, real-time triage telemetry, source explorer, inline editable draft workspace, and real-time Plotly operational dashboards.
* **Containerized the entire platform via multi-stage Docker and Docker Compose**, engineering an executable multi-mode entrypoint (`api`, `ui`, `both`) and holding runtime memory to **616 MB RSS**, operating safely within 1 GB cloud free-tier budgets.
* **Authored a comprehensive test suite of 7 automated modules** (covering data sanitization, decision boundary edge cases, REST endpoints, and UI AppTests) executing via a master test runner with a **100% pass rate across all suites**.

---

### Target Role C: Full-Stack AI Engineer / Generalist
* **Built and deployed TicketPilot**, an enterprise AI support assistant that triages incoming customer complaints across 10 departments, retrieves authoritative solutions from 17,469 articles, and auto-drafts cited replies with 1-click human agent approval.
* **Optimized retrieval performance from 75.0% (dense vector search) to 90.0% (hybrid + cross-encoder reranking)**, leveraging Reciprocal Rank Fusion (RRF) and scoped metadata filtering to eliminate cross-department false positives.
* **Delivered a production-ready system** complete with OpenAPI Swagger documentation (`/docs`), automated health probes (`/health`), Streamlit UI, interactive Plotly dashboards, and full Docker containerization.

---

## 2. Technical Skills Matrix for Resume Header

* **Languages & Frameworks**: Python 3.12, FastAPI, Pydantic, Streamlit, Starlette, Uvicorn, Asyncio
* **Machine Learning & NLP**: Scikit-learn, Sentence-Transformers, Hugging Face Transformers, ChromaDB, BM25 Okapi, Cross-Encoders, TF-IDF, PyTorch, DistilBERT
* **Data & Analytics**: Pandas, NumPy, Plotly, KaTeX, JSONL logging
* **MLOps & DevOps**: Docker, Docker Compose, Git, GitHub Actions, Linux/macOS shell scripting, Memory Profiling (`psutil`)
* **Evaluation & Testing**: Automated Unit Testing, AppTest headless UI testing, Hit Rate@K, Mean Reciprocal Rank (MRR), Macro-F1

---

## 3. LinkedIn / Portfolio Project Description Block

```markdown
**TicketPilot — Enterprise Support Ticket Triage & Grounded RAG Assistant**
*GitHub: https://github.com/Poojarivenkataramana/TicketPilot*

An end-to-end, anti-hallucination customer support AI assistant that automates ticket triage, performs metadata-scoped hybrid vector retrieval, and generates cited support responses with mathematical safety guardrails.

Key Highlights:
• 90.00% Hit Rate@1 and 0.9033 MRR over 17,469 indexed KB articles using Dense ChromaDB + BM25 + Cross-Encoder reranking.
• 0.001 ms triage classification (54.15% Macro-F1 across 10 queues) via an optimized TF-IDF + Linear SVM champion.
• 3-Tier Compound Routing Policy (Auto-Draft, Expedited Supervisor Review, Human Assist) with calibrated confidence thresholds (tau = 0.40).
• Sub-second steady-state latency (609 ms median) and 616 MB RAM memory footprint, operating safely within 1 GB cloud limits.
• Full production stack: FastAPI REST backend, Streamlit support workspace, thread-safe JSONL decision logger, and Docker containerization.
```
