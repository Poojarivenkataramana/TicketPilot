# Sprint 6 Day 28: Repository Cleanup, Resume Bullets & LinkedIn Post

**Project**: TicketPilot — Enterprise Customer Support AI Triage & Grounded RAG Assistant  
**Date**: November 22, 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Day 28 bridges the technical engineering achievements of TicketPilot into career assets and public portfolio visibility.

We accomplished:
1. **Repository Hygiene & Artifact Whitelisting**:
   - Cleaned temporary runtime logs (`logs/*.log` added to `.gitignore`).
   - Whitelisted core production data assets: `!data/knowledge_base_clean.csv` (17,469 clean articles, 14 MB) and `!data/test.csv` (held-out test split, 4.7 MB) so anyone cloning the public repository can run the application, tests, and Docker containers out of the box with zero setup hurdles.
2. **Metric-Driven Resume Bullets ([career/resume_bullets.md](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/career/resume_bullets.md))**:
   - Drafted tailored, XYZ-formatted bullet points for three distinct target roles: **Machine Learning Engineer**, **Senior Python / Backend Software Engineer**, and **Full-Stack AI Engineer**.
   - Integrated full technical skills matrix and 1-paragraph portfolio description blocks.
3. **LinkedIn Announcement Post ([career/linkedin_post.md](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/career/linkedin_post.md))**:
   - Crafted a story-driven, high-engagement post detailing the 30-day technical journey, key architectural innovations (Scoped Hybrid Search, Cross-Encoder reranking, 3-tier routing), engineering trade-offs (Linear SVM vs. DistilBERT), and direct links to the public GitHub repository.
4. **Automated Verification**:
   - Verified via [test_day28_career.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day28_career.py) (`3/3 tests passed, 100%`).

---

## 2. Resume Bullets Overview (XYZ Formula)

### For Machine Learning Engineer / Applied AI Scientist:
* **Architected an end-to-end, anti-hallucination customer support assistant** across 25,131 tickets, combining sub-second classical triage, metadata-scoped hybrid retrieval (ChromaDB + BM25), and cross-encoder neural reranking to achieve a **90.00% Hit Rate@1** and **0.9033 MRR** over 17,469 indexed articles.
* **Engineered a compound three-tier routing policy with refusal guardrails** that calibrated softmax confidence thresholds ($\tau = 0.40$) and cross-encoder relevance gates ($+4.0$), eliminating cascade errors from 40% to 5% and achieving a **0.0% hallucination rate** with 100% cited answers (`[KB-XXXXX]`).
* **Conducted rigorous benchmarking across 4 classification paradigms** (Logistic Regression, frozen `all-MiniLM-L6-v2` embeddings, fine-tuned DistilBERT, and Linear SVM), establishing a production champion delivering **54.32% test accuracy, 54.15% Macro-F1 at 0.001 ms inference latency** while cutting memory footprint by 97% (12 MB vs. 420 MB).
* **Led qualitative human evaluation of 30 production drafts** across accuracy, grounding, tone, and safety, attaining a **4.81 / 5.00 composite score (98.3% pass rate)**.

### For Senior Backend / Python Software Engineer:
* **Developed a high-throughput, low-latency microservice architecture in FastAPI (Python 3.12)** utilizing an async lifespan manager to pre-load ChromaDB vectors and neural models once at startup, ensuring sub-second response times (**609 ms median pipeline latency, 1.52 TPS**).
* **Implemented a thread-safe, high-concurrency Decision Logger** streaming routing telemetry and audit records simultaneously to immutable JSONL streams and tabular CSV logs with zero I/O deadlocks under batch evaluation.
* **Built an interactive full-stack support workspace in Streamlit** featuring dual-pane ticket intake, real-time triage telemetry, source explorer, inline editable draft workspace, and real-time Plotly operational dashboards.
* **Containerized the entire platform via multi-stage Docker and Docker Compose**, engineering an executable multi-mode entrypoint (`api`, `ui`, `both`) and holding runtime memory to **616 MB RSS**, operating safely within 1 GB cloud free-tier budgets.

---

## 3. LinkedIn Announcement Post Highlights

- **Hook**: Addresses the industry reality: *"Why do 95% of customer support AI pilots fail when moved to production?"*
- **Technical Pillars**:
  1. *Sub-Second Neural Triage (0.001 ms)*: Why a 12 MB Linear SVM beat a 420 MB fine-tuned DistilBERT in production.
  2. *Scoped Hybrid Retrieval + Cross-Encoder Reranking*: How scoped search + Cross-Encoder boosted Hit Rate@1 from 75.0% to 90.00% (MRR: 0.9033).
  3. *3-Tier Compound Routing & Anti-Hallucination Guardrails*: Clear mathematical decision boundaries ($\tau = 0.40$, score $= 4.0$).
  4. *Real-World Production Stats*: 609 ms median latency, 4.81/5.00 human score, 616 MB RSS memory.
- **Call-to-Action**: Direct link to the public GitHub repository: [https://github.com/Poojarivenkataramana/TicketPilot](https://github.com/Poojarivenkataramana/TicketPilot).

---

## 4. Definition of Done Check (Sprint Plan Day 28)

- [x] Repository cleaned, `.gitignore` updated, and clean working tree confirmed.
- [x] Core data files (`data/knowledge_base_clean.csv` and `data/test.csv`) tracked for out-of-the-box clone readiness.
- [x] Metric-driven resume bullet points drafted in [career/resume_bullets.md](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/career/resume_bullets.md).
- [x] LinkedIn announcement post drafted in [career/linkedin_post.md](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/career/linkedin_post.md).
- [x] Verified via [test_day28_career.py](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/test_day28_career.py).
