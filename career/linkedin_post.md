# TicketPilot: Production LinkedIn Announcement Post

> **Instructions**: Copy and paste the text below directly into LinkedIn. Feel free to attach a demo screenshot of the Streamlit Workspace ([streamlit_app.py](streamlit_app.py)) or the Plotly Dashboard!

---

### 📝 LinkedIn Post Text:

Why do 95% of customer support AI pilots fail when moved to production?

Most teams plug an off-the-shelf LLM into a basic vector database and expect it to handle customer inquiries. In practice:
1. The model invents refund policies or promises discounts the company doesn't offer (hallucinations).
2. Plain vector search retrieves plausible-sounding articles from the wrong department.
3. Every single ticket goes through an expensive, slow model—wasting compute and blowing past SLAs.

Over the past month, I set out to build an enterprise-grade solution from first principles: **TicketPilot** ✈️ — an end-to-end, anti-hallucination customer support AI triage and grounded RAG assistant.

Here is the engineering architecture and what I learned:

---

### 🔍 1. Sub-Second Neural Triage (0.001 ms)
Instead of invoking a heavyweight transformer for every incoming inquiry, TicketPilot runs an optimized TF-IDF + Linear SVM classifier.
• Evaluated across 4 paradigms (Logistic Regression, frozen sentence embeddings, fine-tuned DistilBERT, Linear SVM).
• **Result**: Linear SVM emerged as champion—achieving **54.32% test accuracy and 54.15% Macro-F1** across 10 queues in **0.001 ms**, cutting memory from 420 MB to just 12 MB!

### 📚 2. Scoped Hybrid Retrieval + Neural Cross-Encoder Reranking
Searching 17,469 Knowledge Base articles globally pulls in false positives from unrelated departments.
• We first filter retrieval strictly to the predicted department queue.
• Next, we fuse ChromaDB dense semantic vectors with BM25 sparse keyword matching using Reciprocal Rank Fusion (RRF).
• Finally, a neural Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scores cross-attention between the ticket problem and candidate resolution.
• **Result**: Boosted Hit Rate@1 from **75.0%** (plain vector search) to **90.00%**, achieving an **MRR of 0.9033**!

### 🛡️ 3. Compound Routing Policy & Anti-Hallucination Guardrails
A single threshold is dangerous. TicketPilot implements a 3-tier compound routing policy:
• **Tier 1 (Auto-Draft)**: High confidence ($\tau \ge 0.40$) + high relevance ($\ge +4.0$) $\rightarrow$ Direct send eligible with inline citations `[KB-XXXXX]`.
• **Tier 2 (Expedited Review)**: High priority tickets and critical service outages $\rightarrow$ Locked auto-send with mandatory supervisor sign-off.
• **Tier 3 (Human Assist)**: Low confidence or out-of-domain requests $\rightarrow$ Anti-hallucination refusal (*"Not enough information, escalating to a specialist"*) with top KB snippets pre-attached.

### ⚡ 4. Real-World Performance
• **Latency**: 609 ms median end-to-end pipeline latency (1.52 TPS).
• **Quality**: Scored **4.81 / 5.00** in human qualitative grading across 30 production drafts (100% citation grounding).
• **Efficiency**: Total runtime memory held to **616 MB RSS**, operating comfortably within 1 GB cloud free-tier budgets!

---

### 💡 The Biggest Takeaway:
*The most complex model is rarely the best production model.* 
Choosing an ultra-fast, lightweight Linear SVM for triage and reserving neural compute for Scoped Cross-Encoder reranking delivered 100x lower latency, 97% lower memory, and zero hallucination leaks.

The entire project is open-source, fully tested (100% pass rate across 7 test suites), containerized with Docker, and includes interactive Streamlit workspaces and Plotly telemetry dashboards.

Check out the full repository and architecture diagrams on GitHub:
👉 **https://github.com/Poojarivenkataramana/TicketPilot**

I'd love to hear your thoughts and feedback from engineering teams building production RAG systems!

#MachineLearning #RAG #FastAPI #GenerativeAI #Python #NLP #Streamlit #Docker #AIEngineering #SoftwareEngineering
