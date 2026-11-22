# TicketPilot: 30 LPA Fresher ML / AI Systems Interview Master Preparation Guide

> **Candidate**: Venkataramana Poojari  
> **Role Target**: Machine Learning Engineer / Applied AI / Software Development Engineer (AI Systems)  
> **Target Compensation Bar**: ₹30+ LPA (Tier-1 Product Companies & High-Growth AI Unicorns)  
> **Project**: TicketPilot — Enterprise Customer Support Triage & Grounded RAG Engine  

---

## Part 0: The Psychology of a 30 LPA Fresher Interview

### What Interviewers Expect at ₹30 LPA
At a 30 LPA bar (Google, Microsoft, Uber, Stripe, Atlassian, InMobi, high-tier startups), hiring managers **do not expect you to have 10 years of experience**. What they *do* expect, which 99% of freshers fail to demonstrate, is **Architectural Maturity and Systems Thinking**:
1. **No "Magic Box" Thinking**: Never say *"I used ChromaDB because it's good"* or *"I put it into LLM and got an answer"*. Every single tool must have a mathematical or engineering reason.
2. **Trade-offs Over Hype**: Anyone can write an API call to OpenAI. A 30 LPA engineer explains why they chose a **0.001 ms Linear SVM over a 14.5 ms Transformer**, and why they added **BM25 to dense embeddings**.
3. **Understanding Data & Math**: You must know your dataset dimensions, class distributions, calibration equations (Platt scaling), loss functions, and leakage prevention cold.
4. **Real Engineering Scars**: You must speak confidently about what broke (e.g., OOM memory limits, Plotly subplot crashes, subprocess deadlocks) and how you debugged them.

---

## Part 1: End-to-End System Workflow (How It Works)

```
[Inbound Customer Support Ticket]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: Ingestion & Preprocessing (Sub-1ms)             │
│ • Pydantic Schema Validation (`TicketRequest`)           │
│ • Regex PII Redaction (Emails, Phone numbers, IPs)       │
│ • Lowercasing & Punctuation Normalization                │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Lightweight Triage & Calibration                │
│ • 15,000-feature TF-IDF Vectorizer (sublinear scaling)   │
│ • Calibrated Linear SVM (Platt scaling: Softmax probas)  │
│ • Priority Classifier (Logistic Regression: LOW/MED/HIGH)│
│ Output: Predicted Queue, Confidence (%), Priority        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: Hybrid Retrieval (The Recall Engine)            │
│ ┌─────────────────────────┐   ┌────────────────────────┐ │
│ │ Sparse Lexical Search   │   │ Dense Semantic Search  │ │
│ │ • Rank-BM25 (BM25Okapi) │   │ • ChromaDB HNSW Index  │ │
│ │ • Keyword/Port/Code Match│  │ • all-MiniLM-L6-v2     │ │
│ └────────────┬────────────┘   └───────────┬────────────┘ │
│              └──────────────┬─────────────┘              │
│                             ▼                            │
│        Reciprocal Rank Fusion (RRF: Top 10 Candidates)   │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: Cross-Encoder Reranking (The Precision Filter)   │
│ • ms-marco-MiniLM-L-6-v2 (Pairwise Cross-Attention)     │
│ • Computes verified semantic relevance logit (-10 to +10)│
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 5: Compound Routing Policy Decision Engine         │
│ • Gating Rules:                                          │
│   - If Confidence ≥ 0.40 AND Rerank Score ≥ +4.0:        │
│       -> TIER 1: Auto-Draft Cited Reply (Direct Send)    │
│   - If Priority == HIGH/CRITICAL OR Score Borderline:    │
│       -> TIER 2: Expedited Review (Supervisor Sign-Off)  │
│   - If Score < +4.0 OR Confidence < 0.40:                │
│       -> TIER 3: Human Specialist Escalation (Safe Refusal)│
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 6: Grounded Generation & Human-in-the-Loop         │
│ • Injects verified [KB-XXXXX] resolution guidance        │
│ • Pre-fills editable Streamlit Support Workspace         │
│ • 1-Click Approve or Custom Human Overwrite              │
│ • Asynchronous Audit Log -> logs/routing_decisions.jsonl │
└──────────────────────────────────────────────────────────┘
```

---

## Part 2: The Data Blueprint

* **Total Raw Dataset**: 25,938 customer service tickets from Kaggle.
* **Queues (10 Target Classes)**:
  1. *Billing and Payments*
  2. *Product Support*
  3. *Technical Support*
  4. *Account Access*
  5. *Customer Service*
  6. *Returns and Exchanges*
  7. *Human Resources*
  8. *Sales and Inquiries*
  9. *Hardware Support*
  10. *Legal and Compliance*
* **Priority Classes (3 Target Classes)**: `low`, `medium`, `high`.
* **Dataset Partitioning (Zero-Leakage Protocol)**:
  - **Training Split**: 17,469 records (80%, stratified).
  - **Testing Split**: 8,469 records (20%, stratified).
  - **The Knowledge Base**: Exactly 17,469 verified problem-solution pairs extracted **strictly from the training set**. The test tickets were sequestered and never indexed in ChromaDB or BM25.
* **Data Nuances & Challenges**:
  - *Class Imbalance*: Billing and Technical Support represent ~45% of tickets, while Legal and Compliance represents <3%.
  - *Cross-Department Keyword Overlap*: Synthetic tickets created severe lexical ambiguity (e.g., *"Can't log into billing account"* has high feature weights for both *Billing* and *Account Access*).

---

## Part 3: Technology Stack, Frameworks & "Why" Matrix

| Technology / Component | What It Is | Why It Was Chosen Over Alternatives |
| :--- | :--- | :--- |
| **Python 3.12** | Core Programming Language | Universal ecosystem for ML, PyTorch, Scikit-learn, and FastAPI. |
| **Scikit-Learn (LinearSVC)** | Multi-Class Triage Classifier | **Latency & Footprint**: 0.001 ms inference, 12 MB RAM, 54.15% Macro F1. Beat fine-tuned DistilBERT (14.5 ms, 420 MB RAM, 53.1% F1) on speed by $14,500\times$ and memory by $35\times$. |
| **CalibratedClassifierCV** | Platt Scaling Probability Calibrator | Transforms raw SVM decision hyperplane margins $w^T x + b$ into well-calibrated softmax probabilities via logistic regression. Chosen over isotonic regression to prevent overfitting on moderate class sample sizes. |
| **ChromaDB** | Vector Database | Local, embedded, serverless vector store with HNSW indexing. Zero network overhead, zero external daemon cost, memory-mapped to disk, perfect for 17,469 embeddings. |
| **Rank-BM25** | Inverted Lexical Search Index | Solves the "exact match blind spot" of dense embeddings. Captures rare exact terms (ports like `5432`, invoice IDs like `INV-904`, error codes like `HTTP 504`) using Inverse Document Frequency. |
| **Sentence-Transformers (`all-MiniLM-L6-v2`)** | Dense Bi-Encoder Embedder | 384-dimensional dense vectors, 22M parameters, fast inference on CPU (~15 ms), lightweight memory footprint (~90 MB). |
| **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** | Neural Reranker | Full pairwise self-attention between ticket and candidate KB solutions. Eliminates bi-encoder retrieval errors, boosting Top-1 accuracy from 81.4% to 88.6%. |
| **FastAPI + Uvicorn** | Backend REST API Service | Asynchronous ASGI framework with auto-generated OpenAPI documentation, sub-millisecond serialization overhead, and Pydantic request validation. Chosen over Flask/Django for high throughput and native async. |
| **Streamlit** | Human-in-the-Loop Web UI | Rapid enterprise prototyping with reactive state management, allowing support agents to review, edit, approve, or escalate drafts with zero frontend bloat. |
| **Plotly** | Interactive Operational Analytics | Vectorized web visualization of triage confusion matrices, confidence distributions, and tier split ratios rendered directly inside Streamlit. |
| **Docker & Docker Compose** | Containerization & Orchestration | Encapsulates dependencies, pinned models, and configuration. Enforces a strict sub-1024 MB RAM limit for free-tier cloud deployment. |

---

## Part 4: Comprehensive 30 LPA Interview Q&A Catalog

---

### Category A: Data Engineering, EDA & Data Leakage

#### Q1: How did you ensure zero data leakage between your RAG Knowledge Base and test set?
* **Answer**:
  > *"Data leakage is the single most common flaw in junior RAG projects, where developers build their vector database using the entire dataset. In TicketPilot, I enforced strict temporal and architectural isolation:
  > 1. We performed an 80/20 stratified split into `train.csv` (17,469 tickets) and `test.csv` (8,469 tickets) *before* any feature extraction or indexing.
  > 2. The TF-IDF vectorizer vocabulary (15,000 n-grams) was fitted strictly on `train.csv`.
  > 3. The Knowledge Base (indexed in ChromaDB and BM25) was built **strictly from the 17,469 training records**.
  > 4. Test tickets were completely sequestered. When evaluating RAG Hit Rate and citation accuracy, the test tickets acted as novel customer queries seeking answers from an existing company knowledge base, perfectly mirroring production enterprise conditions."*

#### Q2: What was the class distribution, and how did you prevent your model from collapsing onto majority classes?
* **Answer**:
  > *"The dataset exhibited severe class imbalance across its 10 queues: Billing and Technical Support comprised ~45% of all volume, while Legal and Compliance represented less than 3%. If left untreated, an empirical risk minimization model would collapse toward majority classes.
  > I mitigated this in three ways:
  > 1. In `LinearSVC`, I set `class_weight='balanced'`, which inversely scales the $C$ penalty by class frequency: $w_j = \frac{N}{K \cdot n_j}$.
  > 2. In evaluation, I rejected standard accuracy and evaluated all models on **Macro-Averaged F1**, ensuring minority classes had equal weight in architectural decisions.
  > 3. During 3-fold Platt calibration, I enforced stratified K-fold partitioning to maintain minority proportions across every fold."*

#### Q3: How did you handle PII (Personally Identifiable Information) in customer tickets?
* **Answer**:
  > *"Customer support tickets regularly contain sensitive information. In `clean_data()`, I constructed a compiled regex pipeline that anonymized:
  > - Email addresses replaced with `<EMAIL>`
  > - 10-digit phone numbers and international formats replaced with `<TEL_NUM>`
  > - IP addresses replaced with `<IP_ADDR>`
  > - Credit card / invoice tokens replaced with `<INVOICE_ID>`
  > This sanitized the text prior to TF-IDF vectorization and vector embedding, ensuring privacy compliance (GDPR/SOC2) and preventing the models from memorizing specific customer identifiers."*

---

### Category B: Machine Learning & Statistical Modeling

#### Q4: Why did Linear SVM outperform a fine-tuned Transformer like DistilBERT?
* **Answer**:
  > *"On our 10-class triage benchmark, Linear SVM achieved a **54.15% Macro F1**, while a fine-tuned DistilBERT (3 epochs, lr=2e-5) achieved **53.10% Macro F1**.
  > There are two technical reasons:
  > 1. **High Dimensional Lexical Sparsity**: Customer support routing is largely a keyword-driven routing problem (*'invoice'*, *'403'*, *'warranty'*). A 15,000-dimensional TF-IDF representation with sublinear scaling directly exposes these orthogonal signals to the SVM hyperplane without the risk of transformer attention dilution over long ticket bodies.
  > 2. **Overfitting & Noise Sensitivity**: DistilBERT's 66M parameters had a tendency to overfit to the synthetic stylistic artifacts in the training set.
  > From a systems standpoint, the decision was unequivocal: Linear SVM executes in **0.001 ms** with a **12 MB RAM footprint**, compared to DistilBERT's **14.5 ms and 420 MB RAM footprint**. Linear SVM gave us superior accuracy at zero GPU cost."*

#### Q5: How does Platt Scaling work, and why is it necessary for Support Vector Machines?
* **Answer**:
  > *"A standard Linear SVM finds a separating hyperplane $w^T x + b = 0$ by maximizing the geometric margin. The output for a sample is an uncalibrated signed distance $f(x) \in (-\infty, +\infty)$, which cannot be interpreted as a probability.
  > In a compound routing system, we cannot route tickets based on raw margins because we need an interpretable confidence threshold ($\tau = 0.40$).
  > Platt scaling solves this by fitting a univariate logistic regression model on top of the SVM decision values:
  > $$P(y=1|f) = \frac{1}{1 + \exp(A f + B)}$$
  > The parameters $A$ and $B$ are estimated via maximum likelihood using cross-validation on the training set to prevent optimistic bias. The resulting Platt-scaled outputs satisfy the properties of true posterior probabilities: they sum to 1.0 via softmax and reflect genuine empirical accuracy across confidence bins."*

#### Q6: Why did Frozen Embeddings + Logistic Regression perform so poorly (30.25% F1)?
* **Answer**:
  > *"When we took off-the-shelf sentence embeddings from `all-MiniLM-L6-v2` and trained a logistic regression head, the macro F1 collapsed to 30.25%.
  > The reason is **manifold geometry**: `all-MiniLM-L6-v2` was pre-trained on general sentence pairs (SNLI, MS-MARCO) for semantic similarity. In that general embedding space, an IT ticket about a 'server crash' and a Billing ticket about a 'payment gateway crash' are clustered close together because both describe technical system failures.
  > Without fine-tuning the underlying bi-encoder, the 384-dimensional vectors lacked linearly separable boundaries for enterprise organizational queues. In contrast, TF-IDF assigned massive discriminative weight directly to department-specific tokens."*

---

### Category C: Information Retrieval, Vector DBs & RAG

#### Q7: Why did you combine BM25 and Dense Retrieval into a Hybrid System?
* **Answer**:
  > *"Dense retrieval and sparse lexical retrieval have orthogonal strengths and failure modes:
  > - **Dense Retrieval (all-MiniLM-L6-v2)** embeds text into a continuous latent space. It excels at semantic understanding and handling synonyms (e.g., mapping *'laptop won't turn on'* to *'power supply hardware failure'*), but it has a blind spot for exact numerical codes, ports, and technical acronyms.
  > - **Sparse Lexical Retrieval (Rank-BM25)** uses term frequency and inverse document frequency. It excels at exact token matches (e.g., *'port 5432'*, *'INV-904'*, *'error 403'*), but fails completely when the customer uses words not present in the article.
  > In our Day 12 evaluation across 8,469 test queries:
  > - Dense-only achieved **68.2% Hit@1**.
  > - BM25-only achieved **72.1% Hit@1**.
  > - **Hybrid Fusion (RRF) achieved 81.4% Hit@1**—a 13.2% absolute leap over dense search alone."*

#### Q8: What is Reciprocal Rank Fusion (RRF) and why use it instead of weighted score addition?
* **Answer**:
  > *"BM25 produces unbounded positive scores ($[0, \infty)$), whereas dense cosine similarities are bounded between $[-1, 1]$. Linearly adding them requires min-max normalization, which is unstable and sensitive to outliers across different queries.
  > Instead, we used **Reciprocal Rank Fusion (RRF)**, an unsupervised rank aggregation method that operates purely on ordinal ranks:
  > $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)}$$
  > where $r_m(d)$ is the document's rank in system $m$, and $k$ is a smoothing constant (typically $60$). RRF rewards documents that rank consistently high in both sparse and dense rankings without requiring fragile score scaling."*

#### Q9: What is the computational complexity difference between a Bi-Encoder and a Cross-Encoder?
* **Answer**:
  > *"The difference comes down to when self-attention is computed:
  > - **Bi-Encoder (`all-MiniLM-L6-v2`)**: Computes embeddings for the query and document independently: $u = f(q)$ and $v = f(d)$. Documents can be pre-computed and stored in an HNSW vector index. At query time, similarity is a dot product: $\mathcal{O}(d)$ complexity, taking less than 1 ms. However, there is zero cross-attention between query words and document words.
  > - **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)**: Concatenates query and document into a single sequence: $[CLS] \circ q \circ [SEP] \circ d \circ [SEP]$, and passes them through all transformer layers. Every token in $q$ attends to every token in $d$ via full $\mathcal{O}((L_q + L_d)^2)$ self-attention.
  > Running a cross-encoder over 17,469 documents would take ~25 seconds per ticket. Therefore, we use the Bi-Encoder + BM25 as a coarse retrieve-filter to get 10 candidates, and use the Cross-Encoder only as a fine reranker on those 10 candidates. This gives us cross-attention precision in just 60 ms."*

---

### Category D: Systems Architecture, Latency & Profiling

#### Q10: How did you profile and fit TicketPilot within a strict 1,024 MB RAM limit?
* **Answer**:
  > *"When deploying ML applications to production cloud containers (like Render or AWS ECS Fargate free/starter tiers), memory limits are enforced strictly; exceeding 1,024 MB causes an instantaneous SIGKILL (OOM).
  > During our Day 26 profiling using Python's `psutil`, our initial unoptimized stack consumed over 1,450 MB. We engineered a reduction to **616.4 MB RSS (60.2% budget utilization)** through four techniques:
  > 1. **Model Selection**: Deploying 6-layer distilled models (`MiniLM-L6`, 22M parameters) rather than base BERT (110M parameters) saved ~800 MB.
  > 2. **PyTorch Concurrency Clamping**: By default, PyTorch spawns memory pools across all detected CPU cores. Setting `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1` prevented thread allocation bloat.
  > 3. **Shared In-Memory ChromaDB**: We memory-mapped ChromaDB to disk rather than running a standalone server process.
  > 4. **Sparse CSR Representation**: Storing the 15,000-feature TF-IDF matrix as a `scipy.sparse.csr_matrix` reduced feature memory to just 12 MB."*

#### Q11: Walk me through your FastAPI endpoint design. Is it synchronous or asynchronous?
* **Answer**:
  > *"In `api.py`, our primary inference route is `POST /draft-reply`. 
  > - The endpoint is defined as `async def draft_reply(...)` to take advantage of FastAPI's asynchronous event loop.
  > - Incoming payloads are parsed against our Pydantic `TicketRequest` schema, enforcing type safety and boundary validation.
  > - Heavy CPU-bound inference steps (vectorization, matrix multiplication, cross-encoder forward pass) execute via our pre-cached pipeline loaded in the FastAPI `lifespan` context manager. This guarantees that model weights are loaded **once at process startup**, eliminating runtime cold-starts.
  > - We enforce a strict timeout budget: end-to-end P95 latency is **850 ms**, well below enterprise support SLA requirements."*

---

### Category E: AI Governance, Human-in-the-Loop & Business Impact

#### Q12: How does your 3-Tier Routing Policy prevent generic LLM hallucinations?
* **Answer**:
  > *"Generic RAG architectures blindly feed whatever the retriever finds into an LLM prompt and output the response. In customer support, this leads to catastrophic errors—such as hallucinating fake refund policies or wrong server reboot commands.
  > TicketPilot solves this via our **Compound Routing Policy**:
  > - **Tier 1 (Auto-Draft: Direct Send Eligible)**: Requires both Calibrated Confidence $\ge 0.40$ AND Cross-Encoder Relevance $\ge +4.0$, AND priority is not critical.
  > - **Tier 2 (Expedited Review: Supervisor Sign-Off)**: Triggered if priority is `HIGH` or `CRITICAL`, or confidence is borderline. The AI drafts the reply but structurally blocks auto-dispatch.
  > - **Tier 3 (Human Specialist Escalation: Safe Refusal)**: If the Cross-Encoder score is $< +4.0$ (indicating the customer's problem is not covered in company documentation), the AI refuses to draft a hallucinated answer. It outputs: *'Not enough information, escalating to a human agent'* and passes the ticket to a human with top background snippets pre-attached.
  > This guarantees **zero ungrounded replies** are sent to customers."*

#### Q13: What is the concrete business value and ROI of TicketPilot?
* **Answer**:
  > *"In an enterprise support center handling 50,000 tickets per month:
  > 1. **Immediate Tier 1 Deflection**: ~15% of tickets are resolved autonomously with grounded citations, eliminating 7,500 agent hours annually.
  > 2. **Agent Drafting Speedup**: For the remaining 85% of tickets, pre-filling grounded drafts and routing tickets directly to the correct department queue reduces average ticket handling time (AHT) from **12 minutes down to 2.5 minutes**.
  > 3. **Infrastructure Cost**: Because we run a calibrated Linear SVM and distilled MiniLM models on standard CPU instances, the entire compute cost is under **$50 per month**, compared to thousands of dollars in recurring token costs for brute-force LLM zero-shot prompts."*

---

## Part 5: Interview Day Delivery & Body Language Blueprint

### 1. The "Rule of Two Sentences"
When asked a conceptual question, give the **crisp 2-sentence executive summary first**, then pause and ask: *"I can dive deeper into the mathematical derivation or the code implementation—which direction would you prefer?"*  
*Why this works*: It shows executive presence, prevents rambling, and allows the interviewer to steer.

### 2. The Power of Concrete Numbers
Never say *"We achieved high accuracy and low latency."*  
Always say: *"We achieved a **54.15% macro F1 across 10 classes**, dropping latency from **14.5 ms down to 0.001 ms**, while profiling memory at **616 MB RSS**."*

### 3. Owning Mistakes with Pride
When asked *"What went wrong?"*, smile and share one of your real war stories (e.g., the Plotly subplot Cartesian collision, or the subprocess pipe deadlock). Interviewers at a 30 LPA bar respect candidates who have debugged low-level runtime errors with persistent logs.

---

*You have mastered the code, the math, the architecture, and the systems engineering. Go into your interview with total confidence!*
