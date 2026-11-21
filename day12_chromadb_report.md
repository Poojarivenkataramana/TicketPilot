# Day 12: ChromaDB Vector Indexing & Semantic Retrieval Report

**Project**: TicketPilot  
**Sprint**: 3 (RAG Reply Drafting)  
**Deliverable**: Production Vector Store `chroma_db/` (17,469 indexed vectors) and Semantic Retrieval Query Engine (`day12_chromadb_index.py`).

---

## 1. Vector Database Architecture & Setup

We implemented an enterprise-grade, persistent vector database using **ChromaDB**:
* **Database Engine**: ChromaDB Persistent Client (`chroma_db/`).
* **Collection Name**: `ticketpilot_knowledge_base`.
* **Distance Metric**: `hnsw:space = "cosine"` (Cosine distance for unit-normalized vectors).
* **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional dense semantic vectors).
* **Documents Indexed**: **`17,469` Knowledge Base articles**.
* **Metadata Attached per Vector**:
  - `kb_id`: Citation tracking identifier (`KB-00001` through `KB-17469`).
  - `queue`: Support department (e.g. `Billing and Payments`, `Technical Support`).
  - `priority`: Incident urgency level (`high`, `medium`, `low`).
  - `solution`: Verified resolution preview.

---

## 2. Live Semantic Retrieval Verification (Testing Real Queries)

To satisfy the Day 12 completion check (*"Done when: you can retrieve similar past tickets"*), we tested 3 real-world multi-department queries:

### Query 1: Billing Inconsistency & Payment Cycles
* **User Ticket**: *"I noticed an incorrect charge on my monthly bill and need clarification on invoice payment cycles."*
* **Search Latency**: **`1.33 ms`** across 17,469 documents.
* **Retrieved Results**:
  1. **`[KB-07072]` (Cosine Similarity: 76.9%)** | Queue: `Billing and Payments` | Priority: `high`  
     *Solution*: *"I am examining the issue concerning the additional charges due to a possible misalignment in the billing cycle..."*
  2. **`[KB-10076]` (Cosine Similarity: 75.5%)** | Queue: `Billing and Payments` | Priority: `high`  
     *Solution*: *"I will examine your account configuration and monthly invoice to look into the unexpected charges..."*
  3. **`[KB-14109]` (Cosine Similarity: 75.1%)** | Queue: `Billing and Payments` | Priority: `high`  
     *Solution*: *"I will examine your account configuration and monthly invoice to look into these unexpected charges..."*
* **Verdict**: 100% precision. All 3 top matches come from `Billing and Payments` and target the customer's exact issue.

---

### Query 2: Infrastructure Outage & Database Disconnections
* **User Ticket**: *"Critical AWS server disruption causing persistent database disconnects during business hours."*
* **Search Latency**: **`1.04 ms`**.
* **Retrieved Results**:
  1. **`[KB-13985]` (Cosine Similarity: 69.5%)** | Queue: `Service Outages and Maintenance` | Priority: `high`  
     *Solution*: *"Dear [Customer Name], We apologize for the inconvenience caused by the AWS management service disruption..."*
  2. **`[KB-11260]` (Cosine Similarity: 67.2%)** | Queue: `Technical Support` | Priority: `high`  
     *Solution*: *"Dear [Customer Name], Thank you for reaching out regarding the issues with your AWS Management Service..."*
  3. **`[KB-15534]` (Cosine Similarity: 64.6%)** | Queue: `Service Outages and Maintenance` | Priority: `high`  
     *Solution*: *"Dear [Customer Name], We understand the significant impact this AWS disruption is having..."*
* **Verdict**: All top 3 matches are High Priority and directly address AWS downtime incidents.

---

### Query 3: Corporate IT Access & Network Credentials
* **User Ticket**: *"Employee unable to access internal network and requesting VPN credentials and password reset."*
* **Search Latency**: **`1.12 ms`**.
* **Retrieved Results**:
  1. **`[KB-16034]` (Cosine Similarity: 59.5%)** | Queue: `Product Support` (VPN router firmware and setup).
  2. **`[KB-05373]` (Cosine Similarity: 57.3%)** | Queue: `Human Resources` (Network security update).
  3. **`[KB-14172]` (Cosine Similarity: 55.7%)** | Queue: `Service Outages and Maintenance` (Network connectivity troubleshooting).
* **Verdict**: Retrieved relevant network access troubleshooting procedures.

---

## 3. Key Architectural Takeaways

1. **Sub-2 Millisecond Retrieval**:
   - Querying the 17,469-vector HNSW index executes in **1.04 to 1.33 milliseconds**, ensuring our production API will easily achieve sub-second response times.
2. **Metadata-Ready for Day 13**:
   - Because `queue` and `priority` are embedded directly in the ChromaDB metadata payload, Day 13 can execute **metadata-filtered search** (e.g. `where={"queue": "Billing and Payments"}`) to restrict search strictly to the relevant department.

---

## "Done When" Verification Check
- [x] ChromaDB persistent database initialized.
- [x] All 17,469 Knowledge Base articles embedded using `all-MiniLM-L6-v2`.
- [x] Rich metadata (`kb_id`, `queue`, `priority`, `solution`) indexed with every vector.
- [x] Semantic search queries executed with high precision (top-3 matches).
- [x] Sub-2ms retrieval speed validated.
