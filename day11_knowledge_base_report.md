# Day 11: Clean Knowledge Base Construction & Leakage Audit Report

**Project**: TicketPilot  
**Sprint**: 3 (RAG Reply Drafting)  
**Deliverable**: Production Knowledge Base `data/knowledge_base_clean.csv` (17,469 verified articles) with strict anti-leakage audit (`day11_build_knowledge_base.py`).

---

## 1. Overview: Preparing Ground Truth for RAG

In Sprint 3, our goal is to build an automated **Retrieval-Augmented Generation (RAG)** copilot that drafts cited responses to incoming customer tickets.

To prevent hallucinations, the LLM must **only reference verified company solutions**. On Day 11, we converted the historical resolutions in `data/train.csv` into a structured, production Knowledge Base (KB).

---

## 2. Placeholder Cleaning & Normalization

The raw training dataset contained dozens of raw synthetic template tags. We implemented regex token standardization to replace these with clean, professional placeholders:

| Raw Tag | Cleaned KB Token | Purpose |
| :--- | :--- | :--- |
| `<name>` | `[Customer Name]` | Protects customer privacy. |
| `<Customer Support Agent>`, `<Your Name>`, `<name of agent>` | `[Support Agent]` | Replaces agent signature tags. |
| `<tel_num>`, `<contact_number>` | `[Support Hotline]` | Standardizes phone number placeholders. |
| `<email>` | `support@company.com` | Standardizes email contact points. |
| `<website_url>`, `<service_link>` | `https://support.company.com` | Standardizes help desk links. |
| `<order_num>`, `<case_num>`, `<acc_num>` | `[Order #]`, `[Case #]`, `[Account #]` | Standardizes transaction identifiers. |
| `<br>` | `\n` | Converts HTML breaks to clean text breaks. |

---

## 3. Knowledge Base Article Schema

Every record in `data/knowledge_base_clean.csv` contains structured fields engineered for vector indexing in Day 12:

```csv
kb_id,queue,priority,problem,solution
KB-00001,Technical Support,high,"Server disconnection error...","Investigate AWS logs..."
```

* **`kb_id`**: Unique permanent citation tag (`KB-00001` to `KB-17469`) used by the LLM in Day 14 to provide verifiable source citations.
* **`queue`**: Department metadata enabling **metadata-filtered vector search** (Day 13).
* **`priority`**: Severity metadata.
* **`problem`**: Cleaned historical customer query text.
* **`solution`**: Cleaned, verified resolution steps written by support agents.

---

## 4. Strict Test Set Leakage Audit

The explicit success criterion for Day 11 is: **"Done when: the KB file is ready, with no test tickets in it."**

We cross-checked all 17,469 KB articles against the 3,770 held-out test tickets (`data/test.csv`):
* **Test Question Overlap in KB**: **`0 tickets` (0.00%)**
* **Test Solution Overlap in KB**: **`0 tickets` (0.00%)**
* **Verification Status**: **100% PASSED**.

No question text, customer complaint, or resolution answer from the test set exists anywhere in the Knowledge Base.

---

## 5. Knowledge Base Breakdown by Department

| Department Queue | Total Articles in KB | Percentage |
| :--- | :---: | :---: |
| **Technical Support** | 5,095 | 29.17% |
| **Product Support** | 3,237 | 18.53% |
| **Customer Service** | 2,627 | 15.04% |
| **IT Support** | 2,082 | 11.92% |
| **Billing and Payments** | 1,763 | 10.09% |
| **Returns and Exchanges** | 865 | 4.95% |
| **Service Outages and Maintenance** | 689 | 3.94% |
| **Sales and Pre-Sales** | 532 | 3.05% |
| **Human Resources** | 330 | 1.89% |
| **General Inquiry** | 249 | 1.43% |
| **Total Clean KB Articles** | **17,469** | **100.0%** |

---

## "Done When" Verification Check
- [x] Knowledge base constructed strictly from training set answers.
- [x] Placeholders and synthetic tags cleaned into professional documentation tokens.
- [x] Redundant duplicate templates deduplicated.
- [x] Citation IDs (`kb_id`) and routing metadata (`queue`, `priority`) attached.
- [x] Zero test tickets verified in the knowledge base.
- [x] Saved to `data/knowledge_base_clean.csv` (14.45 MB).
