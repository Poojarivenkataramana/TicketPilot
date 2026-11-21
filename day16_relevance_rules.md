# Day 16: Evaluation Test Set Charter & Ground-Truth Relevance Rules

**Project**: TicketPilot  
**Sprint**: 4 (Rigorous Evaluation — Days 16 to 20)  
**Deliverables**:
- Evaluation Test Set (N=60): [`data/eval_test_set_60.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/data/eval_test_set_60.csv)
- Ground-Truth Relevance Specification: [`day16_relevance_rules.md`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day16_relevance_rules.md)
- Stratification Visual Dashboard: [`charts/chart16_eval_set_distribution.html`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart16_eval_set_distribution.html)

---

## 1. Purpose & Motivation

In Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) systems, qualitative impressions ("it looks pretty good") are insufficient for enterprise deployment or engineering interviews. To compute rigorous mathematical metrics on Day 17—specifically **Hit Rate@k ($HR@k$)** and **Mean Reciprocal Rank ($MRR$)**—we must establish:

1. A locked, uncorrupted evaluation benchmark of **60 held-out test tickets** drawn from `data/test.csv`.
2. An explicit, standardized, and reproducible definition of what constitutes a **"Relevant Answer"**.

Without formal relevance rules, retrieval metrics cannot be objectively verified. This document serves as the formal **Evaluation Charter** for TicketPilot Sprint 4.

---

## 2. The 4-Tier Relevance Grading Rubric

Every candidate Knowledge Base document retrieved for a customer support ticket is evaluated against a 4-tier graded scale ($R \in \{0, 1, 2, 3\}$):

```
       Grade 3: Fully Relevant (Gold Standard)  ──► Complete, actionable, exact resolution
      ┌────────────────────────────────────────┐
      │ Grade 2: Partially Relevant            │  ──► Correct intent & helpful guidance
──────┼────────────────────────────────────────┼────────────────────────────────────────
▲ HIT │ Grade 1: Marginally Relevant           │  ──► Topical overlap, but non-actionable
▼ MISS│                                        │
      │ Grade 0: Irrelevant / Distractor       │  ──► Wrong problem or wrong department
      └────────────────────────────────────────┘
```

### Detailed Grading Criteria

| Grade | Level | Operational Definition | Example (Query: "PostgreSQL port 5432 connection timeout") |
| :---: | :--- | :--- | :--- |
| **0** | **Irrelevant (Distractor)** | The document addresses an entirely different department or unrelated topic. Contains no actionable guidance for the customer's problem. | KB article discussing 14-day sneaker return packaging guidelines. |
| **1** | **Marginally Relevant** | The document shares general topical keywords or department vocabulary, but does not address the customer's specific root cause or provide actionable troubleshooting. | KB article explaining how to install pgAdmin GUI on Windows, but mentioning nothing about timeouts or port 5432 configuration. |
| **2** | **Partially Relevant** | The document addresses the customer's core intent and provides sound diagnostic direction, but misses secondary specifics (e.g. addresses general firewall timeouts without naming `pg_hba.conf`). | KB article describing general TCP socket timeouts and server firewall whitelisting procedures. |
| **3** | **Fully Relevant (Gold Standard)** | The document directly, accurately, and comprehensively provides the exact diagnostic steps, configuration directives, or official organizational policy needed to resolve the ticket without human escalation. | KB article detailing `max_connections` in `postgresql.conf`, `pg_hba.conf` client whitelisting, and service restarts. |

---

## 3. Mathematical Metric Definitions for Day 17

To quantitatively benchmark Plain Vector Search vs. Hybrid Search vs. Hybrid + Cross-Encoder Reranking on Day 17, we map graded relevance to a **Binary Relevance Threshold**:

$$R_{\text{binary}} = \begin{cases} 1 & \text{if Grade } \ge 2 \quad (\text{Hit}) \\ 0 & \text{if Grade } < 2 \quad (\text{Miss}) \end{cases}$$

A retrieval candidate is declared a **Hit ($R=1$)** if and only if it provides at least **Partially Relevant (Grade 2)** actionable guidance.

### A. Hit Rate at Rank $k$ ($\text{Hit Rate}@k$)
Measures the proportion of test tickets for which at least one relevant document ($R=1$) appears within the top-$k$ retrieved candidates:

$$\text{Hit Rate}@k = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\sum_{j=1}^{k} R_{i, j} \ge 1\right)$$

Where $N=60$, and $R_{i, j} \in \{0, 1\}$ is the binary relevance of the document at rank $j$ for ticket $i$. We evaluate at $k \in \{1, 3, 5\}$.

### B. Mean Reciprocal Rank (MRR)
Measures where the **first relevant document** appears in the ranked list. Higher rank yields a higher score:

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the rank position of the first document with $R=1$. If no relevant document is found in the top-$k$, the reciprocal rank for that query is $0$.

---

## 4. Department-Specific Ground-Truth Relevance Rules

To eliminate subjective evaluator bias, candidate KB articles must satisfy the following department-specific standards:

1. **Billing and Payments**:
   - Must address invoice reconciliation, payment gateway errors, duplicate transaction refunds, or credit card updates.
   - *Hit Rule*: Must state the refund timeframe (e.g., 3–5 business days) or specific portal navigation.
2. **Technical Support**:
   - Must address software runtime crashes, database connection failures, API endpoints, or code stack traces.
   - *Hit Rule*: Must provide specific configuration parameters, service restart procedures, or log inspection commands.
3. **IT Support**:
   - Must address internal employee hardware, VPN connectivity, LDAP/Active Directory authentication, or device provisioning.
   - *Hit Rule*: Must identify credential reset paths or network configuration steps.
4. **Returns and Exchanges**:
   - Must address RMA authorization, prepaid shipping labels, return windows, or product condition criteria.
   - *Hit Rule*: Must clarify the return eligibility window (e.g. 14 or 30 days) and packaging requirements.
5. **Service Outages and Maintenance**:
   - Must address server downtime, cloud cluster health, scheduled maintenance windows, or failover status.
   - *Hit Rule*: Must provide status page reference or infrastructure recovery verification steps.
6. **Product Support**:
   - Must address SaaS software functionality, feature integrations, dashboard configuration, or workflow automations.
   - *Hit Rule*: Must identify the specific feature settings or integration path.
7. **Sales and Pre-Sales**:
   - Must address subscription tiers, volume licensing, enterprise quotes, or contract terms.
   - *Hit Rule*: Must state pricing tier structures or enterprise sales representative contact procedures.
8. **Human Resources**:
   - Must address employee onboarding, payroll sync, employee records, benefits, or HR portal access.
   - *Hit Rule*: Must identify employee portal sections or HR administration verification procedures.
9. **Customer Service**:
   - Must address general account inquiries, profile updates, order tracking, or service requests.
   - *Hit Rule*: Must provide direct account management steps or customer verification guidance.
10. **General Inquiry**:
    - Must address company office locations, general consulting offerings, operating hours, or high-level services.
    - *Hit Rule*: Must provide factual company information without deflecting.

---

## 5. Stratification Breakdown of the 60 Evaluation Tickets

The test set [`data/eval_test_set_60.csv`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/data/eval_test_set_60.csv) contains exactly 60 tickets stratified as follows:

| Department Queue | Ticket Count | Priority Mix (High / Med / Low) | Mean Text Length (Words) | Core Intent Focus |
| :--- | :---: | :---: | :---: | :--- |
| **Billing and Payments** | 6 | 2 High / 3 Med / 1 Low | 154 words | Duplicate charges, invoice discrepancies, payment gateway |
| **Customer Service** | 6 | 2 High / 3 Med / 1 Low | 162 words | Account updates, general assistance, service verification |
| **General Inquiry** | 6 | 1 High / 3 Med / 2 Low | 148 words | IT consulting, company services, operational details |
| **Human Resources** | 6 | 2 High / 3 Med / 1 Low | 158 words | Employee data sync, payroll onboarding, HR records |
| **IT Support** | 6 | 2 High / 3 Med / 1 Low | 166 words | Hardware malfunctions, tool upgrades, access credentials |
| **Product Support** | 6 | 1 High / 4 Med / 1 Low | 152 words | Marketing tools integration, software feature workflows |
| **Returns and Exchanges** | 6 | 2 High / 3 Med / 1 Low | 160 words | Defective item returns, RMA requests, refund tracking |
| **Sales and Pre-Sales** | 6 | 2 High / 3 Med / 1 Low | 155 words | Enterprise SaaS pricing, data security pre-sales, contracts |
| **Service Outages** | 6 | 2 High / 4 Med / 0 Low | 164 words | Server capacity, unexpected service disruption, downtime |
| **Technical Support** | 6 | 1 High / 4 Med / 1 Low | 170 words | API integration failures, system errors, database bugs |
| **Total Benchmark** | **`60`** | **17 High / 33 Med / 10 Low** | **158.9 words** | **100% Departmental & Priority Coverage** |

*Interactive Visual Dashboard*: [`charts/chart16_eval_set_distribution.html`](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart16_eval_set_distribution.html)

---

## "Done When" Verification Check
- [x] Curated test set of 60 tickets extracted from held-out test data (`data/eval_test_set_60.csv`).
- [x] Stratified across all 10 departments (exactly 6 tickets per queue) and all 3 priority levels.
- [x] Non-null subjects, bodies, and clean explainable metadata verified.
- [x] Standardized 4-tier relevance grading rubric (Grades 0 to 3) defined.
- [x] Binary relevance threshold ($R \ge 2$) and mathematical formulas for Hit Rate@k and MRR documented.
- [x] Department-specific ground-truth acceptance rules written down.
- [x] **The test set and rules are written down and version-controlled.**
