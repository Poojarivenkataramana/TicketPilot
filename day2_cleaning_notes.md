# Day 2: Data Cleaning & Preprocessing Report

**Project**: TicketPilot  
**Sprint**: 1 (Understand Data & Build Baseline)  
**Deliverable**: `cleaned_tickets.csv`  

---

## 1. Exact Row Count Waterfall (Step-by-Step Explanation)

| Step | Action | Remaining Rows | Delta / Explanation |
|:---|:---|:---:|:---|
| **1. Combined Raw Input** | Merged candidate files with support replies (`df1`, `df4`, `df5`) | **52,587** | `28,587` (df1) + `4,000` (df4) + `20,000` (df5) |
| **2. English Filter** | Kept only `language == 'en'` | **29,652** | **-22,935 non-English tickets** (German, Spanish, French, Portuguese removed) |
| **3. Null Value Handling** | Dropped rows missing `body` or `answer`<br>*(Filled missing `subject` with `""`)* | **29,644** | **-8 corrupt rows** missing ticket body or resolution answer.<br>*(3,810 tickets had missing subject lines but valid full bodies; preserving them maintains 3.8k real customer queries)* |
| **4. Feature Engineering** | Combined `subject` + `body` into single `text` field | **29,644** | Format: `"[subject] [body]"` with collapsed whitespaces |
| **5. Deduplication** | Case-insensitive deduplication on `text` | **25,131** | **-4,513 duplicate tickets removed** across source files to prevent severe train/test data leakage |
| **Final Clean Count** | **`cleaned_tickets.csv`** | **25,131** | **One clean, fully explainable file** |

*(Note: If a strict policy is preferred where tickets lacking an explicit email subject are discarded entirely, the dataset yields **22,134** rows).*

---

## 2. Final Label Sets & Class Distributions

### Queue Labels (10 Target Classes)
Target variable for routing customer tickets to the correct team:

| Queue Class | Ticket Count | Percentage |
|:---|:---:|:---:|
| **Technical Support** | 7,325 | 29.15% |
| **Product Support** | 4,659 | 18.54% |
| **Customer Service** | 3,783 | 15.05% |
| **IT Support** | 3,000 | 11.94% |
| **Billing and Payments** | 2,539 | 10.10% |
| **Returns and Exchanges** | 1,238 | 4.93% |
| **Service Outages and Maintenance** | 988 | 3.93% |
| **Sales and Pre-Sales** | 766 | 3.05% |
| **Human Resources** | 476 | 1.89% |
| **General Inquiry** | 357 | 1.42% |
| **Total** | **25,131** | **100.0%** |

> **Class Imbalance Strategy**: The top 5 queues account for ~85% of volume. As noted in Sprint 1 guidelines, macro-averaged metrics (Macro-F1) will be used alongside accuracy in Days 4 & 5 to ensure smaller classes (e.g., General Inquiry, HR) are not dominated.

---

### Priority Labels (3 Target Classes)
Target variable for determining SLA and triage escalation:

| Priority Class | Ticket Count | Percentage |
|:---|:---:|:---:|
| **Medium** | 10,306 | 41.01% |
| **High** | 9,742 | 38.76% |
| **Low** | 5,083 | 20.23% |
| **Total** | **25,131** | **100.0%** |

---

## 3. "Done When" Verification Check

- [x] English-only filter applied (`language == 'en'`).
- [x] Corrupt missing values removed.
- [x] Duplicates eliminated to prevent data leakage.
- [x] `subject` and `body` combined into unified `text` feature.
- [x] Target label sets finalized (`queue`: 10 classes, `priority`: 3 classes).
- [x] Exported to `cleaned_tickets.csv` with a **100% explainable row count (25,131 rows)**.
