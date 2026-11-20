# Day 1: Dataset Inspection & Selection Notes

**Project**: TicketPilot (Support Ticket Triage & RAG System)  
**Date**: Day 1 of 30  
**Source Dataset**: Kaggle Multilingual Customer Support Tickets (`tobiasbueck/multilingual-customer-support-tickets`)

---

## 1. Summary of All 5 CSV Files

| Reference | Filename | Row Count | Languages | Key Columns | Includes Support Replies (`answer`)? | Notes / Status |
|:---|:---|:---:|:---:|:---|:---:|:---|
| **df1** | `aa_dataset-tickets-multi-lang-5-2-50-version.csv` | 28,587 | `en` (16,338)<br>`de` (12,249) | `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `tag_1`..`tag_8` | **Yes** | Primary multi-language dataset with agent replies and rich tagging. |
| **df2** | `dataset-tickets-german_normalized.csv` | 2,125 | `de` (2,125) | `subject`, `body`, `queue`, `priority`, `language` | **No** | German-only normalized tickets without resolution replies. |
| **df3** | `dataset-tickets-german_normalized_50_5_2.csv` | 13,178 | `de` (13,178) | `subject`, `body`, `queue`, `priority`, `language` | **No** | Larger German ticket set without resolution replies. |
| **df4** | `dataset-tickets-multi-lang3-4k.csv` | 4,000 | `en` (1,391)<br>`de` (848)<br>`es` (812)<br>`fr` (476)<br>`pt` (473) | `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `business_type`, `tag_1`..`tag_9` | **Yes** | 5-language slice including business type. |
| **df5** | `dataset-tickets-multi-lang-4-20k.csv` | 20,000 | `en` (11,923)<br>`de` (8,077) | `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `tag_1`..`tag_8` | **Yes** | 20k ticket subset with answers. |

---

## 2. Cross-Dataset Overlap & Data Leakage Inspection

When merging `df1`, `df4`, and `df5` (which contain the required `answer` column):
* Total combined rows across the 3 files: **52,587 rows**
* Cleaned normalized bodies appearing in more than 1 file: **8,396 unique ticket bodies**
* Total duplicate rows identified across files: **8,420 rows**
* **Deduplicated candidate pool**: **44,167 unique tickets**

---

## 3. Decisions & Next Steps

1. **Which files to use**:
   - **Use `df1`, `df4`, and `df5`**: These contain both the input ticket (`subject` + `body`) and the resolution (`answer`), which is mandatory for building the RAG knowledge base in Sprint 3.
   - **Exclude `df2` and `df3`**: They lack the `answer` column and are German-only without target agent responses.
2. **Day 1 Completion**:
   - Project directory created.
   - Virtual environment configured.
   - Git repository initialized.
   - All 5 CSV files inspected with notes recorded.

**Done When Check**:
- [x] Short note explaining each file, row counts, columns, languages, and overlaps.
- [x] Decision made on which dataset(s) to move forward with.
