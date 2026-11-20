# Sprint 1: Final Baseline Results & Error Analysis Report

**Project**: TicketPilot  
**Milestone**: Sprint 1 Completion (Days 1–5)  
**Deliverables**: 
1. `models/baseline_queue_model.pkl` (Queue Baseline)
2. `models/baseline_priority_model.pkl` (Priority Baseline)
3. `charts/confusion_matrix_queue.html` & `charts/confusion_matrix_priority.html`
4. `results_table.csv` & Git commit

---

## 1. Official Sprint 1 Results Table (Validation Set Benchmark)

| Target Task | Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted F1 | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`queue` (10 Classes)** | TF-IDF + Logistic Regression (Balanced) | **47.00%** | 43.42% | 53.37% | **46.50%** | 47.22% | < 2 ms |
| **`priority` (3 Classes)** | TF-IDF + Logistic Regression (Balanced) | **56.55%** | 55.23% | 56.82% | **55.61%** | 56.80% | < 1 ms |

> **Key Takeaway**: These two models serve as our honest, fast, zero-GPU baseline benchmarks. In Sprint 2, every advanced model (SVM, Sentence Transformers, DistilBERT) will be evaluated against these exact numbers.

---

## 2. Priority Baseline Breakdown (3 Classes)

* **Overall Accuracy**: **56.55%**
* **Macro-F1 Score**: **55.61%**

| Priority Tier | Precision | Recall | F1-Score | Support (Val) | Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **High** | **0.6204** | **0.5994** | **0.6097** | 1,453 | Strongest performer; detects critical system downtime and urgent issues. |
| **Medium** | **0.6026** | **0.5297** | **0.5638** | 1,580 | Often overlaps with High and Low boundaries. |
| **Low** | **0.4340** | **0.5753** | **0.4947** | 737 | Balanced weighting preserves 57.5% recall for low-urgency general inquiries. |

---

## 3. Confusion Matrix Diagnostic & Error Analysis

### A. Queue Confusion Analysis
* **Interactive Chart**: [confusion_matrix_queue.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/confusion_matrix_queue.html)
* **Diagnosis of Off-Diagonal Confusion**:
  1. **`Technical Support` $\leftrightarrow$ `IT Support` (270 cross-errors)**: The single largest source of model confusion. Because both departments frequently mention words like *"server"*, *"data"*, *"crash"*, and *"login"*, linear n-gram TF-IDF cannot differentiate whether the user is an internal employee needing corporate IT help or an external user encountering a software bug.
  2. **`Technical Support` $\leftrightarrow$ `Product Support` (262 cross-errors)**: Product feature malfunctions versus deeper technical defects share identical complaints.
  3. **`Customer Service` $\leftrightarrow$ `Product Support` (170 cross-errors)**: General inquiries that mention product names get misrouted.
* **Why Sprint 2 is Justified**: This failure pattern proves that bag-of-words keyword matching has hit its ceiling. Contextual transformer embeddings (DistilBERT) are required to understand sentence syntax and speaker intent.

### B. Priority Confusion Analysis
* **Interactive Chart**: [confusion_matrix_priority.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/confusion_matrix_priority.html)
* **Confusion Matrix Counts**:
  - **True High**: 871 correctly predicted as High; 374 predicted as Medium; only 208 misclassified as Low.
  - **True Medium**: 837 correctly predicted as Medium; 398 predicted as High; 345 predicted as Low.
  - **True Low**: 424 correctly predicted as Low; 178 predicted as Medium; 135 misclassified as High.
* **Key Finding**: Errors are predominantly between adjacent classes (High $\leftrightarrow$ Medium, Medium $\leftrightarrow$ Low). Catastrophic errors (True High misclassified as Low) occur in only **14.3%** of high-priority cases. In Sprint 5, we will implement threshold tuning to virtually eliminate False Lows.

---

## 4. Sprint 1 Retrospective (Days 1 to 5)

| Day | Goal | Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Day 1** | Inspect 5 CSVs, detect leakage | Audit notes in `day1_dataset_notes.md`, identified 8.4k duplicates across candidate files. | ✅ Done |
| **Day 2** | Clean data, English-only, combine text | `cleaned_tickets.csv` (25,131 explainable rows), `day2_cleaning_notes.md`. | ✅ Done |
| **Day 3** | EDA, label counts, text lengths | 5 interactive Plotly charts in `charts/`, 5 written insights in `day3_eda_insights.md`. | ✅ Done |
| **Day 4** | Train/val/test split & Queue baseline | Stratified splits in `data/`, Logistic Regression baseline (**46.50% Macro-F1**), `day4_queue_baseline_report.md`. | ✅ Done |
| **Day 5** | Priority baseline, confusion matrices, results table | Priority baseline (**55.61% Macro-F1**), 2 confusion matrices, `results_table.csv` committed to Git. | ✅ Done |

---

## "Done When" Verification Check
- [x] Baseline model for `priority` trained and evaluated.
- [x] Visual confusion matrices generated for both `queue` and `priority`.
- [x] Off-diagonal error analysis documented.
- [x] Official Results Table saved as CSV and Markdown.
- [x] Results table committed to Git repository.
