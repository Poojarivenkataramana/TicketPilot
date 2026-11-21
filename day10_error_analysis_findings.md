# Day 10: Error Analysis & Confidence Threshold Findings Note

**Project**: TicketPilot  
**Sprint**: 2 Completion (Better Models & Error Analysis)  
**Deliverables**:
1. Root-Cause Audit of 50 Misclassified Test Tickets
2. Confidence Threshold Simulation Table & Production Policy
3. Interactive Trade-off Curve: `charts/chart10_confidence_threshold_tradeoff.html`

---

## 1. Qualitative Audit: Grouping 50 Misclassified Tickets

We conducted a line-by-line manual audit of 50 randomly selected misclassified tickets from the held-out test set (`data/test_predictions_distilbert.csv`) to understand *why* the models failed.

### The 5 Root-Cause Failure Categories:

| Failure Category | % of Errors | Root Cause & Real Test Sample |
| :--- | :---: | :--- |
| **1. Multi-Intent / Blended Queries** | **68.0%** | **Customer asks about two domains at once.**<br>*Example*: *"Request for Information on API Integration Capabilities..."* (True: `Technical Support` \| Predicted: `Sales and Pre-Sales`, Conf: 22.0%). A prospective customer evaluating technical APIs before buying bridges both departments. |
| **2. Vague / Ultra-Short Queries** | **22.0%** | **Insufficient semantic context for NLP models.**<br>*Example*: *"Assistance Needed. Urgent help required with access."* (True: `IT Support` \| Predicted: `Customer Service`, Conf: 18.2%). Lacks department-specific keywords. |
| **3. Tech Support vs. IT Support Ambiguity** | **4.0%** | **Identical vocabulary (*database*, *server*, *login*, *slow*).**<br>*Example*: *"Encountering slow performance with SaaS tool due to high database load..."* (True: `Technical Support` \| Predicted: `IT Support`, Conf: 24.9%). |
| **4. Product Support vs. Pre-Sales Inquiries** | **4.0%** | **Questions regarding feature capabilities.**<br>*Example*: *"Combine Keras and Smartsheet for Marketing Tools..."* (True: `Product Support` \| Predicted: `Sales and Pre-Sales`, Conf: 17.3%). |
| **5. Outage vs. Defect Confusion** | **2.0%** | **High-severity technical bugs presenting like downtime.**<br>*Example*: *"Reported System Failure: sporadic system failures during trading hours..."* (True: `Technical Support` \| Predicted: `Service Outages and Maintenance`, Conf: 33.8%). |

---

## 2. Confidence Threshold Simulation & Production Routing Policy

In customer support automation, **a model should never guess blindly**. By establishing a confidence threshold $\tau$, tickets can be bifurcated into:
1. **Auto-Triaged ($\text{Confidence} \ge \tau$)**: Routed instantly with high precision.
2. **Human Review ($\text{Confidence} < \tau$)**: Sent to a human coordinator for manual assignment.

### Simulation Results Across Thresholds (Held-Out Test Set):

| Threshold ($\tau$) | Auto-Triage Coverage (% Automated) | Human Escalation Rate | Accuracy on Auto-Triaged Tickets | Impact / Trade-off |
| :---: | :---: | :---: | :---: | :--- |
| **None (0.15)** | 99.6% | 0.4% | 46.67% | High volume, low accuracy (raw baseline). |
| **0.25** | 62.0% | 38.0% | 57.53% | +10.8% accuracy gain. |
| **0.30** | 40.8% | 59.2% | 65.39% | Solid balance for non-critical routing. |
| **0.35 (Recommended)** 🎯 | **26.6%** | **73.4%** | **73.53%** | **Sweet spot: 1 in 4 tickets automated with ~74% accuracy!** |
| **0.40** | 18.6% | 81.4% | **80.37%** | High-reliability tier (>80% accuracy). |
| **0.50** | 11.6% | 88.4% | **89.29%** | Near-human accuracy on clear-cut tickets. |
| **0.60** | 9.1% | 90.9% | **93.60%** | Ultra-high confidence corporate accounts. |

*Interactive Visual Trade-off*: [chart10_confidence_threshold_tradeoff.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart10_confidence_threshold_tradeoff.html)

---

## 3. Recommended Production Policy for TicketPilot

Based on this audit, we establish the **Two-Tier Triage Architecture** for Sprint 5:
* **Rule 1 (Auto-Triage Tier)**: If $\text{Confidence} \ge 0.35$, immediately route the ticket to the predicted `queue` and trigger RAG auto-drafting.
* **Rule 2 (Human Fallback Tier)**: If $\text{Confidence} < 0.35$, route the ticket to the general human triage pool with the model's top-2 predictions flagged as suggestions.
* **Rule 3 (Outage Priority Bypass)**: If priority is classified as `High` or keywords mention *"outage / breach"*, immediately notify Tier-3 on-call regardless of queue confidence score.

---

## 4. Sprint 2 Retrospective (Days 6 to 10)

| Day | Goal | Deliverable | Status |
| :---: | :--- | :--- | :---: |
| **Day 6** | 3 classical models with 5-fold CV | `day6_classical_models.py` (Linear SVM achieved **54.50% Macro-F1**, beating Logistic Regression by +8.0%). | ✅ Done |
| **Day 7** | Sentence embeddings feature extraction | `all-MiniLM-L6-v2` dense vectors cached in `data/`; discovered frozen embeddings dilute key support tokens (**30.25% F1**). | ✅ Done |
| **Day 8** | DistilBERT fine-tuning without overfitting | `day8_fine_tune_distilbert.py` with inverse class weights + early stopping + Colab GPU notebook. | ✅ Done |
| **Day 9** | Test set evaluation & confidence scores | Evaluated on held-out test set; proved confidence gap (**45.8% correct vs 26.6% wrong**); updated results table. | ✅ Done |
| **Day 10** | Error analysis & threshold testing | Audited 50 errors across 5 root-cause buckets; proved $\tau=0.35$ yields **73.5% accuracy**; produced findings note. | ✅ Done |

---

## "Done When" Verification Check
- [x] 50 wrong predictions audited and categorized into root-cause buckets.
- [x] Confidence threshold trade-off simulated across $\tau \in [0.15, 0.60]$.
- [x] Interactive Plotly curve generated in `charts/`.
- [x] One-page findings note produced and committed to Git.
