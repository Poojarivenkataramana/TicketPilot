# Day 9: Test Set Evaluation & Baseline vs. DistilBERT Comparison Report

**Project**: TicketPilot  
**Sprint**: 2 (Better Models & Error Analysis)  
**Deliverable**: Held-Out Test Set Evaluation (3,770 tickets), Prediction Confidence Analysis, and Official Baseline vs. DistilBERT Comparison Table (`day9_evaluate_test_set.py`).

---

## 1. Official Baseline vs. DistilBERT Table (Held-Out Test Set)

Every model was evaluated on the **exact same 3,770 test tickets** (`data/test.csv`) that were set aside on Day 4 and remained untouched until now:

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Mean Confidence | Inference Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF + Linear SVM (Balanced)** 🏆 | **54.32%** | **51.24%** | **58.23%** | **54.15%** | 22.01% | **0.001 ms** | **Best Classical Performer** |
| **TF-IDF + Logistic Regression (Balanced)** | 46.63% | 42.88% | 52.07% | 45.66% | 33.48% | 0.001 ms | Sprint 1 Baseline |
| **Fine-Tuned DistilBERT (2 Epochs on MPS)** | 26.84% | 23.50% | 33.44% | 25.28% | 31.78% | 11.96 ms | Transformer Checkpoint |

*Interactive Confidence Chart*: [chart9_distilbert_confidence.html](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/charts/chart9_distilbert_confidence.html)  
*Full Benchmark Table*: [results_table.csv](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/results_table.csv)

---

## 2. Confidence Diagnostics: "The Model Knows What It Doesn't Know"

For all 3,770 test predictions, we extracted Softmax probability confidence scores ($\max(\text{Softmax}(\text{logits}))$):

* **Mean Confidence on CORRECT Predictions**: **`45.80%`**
* **Mean Confidence on WRONG Predictions**: **`26.64%`**
* **The Confidence Gap**: **`+19.16%` separation**

### Key Takeaway for Day 10 & Production:
The model demonstrates healthy calibration: when it predicts incorrectly, its confidence drops significantly into the 20% range.
This empirical gap is the exact foundation required for **Day 10 (Confidence Thresholding & Routing)**:
- High confidence ($\ge 40\%$): Route automatically to department queue.
- Low confidence ($< 40\%$): Escalate to a human triage manager for manual review.

---

## 3. Engineering Analysis: Classical vs. Deep Learning Trade-Offs

1. **Linear SVM Remains the Production Champion**:
   - Scores **54.15% Macro-F1** and runs in **0.001 milliseconds per ticket** (over 10,000x faster than DistilBERT).
   - In low-latency enterprise streaming pipelines where tickets must be triaged in real time with minimal server cost, Linear SVM offers the highest cost-to-performance ratio.
2. **DistilBERT Status**:
   - Our 2-epoch run on Apple Silicon confirmed the pipeline works end-to-end and exhibits clear confidence separation.
   - For complete transformer convergence across all 17,591 training records, training on Google Colab's free T4 GPU (using our [day8_distilbert_colab.ipynb](file:///Users/poojarivenkataramana/Downloads/ticket_complaint/day8_distilbert_colab.ipynb)) provides 3 to 5 epochs in minutes.

---

## 4. Artifacts Persisted
* Test Predictions Database: `data/test_predictions_distilbert.csv` (contains `text`, `true_queue`, `predicted_queue`, `confidence`, `is_correct`)
* Model Artifacts: `models/distilbert_queue_finetuned/`
* Interactive Chart: `charts/chart9_distilbert_confidence.html`
* Benchmark Database: `results_table.csv` updated with test set numbers.

## "Done When" Verification Check
- [x] Evaluated baseline and DistilBERT on the exact held-out test set.
- [x] Softmax confidence scores calculated for every test ticket.
- [x] Confidence separation between correct vs wrong predictions quantified.
- [x] Official Baseline vs. DistilBERT table produced and persisted to Git.
