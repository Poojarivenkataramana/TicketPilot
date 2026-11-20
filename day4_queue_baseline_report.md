# Day 4: Baseline Model Report for Queue Classification

**Project**: TicketPilot  
**Sprint**: 1 (Understand Data & Build Baseline)  
**Deliverable**: Baseline model scores for `queue` classification, trained on TF-IDF + Logistic Regression with train/val/test splits.

---

## 1. Stratified Train / Validation / Test Split (Reproducible)

To prevent data leakage and guarantee that every evaluation is honest:
* Total Cleaned Dataset: **25,131 tickets**
* **Train Split (70.0%)**: **17,591 tickets** — Used exclusively for fitting TF-IDF vocabulary and learning model weights.
* **Validation Split (15.0%)**: **3,770 tickets** — Used for tuning, baseline benchmarking, and threshold decisions.
* **Test Split (15.0%)**: **3,770 tickets** — **Held out strictly untouched** until final Sprint evaluation.

All three splits were stratified on `queue` using `random_state=42` and saved to `data/` for all subsequent sprints.

---

## 2. Feature Extraction & Data Leakage Prevention

* **Vectorization**: `TfidfVectorizer` (unigrams + bigrams, `max_features=15,000`, `sublinear_tf=True`, English stop words removed).
* **Strict Anti-Leakage Rule**: The vectorizer was fitted **only on the 17,591 training examples**. Validation and test matrices were generated strictly via `.transform()`. No test text statistics leaked into the feature vocabulary.

---

## 3. Baseline Model Performance (Validation Set)

We evaluated two variations:
1. **Standard Logistic Regression**: Default unweighted loss.
2. **Balanced Logistic Regression**: Loss inversely proportional to class frequencies (`class_weight='balanced'`).

### Comparison Table:

| Metric | Standard Model | Balanced Model (Selected Baseline) | Impact / Why Balanced Wins |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 50.66% | 47.00% | Standard model inflated accuracy by over-predicting majority classes. |
| **Macro-Precision** | 64.02% | 43.42% | Balanced trades minor precision for dramatic recall gains. |
| **Macro-Recall** | 34.53% | **53.37%** | **+18.84% jump in finding tickets across all classes**. |
| **Macro-F1 Score** | 38.56% | **46.50%** | **+7.94% overall gain** across the 10 departments. |
| **Weighted F1 Score** | 47.96% | 47.22% | Remains consistent. |

---

## 4. Class-by-Class Analysis (Balanced Baseline Model)

| Queue Class | Precision | Recall | F1-Score | Support (Val) | Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Billing and Payments** | **0.7724** | **0.7480** | **0.7600** | 381 | Strongest performing queue; highly distinct vocabulary (*invoice*, *payment*). |
| **Service Outages & Maint.** | 0.4569 | **0.7162** | 0.5579 | 148 | High recall ensures critical outages are properly surfaced. |
| **Technical Support** | 0.5829 | 0.4349 | 0.4982 | 1,099 | Largest class; suffers cross-confusion with IT Support. |
| **General Inquiry** | 0.3232 | **0.5926** | **0.4183** | 54 | Standard model had **0.0% recall**; balanced model resurrected it to 59.3%! |
| **Returns and Exchanges** | 0.3561 | 0.5054 | 0.4178 | 186 | Steady performance across refund/exchange queries. |
| **IT Support** | 0.3785 | 0.4533 | 0.4125 | 450 | Cross-confusion with Product Support & Technical Support. |
| **Product Support** | 0.4617 | 0.3705 | 0.4111 | 699 | Moderate performance on hardware/software feature issues. |
| **Human Resources** | 0.3333 | 0.5211 | 0.4066 | 71 | Standard model had 15.5% recall; balanced model jumped to 52.1%. |
| **Sales and Pre-Sales** | 0.2829 | **0.6348** | 0.3914 | 115 | Recall jumped from 10.4% to 63.5%. |
| **Customer Service** | 0.3946 | 0.3598 | 0.3764 | 567 | Catch-all class; needs deep semantic embeddings in Sprint 2. |
| **Macro Average** | **0.4342** | **0.5337** | **0.4650** | **3,770** | **Honest 10-class baseline**. |

---

## 5. Artifacts Persisted
* Splits: `data/train.csv`, `data/val.csv`, `data/test.csv`
* Models: `models/tfidf_vectorizer.pkl` (652 KB), `models/baseline_queue_model.pkl` (1.1 MB)

## "Done When" Verification Check
- [x] Stratified train, val, test split established.
- [x] TF-IDF + Logistic Regression pipeline trained with strict leakage prevention.
- [x] Baseline metrics for `queue` calculated, reported, and compared (Accuracy: 47.00%, Macro-F1: 46.50%).
