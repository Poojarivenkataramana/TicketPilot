"""
TicketPilot - Sprint 2, Day 10: Error Analysis & Confidence Threshold Testing
-----------------------------------------------------------------------------
1. Extract 50 misclassified tickets from the held-out test set.
2. Group the 50 wrong predictions into 5 root-cause failure categories:
   - Category A: Technical Support vs IT Support Semantic Overlap
   - Category B: Multi-Intent / Blended Department Queries
   - Category C: Vague / Low-Information Short Queries
   - Category D: Subject-Body Disconnect / Mislabeled Edge Cases
   - Category E: Service Outages vs Technical Bug Ambiguity
3. Simulate Confidence Thresholds (0.10 to 0.70) measuring:
   - Auto-Triage Rate (Coverage %)
   - Human Escalation Rate (%)
   - High-Precision Accuracy on Auto-Triaged Tickets (%)
4. Export interactive Plotly curve (charts/chart10_confidence_threshold_tradeoff.html).
5. Produce Day 10 One-Page Findings Note.
"""

import os
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

def run_day10_error_analysis():
    print("=" * 70)
    print("TicketPilot - Sprint 2, Day 10: Deep Error Analysis & Thresholding")
    print("=" * 70)

    data_dir = "data"
    charts_dir = "charts"
    models_dir = "models"
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Load Test Set and Classical/Transformer Models
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    clf_svm = joblib.load(os.path.join(models_dir, "best_classical_linear_svm_queue.pkl"))
    clf_lr = joblib.load(os.path.join(models_dir, "baseline_queue_model.pkl"))

    X_test = tfidf.transform(test_df["text"])
    y_test = test_df["queue"]

    # Calibrated probabilities from Logistic Regression and SVM
    probs_lr = clf_lr.predict_proba(X_test)
    preds_lr = clf_lr.predict(X_test)
    conf_lr = np.max(probs_lr, axis=1)

    preds_svm = clf_svm.predict(X_test)
    dec_svm = clf_svm.decision_function(X_test)
    probs_svm = np.exp(dec_svm) / np.sum(np.exp(dec_svm), axis=1, keepdims=True)
    conf_svm = np.max(probs_svm, axis=1)

    # Create full test analysis dataframe
    analysis_df = pd.DataFrame({
        "text": test_df["text"],
        "true_queue": y_test,
        "pred_svm": preds_svm,
        "conf_svm": conf_svm.round(4),
        "correct_svm": (y_test == preds_svm),
        "pred_lr": preds_lr,
        "conf_lr": conf_lr.round(4),
        "correct_lr": (y_test == preds_lr)
    })

    # -------------------------------------------------------------
    # 2. Extract and Audit 50 Wrong Predictions
    # -------------------------------------------------------------
    print("\n--- Step 1: Auditing 50 Misclassified Test Tickets ---")
    wrong_tickets = analysis_df[~analysis_df["correct_svm"]].copy().reset_index(drop=True)
    sample_50 = wrong_tickets.head(50)
    print(f"Total wrong predictions in test set (Linear SVM): {len(wrong_tickets):,} / {len(test_df):,}")

    # Categorize the 50 errors into 5 buckets
    buckets = {
        "A. Technical vs. IT Support Lexical Overlap": 0,
        "B. Product Support vs. Customer Service Ambiguity": 0,
        "C. Outage vs. Incident Disruption Confusion": 0,
        "D. Multi-Intent / Blended Department Queries": 0,
        "E. Vague / Ultra-Short Queries": 0
    }

    categorized_examples = []
    for idx, row in sample_50.iterrows():
        true_q = row["true_queue"]
        pred_q = row["pred_svm"]
        text_lower = row["text"].lower()

        if (true_q in ["Technical Support", "IT Support"]) and (pred_q in ["Technical Support", "IT Support"]):
            cat = "A. Technical vs. IT Support Lexical Overlap"
        elif (true_q in ["Product Support", "Customer Service"]) and (pred_q in ["Product Support", "Customer Service"]):
            cat = "B. Product Support vs. Customer Service Ambiguity"
        elif "outage" in text_lower or "disruption" in text_lower or "downtime" in text_lower:
            cat = "C. Outage vs. Incident Disruption Confusion"
        elif len(row["text"].split()) < 30:
            cat = "E. Vague / Ultra-Short Queries"
        else:
            cat = "D. Multi-Intent / Blended Department Queries"

        buckets[cat] += 1
        if len(categorized_examples) < 15:
            categorized_examples.append({
                "true": true_q,
                "pred": pred_q,
                "conf": row["conf_svm"],
                "category": cat,
                "snippet": row["text"][:110] + "..."
            })

    print("\nDistribution of 50 Errors Across Root-Cause Categories:")
    for b_name, count in buckets.items():
        pct = (count / 50) * 100
        print(f"  {b_name:<50}: {count:2d} ({pct:.1f}%)")

    # -------------------------------------------------------------
    # 3. Confidence Threshold Testing & Routing Simulation
    # -------------------------------------------------------------
    print("\n--- Step 2: Simulating Production Confidence Thresholds ---")
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    thresh_results = []

    for tau in thresholds:
        # Tickets where confidence >= tau are Auto-Triaged
        auto_mask = analysis_df["conf_lr"] >= tau
        auto_count = auto_mask.sum()
        escalated_count = len(analysis_df) - auto_count

        coverage_pct = (auto_count / len(analysis_df)) * 100
        escalation_pct = (escalated_count / len(analysis_df)) * 100

        if auto_count > 0:
            auto_acc = (analysis_df[auto_mask]["correct_lr"].sum() / auto_count) * 100
        else:
            auto_acc = 100.0

        thresh_results.append({
            "Threshold (tau)": f"{tau:.2f}",
            "Auto-Triage Rate (Coverage)": f"{coverage_pct:.1f}%",
            "Human Review Rate": f"{escalation_pct:.1f}%",
            "Accuracy on Auto-Triaged": f"{auto_acc:.2f}%",
            "coverage_num": coverage_pct,
            "accuracy_num": auto_acc
        })

    thresh_df = pd.DataFrame(thresh_results)
    print("\nCONFIDENCE THRESHOLD SIMULATION TABLE (LOGISTIC REGRESSION / SOFTMAX):")
    print(thresh_df[["Threshold (tau)", "Auto-Triage Rate (Coverage)", "Human Review Rate", "Accuracy on Auto-Triaged"]].to_string(index=False))

    # -------------------------------------------------------------
    # 4. Generate Interactive Plotly Trade-off Curve
    # -------------------------------------------------------------
    fig = go.Figure()
    taus = [float(r["Threshold (tau)"]) for r in thresh_results]
    coverages = [r["coverage_num"] for r in thresh_results]
    accuracies = [r["accuracy_num"] for r in thresh_results]

    fig.add_trace(go.Scatter(
        x=taus, y=accuracies,
        mode="lines+markers",
        name="Accuracy on Auto-Triaged Tickets",
        line=dict(color="#00CC96", width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=taus, y=coverages,
        mode="lines+markers",
        name="Auto-Triage Coverage Rate (% Tickets Automated)",
        line=dict(color="#636EFA", width=3, dash="dot"),
        marker=dict(size=8)
    ))

    # Recommended threshold vertical line (e.g. tau=0.35)
    fig.add_vline(x=0.35, line_dash="dash", line_color="#EF553B", annotation_text="Recommended Threshold (tau=0.35)", annotation_position="top left")

    fig.update_layout(
        title="Day 10: Confidence Threshold Trade-Off (Coverage vs. Accuracy)",
        xaxis_title="Confidence Threshold (Softmax Probability)",
        yaxis_title="Percentage (%)",
        template="plotly_white",
        height=550,
        width=850
    )
    chart10_path = os.path.join(charts_dir, "chart10_confidence_threshold_tradeoff.html")
    fig.write_html(chart10_path)
    print(f"\nSaved threshold trade-off curve to: {chart10_path}")

    return {
        "buckets": buckets,
        "sample_errors": categorized_examples,
        "threshold_table": thresh_df
    }

if __name__ == "__main__":
    run_day10_error_analysis()
