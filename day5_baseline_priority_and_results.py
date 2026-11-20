"""
TicketPilot - Sprint 1, Day 5: Priority Baseline, Confusion Matrices & Results Table
-----------------------------------------------------------------------------------
1. Train Priority Baseline Model (TF-IDF + Logistic Regression).
2. Generate Confusion Matrices for both Queue (10x10) and Priority (3x3).
3. Export interactive Plotly heatmaps for error analysis.
4. Build and save the Unified Sprint 1 Results Table.
5. Save priority model artifact in 'models/'.
"""

import os
import joblib
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

def run_day5_priority_and_results():
    print("=" * 65)
    print("TicketPilot - Sprint 1, Day 5: Priority Baseline & Results Table")
    print("=" * 65)

    data_dir = "data"
    models_dir = "models"
    charts_dir = "charts"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Load Data Splits
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    print(f"Loaded splits: Train={len(train_df):,} | Val={len(val_df):,}")

    # Load previously fitted TF-IDF vectorizer
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError("Missing tfidf_vectorizer.pkl. Run Day 4 first.")
    tfidf = joblib.load(tfidf_path)

    X_train = tfidf.transform(train_df["text"])
    X_val = tfidf.transform(val_df["text"])

    # -------------------------------------------------------------
    # 2. Train Priority Baseline Model
    # -------------------------------------------------------------
    print("\n--- Step 1: Training Priority Baseline Model ---")
    y_train_p = train_df["priority"]
    y_val_p = val_df["priority"]

    # Compare standard vs balanced for priority
    clf_p_std = LogisticRegression(max_iter=1000, random_state=42)
    clf_p_std.fit(X_train, y_train_p)
    preds_p_std = clf_p_std.predict(X_val)

    clf_p_bal = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_p_bal.fit(X_train, y_train_p)
    preds_p_bal = clf_p_bal.predict(X_val)

    acc_p_std = accuracy_score(y_val_p, preds_p_std)
    _, _, f1_p_std, _ = precision_recall_fscore_support(y_val_p, preds_p_std, average="macro", zero_division=0)

    acc_p_bal = accuracy_score(y_val_p, preds_p_bal)
    p_p_bal, r_p_bal, f1_p_bal, _ = precision_recall_fscore_support(y_val_p, preds_p_bal, average="macro", zero_division=0)
    w_p_p, w_r_p, w_f1_p, _ = precision_recall_fscore_support(y_val_p, preds_p_bal, average="weighted", zero_division=0)

    print(f"Priority Standard Model: Accuracy={acc_p_std*100:.2f}% | Macro-F1={f1_p_std*100:.2f}%")
    print(f"Priority Balanced Model: Accuracy={acc_p_bal*100:.2f}% | Macro-F1={f1_p_bal*100:.2f}% (Selected)")

    print("\nDetailed Priority Classification Report (Balanced Model):")
    p_report = classification_report(y_val_p, preds_p_bal, digits=4, zero_division=0)
    print(p_report)

    # Save best priority model
    joblib.dump(clf_p_bal, os.path.join(models_dir, "baseline_priority_model.pkl"))
    print(f"Saved baseline priority model to '{models_dir}/baseline_priority_model.pkl'")

    # -------------------------------------------------------------
    # 3. Generate Confusion Matrices
    # -------------------------------------------------------------
    print("\n--- Step 2: Generating Confusion Matrices ---")

    # A. Queue Confusion Matrix
    clf_q = joblib.load(os.path.join(models_dir, "baseline_queue_model.pkl"))
    y_val_q = val_df["queue"]
    preds_q = clf_q.predict(X_val)
    queue_labels = sorted(list(y_val_q.unique()))

    cm_q = confusion_matrix(y_val_q, preds_q, labels=queue_labels)
    cm_q_norm = confusion_matrix(y_val_q, preds_q, labels=queue_labels, normalize="true")

    fig_cm_q = px.imshow(
        cm_q_norm,
        x=queue_labels,
        y=queue_labels,
        color_continuous_scale="Viridis",
        labels=dict(x="Predicted Queue", y="True Queue", color="Normalized Rate"),
        title="Confusion Matrix: Queue Classification (Normalized by True Class)"
    )
    fig_cm_q.update_layout(xaxis_tickangle=-45, height=750, width=850, template="plotly_white")
    cm_q_path = os.path.join(charts_dir, "confusion_matrix_queue.html")
    fig_cm_q.write_html(cm_q_path)
    print(f"Saved: {cm_q_path}")

    # B. Priority Confusion Matrix
    priority_labels = ["high", "medium", "low"]
    cm_p = confusion_matrix(y_val_p, preds_p_bal, labels=priority_labels)
    cm_p_norm = confusion_matrix(y_val_p, preds_p_bal, labels=priority_labels, normalize="true")

    fig_cm_p = px.imshow(
        cm_p_norm,
        x=priority_labels,
        y=priority_labels,
        text_auto=".2%",
        color_continuous_scale="Reds",
        labels=dict(x="Predicted Priority", y="True Priority", color="Normalized Rate"),
        title="Confusion Matrix: Priority Classification (Normalized by True Class)"
    )
    fig_cm_p.update_layout(height=500, width=600, template="plotly_white")
    cm_p_path = os.path.join(charts_dir, "confusion_matrix_priority.html")
    fig_cm_p.write_html(cm_p_path)
    print(f"Saved: {cm_p_path}")

    # -------------------------------------------------------------
    # 4. Build Unified Results Table
    # -------------------------------------------------------------
    print("\n--- Step 3: Compiling Unified Results Table ---")
    
    # Load Queue Metrics from Day 4
    acc_q = accuracy_score(y_val_q, preds_q)
    p_q, r_q, f1_q, _ = precision_recall_fscore_support(y_val_q, preds_q, average="macro", zero_division=0)
    _, _, w_f1_q, _ = precision_recall_fscore_support(y_val_q, preds_q, average="weighted", zero_division=0)

    results_data = [
        {
            "Sprint": "Sprint 1 (Baseline)",
            "Model": "TF-IDF + Logistic Regression (Balanced)",
            "Target Task": "Queue (10 Classes)",
            "Accuracy": f"{acc_q * 100:.2f}%",
            "Macro Precision": f"{p_q * 100:.2f}%",
            "Macro Recall": f"{r_q * 100:.2f}%",
            "Macro-F1": f"{f1_q * 100:.2f}%",
            "Weighted F1": f"{w_f1_q * 100:.2f}%"
        },
        {
            "Sprint": "Sprint 1 (Baseline)",
            "Model": "TF-IDF + Logistic Regression (Balanced)",
            "Target Task": "Priority (3 Classes)",
            "Accuracy": f"{acc_p_bal * 100:.2f}%",
            "Macro Precision": f"{p_p_bal * 100:.2f}%",
            "Macro Recall": f"{r_p_bal * 100:.2f}%",
            "Macro-F1": f"{f1_p_bal * 100:.2f}%",
            "Weighted F1": f"{w_f1_p * 100:.2f}%"
        }
    ]

    results_df = pd.DataFrame(results_data)
    results_csv_path = "results_table.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Exported: {results_csv_path}")

    print("\n" + "=" * 70)
    print("SPRINT 1 OFFICIAL BASELINE RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return {
        "results_table": results_df,
        "queue_cm": cm_q,
        "priority_cm": cm_p
    }

if __name__ == "__main__":
    run_day5_priority_and_results()
