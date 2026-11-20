"""
TicketPilot - Sprint 2, Day 6: Classical Model Comparison (Logistic Regression vs. Linear SVM vs. Naive Bayes)
-------------------------------------------------------------------------------------------------------------
1. Evaluate 3 classical models on Queue classification:
   - Model 1: Logistic Regression (class_weight='balanced')
   - Model 2: Linear Support Vector Machine (LinearSVC, class_weight='balanced')
   - Model 3: Multinomial Naive Bayes (MultinomialNB, alpha=0.1)
2. Run 5-Fold Stratified Cross-Validation on the Training set.
3. Evaluate on the fixed Validation set (Accuracy, Macro-Precision, Macro-Recall, Macro-F1).
4. Export interactive comparison chart (Plotly) and update 'results_table.csv'.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

def run_day6_classical_comparison():
    print("=" * 70)
    print("TicketPilot - Sprint 2, Day 6: 3 Classical Models Comparison")
    print("=" * 70)

    data_dir = "data"
    models_dir = "models"
    charts_dir = "charts"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Load Data Splits & Vectorizer
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    print(f"Loaded splits: Train={len(train_df):,} | Val={len(val_df):,}")

    tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    X_train = tfidf.transform(train_df["text"])
    X_val = tfidf.transform(val_df["text"])
    y_train = train_df["queue"]
    y_val = val_df["queue"]

    # 2. Define the 3 Classical Models
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        ),
        "Linear SVM (LinearSVC)": LinearSVC(
            class_weight="balanced", C=0.5, random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.1
        )
    }

    # 3. 5-Fold Stratified Cross-Validation on Training Data
    print("\n--- Step 1: Running 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro"]

    cv_results = {}
    for name, model in models.items():
        print(f"Evaluating {name} with 5-fold CV...")
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_acc_mean = scores["test_accuracy"].mean()
        cv_acc_std = scores["test_accuracy"].std()
        cv_f1_mean = scores["test_f1_macro"].mean()
        cv_f1_std = scores["test_f1_macro"].std()
        cv_results[name] = {
            "cv_acc_mean": cv_acc_mean,
            "cv_acc_std": cv_acc_std,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std
        }
        print(f"  -> CV Accuracy: {cv_acc_mean*100:.2f}% (±{cv_acc_std*100:.2f}%) | CV Macro-F1: {cv_f1_mean*100:.2f}% (±{cv_f1_std*100:.2f}%)")

    # 4. Train on Full Training Set and Evaluate on Validation Set
    print("\n--- Step 2: Training on Full Train Set & Evaluating on Val Set ---")
    val_results = []
    trained_models = {}

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        trained_models[name] = model

        # Inference speed test (3,770 val samples)
        start_inf = time.time()
        val_preds = model.predict(X_val)
        inf_time_per_ticket = ((time.time() - start_inf) / len(val_df)) * 1000  # ms

        acc = accuracy_score(y_val, val_preds)
        p, r, f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
        w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="weighted", zero_division=0)

        val_results.append({
            "Sprint": "Sprint 2 (Classical)",
            "Model": name,
            "Target Task": "Queue (10 Classes)",
            "CV Macro-F1": f"{cv_results[name]['cv_f1_mean']*100:.2f}% (±{cv_results[name]['cv_f1_std']*100:.2f}%)",
            "Accuracy": f"{acc * 100:.2f}%",
            "Macro Precision": f"{p * 100:.2f}%",
            "Macro Recall": f"{r * 100:.2f}%",
            "Macro-F1": f"{f1 * 100:.2f}%",
            "Weighted F1": f"{w_f1 * 100:.2f}%",
            "Inference Latency": f"{inf_time_per_ticket:.3f} ms"
        })

    comparison_df = pd.DataFrame(val_results)
    print("\n" + "=" * 80)
    print("3 CLASSICAL MODELS COMPARISON TABLE (QUEUE CLASSIFICATION)")
    print("=" * 80)
    print(comparison_df[["Model", "CV Macro-F1", "Accuracy", "Macro-F1", "Macro Recall", "Inference Latency"]].to_string(index=False))

    # Save best performing classical model
    best_model_name = "Linear SVM (LinearSVC)"
    joblib.dump(trained_models[best_model_name], os.path.join(models_dir, "best_classical_linear_svm_queue.pkl"))
    print(f"\nSaved best classical model ({best_model_name}) to '{models_dir}/best_classical_linear_svm_queue.pkl'")

    # 5. Export Visual Plotly Comparison Chart
    fig = go.Figure()
    model_names = [r["Model"] for r in val_results]
    macro_f1_scores = [float(r["Macro-F1"].replace("%", "")) for r in val_results]
    accuracies = [float(r["Accuracy"].replace("%", "")) for r in val_results]
    recalls = [float(r["Macro Recall"].replace("%", "")) for r in val_results]

    fig.add_trace(go.Bar(name="Macro-F1 Score", x=model_names, y=macro_f1_scores, marker_color="#636EFA", text=[f"{s:.2f}%" for s in macro_f1_scores], textposition="auto"))
    fig.add_trace(go.Bar(name="Macro Recall", x=model_names, y=recalls, marker_color="#00CC96", text=[f"{s:.2f}%" for s in recalls], textposition="auto"))
    fig.add_trace(go.Bar(name="Overall Accuracy", x=model_names, y=accuracies, marker_color="#AB63FA", text=[f"{s:.2f}%" for s in accuracies], textposition="auto"))

    fig.update_layout(
        barmode="group",
        title="Day 6: Comparison of 3 Classical ML Models on Queue Classification",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 70],
        template="plotly_white",
        height=550,
        width=850
    )
    chart6_path = os.path.join(charts_dir, "chart6_classical_models_comparison.html")
    fig.write_html(chart6_path)
    print(f"Saved interactive comparison chart to: {chart6_path}")

    # 6. Update results_table.csv
    existing_results = pd.read_csv("results_table.csv")
    # Avoid duplicate additions
    updated_results = pd.concat([existing_results, comparison_df[existing_results.columns]], ignore_index=True)
    updated_results.drop_duplicates(subset=["Model", "Target Task"], keep="last", inplace=True)
    updated_results.to_csv("results_table.csv", index=False)
    print(f"Updated results_table.csv with Day 6 classical models.")

    return comparison_df

if __name__ == "__main__":
    run_day6_classical_comparison()
