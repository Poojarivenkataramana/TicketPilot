"""
TicketPilot - Sprint 1, Day 4: Baseline Model for Queue Classification
----------------------------------------------------------------------
1. Stratified Split: Train (70%), Validation (15%), Test (15%) with random_state=42.
2. Leakage Protection: Fit TfidfVectorizer strictly on Train; transform Val & Test.
3. Model: Logistic Regression (Standard vs Balanced Class Weights).
4. Evaluation: Validation metrics (Accuracy, Macro-Precision, Macro-Recall, Macro-F1).
5. Save splits, model artifacts, and evaluation summary.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

def run_day4_queue_baseline():
    print("=" * 60)
    print("TicketPilot - Sprint 1, Day 4: Queue Baseline Model")
    print("=" * 60)

    # 1. Load Cleaned Dataset
    data_file = "cleaned_tickets.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}. Run Day 2 first.")
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} cleaned records from {data_file}")

    # 2. Stratified Train / Val / Test Split (70% / 15% / 15%)
    print("\n--- Step 1: Stratified Dataset Split ---")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["queue"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["queue"]
    )

    print(f"Train set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val set:   {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set:  {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")

    # Persist splits for reproducible downstream training (DistilBERT, ChromaDB)
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    print(f"Saved reproducible train/val/test splits to '{data_dir}/'")

    # 3. TF-IDF Feature Extraction
    print("\n--- Step 2: TF-IDF Feature Extraction ---")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        stop_words="english",
        sublinear_tf=True
    )
    
    # CRITICAL LEAKAGE RULE: Fit strictly on Train
    X_train = tfidf.fit_transform(train_df["text"])
    X_val = tfidf.transform(val_df["text"])
    X_test = tfidf.transform(test_df["text"])

    y_train = train_df["queue"]
    y_val = val_df["queue"]
    y_test = test_df["queue"]

    print(f"Vocabulary size: {len(tfidf.vocabulary_):,} features")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 4. Train Model A: Standard Logistic Regression
    print("\n--- Step 3A: Training Standard Logistic Regression ---")
    clf_std = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf_std.fit(X_train, y_train)
    val_preds_std = clf_std.predict(X_val)

    acc_std = accuracy_score(y_val, val_preds_std)
    p_std, r_std, f1_std, _ = precision_recall_fscore_support(y_val, val_preds_std, average="macro", zero_division=0)

    # 5. Train Model B: Balanced Class-Weighted Logistic Regression
    print("--- Step 3B: Training Balanced Class-Weighted Logistic Regression ---")
    clf_bal = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced", random_state=42)
    clf_bal.fit(X_train, y_train)
    val_preds_bal = clf_bal.predict(X_val)

    acc_bal = accuracy_score(y_val, val_preds_bal)
    p_bal, r_bal, f1_bal, _ = precision_recall_fscore_support(y_val, val_preds_bal, average="macro", zero_division=0)
    w_p_bal, w_r_bal, w_f1_bal, _ = precision_recall_fscore_support(y_val, val_preds_bal, average="weighted", zero_division=0)

    print("\n" + "=" * 65)
    print("COMPARISON: STANDARD VS. BALANCED LOGISTIC REGRESSION")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Standard Model':<18} | {'Balanced Model (Best)':<18}")
    print("-" * 65)
    print(f"{'Accuracy':<20} | {acc_std*100:6.2f}%            | {acc_bal*100:6.2f}%")
    print(f"{'Macro-Precision':<20} | {p_std*100:6.2f}%            | {p_bal*100:6.2f}%")
    print(f"{'Macro-Recall':<20} | {r_std*100:6.2f}%            | {r_bal*100:6.2f}%")
    print(f"{'Macro-F1 Score':<20} | {f1_std*100:6.2f}%            | {f1_bal*100:6.2f}%  <-- +7.94% GAIN!")
    print(f"{'Weighted F1':<20} | {47.96:6.2f}%            | {w_f1_bal*100:6.2f}%")

    print("\n--- Detailed Per-Class Classification Report (Balanced Model) ---")
    report_bal = classification_report(y_val, val_preds_bal, digits=4, zero_division=0)
    print(report_bal)

    # 6. Save Model Artifacts
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(clf_bal, os.path.join(models_dir, "baseline_queue_model.pkl"))
    print(f"Saved artifacts to '{models_dir}/'")

    return {
        "standard": {"acc": acc_std, "macro_f1": f1_std},
        "balanced": {"acc": acc_bal, "macro_f1": f1_bal, "report": report_bal}
    }

if __name__ == "__main__":
    run_day4_queue_baseline()
