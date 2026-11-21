"""
TicketPilot - Sprint 2, Day 7: Sentence Embeddings Feature Extraction & Classification
-------------------------------------------------------------------------------------
1. Load Train and Validation sets.
2. Generate dense vector representations using all-MiniLM-L6-v2 (384 dimensions).
3. Cache embeddings to disk (data/train_emb_minilm.npy, data/val_emb_minilm.npy).
4. Train downstream classification heads (Logistic Regression & Linear SVM) on embeddings.
5. Evaluate on Validation set (Accuracy, Macro-Precision, Macro-Recall, Macro-F1).
6. Compare with TF-IDF baselines and update results_table.csv.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

def run_day7_embeddings():
    print("=" * 70)
    print("TicketPilot - Sprint 2, Day 7: Sentence Embeddings Classification")
    print("=" * 70)

    data_dir = "data"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    print(f"Loaded splits: Train={len(train_df):,} | Val={len(val_df):,}")

    train_emb_path = os.path.join(data_dir, "train_emb_minilm.npy")
    val_emb_path = os.path.join(data_dir, "val_emb_minilm.npy")

    # 2. Compute or Load Cached Sentence Embeddings
    model_name = "all-MiniLM-L6-v2"
    if os.path.exists(train_emb_path) and os.path.exists(val_emb_path):
        print(f"\nLoading cached embeddings from {data_dir}/...")
        X_train_emb = np.load(train_emb_path)
        X_val_emb = np.load(val_emb_path)
        print(f"Loaded train embeddings shape: {X_train_emb.shape}")
        print(f"Loaded val embeddings shape:   {X_val_emb.shape}")
    else:
        print(f"\nLoading SentenceTransformer model: '{model_name}'...")
        embedder = SentenceTransformer(model_name)
        print(f"Embedding dimension: {embedder.get_sentence_embedding_dimension()}")

        print(f"\nEncoding {len(train_df):,} training tickets...")
        start_time = time.time()
        X_train_emb = embedder.encode(
            train_df["text"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"Train encoding finished in {time.time() - start_time:.1f}s. Shape: {X_train_emb.shape}")

        print(f"\nEncoding {len(val_df):,} validation tickets...")
        start_val = time.time()
        X_val_emb = embedder.encode(
            val_df["text"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"Val encoding finished in {time.time() - start_val:.1f}s. Shape: {X_val_emb.shape}")

        # Cache to disk
        np.save(train_emb_path, X_train_emb)
        np.save(val_emb_path, X_val_emb)
        print(f"Cached dense vectors to '{data_dir}/'")

    y_train = train_df["queue"]
    y_val = val_df["queue"]

    # 3. Train Classifiers on Dense Embeddings
    print("\n--- Training Classifiers on Dense Embeddings ---")
    
    # Head 1: Logistic Regression on Dense Vectors
    print("1. Training Logistic Regression (Balanced) on embeddings...")
    clf_lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    clf_lr.fit(X_train_emb, y_train)
    val_preds_lr = clf_lr.predict(X_val_emb)

    acc_lr = accuracy_score(y_val, val_preds_lr)
    p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(y_val, val_preds_lr, average="macro", zero_division=0)
    w_p_lr, w_r_lr, w_f1_lr, _ = precision_recall_fscore_support(y_val, val_preds_lr, average="weighted", zero_division=0)

    # Head 2: Linear SVM on Dense Vectors
    print("2. Training Linear SVM (Balanced) on embeddings...")
    clf_svm = LinearSVC(class_weight="balanced", C=0.5, random_state=42)
    clf_svm.fit(X_train_emb, y_train)
    val_preds_svm = clf_svm.predict(X_val_emb)

    acc_svm = accuracy_score(y_val, val_preds_svm)
    p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_val, val_preds_svm, average="macro", zero_division=0)
    w_p_svm, w_r_svm, w_f1_svm, _ = precision_recall_fscore_support(y_val, val_preds_svm, average="weighted", zero_division=0)

    # Select best head
    if f1_svm >= f1_lr:
        best_name = "MiniLM Embeddings + Linear SVM"
        best_model = clf_svm
        best_preds = val_preds_svm
        best_acc, best_p, best_r, best_f1, best_w_f1 = acc_svm, p_svm, r_svm, f1_svm, w_f1_svm
    else:
        best_name = "MiniLM Embeddings + Logistic Regression"
        best_model = clf_lr
        best_preds = val_preds_lr
        best_acc, best_p, best_r, best_f1, best_w_f1 = acc_lr, p_lr, r_lr, f1_lr, w_f1_lr

    print("\n" + "=" * 70)
    print("DAY 7: SENTENCE EMBEDDINGS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {best_name}")
    print(f"Accuracy:        {best_acc * 100:.2f}%")
    print(f"Macro-Precision: {best_p * 100:.2f}%")
    print(f"Macro-Recall:    {best_r * 100:.2f}%")
    print(f"Macro-F1 Score:  {best_f1 * 100:.2f}%")
    print(f"Weighted F1:     {best_w_f1 * 100:.2f}%")

    print("\nDetailed Classification Report:")
    report = classification_report(y_val, best_preds, digits=4, zero_division=0)
    print(report)

    # Save artifact
    joblib.dump(best_model, os.path.join(models_dir, "embedding_classifier_queue.pkl"))
    print(f"Saved best embedding classifier to '{models_dir}/embedding_classifier_queue.pkl'")

    # Update results_table.csv
    results_path = "results_table.csv"
    existing_df = pd.read_csv(results_path)
    new_entry = pd.DataFrame([{
        "Sprint": "Sprint 2 (Embeddings)",
        "Model": best_name,
        "Target Task": "Queue (10 Classes)",
        "Accuracy": f"{best_acc * 100:.2f}%",
        "Macro Precision": f"{best_p * 100:.2f}%",
        "Macro Recall": f"{best_r * 100:.2f}%",
        "Macro-F1": f"{best_f1 * 100:.2f}%",
        "Weighted F1": f"{best_w_f1 * 100:.2f}%"
    }])
    updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
    updated_df.drop_duplicates(subset=["Model", "Target Task"], keep="last", inplace=True)
    updated_df.to_csv(results_path, index=False)
    print(f"Updated '{results_path}' with Day 7 embedding model scores.")

    return {
        "accuracy": best_acc,
        "macro_f1": best_f1,
        "report": report
    }

if __name__ == "__main__":
    run_day7_embeddings()
