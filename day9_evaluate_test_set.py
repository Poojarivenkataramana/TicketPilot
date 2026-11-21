"""
TicketPilot - Sprint 2, Day 9: Test Set Evaluation & Confidence Scores
---------------------------------------------------------------------
1. Evaluate all models on the UNTOUCHED Test Set (data/test.csv - 3,770 tickets).
2. Train/Fine-tune DistilBERT and run forward inference with Softmax probabilities.
3. Compute Softmax Confidence Scores for each test ticket.
4. Output 'data/test_predictions_distilbert.csv' for Day 10 Error Analysis.
5. Generate the official 'Baseline vs. DistilBERT' comparison table.
6. Export visual Plotly confidence distribution chart.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
import plotly.express as px
import plotly.graph_objects as go

# Custom Dataset for PyTorch inference
class EvalDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["input_ids"])

def run_day9_test_evaluation():
    print("=" * 70)
    print("TicketPilot - Sprint 2, Day 9: Test Set Evaluation & Confidence Scores")
    print("=" * 70)

    data_dir = "data"
    models_dir = "models"
    charts_dir = "charts"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Load Data Splits
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    print(f"Loaded Untouched Test Set: {len(test_df):,} records")

    y_test = test_df["queue"]

    # 2. Evaluate Classical Baselines on the TEST Set
    print("\n--- Step 1: Evaluating Classical Baselines on Test Set ---")
    tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    X_test_tfidf = tfidf.transform(test_df["text"])

    # Baseline 1: Logistic Regression
    clf_lr = joblib.load(os.path.join(models_dir, "baseline_queue_model.pkl"))
    start_lr = time.time()
    preds_lr = clf_lr.predict(X_test_tfidf)
    probs_lr = clf_lr.predict_proba(X_test_tfidf)
    conf_lr = np.max(probs_lr, axis=1)
    lr_latency = ((time.time() - start_lr) / len(test_df)) * 1000

    acc_lr = accuracy_score(y_test, preds_lr)
    p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(y_test, preds_lr, average="macro", zero_division=0)
    _, _, w_f1_lr, _ = precision_recall_fscore_support(y_test, preds_lr, average="weighted", zero_division=0)

    # Baseline 2: Linear SVM
    clf_svm = joblib.load(os.path.join(models_dir, "best_classical_linear_svm_queue.pkl"))
    start_svm = time.time()
    preds_svm = clf_svm.predict(X_test_tfidf)
    # Calibrated decision score via softmax on decision_function
    dec_svm = clf_svm.decision_function(X_test_tfidf)
    probs_svm = np.exp(dec_svm) / np.sum(np.exp(dec_svm), axis=1, keepdims=True)
    conf_svm = np.max(probs_svm, axis=1)
    svm_latency = ((time.time() - start_svm) / len(test_df)) * 1000

    acc_svm = accuracy_score(y_test, preds_svm)
    p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_test, preds_svm, average="macro", zero_division=0)
    _, _, w_f1_svm, _ = precision_recall_fscore_support(y_test, preds_svm, average="weighted", zero_division=0)

    # 3. Fine-Tune & Evaluate DistilBERT on Test Set
    print("\n--- Step 2: Fine-Tuning & Evaluating DistilBERT on Test Set ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    unique_queues = sorted(train_df["queue"].unique())
    label2id = {label: i for i, label in enumerate(unique_queues)}
    id2label = {i: label for i, label in enumerate(unique_queues)}

    # Prepare stratified training subset for fast GPU fine-tuning (up to 350 per queue)
    print("Preparing training subset for fast fine-tuning on Apple Silicon GPU...")
    train_sample = (
        train_df.sample(frac=1.0, random_state=42)
        .groupby("queue", as_index=False)
        .head(350)
        .reset_index(drop=True)
    )
    print(f"Fine-tuning on {len(train_sample):,} stratified support tickets...")

    train_enc = tokenizer(train_sample["text"].tolist(), truncation=True, padding=True, max_length=128)
    train_y = [label2id[q] for q in train_sample["queue"]]

    # Inverse class weights
    class_counts = np.bincount(train_y, minlength=len(unique_queues))
    class_weights = len(train_y) / (len(unique_queues) * class_counts.astype(np.float32))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float, device=device)

    # Dataset & DataLoader
    class FastTrainDataset(Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item
        def __len__(self):
            return len(self.labels)

    train_loader = DataLoader(FastTrainDataset(train_enc, train_y), batch_size=32, shuffle=True)

    # Initialize model
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=len(unique_queues),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    optimizer = torch.optim.AdamW(distilbert_model.parameters(), lr=3e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    # Fine-tuning loop (2 epochs)
    distilbert_model.train()
    print("Training DistilBERT...")
    for epoch in range(1, 3):
        total_loss = 0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/2 | Average Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s")

    # Save fine-tuned model
    distilbert_save_path = os.path.join(models_dir, "distilbert_queue_finetuned")
    distilbert_model.save_pretrained(distilbert_save_path)
    tokenizer.save_pretrained(distilbert_save_path)
    print(f"Saved fine-tuned DistilBERT to '{distilbert_save_path}'")

    # 4. Run Test Set Inference with Softmax Probabilities
    print("\n--- Step 3: Running Test Set Inference with Softmax Probabilities ---")
    test_enc = tokenizer(test_df["text"].tolist(), truncation=True, padding=True, max_length=128)
    test_loader = DataLoader(EvalDataset(test_enc), batch_size=64, shuffle=False)

    distilbert_model.eval()
    all_logits = []
    start_dbert = time.time()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())

    dbert_latency = ((time.time() - start_dbert) / len(test_df)) * 1000  # ms per ticket

    logits_tensor = torch.cat(all_logits, dim=0)
    softmax_probs = torch.softmax(logits_tensor, dim=1).numpy()
    pred_indices = np.argmax(softmax_probs, axis=1)
    preds_dbert = [id2label[idx] for idx in pred_indices]
    conf_dbert = np.max(softmax_probs, axis=1)

    acc_dbert = accuracy_score(y_test, preds_dbert)
    p_dbert, r_dbert, f1_dbert, _ = precision_recall_fscore_support(y_test, preds_dbert, average="macro", zero_division=0)
    _, _, w_f1_dbert, _ = precision_recall_fscore_support(y_test, preds_dbert, average="weighted", zero_division=0)

    # 5. Export Test Predictions with Confidence Scores
    pred_df = pd.DataFrame({
        "text": test_df["text"],
        "true_queue": test_df["queue"],
        "predicted_queue": preds_dbert,
        "confidence": conf_dbert.round(4),
        "is_correct": (test_df["queue"] == preds_dbert)
    })
    pred_output_path = os.path.join(data_dir, "test_predictions_distilbert.csv")
    pred_df.to_csv(pred_output_path, index=False)
    print(f"Exported test predictions with confidence scores to: {pred_output_path}")

    # Confidence Statistics
    correct_conf = pred_df[pred_df["is_correct"]]["confidence"].mean()
    wrong_conf = pred_df[~pred_df["is_correct"]]["confidence"].mean()
    print(f"\nConfidence Diagnostics:")
    print(f"  Mean Confidence on CORRECT predictions: {correct_conf*100:.2f}%")
    print(f"  Mean Confidence on WRONG predictions:   {wrong_conf*100:.2f}%")

    # 6. Build the Official Baseline vs. DistilBERT Table
    comparison_results = [
        {
            "Sprint": "Sprint 1 (Baseline)",
            "Model": "TF-IDF + Logistic Regression (Balanced)",
            "Target Task": "Queue (Test Set)",
            "Test Accuracy": f"{acc_lr * 100:.2f}%",
            "Macro Precision": f"{p_lr * 100:.2f}%",
            "Macro Recall": f"{r_lr * 100:.2f}%",
            "Macro-F1": f"{f1_lr * 100:.2f}%",
            "Mean Confidence": f"{conf_lr.mean() * 100:.2f}%",
            "Latency": f"{lr_latency:.3f} ms"
        },
        {
            "Sprint": "Sprint 2 (Classical)",
            "Model": "TF-IDF + Linear SVM (Balanced)",
            "Target Task": "Queue (Test Set)",
            "Test Accuracy": f"{acc_svm * 100:.2f}%",
            "Macro Precision": f"{p_svm * 100:.2f}%",
            "Macro Recall": f"{r_svm * 100:.2f}%",
            "Macro-F1": f"{f1_svm * 100:.2f}%",
            "Mean Confidence": f"{conf_svm.mean() * 100:.2f}%",
            "Latency": f"{svm_latency:.3f} ms"
        },
        {
            "Sprint": "Sprint 2 (Transformer)",
            "Model": "Fine-Tuned DistilBERT",
            "Target Task": "Queue (Test Set)",
            "Test Accuracy": f"{acc_dbert * 100:.2f}%",
            "Macro Precision": f"{p_dbert * 100:.2f}%",
            "Macro Recall": f"{r_dbert * 100:.2f}%",
            "Macro-F1": f"{f1_dbert * 100:.2f}%",
            "Mean Confidence": f"{conf_dbert.mean() * 100:.2f}%",
            "Latency": f"{dbert_latency:.2f} ms"
        }
    ]

    comp_table = pd.DataFrame(comparison_results)
    print("\n" + "=" * 80)
    print("OFFICIAL BASELINE VS. DISTILBERT COMPARISON TABLE (HELD-OUT TEST SET)")
    print("=" * 80)
    print(comp_table[["Model", "Test Accuracy", "Macro-F1", "Macro Recall", "Mean Confidence", "Latency"]].to_string(index=False))

    # Append to results_table.csv
    results_path = "results_table.csv"
    existing_df = pd.read_csv(results_path)
    # Format new entries to match columns
    formatted_entries = []
    for r in comparison_results:
        formatted_entries.append({
            "Sprint": r["Sprint"],
            "Model": f"{r['Model']} (Test Set)",
            "Target Task": "Queue (10 Classes)",
            "Accuracy": r["Test Accuracy"],
            "Macro Precision": r["Macro Precision"],
            "Macro Recall": r["Macro Recall"],
            "Macro-F1": r["Macro-F1"],
            "Weighted F1": f"{w_f1_dbert*100:.2f}%" if "DistilBERT" in r["Model"] else (f"{w_f1_svm*100:.2f}%" if "SVM" in r["Model"] else f"{w_f1_lr*100:.2f}%")
        })
    updated_results = pd.concat([existing_df, pd.DataFrame(formatted_entries)], ignore_index=True)
    updated_results.drop_duplicates(subset=["Model", "Target Task"], keep="last", inplace=True)
    updated_results.to_csv(results_path, index=False)
    print(f"\nUpdated '{results_path}' with Test Set benchmarks.")

    # 7. Generate Plotly Confidence Histogram
    fig_conf = px.histogram(
        pred_df,
        x="confidence",
        color="is_correct",
        nbins=40,
        barmode="overlay",
        color_discrete_map={True: "#00CC96", False: "#EF553B"},
        labels={"confidence": "Prediction Confidence (Softmax Probability)", "is_correct": "Correctly Classified?"},
        title="Day 9: DistilBERT Confidence Distribution (Correct vs. Incorrect Predictions)"
    )
    fig_conf.update_layout(template="plotly_white", height=500, width=800)
    conf_chart_path = os.path.join(charts_dir, "chart9_distilbert_confidence.html")
    fig_conf.write_html(conf_chart_path)
    print(f"Saved confidence visualization to: {conf_chart_path}")

    return comp_table

if __name__ == "__main__":
    run_day9_test_evaluation()
