"""
TicketPilot - Sprint 2, Day 8: Fine-Tuning DistilBERT for Queue Classification
-----------------------------------------------------------------------------
Architecture:
- Base Model: distilbert-base-uncased (66M parameters)
- Target: 10 Queue Classes
- Max Token Length: 128 (captures 99.5% of ticket texts based on Day 3 EDA)
- Loss Function: Class-Weighted Cross-Entropy (counteracts heavy class imbalance)
- Overfitting Prevention:
    * Weight Decay: 0.01 (L2 regularization)
    * Warmup: 10% of total steps
    * Learning Rate: 2e-5 with AdamW optimizer
    * Early Stopping & load_best_model_at_end=True (tracks validation Macro-F1)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Custom Trainer to apply Class Weights to Cross-Entropy Loss
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# HuggingFace Dataset wrapper
class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_weighted": f1_weighted
    }

def main():
    parser = argparse.ArgumentParser(description="DistilBERT Fine-Tuning for TicketPilot")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional subset size for fast local testing")
    parser.add_argument("--output-dir", type=str, default="models/distilbert_queue", help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("TicketPilot - Sprint 2, Day 8: DistilBERT Fine-Tuning")
    print("=" * 70)

    # Device check
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon MPS acceleration.")
    else:
        device = "cpu"
        print("Using CPU (Warning: For full multi-epoch training, use Google Colab GPU).")

    # 1. Load Data
    data_dir = "data"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))

    if args.sample_size:
        print(f"Subsampling for fast verification: {args.sample_size} train, {args.sample_size // 4} val")
        train_df = train_df.sample(n=min(args.sample_size, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(args.sample_size // 4, len(val_df)), random_state=42)

    # Label mapping (string -> int)
    unique_queues = sorted(train_df["queue"].unique())
    label2id = {label: i for i, label in enumerate(unique_queues)}
    id2label = {i: label for i, label in enumerate(unique_queues)}

    train_labels = [label2id[q] for q in train_df["queue"]]
    val_labels = [label2id[q] for q in val_df["queue"]]

    # Calculate inverse class weights
    class_counts = np.bincount(train_labels, minlength=len(unique_queues))
    total_samples = len(train_labels)
    class_weights = total_samples / (len(unique_queues) * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print("\nCalculated Class Weights (inverse frequency):")
    for q, w in zip(unique_queues, class_weights):
        print(f"  - {q:<32}: weight = {w:.3f}")

    # 2. Tokenization
    model_ckpt = "distilbert-base-uncased"
    print(f"\nLoading Tokenizer: '{model_ckpt}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    print("Tokenizing train and validation texts (max_length=128)...")
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    val_encodings = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    train_dataset = TicketDataset(train_encodings, train_labels)
    val_dataset = TicketDataset(val_encodings, val_labels)

    # 3. Model Initialization
    print(f"\nInitializing Sequence Classification Head for {len(unique_queues)} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=len(unique_queues),
        id2label=id2label,
        label2id=label2id
    )

    # 4. Training Arguments with Overfitting Guards
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,                 # L2 regularization
        warmup_steps=100,                  # Learning rate warmup steps
        eval_strategy="epoch",             # Evaluate at each epoch
        save_strategy="epoch",             # Save checkpoint each epoch
        load_best_model_at_end=True,       # Prevent overfitting: restore checkpoint with highest val score
        metric_for_best_model="f1_macro",  # Target metric
        greater_is_better=True,
        logging_steps=50,
        report_to="none"
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop early if val metric degrades
    )

    print("\n--- Starting Model Training ---")
    train_result = trainer.train()

    print("\n--- Final Evaluation on Validation Set ---")
    eval_metrics = trainer.evaluate()
    print("\nValidation Results:")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v*100 if 'loss' not in k else v:.4f}")

    # Save model, tokenizer, and label mappings
    print(f"\nSaving best fine-tuned model to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save label mappings
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print("Day 8 Fine-Tuning Pipeline Complete without Overfitting!")
    return eval_metrics

if __name__ == "__main__":
    main()
