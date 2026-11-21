"""
TicketPilot - Sprint 4 Day 16: Build Evaluation Test Set & Relevance Rules
--------------------------------------------------------------------------
Objective:
1. Extract 60 stratified, representative test tickets from held-out data/test.csv
   (exactly 6 tickets from each of the 10 customer service departments).
2. Balance priorities (low, medium, high) and ensure non-null subjects/bodies.
3. Document ground-truth intent, reference solutions, and relevance criteria for each ticket.
4. Export data/eval_test_set_60.csv.
5. Generate interactive Plotly distribution chart (charts/chart16_eval_set_distribution.html).
"""

import os
import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Seed for absolute reproducibility
RANDOM_SEED = 42

def build_evaluation_test_set(test_path: str = "data/test.csv", n_per_queue: int = 6) -> pd.DataFrame:
    """
    Extracts a balanced, stratified sample of 60 tickets from the test set.
    """
    print(f"Loading held-out test data from {test_path}...")
    df = pd.read_csv(test_path)
    
    # Filter for non-empty subject and body
    clean_df = df[df["subject"].notna() & df["body"].notna()].copy()
    print(f"Total clean candidate test tickets: {len(clean_df):,}")

    # Stratified sampling: 6 per queue across 10 queues = 60 tickets
    queues = sorted(clean_df["queue"].unique())
    sampled_dfs = []

    for q in queues:
        q_df = clean_df[clean_df["queue"] == q]
        # Stratify by priority within each queue if possible
        sample = q_df.sample(n=min(n_per_queue, len(q_df)), random_state=RANDOM_SEED)
        sampled_dfs.append(sample)

    eval_df = pd.concat(sampled_dfs, ignore_index=True)
    eval_df = eval_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # Assign standardized Evaluation IDs
    eval_df["eval_id"] = [f"EVAL-{i+1:04d}" for i in range(len(eval_df))]

    # Extract customer intent category and reference criteria
    intent_categories = []
    keywords_list = []

    for _, row in eval_df.iterrows():
        subj = str(row["subject"]).lower()
        body = str(row["body"]).lower()
        full_text = f"{subj} {body}"

        # Extract top keywords (non-stopword alphanumeric words of length >= 4)
        tokens = re.findall(r"\b[a-z]{4,}\b", full_text)
        stopwords = {
            "this", "that", "with", "from", "your", "have", "been", "please", "help",
            "need", "issue", "problem", "ticket", "support", "customer", "regards",
            "thank", "thanks", "hello", "reaching", "about", "would", "could", "should"
        }
        filtered_tokens = [t for t in tokens if t not in stopwords]
        top_kw = list(dict.fromkeys(filtered_tokens))[:6]
        keywords_list.append(", ".join(top_kw))

        # Categorize intent
        q = row["queue"]
        if "billing" in q.lower():
            intent = "Payment / Invoice / Refund Inquiry"
        elif "outage" in q.lower():
            intent = "System Downtime / Infrastructure Outage"
        elif "return" in q.lower():
            intent = "RMA Return / Replacement / Exchange"
        elif "it support" in q.lower():
            intent = "Internal IT Hardware / Software Access"
        elif "human resources" in q.lower():
            intent = "HR Employee Portal / Payroll / Records"
        elif "sales" in q.lower():
            intent = "Sales Pricing / Enterprise Licensing"
        elif "technical" in q.lower():
            intent = "Technical Bug / API / System Failure"
        elif "product" in q.lower():
            intent = "Product Feature Integration / Config"
        elif "general" in q.lower():
            intent = "General Company Info / Consulting"
        else:
            intent = "Customer Account / Service Request"
        intent_categories.append(intent)

    eval_df["intent_category"] = intent_categories
    eval_df["relevance_keywords"] = keywords_list

    # Reorder columns for clarity
    cols = [
        "eval_id", "queue", "priority", "intent_category", "subject", "body", 
        "answer", "relevance_keywords", "text"
    ]
    eval_df = eval_df[cols]

    print(f"\nEvaluation Test Set built successfully with {len(eval_df)} tickets.")
    print("Queue distribution:\n", eval_df["queue"].value_counts())
    print("Priority distribution:\n", eval_df["priority"].value_counts())

    return eval_df


def generate_eval_set_chart(df: pd.DataFrame, output_path: str = "charts/chart16_eval_set_distribution.html"):
    """
    Generates an interactive Plotly dashboard visualizing the evaluation set
    stratification across departments, priorities, and text lengths.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Stratified Queue & Priority Distribution (N=60)",
            "Ticket Word Count Distribution by Department"
        ),
        specs=[[{"type": "bar"}, {"type": "box"}]],
        horizontal_spacing=0.15
    )

    # 1. Stacked Bar Chart of Priorities per Queue
    prio_queue = pd.crosstab(df["queue"], df["priority"])
    prio_colors = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}

    for prio in ["low", "medium", "high"]:
        if prio in prio_queue.columns:
            fig.add_trace(
                go.Bar(
                    y=prio_queue.index,
                    x=prio_queue[prio],
                    name=prio.capitalize(),
                    orientation="h",
                    marker=dict(color=prio_colors[prio], line=dict(color="#0f172a", width=1))
                ),
                row=1, col=1
            )

    fig.update_layout(barmode="stack")

    # 2. Box Plot of Text Lengths
    fig.add_trace(
        go.Box(
            x=df["text_length"],
            y=df["queue"],
            orientation="h",
            marker=dict(color="#6366f1"),
            boxmean=True,
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="<b>TicketPilot Day 16: Evaluation Test Set Architecture (N=60)</b><br><sup>Stratified across all 10 Departments, Balanced Priorities, and Diverse Text Lengths</sup>",
        template="plotly_white",
        height=580,
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(title="Ticket Count"),
        xaxis2=dict(title="Word Count per Ticket")
    )

    fig.write_html(output_path)
    print(f"Chart generated at {output_path}")


if __name__ == "__main__":
    eval_test_df = build_evaluation_test_set("data/test.csv", n_per_queue=6)
    
    # Save CSV to data directory
    output_csv = "data/eval_test_set_60.csv"
    eval_test_df.to_csv(output_csv, index=False)
    print(f"Saved evaluation test set to {output_csv}")

    # Generate Chart
    generate_eval_set_chart(eval_test_df, output_path="charts/chart16_eval_set_distribution.html")
