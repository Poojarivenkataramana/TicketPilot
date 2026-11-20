"""
TicketPilot - Sprint 1, Day 3: Exploratory Data Analysis (EDA)
--------------------------------------------------------------
Generates 5 Plotly charts and extracts data-backed insights:
1. Chart 1: Queue Volume Distribution & Class Imbalance (Bar Chart)
2. Chart 2: Priority Composition by Queue (Stacked Bar Chart)
3. Chart 3: Ticket Word Count Distribution by Department (Box Plot)
4. Chart 4: Top Distinctive Keywords per Queue via TF-IDF (Subplot Bars)
5. Chart 5: Ticket Length vs. Priority Urgency (Violin/Box Plot)
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer

def run_day3_eda():
    print("=" * 60)
    print("TicketPilot - Sprint 1, Day 3: Exploratory Data Analysis")
    print("=" * 60)

    data_file = "cleaned_tickets.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Cleaned dataset '{data_file}' not found. Run Day 2 first.")

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} cleaned tickets from {data_file}")

    # Create charts directory
    charts_dir = "charts"
    os.makedirs(charts_dir, exist_ok=True)

    # Feature Engineering for EDA: Word Counts & Char Counts
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    df["char_count"] = df["text"].astype(str).apply(len)

    print("\n--- Summary Statistics of Ticket Word Counts ---")
    print(df["word_count"].describe())

    # -------------------------------------------------------------
    # CHART 1: Queue Volume Distribution & Class Imbalance
    # -------------------------------------------------------------
    queue_counts = df["queue"].value_counts().reset_index()
    queue_counts.columns = ["queue", "count"]
    queue_counts["percentage"] = (queue_counts["count"] / len(df) * 100).round(2)

    fig1 = px.bar(
        queue_counts,
        x="queue",
        y="count",
        text=queue_counts.apply(lambda r: f"{r['count']:,}<br>({r['percentage']}%)", axis=1),
        title="Chart 1: Ticket Volume Distribution Across Queues (Class Imbalance)",
        labels={"queue": "Support Queue", "count": "Ticket Count"},
        color="count",
        color_continuous_scale="Blues"
    )
    fig1.update_layout(xaxis_tickangle=-35, showlegend=False, template="plotly_white")
    chart1_path = os.path.join(charts_dir, "chart1_queue_distribution.html")
    fig1.write_html(chart1_path)
    print(f"Saved: {chart1_path}")

    # -------------------------------------------------------------
    # CHART 2: Priority Composition by Queue
    # -------------------------------------------------------------
    queue_priority = df.groupby(["queue", "priority"]).size().reset_index(name="count")
    # Calculate percentage within each queue
    queue_totals = df.groupby("queue").size().to_dict()
    queue_priority["pct_of_queue"] = queue_priority.apply(
        lambda r: round((r["count"] / queue_totals[r["queue"]]) * 100, 2), axis=1
    )

    fig2 = px.bar(
        queue_priority,
        x="queue",
        y="count",
        color="priority",
        color_discrete_map={"high": "#EF553B", "medium": "#FFA15A", "low": "#00CC96"},
        category_orders={"priority": ["high", "medium", "low"]},
        title="Chart 2: Priority Breakdown Across Support Queues",
        labels={"queue": "Support Queue", "count": "Ticket Count", "priority": "Priority"},
        barmode="stack"
    )
    fig2.update_layout(xaxis_tickangle=-35, template="plotly_white")
    chart2_path = os.path.join(charts_dir, "chart2_priority_by_queue.html")
    fig2.write_html(chart2_path)
    print(f"Saved: {chart2_path}")

    # -------------------------------------------------------------
    # CHART 3: Ticket Word Count Distribution by Department
    # -------------------------------------------------------------
    fig3 = px.box(
        df,
        x="queue",
        y="word_count",
        color="queue",
        title="Chart 3: Ticket Word Count Distribution Across Queues",
        labels={"queue": "Support Queue", "word_count": "Word Count"},
        points=False  # Hide thousands of jitter dots for cleaner box
    )
    fig3.update_layout(xaxis_tickangle=-35, showlegend=False, template="plotly_white", yaxis_range=[0, 300])
    chart3_path = os.path.join(charts_dir, "chart3_word_count_by_queue.html")
    fig3.write_html(chart3_path)
    print(f"Saved: {chart3_path}")

    # -------------------------------------------------------------
    # CHART 4: Distinctive Keywords per Queue via Class-Specific TF-IDF
    # -------------------------------------------------------------
    queues = df["queue"].unique()
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["text"])
    feature_names = np.array(tfidf.get_feature_names_out())

    top_n_keywords = {}
    for q in queues:
        mask = (df["queue"] == q).values
        sub_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        top_indices = sub_mean.argsort()[::-1][:5]
        top_n_keywords[q] = [(feature_names[i], round(sub_mean[i], 4)) for i in top_indices]

    # Create a 2x5 subplot of top keywords for all 10 queues
    rows, cols = 5, 2
    queue_list = list(top_n_keywords.keys())
    fig4 = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=queue_list,
        horizontal_spacing=0.15,
        vertical_spacing=0.08
    )

    for idx, q in enumerate(queue_list):
        r = (idx // cols) + 1
        c = (idx % cols) + 1
        words = [w[0] for w in top_n_keywords[q]][::-1]
        scores = [w[1] for w in top_n_keywords[q]][::-1]
        fig4.add_trace(
            go.Bar(x=scores, y=words, orientation='h', name=q, showlegend=False, marker_color="#636EFA"),
            row=r, col=c
        )

    fig4.update_layout(
        title_text="Chart 4: Top 5 Distinctive N-Grams (TF-IDF) per Support Queue",
        height=1200,
        width=1000,
        template="plotly_white"
    )
    chart4_path = os.path.join(charts_dir, "chart4_top_keywords_per_queue.html")
    fig4.write_html(chart4_path)
    print(f"Saved: {chart4_path}")

    # -------------------------------------------------------------
    # CHART 5: Ticket Length vs Priority Urgency
    # -------------------------------------------------------------
    fig5 = px.box(
        df,
        x="priority",
        y="word_count",
        color="priority",
        color_discrete_map={"high": "#EF553B", "medium": "#FFA15A", "low": "#00CC96"},
        category_orders={"priority": ["high", "medium", "low"]},
        title="Chart 5: Word Count vs. Ticket Priority Level",
        labels={"priority": "Priority Level", "word_count": "Word Count"},
        points=False
    )
    fig5.update_layout(template="plotly_white", yaxis_range=[0, 300])
    chart5_path = os.path.join(charts_dir, "chart5_length_vs_priority.html")
    fig5.write_html(chart5_path)
    print(f"Saved: {chart5_path}")

    print("\n--- Top Distinctive Keywords by Queue ---")
    for q, kws in top_n_keywords.items():
        kw_str = ", ".join([f"{w} ({s})" for w, s in kws])
        print(f"  {q:<32}: {kw_str}")

    return {
        "summary": df["word_count"].describe().to_dict(),
        "top_keywords": top_n_keywords,
        "queue_priority": queue_priority
    }

if __name__ == "__main__":
    run_day3_eda()
