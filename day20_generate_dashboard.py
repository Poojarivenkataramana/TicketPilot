"""
TicketPilot - Sprint 4 Day 20: Executive Summary Visual Dashboard
-----------------------------------------------------------------
Generates charts/chart20_sprint4_executive_summary.html consolidating all Sprint 4 evaluation findings:
1. Retrieval Benchmark (Hit Rate@1 & MRR)
2. Qualitative Quality Scores (Accuracy, Grounding, Tone, Safety)
3. Steady-State Latency Percentiles (p50, p90, p95, p99)
4. Production Routing Tiers Distribution
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def generate_day20_dashboard(output_path: str = "charts/chart20_sprint4_executive_summary.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1. Retrieval Architecture Benchmark (N=60)",
            "2. Qualitative Draft Scores (1 to 5 Scale, N=30)",
            "3. Latency Percentiles Profile (ms)",
            "4. Production Routing Tiers Policy"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "pie"}]
        ],
        horizontal_spacing=0.15,
        vertical_spacing=0.18
    )

    # 1. Retrieval Benchmark (Hit Rate@1 & MRR)
    methods = ["Plain Vector", "Hybrid Search", "Hybrid + Rerank"]
    hr1 = [75.0, 83.33, 90.0]
    mrr_x100 = [81.39, 86.39, 90.33]

    fig.add_trace(
        go.Bar(name="Hit Rate@1 (%)", x=methods, y=hr1, text=[f"{v:.1f}%" for v in hr1], textposition="outside", marker_color="#3b82f6"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="MRR (x100)", x=methods, y=mrr_x100, text=[f"{v/100:.3f}" for v in mrr_x100], textposition="outside", marker_color="#10b981"),
        row=1, col=1
    )

    # 2. Qualitative Draft Scores
    dimensions = ["Accuracy", "Grounding", "Tone", "Safety", "Composite"]
    scores = [4.90, 4.97, 4.40, 4.97, 4.81]
    bar_colors = ["#3b82f6", "#10b981", "#f59e0b", "#6366f1", "#06b6d4"]

    fig.add_trace(
        go.Bar(
            x=dimensions,
            y=scores,
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
            marker=dict(color=bar_colors, line=dict(color="#0f172a", width=1.2)),
            showlegend=False
        ),
        row=1, col=2
    )

    # 3. Latency Percentiles
    percentiles = ["Mean", "p50 (Med)", "p90", "p95", "p99"]
    lat_values = [656.6, 609.2, 940.2, 1161.0, 1387.2]
    p_colors = ["#0284c7", "#10b981", "#f59e0b", "#f97316", "#ef4444"]

    fig.add_trace(
        go.Bar(
            x=percentiles,
            y=lat_values,
            text=[f"{v:.0f}ms" for v in lat_values],
            textposition="outside",
            marker=dict(color=p_colors, line=dict(color="#0f172a", width=1.2)),
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. Production Routing Tiers Distribution (Simulated from 60 tickets with compound policy)
    tier_labels = ["Tier 1: Auto-Draft (28%)", "Tier 2: Expedited Review (22%)", "Tier 3: Human Assist (50%)"]
    tier_values = [28, 22, 50]
    tier_colors = ["#10b981", "#f59e0b", "#6366f1"]

    fig.add_trace(
        go.Pie(
            labels=tier_labels,
            values=tier_values,
            hole=0.45,
            marker=dict(colors=tier_colors, line=dict(color="#0f172a", width=1.5)),
            textinfo="label",
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_layout(
        title="<b>TicketPilot Sprint 4: Executive Evaluation Dashboard & Decision Architecture</b><br><sup>Empirical Synthesis of Information Retrieval, Human Quality Scores, Latency Profile, and Routing Policy</sup>",
        template="plotly_white",
        height=720,
        font=dict(family="Inter, sans-serif", size=12),
        barmode="group",
        yaxis1=dict(title="Score (%) / MRR", range=[0, 115]),
        yaxis2=dict(title="Score (1 to 5)", range=[0, 5.8]),
        yaxis3=dict(title="Latency (ms)", range=[0, 1600])
    )

    fig.write_html(output_path)
    print(f"Executive dashboard generated at {output_path}")

if __name__ == "__main__":
    generate_day20_dashboard()
