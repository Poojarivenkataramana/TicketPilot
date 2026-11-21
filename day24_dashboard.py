"""
TicketPilot - Sprint 5 Day 24: Plotly Operational Dashboard
-----------------------------------------------------------
Processes a test batch of 100 tickets and generates an interactive,
executive operational dashboard tracking:
1. Volume by Queue (Predicted vs. Ground Truth distributions)
2. Priority Mix (High, Medium, Low breakdown)
3. Softmax Confidence Distribution (with calibrated cutoff tau=0.40)
4. Routing Policy Tier Distribution & Auto-Draft Rate
5. Decision Quadrant (Confidence vs. Cross-Encoder Rerank Score)
6. End-to-End Latency Decomposition (ms per ticket)

Output Artifacts:
- data/day24_batch_100_input.csv: Reproducible 100-ticket test sample (10/queue)
- data/day24_batch_100_results.csv: Full per-ticket execution metrics
- charts/chart24_operational_dashboard.html: Interactive executive Plotly dashboard
"""

import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from day15_end_to_end_pipeline import TicketPilotPipeline
from routing_policy import evaluate_routing_policy, CONFIDENCE_THRESHOLD, RERANK_RELEVANCE_THRESHOLD
from decision_logger import decision_logger


def prepare_test_batch(test_path: str = "data/test.csv", batch_size: int = 100, seed: int = 42) -> pd.DataFrame:
    """Extracts a balanced stratified sample of 10 tickets per queue (total 100 tickets)."""
    df = pd.read_csv(test_path)
    tickets_per_queue = batch_size // df["queue"].nunique()
    
    sample_dfs = []
    for queue_name, group in df.groupby("queue"):
        sample_n = min(len(group), tickets_per_queue)
        sample_dfs.append(group.sample(n=sample_n, random_state=seed))
    
    batch_df = pd.concat(sample_dfs).reset_index(drop=True)
    if len(batch_df) < batch_size:
        remainder = batch_size - len(batch_df)
        remaining_pool = df[~df.index.isin(batch_df.index)]
        batch_df = pd.concat([batch_df, remaining_pool.sample(n=remainder, random_state=seed)]).reset_index(drop=True)
    
    os.makedirs("data", exist_ok=True)
    batch_df.to_csv("data/day24_batch_100_input.csv", index=False)
    print(f"Loaded test batch of {len(batch_df)} tickets across {batch_df['queue'].nunique()} queues.")
    return batch_df


def run_batch_evaluation(batch_df: pd.DataFrame, pipeline: TicketPilotPipeline = None, force_rerun: bool = False) -> pd.DataFrame:
    """Runs all batch tickets through the TicketPilot pipeline and evaluates routing policy."""
    results_path = "data/day24_batch_100_results.csv"
    if not force_rerun and os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        if len(existing_df) == len(batch_df):
            print(f"Loaded existing batch results ({len(existing_df)} tickets) from {results_path}")
            return existing_df

    if pipeline is None:
        pipeline = TicketPilotPipeline(
            models_dir="models",
            kb_path="data/knowledge_base_clean.csv",
            chroma_path="chroma_db",
            relevance_threshold=RERANK_RELEVANCE_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
    
    records = []
    print(f"\nProcessing {len(batch_df)} test batch tickets through TicketPilot...")
    t_batch_start = time.perf_counter()
    
    for idx, row in batch_df.iterrows():
        t_start = time.perf_counter()
        ticket_id = f"BATCH-TCK-{idx+1:03d}"
        subject = str(row.get("subject", "")) if pd.notna(row.get("subject")) else ""
        body = str(row.get("body", "")) if pd.notna(row.get("body")) else ""
        full_text = f"{subject} {body}".strip()
        gt_queue = str(row.get("queue", "Unknown"))
        gt_priority = str(row.get("priority", "medium")).lower()

        ticket_input = {
            "ticket_id": ticket_id,
            "subject": subject,
            "body": body,
            "text": full_text
        }

        # Run pipeline
        res = pipeline.run_ticket(ticket_input)
        latency_ms = (time.perf_counter() - t_start) * 1000

        pred_queue = res["pred_queue"]
        confidence = float(res["confidence"])
        pred_priority = str(res["pred_priority"]).lower()
        top_rerank_score = float(res.get("top_rerank_score", -99.0))
        top_kb_id = str(res.get("top_kb_id", "None"))
        draft_text = str(res.get("draft_text", ""))
        citations = res.get("found_citations", "None")

        # Evaluate routing policy
        routing = evaluate_routing_policy(
            pred_queue=pred_queue,
            confidence=confidence,
            pred_priority=pred_priority,
            top_rerank_score=top_rerank_score
        )

        # Log decision to persistent logger
        decision_logger.log_decision(
            ticket_id=ticket_id,
            subject=subject,
            body=body,
            predicted_queue=pred_queue,
            confidence=confidence,
            predicted_priority=pred_priority,
            routing_tier=routing["tier"],
            action=routing["action"],
            top_kb_id=top_kb_id,
            rerank_score=top_rerank_score,
            auto_send_eligible=routing["auto_send_eligible"],
            requires_human_approval=routing["requires_human_approval"],
            citations=[c.strip() for c in citations.split(",") if c.strip() and c.strip() != "None"],
            draft_text=draft_text,
            latency_ms=latency_ms
        )

        records.append({
            "ticket_id": ticket_id,
            "subject": subject[:60] + "..." if len(subject) > 60 else subject,
            "ground_truth_queue": gt_queue,
            "predicted_queue": pred_queue,
            "queue_correct": int(pred_queue.strip().lower() == gt_queue.strip().lower()),
            "ground_truth_priority": gt_priority,
            "predicted_priority": pred_priority,
            "priority_correct": int(pred_priority.strip().lower() == gt_priority.strip().lower()),
            "confidence": confidence,
            "top_kb_id": top_kb_id,
            "top_rerank_score": top_rerank_score,
            "routing_tier": routing["tier"],
            "routing_label": routing["tier_label"],
            "action": routing["action"],
            "auto_send_eligible": int(routing["auto_send_eligible"]),
            "requires_human_approval": int(routing["requires_human_approval"]),
            "latency_ms": latency_ms
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(batch_df):
            elapsed = time.perf_counter() - t_batch_start
            print(f"   [{idx+1:03d}/{len(batch_df):03d}] Processed ({elapsed:.1f}s elapsed, {elapsed/(idx+1)*1000:.1f}ms/tck)")

    results_df = pd.DataFrame(records)
    results_df.to_csv("data/day24_batch_100_results.csv", index=False)
    print(f"\nBatch processing complete! Saved {len(results_df)} records to data/day24_batch_100_results.csv")
    return results_df


def generate_plotly_dashboard(results_df: pd.DataFrame, output_html: str = "charts/chart24_operational_dashboard.html"):
    """
    Constructs an interactive 6-panel operational dashboard tracking:
    1. Volume by Queue (Predicted vs Ground Truth)
    2. Priority Mix
    3. Softmax Confidence Distribution
    4. Routing Tier & Auto-Draft Rate
    5. Decision Quadrant (Confidence vs Rerank Score)
    6. End-to-End Processing Latency
    """
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    # 1. Summary Statistics for Cards
    total_tickets = len(results_df)
    tier1_count = sum(results_df["routing_tier"] == "TIER_1_AUTO_DRAFT")
    tier2_count = sum(results_df["routing_tier"] == "TIER_2_EXPEDITED_REVIEW")
    tier3_count = sum(results_df["routing_tier"] == "TIER_3_HUMAN_ASSIST")
    auto_draft_rate = (tier1_count / total_tickets) * 100
    expedited_rate = (tier2_count / total_tickets) * 100
    human_assist_rate = (tier3_count / total_tickets) * 100
    mean_conf = results_df["confidence"].mean() * 100
    median_latency = results_df["latency_ms"].median()
    queue_accuracy = results_df["queue_correct"].mean() * 100

    # 2. Build Multi-Panel Subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "<b>1. Volume by Queue (Predicted vs. Ground Truth)</b>",
            "<b>2. Priority Mix Breakdown</b>",
            "<b>3. Softmax Confidence Distribution</b>",
            "<b>4. Routing Policy & Auto-Draft Rate</b>",
            "<b>5. Decision Space (Confidence vs. Rerank Score)</b>",
            "<b>6. End-to-End Latency Profile (ms)</b>"
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "histogram"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "histogram"}]
        ],
        vertical_spacing=0.11,
        horizontal_spacing=0.10
    )

    # Panel 1: Volume by Queue (Predicted vs Ground Truth)
    pred_queue_counts = results_df["predicted_queue"].value_counts().sort_index()
    gt_queue_counts = results_df["ground_truth_queue"].value_counts().reindex(pred_queue_counts.index, fill_value=0)
    
    fig.add_trace(
        go.Bar(
            y=pred_queue_counts.index,
            x=pred_queue_counts.values,
            orientation="h",
            name="Predicted Volume",
            marker_color="#2563eb",
            opacity=0.85,
            text=pred_queue_counts.values,
            textposition="auto"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            y=gt_queue_counts.index,
            x=gt_queue_counts.values,
            orientation="h",
            name="Ground Truth Baseline",
            marker_color="#94a3b8",
            opacity=0.55
        ),
        row=1, col=1
    )

    # Panel 2: Priority Mix (Donut chart)
    prio_counts = results_df["predicted_priority"].value_counts()
    prio_colors = {"high": "#dc2626", "medium": "#f59e0b", "low": "#10b981"}
    colors_list = [prio_colors.get(p.lower(), "#64748b") for p in prio_counts.index]
    
    fig.add_trace(
        go.Pie(
            labels=[p.capitalize() for p in prio_counts.index],
            values=prio_counts.values,
            hole=0.45,
            name="Priority Mix",
            marker=dict(colors=colors_list),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        ),
        row=1, col=2
    )

    # Panel 3: Confidence Distribution (Histogram with threshold line)
    fig.add_trace(
        go.Histogram(
            x=results_df["confidence"],
            nbinsx=15,
            marker_color="#6366f1",
            opacity=0.75,
            name="Confidence Scores"
        ),
        row=2, col=1
    )

    # Panel 4: Routing Policy & Auto-Draft Rate (Donut chart)
    tier_labels = ["Tier 1: Auto-Draft", "Tier 2: Expedited Review", "Tier 3: Human Assist"]
    tier_values = [tier1_count, tier2_count, tier3_count]
    tier_colors = ["#16a34a", "#d97706", "#4f46e5"]

    fig.add_trace(
        go.Pie(
            labels=tier_labels,
            values=tier_values,
            hole=0.48,
            name="Routing Policy",
            marker=dict(colors=tier_colors),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        ),
        row=2, col=2
    )

    # Panel 5: Decision Space Scatter Plot (Confidence vs. Rerank Score)
    tier_color_map = {
        "TIER_1_AUTO_DRAFT": "#16a34a",
        "TIER_2_EXPEDITED_REVIEW": "#d97706",
        "TIER_3_HUMAN_ASSIST": "#4f46e5"
    }

    for tier_code, color in tier_color_map.items():
        sub_df = results_df[results_df["routing_tier"] == tier_code]
        fig.add_trace(
            go.Scatter(
                x=sub_df["confidence"],
                y=sub_df["top_rerank_score"],
                mode="markers",
                name=tier_code.replace("_", " ").title(),
                marker=dict(size=9, color=color, opacity=0.8),
                text=sub_df["ticket_id"] + " (" + sub_df["predicted_queue"] + ")",
                hoverinfo="text+x+y"
            ),
            row=3, col=1
        )

    # Panel 6: Latency Distribution (Histogram)
    fig.add_trace(
        go.Histogram(
            x=results_df["latency_ms"],
            nbinsx=15,
            marker_color="#0ea5e9",
            opacity=0.75,
            name="Latency (ms)"
        ),
        row=3, col=2
    )

    # Threshold indicator lines using explicit Cartesian shapes
    # Confidence line in histogram (row 2, col 1: xref="x2", yref="y2 domain")
    fig.add_shape(
        type="line",
        x0=CONFIDENCE_THRESHOLD, x1=CONFIDENCE_THRESHOLD,
        y0=0, y1=1,
        xref="x2", yref="y2 domain",
        line=dict(color="#ef4444", width=2, dash="dash")
    )
    fig.add_annotation(
        x=CONFIDENCE_THRESHOLD, y=0.95,
        xref="x2", yref="y2 domain",
        text=f"Cutoff ({CONFIDENCE_THRESHOLD:.0%})",
        showarrow=False,
        xanchor="left",
        font=dict(color="#ef4444", size=11)
    )

    # Decision quadrant lines in scatter plot (row 3, col 1: xref="x3", yref="y3")
    fig.add_shape(
        type="line",
        x0=CONFIDENCE_THRESHOLD, x1=CONFIDENCE_THRESHOLD,
        y0=0, y1=1,
        xref="x3", yref="y3 domain",
        line=dict(color="#64748b", width=1.5, dash="dash")
    )
    fig.add_shape(
        type="line",
        x0=0, x1=1,
        y0=RERANK_RELEVANCE_THRESHOLD, y1=RERANK_RELEVANCE_THRESHOLD,
        xref="x3 domain", yref="y3",
        line=dict(color="#64748b", width=1.5, dash="dash")
    )

    fig.update_layout(
        title={
            "text": "<b>TicketPilot Operational Monitoring Dashboard</b><br><span style='font-size:14px;color:#64748b'>Batch Evaluation Performance on 100 Test Tickets across 10 Support Queues</span>",
            "y": 0.98, "x": 0.5, "xanchor": "center", "yanchor": "top"
        },
        template="plotly_white",
        height=1080,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=100, b=80),
        barmode="group"
    )

    fig.update_xaxes(title_text="Ticket Count", row=1, col=1)
    fig.update_xaxes(title_text="Softmax Confidence Score", row=2, col=1)
    fig.update_xaxes(title_text="Softmax Confidence (tau >= 0.40)", row=3, col=1)
    fig.update_yaxes(title_text="Cross-Encoder Rerank Score", row=3, col=1)
    fig.update_xaxes(title_text="Latency per Ticket (ms)", row=3, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)

    # Generate standalone executive HTML with KPI Cards
    plotly_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    full_page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TicketPilot - Operational Dashboard (100-Ticket Batch)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 24px;
        }}
        .header-container {{
            max-width: 1400px;
            margin: 0 auto 20px auto;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: #ffffff;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            text-align: center;
        }}
        .kpi-title {{
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: #0f172a;
        }}
        .kpi-caption {{
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 4px;
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="header-container">
        <h1 style="margin-bottom: 4px;">✈️ TicketPilot Operational Telemetry</h1>
        <p style="color: #64748b; margin-top: 0; margin-bottom: 20px;">
            Production Quality & Routing Performance on 100 Stratified Test Tickets
        </p>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Batch Size</div>
                <div class="kpi-value">{total_tickets}</div>
                <div class="kpi-caption">Stratified 10 Queues</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Auto-Draft Rate</div>
                <div class="kpi-value" style="color: #16a34a;">{auto_draft_rate:.1f}%</div>
                <div class="kpi-caption">{tier1_count} Direct Send Eligible</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Expedited Review</div>
                <div class="kpi-value" style="color: #d97706;">{expedited_rate:.1f}%</div>
                <div class="kpi-caption">{tier2_count} High Risk / Outages</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Human Assist</div>
                <div class="kpi-value" style="color: #4f46e5;">{human_assist_rate:.1f}%</div>
                <div class="kpi-caption">{tier3_count} Low Confidence / Ungrounded</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Queue Accuracy</div>
                <div class="kpi-value" style="color: #2563eb;">{queue_accuracy:.1f}%</div>
                <div class="kpi-caption">Top-1 Triage Match</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Median Latency</div>
                <div class="kpi-value">{median_latency:.0f} ms</div>
                <div class="kpi-caption">Steady-State Pipeline</div>
            </div>
        </div>
    </div>
    <div class="dashboard-container">
        {plotly_html}
    </div>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_page_html)

    print(f"Exported interactive operational dashboard to {output_html}")
    return fig


def main():
    print("=" * 70)
    print("TICKETPILOT SPRINT 5 DAY 24: OPERATIONAL DASHBOARD GENERATOR")
    print("=" * 70)

    # 1. Prepare 100-ticket test batch
    batch_df = prepare_test_batch(test_path="data/test.csv", batch_size=100, seed=42)

    # 2. Run evaluation
    results_df = run_batch_evaluation(batch_df)

    # 3. Generate Plotly dashboard
    generate_plotly_dashboard(results_df, output_html="charts/chart24_operational_dashboard.html")

    # 4. Print Executive Operational Summary
    print("\n" + "=" * 70)
    print("EXECUTIVE OPERATIONAL DASHBOARD SUMMARY (100 TICKETS)")
    print("=" * 70)
    print(f"Total Test Tickets Processed:     {len(results_df)}")
    print(f"Triage Queue Top-1 Accuracy:      {results_df['queue_correct'].mean():.2%}")
    print(f"Priority Classification Accuracy: {results_df['priority_correct'].mean():.2%}")
    print(f"Mean Softmax Confidence:          {results_df['confidence'].mean():.2%}")
    print(f"Median Pipeline Latency:          {results_df['latency_ms'].median():.1f} ms")
    print("\nRouting Policy Tier Breakdown:")
    for tier, count in results_df["routing_tier"].value_counts().items():
        print(f"  - {tier:25s}: {count:3d} ({count/len(results_df):.1%})")

    print("\nPredicted Queue Volume Distribution:")
    for q, count in results_df["predicted_queue"].value_counts().items():
        print(f"  - {q:32s}: {count:3d}")


if __name__ == "__main__":
    main()
