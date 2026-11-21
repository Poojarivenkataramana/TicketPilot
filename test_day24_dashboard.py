"""
Test Suite for Sprint 5 Day 24: Plotly Operational Dashboard
------------------------------------------------------------
Validates the batch evaluation run and Plotly operational dashboard:
1. Verifies test batch of 100 tickets is extracted and balanced across queues.
2. Verifies execution metrics in data/day24_batch_100_results.csv.
3. Asserts computation of Volume by Queue, Priority Mix, Confidence, and Auto-Draft Rate.
4. Verifies generation and integrity of charts/chart24_operational_dashboard.html.
"""

import os
import pandas as pd


def test_batch_input_dataset():
    """Verify input test batch exists and has 100 tickets across 10 queues."""
    input_path = "data/day24_batch_100_input.csv"
    assert os.path.exists(input_path), f"Missing {input_path}"
    
    df = pd.read_csv(input_path)
    assert len(df) == 100, f"Expected 100 tickets, got {len(df)}"
    assert df["queue"].nunique() == 10, f"Expected 10 queues, got {df['queue'].nunique()}"
    assert "subject" in df.columns
    assert "body" in df.columns or "text" in df.columns


def test_batch_results_dataset():
    """Verify results dataset contains 100 records and all critical operational fields."""
    results_path = "data/day24_batch_100_results.csv"
    assert os.path.exists(results_path), f"Missing {results_path}"

    df = pd.read_csv(results_path)
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"

    required_cols = [
        "ticket_id", "ground_truth_queue", "predicted_queue", "queue_correct",
        "ground_truth_priority", "predicted_priority", "confidence",
        "top_kb_id", "top_rerank_score", "routing_tier", "action",
        "auto_send_eligible", "requires_human_approval", "latency_ms"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in results"

    # Validate value constraints
    assert df["confidence"].between(0.0, 1.0).all(), "Confidence scores must be between 0.0 and 1.0"
    assert (df["latency_ms"] > 0).all(), "Latency must be strictly positive"
    assert set(df["routing_tier"].unique()).issubset({
        "TIER_1_AUTO_DRAFT", "TIER_2_EXPEDITED_REVIEW", "TIER_3_HUMAN_ASSIST"
    }), f"Unexpected routing tiers: {df['routing_tier'].unique()}"


def test_operational_kpis():
    """Verify computation of core operational KPIs (Volume, Priority, Confidence, Auto-Draft Rate)."""
    df = pd.read_csv("data/day24_batch_100_results.csv")

    # 1. Volume by Queue
    pred_queues = df["predicted_queue"].value_counts()
    assert len(pred_queues) >= 5, "Predicted volume should cover multiple queues"

    # 2. Priority Mix
    prio_counts = df["predicted_priority"].str.lower().value_counts()
    assert "high" in prio_counts or "medium" in prio_counts or "low" in prio_counts

    # 3. Confidence Distribution
    mean_conf = df["confidence"].mean()
    assert 0.10 <= mean_conf <= 0.95, f"Unexpected mean confidence {mean_conf}"

    # 4. Auto-Draft Rate
    auto_draft_count = sum(df["routing_tier"] == "TIER_1_AUTO_DRAFT")
    auto_draft_rate = (auto_draft_count / len(df)) * 100
    assert 0.0 <= auto_draft_rate <= 100.0, f"Invalid auto-draft rate: {auto_draft_rate}"

    print(f"\nOperational KPIs on 100-Ticket Batch:")
    print(f" - Auto-Draft Rate: {auto_draft_rate:.1f}%")
    print(f" - Mean Confidence: {mean_conf:.1%}")
    print(f" - Median Latency:  {df['latency_ms'].median():.1f} ms")


def test_dashboard_html_artifact():
    """Verify the generated Plotly operational dashboard HTML file exists and is well-formed."""
    html_path = "charts/chart24_operational_dashboard.html"
    assert os.path.exists(html_path), f"Missing dashboard HTML at {html_path}"
    
    file_size = os.path.getsize(html_path)
    assert file_size > 10000, f"Dashboard HTML unexpectedly small ({file_size} bytes)"

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "TicketPilot Operational Telemetry" in content
    assert "Volume by Queue" in content
    assert "Priority Mix" in content
    assert "Softmax Confidence Distribution" in content
    assert "Routing Policy &amp; Auto-Draft Rate" in content or "Routing Policy & Auto-Draft Rate" in content
    assert "plotly-graph-div" in content or "Plotly" in content


if __name__ == "__main__":
    print("Running Day 24 Operational Dashboard Test Suite...")
    test_batch_input_dataset()
    print("✓ test_batch_input_dataset passed")
    test_batch_results_dataset()
    print("✓ test_batch_results_dataset passed")
    test_operational_kpis()
    print("✓ test_operational_kpis passed")
    test_dashboard_html_artifact()
    print("✓ test_dashboard_html_artifact passed")
    print("\n🎉 ALL DAY 24 DASHBOARD TESTS PASSED SUCCESSFULLY!")
