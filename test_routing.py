"""
Unit Tests for Three-Tier Compound Routing Policy (Day 25)
----------------------------------------------------------
Tests mathematical decision boundaries established during Sprint 4/5:
- Tier 1: Auto-Draft (Confidence >= 0.40 AND Rerank Score >= 4.0 AND Priority != High)
- Tier 2: Expedited Review (Priority == High OR Queue == Outages)
- Tier 3: Human Assist (Low Confidence OR Low Relevance)
- Boundary condition checks (0.40 vs 0.399, 4.0 vs 3.99)
- Flag assertions: auto_send_eligible, requires_human_approval
"""

from routing_policy import evaluate_routing_policy, CONFIDENCE_THRESHOLD, RERANK_RELEVANCE_THRESHOLD


def test_tier1_auto_draft_eligibility():
    """Validates clear Tier 1 auto-draft case."""
    res = evaluate_routing_policy(
        pred_queue="Billing and Payments",
        confidence=0.85,
        pred_priority="medium",
        top_rerank_score=7.2
    )
    assert res["tier"] == "TIER_1_AUTO_DRAFT"
    assert res["auto_send_eligible"] is True
    assert res["requires_human_approval"] is False
    assert "Safe for auto-drafting" in res["rationale"]


def test_tier2_high_priority_override():
    """Validates that High priority forces Tier 2 even with high confidence and relevance."""
    res = evaluate_routing_policy(
        pred_queue="Billing and Payments",
        confidence=0.95,
        pred_priority="high",
        top_rerank_score=9.0
    )
    assert res["tier"] == "TIER_2_EXPEDITED_REVIEW"
    assert res["auto_send_eligible"] is False
    assert res["requires_human_approval"] is True
    assert "Requires supervisor sign-off" in res["rationale"]


def test_tier2_critical_outage_override():
    """Validates that Service Outages queue forces Tier 2 even with low priority."""
    res = evaluate_routing_policy(
        pred_queue="Service Outages and Maintenance",
        confidence=0.75,
        pred_priority="low",
        top_rerank_score=6.5
    )
    assert res["tier"] == "TIER_2_EXPEDITED_REVIEW"
    assert res["auto_send_eligible"] is False
    assert res["requires_human_approval"] is True


def test_tier3_low_confidence():
    """Validates that confidence below 0.40 triggers Tier 3 escalation."""
    res = evaluate_routing_policy(
        pred_queue="Product Support",
        confidence=0.32,
        pred_priority="low",
        top_rerank_score=6.0
    )
    assert res["tier"] == "TIER_3_HUMAN_ASSIST"
    assert res["auto_send_eligible"] is False
    assert res["requires_human_approval"] is True
    assert "below 40%" in res["rationale"]


def test_tier3_low_rerank_score():
    """Validates that rerank score below 4.0 triggers Tier 3 escalation."""
    res = evaluate_routing_policy(
        pred_queue="Technical Support",
        confidence=0.70,
        pred_priority="medium",
        top_rerank_score=2.5
    )
    assert res["tier"] == "TIER_3_HUMAN_ASSIST"
    assert res["auto_send_eligible"] is False
    assert res["requires_human_approval"] is True
    assert "below +4.0" in res["rationale"]


def test_decision_boundaries_exact_thresholds():
    """Validates exact threshold edge cases at 0.40 and 4.0."""
    # Exactly on boundary -> Tier 1
    res_edge = evaluate_routing_policy(
        pred_queue="Customer Service",
        confidence=CONFIDENCE_THRESHOLD,  # 0.40
        pred_priority="low",
        top_rerank_score=RERANK_RELEVANCE_THRESHOLD  # 4.0
    )
    assert res_edge["tier"] == "TIER_1_AUTO_DRAFT"

    # Just below confidence boundary (0.399) -> Tier 3
    res_below_conf = evaluate_routing_policy(
        pred_queue="Customer Service",
        confidence=CONFIDENCE_THRESHOLD - 0.001,
        pred_priority="low",
        top_rerank_score=4.5
    )
    assert res_below_conf["tier"] == "TIER_3_HUMAN_ASSIST"

    # Just below rerank score boundary (3.99) -> Tier 3
    res_below_score = evaluate_routing_policy(
        pred_queue="Customer Service",
        confidence=0.60,
        pred_priority="low",
        top_rerank_score=RERANK_RELEVANCE_THRESHOLD - 0.01
    )
    assert res_below_score["tier"] == "TIER_3_HUMAN_ASSIST"


if __name__ == "__main__":
    print("Running Routing Policy Unit Tests...")
    test_tier1_auto_draft_eligibility()
    test_tier2_high_priority_override()
    test_tier2_critical_outage_override()
    test_tier3_low_confidence()
    test_tier3_low_rerank_score()
    test_decision_boundaries_exact_thresholds()
    print("✓ All 6 routing policy unit tests passed!")
