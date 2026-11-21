"""
TicketPilot - Sprint 4 Day 20: Production Routing Policy & Decision Thresholds
------------------------------------------------------------------------------
Encodes the mathematical decision boundaries established during Sprint 4 evaluation.
Maps incoming tickets into three operational tiers:
1. TIER 1: Fully Automated Auto-Draft (Confidence >= 0.40 AND Rerank Score >= 4.0 AND Priority != High)
2. TIER 2: Expedited Human Review (Priority == High OR Queue == Outages)
3. TIER 3: Human Agent Review & Assist (Low Confidence OR Low Relevance)
"""

from typing import Dict, Any

# Production Decision Boundaries
CONFIDENCE_THRESHOLD = 0.40
RERANK_RELEVANCE_THRESHOLD = 4.0
HIGH_RISK_QUEUES = {"Service Outages and Maintenance"}

def evaluate_routing_policy(
    pred_queue: str,
    confidence: float,
    pred_priority: str,
    top_rerank_score: float
) -> Dict[str, Any]:
    """
    Evaluates production routing policy and returns tier, action, and rationale.
    """
    is_high_risk = (pred_priority.lower() == "high") or (pred_queue in HIGH_RISK_QUEUES)
    is_high_confidence = (confidence >= CONFIDENCE_THRESHOLD)
    is_high_relevance = (top_rerank_score >= RERANK_RELEVANCE_THRESHOLD)

    # Tier 2: High-Priority / Critical Escalation
    if is_high_risk:
        return {
            "tier": "TIER_2_EXPEDITED_REVIEW",
            "tier_label": "Tier 2: Expedited Human Review",
            "action": "Draft Cited Reply & Escalate for Immediate Human Sign-Off",
            "auto_send_eligible": False,
            "requires_human_approval": True,
            "rationale": f"High risk ticket (Priority: {pred_priority}, Queue: {pred_queue}). Requires supervisor sign-off."
        }

    # Tier 1: Fully Automated Auto-Draft
    if is_high_confidence and is_high_relevance:
        return {
            "tier": "TIER_1_AUTO_DRAFT",
            "tier_label": "Tier 1: Automated Auto-Draft",
            "action": "Auto-Draft Cited Reply",
            "auto_send_eligible": True,
            "requires_human_approval": False,
            "rationale": f"High confidence ({confidence:.1%}) and high KB relevance ({top_rerank_score:+.2f}). Safe for auto-drafting."
        }

    # Tier 3: Human Agent Review & Assist
    reasons = []
    if not is_high_confidence:
        reasons.append(f"Confidence ({confidence:.1%}) below {CONFIDENCE_THRESHOLD:.0%}")
    if not is_high_relevance:
        reasons.append(f"Rerank score ({top_rerank_score:+.2f}) below {RERANK_RELEVANCE_THRESHOLD:+.1f}")

    return {
        "tier": "TIER_3_HUMAN_ASSIST",
        "tier_label": "Tier 3: Human Agent Review",
        "action": "Escalate to Human Agent with Top KB Snippets Pre-Attached",
        "auto_send_eligible": False,
        "requires_human_approval": True,
        "rationale": f"Safely escalated: {'; '.join(reasons)}. Prevents misrouting and hallucination."
    }

if __name__ == "__main__":
    # Test sample scenarios
    print("Testing Production Routing Policy:")
    cases = [
        ("Billing and Payments", 0.85, "medium", 6.97),
        ("Service Outages and Maintenance", 0.89, "high", 8.50),
        ("Product Support", 0.22, "medium", 1.20),
        ("General Inquiry", 0.38, "low", -1.50)
    ]
    for q, conf, prio, score in cases:
        decision = evaluate_routing_policy(q, conf, prio, score)
        print(f"\n[{q} | Conf: {conf:.1%} | Prio: {prio} | Score: {score:+.2f}]")
        print(f" -> {decision['tier_label']}: {decision['action']}")
        print(f"    Rationale: {decision['rationale']}")
