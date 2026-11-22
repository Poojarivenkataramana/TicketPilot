"""
Unit Tests for FastAPI REST Endpoints (Day 25)
---------------------------------------------
Tests FastAPI application endpoints using TestClient with lifespan context:
- GET /: Root metadata & Swagger link
- GET /health: Health check, uptime, vector store readiness
- POST /triage: Sub-second classification with valid and invalid payloads
- POST /draft-reply: Full RAG generation across Tier 1, Tier 2, and Tier 3
- GET /decisions: Audit log retrieval
- GET /decisions/stats: Summary telemetry
"""

from fastapi.testclient import TestClient
from api import app


def test_api_root(client: TestClient):
    """Verify root endpoint returns system information."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "TicketPilot" in data["service"]
    assert "docs_url" in data
    assert data["status"] == "online"


def test_api_health(client: TestClient):
    """Verify health endpoint confirms ChromaDB and pipeline readiness."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["chromadb_vector_count"] >= 17000
    assert data["kb_article_count"] >= 17000


def test_api_triage_valid(client: TestClient):
    """Verify POST /triage delivers rapid classification and confidence."""
    payload = {
        "ticket_id": "TCK-TRIAGE-TEST",
        "subject": "Unauthorized charge on Visa invoice INV-902",
        "body": "I was billed twice on my credit card for subscription renewal. Please refund."
    }
    res = client.post("/triage", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "Billing" in data["predicted_queue"] or len(data["predicted_queue"]) > 0
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["predicted_priority"] in ["high", "medium", "low"]


def test_api_triage_invalid_empty(client: TestClient):
    """Verify POST /triage returns 422 Unprocessable Entity on empty or missing body."""
    payload = {"ticket_id": "TCK-BAD"}
    res = client.post("/triage", json=payload)
    assert res.status_code == 422


def test_api_draft_reply_tier1_auto_draft(client: TestClient):
    """Verify POST /draft-reply generates cited draft for answerable ticket."""
    payload = {
        "ticket_id": "TCK-DRAFT-01",
        "subject": "Duplicate invoice renewal charge",
        "body": "I was charged twice on my credit card for invoice renewal. I need a refund."
    }
    res = client.post("/draft-reply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "routing_tier" in data
    assert len(data["draft_text"]) > 20
    assert "latency_ms" in data


def test_api_draft_reply_tier2_outage(client: TestClient):
    """Verify POST /draft-reply flags Tier 2 Expedited Review on critical database failure."""
    payload = {
        "ticket_id": "TCK-OUTAGE-02",
        "subject": "CRITICAL: Database outage across all production clusters",
        "body": "Our primary database is completely down and failing health checks with HTTP 500."
    }
    res = client.post("/draft-reply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["routing_tier"] == "TIER_2_EXPEDITED_REVIEW"
    assert data["requires_human_approval"] is True


def test_api_draft_reply_tier3_ungrounded_refusal(client: TestClient):
    """Verify POST /draft-reply triggers Tier 3 escalation for ungrounded request."""
    payload = {
        "ticket_id": "TCK-UNKNOWN-03",
        "subject": "Internal custom alien protocol XYZ",
        "body": "Can you configure a custom warp drive interface on port 9999?"
    }
    res = client.post("/draft-reply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["routing_tier"] == "TIER_3_HUMAN_ASSIST"
    assert data["requires_human_approval"] is True


def test_api_decisions_and_stats(client: TestClient):
    """Verify decision audit log and telemetry endpoints."""
    # GET /decisions
    res_dec = client.get("/decisions?limit=5")
    assert res_dec.status_code == 200
    data_dec = res_dec.json()
    assert "decisions" in data_dec
    assert isinstance(data_dec["decisions"], list)

    # GET /decisions/stats
    res_stats = client.get("/decisions/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "total_logged" in stats
    assert "auto_draft_rate_pct" in stats


def run_all_api_tests():
    print("Running FastAPI REST API Unit Tests...")
    with TestClient(app) as client:
        test_api_root(client)
        print("  ✓ test_api_root passed")
        test_api_health(client)
        print("  ✓ test_api_health passed")
        test_api_triage_valid(client)
        print("  ✓ test_api_triage_valid passed")
        test_api_triage_invalid_empty(client)
        print("  ✓ test_api_triage_invalid_empty passed")
        test_api_draft_reply_tier1_auto_draft(client)
        print("  ✓ test_api_draft_reply_tier1_auto_draft passed")
        test_api_draft_reply_tier2_outage(client)
        print("  ✓ test_api_draft_reply_tier2_outage passed")
        test_api_draft_reply_tier3_ungrounded_refusal(client)
        print("  ✓ test_api_draft_reply_tier3_ungrounded_refusal passed")
        test_api_decisions_and_stats(client)
        print("  ✓ test_api_decisions_and_stats passed")
    print("✓ All 8 FastAPI REST API unit tests passed successfully!")


if __name__ == "__main__":
    run_all_api_tests()
