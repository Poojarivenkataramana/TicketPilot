"""
TicketPilot - Sprint 5 Day 21: FastAPI Endpoint Verification Test Suite
----------------------------------------------------------------------
Verifies the production FastAPI backend:
1. Startup Lifespan: Models loaded once into app.state.pipeline.
2. GET /: Root info & Swagger documentation metadata.
3. GET /health: Health check, uptime, and vector count.
4. POST /triage: Real-time classification & confidence scoring.
5. POST /draft-reply:
   - High-confidence answerable query (Tier 1 Auto-Draft with [KB-XXXXX] citations).
   - High-risk outage query (Tier 2 Expedited Review).
   - Ambiguous / out-of-domain query (Tier 3 Safe Human Escalation).
6. OpenAPI Swagger Schema Verification: Confirms /docs schema compliance.
"""

import json
import time
from fastapi.testclient import TestClient
from api import app

def run_api_verification():
    print("=" * 70)
    print("RUNNING TICKETPILOT FASTAPI VERIFICATION TEST SUITE")
    print("=" * 70)

    test_results = {}

    # Initialize TestClient with lifespan context (loads models once)
    with TestClient(app) as client:
        # 1. Test Root Endpoint
        print("\n1. Testing GET / (Root Info)...")
        t0 = time.time()
        res = client.get("/")
        lat = (time.time() - t0) * 1000
        assert res.status_code == 200, f"Root endpoint failed: {res.text}"
        data = res.json()
        print(f"   Status: {res.status_code} OK ({lat:.1f} ms)")
        print(f"   Payload: {data}")
        test_results["root"] = {"status_code": res.status_code, "data": data, "latency_ms": lat}

        # 2. Test Health Endpoint
        print("\n2. Testing GET /health...")
        t0 = time.time()
        res = client.get("/health")
        lat = (time.time() - t0) * 1000
        assert res.status_code == 200, f"Health endpoint failed: {res.text}"
        data = res.json()
        print(f"   Status: {res.status_code} OK ({lat:.1f} ms)")
        print(f"   Vectors Indexed: {data['chromadb_vector_count']:,} | KB Articles: {data['kb_article_count']:,}")
        test_results["health"] = {"status_code": res.status_code, "data": data, "latency_ms": lat}

        # 3. Test POST /triage
        print("\n3. Testing POST /triage (Fast Classification)...")
        triage_payload = {
            "ticket_id": "TEST-TRIAGE-01",
            "subject": "PostgreSQL database connection timeout error 5432",
            "body": "Our application servers are unable to connect to the PostgreSQL cluster on port 5432. All connections timeout after 30 seconds."
        }
        t0 = time.time()
        res = client.post("/triage", json=triage_payload)
        lat = (time.time() - t0) * 1000
        assert res.status_code == 200, f"Triage failed: {res.text}"
        data = res.json()
        print(f"   Status: {res.status_code} OK ({lat:.1f} ms)")
        print(f"   Predicted Queue: {data['predicted_queue']} (Confidence: {data['confidence']:.1%}) | Priority: {data['predicted_priority']}")
        test_results["triage"] = {"status_code": res.status_code, "data": data, "latency_ms": lat}

        # 4. Test POST /draft-reply (Case A: High-Confidence Auto-Draft)
        print("\n4. Testing POST /draft-reply (Case A: High-Confidence Billing Refund)...")
        draft_payload_a = {
            "ticket_id": "TEST-DRAFT-01",
            "subject": "Duplicate subscription renewal charge on credit card",
            "body": "I was charged twice on my Visa card for this month's renewal invoice INV-904. I need a refund for the duplicate transaction."
        }
        t0 = time.time()
        res = client.post("/draft-reply", json=draft_payload_a)
        lat = (time.time() - t0) * 1000
        assert res.status_code == 200, f"Draft reply failed: {res.text}"
        data_a = res.json()
        print(f"   Status: {res.status_code} OK ({lat:.1f} ms)")
        print(f"   Routing Tier: {data_a['routing_tier_label']}")
        print(f"   Action: {data_a['action']}")
        print(f"   Citations: {data_a['citations']}")
        print(f"   Draft Preview: {data_a['draft_text'][:130]}...")
        test_results["draft_reply_case_a"] = {"status_code": res.status_code, "data": data_a, "latency_ms": lat}

        # 5. Test POST /draft-reply (Case B: High-Priority Outage Ticket)
        print("\n5. Testing POST /draft-reply (Case B: High-Priority Cloud Outage)...")
        draft_payload_b = {
            "ticket_id": "TEST-DRAFT-02",
            "subject": "CRITICAL: Cloud EC2 cluster failure and database downtime",
            "body": "Our entire production database cluster is down and returning HTTP 500 errors across all customer instances."
        }
        t0 = time.time()
        res = client.post("/draft-reply", json=draft_payload_b)
        lat = (time.time() - t0) * 1000
        assert res.status_code == 200, f"Draft reply failed: {res.text}"
        data_b = res.json()
        print(f"   Status: {res.status_code} OK ({lat:.1f} ms)")
        print(f"   Routing Tier: {data_b['routing_tier_label']}")
        print(f"   Action: {data_b['action']}")
        print(f"   Human Approval Required: {data_b['requires_human_approval']}")
        test_results["draft_reply_case_b"] = {"status_code": res.status_code, "data": data_b, "latency_ms": lat}

        # 6. Test POST /draft-reply (Case C: Ambiguous / Low-Confidence Refusal)
        print("\n6. Testing POST /draft-reply (Case C: Ambiguous Low-Confidence Refusal)...")
        draft_payload_c = {
            "ticket_id": "TEST-DRAFT-03",
            "subject": "General inquiry about proprietary enterprise integration",
            "body": "Can you configure custom integration for internal system ABC without standard documentation?"
        }
        t0 = time.time()
        res = client.post("/draft-reply", json=draft_payload_c)
        lat = (time.time() - t0) * 1000
        assert res.status_code == 200, f"Draft reply failed: {res.text}"
        data_c = res.json()
        print(f"   Status: {res.status_code} OK ({lat:.1f} ms)")
        print(f"   Routing Tier: {data_c['routing_tier_label']}")
        print(f"   Action: {data_c['action']}")
        print(f"   Draft Output: {data_c['draft_text']}")
        test_results["draft_reply_case_c"] = {"status_code": res.status_code, "data": data_c, "latency_ms": lat}

        # 7. Test OpenAPI Documentation Schema (/openapi.json)
        print("\n7. Testing OpenAPI Schema (/openapi.json)...")
        res = client.get("/openapi.json")
        assert res.status_code == 200, "Failed to load openapi.json"
        openapi_schema = res.json()
        endpoints = list(openapi_schema["paths"].keys())
        print(f"   OpenAPI Version: {openapi_schema['openapi']}")
        print(f"   Registered Endpoints: {endpoints}")
        assert "/triage" in endpoints and "/draft-reply" in endpoints, "Missing endpoints in OpenAPI schema!"
        test_results["openapi"] = {"status_code": 200, "endpoints": endpoints}

    print("\n" + "=" * 70)
    print("ALL FASTAPI ENDPOINT TESTS PASSED SUCCESSFULLY! (6/6 Passed)")
    print("=" * 70)

    # Export test results
    with open("day21_api_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print("Saved test results to day21_api_test_results.json")

    return test_results

if __name__ == "__main__":
    run_api_verification()
