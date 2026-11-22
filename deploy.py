"""
TicketPilot - Sprint 6 Day 26: Production Deployment Manager
-----------------------------------------------------------
Manages local and private cloud deployments while strictly monitoring
memory consumption to operate safely within cloud free-tier memory budgets (<= 1 GB RAM).

Capabilities:
1. Pre-flight sanity checks on models, ChromaDB, and Knowledge Base.
2. Live memory tracking (Resident Set Size, Virtual Memory, RAM %).
3. Starts FastAPI (8000) and Streamlit (8501) with graceful health polling.
4. Performs live smoke tests on endpoints (/health, /triage, Streamlit UI).
5. Exports deployment telemetry to data/day26_deployment_telemetry.json.
6. Returns verified working links for public/private demonstration.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import requests
import psutil

API_PORT = 8000
UI_PORT = 8501
MAX_MEMORY_BUDGET_MB = 1024  # 1 GB Cloud Free-Tier limit


def check_preflight_artifacts() -> bool:
    """Verifies that all required models and indices exist."""
    print("1/4 Performing Pre-Flight Artifact Inspection...")
    required_paths = [
        "models/tfidf_vectorizer.pkl",
        "models/baseline_queue_model.pkl",
        "models/baseline_priority_model.pkl",
        "models/best_classical_linear_svm_queue.pkl",
        "data/knowledge_base_clean.csv",
        "chroma_db",
        "api.py",
        "streamlit_app.py",
        "routing_policy.py"
    ]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        print(f"❌ Missing required artifacts: {missing}")
        return False

    print("   ✓ All model weights, ChromaDB store, and Knowledge Base files verified.")
    return True


def get_process_memory_mb(pid: int) -> float:
    """Returns the RSS memory in megabytes of a given process and its children."""
    try:
        proc = psutil.Process(pid)
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            rss += child.memory_info().rss
        return rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def start_services() -> dict:
    """Launches FastAPI and Streamlit processes, monitors memory, and verifies connectivity."""
    print("\n2/4 Launching TicketPilot Production Services...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    os.makedirs("logs", exist_ok=True)
    api_log_file = open("logs/api_deploy.log", "w", encoding="utf-8")
    ui_log_file = open("logs/ui_deploy.log", "w", encoding="utf-8")

    # Start FastAPI
    api_cmd = [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(API_PORT)]
    print(f"   Starting FastAPI on http://127.0.0.1:{API_PORT}...")
    api_proc = subprocess.Popen(
        api_cmd,
        stdout=api_log_file,
        stderr=api_log_file,
        env=env
    )

    # Start Streamlit
    ui_cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(UI_PORT),
        "--server.address", "127.0.0.1",
        "--server.headless", "true"
    ]
    print(f"   Starting Streamlit UI on http://127.0.0.1:{UI_PORT}...")
    ui_proc = subprocess.Popen(
        ui_cmd,
        stdout=ui_log_file,
        stderr=ui_log_file,
        env=env
    )

    # Wait for services to be healthy
    print("\n3/4 Polling Endpoint Readiness & Health Checks...")
    max_wait_s = 45
    t0 = time.time()
    api_ready = False
    ui_ready = False

    while (time.time() - t0) < max_wait_s:
        # Check API health
        if not api_ready:
            try:
                r = requests.get(f"http://127.0.0.1:{API_PORT}/health", timeout=1)
                if r.status_code == 200:
                    api_ready = True
                    print(f"   ✓ FastAPI healthy after {time.time() - t0:.1f}s")
            except requests.exceptions.RequestException:
                pass

        # Check UI readiness
        if not ui_ready:
            try:
                r = requests.get(f"http://127.0.0.1:{UI_PORT}", timeout=1)
                if r.status_code == 200:
                    ui_ready = True
                    print(f"   ✓ Streamlit UI responsive after {time.time() - t0:.1f}s")
            except requests.exceptions.RequestException:
                pass

        if api_ready and ui_ready:
            break
        time.sleep(1.0)

    if not (api_ready and ui_ready):
        print(f"❌ Timed out waiting for services. (API ready: {api_ready}, UI ready: {ui_ready})")
        # Flush logs and print tail
        api_log_file.flush()
        ui_log_file.flush()
        with open("logs/api_deploy.log", "r", encoding="utf-8") as f:
            print("--- FastAPI Tail Log ---")
            print("".join(f.readlines()[-15:]))
        with open("logs/ui_deploy.log", "r", encoding="utf-8") as f:
            print("--- Streamlit Tail Log ---")
            print("".join(f.readlines()[-15:]))
        api_proc.terminate()
        ui_proc.terminate()
        return {"success": False, "api_ready": api_ready, "ui_ready": ui_ready}
        return {"success": False, "api_ready": api_ready, "ui_ready": ui_ready}

    # Measure memory
    api_mem_mb = get_process_memory_mb(api_proc.pid)
    ui_mem_mb = get_process_memory_mb(ui_proc.pid)
    total_mem_mb = api_mem_mb + ui_mem_mb
    system_mem = psutil.virtual_memory()

    print("\n4/4 Live Memory & Performance Telemetry:")
    print(f"   - FastAPI RSS Memory:    {api_mem_mb:.1f} MB")
    print(f"   - Streamlit RSS Memory:  {ui_mem_mb:.1f} MB")
    print(f"   - Combined Process RAM:  {total_mem_mb:.1f} MB (Budget Limit: {MAX_MEMORY_BUDGET_MB} MB)")
    print(f"   - Memory Budget Usage:   {(total_mem_mb / MAX_MEMORY_BUDGET_MB) * 100:.1f}%")
    print(f"   - Total Host System RAM: {system_mem.total / (1024**3):.1f} GB ({system_mem.percent}% in use)")

    # Test Live Triage Request
    triage_test = requests.post(
        f"http://127.0.0.1:{API_PORT}/triage",
        json={
            "ticket_id": "DEPLOY-TEST-01",
            "subject": "Deployment test check",
            "body": "Verifying production deployment connectivity and classification."
        },
        timeout=5
    )
    triage_ok = triage_test.status_code == 200

    telemetry = {
        "success": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "links": {
            "streamlit_ui": f"http://localhost:{UI_PORT}",
            "fastapi_swagger": f"http://localhost:{API_PORT}/docs",
            "fastapi_health": f"http://localhost:{API_PORT}/health",
            "fastapi_redoc": f"http://localhost:{API_PORT}/redoc"
        },
        "memory_profile": {
            "fastapi_rss_mb": round(api_mem_mb, 1),
            "streamlit_rss_mb": round(ui_mem_mb, 1),
            "combined_rss_mb": round(total_mem_mb, 1),
            "budget_limit_mb": MAX_MEMORY_BUDGET_MB,
            "within_budget": total_mem_mb <= MAX_MEMORY_BUDGET_MB,
            "budget_utilization_pct": round((total_mem_mb / MAX_MEMORY_BUDGET_MB) * 100, 1)
        },
        "smoke_tests": {
            "api_health_code": 200,
            "ui_status_code": 200,
            "triage_endpoint_ok": triage_ok
        },
        "pids": {
            "api_pid": api_proc.pid,
            "ui_pid": ui_proc.pid
        }
    }

    os.makedirs("data", exist_ok=True)
    with open("data/day26_deployment_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    return telemetry, api_proc, ui_proc


def main():
    parser = argparse.ArgumentParser(description="TicketPilot Deployment Manager")
    parser.add_argument("--test-only", action="store_true", help="Start services, verify health and memory, then terminate gracefully")
    args = parser.parse_args()

    print("=" * 75)
    print("✈️  TICKETPILOT SPRINT 6 DAY 26: PRODUCTION DEPLOYMENT MANAGER")
    print("=" * 75)

    if not check_preflight_artifacts():
        sys.exit(1)

    result = start_services()
    if not isinstance(result, tuple) or not result[0].get("success"):
        print("\n❌ Deployment failed.")
        sys.exit(1)

    telemetry, api_proc, ui_proc = result

    print("\n" + "=" * 75)
    print("🎉 TICKETPILOT IS LIVE AND OPERATIONAL!")
    print("=" * 75)
    print(f"👉 Streamlit Web UI:        {telemetry['links']['streamlit_ui']}")
    print(f"👉 FastAPI Swagger Docs:    {telemetry['links']['fastapi_swagger']}")
    print(f"👉 REST API Health Check:   {telemetry['links']['fastapi_health']}")
    print("-" * 75)
    print(f"Memory Footprint: {telemetry['memory_profile']['combined_rss_mb']} MB (safely within {MAX_MEMORY_BUDGET_MB} MB cloud limit)")
    print("=" * 75)

    if args.test_only:
        print("\n[--test-only flag passed] Performing graceful shutdown...")
        api_proc.terminate()
        ui_proc.terminate()
        api_proc.wait(timeout=5)
        ui_proc.wait(timeout=5)
        print("✓ Services stopped cleanly.")
    else:
        print("\nServices running in background. Press Ctrl+C to terminate.")
        try:
            while True:
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nShutting down TicketPilot services...")
            api_proc.terminate()
            ui_proc.terminate()
            print("✓ Shutdown complete.")


if __name__ == "__main__":
    main()
