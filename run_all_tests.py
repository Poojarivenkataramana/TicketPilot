"""
TicketPilot - Sprint 5 Day 25: Master Test Runner
-------------------------------------------------
Orchestrates and executes all unit, integration, and UI test suites:
1. test_cleaning.py: Text cleaning, normalization, and deduplication
2. test_routing.py: Three-tier routing policy and boundary conditions
3. test_api.py: FastAPI REST endpoints, schema validation, and health checks
4. test_day22_logging.py: Thread-safe Decision Logger audit stream
5. test_day23_streamlit.py: Streamlit AppTest headless interface
6. test_day24_dashboard.py: Operational Plotly dashboard metrics & batch runs
7. verify_docker_setup.py: Docker container configuration validator
"""

import sys
import time
import subprocess
import json

SUITES = [
    ("Text Cleaning & Sanitization", "test_cleaning.py"),
    ("Compound Routing Policy", "test_routing.py"),
    ("FastAPI REST Endpoints", "test_api.py"),
    ("Decision Logger & Audit Streams", "test_day22_logging.py"),
    ("Streamlit Interactive Web UI", "test_day23_streamlit.py"),
    ("Plotly Operational Dashboard", "test_day24_dashboard.py"),
    ("Docker Container Setup", "verify_docker_setup.py"),
]


def run_master_suite():
    print("=" * 75)
    print("✈️  TICKETPILOT SPRINT 5 DAY 25: COMPREHENSIVE TEST RUNNER")
    print("=" * 75)

    summary = []
    t_suite_start = time.perf_counter()

    for name, script_path in SUITES:
        print(f"\n[RUNNING] {name} ({script_path})...")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = time.perf_counter() - t0

        if proc.returncode == 0:
            print(f"  ✓ PASS ({duration:.2f}s)")
            summary.append({
                "suite": name,
                "script": script_path,
                "status": "PASS",
                "duration_s": duration
            })
        else:
            print(f"  ❌ FAIL ({duration:.2f}s)")
            print("  --- ERROR OUTPUT ---")
            print(proc.stderr or proc.stdout)
            print("  --------------------")
            summary.append({
                "suite": name,
                "script": script_path,
                "status": "FAIL",
                "duration_s": duration,
                "error": proc.stderr or proc.stdout
            })

    total_time = time.perf_counter() - t_suite_start
    total_passed = sum(1 for s in summary if s["status"] == "PASS")
    total_failed = sum(1 for s in summary if s["status"] == "FAIL")

    print("\n" + "=" * 75)
    print("TEST EXECUTION SCORECARD")
    print("=" * 75)
    for s in summary:
        status_icon = "✓" if s["status"] == "PASS" else "❌"
        print(f"  {status_icon} {s['suite']:35s} | {s['status']:4s} | {s['duration_s']:6.2f}s")

    print("-" * 75)
    print(f"Total Suites: {len(summary)} | Passed: {total_passed} | Failed: {total_failed} | Time: {total_time:.2f}s")
    print("=" * 75)

    if total_failed == 0:
        print("🎉 ALL TEST SUITES PASSED WITH 100% SUCCESS RATE!")
        return 0
    else:
        print(f"⚠️ {total_failed} TEST SUITE(S) FAILED.")
        return 1


if __name__ == "__main__":
    exit_code = run_master_suite()
    sys.exit(exit_code)
