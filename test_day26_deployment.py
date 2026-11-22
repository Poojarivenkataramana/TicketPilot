"""
Test Suite for Sprint 6 Day 26: Production Deployment & Memory Profiling
-----------------------------------------------------------------------
Verifies:
1. Deployment configuration files (.streamlit/config.toml, render.yaml).
2. deploy.py execution in test-only mode.
3. Verification of live links and endpoints (Streamlit 8501, FastAPI 8000).
4. Memory budget compliance (RSS <= 1024 MB).
5. JSON telemetry output integrity.
"""

import os
import json
import subprocess
import sys


def test_deployment_configs():
    """Verify presence of cloud and container deployment manifests."""
    assert os.path.exists(".streamlit/config.toml"), "Missing .streamlit/config.toml"
    assert os.path.exists("render.yaml"), "Missing render.yaml"
    assert os.path.exists("deploy.py"), "Missing deploy.py"

    with open(".streamlit/config.toml", "r", encoding="utf-8") as f:
        content = f.read()
    assert "headless = true" in content
    assert "port = 8501" in content

    with open("render.yaml", "r", encoding="utf-8") as f:
        r_content = f.read()
    assert "ticketpilot-api" in r_content
    assert "ticketpilot-ui" in r_content


def test_deploy_script_execution():
    """Executes deploy.py with --test-only and asserts clean startup and shutdown."""
    proc = subprocess.run(
        [sys.executable, "deploy.py", "--test-only"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90
    )
    assert proc.returncode == 0, f"deploy.py failed: {proc.stderr or proc.stdout}"
    assert "TICKETPILOT IS LIVE AND OPERATIONAL" in proc.stdout
    assert "FastAPI healthy" in proc.stdout
    assert "Streamlit UI responsive" in proc.stdout


def test_telemetry_artifact():
    """Verifies that data/day26_deployment_telemetry.json is well-formed and compliant."""
    telemetry_path = "data/day26_deployment_telemetry.json"
    assert os.path.exists(telemetry_path), f"Missing {telemetry_path}"

    with open(telemetry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["success"] is True
    assert "http://localhost:8501" in data["links"]["streamlit_ui"]
    assert "http://localhost:8000/docs" in data["links"]["fastapi_swagger"]
    assert data["memory_profile"]["within_budget"] is True
    assert data["memory_profile"]["combined_rss_mb"] <= 1024

    print(f"\nDeployment Telemetry Verified:")
    print(f" - UI Link:      {data['links']['streamlit_ui']}")
    print(f" - Swagger Docs: {data['links']['fastapi_swagger']}")
    print(f" - Memory RSS:   {data['memory_profile']['combined_rss_mb']} MB (Budget: 1024 MB)")
    print(f" - Budget Used:  {data['memory_profile']['budget_utilization_pct']}%")


if __name__ == "__main__":
    print("Running Day 26 Deployment Test Suite...")
    test_deployment_configs()
    print("✓ test_deployment_configs passed")
    test_deploy_script_execution()
    print("✓ test_deploy_script_execution passed")
    test_telemetry_artifact()
    print("✓ test_telemetry_artifact passed")
    print("\n🎉 ALL DAY 26 DEPLOYMENT TESTS PASSED SUCCESSFULLY!")
