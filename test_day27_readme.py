"""
Test Suite for Sprint 6 Day 27: README and Documentation Verification
---------------------------------------------------------------------
Verifies that README.md satisfies all Day 27 requirements:
1. File existence and depth (> 10,000 characters).
2. End-to-end architecture diagram (Mermaid flowchart).
3. Quantitative performance tables (triage baselines, retrieval metrics, qualitative scores, latency).
4. Explicit disclosure that data is synthetic with associated technical limitations.
5. Interactive demo walkthrough and UI wireframe.
6. Getting started and containerization instructions.
"""

import os


def test_readme_exists_and_substantial():
    """Verify README.md exists and contains detailed documentation."""
    readme_path = "README.md"
    assert os.path.exists(readme_path), "README.md is missing!"
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert len(content) > 10000, f"README.md is too brief ({len(content)} characters)"
    return content


def test_readme_architecture_diagram(content: str):
    """Verify Mermaid architecture diagram is present."""
    assert "```mermaid" in content, "Missing Mermaid architecture diagram"
    assert "Scoped Hybrid Vector Retrieval" in content or "Hybrid Vector Retrieval" in content
    assert "Compound Routing Policy" in content


def test_readme_results_tables(content: str):
    """Verify all 4 core benchmark results tables are present."""
    # 1. Triage baseline table
    assert "Linear SVM" in content
    assert "54.32%" in content or "54.15%" in content

    # 2. Retrieval benchmark table
    assert "Hit Rate@1" in content
    assert "0.9033" in content or "90.00%" in content

    # 3. Human qualitative scoring table
    assert "4.81 / 5.00" in content or "4.81" in content

    # 4. Latency table
    assert "656.55 ms" in content or "609.16 ms" in content


def test_readme_limitations_and_synthetic_data(content: str):
    """Verify explicit statement that data is synthetic and documents limitations."""
    assert "synthetic" in content.lower(), "Missing explicit mention that data is synthetic"
    assert "Limitations" in content, "Missing limitations section"


def test_readme_demo_and_usage(content: str):
    """Verify demo walkthrough, REST API examples, and run instructions."""
    assert "http://localhost:8501" in content
    assert "http://localhost:8000/docs" in content
    assert "POST /triage" in content
    assert "POST /draft-reply" in content
    assert "docker compose" in content or "docker run" in content


if __name__ == "__main__":
    print("Running Day 27 README Verification Test Suite...")
    doc = test_readme_exists_and_substantial()
    print("✓ test_readme_exists_and_substantial passed")
    test_readme_architecture_diagram(doc)
    print("✓ test_readme_architecture_diagram passed")
    test_readme_results_tables(doc)
    print("✓ test_readme_results_tables passed")
    test_readme_limitations_and_synthetic_data(doc)
    print("✓ test_readme_limitations_and_synthetic_data passed")
    test_readme_demo_and_usage(doc)
    print("✓ test_readme_demo_and_usage passed")
    print("\n🎉 ALL DAY 27 README VERIFICATION TESTS PASSED SUCCESSFULLY!")
