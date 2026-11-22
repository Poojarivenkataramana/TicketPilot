"""
TicketPilot - Sprint 6 Day 30: Test Suite for Mock Interview & Weak Spot Remediation
------------------------------------------------------------------------------------
Verifies:
1. Markdown file existence and word count (>2,000 words).
2. Coverage of all 5 Mock Interview Phases:
   - Phase 1: Architecture & System Design
   - Phase 2: Machine Learning & Statistical Modeling
   - Phase 3: Information Retrieval & RAG Architecture
   - Phase 4: Production Systems, Profiling & Deployment
   - Phase 5: Behavioral & Operational Governance
3. Coverage of the 4 Weak Spots Remediation Battlecards (F1 ceiling, ChromaDB scale, TF-IDF OOV, Synthetic data).
4. Coverage of Engineering War Stories Log.
5. Presence of the complete 30-Day Sprint Plan Retrospective.
"""

import os

FILE_PATH = "career/mock_interview_and_feedback.md"


def test_file_exists_and_substantial():
    assert os.path.exists(FILE_PATH), f"Missing {FILE_PATH}"
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    words = content.split()
    assert len(words) >= 2000, f"Expected at least 2000 words, got {len(words)}"
    print(f"✅ PASSED: Document length verified ({len(words)} words total).")


def test_mock_interview_5_phases():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Phase 1: Architecture" in content, "Missing Phase 1: Architecture"
    assert "Phase 2: Machine Learning" in content, "Missing Phase 2: Machine Learning"
    assert "Phase 3: Information Retrieval" in content, "Missing Phase 3: Information Retrieval"
    assert "Phase 4: Production Systems" in content, "Missing Phase 4: Production Systems"
    assert "Phase 5: Behavioral" in content, "Missing Phase 5: Behavioral"
    print("✅ PASSED: All 5 Technical Mock Interview phases verified.")


def test_weak_spots_remediation():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Weak Spots Identification & Remediation" in content, "Missing Weak Spots Section"
    assert "54% Macro F1" in content or "54%" in content, "Missing 54% F1 defense"
    assert "ChromaDB" in content, "Missing ChromaDB scalability defense"
    assert "TF-IDF" in content, "Missing TF-IDF vocabulary adaptation defense"
    assert "synthetic" in content.lower(), "Missing synthetic data disclosure defense"
    print("✅ PASSED: All 4 Weak Spot Remediation Battlecards verified.")


def test_war_stories_log():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Engineering \"War Stories\" Log" in content or "War Stories" in content, "Missing War Stories section"
    assert "Subprocess Pipe" in content or "deadlock" in content.lower(), "Missing Subprocess incident"
    assert "Plotly" in content, "Missing Plotly subplot incident"
    assert "1024 MB" in content, "Missing memory limit incident"
    print("✅ PASSED: Production War Stories Log verified.")


def test_retrospective_completion():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    assert "30-Day Sprint Plan Completion Retrospective" in content or "Retrospective" in content, "Missing Retrospective table"
    for sprint in ["Sprint 1", "Sprint 2", "Sprint 3", "Sprint 4", "Sprint 5", "Sprint 6"]:
        assert sprint in content, f"Missing {sprint} in retrospective table"
    print("✅ PASSED: All 6 Sprints and 30 Days verified complete.")


if __name__ == "__main__":
    test_file_exists_and_substantial()
    test_mock_interview_5_phases()
    test_weak_spots_remediation()
    test_war_stories_log()
    test_retrospective_completion()
    print("\n🎉 ALL DAY 30 CHECKS PASSED SUCCESSFULLY! THE 30-DAY SPRINT PLAN IS 100% COMPLETE!")
