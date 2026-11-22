"""
TicketPilot - Sprint 6 Day 29: Test Suite for Interview Pitch & Technical Q&A
------------------------------------------------------------------------------
Verifies:
1. Markdown file existence and word count (>2,000 words).
2. Word-count of 2-Minute Elevator Pitch script (must be 250-350 words for 2-min spoken delivery).
3. Coverage of all 5 mandatory deep technical interview questions:
   - Question 1: Model choice (Linear SVM vs. DistilBERT vs. frozen embeddings, latency, RAM)
   - Question 2: Error analysis & cascading failure prevention
   - Question 3: Threshold calibration (confidence 0.40, Cross-Encoder +4.0)
   - Question 4: Zero data leakage protocol
   - Question 5: RAG evaluation methodology (Hit@K, MRR, citation grounding)
4. Presence of Self-Recording practice rubric.
"""

import os
import re

FILE_PATH = "career/interview_pitch_and_qa.md"


def test_file_exists_and_substantial():
    assert os.path.exists(FILE_PATH), f"Missing {FILE_PATH}"
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    words = content.split()
    assert len(words) >= 2000, f"Expected at least 2000 words, got {len(words)}"
    print(f"✅ PASSED: Document length verified ({len(words)} words total).")


def test_elevator_pitch_pacing():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract elevator pitch quote
    match = re.search(r'### Word-for-Word Speaking Script\s+>\s+"(.*?)"', content, re.DOTALL)
    assert match, "Could not find word-for-word speaking script blockquote"

    pitch_text = match.group(1).replace("\n>", " ").strip()
    words = pitch_text.split()
    word_count = len(words)

    # 2-minute speech at 130-150 words/min requires 250 to 350 words
    assert 250 <= word_count <= 350, (
        f"Elevator pitch word count ({word_count}) is outside the 250-350 word range for a 2-minute pitch."
    )
    print(f"✅ PASSED: 2-minute pitch pacing verified ({word_count} words, ~1 min 55 sec spoken).")


def test_5_deep_technical_questions_covered():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Question 1: Model Choice & Cost/Latency
    assert "Question 1: Model Choice" in content, "Missing Question 1"
    assert "54.15%" in content, "Missing Linear SVM 54.15% F1 metric"
    assert "DistilBERT" in content, "Missing DistilBERT comparison"
    assert "0.001 ms" in content or "0.001" in content, "Missing latency metric"
    assert "12 MB" in content, "Missing memory footprint metric"

    # Question 2: Error Analysis & Cascading Failures
    assert "Question 2: Error Analysis" in content, "Missing Question 2"
    assert "cascading" in content.lower(), "Missing discussion of cascading errors"
    assert "Compound Routing Policy" in content or "compound routing" in content.lower(), "Missing routing policy explanation"

    # Question 3: Threshold Calibration
    assert "Question 3: Decision Threshold Calibration" in content, "Missing Question 3"
    assert "0.40" in content, "Missing 0.40 confidence cutoff"
    assert "+4.0" in content, "Missing +4.0 Cross-Encoder threshold"
    assert "Platt" in content or "CalibratedClassifierCV" in content, "Missing calibration method"

    # Question 4: Data Leakage Prevention
    assert "Question 4: Data Leakage" in content, "Missing Question 4"
    assert "17,469" in content, "Missing train set article count"
    assert "8,469" in content, "Missing test set count"
    assert "sequest" in content.lower() or "isolation" in content.lower() or "partition" in content.lower(), "Missing leakage mitigation details"

    # Question 5: RAG Evaluation Methodology
    assert "Question 5: RAG Evaluation" in content, "Missing Question 5"
    assert "Hit Rate" in content or "Hit@" in content, "Missing Hit Rate metric"
    assert "MRR" in content, "Missing MRR metric"
    assert "[KB-" in content, "Missing citation grounding verification"

    print("✅ PASSED: All 5 deep technical interview topics thoroughly covered with exact metrics.")


def test_self_recording_rubric():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Candidate Practice & Self-Recording Guide" in content, "Missing recording guide"
    assert "Voice Memos" in content or "QuickTime" in content, "Missing tool recommendations"
    assert "Rubric" in content or "Self-Check" in content, "Missing self-evaluation rubric"
    print("✅ PASSED: Candidate self-recording practice guide and evaluation rubric verified.")


if __name__ == "__main__":
    test_file_exists_and_substantial()
    test_elevator_pitch_pacing()
    test_5_deep_technical_questions_covered()
    test_self_recording_rubric()
    print("\n🎉 ALL DAY 29 CHECKS PASSED SUCCESSFULLY!")
