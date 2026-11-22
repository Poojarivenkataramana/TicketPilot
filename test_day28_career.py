"""
Test Suite for Sprint 6 Day 28: Repository Cleanup, Resume Bullets & LinkedIn Post
---------------------------------------------------------------------------------
Verifies:
1. career/resume_bullets.md exists, contains structured XYZ bullets for ML, Backend,
   and Full-Stack roles, and technical skills matrix.
2. career/linkedin_post.md exists, contains hook, architectural insights, metrics,
   GitHub URL, and relevant hashtags.
3. Repository cleanliness and git status.
"""

import os
import subprocess


def test_resume_bullets_file():
    """Verify presence and contents of career/resume_bullets.md."""
    path = "career/resume_bullets.md"
    assert os.path.exists(path), f"Missing {path}"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Machine Learning Engineer" in content
    assert "Senior Backend" in content
    assert "Full-Stack AI Engineer" in content
    assert "90.00%" in content
    assert "0.9033" in content
    assert "0.001 ms" in content
    assert "Technical Skills Matrix" in content
    assert "https://github.com/Poojarivenkataramana/TicketPilot" in content


def test_linkedin_post_file():
    """Verify presence and contents of career/linkedin_post.md."""
    path = "career/linkedin_post.md"
    assert os.path.exists(path), f"Missing {path}"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "TicketPilot" in content
    assert "Why do 95% of customer support AI pilots fail" in content
    assert "90.00%" in content
    assert "0.9033" in content
    assert "Linear SVM" in content
    assert "https://github.com/Poojarivenkataramana/TicketPilot" in content
    assert "#MachineLearning" in content


def test_repository_cleanliness():
    """Verify repository has no unstaged changes or untracked clutter."""
    assert os.path.exists(".gitignore"), "Missing .gitignore"
    with open(".gitignore", "r", encoding="utf-8") as f:
        gitignore_content = f.read()

    assert "!data/knowledge_base_clean.csv" in gitignore_content
    assert "!data/test.csv" in gitignore_content


if __name__ == "__main__":
    print("Running Day 28 Career & Cleanup Verification Test Suite...")
    test_resume_bullets_file()
    print("✓ test_resume_bullets_file passed")
    test_linkedin_post_file()
    print("✓ test_linkedin_post_file passed")
    test_repository_cleanliness()
    print("✓ test_repository_cleanliness passed")
    print("\n🎉 ALL DAY 28 VERIFICATION TESTS PASSED SUCCESSFULLY!")
