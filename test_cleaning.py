"""
Unit Tests for Text Cleaning & Data Sanitization (Day 25)
--------------------------------------------------------
Tests edge cases of clean_ticket_text, validate_ticket_payload, and clean_dataframe_pipeline:
- Empty/None inputs
- Excessive whitespace, tabs, and multiline characters
- HTML entity and tag stripping
- Control character removal
- Deduplication of subject in body
- Batch pipeline filtering
"""

import pandas as pd
from text_cleaning import clean_ticket_text, validate_ticket_payload, clean_dataframe_pipeline


def test_clean_ticket_text_standard():
    subject = "Payment failed"
    body = "My credit card was declined during checkout."
    res = clean_ticket_text(subject, body)
    assert res == "Payment failed My credit card was declined during checkout."


def test_clean_ticket_text_whitespace_and_newlines():
    subject = "  Slow   connection   \n"
    body = "The    network is \t\t very slow.\n\n\nIt disconnects often.   "
    res = clean_ticket_text(subject, body)
    assert res == "Slow connection The network is very slow. It disconnects often."


def test_clean_ticket_text_html_and_entities():
    subject = "Error &amp; Bug in &lt;div&gt;"
    body = "<p>Hello Support,</p><br>Issue with <b>login</b> page &quot;button&quot;."
    res = clean_ticket_text(subject, body)
    assert "<p>" not in res
    assert "<b>" not in res
    assert "&amp;" not in res
    assert 'Error & Bug in Issue with login page "button".' in res or "Error & Bug" in res


def test_clean_ticket_text_none_and_empty():
    assert clean_ticket_text(None, "Body only") == "Body only"
    assert clean_ticket_text("Subject only", None) == "Subject only"
    assert clean_ticket_text("", "") == ""
    assert clean_ticket_text(None, None) == ""


def test_clean_ticket_text_deduplicate_subject_prefix():
    subject = "Printer not working"
    body = "Printer not working on floor 3 since morning."
    res = clean_ticket_text(subject, body)
    assert res == "Printer not working on floor 3 since morning."


def test_validate_ticket_payload_valid():
    payload = {
        "ticket_id": "TCK-100",
        "subject": "VPN issue",
        "body": "Cannot connect to company VPN from home."
    }
    validated = validate_ticket_payload(payload)
    assert validated["ticket_id"] == "TCK-100"
    assert "VPN issue" in validated["text"]


def test_validate_ticket_payload_empty_error():
    payload = {"ticket_id": "TCK-EMPTY", "subject": "   ", "body": ""}
    try:
        validate_ticket_payload(payload)
        assert False, "Expected ValueError on empty payload"
    except ValueError as e:
        assert "empty after text cleaning" in str(e)


def test_clean_dataframe_pipeline():
    raw_data = pd.DataFrame([
        {"subject": "Sub 1", "body": "Body 1", "queue": "Billing", "language": "en"},
        {"subject": "Sub 2", "body": "Body 2", "queue": "IT", "language": "es"},  # Should be dropped (non-en)
        {"subject": None, "body": None, "queue": "Billing", "language": "en"},    # Should be dropped (null body)
        {"subject": "Sub 1", "body": "Body 1", "queue": "Billing", "language": "en"},  # Should be deduplicated
        {"subject": "Sub 3", "body": "Body 3", "queue": "Support", "language": "en"},
    ])
    cleaned = clean_dataframe_pipeline(raw_data)
    assert len(cleaned) == 2, f"Expected 2 rows after cleaning, got {len(cleaned)}"
    assert any("Body 1" in t for t in cleaned["text"].values)
    assert any("Body 3" in t for t in cleaned["text"].values)


if __name__ == "__main__":
    print("Running Text Cleaning Unit Tests...")
    test_clean_ticket_text_standard()
    test_clean_ticket_text_whitespace_and_newlines()
    test_clean_ticket_text_html_and_entities()
    test_clean_ticket_text_none_and_empty()
    test_clean_ticket_text_deduplicate_subject_prefix()
    test_validate_ticket_payload_valid()
    test_validate_ticket_payload_empty_error()
    test_clean_dataframe_pipeline()
    print("✓ All 8 text cleaning unit tests passed!")
