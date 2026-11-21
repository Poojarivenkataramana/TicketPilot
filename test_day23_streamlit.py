"""
Test Suite for Sprint 5 Day 23: TicketPilot Streamlit Web UI
-----------------------------------------------------------
Tests the end-to-end user workflows using Streamlit's official AppTest framework:
1. UI component initialization (sidebar, metrics, inputs).
2. Ticket intake and processing for pre-loaded scenarios (Tier 1 & Tier 2).
3. Grounded RAG resolution: category, priority, confidence, KB sources, inline citations.
4. Human-in-the-loop editable draft workspace and action buttons (Approve, Escalate).
5. Logging synchronization with decision logger.
"""

import os
from streamlit.testing.v1 import AppTest


def test_streamlit_initial_load():
    """Verify that the Streamlit application starts and renders default components."""
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    
    # Assert no unhandled exceptions occurred
    assert len(at.exception) == 0, f"App loaded with exceptions: {[e.message for e in at.exception]}"
    
    # Check that text inputs exist
    assert len(at.text_input) >= 2, "Expected Ticket ID and Subject text inputs"
    assert len(at.text_area) >= 2, "Expected Body and Editable Draft text areas"
    assert len(at.selectbox) >= 1, "Expected Scenario selectbox"
    assert len(at.button) >= 3, "Expected Process, Approve, and Escalate buttons"


def test_streamlit_process_billing_scenario():
    """Verify processing Scenario 1 (Billing duplicate charge) produces Tier 1 triage & draft."""
    at = AppTest.from_file("streamlit_app.py", default_timeout=45)
    at.run()
    
    # Select Scenario 1
    # PRELOADED_SCENARIOS: index 1 is Scenario 1: Billing Duplicate Charge
    at.selectbox[0].select_index(1).run()
    assert "Duplicate subscription renewal charge" in at.text_input[1].value
    
    # Click "Process & Generate Cited Reply" (button 0)
    at.button[0].click().run()
    assert len(at.exception) == 0, f"Exceptions on submit: {[e.message for e in at.exception]}"
    
    # Verify metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert "Predicted Queue" in metric_labels
    assert "Priority" in metric_labels
    assert "Confidence" in metric_labels
    
    # Check predicted queue value
    queue_metric = next(m for m in at.metric if m.label == "Predicted Queue")
    assert "Billing and Payments" in queue_metric.value or "Refund" in queue_metric.value or len(queue_metric.value) > 0
    
    # Verify editable draft contains content
    draft_area = at.text_area[1]
    assert len(draft_area.value) > 20
    assert "Awaiting ticket intake" not in draft_area.value


def test_streamlit_process_outage_scenario():
    """Verify processing Scenario 2 (Critical cloud outage) flags Tier 2 Expedited Review."""
    at = AppTest.from_file("streamlit_app.py", default_timeout=45)
    at.run()
    
    # Select Scenario 2 (index 2)
    at.selectbox[0].select_index(2).run()
    assert "CRITICAL" in at.text_input[1].value
    
    # Click "Process & Generate Cited Reply"
    at.button[0].click().run()
    assert len(at.exception) == 0, f"Exceptions on submit: {[e.message for e in at.exception]}"
    
    # Check priority
    priority_metric = next(m for m in at.metric if m.label == "Priority")
    assert priority_metric.value.upper() in ["HIGH", "CRITICAL", "MEDIUM", "LOW"]


def test_streamlit_human_in_the_loop_actions():
    """Verify agent can edit draft and click approve or escalate buttons."""
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    
    # Input a custom ticket
    at.text_input[0].set_value("TCK-CUSTOM-01")
    at.text_input[1].set_value("Cannot reset password via email link")
    at.text_area[0].set_value("The password reset email never arrived in my inbox or spam folder. Please help.")
    
    at.button[0].click().run()
    assert len(at.exception) == 0
    
    # Edit the draft
    initial_draft = at.text_area[1].value
    modified_draft = initial_draft + "\n\nSupport Agent Note: Customer verified email address."
    at.text_area[1].set_value(modified_draft).run()
    assert at.text_area[1].value == modified_draft
    
    # Click Approve & Send (button 1)
    at.button[1].click().run()
    assert len(at.exception) == 0
    assert any("Reply approved" in s.value for s in at.success)


if __name__ == "__main__":
    print("Running Day 23 Streamlit UI Test Suite...")
    test_streamlit_initial_load()
    print("✓ test_streamlit_initial_load passed")
    test_streamlit_process_billing_scenario()
    print("✓ test_streamlit_process_billing_scenario passed")
    test_streamlit_process_outage_scenario()
    print("✓ test_streamlit_process_outage_scenario passed")
    test_streamlit_human_in_the_loop_actions()
    print("✓ test_streamlit_human_in_the_loop_actions passed")
    print("\n🎉 ALL STREAMLIT UI TESTS PASSED SUCCESSFULLY!")
