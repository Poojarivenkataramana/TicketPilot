"""
TicketPilot - Sprint 5 Day 25: Text Cleaning & Data Sanitization Module
----------------------------------------------------------------------
Provides production-grade text sanitization, normalization, and validation
for single ticket intake and batch dataset pipelines.
"""

import re
import html
from typing import Dict, Any, Optional
import pandas as pd


def clean_ticket_text(subject: Optional[str], body: Optional[str]) -> str:
    """
    Cleans, normalizes, and combines ticket subject and body into a single coherent text.
    Handles HTML entities, tags, control characters, excessive whitespace, and nulls.
    """
    subj = str(subject).strip() if subject is not None and pd.notna(subject) else ""
    bdy = str(body).strip() if body is not None and pd.notna(body) else ""

    # Remove HTML entities and tags
    subj = html.unescape(subj)
    bdy = html.unescape(bdy)
    subj = re.sub(r"<[^>]+>", " ", subj)
    bdy = re.sub(r"<[^>]+>", " ", bdy)

    # Combine subject and body
    if subj and bdy:
        # Avoid duplicate subject if body already starts with the exact subject
        if bdy.lower().startswith(subj.lower()):
            combined = bdy
        else:
            combined = f"{subj} {bdy}"
    elif subj:
        combined = subj
    else:
        combined = bdy

    # Remove control characters (except standard newlines/tabs during regex cleanup)
    combined = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", combined)

    # Normalize excessive whitespace, tabs, and newlines into single spaces
    combined = re.sub(r"\s+", " ", combined).strip()

    return combined


def validate_ticket_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Validates ticket dictionary schema and returns sanitized fields.
    Raises ValueError on missing or empty required text.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Payload must be a dictionary, got {type(payload)}")

    ticket_id = str(payload.get("ticket_id", "TCK-UNKNOWN")).strip()
    subject = str(payload.get("subject", "")).strip() if payload.get("subject") is not None else ""
    body = str(payload.get("body", "")).strip() if payload.get("body") is not None else ""

    full_cleaned = clean_ticket_text(subject, body)
    if not full_cleaned:
        raise ValueError("Ticket content is empty after text cleaning. Subject or body must contain text.")

    return {
        "ticket_id": ticket_id,
        "subject": subject,
        "body": body,
        "text": full_cleaned
    }


def clean_dataframe_pipeline(df: pd.DataFrame, strict_subject: bool = False) -> pd.DataFrame:
    """
    Batch cleaning pipeline for raw ticketing datasets:
    1. Filters for English tickets ('language' == 'en' if column exists)
    2. Drops rows with missing body or queue
    3. Cleans subject and body
    4. Deduplicates identical tickets
    """
    cleaned = df.copy()

    # Filter language if column present
    if "language" in cleaned.columns:
        cleaned = cleaned[cleaned["language"] == "en"].copy()

    # Drop missing critical columns
    req_cols = [c for c in ["body", "queue"] if c in cleaned.columns]
    cleaned = cleaned.dropna(subset=req_cols).copy()

    if strict_subject and "subject" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["subject"]).copy()
    elif "subject" in cleaned.columns:
        cleaned["subject"] = cleaned["subject"].fillna("")

    # Clean text column
    if "subject" in cleaned.columns and "body" in cleaned.columns:
        cleaned["text"] = cleaned.apply(
            lambda r: clean_ticket_text(r["subject"], r["body"]),
            axis=1
        )
    elif "body" in cleaned.columns:
        cleaned["text"] = cleaned["body"].apply(lambda b: clean_ticket_text("", b))

    # Deduplicate on text
    if "text" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return cleaned
