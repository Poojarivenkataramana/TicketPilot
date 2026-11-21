"""
TicketPilot - Sprint 5 Day 22: Production Routing Decision Logger
-----------------------------------------------------------------
Provides thread-safe audit logging for every triage and reply drafting decision.
Stores structured audit records in:
- logs/routing_decisions.jsonl (High-throughput append-only JSON Lines stream)
- logs/routing_decisions.csv (Relational snapshot for analytics & dashboarding)

Captured Attributes:
- log_id, timestamp (ISO 8601 UTC)
- ticket_id, subject, body_preview
- predicted_queue, confidence, predicted_priority
- top_kb_id, rerank_score
- routing_tier (TIER_1_AUTO_DRAFT, TIER_2_EXPEDITED_REVIEW, TIER_3_HUMAN_ASSIST)
- decision_category ('auto-draft' vs 'human review')
- auto_send_eligible, requires_human_approval
- draft_preview, latency_ms
"""

import os
import csv
import json
import uuid
import datetime
import threading
from typing import Dict, List, Any, Optional
import pandas as pd

LOGS_DIR = "logs"
JSONL_LOG_PATH = os.path.join(LOGS_DIR, "routing_decisions.jsonl")
CSV_LOG_PATH = os.path.join(LOGS_DIR, "routing_decisions.csv")

CSV_FIELDNAMES = [
    "log_id", "timestamp", "ticket_id", "subject", "body_preview",
    "predicted_queue", "confidence", "predicted_priority",
    "top_kb_id", "rerank_score", "routing_tier", "decision_category",
    "auto_send_eligible", "requires_human_approval",
    "citations", "draft_preview", "latency_ms"
]


class DecisionLogger:
    """
    Thread-safe production decision logging engine for TicketPilot.
    """

    def __init__(self, log_dir: str = LOGS_DIR):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, "routing_decisions.jsonl")
        self.csv_path = os.path.join(self.log_dir, "routing_decisions.csv")
        self._lock = threading.Lock()
        self._init_csv()

    def _init_csv(self):
        """Initializes CSV header if file does not exist."""
        with self._lock:
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                    writer.writeheader()

    def log_decision(
        self,
        ticket_id: str,
        subject: str,
        body: str,
        predicted_queue: str,
        confidence: float,
        predicted_priority: str,
        routing_tier: str,
        action: str,
        top_kb_id: Optional[str] = None,
        rerank_score: Optional[float] = None,
        auto_send_eligible: bool = False,
        requires_human_approval: bool = True,
        citations: Optional[List[str]] = None,
        draft_text: str = "",
        latency_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Logs a single routing decision record to both JSONL and CSV.
        """
        now = datetime.datetime.utcnow().isoformat() + "Z"
        log_id = f"LOG-{uuid.uuid4().hex[:8].upper()}"

        # Category: 'auto-draft' vs 'human review'
        if routing_tier == "TIER_1_AUTO_DRAFT":
            decision_category = "auto-draft"
        else:
            decision_category = "human review"

        body_preview = (body[:100] + "...") if len(body) > 100 else body
        draft_preview = (draft_text.replace("\n", " ")[:120] + "...") if len(draft_text) > 120 else draft_text.replace("\n", " ")
        citations_str = ", ".join(citations) if citations else "None"

        record = {
            "log_id": log_id,
            "timestamp": now,
            "ticket_id": ticket_id or "TCK-UNKNOWN",
            "subject": subject,
            "body_preview": body_preview,
            "predicted_queue": predicted_queue,
            "confidence": round(float(confidence), 4),
            "predicted_priority": predicted_priority,
            "top_kb_id": top_kb_id or "None",
            "rerank_score": round(float(rerank_score), 2) if rerank_score is not None else -99.0,
            "routing_tier": routing_tier,
            "decision_category": decision_category,
            "auto_send_eligible": bool(auto_send_eligible),
            "requires_human_approval": bool(requires_human_approval),
            "citations": citations_str,
            "draft_preview": draft_preview,
            "latency_ms": round(float(latency_ms), 2)
        }

        with self._lock:
            # 1. Append to JSONL
            with open(self.jsonl_path, mode="a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(record) + "\n")

            # 2. Append to CSV (write header if file newly created or empty)
            file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)

        return record

    def get_recent_decisions(
        self,
        limit: int = 50,
        tier: Optional[str] = None,
        queue: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Reads recent decisions from the JSONL log with optional filtering.
        """
        if not os.path.exists(self.jsonl_path):
            return []

        decisions = []
        with self._lock:
            with open(self.jsonl_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            decisions.append(json.loads(line.strip()))
                        except Exception:
                            continue

        # Reverse to get newest first
        decisions.reverse()

        # Apply filters
        if tier:
            decisions = [d for d in decisions if d.get("routing_tier") == tier]
        if queue:
            decisions = [d for d in decisions if d.get("predicted_queue") == queue]

        return decisions[:limit]

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Computes real-time telemetry metrics over all logged decisions.
        """
        if not os.path.exists(self.jsonl_path):
            return {
                "total_logged": 0,
                "auto_draft_count": 0,
                "human_review_count": 0,
                "auto_draft_rate_pct": 0.0,
                "human_review_rate_pct": 0.0,
                "avg_confidence_pct": 0.0,
                "avg_latency_ms": 0.0
            }

        decisions = []
        with self._lock:
            with open(self.jsonl_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            decisions.append(json.loads(line.strip()))
                        except Exception:
                            continue

        total = len(decisions)
        if total == 0:
            return {
                "total_logged": 0,
                "auto_draft_count": 0,
                "human_review_count": 0,
                "auto_draft_rate_pct": 0.0,
                "human_review_rate_pct": 0.0,
                "avg_confidence_pct": 0.0,
                "avg_latency_ms": 0.0
            }

        df = pd.DataFrame(decisions)
        auto_drafts = int((df["decision_category"] == "auto-draft").sum())
        human_reviews = total - auto_drafts

        return {
            "total_logged": total,
            "auto_draft_count": auto_drafts,
            "human_review_count": human_reviews,
            "auto_draft_rate_pct": round((auto_drafts / total) * 100, 2),
            "human_review_rate_pct": round((human_reviews / total) * 100, 2),
            "avg_confidence_pct": round(df["confidence"].mean() * 100, 2),
            "avg_latency_ms": round(df["latency_ms"].mean(), 2),
            "queue_distribution": df["predicted_queue"].value_counts().to_dict(),
            "tier_distribution": df["routing_tier"].value_counts().to_dict()
        }


# Global singleton instance
decision_logger = DecisionLogger()
