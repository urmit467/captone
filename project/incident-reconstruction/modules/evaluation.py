"""Evaluation metrics comparing predictions against synthetic ground truth.

Metrics are computed against the ``ground_truth`` collection, never against the
system's own predictions.
"""

from __future__ import annotations

import logging

from database.db_operations import DBOperations
from modules.threat_engine import ThreatEngine

logger = logging.getLogger(__name__)


class Evaluator:
    """Compares affected-person predictions with ground truth."""

    def __init__(self, db_ops: Optional[DBOperations] = None) -> None:
        self.db = db_ops or DBOperations()
        self.threat = ThreatEngine()

    def evaluate(self, incident_id: str, threshold_high=None, threshold_medium=None) -> dict:
        if threshold_high is not None or threshold_medium is not None:
            self.threat.update_thresholds(
                threshold_high or self.threat.high,
                threshold_medium or self.threat.medium,
            )

        predictions = self.db.list_predictions(incident_id)
        ground_truth = {g["person_id"]: g for g in self.db.list_ground_truth(incident_id)}
        if not predictions or not ground_truth:
            if not predictions and not ground_truth:
                error = (
                    "No predictions and no ground truth for this incident. "
                    "Run reconstruction first, and generate demo data for ground truth."
                )
            elif not predictions:
                error = (
                    "No predictions for this incident yet. "
                    "Run reconstruction first."
                )
            else:
                error = (
                    "No ground truth for this incident. "
                    "Ground truth only exists for demo datasets (e.g. INC_001). "
                    "Insert ground truth rows manually to evaluate a custom incident."
                )
            return {
                "incident_id": incident_id,
                "error": error,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "localization_accuracy": 0.0,
                "path_reconstruction_accuracy": 0.0,
                "travel_time_error_minutes": None,
                "details": [],
            }

        # Affected = present on the affected path at incident time.
        y_true = {pid: bool(g.get("present_on_affected_path")) for pid, g in ground_truth.items()}
        # Predicted affected: probability >= MEDIUM threshold.
        y_pred = {
            p["person_id"]: self.threat.classify(p["probability"]) in ("HIGH", "MEDIUM")
            for p in predictions
        }

        all_pids = set(y_true) | set(y_pred)
        tp = sum(1 for p in all_pids if y_true.get(p) and y_pred.get(p))
        fp = sum(1 for p in all_pids if not y_true.get(p) and y_pred.get(p))
        fn = sum(1 for p in all_pids if y_true.get(p) and not y_pred.get(p))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # Incident-time localization accuracy: how often the predicted path at T
        # matches the affected path for people predicted affected.
        loc_correct = loc_total = 0
        path_correct = path_total = 0
        travel_errors = []
        details = []
        for p in predictions:
            pid = p["person_id"]
            gt = ground_truth.get(pid)
            if not gt:
                continue
            pred_affected = y_pred.get(pid, False)
            true_affected = y_true.get(pid, False)
            pred_path = (p.get("reconstructed_path") or [])
            predicted_at_incident = pred_path[-1] if pred_path else None
            loc_ok = bool(pred_affected and predicted_at_incident == gt.get("affected_path_id"))
            loc_total += 1
            loc_correct += 1 if loc_ok else 0

            actual_seq = gt.get("actual_path_sequence") or []
            path_ok = bool(actual_seq and pred_path and pred_path == actual_seq)
            path_total += 1
            path_correct += 1 if path_ok else 0

            # Travel-time estimation error vs ground truth entry/exit.
            if gt.get("actual_entry_time") and p.get("predicted_entry_time"):
                from dateutil import parser
                e1 = parser.isoparse(p["predicted_entry_time"])
                e2 = parser.isoparse(gt["actual_entry_time"])
                travel_errors.append(abs((e1 - e2).total_seconds()) / 60.0)

            details.append({
                "person_id": pid,
                "true_affected": true_affected,
                "predicted_affected": pred_affected,
                "probability": p.get("probability"),
                "risk_level": p.get("risk_level"),
                "localization_correct": loc_ok,
                "path_reconstruction_correct": path_ok,
            })

        avg_travel_error = (sum(travel_errors) / len(travel_errors)) if travel_errors else None

        return {
            "incident_id": incident_id,
            "thresholds": {"high": self.threat.high, "medium": self.threat.medium},
            "counts": {
                "true_affected": sum(1 for v in y_true.values() if v),
                "predicted_affected": sum(1 for v in y_pred.values() if v),
                "tp": tp, "fp": fp, "fn": fn,
            },
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "localization_accuracy": loc_correct / loc_total if loc_total else 0.0,
            "path_reconstruction_accuracy": path_correct / path_total if path_total else 0.0,
            "travel_time_error_minutes": avg_travel_error,
            "details": details,
        }