"""Affected-person classification (HIGH / MEDIUM / LOW) on top of the engine.

These categories indicate potential exposure to an incident, not guilt.
"""

from __future__ import annotations

import logging
from typing import Optional

from utils.config import THRESHOLD_HIGH, THRESHOLD_MEDIUM, validate_thresholds

logger = logging.getLogger(__name__)


class ThreatEngine:
    """Classifies reconstruction results into risk bands."""

    def __init__(self, high: float = THRESHOLD_HIGH, medium: float = THRESHOLD_MEDIUM) -> None:
        validate_thresholds(high, medium)
        self.high = high
        self.medium = medium

    def update_thresholds(self, high: float, medium: float) -> None:
        validate_thresholds(high, medium)
        self.high = high
        self.medium = medium

    def classify(self, probability: float) -> str:
        if probability >= self.high:
            return "HIGH"
        if probability >= self.medium:
            return "MEDIUM"
        return "LOW"

    def classify_results(self, results: list[dict]) -> dict[str, list[dict]]:
        bands = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for r in results:
            band = self.classify(float(r.get("probability", 0.0)))
            r["risk_level"] = band
            bands[band].append(r)
        return bands

    def summary(self, results: list[dict]) -> dict:
        bands = self.classify_results(results)
        return {
            "total": len(results),
            "bands": bands,
            "counts": {k: len(v) for k, v in bands.items()},
            "thresholds": {"high": self.high, "medium": self.medium},
        }