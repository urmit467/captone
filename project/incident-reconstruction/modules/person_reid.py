"""Automatic person re-identification across cameras.

Matches person detections to existing person records using appearance features
(color histograms of person crops) so the system needs no manual registration
or cross-camera identity association. Fully automatic.

This is a lightweight, dependency-free ReID approach suitable for an academic
prototype: it compares HSV color histograms of person crops. It is not
intended to replace a deep ReID model in production.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Histogram bins per HSV channel.
_H_BINS = 8
_S_BINS = 8
_V_BINS = 8


def extract_features(image) -> Optional[np.ndarray]:
    """Extract a normalized HSV color-histogram signature from a person crop.

    Returns None if the image is empty or feature extraction fails.
    """
    if image is None or image.size == 0:
        return None
    try:
        resized = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [_H_BINS, _S_BINS, _V_BINS],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
        return hist.flatten().astype(np.float32)
    except Exception as exc:  # pragma: no cover - opencv
        logger.warning("Feature extraction failed: %s", exc)
        return None


def compare_features(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Return histogram-correlation similarity in [0, 1]; higher = more similar."""
    if feat1 is None or feat2 is None:
        return 0.0
    try:
        sim = cv2.compareHist(
            feat1.reshape(_H_BINS, _S_BINS, _V_BINS),
            feat2.reshape(_H_BINS, _S_BINS, _V_BINS),
            cv2.HISTCMP_CORREL,
        )
        return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))
    except Exception:  # pragma: no cover - opencv
        return 0.0


class PersonReID:
    """Automatically matches person detections to existing person records."""

    def __init__(self, db_ops, threshold: float = 0.70) -> None:
        self.db = db_ops
        self.threshold = threshold

    def _person_features(self, person: dict) -> Optional[np.ndarray]:
        metadata = person.get("metadata") or {}
        feat = metadata.get("reid_features")
        if feat is None:
            return None
        try:
            return np.array(feat, dtype=np.float32)
        except (ValueError, TypeError):
            return None

    def match(self, image) -> Optional[str]:
        """Find the best matching existing person for ``image``.

        Returns the person_id if similarity >= threshold, else None.
        """
        feat = extract_features(image)
        if feat is None:
            return None

        best_id: Optional[str] = None
        best_sim = self.threshold

        for person in self.db.list_persons():
            pid = person["person_id"]
            existing_feat = self._person_features(person)
            if existing_feat is None:
                continue
            sim = compare_features(feat, existing_feat)
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        return best_id

    def save_features(self, person_id: str, image) -> None:
        """Store the feature signature for ``person_id`` (merged with existing metadata)."""
        feat = extract_features(image)
        if feat is None:
            return
        person = self.db.get_person(person_id)
        metadata = (person or {}).get("metadata") or {}
        metadata["reid_features"] = feat.tolist()
        try:
            self.db.update_person(person_id, {"metadata": metadata})
        except Exception as exc:  # pragma: no cover - db write
            logger.warning("Failed to save ReID features for %s: %s", person_id, exc)
