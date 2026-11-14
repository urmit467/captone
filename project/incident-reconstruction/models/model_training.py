"""Model training placeholder.

Per the project requirements, Scikit-learn (Random Forest) is used ONLY when
sufficient labeled training data exists. This module documents that boundary
and trains nothing with arbitrary or fabricated labels.
"""

from __future__ import annotations

import logging

from utils.config import THRESHOLD_HIGH, THRESHOLD_MEDIUM

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Placeholder: no model is trained unless real labeled incidents exist."""

    def __init__(self) -> None:
        self.model = None
        self.trained = False

    def has_labeled_data(self) -> bool:
        """True only when a minimum set of labeled incidents is available."""
        # Labeled data = incidents with confirmed ground-truth presence.
        return False

    def train(self) -> str:
        raise NotImplementedError(
            "Random Forest training is disabled until real labeled incident "
            "data exists. The reconstruction engine uses a transparent "
            "mathematical probability estimator instead."
        )

    def predict(self, features) -> str:
        raise NotImplementedError("No trained model available.")