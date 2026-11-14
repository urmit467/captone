"""Person detection with Ultralytics YOLO (COCO class 0 = person).

CPU inference by default; works on a normal Windows PC without a GPU.
"""

from __future__ import annotations

import logging
from typing import Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PersonDetector:
    """Detects people in a single frame using YOLO."""

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.25,
        img_size: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self.img_size = img_size
        self.device = device
        try:
            self.model = YOLO(model_name)
            logger.info("Loaded YOLO model %s (device=%s)", model_name, device)
        except Exception as exc:  # pragma: no cover - model load/Download
            logger.error("Failed to load YOLO model %s: %s", model_name, exc)
            raise

    def detect(self, frame) -> list[dict]:
        """Return person detections: [{bbox, confidence}] in absolute pixels."""
        try:
            results = self.model.predict(
                frame,
                conf=self.conf,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        detections: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != self.PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    }
                )
        logger.debug("Detected %d people", len(detections))
        return detections