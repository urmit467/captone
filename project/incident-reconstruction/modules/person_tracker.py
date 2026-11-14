"""Person tracking with ByteTrack via Ultralytics YOLO.

Keeps tracker state across repeated :meth:`track` calls (``persist=True``) so
the same person keeps the same tracking ID throughout a video.
Note: ByteTrack identities are local to one camera; cross-camera identity is
established manually (see dashboard identity association).
"""

from __future__ import annotations

import logging
from typing import Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PersonTracker:
    """Tracks people across frames with ByteTrack."""

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
            logger.info("ByteTrack tracker ready (model=%s, device=%s)", model_name, device)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load YOLO model %s: %s", model_name, exc)
            raise

    def track(self, frame) -> list[dict]:
        """Run detection + ByteTrack on one frame.

        Returns [{bbox, confidence, tracking_id}] where tracking_id is the
        ByteTrack id local to this camera feed.
        """
        try:
            results = self.model.track(
                frame,
                conf=self.conf,
                imgsz=self.img_size,
                device=self.device,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )
        except Exception as exc:
            logger.error("ByteTrack tracking failed: %s", exc)
            return []

        detections: list[dict] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.id is None:
                continue
            for box, tid in zip(boxes, boxes.id):
                cls = int(box.cls[0])
                if cls != self.PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "tracking_id": int(tid),
                    }
                )
        logger.debug("Tracked %d people", len(detections))
        return detections