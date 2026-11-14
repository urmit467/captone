"""OpenCV video processing: YOLO + ByteTrack per frame, observations to MongoDB.

Every observation stores: person_id (None until manually associated), camera_id,
path_id, timestamp (UTC), bounding box, tracking_id (ByteTrack id local to the
camera), confidence, source='camera'.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Callable, Optional

import cv2

from database.db_operations import DBOperations
from modules.person_detector import PersonDetector
from modules.person_reid import PersonReID
from modules.person_tracker import PersonTracker
from utils.config import UPLOAD_DIR, PROCESSED_DIR
from utils.helpers import isoformat, next_person_id, utc_now

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Processes a CCTV video frame by frame and archives observations."""

    def __init__(
        self,
        db_ops: Optional[DBOperations] = None,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[PersonTracker] = None,
    ) -> None:
        self.db = db_ops or DBOperations()
        self.detector = detector or PersonDetector()
        self.tracker = tracker or PersonTracker()
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        self.crops_dir = os.path.join(PROCESSED_DIR, "crops")
        os.makedirs(self.crops_dir, exist_ok=True)
        # Automatic person re-identification across cameras — no manual
        # registration required.
        self.reid = PersonReID(self.db)

    # ------------------------------------------------------------------ #
    # Processing                                                         #
    # ------------------------------------------------------------------ #
    def process_video(
        self,
        video_path: str,
        camera_id: str,
        path_id: str,
        start_time=None,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """Process a video and store observations.

        ``progress_callback(current_frame, total_frames)`` may be provided by
        the dashboard to show a progress bar.
        """
        start_time = start_time or utc_now()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer = None
        if output_path:
            # Prefer H.264/avc1 (browser-playable) with FMP4 fallback,
            # since OpenCV's default mp4v/FMP4 isn't decodable by browsers.
            writer = None
            for fourcc_name in ("avc1", "mp4v"):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    trial = cv2.VideoWriter(
                        output_path, fourcc, fps,
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                    )
                    if trial.isOpened():
                        writer = trial
                        logger.info("VideoWriter using codec %s", fourcc_name)
                        break
                    trial.release()
                except Exception:
                    continue
            if writer is None:
                raise RuntimeError("Could not open VideoWriter with any codec")

        observations: list[dict] = []
        frame_idx = 0
        processed = 0

        # Track person IDs created during this run so new detections can match
        # against them without re-querying the DB every frame.
        known_persons = {
            p["person_id"]: p for p in self.db.list_persons()
        }

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                if max_frames is not None and processed >= max_frames:
                    break

                timestamp = start_time + timedelta(seconds=frame_idx / fps)

                detections = self.tracker.track(frame)
                h, w = frame.shape[:2]

                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    # Clip the bounding box to the frame bounds and crop the person.
                    x1c = max(0, min(int(x1), w))
                    y1c = max(0, min(int(y1), h))
                    x2c = max(0, min(int(x2), w))
                    y2c = max(0, min(int(y2), h))
                    image_path = None
                    if x2c > x1c and y2c > y1c:
                        crop_name = f"{camera_id}_{int(det['tracking_id'])}_{frame_idx:06d}.jpg"
                        image_path = os.path.join(self.crops_dir, crop_name)
                        try:
                            cv2.imwrite(image_path, frame[y1c:y2c, x1c:x2c])
                        except Exception as exc:  # pragma: no cover - disk write
                            logger.warning("Failed to save person crop: %s", exc)
                            image_path = None

                    # Automatic person matching across cameras via ReID: match
                    # this detection to an existing person by appearance, or
                    # create a new person if it has never been seen before.
                    person_id = None
                    if image_path:
                        person_id = self.reid.match(frame[y1c:y2c, x1c:x2c])
                    if person_id is None or person_id not in known_persons:
                        # No match (or match data stale): create a new person.
                        try:
                            person_id = next_person_id(list(known_persons.keys()))
                            self.db.insert_person(person_id)
                            known_persons[person_id] = {"person_id": person_id}
                            if image_path:
                                self.reid.save_features(person_id, frame[y1c:y2c, x1c:x2c])
                        except Exception as exc:  # pragma: no cover - db write
                            logger.warning("Failed to register person: %s", exc)
                            person_id = None

                    observations.append(
                        {
                            "person_id": person_id,
                            "camera_id": camera_id,
                            "path_id": path_id,
                            "timestamp": timestamp,
                            "bounding_box": {
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            },
                            "tracking_id": str(det["tracking_id"]),
                            "confidence": det["confidence"],
                            "source": "camera",
                            "image_path": image_path,
                        }
                    )

                if writer is not None:
                    frame = self._draw_detections(frame, detections)
                    writer.write(frame)

                processed += 1
                frame_idx += 1
                if progress_callback:
                    progress_callback(frame_idx, total_frames or frame_idx)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        saved = 0
        for obs in observations:
            try:
                self.db.insert_observation(**obs)
                saved += 1
            except Exception as exc:  # pragma: no cover - db write
                logger.warning("Failed to save observation: %s", exc)

        summary = {
            "video_path": video_path,
            "camera_id": camera_id,
            "path_id": path_id,
            "fps": fps,
            "frames_processed": processed,
            "total_frames": total_frames,
            "detections": len(observations),
            "observations_saved": saved,
            "start_time": isoformat(start_time),
        }
        logger.info("Processed video: %s", summary)
        return summary

    # ------------------------------------------------------------------ #
    # Drawing helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_detections(frame, detections: list[dict]):
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            tid = det["tracking_id"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{tid} {conf:.2f}"
            cv2.putText(
                frame, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return frame