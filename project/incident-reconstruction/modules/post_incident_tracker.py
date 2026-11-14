"""Post-incident movement reconstruction.

Analyzes where potentially affected people were seen after an incident and
distinguishes confirmed (camera-observed) from inferred movement.
"""

from __future__ import annotations

import logging
from typing import Optional

from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.timeline_generator import TimelineGenerator
from utils.helpers import parse_timestamp

logger = logging.getLogger(__name__)


class PostIncidentTracker:
    """Reconstructs a person's movement after an incident timestamp."""

    def __init__(
        self,
        db_ops: Optional[DBOperations] = None,
        path_manager: Optional[PathManager] = None,
        timeline_generator: Optional[TimelineGenerator] = None,
    ) -> None:
        self.db = db_ops or DBOperations()
        self.path_manager = path_manager or PathManager(db_ops=self.db)
        if not self.path_manager.is_loaded():
            self.path_manager.load_from_db()
        self.timeline_generator = timeline_generator or TimelineGenerator(
            db_ops=self.db, path_manager=self.path_manager
        )

    def analyze(self, person_id: str, incident_time) -> dict:
        incident_time = parse_timestamp(incident_time)
        timeline = self.timeline_generator.timeline(person_id)
        events = timeline["events"]

        after = [e for e in events if e["timestamp"] and e["timestamp"] > incident_time]
        before = [e for e in events if e["timestamp"] and e["timestamp"] < incident_time]

        if not after:
            status = "No post-incident observations."
            last_path = before[-1]["path_id"] if before else None
            return {
                "person_id": person_id,
                "post_incident_status": status,
                "last_observed_location": last_path,
                "post_incident_trajectory": [],
                "first_post_incident_observation": None,
                "path_history_after": [],
                "confirmed": False,
            }

        first_post = after[0]
        trajectory = [
            e for e in timeline["reconstructed_trajectory"]
            if e["timestamp"] and e["timestamp"] >= incident_time
        ]
        confirmed_steps = [e for e in trajectory if e["source"] == "camera"]
        inferred_steps = [e for e in trajectory if e["source"] == "inferred"]

        return {
            "person_id": person_id,
            "post_incident_status": (
                f"Observed on {first_post['path_id']} at {first_post['timestamp_iso']}"
            ),
            "last_observed_location": trajectory[-1]["path_id"] if trajectory else None,
            "post_incident_trajectory": trajectory,
            "first_post_incident_observation": {
                "path_id": first_post["path_id"],
                "camera_id": first_post["camera_id"],
                "timestamp": first_post["timestamp_iso"],
            },
            "path_history_after": [e["path_id"] for e in trajectory],
            "confirmed": bool(confirmed_steps),
            "confirmed_steps": confirmed_steps,
            "inferred_steps": inferred_steps,
        }