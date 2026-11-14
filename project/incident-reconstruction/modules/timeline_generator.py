"""Person movement timeline generation.

Builds a per-person timeline from stored observations, distinguishes actual
camera observations from inferred path transitions, and reconstructs the
trajectory through the path network.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.travel_time import TravelTimeModel
from utils.helpers import parse_timestamp

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _sort_key(event: dict) -> datetime:
    ts = event.get("timestamp")
    return ts if isinstance(ts, datetime) else EPOCH


class TimelineGenerator:
    """Generates a movement timeline for a single person."""

    def __init__(self, db_ops: Optional[DBOperations] = None,
                 path_manager: Optional[PathManager] = None,
                 travel_model: Optional[TravelTimeModel] = None) -> None:
        self.db = db_ops or DBOperations()
        self.path_manager = path_manager or PathManager(db_ops=self.db)
        if not self.path_manager.is_loaded():
            self.path_manager.load_from_db()
        self.travel_model = travel_model or TravelTimeModel(
            db_ops=self.db, path_manager=self.path_manager
        )

    def observations(self, person_id: str) -> list[dict]:
        return self.db.observations_for_person(person_id)

    def timeline(self, person_id: str) -> dict:
        raw = self.db.observations_for_person(person_id)
        events = []
        for o in raw:
            ts = parse_timestamp(o.get("timestamp"))
            events.append({
                "timestamp": ts,
                "timestamp_iso": o.get("timestamp"),
                "path_id": o.get("path_id"),
                "camera_id": o.get("camera_id"),
                "tracking_id": o.get("tracking_id"),
                "confidence": o.get("confidence"),
                "source": o.get("source", "camera"),
                "bounding_box": o.get("bounding_box") or {},
            })
        events.sort(key=_sort_key)

        if not events:
            return {
                "person_id": person_id,
                "events": [],
                "first_observed": None,
                "last_observed": None,
                "reconstructed_trajectory": [],
                "path_history": [],
                "inferred_transitions": [],
                "event_count": 0,
                "observed_count": 0,
                "inferred_count": 0,
            }

        first = events[0]
        last = events[-1]
        trajectory = self._reconstruct_trajectory(events)
        inferred = [e for e in trajectory if e.get("source") == "inferred"]
        path_history = [e["path_id"] for e in trajectory if e.get("path_id")]

        return {
            "person_id": person_id,
            "events": events,
            "first_observed": first.get("timestamp_iso"),
            "last_observed": last.get("timestamp_iso"),
            "first_observed_dt": first.get("timestamp"),
            "last_observed_dt": last.get("timestamp"),
            "reconstructed_trajectory": trajectory,
            "path_history": path_history,
            "inferred_transitions": inferred,
            "event_count": len(events),
            "observed_count": sum(1 for e in events if e["source"] == "camera"),
            "inferred_count": sum(1 for e in events if e["source"] == "inferred"),
        }

    def _reconstruct_trajectory(self, events: list[dict]) -> list[dict]:
        """Turn sorted observations into a path sequence, inferring transitions
        across unobservable paths using the shortest route between consecutive
        observed paths."""
        if not events:
            return []
        trajectory = [dict(events[0], source=events[0].get("source", "camera"))]
        for prev, curr in zip(events, events[1:]):
            src_path = prev.get("path_id")
            dst_path = curr.get("path_id")
            if src_path == dst_path or not src_path or not dst_path:
                trajectory.append(dict(curr, source=curr.get("source", "camera")))
                continue
            route = self.path_manager.shortest_path(src_path, dst_path)
            if route and len(route) > 2:
                # Intermediate paths were not directly observed: infer them.
                for intermediate in route[1:-1]:
                    trajectory.append({
                        "timestamp": curr["timestamp"],
                        "timestamp_iso": curr.get("timestamp_iso"),
                        "path_id": intermediate,
                        "camera_id": None,
                        "tracking_id": None,
                        "confidence": None,
                        "source": "inferred",
                        "bounding_box": {},
                    })
            trajectory.append(dict(curr, source=curr.get("source", "camera")))
        return trajectory