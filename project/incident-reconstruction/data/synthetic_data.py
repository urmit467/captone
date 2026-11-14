"""Synthetic dataset generator for demonstration and testing.

Generates a small connected path network (P1 -> P2 -> P3 -> P4 -> P5 -> P6
with P4 permanently unobservable), anonymous people, camera observations and
one incident. Ground truth (who was actually on P4 at incident time) is
stored separately in the ``ground_truth`` collection and is NEVER visible to
the reconstruction algorithm.
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime, time, timedelta
from typing import Optional

from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.travel_time import TravelTimeModel
from utils.helpers import isoformat

logger = logging.getLogger(__name__)

DEFAULT_PATHS = [
    ("P1", 90.0),
    ("P2", 110.0),
    ("P3", 80.0),
    ("P4", 70.0),   # permanently unobservable
    ("P5", 100.0),
    ("P6", 90.0),
]
BASE_WALKING_SPEED = 84.0  # meters per minute (~1.4 m/s)


class SyntheticDataGenerator:
    def __init__(
        self,
        db_ops: Optional[DBOperations] = None,
        path_manager: Optional[PathManager] = None,
    ) -> None:
        self.db = db_ops or DBOperations()
        self.path_manager = path_manager or PathManager(db_ops=self.db)
        if not self.path_manager.is_loaded():
            self.path_manager.load_from_db()
        self.travel_model = TravelTimeModel(
            db_ops=self.db, path_manager=self.path_manager
        )

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #
    def clear_demo_data(self) -> None:
        """Remove all demo data so the generator is idempotent."""
        for coll in ["ground_truth", "predictions", "incidents",
                     "trajectories", "observations", "cameras", "paths",
                     "persons"]:
            self.db._collection(coll).delete_many({})  # type: ignore[attr-defined]
        self.path_manager.graph.clear()
        self.path_manager._loaded = False
        logger.info("Cleared demo data")

    def generate_demo(
        self,
        num_people: int = 20,
        incident_time_str: str = "10:30",
        demo_date=None,
        seed: int = 42,
        force_affected: int = 3,
    ) -> dict:
        """Generate a complete demo dataset and register one incident.

        Returns a summary dict including ground-truth affected people.
        """
        rng = random.Random(seed)
        self.clear_demo_data()
        incident_day = demo_date or date.today()
        incident_time = datetime.combine(
            incident_day, time.fromisoformat(incident_time_str)
        )

        # ----- paths ------------------------------------------------------
        if not self.path_manager.is_loaded():
            self.path_manager.load_from_db()
        for pid, length in DEFAULT_PATHS:
            observable = pid != "P4"
            self.path_manager.add_path(
                path_id=pid,
                name=pid,
                length=length,
                average_speed=BASE_WALKING_SPEED,
                is_observable=observable,
            )
        for a, b in [("P1", "P2"), ("P2", "P3"), ("P3", "P4"),
                     ("P4", "P5"), ("P5", "P6")]:
            self.path_manager.connect_paths(a, b)

        # ----- cameras ----------------------------------------------------
        camera_ids = []
        for pid in ["P1", "P2", "P3", "P5", "P6"]:
            cid = f"CAM_{pid}"
            self.db.insert_camera(
                cid, f"Camera {pid}", pid, location=f"Entrance of {pid}"
            )
            self.path_manager.assign_camera(pid, cid)
            camera_ids.append(cid)

        # ----- people + movement ------------------------------------------
        persons = []
        start_of_day = datetime.combine(incident_day, time(9, 0))
        affected_ground_truth = []

        for i in range(1, num_people + 1):
            person_id = f"Person_{i:03d}"
            self.db.insert_person(person_id, {"demo": True})
            persons.append(person_id)

            speed_factor = rng.uniform(0.8, 1.3)
            start_path = rng.choice(["P1", "P2", "P3"])
            start_time = start_of_day + timedelta(minutes=rng.uniform(0, 70))

            path_order = ["P1", "P2", "P3", "P4", "P5", "P6"]
            start_idx = path_order.index(start_path)
            num_paths = rng.randint(1, len(path_order) - start_idx)
            sequence = path_order[start_idx:start_idx + num_paths]

            t = start_time
            path_presence = {}
            for pid in sequence:
                length = self.path_manager.get_path(pid)["length"]
                person_speed = BASE_WALKING_SPEED * speed_factor
                traverse = length / person_speed
                path_presence[pid] = (t, t + timedelta(minutes=traverse))
                t += timedelta(minutes=traverse)

            # Camera observations while on observable paths.
            for pid, (entered, exited) in path_presence.items():
                if pid == "P4":
                    continue  # unobservable: no camera
                span_min = (exited - entered).total_seconds() / 60.0
                if span_min <= 0:
                    continue
                for _ in range(rng.randint(1, 3)):
                    obs_time = entered + timedelta(
                        minutes=rng.uniform(0, span_min)
                    )
                    self.db.insert_observation(
                        person_id=person_id,
                        camera_id=f"CAM_{pid}",
                        path_id=pid,
                        timestamp=obs_time,
                        confidence=rng.uniform(0.85, 0.99),
                        source="camera",
                    )

            # Ground truth for the incident on P4.
            if "P4" in path_presence:
                p_enter, p_exit = path_presence["P4"]
                present = p_enter <= incident_time <= p_exit
            else:
                present = False
            affected_ground_truth.append(person_id) if present else None
            self.db.insert_ground_truth(
                incident_id="INC_001",
                person_id=person_id,
                present_on_affected_path=present,
                actual_path_sequence=sequence,
                actual_entry_time=path_presence.get("P4", (None, None))[0],
                actual_exit_time=path_presence.get("P4", (None, None))[1],
            )

        # Guarantee at least ``force_affected`` people are on P4 at the
        # incident by rebuilding their schedules around it.
        forced = (
            affected_ground_truth[:force_affected]
            if len(affected_ground_truth) >= force_affected
            else persons[:force_affected]
        )
        self._force_affected(forced, incident_time, seed + 1)

        # Recompute ground-truth affected list after forcing.
        affected_final = [
            g["person_id"]
            for g in self.db.list_ground_truth("INC_001")
            if g.get("present_on_affected_path")
        ]

        # ----- incident ---------------------------------------------------
        self.db.insert_incident(
            incident_id="INC_001",
            affected_path_id="P4",
            incident_time=incident_time,
            severity="HIGH",
            description=(
                "Demo incident on unobservable path P4 at 10:30."
            ),
        )
        self.path_manager.refresh()

        summary = {
            "paths": 6,
            "unobservable_paths": ["P4"],
            "cameras": camera_ids,
            "persons": len(persons),
            "incident_id": "INC_001",
            "incident_time": isoformat(incident_time),
            "affected_path": "P4",
            "ground_truth_affected": sorted(affected_final),
        }
        logger.info("Demo dataset generated: %s", summary)
        return summary

    def _force_affected(self, person_ids: list[str], incident_time: datetime,
                        seed: int) -> None:
        """Adjust schedules so each listed person is on P4 at incident time."""
        rng = random.Random(seed)
        for person_id in person_ids:
            enter = incident_time - timedelta(minutes=rng.uniform(1, 8))
            exit_ = incident_time + timedelta(minutes=rng.uniform(1, 8))
            sequence = ["P3", "P4", "P5"]

            self.db.insert_observation(
                person_id=person_id, camera_id="CAM_P3", path_id="P3",
                timestamp=enter - timedelta(minutes=3),
                confidence=0.95, source="camera",
            )
            self.db.insert_observation(
                person_id=person_id, camera_id="CAM_P5", path_id="P5",
                timestamp=exit_ + timedelta(minutes=3),
                confidence=0.95, source="camera",
            )

            coll = self.db._collection("ground_truth")
            coll.update_one(
                {"incident_id": "INC_001", "person_id": person_id},
                {"$set": {
                    "present_on_affected_path": True,
                    "actual_path_sequence": sequence,
                    "actual_entry_time": enter,
                    "actual_exit_time": exit_,
                }},
                upsert=True,
            )