"""Unit tests for the reconstruction engine.

Skipped when MongoDB is not running.
"""

import pytest

from database.connection import ping
from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.reconstruction_engine import ReconstructionEngine
from utils.helpers import parse_timestamp


def _db():
    if not ping():
        pytest.skip("MongoDB not running")
    return DBOperations()


def _setup_person(db, pm, person_id, observations):
    db.insert_person(person_id)
    for path_id, ts_str in observations:
        db.insert_observation(
            person_id=person_id,
            camera_id=f"CAM_{path_id}",
            path_id=path_id,
            timestamp=parse_timestamp(ts_str),
            confidence=0.9,
            source="camera",
        )


def _cleanup(db, path_ids: list[str], person_ids: list[str]) -> None:
    for pid in path_ids:
        try:
            db.delete_path(pid)
        except Exception:
            pass
    for p in person_ids:
        try:
            db.delete_person(p)
        except Exception:
            pass


def test_unreachable_path_gives_zero_probability():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, ["R_A", "R_B"], ["R_PERSON"])
    pm.add_path("R_A", "A", 50.0, 84.0)
    pm.add_path("R_B", "B", 50.0, 84.0)  # disconnected from A
    pm.refresh()
    engine = ReconstructionEngine(db_ops=db, path_manager=pm)

    _setup_person(db, pm, "R_PERSON", [("R_A", "2026-11-07T10:20:00")])
    res = engine.compute_person_probability(
        "R_PERSON", "R_B", parse_timestamp("2026-11-07T10:30:00")
    )
    assert res["probability"] == 0.0
    assert res["possible"] is False


def test_same_path_before_incident_gives_high_probability():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, ["R_X"], ["R_PERSON2"])
    pm.add_path("R_X", "X", 70.0, 84.0)  # ~50 seconds to cross
    pm.refresh()
    engine = ReconstructionEngine(db_ops=db, path_manager=pm)

    seen_at = "2026-11-07T10:28:00"  # 2 minutes before incident, on the path
    _setup_person(db, pm, "R_PERSON2", [("R_X", seen_at)])
    res = engine.compute_person_probability(
        "R_PERSON2", "R_X", parse_timestamp("2026-11-07T10:30:00")
    )
    assert res["possible"] is True
    assert res["probability"] > 50, f"Expected high probability, got {res['probability']}"


def test_no_observations_gives_zero():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, ["R_Z"], ["R_NOPERSON"])
    pm.add_path("R_Z", "Z", 50.0, 84.0)
    pm.refresh()
    engine = ReconstructionEngine(db_ops=db, path_manager=pm)
    db.insert_person("R_NOPERSON")
    res = engine.compute_person_probability(
        "R_NOPERSON", "R_Z", parse_timestamp("2026-11-07T10:30:00")
    )
    assert res["probability"] == 0.0