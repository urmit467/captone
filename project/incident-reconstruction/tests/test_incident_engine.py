"""Incident end-to-end smoke test (reconstruction + classification).

Skipped when MongoDB is not running.
"""

import pytest

from database.connection import ping
from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.reconstruction_engine import ReconstructionEngine
from modules.threat_engine import ThreatEngine
from utils.helpers import parse_timestamp


def _db():
    if not ping():
        pytest.skip("MongoDB not running")
    return DBOperations()


def _cleanup(db, path_id: str, person_ids: list[str]) -> None:
    """Remove test documents so tests are idempotent across runs."""
    try:
        db.delete_one("incidents", {"incident_id": "I_INC"})
    except Exception:
        pass
    try:
        db.delete_predictions("I_INC")
    except Exception:
        pass
    try:
        db.delete_path(path_id)
    except Exception:
        pass
    for pid in person_ids:
        try:
            db.delete_person(pid)
        except Exception:
            pass
    # Remove any leftover test persons from prior runs so that
    # reconstruct_for_incident() (which iterates list_persons()) only sees the
    # persons this test inserts.
    for p in db.list_persons():
        try:
            db.delete_person(p["person_id"])
        except Exception:
            pass


def test_incident_classification_runs():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, "I_P4", ["I_ONE", "I_TWO"])
    pm.add_path("I_P4", "P4", 70.0, 84.0, is_observable=False)
    pm.refresh()

    db.insert_person("I_ONE")
    db.insert_person("I_TWO")
    db.insert_observation("I_ONE", "CAM_I_P4", "I_P4",
                          parse_timestamp("2026-11-07T10:28:00"),
                          confidence=0.9, source="camera")
    db.insert_incident("I_INC", "I_P4",
                       parse_timestamp("2026-11-07T10:30:00"), "HIGH", "test")

    engine = ReconstructionEngine(db_ops=db, path_manager=pm)
    threat = ThreatEngine()
    results = engine.reconstruct_for_incident(
        "I_INC", "I_P4", parse_timestamp("2026-11-07T10:30:00")
    )
    assert len(results) == 2
    bands = threat.classify_results(results)
    assert set(bands) == {"HIGH", "MEDIUM", "LOW"}

    saved = db.list_predictions("I_INC")
    assert len(saved) == 2
    for s in saved:
        assert "risk_level" in s and "probability" in s
