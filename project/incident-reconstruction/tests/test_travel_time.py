"""Unit tests for the travel-time model.

Skipped when MongoDB is not running.
"""

import pytest

from database.connection import ping
from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.travel_time import TravelTimeModel


def _db():
    if not ping():
        pytest.skip("MongoDB not running")
    return DBOperations()


def _cleanup(db, *path_ids):
    for pid in path_ids:
        try:
            db.delete_path(pid)
        except Exception:
            pass


def test_path_stats_and_route_times():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, "TT_A", "TT_B")
    pm.add_path("TT_A", "A", 100.0, 100.0)  # 1 minute expected
    pm.add_path("TT_B", "B", 200.0, 100.0)  # 2 minutes expected
    pm.connect_paths("TT_A", "TT_B")

    model = TravelTimeModel(db_ops=db, path_manager=pm)
    stats = model.path_stats("TT_A")
    assert stats is not None
    assert abs(stats["expected_minutes"] - 1.0) < 1e-9
    assert stats["min_minutes"] < stats["expected_minutes"] < stats["max_minutes"]

    route = model.route_times(["TT_A", "TT_B"])
    assert route is not None
    assert abs(route["total_expected_minutes"] - 3.0) < 1e-9

    pw = model.presence_window_on_path(["TT_A", "TT_B"], "TT_B", start_minutes=10.0)
    assert pw is not None
    assert pw["earliest_presence_minutes"] >= 10.0