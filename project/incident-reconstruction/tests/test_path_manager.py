"""Unit tests for the path manager (NetworkX + MongoDB).

Skipped when MongoDB is not running.
"""

from datetime import datetime, timezone

import pytest

from database.connection import ping
from database.db_operations import DBOperations
from modules.path_manager import PathManager


def _db():
    if not ping():
        pytest.skip("MongoDB not running")
    return DBOperations()


def _cleanup(db, *path_ids):
    """Remove test documents so tests are idempotent across runs."""
    for pid in path_ids:
        try:
            db.delete_path(pid)
        except Exception:
            pass


def test_add_and_connect_paths():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, "T_P1", "T_P2")
    assert pm.add_path("T_P1", "Path 1", 100.0, 84.0)
    assert pm.add_path("T_P2", "Path 2", 100.0, 84.0)
    assert pm.connect_paths("T_P1", "T_P2")
    assert pm.is_connected("T_P1", "T_P2")
    assert pm.shortest_path("T_P1", "T_P2") == ["T_P1", "T_P2"]
    assert "T_P2" in pm.neighbors("T_P1")

    # Duplicate connect is a no-op.
    assert pm.connect_paths("T_P1", "T_P2") is False

    pm.disconnect_paths("T_P1", "T_P2")
    assert not pm.is_connected("T_P1", "T_P2")


def test_observable_flags():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, "T_OBS", "T_UNOBS")
    pm.add_path("T_OBS", "Observable", 50.0, 84.0, is_observable=True)
    pm.add_path("T_UNOBS", "Unobservable", 50.0, 84.0, is_observable=False)
    assert len(pm.observable_paths()) >= 1
    assert len(pm.unobservable_paths()) >= 1


def test_delete_path_removes_connections():
    db = _db()
    pm = PathManager(db_ops=db)
    _cleanup(db, "T_D1", "T_D2")
    pm.add_path("T_D1", "D1", 50.0, 84.0)
    pm.add_path("T_D2", "D2", 50.0, 84.0)
    pm.connect_paths("T_D1", "T_D2")
    pm.delete_path("T_D1")
    assert pm.path_exists("T_D1") is False
    assert not pm.is_connected("T_D1", "T_D2")


def test_unknown_operations_safe():
    db = _db()
    pm = PathManager(db_ops=db)
    assert pm.shortest_path("T_P1", "DOES_NOT_EXIST") is None
    assert pm.has_route("T_P1", "DOES_NOT_EXIST") is False