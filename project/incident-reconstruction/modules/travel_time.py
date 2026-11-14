"""Travel-time estimation over the path network.

Travel time for a path = length / average walking speed. For each path we keep
a minimum / expected / maximum plausible range so the reconstruction engine can
reason about uncertainty instead of assuming instant movement.
"""

from __future__ import annotations

import logging
from typing import Optional

from database.db_operations import DBOperations
from modules.path_manager import PathManager
from utils.config import TRAVEL_MAX_FACTOR, TRAVEL_MIN_FACTOR

logger = logging.getLogger(__name__)


class TravelTimeModel:
    """Estimates travel times across the path network."""

    def __init__(self, db_ops: Optional[DBOperations] = None,
                 path_manager: Optional[PathManager] = None) -> None:
        self.db = db_ops or DBOperations()
        self.path_manager = path_manager or PathManager(db_ops=self.db)
        if not self.path_manager.is_loaded():
            self.path_manager.load_from_db()

    # -- per-path times ------------------------------------------------- #
    def path_stats(self, path_id: str) -> Optional[dict]:
        path = self.path_manager.get_path(path_id)
        if not path:
            return None
        length = float(path.get("length", 0.0) or 0.0)
        speed = float(path.get("average_speed", 0.0) or 0.0)
        if length <= 0 or speed <= 0:
            return None
        expected = length / speed  # minutes if length in meters and speed in m/min
        return {
            "path_id": path_id,
            "name": path.get("name", path_id),
            "length": length,
            "average_speed": speed,
            "expected_minutes": expected,
            "min_minutes": expected * TRAVEL_MIN_FACTOR,
            "max_minutes": expected * TRAVEL_MAX_FACTOR,
        }

    def path_expected_minutes(self, path_id: str) -> Optional[float]:
        stats = self.path_stats(path_id)
        return stats["expected_minutes"] if stats else None

    def path_min_max(self, path_id: str) -> Optional[tuple]:
        stats = self.path_stats(path_id)
        if not stats:
            return None
        return stats["min_minutes"], stats["max_minutes"]

    # -- route times ---------------------------------------------------- #
    def route_times(self, route: list[str]) -> Optional[dict]:
        """Min/expected/max minutes to traverse every path in ``route`` in order.

        To be present on path at index i, a traveler has already traversed the
        paths at indices 0..i-1 and is currently traversing path i.
        """
        if not route:
            return None
        legs = []
        total_min = total_max = total_expected = 0.0
        cumulative = {"min": 0.0, "expected": 0.0, "max": 0.0}
        for pid in route:
            stats = self.path_stats(pid)
            if not stats:
                return None
            legs.append({
                "path_id": pid,
                "min_minutes": stats["min_minutes"],
                "expected_minutes": stats["expected_minutes"],
                "max_minutes": stats["max_minutes"],
            })
            total_min += stats["min_minutes"]
            total_expected += stats["expected_minutes"]
            total_max += stats["max_minutes"]
            cumulative = {
                "min": cumulative["min"] + stats["min_minutes"],
                "expected": cumulative["expected"] + stats["expected_minutes"],
                "max": cumulative["max"] + stats["max_minutes"],
            }
        return {
            "route": route,
            "legs": legs,
            "total_min_minutes": total_min,
            "total_expected_minutes": total_expected,
            "total_max_minutes": total_max,
            "arrival_at_end_cumulative": cumulative,
        }

    def arrival_window_to_index(self, route: list[str], index: int,
                                start_minutes: float = 0.0) -> Optional[dict]:
        """Earliest/latest arrival at route[index] (relative to start_minutes)."""
        if not route or index < 0 or index >= len(route):
            return None
        cum_min = cum_max = 0.0
        for pid in route[:index]:
            stats = self.path_stats(pid)
            if not stats:
                return None
            cum_min += stats["min_minutes"]
            cum_max += stats["max_minutes"]
        return {
            "path_id": route[index],
            "arrival_min_minutes": start_minutes + cum_min,
            "arrival_max_minutes": start_minutes + cum_max,
            "traverse_min_minutes": None,
            "traverse_expected_minutes": None,
            "traverse_max_minutes": None,
        }

    def presence_window_on_path(self, route: list[str], path_id: str,
                                start_minutes: float = 0.0) -> Optional[dict]:
        """[earliest_presence, latest_presence] window on ``path_id`` along route.

        A traveler present on path_id entered sometime in the arrival window and
        leaves after traversing path_id.
        """
        try:
            idx = route.index(path_id)
        except ValueError:
            return None
        arrival = self.arrival_window_to_index(route, index=idx, start_minutes=start_minutes)
        if not arrival:
            return None
        stats = self.path_stats(path_id)
        if not stats:
            return None
        return {
            "path_id": path_id,
            "index": idx,
            "earliest_presence_minutes": arrival["arrival_min_minutes"],
            "latest_presence_minutes": (
                arrival["arrival_max_minutes"] + stats["max_minutes"]
            ),
            "traverse_min_minutes": stats["min_minutes"],
            "traverse_expected_minutes": stats["expected_minutes"],
            "traverse_max_minutes": stats["max_minutes"],
        }