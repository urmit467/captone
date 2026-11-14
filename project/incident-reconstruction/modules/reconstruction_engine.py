"""Hidden-path reconstruction engine (core research module).

Answers: given a person's CCTV observations before and after an incident, the
connected path network and a travel-time model, what is the probability the
person was inside an unobservable affected path at the exact incident time?

The estimator is transparent and explainable: no random probabilities, no
fabricated labels. It combines:

* temporal overlap between the incident and a plausible presence window on the
  affected path (triangular membership),
* route coverage (how many feasible routes between the observations pass
  through the affected path),
* observation recency (closer observations => more confidence),
* the person's actual historical transitions through the path.

Scikit-learn is NOT used here; it is reserved for modules/training when real
labeled incidents exist.
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta
from typing import Optional

from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.timeline_generator import TimelineGenerator
from modules.travel_time import TravelTimeModel
from utils.config import (
    MAX_CROSS_BUFFER_MINUTES,
    RECENCY_SCALE_MINUTES,
    ROUTE_TOLERANCE_MINUTES,
    THRESHOLD_HIGH,
    THRESHOLD_MEDIUM,
    classify,
)
from utils.helpers import clamp, isoformat, parse_timestamp, to_minutes, utc_now

logger = logging.getLogger(__name__)


def _triangular(value, low, high, mode) -> float:
    """Triangular membership in [low, high] peaking at ``mode``; range [0, 1]."""
    if value is None or high <= low or value < low or value > high:
        return 0.0
    if value <= mode:
        return (value - low) / (mode - low) if mode > low else 1.0
    return (high - value) / (high - mode) if high > mode else 1.0


def _minutes_between(later, earlier) -> float:
    return to_minutes((later - earlier).total_seconds())  # type: ignore[arg-type]


class ReconstructionEngine:
    """Estimates incident-time presence on an unobservable path per person."""

    def __init__(
        self,
        db_ops: Optional[DBOperations] = None,
        path_manager: Optional[PathManager] = None,
        travel_model: Optional[TravelTimeModel] = None,
        timeline_generator: Optional[TimelineGenerator] = None,
    ) -> None:
        self.db = db_ops or DBOperations()
        self.path_manager = path_manager or PathManager(db_ops=self.db)
        if not self.path_manager.is_loaded():
            self.path_manager.load_from_db()
        self.travel_model = travel_model or TravelTimeModel(
            db_ops=self.db, path_manager=self.path_manager
        )
        self.timeline_generator = timeline_generator or TimelineGenerator(
            db_ops=self.db, path_manager=self.path_manager,
            travel_model=self.travel_model,
        )

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def reconstruct_for_incident(
        self,
        incident_id: str,
        affected_path_id: str,
        incident_time,
        save: bool = True,
    ) -> list[dict]:
        """Run reconstruction for every person and return ranked results."""
        incident_time = parse_timestamp(incident_time)
        affected = self.path_manager.get_path(affected_path_id)
        if not affected:
            raise ValueError(f"Unknown affected path: {affected_path_id}")

        results = []
        for person in self.db.list_persons():
            person_id = person["person_id"]
            result = self.compute_person_probability(
                person_id, affected_path_id, incident_time
            )
            result["incident_id"] = incident_id
            result["person_id"] = person_id
            result["risk_level"] = classify(
                result["probability"], THRESHOLD_HIGH, THRESHOLD_MEDIUM
            )
            result["explanation"] = self._build_explanation(result)
            results.append(result)

        results.sort(key=lambda r: r["probability"], reverse=True)

        if save:
            self.db.delete_predictions(incident_id)
            for r in results:
                self.db.insert_prediction(
                    incident_id=incident_id,
                    person_id=r["person_id"],
                    probability=r["probability"],
                    predicted_entry_time=r.get("predicted_entry_time"),
                    predicted_exit_time=r.get("predicted_exit_time"),
                    reconstructed_path=r.get("reconstructed_path"),
                    risk_level=r["risk_level"],
                    explanation=r["explanation"],
                    last_observed_location=r.get("last_observed_location"),
                    post_incident_status=r.get("post_incident_status"),
                )
        logger.info(
            "Reconstruction for %s on %s: %d people evaluated",
            incident_id, affected_path_id, len(results),
        )
        return results

    # ------------------------------------------------------------------ #
    # Single-person probability                                          #
    # ------------------------------------------------------------------ #
    def compute_person_probability(
        self, person_id: str, affected_path_id: str, incident_time
    ) -> dict:
        incident_time = parse_timestamp(incident_time)
        affected = self.path_manager.get_path(affected_path_id)
        affected_stats = self.travel_model.path_stats(affected_path_id)

        timeline = self.timeline_generator.timeline(person_id)
        events = timeline["events"]
        ts = parse_timestamp

        before = [e for e in events if e["timestamp"] and e["timestamp"] < incident_time]
        after = [e for e in events if e["timestamp"] and e["timestamp"] > incident_time]
        last = before[-1] if before else None      # most recent observation before
        first = after[0] if after else None        # earliest observation after

        # ------------------------------------------------------------------
        # No observations at all: cannot estimate anything.
        # ------------------------------------------------------------------
        if not last and not first:
            return self._empty_result(
                person_id, incident_time,
                reason="No camera observations for this person.",
            )

        last_path = last["path_id"] if last else None
        first_path = first["path_id"] if first else None

        # Route feasibility helpers.
        def route_minutes(route: list[str]) -> Optional[float]:
            rt = self.travel_model.route_times(route)
            return rt["total_expected_minutes"] if rt else None

        def arrival_to_affected(route: list[str], start_ts) -> Optional[dict]:
            """Earliest/latest arrival AT affected path (start being on it)."""
            try:
                idx = route.index(affected_path_id)
            except ValueError:
                return None
            if idx == 0:
                return {
                    "min_minutes": 0.0,
                    "max_minutes": 0.0,
                    "expected_minutes": 0.0,
                }
            prefix = route[:idx]  # paths traversed before stepping onto affected
            total_min = total_max = total_exp = 0.0
            for pid in prefix:
                st = self.travel_model.path_stats(pid)
                if not st:
                    return None
                total_min += st["min_minutes"]
                total_max += st["max_minutes"]
                total_exp += st["expected_minutes"]
            return {
                "min_minutes": total_min,
                "max_minutes": total_max,
                "expected_minutes": total_exp,
            }

        windows = []  # plausible presence windows on the affected path
        route_coverage = 1.0

        st = affected_stats
        max_cross = float(st["max_minutes"]) + MAX_CROSS_BUFFER_MINUTES if st else 15.0

        # ==================================================================
        # Case 1: last observation is ON the affected path.
        # ==================================================================
        if last and last_path == affected_path_id:
            t_last = last["timestamp"]
            win_start = t_last
            win_end = t_last + timedelta(minutes=max_cross)
            mode = t_last  # most likely still there right after being seen
            windows.append({
                "start": win_start, "end": win_end, "mode": mode,
                "entry_est": t_last,
                "exit_est": t_last + timedelta(minutes=float(st["expected_minutes"]) if st else max_cross / 2),
                "route": [affected_path_id],
                "kind": "observed_on_affected_before",
            })
            route_coverage = 1.0
            # If they are later seen elsewhere, they must have left the
            # affected path in time to reach that later path.
            if first and first_path != affected_path_id:
                route_to = self.path_manager.shortest_path(
                    affected_path_id, first_path
                )
                if route_to:
                    rt = self.travel_model.route_times(route_to)
                    # They must leave the affected path early enough to reach
                    # the later-observed path by its observation time, so the
                    # latest possible presence on the affected path is the
                    # later-observation time minus the travel time (not the
                    # last-observation time plus it).
                    travel_min = rt["total_min_minutes"] if rt else max_cross
                    latest_leave = first["timestamp"] - timedelta(minutes=travel_min)
                    for w in windows:
                        w["end"] = min(w["end"], latest_leave)
                    if latest_leave <= t_last:
                        return self._empty_result(
                            person_id, incident_time,
                            reason=(
                                f"Cannot travel from {affected_path_id} to {first_path} "
                                f"between being seen on {affected_path_id} "
                                f"({last['timestamp_iso']}) and being seen on {first_path} "
                                f"({first['timestamp_iso']})."
                            ),
                        )
                route_coverage = 1.0  # every route starts on the affected path
            base_gap = (
                _minutes_between(first["timestamp"], t_last)
                if first
                else _minutes_between(incident_time, t_last)
            )

        # ==================================================================
        # Case 2: first observation is ON the affected path.
        # ==================================================================
        elif first and first_path == affected_path_id:
            t_first = first["timestamp"]
            win_start = t_first - timedelta(minutes=max_cross)
            win_end = t_first
            mode = win_start  # entered as early as the window allows
            windows.append({
                "start": win_start, "end": win_end, "mode": mode,
                "entry_est": win_start,
                "exit_est": t_first,
                "route": [affected_path_id],
                "kind": "observed_on_affected_after",
            })
            route_coverage = 1.0
            if last and last_path != affected_path_id:
                route_to = self.path_manager.shortest_path(last_path, affected_path_id)
                if route_to:
                    arr = arrival_to_affected(route_to, last["timestamp"])
                    if arr:
                        entry_min = last["timestamp"] + timedelta(minutes=arr["min_minutes"])
                        entry_max = last["timestamp"] + timedelta(minutes=arr["max_minutes"])
                        eff_start = max(win_start, entry_min)
                        eff_end = min(win_end, entry_max, t_first)
                        if eff_start > eff_end:
                            return self._empty_result(
                                person_id, incident_time,
                                reason=(
                                    f"Cannot travel from {last_path} to {affected_path_id} "
                                    f"between the last observation ({last['timestamp_iso']}) "
                                    f"and being seen on {affected_path_id} "
                                    f"({first['timestamp_iso']})."
                                ),
                            )
                        for w in windows:
                            w["start"] = eff_start
                            w["end"] = eff_end
                            w["mode"] = eff_start
                        route_coverage = 1.0
                    else:
                        route_coverage = 0.0
                else:
                    route_coverage = 0.0
            base_gap = _minutes_between(t_first, last["timestamp"]) if last else _minutes_between(t_first, incident_time)

        # ==================================================================
        # Case 3: observations on both sides, neither on the affected path.
        # ==================================================================
        elif last and first:
            src = last_path
            dst = first_path
            gap_min = _minutes_between(first["timestamp"], last["timestamp"])
            routes = self.path_manager.all_simple_paths(src, dst)
            feasible = [
                r for r in routes
                if route_minutes(r) is not None
                and route_minutes(r) <= gap_min + ROUTE_TOLERANCE_MINUTES
            ]
            through = [r for r in feasible if affected_path_id in r]
            if not through:
                return self._empty_result(
                    person_id, incident_time,
                    reason=(
                        f"No physically feasible route from {src} to {dst} passes "
                        f"through {affected_path_id} in the time available "
                        f"({gap_min:.0f} min)."
                    ),
                )
            route_coverage = len(through) / max(len(feasible), 1)

            best_mu = 0.0
            best_window = None
            for route in through:
                arr = arrival_to_affected(route, last["timestamp"])
                if not arr:
                    continue
                enter_min = last["timestamp"] + timedelta(minutes=arr["min_minutes"])
                # Latest arrival at the affected path: they must traverse it
                # and reach the post-incident observation path by the first
                # observation time. They cannot still be on the affected path
                # once seen on the later path. We do *not* cap arrival by the
                # forward travel time, because the person may have lingered on
                # the last observed path before entering the affected path.
                st_min = float(st["min_minutes"]) if st else 0.0
                enter_max = first["timestamp"] - timedelta(minutes=st_min)
                if enter_max <= enter_min:
                    continue
                expected_enter = last["timestamp"] + timedelta(minutes=arr["expected_minutes"])
                travel = st if st else {"min_minutes": 0, "expected_minutes": max_cross, "max_minutes": max_cross}
                presence_start = enter_min
                presence_end = min(
                    enter_max + timedelta(minutes=float(travel["max_minutes"])),
                    first["timestamp"],
                )
                if presence_end <= presence_start:
                    continue
                mode = expected_enter + timedelta(minutes=float(travel["expected_minutes"]) / 2)
                if mode < presence_start:
                    mode = presence_start
                elif mode > presence_end:
                    mode = presence_end
                mu = _triangular(incident_time, presence_start, presence_end, mode)
                if mu > best_mu:
                    best_mu = mu
                    best_window = {
                        "start": presence_start, "end": presence_end, "mode": mode,
                        "entry_est": expected_enter,
                        "exit_est": expected_enter + timedelta(
                            minutes=float(travel["expected_minutes"])
                        ),
                        "route": route,
                        "kind": "via_route",
                    }
            if best_window is None:
                return self._empty_result(person_id, incident_time, reason="No route window computed.")
            windows = [best_window]
            base_gap = gap_min

        # ==================================================================
        # Case 4: only a before-incident observation.
        # ==================================================================
        elif last:
            t_last = last["timestamp"]
            if last_path == affected_path_id:
                win_start = t_last
                win_end = t_last + timedelta(minutes=max_cross)
                mode = t_last
                windows.append({
                    "start": win_start, "end": win_end, "mode": mode,
                    "entry_est": t_last,
                    "exit_est": t_last + timedelta(minutes=max_cross / 2),
                    "route": [affected_path_id],
                    "kind": "observed_on_affected_before",
                })
                route_coverage = 1.0
            else:
                route = self.path_manager.shortest_path(last_path, affected_path_id)
                if not route:
                    return self._empty_result(
                        person_id, incident_time,
                        reason=f"{affected_path_id} is not reachable from {last_path}.",
                    )
                arr = arrival_to_affected(route, t_last)
                if not arr:
                    return self._empty_result(person_id, incident_time, reason="Route window unavailable.")
                enter_min = t_last + timedelta(minutes=arr["min_minutes"])
                enter_max = t_last + timedelta(minutes=arr["max_minutes"])
                expected_enter = t_last + timedelta(minutes=arr["expected_minutes"])
                travel = st if st else {"min_minutes": 0, "expected_minutes": max_cross, "max_minutes": max_cross}
                presence_start = enter_min
                presence_end = enter_max + timedelta(minutes=float(travel["max_minutes"]))
                mode = expected_enter + timedelta(minutes=float(travel["expected_minutes"]) / 2)
                windows.append({
                    "start": presence_start, "end": presence_end, "mode": mode,
                    "entry_est": expected_enter,
                    "exit_est": expected_enter + timedelta(minutes=float(travel["expected_minutes"])),
                    "route": route,
                    "kind": "travel_to_affected",
                })
                route_coverage = 1.0
            base_gap = _minutes_between(incident_time, t_last)

        # ==================================================================
        # Case 5: only an after-incident observation.
        # ==================================================================
        else:
            t_first = first["timestamp"]
            if first_path == affected_path_id:
                win_start = t_first - timedelta(minutes=max_cross)
                win_end = t_first
                mode = win_start
                windows.append({
                    "start": win_start, "end": win_end, "mode": mode,
                    "entry_est": win_start,
                    "exit_est": t_first,
                    "route": [affected_path_id],
                    "kind": "observed_on_affected_after",
                })
                route_coverage = 1.0
            else:
                # No pre-incident anchor: only feasible if they could leave the
                # affected path after T and still reach the observed post path.
                route = self.path_manager.shortest_path(affected_path_id, first_path)
                if not route:
                    return self._empty_result(
                        person_id, incident_time,
                        reason=f"{first_path} is not reachable from {affected_path_id}.",
                    )
                rt = self.travel_model.route_times(route)
                need_minutes = float(rt["total_min_minutes"]) if rt else max_cross
                if _minutes_between(t_first, incident_time) < need_minutes:
                    return self._empty_result(
                        person_id, incident_time,
                        reason=(
                            f"Not enough time between the incident and the observation "
                            f"on {first_path} to have passed through {affected_path_id}."
                        ),
                    )
                # Weak evidence: no before-incident anchor => capped low probability.
                return {
                    "person_id": person_id,
                    "probability": clamp(30.0 * math.exp(-_minutes_between(t_first, incident_time) / 240.0)),
                    "possible": True,
                    "entry_est": incident_time,
                    "exit_est": incident_time + timedelta(
                        minutes=float(st["expected_minutes"]) if st else max_cross / 2
                    ),
                    "reconstructed_path": [affected_path_id, first_path],
                    "last_observed_location": first_path,
                    "post_incident_status": f"Observed on {first_path} after the incident.",
                    "explanation": "",
                    "kind": "post_incident_only",
                }
            base_gap = _minutes_between(t_first, incident_time)

        # ------------------------------------------------------------------
        # Combine temporal, coverage, recency and history factors.
        # ------------------------------------------------------------------
        mu = max(
            _triangular(incident_time, w["start"], w["end"], w["mode"])
            for w in windows
        ) if windows else 0.0

        w = windows[0] if windows else {}
        entry_est = w.get("entry_est", incident_time)
        exit_est = w.get("exit_est", incident_time)
        route = w.get("route", [affected_path_id])

        recency = math.exp(-base_gap / RECENCY_SCALE_MINUTES)
        history = self._history_factor(person_id, affected_path_id)

        probability = clamp(mu * 100.0 * route_coverage * recency * history)

        return {
            "person_id": person_id,
            "probability": probability,
            "possible": probability > 0,
            "entry_est": entry_est,
            "exit_est": exit_est,
            "reconstructed_path": route,
            "last_observed_location": (last_path or first_path),
            "post_incident_status": (
                f"Last seen on {last_path} before the incident."
                if last_path else f"First seen on {first_path} after the incident."
            ),
            "explanation": "",
            "kind": windows[0].get("kind", "unknown") if windows else "none",
            "_mu": mu,
            "_coverage": route_coverage,
            "_recency": recency,
            "_history": history,
            "_windows": [
                {
                    "start": isoformat(wd["start"]),
                    "end": isoformat(wd["end"]),
                    "mode": isoformat(wd["mode"]),
                }
                for wd in windows
            ],
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _history_factor(self, person_id: str, affected_path_id: str) -> float:
        """Boost based on the person's actual observed transitions through the
        affected path. Real observations only; never fabricated."""
        events = self.timeline_generator.observations(person_id)
        if len(events) < 2:
            return 1.0
        ordered = sorted(events, key=lambda e: parse_timestamp(e["timestamp"]) or parse_timestamp("1970-01-01"))
        total = 0
        through = 0
        for prev, curr in zip(ordered, ordered[1:]):
            p_path = prev.get("path_id")
            c_path = curr.get("path_id")
            if not p_path or not c_path or p_path == c_path:
                continue
            route = self.path_manager.shortest_path(p_path, c_path)
            total += 1
            if route and affected_path_id in route:
                through += 1
        if total == 0:
            return 1.0
        return clamp(1.0 + 0.2 * (through / total), 1.0, 1.2)

    def _empty_result(self, person_id: str, incident_time, reason: str) -> dict:
        return {
            "person_id": person_id,
            "probability": 0.0,
            "possible": False,
            "entry_est": None,
            "exit_est": None,
            "reconstructed_path": [],
            "last_observed_location": None,
            "post_incident_status": reason,
            "explanation": reason,
            "kind": "impossible",
            "_mu": 0.0,
            "_coverage": 0.0,
            "_recency": 0.0,
            "_history": 1.0,
            "_windows": [],
        }

    def _build_explanation(self, result: dict) -> str:
        pid = result["person_id"]
        prob = result["probability"]
        kind = result["kind"]
        p = f"{prob:.1f}%"

        base = (
            f"{pid}: {p} probability of being on "
            f"{(result.get('reconstructed_path') or ['the affected path'])[-1]}. "
        )

        if kind == "impossible":
            return base + result.get("post_incident_status", "Physically implausible.")

        if kind == "post_incident_only":
            return (
                base + "Only a post-incident observation exists, so the entry time "
                "is unconstrained and the estimate is low-confidence."
            )

        windows = result.get("_windows") or []
        if windows:
            win = windows[0]
            base += (
                f"Plausible presence window on the affected path: "
                f"{win['start']} to {win['end']} (most likely around {win['mode']}). "
            )
        if result.get("entry_est") and result.get("exit_est"):
            base += (
                f"Estimated entry: {isoformat(result['entry_est'])}; "
                f"estimated exit: {isoformat(result['exit_est'])}. "
            )
        base += (
            f"Evidence strength: temporal-overlap {result.get('_mu', 0):.2f}, "
            f"route coverage {result.get('_coverage', 0):.2f}, "
            f"recency {result.get('_recency', 0):.2f}. "
        )
        if result.get("_history", 1) > 1.0:
            base += "Person has previously been observed traversing the affected path. "
        base += "This is a probability estimate of potential exposure, not a confirmed sighting."
        return base