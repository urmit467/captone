"""CRUD operations on top of the MongoDB connection.

Every public function wraps a single collection operation with validation and
consistent error handling. Timestamps are always stored as UTC ``datetime``
objects and returned to callers as ISO-8601 strings. ``_id`` values are
exposed to callers as stringified ObjectIds under the ``id`` key.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from bson.errors import InvalidId

from database.connection import get_database
from database.schema import COLLECTIONS, INDEXES

logger = logging.getLogger(__name__)


def _serialize(doc: Optional[dict]) -> Optional[dict]:
    """Convert a raw MongoDB document into a caller-friendly dict."""
    if doc is None:
        return None
    out = dict(doc)
    oid = out.pop("_id", None)
    out["id"] = str(oid) if oid is not None else None
    for key, value in list(out.items()):
        if isinstance(value, datetime):
            # Naive datetimes are stored as UTC; label them as UTC rather than
            # re-interpreting them as local time (astimezone on a naive datetime
            # assumes the local zone and would shift the wall-clock value).
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            out[key] = value.astimezone(timezone.utc).isoformat()
        elif isinstance(value, ObjectId):
            out[key] = str(value)
    return out


def _parse_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise ValueError(f"Invalid ObjectId: {value!r}") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DBOperations:
    """Thin wrapper around PyMongo collections with validation."""

    def __init__(self) -> None:
        self.db = get_database()

    # ------------------------------------------------------------------ #
    # Generic helpers                                                    #
    # ------------------------------------------------------------------ #
    def _collection(self, name: str):
        if name not in COLLECTIONS:
            raise ValueError(f"Unknown collection: {name}")
        return self.db[name]

    def insert_one(self, collection: str, document: dict) -> str:
        doc = dict(document)
        if "_id" not in doc or doc["_id"] is None:
            doc.pop("_id", None)
        now = _utc_now()
        doc.setdefault("created_at", now)
        result = self._collection(collection).insert_one(doc)
        logger.debug("Inserted into %s: %s", collection, result.inserted_id)
        return str(result.inserted_id)

    def find_one(self, collection: str, filter_query: dict) -> Optional[dict]:
        doc = self._collection(collection).find_one(filter_query)
        return _serialize(doc)

    def find_many(
        self,
        collection: str,
        filter_query: Optional[dict] = None,
        sort: Optional[list] = None,
        limit: int = 0,
    ) -> list[dict]:
        cursor = self._collection(collection).find(filter_query or {})
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        return [_serialize(doc) for doc in cursor]

    def update_one(self, collection: str, filter_query: dict, set_fields: dict) -> bool:
        if "created_at" in set_fields:
            set_fields = {k: v for k, v in set_fields.items() if k != "created_at"}
        set_fields["updated_at"] = _utc_now()
        result = self._collection(collection).update_one(
            filter_query, {"$set": set_fields}
        )
        return result.modified_count > 0

    def delete_one(self, collection: str, filter_query: dict) -> bool:
        result = self._collection(collection).delete_one(filter_query)
        return result.deleted_count > 0

    def ensure_indexes(self) -> None:
        """Create the indexes defined in database.schema."""
        from database.schema import INDEXES

        for collection, specs in INDEXES.items():
            for spec in specs:
                keys = spec.get("keys", [])
                unique = bool(spec.get("unique", False))
                self._collection(collection).create_index(keys, unique=unique)
        logger.info("Database indexes ensured")

    # ------------------------------------------------------------------ #
    # Persons                                                            #
    # ------------------------------------------------------------------ #
    def insert_person(self, person_id: str, metadata: Optional[dict] = None) -> str:
        if self.find_one("persons", {"person_id": person_id}):
            raise ValueError(f"Person already exists: {person_id}")
        return self.insert_one(
            "persons",
            {"person_id": person_id, "metadata": metadata or {}},
        )

    def get_person(self, person_id: str) -> Optional[dict]:
        return self.find_one("persons", {"person_id": person_id})

    def list_persons(self) -> list[dict]:
        return self.find_many("persons", sort=[("created_at", -1)])

    def update_person(self, person_id: str, fields: dict) -> bool:
        return self.update_one("persons", {"person_id": person_id}, fields)

    def delete_person(self, person_id: str) -> bool:
        return self.delete_one("persons", {"person_id": person_id})

    # ------------------------------------------------------------------ #
    # Cameras                                                            #
    # ------------------------------------------------------------------ #
    def insert_camera(
        self,
        camera_id: str,
        camera_name: str,
        path_id: str,
        location: str = "",
        status: str = "active",
    ) -> str:
        if self.find_one("cameras", {"camera_id": camera_id}):
            raise ValueError(f"Camera already exists: {camera_id}")
        return self.insert_one(
            "cameras",
            {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "path_id": path_id,
                "location": location,
                "status": status,
            },
        )

    def get_camera(self, camera_id: str) -> Optional[dict]:
        return self.find_one("cameras", {"camera_id": camera_id})

    def list_cameras(self) -> list[dict]:
        return self.find_many("cameras", sort=[("camera_id", 1)])

    def update_camera(self, camera_id: str, fields: dict) -> bool:
        return self.update_one("cameras", {"camera_id": camera_id}, fields)

    def delete_camera(self, camera_id: str) -> bool:
        return self.delete_one("cameras", {"camera_id": camera_id})

    def cameras_on_path(self, path_id: str) -> list[dict]:
        return self.find_many("cameras", {"path_id": path_id})

    # ------------------------------------------------------------------ #
    # Paths                                                              #
    # ------------------------------------------------------------------ #
    def insert_path(
        self,
        path_id: str,
        name: str,
        length: float,
        average_speed: float,
        is_observable: bool = True,
        connected_paths: Optional[list[str]] = None,
    ) -> str:
        if self.find_one("paths", {"path_id": path_id}):
            raise ValueError(f"Path already exists: {path_id}")
        return self.insert_one(
            "paths",
            {
                "path_id": path_id,
                "name": name,
                "length": float(length),
                "average_speed": float(average_speed),
                "is_observable": bool(is_observable),
                "connected_paths": connected_paths or [],
            },
        )

    def get_path(self, path_id: str) -> Optional[dict]:
        return self.find_one("paths", {"path_id": path_id})

    def list_paths(self) -> list[dict]:
        return self.find_many("paths", sort=[("path_id", 1)])

    def update_path(self, path_id: str, fields: dict) -> bool:
        return self.update_one("paths", {"path_id": path_id}, fields)

    def delete_path(self, path_id: str) -> bool:
        return self.delete_one("paths", {"path_id": path_id})

    # ------------------------------------------------------------------ #
    # Observations                                                       #
    # ------------------------------------------------------------------ #
    def insert_observation(
        self,
        person_id: str,
        camera_id: str,
        path_id: str,
        timestamp,
        bounding_box: Optional[dict] = None,
        tracking_id: Optional[str] = None,
        confidence: Optional[float] = None,
        source: str = "camera",
        image_path: Optional[str] = None,
    ) -> str:
        if isinstance(timestamp, str):
            timestamp = _parse_iso(timestamp)
        return self.insert_one(
            "observations",
            {
                "person_id": person_id,
                "camera_id": camera_id,
                "path_id": path_id,
                "timestamp": timestamp,
                "bounding_box": bounding_box or {},
                "tracking_id": tracking_id or "",
                "confidence": float(confidence) if confidence is not None else None,
                "source": source,
                "image_path": image_path,
            },
        )

    def list_observations(
        self,
        person_id: Optional[str] = None,
        path_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        start_time=None,
        end_time=None,
        source: Optional[str] = None,
        sort_desc: bool = True,
        limit: int = 0,
    ) -> list[dict]:
        query: dict[str, Any] = {}
        if person_id:
            query["person_id"] = person_id
        if path_id:
            query["path_id"] = path_id
        if camera_id:
            query["camera_id"] = camera_id
        if source:
            query["source"] = source
        if start_time is not None or end_time is not None:
            time_query: dict[str, Any] = {}
            if start_time is not None:
                time_query["$gte"] = _to_datetime(start_time)
            if end_time is not None:
                time_query["$lte"] = _to_datetime(end_time)
            query["timestamp"] = time_query
        sort = [("timestamp", -1 if sort_desc else 1)]
        return self.find_many("observations", query, sort=sort, limit=limit)

    def observations_for_person(self, person_id: str) -> list[dict]:
        return self.list_observations(person_id=person_id)

    def list_unassigned_observations(self) -> list[dict]:
        """Return observations not yet linked to a person (person_id is null)."""
        return self.find_many("observations", {"person_id": None},
                              sort=[("timestamp", -1)])

    def update_observation_person(self, observation_id: str, person_id: str) -> bool:
        """Link an observation to a person by its Mongo ObjectId (string)."""
        return self.update_one("observations",
                               {"_id": _parse_object_id(observation_id)},
                               {"person_id": person_id})

    # ------------------------------------------------------------------ #
    # Trajectories                                                       #
    # ------------------------------------------------------------------ #
    def insert_trajectory(
        self,
        person_id: str,
        path_sequence: list[str],
        start_time,
        end_time,
        source: str = "reconstructed",
        confidence: Optional[float] = None,
    ) -> str:
        return self.insert_one(
            "trajectories",
            {
                "person_id": person_id,
                "path_sequence": list(path_sequence),
                "start_time": _to_datetime(start_time),
                "end_time": _to_datetime(end_time),
                "source": source,
                "confidence": float(confidence) if confidence is not None else None,
            },
        )

    def list_trajectories(self, person_id: Optional[str] = None) -> list[dict]:
        query = {"person_id": person_id} if person_id else {}
        return self.find_many(query, sort=[("start_time", -1)])

    # ------------------------------------------------------------------ #
    # Incidents                                                          #
    # ------------------------------------------------------------------ #
    def insert_incident(
        self,
        incident_id: str,
        affected_path_id: str,
        incident_time,
        severity: str,
        description: str = "",
        status: str = "open",
    ) -> str:
        if self.find_one("incidents", {"incident_id": incident_id}):
            raise ValueError(f"Incident already exists: {incident_id}")
        return self.insert_one(
            "incidents",
            {
                "incident_id": incident_id,
                "affected_path_id": affected_path_id,
                "incident_time": _to_datetime(incident_time),
                "severity": severity,
                "description": description,
                "status": status,
            },
        )

    def get_incident(self, incident_id: str) -> Optional[dict]:
        return self.find_one("incidents", {"incident_id": incident_id})

    def list_incidents(self) -> list[dict]:
        return self.find_many("incidents", sort=[("incident_time", -1)])

    def update_incident(self, incident_id: str, fields: dict) -> bool:
        return self.update_one("incidents", {"incident_id": incident_id}, fields)

    # ------------------------------------------------------------------ #
    # Predictions                                                        #
    # ------------------------------------------------------------------ #
    def insert_prediction(
        self,
        incident_id: str,
        person_id: str,
        probability: float,
        predicted_entry_time=None,
        predicted_exit_time=None,
        reconstructed_path: Optional[list[str]] = None,
        risk_level: str = "LOW",
        explanation: str = "",
        last_observed_location: Optional[str] = None,
        post_incident_status: Optional[str] = None,
    ) -> str:
        return self.insert_one(
            "predictions",
            {
                "incident_id": incident_id,
                "person_id": person_id,
                "probability": float(probability),
                "predicted_entry_time": _to_datetime(predicted_entry_time)
                if predicted_entry_time is not None
                else None,
                "predicted_exit_time": _to_datetime(predicted_exit_time)
                if predicted_exit_time is not None
                else None,
                "reconstructed_path": reconstructed_path or [],
                "risk_level": risk_level,
                "explanation": explanation,
                "last_observed_location": last_observed_location or "",
                "post_incident_status": post_incident_status or "",
            },
        )

    def list_predictions(self, incident_id: str) -> list[dict]:
        return self.find_many(
            "predictions",
            {"incident_id": incident_id},
            sort=[("probability", -1)],
        )

    def delete_predictions(self, incident_id: str) -> int:
        result = self._collection("predictions").delete_many({"incident_id": incident_id})
        return result.deleted_count

    # ------------------------------------------------------------------ #
    # Ground truth                                                       #
    # ------------------------------------------------------------------ #
    def insert_ground_truth(
        self,
        incident_id: str,
        person_id: str,
        present_on_affected_path: bool,
        actual_path_sequence: list[str],
        actual_entry_time=None,
        actual_exit_time=None,
    ) -> str:
        if self.find_one(
            "ground_truth", {"incident_id": incident_id, "person_id": person_id}
        ):
            raise ValueError("Ground truth already exists for this person/incident")
        return self.insert_one(
            "ground_truth",
            {
                "incident_id": incident_id,
                "person_id": person_id,
                "present_on_affected_path": bool(present_on_affected_path),
                "actual_path_sequence": list(actual_path_sequence),
                "actual_entry_time": _to_datetime(actual_entry_time)
                if actual_entry_time is not None
                else None,
                "actual_exit_time": _to_datetime(actual_exit_time)
                if actual_exit_time is not None
                else None,
            },
        )

    def list_ground_truth(self, incident_id: Optional[str] = None) -> list[dict]:
        query = {"incident_id": incident_id} if incident_id else {}
        return self.find_many("ground_truth", query, sort=[("person_id", 1)])


def _to_datetime(value):
    """Coerce a str/datetime into a UTC datetime (naive datetimes treated as UTC)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    from dateutil import parser

    dt = parser.isoparse(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_iso(value: str) -> datetime:
    from dateutil import parser

    dt = parser.isoparse(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)