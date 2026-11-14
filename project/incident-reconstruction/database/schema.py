"""Database schema description: collection names and index definitions.

Collections live in the ``incident_reconstruction`` database. ObjectIds are used
as primary keys; every reference between collections stores the stringified
ObjectId of the referenced document.
"""

DATABASE_NAME = "incident_reconstruction"

COLLECTIONS = [
    "persons",
    "cameras",
    "paths",
    "observations",
    "trajectories",
    "incidents",
    "predictions",
    "ground_truth",
]

# Index specifications: collection -> list of (key, direction) or
# list of dicts with 'keys' and 'unique' options.
INDEXES = {
    "persons": [
        {"keys": [("person_id", 1)], "unique": True},
    ],
    "cameras": [
        {"keys": [("camera_id", 1)], "unique": True},
        {"keys": [("path_id", 1)]},
    ],
    "paths": [
        {"keys": [("path_id", 1)], "unique": True},
        {"keys": [("is_observable", 1)]},
    ],
    "observations": [
        {"keys": [("person_id", 1)]},
        {"keys": [("path_id", 1)]},
        {"keys": [("camera_id", 1)]},
        {"keys": [("timestamp", 1)]},
        {"keys": [("person_id", 1), ("timestamp", 1)]},
    ],
    "trajectories": [
        {"keys": [("person_id", 1)]},
        {"keys": [("source", 1)]},
    ],
    "incidents": [
        {"keys": [("incident_id", 1)], "unique": True},
        {"keys": [("affected_path_id", 1)]},
        {"keys": [("incident_time", 1)]},
    ],
    "predictions": [
        {"keys": [("incident_id", 1)]},
        {"keys": [("person_id", 1)]},
        {"keys": [("risk_level", 1)]},
    ],
    "ground_truth": [
        {"keys": [("incident_id", 1), ("person_id", 1)], "unique": True},
        {"keys": [("person_id", 1)]},
    ],
}