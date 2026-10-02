-- PostgreSQL schema for the incident-time person reconstruction prototype.
-- Persons are anonymised (Person_001 ...). No faces or names are stored.
-- DESIGN version. The working schema (times as seconds, plus SQL functions) is sql/01_schema.sql in the Week 5 package.
-- Install PostgreSQL directly on your computer; no Docker is used.

CREATE TABLE persons (
    id            SERIAL PRIMARY KEY,
    label         TEXT UNIQUE NOT NULL,            -- 'Person_001'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE paths (
    id                TEXT PRIMARY KEY,            -- 'P4'
    length_m          REAL NOT NULL,
    capacity          INTEGER,
    avg_speed_mps     REAL,
    has_camera        BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE path_edges (
    from_path   TEXT REFERENCES paths(id),
    to_path     TEXT REFERENCES paths(id),
    PRIMARY KEY (from_path, to_path)
);

CREATE TABLE cameras (
    id          TEXT PRIMARY KEY,                  -- 'C3'
    path_id     TEXT NOT NULL REFERENCES paths(id),
    location    TEXT,
    clock_offset_s REAL NOT NULL DEFAULT 0         -- for time synchronisation between cameras
);

-- Raw single-camera tracks (Week 3 output)
CREATE TABLE tracks (
    id           SERIAL PRIMARY KEY,
    camera_id    TEXT NOT NULL REFERENCES cameras(id),
    local_track_id INTEGER NOT NULL,               -- ID assigned by ByteTrack/BoT-SORT
    start_time   TIMESTAMPTZ NOT NULL,
    end_time     TIMESTAMPTZ NOT NULL,
    n_frames     INTEGER,
    person_id    INTEGER REFERENCES persons(id),   -- filled by Re-ID (Week 4)
    reid_score   REAL,
    UNIQUE (camera_id, local_track_id, start_time)
);

-- One row per detection (or per sampled frame)
CREATE TABLE observations (
    id          BIGSERIAL PRIMARY KEY,
    person_id   INTEGER REFERENCES persons(id),
    track_id    INTEGER REFERENCES tracks(id),
    camera_id   TEXT NOT NULL REFERENCES cameras(id),
    path_id     TEXT NOT NULL REFERENCES paths(id),
    ts          TIMESTAMPTZ NOT NULL,
    x REAL, y REAL,
    confidence  REAL
);
CREATE INDEX idx_obs_person_ts ON observations (person_id, ts);
CREATE INDEX idx_obs_path_ts   ON observations (path_id, ts);

-- Per-person, per-path presence intervals (observed or inferred)
CREATE TABLE trajectories (
    id          SERIAL PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES persons(id),
    path_id     TEXT NOT NULL REFERENCES paths(id),
    start_time  TIMESTAMPTZ NOT NULL,
    end_time    TIMESTAMPTZ NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('observed', 'inferred')),
    confidence  REAL
);

CREATE TABLE incidents (
    id                SERIAL PRIMARY KEY,
    affected_path_id  TEXT NOT NULL REFERENCES paths(id),
    start_time        TIMESTAMPTZ NOT NULL,
    severity          REAL NOT NULL DEFAULT 1.0,
    status            TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE predictions (
    incident_id          INTEGER NOT NULL REFERENCES incidents(id),
    person_id            INTEGER NOT NULL REFERENCES persons(id),
    predicted_path       TEXT REFERENCES paths(id),
    probability          REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    predicted_entry_time TIMESTAMPTZ,
    predicted_exit_time  TIMESTAMPTZ,
    model_name           TEXT,
    PRIMARY KEY (incident_id, person_id, model_name)
);

CREATE TABLE risk_scores (
    incident_id  INTEGER NOT NULL REFERENCES incidents(id),
    person_id    INTEGER NOT NULL REFERENCES persons(id),
    probability  REAL NOT NULL,
    severity     REAL NOT NULL,
    risk_score   REAL NOT NULL,
    level        TEXT NOT NULL CHECK (level IN ('HIGH','MEDIUM','LOW')),
    status       TEXT NOT NULL DEFAULT 'flagged',
    PRIMARY KEY (incident_id, person_id)
);

-- Evaluation-only table: NEVER read by the inference code.
CREATE TABLE ground_truth_presence (
    person_id   INTEGER NOT NULL REFERENCES persons(id),
    path_id     TEXT NOT NULL REFERENCES paths(id),
    start_time  TIMESTAMPTZ NOT NULL,
    end_time    TIMESTAMPTZ NOT NULL
);
