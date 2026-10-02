-- Week 5 schema (PostgreSQL). Times are stored as DOUBLE PRECISION seconds
-- (simulator clock, or epoch seconds for real footage) so both data sources use one format.
-- Persons are anonymised. Tables marked EVALUATION ONLY are never read by the queries/ or by inference code.

CREATE TABLE IF NOT EXISTS run_meta (            -- settings the SQL functions read (hidden path, speed bounds, ...)
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id    INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE                   -- 'Person_0001'
);

CREATE TABLE IF NOT EXISTS paths (
    id         TEXT PRIMARY KEY,                 -- 'P4'
    length_m   DOUBLE PRECISION NOT NULL,
    has_camera BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS path_edges (          -- directed: a person can walk from_path -> to_path
    from_path TEXT NOT NULL REFERENCES paths(id),
    to_path   TEXT NOT NULL REFERENCES paths(id),
    PRIMARY KEY (from_path, to_path)
);

CREATE TABLE IF NOT EXISTS cameras (
    id      TEXT PRIMARY KEY,                    -- 'C3'
    path_id TEXT NOT NULL REFERENCES paths(id)
);

CREATE TABLE IF NOT EXISTS observations (        -- one row per camera sighting of a (global) person
    id        BIGSERIAL PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES persons(id),
    camera_id TEXT,
    path_id   TEXT NOT NULL REFERENCES paths(id),
    ts        DOUBLE PRECISION NOT NULL,
    source    TEXT                               -- 'sim' | 'track'
);
CREATE INDEX IF NOT EXISTS idx_obs_person_ts ON observations (person_id, ts);
CREATE INDEX IF NOT EXISTS idx_obs_ts        ON observations (ts);
CREATE INDEX IF NOT EXISTS idx_obs_path_ts   ON observations (path_id, ts);

CREATE TABLE IF NOT EXISTS visits (              -- timeline: one uninterrupted stay of a person on a path
    id        BIGSERIAL PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES persons(id),
    visit_idx INTEGER NOT NULL,                  -- 1,2,3... in time order for that person
    path_id   TEXT NOT NULL REFERENCES paths(id),
    enter_ts  DOUBLE PRECISION NOT NULL,         -- first sighting on the path
    exit_ts   DOUBLE PRECISION NOT NULL,         -- last sighting on the path
    n_obs     INTEGER NOT NULL,
    UNIQUE (person_id, visit_idx)
);
CREATE INDEX IF NOT EXISTS idx_visits_path_time ON visits (path_id, enter_ts, exit_ts);

CREATE TABLE IF NOT EXISTS trajectories (        -- observed now (Week 5); inferred hidden-path stays are added in Week 6/7
    id         BIGSERIAL PRIMARY KEY,
    person_id  INTEGER NOT NULL REFERENCES persons(id),
    path_id    TEXT NOT NULL REFERENCES paths(id),
    start_time DOUBLE PRECISION NOT NULL,
    end_time   DOUBLE PRECISION NOT NULL,
    source     TEXT NOT NULL CHECK (source IN ('observed', 'inferred')),
    confidence DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS route_times (         -- filled by build_route_times(): walking-time bounds between paths
    from_path   TEXT NOT NULL REFERENCES paths(id),
    to_path     TEXT NOT NULL REFERENCES paths(id),
    via_hidden  BOOLEAN NOT NULL,                -- does the route pass through the hidden path?
    n_routes    INTEGER NOT NULL,
    min_seconds DOUBLE PRECISION NOT NULL,       -- fastest possible (v_max)
    max_seconds DOUBLE PRECISION NOT NULL,       -- slowest plausible (v_min)
    PRIMARY KEY (from_path, to_path, via_hidden)
);

CREATE TABLE IF NOT EXISTS incidents (
    id               SERIAL PRIMARY KEY,
    affected_path_id TEXT NOT NULL REFERENCES paths(id),
    start_time       DOUBLE PRECISION NOT NULL,
    severity         DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    status           TEXT NOT NULL DEFAULT 'open'
);

-- Used from Week 6/7 (created now so the schema is complete)
CREATE TABLE IF NOT EXISTS predictions (
    incident_id          INTEGER NOT NULL REFERENCES incidents(id),
    person_id            INTEGER NOT NULL REFERENCES persons(id),
    model_name           TEXT NOT NULL,
    probability          DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
    route                TEXT,
    predicted_entry_time DOUBLE PRECISION,
    predicted_exit_time  DOUBLE PRECISION,
    PRIMARY KEY (incident_id, person_id, model_name)
);
CREATE TABLE IF NOT EXISTS risk_scores (
    incident_id INTEGER NOT NULL REFERENCES incidents(id),
    person_id   INTEGER NOT NULL REFERENCES persons(id),
    probability DOUBLE PRECISION NOT NULL,
    severity    DOUBLE PRECISION NOT NULL,
    risk_score  DOUBLE PRECISION NOT NULL,
    level       TEXT NOT NULL CHECK (level IN ('HIGH', 'MEDIUM', 'LOW', 'PENDING')),
    status      TEXT NOT NULL DEFAULT 'flagged',
    PRIMARY KEY (incident_id, person_id)
);

-- EVALUATION ONLY: true answers from the simulator / hand-logged real runs. Inference never reads these.
CREATE TABLE IF NOT EXISTS ground_truth_presence (
    person_id  INTEGER NOT NULL,
    path_id    TEXT NOT NULL,
    start_time DOUBLE PRECISION NOT NULL,
    end_time   DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS ground_truth_labels (
    incident_id INTEGER NOT NULL,
    person_id   INTEGER NOT NULL,
    in_hidden   INTEGER NOT NULL,
    PRIMARY KEY (incident_id, person_id)
);
