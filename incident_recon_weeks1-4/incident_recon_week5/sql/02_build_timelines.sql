-- Builds the per-person timeline (visits + trajectories) from raw observations using SQL window functions.
--   "gaps and islands": a new visit starts whenever the path changes or there is a long silence.
--   Timestamp jitter can make sightings at a path boundary alternate (P2,P3,P2,P3). A very short visit sandwiched
--   between two visits of the same path is relabelled and merged ("flip repair"); repeated until stable.
-- Usage:  SELECT build_visits();                -- defaults: 300 s silence splits a visit, 12 s flip window
--         SELECT build_visits(600, 10);

CREATE OR REPLACE FUNCTION build_visits(split_gap DOUBLE PRECISION DEFAULT 300, flip_s DOUBLE PRECISION DEFAULT 12)
RETURNS INTEGER LANGUAGE plpgsql AS $$
DECLARE
    n_fix    INTEGER;
    it       INTEGER := 0;
    n_visits INTEGER;
BEGIN
    DROP TABLE IF EXISTS obs_work;
    CREATE TEMP TABLE obs_work AS
        SELECT id, person_id, path_id, ts, 0::bigint AS visit_no FROM observations;
    CREATE INDEX ON obs_work (person_id, ts, id);

    LOOP
        -- 1. number the visits of every person
        UPDATE obs_work w SET visit_no = x.vn
        FROM (
            SELECT id, SUM(is_start) OVER (PARTITION BY person_id ORDER BY ts, id) AS vn
            FROM (
                SELECT id, person_id, ts,
                       CASE WHEN path_id IS DISTINCT FROM LAG(path_id) OVER p
                              OR ts - LAG(ts) OVER p > split_gap THEN 1 ELSE 0 END AS is_start
                FROM obs_work
                WINDOW p AS (PARTITION BY person_id ORDER BY ts, id)
            ) s
        ) x
        WHERE w.id = x.id;

        -- 2. one row per visit
        DROP TABLE IF EXISTS vis_work;
        CREATE TEMP TABLE vis_work AS
            SELECT person_id, visit_no, MIN(path_id) AS path_id,
                   MIN(ts) AS enter_ts, MAX(ts) AS exit_ts, COUNT(*)::int AS n_obs
            FROM obs_work GROUP BY person_id, visit_no;

        -- 3. flips: short visit (<= 3 sightings, <= flip_s long) between two visits of the same other path
        DROP TABLE IF EXISTS flips;
        CREATE TEMP TABLE flips AS
            SELECT person_id, visit_no, prev_path AS fix_path
            FROM (
                SELECT *, LAG(path_id)   OVER p AS prev_path, LEAD(path_id)  OVER p AS next_path,
                          LAG(exit_ts)   OVER p AS prev_exit, LEAD(enter_ts) OVER p AS next_enter
                FROM vis_work
                WINDOW p AS (PARTITION BY person_id ORDER BY visit_no)
            ) q
            WHERE prev_path = next_path AND prev_path <> path_id
              AND exit_ts - enter_ts <= flip_s AND n_obs <= 3
              AND next_enter - prev_exit <= 2 * flip_s;

        SELECT COUNT(*) INTO n_fix FROM flips;
        EXIT WHEN n_fix = 0 OR it >= 5;
        UPDATE obs_work w SET path_id = f.fix_path
        FROM flips f WHERE w.person_id = f.person_id AND w.visit_no = f.visit_no;
        it := it + 1;
    END LOOP;

    TRUNCATE visits RESTART IDENTITY;
    INSERT INTO visits (person_id, visit_idx, path_id, enter_ts, exit_ts, n_obs)
        SELECT person_id, ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY enter_ts), path_id, enter_ts, exit_ts, n_obs
        FROM vis_work;
    GET DIAGNOSTICS n_visits = ROW_COUNT;

    DELETE FROM trajectories WHERE source = 'observed';
    INSERT INTO trajectories (person_id, path_id, start_time, end_time, source, confidence)
        SELECT person_id, path_id, enter_ts, exit_ts, 'observed', 1.0 FROM visits;
    RETURN n_visits;
END $$;
