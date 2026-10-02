-- Project algorithm steps 2-4, in SQL:
--   2. retrieve everyone sighted within +-p_win seconds of the incident
--   3. find each person's LAST sighting before T (path a, time ta) and FIRST sighting after T (path b, time tb)
--   4. decide whether the person could physically have been inside the hidden path at T
-- Week 6 turns the 'possible' rows into probabilities. Reads only observations / incidents / route_times / run_meta.

CREATE OR REPLACE FUNCTION incident_candidates(p_incident INTEGER, p_win DOUBLE PRECISION DEFAULT 900)
RETURNS TABLE (person_id INTEGER, a TEXT, ta DOUBLE PRECISION, b TEXT, tb DOUBLE PRECISION,
               gap_s DOUBLE PRECISION, min_crossing_s DOUBLE PRECISION, status TEXT)
LANGUAGE sql STABLE AS $$
    WITH inc AS (SELECT start_time AS t FROM incidents WHERE id = p_incident),
    meta AS (
        SELECT (SELECT value FROM run_meta WHERE key = 'hidden_path') AS hid,
               (SELECT value::double precision FROM run_meta WHERE key = 'time_slack_s') AS slack
    ),
    cand AS (
        SELECT DISTINCT o.person_id FROM observations o CROSS JOIN inc
        WHERE o.ts BETWEEN inc.t - p_win AND inc.t + p_win
    ),
    bef AS (
        SELECT DISTINCT ON (o.person_id) o.person_id, o.path_id AS a, o.ts AS ta
        FROM observations o JOIN cand c ON c.person_id = o.person_id CROSS JOIN inc
        WHERE o.ts < inc.t ORDER BY o.person_id, o.ts DESC
    ),
    aft AS (
        SELECT DISTINCT ON (o.person_id) o.person_id, o.path_id AS b, o.ts AS tb
        FROM observations o JOIN cand c ON c.person_id = o.person_id CROSS JOIN inc
        WHERE o.ts >= inc.t ORDER BY o.person_id, o.ts ASC
    )
    SELECT c.person_id, bef.a, bef.ta, aft.b, aft.tb,
           aft.tb - bef.ta AS gap_s,
           rt.min_seconds  AS min_crossing_s,
           CASE
             WHEN bef.a IS NULL THEN 'no sighting before'
             WHEN aft.b IS NULL THEN
                  CASE WHEN EXISTS (SELECT 1 FROM path_edges e, meta WHERE e.from_path = bef.a AND e.to_path = meta.hid)
                       THEN 'pending: may be inside, not seen again yet' ELSE 'no sighting after' END
             WHEN bef.a = aft.b THEN 'same path before and after'
             WHEN rt.min_seconds IS NULL THEN 'no route through hidden path'
             WHEN aft.tb - bef.ta < rt.min_seconds - (SELECT slack FROM meta) THEN 'impossible: reached too fast'
             ELSE 'possible'
           END AS status
    FROM cand c
    LEFT JOIN bef ON bef.person_id = c.person_id
    LEFT JOIN aft ON aft.person_id = c.person_id
    LEFT JOIN route_times rt ON rt.from_path = bef.a AND rt.to_path = aft.b AND rt.via_hidden
$$;
