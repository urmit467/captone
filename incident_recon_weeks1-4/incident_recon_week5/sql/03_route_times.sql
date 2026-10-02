-- Walking-time bounds between every pair of paths, found by walking the path graph with a recursive CTE.
--   min_seconds = (total length of the paths strictly between A and B) / v_max      (fastest possible)
--   max_seconds = same length / v_min                                               (slowest plausible)
-- Rows are split by whether the route passes through the hidden path.
-- Reads run_meta keys: hidden_path, v_min_mps, v_max_mps.   Usage:  SELECT build_route_times();

CREATE OR REPLACE FUNCTION build_route_times() RETURNS INTEGER LANGUAGE plpgsql AS $$
DECLARE
    hid  TEXT;
    vmax DOUBLE PRECISION;
    vmin DOUBLE PRECISION;
    n    INTEGER;
BEGIN
    SELECT value INTO hid FROM run_meta WHERE key = 'hidden_path';
    SELECT value::double precision INTO vmax FROM run_meta WHERE key = 'v_max_mps';
    SELECT value::double precision INTO vmin FROM run_meta WHERE key = 'v_min_mps';
    DELETE FROM route_times;
    INSERT INTO route_times (from_path, to_path, via_hidden, n_routes, min_seconds, max_seconds)
    WITH RECURSIVE r (start_path, cur, route, inter_len, via_hidden) AS (
        SELECT e.from_path, e.to_path, ARRAY[e.from_path, e.to_path], 0::double precision, FALSE
        FROM path_edges e
        UNION ALL
        SELECT r.start_path, e.to_path, r.route || e.to_path,
               r.inter_len + p.length_m,                -- r.cur becomes an intermediate path
               r.via_hidden OR (r.cur = hid)
        FROM r
        JOIN path_edges e ON e.from_path = r.cur
        JOIN paths p      ON p.id = r.cur
        WHERE NOT (e.to_path = ANY (r.route))           -- simple routes only (no loops)
    )
    SELECT start_path, cur, via_hidden, COUNT(*), MIN(inter_len) / vmax, MAX(inter_len) / vmin
    FROM r GROUP BY start_path, cur, via_hidden;
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END $$;
