-- Who was seen on a path connected to the hidden path within [T - W, T + W]?
-- psql:    \set T 5000
--          \set W 900
--          \i sql/queries/q1_seen_near_hidden_path.sql
SELECT v.person_id, v.path_id, round(v.enter_ts::numeric, 1) AS first_seen, round(v.exit_ts::numeric, 1) AS last_seen, v.n_obs
FROM visits v
WHERE v.path_id IN (SELECT from_path FROM path_edges WHERE to_path   = (SELECT value FROM run_meta WHERE key = 'hidden_path')
                    UNION
                    SELECT to_path   FROM path_edges WHERE from_path = (SELECT value FROM run_meta WHERE key = 'hidden_path'))
  AND v.exit_ts >= :T - :W AND v.enter_ts <= :T + :W
ORDER BY v.person_id, v.enter_ts;
