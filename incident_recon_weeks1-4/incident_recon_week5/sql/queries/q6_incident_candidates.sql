-- The candidate table for one incident, with the feasibility decision per person.   psql: \set incident_id 5 \set W 900
SELECT person_id, a AS last_path, round(ta::numeric, 1) AS last_ts, b AS next_path, round(tb::numeric, 1) AS next_ts,
       round(gap_s::numeric, 1) AS gap_s, round(min_crossing_s::numeric) AS min_crossing_s, status
FROM incident_candidates(:incident_id, :W)
ORDER BY (status = 'possible') DESC, gap_s;
