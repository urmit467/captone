-- For everyone sighted near time T: last sighting before T and first sighting after T.   psql: \set T 5000 \set W 900
WITH cand AS (SELECT DISTINCT person_id FROM observations WHERE ts BETWEEN :T - :W AND :T + :W),
bef AS (SELECT DISTINCT ON (o.person_id) o.person_id, o.path_id AS last_path, o.ts AS last_ts
        FROM observations o JOIN cand USING (person_id) WHERE o.ts < :T ORDER BY o.person_id, o.ts DESC),
aft AS (SELECT DISTINCT ON (o.person_id) o.person_id, o.path_id AS next_path, o.ts AS next_ts
        FROM observations o JOIN cand USING (person_id) WHERE o.ts >= :T ORDER BY o.person_id, o.ts ASC)
SELECT c.person_id, bef.last_path, round(bef.last_ts::numeric, 1) AS last_ts, aft.next_path, round(aft.next_ts::numeric, 1) AS next_ts,
       round((aft.next_ts - bef.last_ts)::numeric, 1) AS gap_s
FROM cand c LEFT JOIN bef USING (person_id) LEFT JOIN aft USING (person_id)
ORDER BY c.person_id;
