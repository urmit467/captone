-- Data-quality overview: visits and stay duration per path (the hidden path has none - nobody films it).
SELECT p.id AS path_id, p.has_camera, COUNT(v.id) AS visits,
       round(AVG(v.exit_ts - v.enter_ts)::numeric, 1) AS avg_stay_s,
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY v.exit_ts - v.enter_ts))::numeric, 1) AS median_stay_s
FROM paths p LEFT JOIN visits v ON v.path_id = p.id
GROUP BY p.id, p.has_camera ORDER BY p.id;
