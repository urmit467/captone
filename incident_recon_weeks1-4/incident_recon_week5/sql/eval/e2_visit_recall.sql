-- EVALUATION ONLY. Of the true stays on camera-covered paths, how many have a matching visit in the timeline?
-- (A stay is "found" if a visit on the same path overlaps it.) Lower for sparse/noisy cameras.
SELECT g.path_id, COUNT(*) AS true_stays,
       SUM(CASE WHEN EXISTS (SELECT 1 FROM visits v WHERE v.person_id = g.person_id AND v.path_id = g.path_id
                             AND v.enter_ts <= g.end_time + 5 AND v.exit_ts >= g.start_time - 5) THEN 1 ELSE 0 END) AS found,
       round(100.0 * SUM(CASE WHEN EXISTS (SELECT 1 FROM visits v WHERE v.person_id = g.person_id AND v.path_id = g.path_id
                             AND v.enter_ts <= g.end_time + 5 AND v.exit_ts >= g.start_time - 5) THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_found
FROM ground_truth_presence g JOIN paths p ON p.id = g.path_id AND p.has_camera
GROUP BY g.path_id ORDER BY g.path_id;
