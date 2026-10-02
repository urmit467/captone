-- One person's timeline.   psql:  \set pid 102
SELECT visit_idx, path_id, round(enter_ts::numeric, 1) AS enter_s, round(exit_ts::numeric, 1) AS exit_s,
       round((exit_ts - enter_ts)::numeric, 1) AS stay_s, n_obs
FROM visits WHERE person_id = :pid ORDER BY visit_idx;
