-- Walking-time bounds between paths (built by build_route_times()). Hidden-path crossings listed first.
SELECT from_path, to_path, via_hidden, n_routes, round(min_seconds::numeric) AS min_s, round(max_seconds::numeric) AS max_s
FROM route_times WHERE from_path IN (SELECT from_path FROM path_edges WHERE to_path = (SELECT value FROM run_meta WHERE key = 'hidden_path'))
ORDER BY via_hidden DESC, from_path, to_path;
