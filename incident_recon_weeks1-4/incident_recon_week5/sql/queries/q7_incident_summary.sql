-- How many people fall into each feasibility class for one incident.
SELECT status, COUNT(*) AS people FROM incident_candidates(:incident_id, :W) GROUP BY status ORDER BY people DESC;
