-- EVALUATION ONLY (uses ground truth). Does the Week 5 feasibility filter keep the true occupants and drop the rest?
-- recall_of_filter should be close to 1.0 for people that have sightings on both sides; the 'impossible' rows should contain no true occupants.
WITH c AS (
    SELECT i.id AS incident_id, ic.*
    FROM incidents i, LATERAL incident_candidates(i.id, :W) ic
)
SELECT c.status, SUM(l.in_hidden) AS truly_inside, COUNT(*) - SUM(l.in_hidden) AS not_inside, COUNT(*) AS total
FROM c JOIN ground_truth_labels l ON l.incident_id = c.incident_id AND l.person_id = c.person_id
GROUP BY c.status ORDER BY total DESC;
