# Week 2 - Simulator and hidden ground truth

`python simulate.py` writes to `data/`:

| File | Contents | May the algorithm read it? |
|---|---|---|
| observations.csv | person_id, path_id, timestamp (camera sightings only, P4 never appears) | yes |
| segments.csv | person_id, path_id, enter, exit for EVERY path incl. P4 | NO - evaluation only |
| incidents.csv | incident_id, time | yes |
| labels.csv | incident_id, person_id, in_hidden (truth) | NO - evaluation only |

All knobs are in `config.py`: path lengths/speeds, branching probabilities, dwell, camera sampling period, detection probability,
timestamp jitter, number of people/incidents, random seed. Noise switches for later robustness tests: `P_DETECT`, `TIME_JITTER_S`, `REID_SWAP_PROB` (not yet used).

Note: with the default settings the task is easy (P4 takes ~250 s, cameras sample every 3 s). Make it harder for Week 7-8
experiments: lower `P_DETECT`, raise `OBS_PERIOD_S`, widen `SPEED_SIGMA`, add dwell.
