# Week 5 - Timelines and database

Goal: turn raw sightings into per-person timelines stored in PostgreSQL, put the path graph in NetworkX, and provide the queries
the incident engine needs ("who was seen on neighbours of P4 in [T-15 min, T+15 min]?", "who could be inside P4 at T?").

```
observations (DB)  ->  timeline.build_segments  ->  trajectories (DB, source='observed')
                   ->  timeline.build_transitions (gap + route feasibility per consecutive pair)
graph.PathGraph (NetworkX DiGraph)  +  queries.py  ->  seen_on_neighbors / incident_context / person_timeline
```

| File | Purpose |
|---|---|
| `graph.py` | `PathGraph`: NetworkX DiGraph from `path_graph.json`, neighbours, all routes a->b with travel-time windows, `gap_feasibility(a, b, gap)` |
| `timeline.py` | sightings -> segments (path, first_seen, last_seen); flicker removal; transitions labelled direct / hidden / either / impossible |
| `queries.py` | `person_timeline`, `seen_on_neighbors`, `incident_context` |
| `db.py`, `timeline_store.py` | PostgreSQL (psycopg2) or SQLite backend; `trajectories` table = the stored timeline |
| `load_to_db.py` | `sim` (Week 2 data) and `real` (Week 3 track CSVs + Week 4 `global_timeline.csv`) loaders |
| `build_timelines.py` | DB observations -> segments -> `trajectories` |
| `query_timelines.py` | CLI: `person`, `neighbors` (pandas or `--sql`), `context` |
| `validate_timelines.py` | evaluation on simulated data against the hidden ground truth |
| `test_timelines.py` | 13 unit tests (`python -m unittest -v test_timelines`) |

## Run
```
python load_to_db.py sim --db sqlite:///week5.db --reset --with-ground-truth     # or postgresql://recon:recon@localhost:5432/recon
python build_timelines.py --db sqlite:///week5.db
python query_timelines.py --db sqlite:///week5.db person 85
python query_timelines.py --db sqlite:///week5.db neighbors P4 --t 1850.7 --window 900
python query_timelines.py --db sqlite:///week5.db context P4 --t 1850.7 --window 900
python validate_timelines.py                     # no DB needed
```
Real footage: `load_to_db.py real --tracks "../week3_detection_tracking/output/tracks/*.csv" --timeline ../week4_reid/global_timeline.csv --reset`,
then pass `--t0 1970-01-01T00:00:00+00:00` to the build/query scripts (Week 3 writes epoch seconds).

## Design decisions
* **Segments use first/last SEEN times.** True entry is earlier and true exit later (median ~1.8 s on the default simulator, ~12-14 s on the harsh one).
  Week 6 must model that, not treat `first_seen` as the entry time.
* **Flicker removal.** +-0.5 s timestamp jitter swaps sightings at path boundaries (P2 P2 P3 P2 P3). Without cleaning this created 73 impossible
  backward moves; with it 6 remain (adjacent-path gaps of 15-22 s caused by runs of missed detections, just above `time_slack_s` = 15).
* **Feasibility rule** = same as Week 4 gating: gap in [L/v_max - jitter, L/v_min + slack], L = length of the intermediate paths.
  The upper bound is a HARD cut. Your own example (P3 10:21 -> P5 10:35 = 840 s over 300 m = 0.36 m/s) fails the default bound (765 s), so use
  `--extra-dwell` (seconds) if participants may linger in the hidden path. Default 0 keeps Week 4 and Week 5 consistent.
* `incident_context` only FILTERS. It gives statuses (`seen_at_T`, `hidden_feasible`, `not_feasible`, `no_before`, `no_after`) and `hidden_only`
  (gap explained only by routes through P4). Probabilities are Week 6.
* `paths.avg_speed_mps` / `capacity` are left NULL on purpose: Week 6 fits travel times from data rather than reading simulator parameters.
* `ground_truth_presence` is filled only with `--with-ground-truth`; only `validate_timelines.py` reads it.

## Results (simulator, `validate_timelines.py`)
| Data | Segments built / true | Candidate recall | Candidate precision | Notes |
|---|---|---|---|---|
| default config (3000 people, 300 incidents) | 11011 / 11011, all matched | 1.0000 (10219/10219) | 0.985 (152 false candidates) | first_seen - true_enter: median 1.75 s, p95 5.4 s |
| harsh config: P_DETECT 0.4, period 8 s, dwell 0.4 x 90 s, speed sigma 0.35; `--max-gap 120 --extra-dwell 120` | 11160 / 11162 | 0.9983 (11535/11555) | 0.903 (1240 false) | 17 of 20 lost positives had no sighting after T in the window, 3 not_feasible |

The harsh run with the default `--max-gap 60` fragmented segments (12629 built) and with `--extra-dwell 0` lost 40 positives instead of 20, so both knobs matter once noise is realistic.
"Candidate precision" here is of the feasibility filter only; it is not the Week 6/7 flagging precision.

## What was and was not tested here
* Tested: all of the above on SQLite, the 13 unit tests (incl. the spec's 10:30 example and the "P5 seen before minimum P3->P4 time" exclusion),
  SQL query == pandas query, DB round trip, and the `real` loader on fabricated Week 3/4-format files derived from simulator data.
* **Not tested: the PostgreSQL path** (`psycopg2`, COPY bulk insert, `setval`). No Postgres or psycopg2 was available, so run `load_to_db.py sim --db postgresql://...`
  once on your Docker instance and tell me if anything breaks. `schema.sql` was not changed.
* **Not tested on real footage**; real Re-ID errors will produce `impossible` transitions, which `build_timelines.py` reports as a warning.
* Both simulator configs are synthetic; the recall numbers say the pipeline is consistent, not how it will do on your recordings.

## Hand-off to Week 6
Inputs for the probabilistic model: `incident_context(...)` rows (`before_path`, `before_dt_s`, `after_path`, `after_dt_s`, `gap_s`, `hidden_ok`, `hidden_only`)
for the candidate set, and `build_transitions(...)` (all observed gaps per path pair) to fit per-edge travel-time distributions.
Fit them ONLY on transitions with kind `direct` where the route is unambiguous (and on non-hidden pairs); `either`/`hidden` gaps are the ones being explained.
