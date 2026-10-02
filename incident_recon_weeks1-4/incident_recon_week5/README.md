# Week 5 - Timelines in PostgreSQL (plain SQL, no Docker)

**Goal of the week:** put every sighting into a database, turn the sightings into a **timeline per person** (which path, from when to when),
and answer the first questions of the incident algorithm *in SQL*: who was near the hidden path at time T, who could physically have been inside it?

Everything is plain SQL files you can read, edit and run in psql or pgAdmin. Python is only used to create the database, load files and print results.

## 1. Install PostgreSQL (once, no Docker)
| System | How |
|---|---|
| Windows | Download the installer from postgresql.org/download/windows (EDB). Keep the defaults, **remember the password you set for user `postgres`**, keep port 5432. pgAdmin is installed with it. |
| macOS | `brew install postgresql@16` then `brew services start postgresql@16` |
| Ubuntu | `sudo apt install postgresql` (then `sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'yourpassword'"`) |

Python packages: `pip install -r requirements.txt` (numpy, pandas, psycopg2-binary, pytest).

## 2. Run it
Windows PowerShell (one command per line, no backslashes):
```
$env:RECON_DSN = "postgresql://postgres:YOURPASSWORD@localhost:5432/recon"
python simulate.py --preset hard --n-persons 1200 --n-incidents 60
python setup_db.py --reset
python load_data.py --source sim --data-dir data/hard
python check_timelines.py
python run_query.py sql/queries/q6_incident_candidates.sql --p incident_id=5 --p W=900
```
Linux/macOS: use `export RECON_DSN=postgresql://postgres:YOURPASSWORD@localhost:5432/recon`. Presets `easy`, `medium`, `hard` control how noisy the cameras are
(datasets for `easy` and `hard` are already in `data/`, so you can skip `simulate.py`). If your password has special characters, URL-encode them (`@` -> `%40`).

`setup_db.py` creates the database `recon` itself if it does not exist.

## 3. What is in the folder
| File | What it does |
|---|---|
| `sql/01_schema.sql` | All tables (persons, paths, path_edges, cameras, observations, visits, trajectories, route_times, incidents, predictions, risk_scores, and the evaluation-only ground-truth tables). Times are seconds (float) so simulated and real footage use one format. |
| `sql/02_build_timelines.sql` | `build_visits()` - turns sightings into visits with window functions (new visit when the path changes or after 300 s of silence), repairs timestamp-jitter "flips" (P2,P3,P2,P3 -> P2,P3), fills `trajectories`. |
| `sql/03_route_times.sql` | `build_route_times()` - recursive CTE that walks the path graph and stores, for every pair of paths, the fastest and slowest walking time, split by whether the route passes through the hidden path. |
| `sql/04_incident_candidates.sql` | `incident_candidates(incident_id, window)` - the incident algorithm steps 2-4: everyone sighted near T, their last sighting before T and first after T, and a feasibility decision. |
| `sql/queries/q1..q7` | Ready-made queries (see below). Run in psql/pgAdmin or with `run_query.py`. |
| `sql/eval/e1, e2` | **Evaluation only** (uses ground truth): does the feasibility filter keep the true occupants? are true stays found in the timeline? |
| `simulate.py` | Simulator with `easy / medium / hard` camera-noise presets (hidden ground truth kept in separate files). |
| `setup_db.py`, `load_data.py`, `run_query.py`, `dbutil.py` | Create DB + schema, load data (simulated or real tracks), run a .sql file with parameters. |
| `check_timelines.py`, `timelines_ref.py` | Cross-check: SQL timeline vs an independent pandas implementation (must be identical). |
| `tests/test_week5.py` | 5 tests incl. the full pipeline on a real PostgreSQL. |

### The queries
| File | Question it answers | Parameters |
|---|---|---|
| q1 | Who was seen on a path next to the hidden path in [T-W, T+W]? | `T`, `W` |
| q2 | One person's timeline | `pid` |
| q3 | For everyone near T: last sighting before T and first after T | `T`, `W` |
| q4 | Visits and stay-time per path (data-quality overview) | - |
| q5 | Walking-time bounds between paths, hidden-path routes first | - |
| q6 | Candidate table of one incident with the feasibility decision | `incident_id`, `W` |
| q7 | Counts per feasibility class for one incident | `incident_id`, `W` |

In psql: `\set incident_id 5` `\set W 900` `\i sql/queries/q6_incident_candidates.sql` (or `psql -v incident_id=5 -v W=900 -f ...`).
These files were run both through `run_query.py` and through the `psql` command line; results are identical.

### Feasibility classes (`incident_candidates.status`)
`possible` (could have crossed the hidden path in the time available - **these go to Week 6 for a probability**) |
`impossible: reached too fast` (arrived on the other side sooner than the fastest crossing) |
`same path before and after` | `no route through hidden path` | `no sighting before` / `no sighting after` |
`pending: may be inside, not seen again yet` (last seen on a path leading into the hidden path, no later sighting yet).

## 4. Results (simulated data, 1200 people, 60 incidents each)
Camera presets: **easy** = 85% detection, 3 s sampling; **hard** = 35% detection, 10 s sampling, much more walking-speed variation and pausing.

**Timelines** (`check_timelines.py`, `sql/eval/e2`): the SQL timeline is **identical** to the independent pandas implementation (4,364 / 4,398 visits).
Share of true stays on camera paths that appear in the timeline: **100%** (easy), **98.2 - 99.8%** (hard).

**Feasibility filter** (`sql/eval/e1`, all candidates of all incidents):
| | true occupants kept as `possible` | true occupants wrongly dropped | non-occupants still `possible` |
|---|---|---|---|
| easy | 845 / 845 | 0 | 14 |
| hard | 897 of 898 (the 898th is `pending`) | 0 | 156 (of 1,053) |

So after Week 5 the question "who could have been inside P4?" is narrowed from about 200 candidates per incident to about 17 (hard) or 14 (easy) people, and no true occupant is lost.
Telling the 156 non-occupants apart from the 897 occupants is exactly the job of the Week 6 probability model.

### Finding: the speed limit is a trade-off
The rule "arrived too fast to have crossed" uses the maximum walking speed `walking_speed_bounds_mps[1]` from `configs/path_graph.json`.
With 2.5 m/s, 3 true occupants (fast walkers crossing P4 at 3.2 - 4.0 m/s) were wrongly dropped on the hard data. Sweep on the hard dataset:

| v_max (m/s) | occupants kept | wrongly dropped | non-occupants kept | non-occupants dropped |
|---|---|---|---|---|
| 2.0 | 890 | 7 | 138 | 92 |
| 2.5 | 894 | 3 | 143 | 87 |
| **3.0 (default)** | **897** | **0** | 156 | 74 |
| 4.0 | 897 | 0 | 172 | 58 |
| 5.0 | 897 | 0 | 200 | 30 |

Pick v_max from your own participants (include someone walking fast). Too low loses real people; too high lets in more impossible cases that Week 6 must then reject.
To change it without reloading: `UPDATE run_meta SET value='3.5' WHERE key='v_max_mps'; SELECT build_route_times();`

## 5. Using real footage
After Weeks 3-4 you have `global_timeline.csv`. Load it with
`python load_data.py --source tracks --tracks path\to\global_timeline.csv`
(each track becomes sightings every 2 s; change with `--sample-s`). Then run `python check_timelines.py` and the same queries.
Incidents for real footage: `INSERT INTO incidents (affected_path_id, start_time) VALUES ('P4', 1760436600.0);` (epoch seconds of your chosen timestamp), then `q6` with that incident's id.
Camera ids in the file must match `configs/path_graph.json` (C1, C2, C3, C5, C6, C7).

## 6. Known limits (carried into Week 6)
- People with a sighting before T but none after (`pending`) cannot be scored yet; Week 6 handles only people seen on both sides. Right after a real incident this matters, because people still inside have not reappeared.
- The timeline only knows what cameras saw; a stay shorter than the sampling gap can be missed (1-2% on the hard data).
- Ground-truth tables are for evaluation only; a test checks that no inference SQL mentions them.

## 7. Hand-off to Week 6
Week 6 reads `incident_candidates(...)` rows with `status = 'possible'`: for each person it has the anchor sightings (a, ta) and (b, tb), the gap, and the fastest crossing time,
and turns them into **P(inside the hidden path at T)** using fitted travel-time distributions.
