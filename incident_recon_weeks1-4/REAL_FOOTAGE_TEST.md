# Testing the whole pipeline on real footage (Weeks 1-5)

Windows PowerShell, no Docker. Start from the folder that contains week1_design ... incident_recon_week5. Activate the venv first (`.venv\Scripts\activate`).
Keep ONE time convention everywhere: ISO times without a zone are treated as UTC (Week 3 `--start-time`, route sheet, incident times).

## A. Before recording (Week 1 recording_plan.md, short version)
1. Layout: 5-7 connected paths, one camera per observable path (C1 C2 C3 C5 C6 C7), NO camera on P4. Edit `path_graph.json` (lengths, edges) to match the real place.
2. 10-20 consenting participants, clearly different clothing colours. Ids Person_001..; no names next to footage.
3. Clock sync: before and after each session show a running phone clock to every camera. Write down each camera's offset (seconds to ADD to match the reference clock).
4. File names: `C3_2026-10-14T10-00-00.mp4` (camera + wall-clock start).
5. Scripted runs (>= 10, hold out 3-4 you never tune on): vary speed, one person pausing in P4, someone taking the bypass P3->P5, people entering close together.
6. Ground truth: a person with a stopwatch at the P4 entrance/exit logs, for every participant and path INCLUDING P4, the enter and exit time.

## B. Files you create by hand
`route_sheet.csv` (true presence, all paths incl. P4; times ISO or epoch):
```
person,path_id,enter,exit
Person_001,P3,2026-10-14T10:21:00,2026-10-14T10:24:10
Person_001,P4,2026-10-14T10:24:10,2026-10-14T10:34:40
Person_001,P5,2026-10-14T10:34:40,2026-10-14T10:36:00
```
`reid_labels.csv` (hand label for ~50-100 tracks; look at the crops in week3 output/crops):
```
camera_id,track_id,person_id
C3,12,Person_001
C5,4,Person_001
```
The same file feeds Week 4 `evaluate_reid.py` and the Week 5 ground-truth helper.

## C. Run
```powershell
# Week 3: all cameras of one session + track_summary.csv
cd week3_detection_tracking
python run_all_cameras.py --videos D:\session1 --offsets C1=0.4,C3=-0.2 --device cuda:0
#   first check with --dry-run; use --model yolo11s.pt on a GPU for better detection; watch an annotated video (--save-video) for ID switches

# Week 4: Re-ID + accuracy number
cd ..\week4_reid
python extract_embeddings.py --crops ../week3_detection_tracking/output/crops --backend hist
#   better with OSNet: --backend osnet --weights osnet_x1_0_market.pth --device cuda:0   (run sanity_check_embeddings.py first)
python match_tracks.py --tracks ../week3_detection_tracking/output/track_summary.csv --embeddings track_embeddings.npz --graph ../week1_design/path_graph.json --out global_timeline.csv
python evaluate_reid.py --timeline global_timeline.csv --labels reid_labels.csv
#   tune --max-dist on your labelled tracks (maximise pair-F1). Weak Re-ID? use --backend hist with distinct clothing.

# Week 5: database + timelines
cd ..\incident_recon_week5
$env:RECON_DSN = "postgresql://postgres:YOURPASSWORD@localhost:5432/recon"
python setup_db.py --reset
python load_data.py --source tracks --tracks ..\week4_reid\global_timeline.csv
python check_timelines.py

# Ground truth for evaluation (pick the artificial incident times yourself; several per session is fine)
python real_footage_ground_truth.py --timeline ..\week4_reid\global_timeline.csv --labels ..\week4_reid\reid_labels.csv --route route_sheet.csv --incidents 2026-10-14T10:30:00,2026-10-14T10:41:20 --dsn $env:RECON_DSN

# Results
python run_query.py sql/queries/q7_incident_summary.sql --p incident_id=1 --p W=900
python run_query.py sql/queries/q6_incident_candidates.sql --p incident_id=1 --p W=900
python run_query.py sql/eval/e1_feasibility_filter.sql --p W=900
python run_query.py sql/eval/e2_visit_recall.sql
```
Run the ground-truth helper AFTER `load_data.py` (load_data wipes those tables). Re-running load_data means re-running the helper.

## D. What to look at, and what usually goes wrong
| Check | Good sign | If not |
|---|---|---|
| `evaluate_reid.py` | high link precision, few merged ids | Re-ID is the weakest link: distinct clothing, tune `--max-dist`, OSNet weights |
| helper report | purity ~1.0, few fragmented participants | every fragmented/merged identity leaks into the Week 5 numbers |
| `check_timelines.py` | "OK - identical", 0 same-path repeats | camera clock offsets wrong, or Re-ID errors |
| e2 | most true stays on camera paths found | lower `--min-frames`/conf in Week 3, check camera coverage |
| e1 | true occupants never `impossible` | see below |

* True occupants wrongly `impossible`: a participant walked faster/slower than the bounds. Raise `v_max` or lower `v_min` without reloading:
  `UPDATE run_meta SET value='3.5' WHERE key='v_max_mps'; SELECT build_route_times();` (run in pgAdmin / psql). Someone pausing in P4 for more than about
  (300 m / v_min) seconds also fails the upper bound.
* `pending` rows: people last seen before T and not yet seen after T. Normal if the recording ends right after T; choose T so everyone has time to re-appear.
* Clocks: a 5-10 s offset between cameras shows up as wrong gaps between neighbouring paths. Redo the offsets from your sync clip.
* Keep the held-out runs untouched until the final evaluation.
