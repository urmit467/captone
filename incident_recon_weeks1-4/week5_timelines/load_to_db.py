"""Load observations into the database (Week 1 schema).

Simulated data (Week 2):
  python load_to_db.py sim  --db sqlite:///week5.db --data ../week2_simulator/data --reset
  python load_to_db.py sim  --db postgresql://recon:recon@localhost:5432/recon --data ../week2_simulator/data --reset
    add --with-ground-truth to also fill ground_truth_presence (evaluation only; build/query code never reads it)

Real footage (Week 3 + Week 4):
  python load_to_db.py real --db ... --tracks "../week3_detection_tracking/output/tracks/*.csv" \
        --timeline ../week4_reid/global_timeline.csv --reset

IDs: persons.id = Week-2 person_id + 1 (or Week-4 global_person_id + 1), label 'Person_001'...; incidents.id = Week-2 incident_id + 1.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import connect, secs_to_ts, SIM_T0, EPOCH0

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "..", "week1_design", "schema.sql")


def init_schema(db, reset):
    if reset:
        for t in ["trajectories", "risk_scores", "predictions", "ground_truth_presence", "observations", "tracks",
                  "incidents", "cameras", "path_edges", "paths", "persons"]:
            db.execute(f"DROP TABLE IF EXISTS {t}" + (" CASCADE" if db.kind == "pg" else ""))
    db.run_script(open(SCHEMA).read())


def load_network(db, graph_json):
    with open(graph_json) as f:
        g = json.load(f)
    db.insert_df("paths", pd.DataFrame([dict(id=p, length_m=d["length_m"], has_camera=d["camera"] is not None)
                                        for p, d in g["paths"].items()]))   # capacity / avg_speed_mps left NULL on purpose:
    db.insert_df("path_edges", pd.DataFrame(g["edges"], columns=["from_path", "to_path"]))   # Week 6 FITS speeds from data
    db.insert_df("cameras", pd.DataFrame([dict(id=d["camera"], path_id=p, clock_offset_s=0.0)
                                          for p, d in g["paths"].items() if d["camera"]]))
    return {p: d["camera"] for p, d in g["paths"].items()}


def load_persons(db, ids):
    ids = sorted(set(int(i) for i in ids))
    db.insert_df("persons", pd.DataFrame({"id": [i + 1 for i in ids], "label": [f"Person_{i + 1:03d}" for i in ids]}))
    db.reset_serial("persons")


def cmd_sim(a):
    db = connect(a.db)
    init_schema(db, a.reset)
    path_cam = load_network(db, a.graph)
    t0 = pd.Timestamp(a.t0)
    obs = pd.read_csv(f"{a.data}/observations.csv")
    load_persons(db, obs.person_id.unique())
    out = pd.DataFrame({"person_id": obs.person_id + 1, "camera_id": obs.path_id.map(path_cam), "path_id": obs.path_id,
                        "ts": secs_to_ts(obs.timestamp, t0)})
    db.insert_df("observations", out)
    inc = pd.read_csv(f"{a.data}/incidents.csv")
    db.insert_df("incidents", pd.DataFrame({"id": inc.incident_id + 1, "affected_path_id": "P4",
                                            "start_time": secs_to_ts(inc.time, t0), "severity": 1.0, "status": "open"}))
    db.reset_serial("incidents")
    if a.with_ground_truth:
        gt = pd.read_csv(f"{a.data}/segments.csv")
        db.insert_df("ground_truth_presence", pd.DataFrame({"person_id": gt.person_id + 1, "path_id": gt.path_id,
                                                            "start_time": secs_to_ts(gt.enter, t0), "end_time": secs_to_ts(gt.exit, t0)}))
    print(f"loaded {len(out)} observations, {obs.person_id.nunique()} persons, {len(inc)} incidents"
          + (", ground truth" if a.with_ground_truth else "") + f" -> {a.db}")


def cmd_real(a):
    db = connect(a.db)
    init_schema(db, a.reset)
    path_cam = load_network(db, a.graph)
    cam_path = {c: p for p, c in path_cam.items() if c}
    tl = pd.read_csv(a.timeline)                                     # global_person_id,label,camera_id,path_id,track_id,start_time,end_time,match_dist
    load_persons(db, tl.global_person_id.unique())
    frames = pd.concat([pd.read_csv(f) for p in a.tracks for f in glob.glob(p)], ignore_index=True)
    n_fr = frames.groupby(["camera_id", "track_id"]).size().rename("n_frames").reset_index()
    tl = tl.merge(n_fr, on=["camera_id", "track_id"], how="left").reset_index(drop=True)
    tl["tid"] = np.arange(1, len(tl) + 1)
    t0 = pd.Timestamp(a.t0)
    db.insert_df("tracks", pd.DataFrame({
        "id": tl.tid, "camera_id": tl.camera_id, "local_track_id": tl.track_id,
        "start_time": secs_to_ts(tl.start_time, t0), "end_time": secs_to_ts(tl.end_time, t0),
        "n_frames": tl.n_frames, "person_id": tl.global_person_id + 1, "reid_score": 1.0 - tl.match_dist}))
    db.reset_serial("tracks")
    fr = frames.merge(tl[["camera_id", "track_id", "global_person_id", "tid"]], on=["camera_id", "track_id"], how="inner")
    fr = fr.iloc[:: a.obs_stride]
    db.insert_df("observations", pd.DataFrame({
        "person_id": fr.global_person_id + 1, "track_id": fr.tid, "camera_id": fr.camera_id,
        "path_id": fr.camera_id.map(cam_path), "ts": secs_to_ts(fr.timestamp, t0),
        "x": fr.foot_x, "y": fr.foot_y, "confidence": fr.confidence}))
    print(f"loaded {len(tl)} tracks, {len(fr)} observations, {tl.global_person_id.nunique()} persons -> {a.db}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("sim", cmd_sim), ("real", cmd_real)]:
        p = sub.add_parser(name)
        p.add_argument("--db", required=True)
        p.add_argument("--graph", default=os.path.join(HERE, "..", "week1_design", "path_graph.json"))
        p.add_argument("--reset", action="store_true", help="drop and recreate all tables first")
        p.set_defaults(fn=fn)
        if name == "sim":
            p.add_argument("--data", default=os.path.join(HERE, "..", "week2_simulator", "data"))
            p.add_argument("--t0", default=str(SIM_T0))
            p.add_argument("--with-ground-truth", action="store_true")
        else:
            p.add_argument("--tracks", nargs="+", required=True, help="Week 3 per-camera track CSVs (globs ok)")
            p.add_argument("--timeline", required=True, help="Week 4 global_timeline.csv")
            p.add_argument("--t0", default=str(EPOCH0))
            p.add_argument("--obs-stride", type=int, default=1, help="keep every n-th detection (25 fps video is dense)")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
