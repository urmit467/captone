"""Load observations into PostgreSQL, then build timelines and route times with the SQL functions.

Simulated data (from simulate.py):
  python load_data.py --source sim --data-dir data/hard
Real footage (global_timeline.csv from Week 4):
  python load_data.py --source tracks --tracks ../incident_recon_weeks1-4/week4_reid/global_timeline.csv
Each track becomes sightings at its start, its end and every --sample-s seconds in between.
"""
import argparse, io, json, os
import numpy as np, pandas as pd
from dbutil import DEFAULT_DSN, connect


def copy_df(conn, table, df, columns):
    buf = io.StringIO(); df[columns].to_csv(buf, index=False, header=False); buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)", buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--source", choices=["sim", "tracks"], default="sim")
    ap.add_argument("--data-dir", default="data/easy")
    ap.add_argument("--tracks", default=None, help="global_timeline.csv (for --source tracks)")
    ap.add_argument("--sample-s", type=float, default=2.0, help="sighting spacing inside a real track")
    ap.add_argument("--graph", default="configs/path_graph.json")
    a = ap.parse_args()

    g = json.load(open(a.graph))
    cam_of = {p: d["camera"] for p, d in g["paths"].items()}
    conn = connect(a.dsn)
    with conn.cursor() as cur:   # clear data tables (schema and functions stay)
        cur.execute("TRUNCATE predictions, risk_scores, incidents, trajectories, visits, route_times, observations, persons, "
                    "cameras, path_edges, paths, run_meta, ground_truth_presence, ground_truth_labels RESTART IDENTITY CASCADE")
        cur.executemany("INSERT INTO paths VALUES (%s,%s,%s)", [(p, d["length_m"], bool(d["camera"])) for p, d in g["paths"].items()])
        cur.executemany("INSERT INTO path_edges VALUES (%s,%s)", g["edges"])
        cur.executemany("INSERT INTO cameras VALUES (%s,%s)", [(d["camera"], p) for p, d in g["paths"].items() if d["camera"]])
        meta = dict(hidden_path=g["hidden_path"], v_min_mps=g["walking_speed_bounds_mps"][0], v_max_mps=g["walking_speed_bounds_mps"][1],
                    time_slack_s=g["time_slack_s"], source=a.source)

        if a.source == "sim":
            sm = json.load(open(os.path.join(a.data_dir, "meta.json")))
            meta.update(p_detect=sm["p_detect"], obs_period_s=sm["obs_period"], preset=sm["preset"])
            obs = pd.read_csv(os.path.join(a.data_dir, "observations.csv"))
            persons = pd.DataFrame({"id": np.sort(obs.person_id.unique())})
            persons["label"] = persons.id.map(lambda i: f"Person_{i + 1:04d}")
            obs = obs.rename(columns={"timestamp": "ts"}); obs["camera_id"] = obs.path_id.map(cam_of); obs["source"] = "sim"
        else:
            tr = pd.read_csv(a.tracks)
            meta.update(p_detect=0.95, obs_period_s=a.sample_s)
            persons = pd.DataFrame({"id": np.sort(tr.global_person_id.unique())}); persons["label"] = persons.id.map(lambda i: f"Person_{i + 1:04d}")
            rows = []
            for r in tr.itertuples():
                ts = np.unique(np.append(np.arange(r.start_time, r.end_time, a.sample_s), r.end_time))
                rows += [(r.global_person_id, r.camera_id, r.path_id, t, "track") for t in ts]
            obs = pd.DataFrame(rows, columns=["person_id", "camera_id", "path_id", "ts", "source"])
        cur.executemany("INSERT INTO run_meta VALUES (%s,%s)", [(k, str(v)) for k, v in meta.items()])
    copy_df(conn, "persons", persons, ["id", "label"])
    copy_df(conn, "observations", obs.sort_values("ts"), ["person_id", "camera_id", "path_id", "ts", "source"])
    print(f"loaded {len(persons)} persons, {len(obs)} observations")

    if a.source == "sim":    # incidents + evaluation-only truth
        inc = pd.read_csv(os.path.join(a.data_dir, "incidents.csv"))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO incidents (id, affected_path_id, start_time, status) VALUES (%s,%s,%s,'simulated')",
                            [(int(r.incident_id) + 1, g["hidden_path"], float(r.time)) for r in inc.itertuples()])
            cur.execute("SELECT setval(pg_get_serial_sequence('incidents','id'), (SELECT MAX(id) FROM incidents))")
        seg = pd.read_csv(os.path.join(a.data_dir, "segments.csv")).rename(columns={"enter": "start_time", "exit": "end_time"})
        copy_df(conn, "ground_truth_presence", seg, ["person_id", "path_id", "start_time", "end_time"])
        lab = pd.read_csv(os.path.join(a.data_dir, "labels.csv")); lab["incident_id"] += 1
        copy_df(conn, "ground_truth_labels", lab, ["incident_id", "person_id", "in_hidden"])
        print(f"loaded {len(inc)} incidents (ids 1..{len(inc)}) and the evaluation-only ground truth")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT build_visits()"); nv = cur.fetchone()[0]
        cur.execute("SELECT build_route_times()"); nr = cur.fetchone()[0]
    conn.commit()
    print(f"built {nv} visits and {nr} route-time rows with SQL")


if __name__ == "__main__":
    main()
