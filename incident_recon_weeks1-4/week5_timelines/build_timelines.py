
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import connect, SIM_T0
from graph import PathGraph
from timeline import build_segments, build_transitions
from timeline_store import load_sightings, save_segments

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--graph", default=os.path.join(HERE, "..", "week1_design", "path_graph.json"))
    ap.add_argument("--t0", default=str(SIM_T0))
    ap.add_argument("--max-gap", type=float, default=60.0)
    ap.add_argument("--extra-dwell", type=float, default=0.0, help="extra seconds allowed for lingering in hidden paths (see graph.py)")
    ap.add_argument("--export", default=None)
    a = ap.parse_args()

    db, t0 = connect(a.db), pd.Timestamp(a.t0)
    g = PathGraph(a.graph, extra_dwell_s=a.extra_dwell)
    sight = load_sightings(db, t0)
    seg = build_segments(sight, max_same_path_gap_s=a.max_gap)
    tr = build_transitions(seg, g)
    save_segments(db, seg, t0)
    print(f"{len(sight)} sightings -> {len(seg)} segments for {seg.person_id.nunique()} persons -> trajectories table")
    print("path visits:", seg.path_id.value_counts().sort_index().to_dict())
    print("transitions:", tr.kind.value_counts().to_dict())
    bad = tr[tr.kind == "impossible"]
    if len(bad):
        print(f"WARNING: {len(bad)} transitions fit no route (Re-ID error, clock offset, or a missed segment) - inspect with --export")
    if a.export:
        os.makedirs(a.export, exist_ok=True)
        seg.to_csv(f"{a.export}/segments.csv", index=False)
        tr.to_csv(f"{a.export}/transitions.csv", index=False)


if __name__ == "__main__":
    main()
