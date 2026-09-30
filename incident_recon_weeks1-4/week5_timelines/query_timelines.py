"""Timeline queries against the database.  Times: seconds since t0 (simulator) or ISO timestamps.

  python query_timelines.py --db sqlite:///week5.db person 84
  python query_timelines.py --db sqlite:///week5.db neighbors P4 --t 1850.7 --window 900 [--direction in|out|both] [--sql]
  python query_timelines.py --db sqlite:///week5.db context   P4 --t 1850.7 --window 900     # who could be inside P4 at T?
"""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import connect, SIM_T0
from graph import PathGraph
from timeline import build_transitions
from timeline_store import load_segments, parse_time, iso
from queries import person_timeline, seen_on_neighbors, incident_context

HERE = os.path.dirname(os.path.abspath(__file__))

NEIGHBOR_SQL = """
SELECT DISTINCT t.person_id, t.path_id, t.start_time, t.end_time
FROM trajectories t
JOIN path_edges e ON (e.to_path = t.path_id AND e.from_path = ?) OR (e.from_path = t.path_id AND e.to_path = ?)
WHERE t.source = 'observed' AND t.start_time <= ? AND t.end_time >= ?
ORDER BY t.person_id, t.start_time
"""   # plain-SQL twin of queries.seen_on_neighbors(direction='both'); the test suite checks the two agree


def seen_on_neighbors_sql(db, path, t_lo, t_hi, t0):
    """Same question as queries.seen_on_neighbors, answered by the database (t_lo/t_hi in seconds since t0)."""
    return db.fetch_df(NEIGHBOR_SQL, (path, path, iso(t_hi, t0), iso(t_lo, t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--graph", default=os.path.join(HERE, "..", "week1_design", "path_graph.json"))
    ap.add_argument("--t0", default=str(SIM_T0))
    ap.add_argument("--extra-dwell", type=float, default=0.0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("person"); p.add_argument("person_id", type=int, help="database persons.id")
    for n in ("neighbors", "context"):
        p = sub.add_parser(n)
        p.add_argument("path")
        p.add_argument("--t", required=True)
        p.add_argument("--window", type=float, default=900.0)
        if n == "neighbors":
            p.add_argument("--direction", default="both", choices=["in", "out", "both"])
            p.add_argument("--sql", action="store_true", help="answer with the SQL query instead of pandas")
    a = ap.parse_args()

    db, t0, g = connect(a.db), pd.Timestamp(a.t0), PathGraph(a.graph, extra_dwell_s=a.extra_dwell)
    seg = load_segments(db, t0)
    pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.2f}".format)
    if a.cmd == "person":
        print(person_timeline(seg, build_transitions(seg, g), a.person_id).to_string(index=False))
    else:
        T = parse_time(a.t, t0)
        if a.cmd == "neighbors":
            if a.sql:
                print(seen_on_neighbors_sql(db, a.path, T - a.window, T + a.window, t0).to_string(index=False))
            else:
                print(seen_on_neighbors(seg, g, a.path, T - a.window, T + a.window, a.direction).to_string(index=False))
        else:
            ctx = incident_context(seg, g, a.path, T, a.window)
            print(ctx.status.value_counts().to_string())
            print("\nhidden-feasible candidates (strongest evidence first):")
            c = ctx[ctx.status == "hidden_feasible"].sort_values(["hidden_only", "gap_s"], ascending=[False, True])
            print(c[["person_id", "before_path", "before_dt_s", "after_path", "after_dt_s", "gap_s", "hidden_only"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
