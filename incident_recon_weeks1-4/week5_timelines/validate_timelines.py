"""Week 5 validation on SIMULATED data (evaluation only - reads the hidden ground truth, which inference code never does).

  python validate_timelines.py --data ../week2_simulator/data --graph ../week1_design/path_graph.json

1. Segment quality   built observable segments vs true segments (same person + path): entry/exit timing error, missing/extra.
2. Incident context  for every simulated incident, run queries.incident_context and check that the people who were truly in the
                     hidden path at T are kept as 'hidden_feasible' (candidate recall) and how many non-affected people are
                     already excluded (candidate precision).  This is a FILTER, not the Week 6 probability model.
"""
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import PathGraph
from timeline import to_sightings, build_segments, build_transitions
from queries import incident_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../week2_simulator/data")
    ap.add_argument("--graph", default="../week1_design/path_graph.json")
    ap.add_argument("--window", type=float, default=900.0)
    ap.add_argument("--slack", type=float, default=None, help="override time_slack_s of the graph json")
    ap.add_argument("--max-gap", type=float, default=60.0, help="unseen seconds on the SAME path that start a new segment")
    ap.add_argument("--extra-dwell", type=float, default=0.0, help="extra seconds allowed for lingering inside hidden paths")
    a = ap.parse_args()

    g = PathGraph(a.graph, slack_s=a.slack, extra_dwell_s=a.extra_dwell)
    obs = pd.read_csv(f"{a.data}/observations.csv")
    seg = build_segments(to_sightings(obs), max_same_path_gap_s=a.max_gap)
    tr = build_transitions(seg, g)

    # ---- 1. segment quality --------------------------------------------------------------------------------------
    true = pd.read_csv(f"{a.data}/segments.csv")
    true_obs = true[true.path_id != g.hidden].copy()
    true_obs["k"] = true_obs.groupby(["person_id", "path_id"]).cumcount()
    seg["k"] = seg.groupby(["person_id", "path_id"]).cumcount()
    m = seg.merge(true_obs, on=["person_id", "path_id", "k"], how="outer", indicator=True)
    both = m[m._merge == "both"]
    print("== 1. segments ==")
    print(f"built {len(seg)}   true observable {len(true_obs)}   matched {len(both)}   only-built {(m._merge=='left_only').sum()}   only-true {(m._merge=='right_only').sum()}")
    de, dx = both.first_seen - both.enter, both.exit - both.last_seen
    print(f"first_seen - true_enter : median {de.median():.2f}s  p95 {de.quantile(.95):.2f}s  (positive = seen after entering)")
    print(f"true_exit - last_seen   : median {dx.median():.2f}s  p95 {dx.quantile(.95):.2f}s")
    print("transition kinds:", tr.kind.value_counts().to_dict())

    # ---- 2. incident context vs ground truth --------------------------------------------------------------------
    inc = pd.read_csv(f"{a.data}/incidents.csv")
    lab = pd.read_csv(f"{a.data}/labels.csv")
    rows = []
    for r in inc.itertuples():
        ctx = incident_context(seg, g, g.hidden, r.time, a.window)
        L = lab[lab.incident_id == r.incident_id]
        x = L.merge(ctx[["person_id", "status", "hidden_only"]], on="person_id", how="left")
        x["status"] = x.status.fillna("not_in_context")
        pos, neg = x[x.in_hidden == 1], x[x.in_hidden == 0]
        rows.append(dict(incident_id=r.incident_id, n_pos=len(pos), pos_kept=(pos.status == "hidden_feasible").sum(),
                         pos_status=pos.status.value_counts().to_dict(),
                         neg_kept=(neg.status == "hidden_feasible").sum(), n_neg=len(neg)))
    R = pd.DataFrame(rows)
    tp, fn = R.pos_kept.sum(), R.n_pos.sum() - R.pos_kept.sum()
    fp = R.neg_kept.sum()
    print("\n== 2. incident context (feasibility filter) over", len(R), "incidents ==")
    print(f"truly in P4 at T: {R.n_pos.sum()}   kept as hidden_feasible: {tp}   lost: {fn}   -> candidate recall {tp / max(tp + fn, 1):.4f}")
    print(f"non-affected people kept as hidden_feasible (false candidates): {fp}   -> candidate precision {tp / max(tp + fp, 1):.4f}")
    lost = {}
    for d in R.pos_status:
        for k, v in d.items():
            if k != "hidden_feasible":
                lost[k] = lost.get(k, 0) + v
    print("where the lost positives ended up:", lost or "none")
    print(f"mean per incident: {R.n_pos.mean():.1f} truly affected, {(R.pos_kept + R.neg_kept).mean():.1f} candidates")
    print("NOTE: the default simulator noise is mild (week2 README). Re-run with a harsher simulator config to see the filter degrade.")


if __name__ == "__main__":
    main()
