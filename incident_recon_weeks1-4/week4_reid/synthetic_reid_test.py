"""Re-ID matcher test on the Week 2 simulator (no video needed).

Each simulated camera visit becomes a "track". Appearance embeddings are synthetic: every person has a random vector, blended
with a shared "clothing cluster" vector (alpha = how alike people in a cluster look) plus per-track noise.
Compares matching WITH vs WITHOUT the topology/time gating -> shows why the path graph helps Re-ID.

  python synthetic_reid_test.py --n-persons 800 --noise 0.7 --alpha 0.6 --clusters 6
"""
import argparse, os, sys
import numpy as np, pandas as pd
from camera_topology import CameraTopology
from match_tracks import match_tracks
from evaluate_reid import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))


def visits(obs):
    obs = obs.sort_values(["person_id", "timestamp"])
    rows = []
    for pid, g in obs.groupby("person_id", sort=False):
        p, t = g.path_id.to_numpy(), g.timestamp.to_numpy()
        s = 0
        for i in range(1, len(p) + 1):
            if i == len(p) or p[i] != p[s]:
                rows.append((pid, p[s], t[s], t[i - 1])); s = i
    return pd.DataFrame(rows, columns=["person_id", "path_id", "start_time", "end_time"])


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default=os.path.join(HERE, "../week2_simulator/data/observations.csv"))
    ap.add_argument("--graph", default=os.path.join(HERE, "../week1_design/path_graph.json"))
    ap.add_argument("--n-persons", type=int, default=800)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.7, help="per-track embedding noise (norm)")
    ap.add_argument("--alpha", type=float, default=0.6, help="similarity of people within a clothing cluster (0..1)")
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--max-dist", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    topo = CameraTopology(a.graph)
    obs = pd.read_csv(a.obs)
    obs = obs[obs.person_id < a.n_persons]
    tr = visits(obs)
    tr["camera_id"] = tr.path_id.map(topo.path_cam)
    tr = tr.sort_values("start_time").reset_index(drop=True)

    cluster = unit(rng.normal(size=(a.clusters, a.dim)))
    pvec = unit(a.alpha * cluster[np.arange(a.n_persons) % a.clusters] + (1 - a.alpha) * unit(rng.normal(size=(a.n_persons, a.dim))))
    emb = unit(pvec[tr.person_id.to_numpy()] + a.noise * rng.normal(size=(len(tr), a.dim)) / np.sqrt(a.dim))
    print(f"{len(tr)} tracks from {a.n_persons} persons | noise={a.noise} alpha={a.alpha} clusters={a.clusters}\n")

    rows = {}
    for name, tp in (("appearance only + time order", None), ("appearance + topology gating", topo)):
        res = match_tracks(tr, emb, tp, max_dist=a.max_dist)
        rows[name] = evaluate(res.global_id, tr.person_id, tr.start_time)
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
