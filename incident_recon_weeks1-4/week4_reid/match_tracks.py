"""Week 4b: link single-camera tracks into global person identities.

A track may join an existing identity only if
  (1) the camera transition and time gap are physically possible (camera_topology.py), and
  (2) its appearance is close to the identity's gallery (cosine distance <= max_dist).
Tracks starting within `batch_s` seconds are assigned together with the Hungarian algorithm (one identity per track).
Otherwise the track starts a new identity.

  python match_tracks.py --tracks ../week3_detection_tracking/output/track_summary.csv \
      --embeddings track_embeddings.npz --graph ../week1_design/path_graph.json --out global_timeline.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from camera_topology import CameraTopology

BIG = 1e6


def match_tracks(tracks, emb, topo=None, max_dist=0.35, batch_s=5.0, horizon_s=900.0, jitter_s=2.0):
    """tracks: DataFrame(camera_id, start_time, end_time); emb: (N, D) L2-normalised rows aligned with tracks.
    topo=None -> no camera/time-gap gating (only 'earlier track must have ended' within horizon_s): ablation baseline.
    Returns tracks + global_id + match_dist."""
    tracks = tracks.reset_index(drop=True)
    n, D = emb.shape
    start, end = tracks.start_time.to_numpy(float), tracks.end_time.to_numpy(float)
    cam = tracks.camera_id.map(topo.idx).to_numpy() if topo is not None else np.zeros(n, int)
    horizon = topo.max_gap if topo is not None else horizon_s

    gsum = np.zeros((n, D)); last_cam = np.zeros(n, int); last_end = np.zeros(n)
    alive = np.zeros(n, bool); n_id = 0
    gid = np.full(n, -1); dist_out = np.full(n, np.nan)

    order = np.argsort(start, kind="stable")
    i = 0
    while i < n:
        j = i
        while j < n and start[order[j]] - start[order[i]] <= batch_s:
            j += 1
        b = order[i:j]; i = j
        alive[:n_id] &= last_end[:n_id] >= start[b].min() - horizon - 1.0     # forget identities that are too old
        act = np.flatnonzero(alive[:n_id])
        cost = np.full((len(b), len(act) + len(b)), BIG)
        cost[np.arange(len(b)), len(act) + np.arange(len(b))] = max_dist       # "start a new identity" option
        if len(act):
            G = gsum[act] / np.maximum(np.linalg.norm(gsum[act], axis=1, keepdims=True), 1e-9)
            dist = 1.0 - emb[b] @ G.T
            gap = start[b][:, None] - last_end[act][None, :]
            if topo is not None:
                ok = topo.allowed(last_cam[act][None, :], cam[b][:, None], gap)
            else:
                ok = (gap >= -jitter_s) & (gap <= horizon)
            cost[:, :len(act)] = np.where(ok & (dist <= max_dist), dist, BIG)
        r, c = linear_sum_assignment(cost)
        for ri, ci in zip(r, c):
            t = b[ri]
            if ci < len(act) and cost[ri, ci] < BIG:
                k = act[ci]; dist_out[t] = cost[ri, ci]
            else:
                k = n_id; n_id += 1; alive[k] = True
            gid[t] = k
            gsum[k] += emb[t]; last_cam[k] = cam[t]; last_end[k] = max(last_end[k], end[t])
    out = tracks.copy()
    out["global_id"] = gid; out["match_dist"] = dist_out
    return out


def load_embeddings(npz_path, tracks):
    z = np.load(npz_path, allow_pickle=True)
    lut = {k: e for k, e in zip(z["keys"], z["emb"])}
    keys = tracks.camera_id.astype(str) + "|" + tracks.track_id.astype(int).astype(str)
    have = keys.isin(lut)
    if (~have).any():
        print(f"warning: {(~have).sum()} tracks have no embedding (no crops) and are skipped")
    tracks = tracks[have].reset_index(drop=True)
    return tracks, np.stack([lut[k] for k in keys[have]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--graph", default="../week1_design/path_graph.json")
    ap.add_argument("--max-dist", type=float, default=0.35, help="max cosine distance to accept a match")
    ap.add_argument("--no-topology", action="store_true", help="ablation: disable camera/time-gap gating")
    ap.add_argument("--out", default="global_timeline.csv")
    a = ap.parse_args()
    topo = CameraTopology(a.graph)
    tr, emb = load_embeddings(a.embeddings, pd.read_csv(a.tracks))
    res = match_tracks(tr, emb, None if a.no_topology else topo, max_dist=a.max_dist)
    res["path_id"] = res.camera_id.map(topo.cam_path)
    res["label"] = res.global_id.map(lambda k: f"Person_{k + 1:03d}")
    cols = ["global_id", "label", "camera_id", "path_id", "track_id", "start_time", "end_time", "match_dist"]
    res.sort_values(["global_id", "start_time"])[cols].rename(columns={"global_id": "global_person_id"}).to_csv(a.out, index=False)
    print(f"{len(res)} tracks -> {res.global_id.nunique()} global identities -> {a.out}")


if __name__ == "__main__":
    main()
