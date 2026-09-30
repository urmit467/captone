"""Which track can follow which? Builds allowed (camera_A -> camera_B, time gap) intervals from the path graph.

gap = start_time(track B) - end_time(track A).
For every simple route pa -> ... -> pb through the graph, intermediate paths add L metres of walking:
    gap in [ L / v_max - jitter ,  L / v_min + slack ]
Adjacent paths (L = 0) -> gap ~ 0.  Crossing the hidden path P4 (300 m) -> gap >= ~118 s.
This is what stops the Re-ID matcher from linking a person on C3 to someone on C7 four seconds later.
"""
import json
from collections import defaultdict
import numpy as np


class CameraTopology:
    def __init__(self, graph_path, allow_same_camera=True, jitter_s=2.0):
        g = json.load(open(graph_path))
        vmin, vmax = g["walking_speed_bounds_mps"]
        slack = g["time_slack_s"]
        length = {p: d["length_m"] for p, d in g["paths"].items()}
        self.cam_path = {d["camera"]: p for p, d in g["paths"].items() if d["camera"]}
        self.path_cam = {p: c for c, p in self.cam_path.items()}
        self.cams = sorted(self.cam_path)
        self.idx = {c: i for i, c in enumerate(self.cams)}
        adj = defaultdict(list)
        for a, b in g["edges"]:
            adj[a].append(b)

        def routes(a, b):
            out = []
            def dfs(n, seen, L):
                if n == b and len(seen) > 1:
                    out.append(L); return
                for q in adj[n]:
                    if q not in seen:
                        dfs(q, seen | {q}, L + (length[q] if q != b else 0))
            dfs(a, {a}, 0)
            return out

        iv = defaultdict(list)
        for ca in self.cams:
            for cb in self.cams:
                pa, pb = self.cam_path[ca], self.cam_path[cb]
                if ca == cb:
                    if allow_same_camera:                      # re-linking a track fragment after occlusion
                        iv[(ca, cb)].append((-jitter_s, 2 * slack))
                else:
                    for L in sorted(set(routes(pa, pb))):
                        iv[(ca, cb)].append((L / vmax - jitter_s, L / vmin + slack))
        K, n = max((len(v) for v in iv.values()), default=1), len(self.cams)
        self.lo = np.full((K, n, n), np.inf)
        self.hi = np.full((K, n, n), -np.inf)
        for (ca, cb), lst in iv.items():
            for k, (lo, hi) in enumerate(lst):
                self.lo[k, self.idx[ca], self.idx[cb]] = lo
                self.hi[k, self.idx[ca], self.idx[cb]] = hi
        self.max_gap = float(self.hi[np.isfinite(self.hi)].max())
        self.intervals = dict(iv)

    def allowed(self, i, j, gap):
        """i: index of camera of the earlier track, j: of the later track, gap in seconds (broadcastable arrays)."""
        ok = np.zeros(np.broadcast(i, j, gap).shape, bool)
        for k in range(self.lo.shape[0]):
            ok |= (gap >= self.lo[k][i, j]) & (gap <= self.hi[k][i, j])
        return ok


if __name__ == "__main__":
    import sys
    t = CameraTopology(sys.argv[1] if len(sys.argv) > 1 else "../week1_design/path_graph.json")
    for (a, b), v in sorted(t.intervals.items()):
        if v and a != b:
            print(a, "->", b, [(round(lo), round(hi)) for lo, hi in v])
