"""Standalone simulator with difficulty presets. Writes data/<preset>/ :
  observations.csv  person_id,path_id,timestamp        (what cameras saw - the only thing inference may use)
  segments.csv      person_id,path_id,enter,exit       (TRUTH incl. hidden path - evaluation only)
  incidents.csv     incident_id,time,n_inside_total
  labels.csv        incident_id,person_id,in_hidden    (TRUTH - evaluation only)
  meta.json         camera parameters the model is allowed to know (p_detect, obs_period)
"""
import json, os
import numpy as np, pandas as pd

PRESETS = {   # camera sampling / walking variability. "hard" is the one that stresses the method.
    "easy":   dict(p_detect=0.85, obs_period=3.0,  jitter=0.5, speed_sigma=0.20, segment_sigma=0.10, dwell_prob=0.15, dwell_mean=30),
    "medium": dict(p_detect=0.60, obs_period=5.0,  jitter=1.0, speed_sigma=0.30, segment_sigma=0.15, dwell_prob=0.30, dwell_mean=60),
    "hard":   dict(p_detect=0.35, obs_period=10.0, jitter=2.0, speed_sigma=0.40, segment_sigma=0.20, dwell_prob=0.50, dwell_mean=90),
}
TRANSITIONS = {"P1": {"P2": 1.0}, "P2": {"P3": 1.0}, "P3": {"P4": 0.6, "P5": 0.4},
               "P4": {"P5": 0.35, "P6": 0.35, "P7": 0.30}, "P5": {"P6": 0.6, "EXIT": 0.4},
               "P6": {"P7": 0.2, "EXIT": 0.8}, "P7": {"EXIT": 1.0}}
ENTRY = {"P1": 0.5, "P2": 0.2, "P3": 0.3}
SPEED = {"P4": 1.2}; DEFAULT_SPEED = 1.3


class _Graph:
    def __init__(self, path):
        raw = json.load(open(path))
        self.hidden = raw["hidden_path"]; self.edges = raw["edges"]
        self.length = {p: float(d["length_m"]) for p, d in raw["paths"].items()}
        self.has_camera = {p: bool(d["camera"]) for p, d in raw["paths"].items()}


def _person(pid, t0, rng, g, c):
    mult = rng.lognormal(0, c["speed_sigma"])
    path = rng.choice(list(ENTRY), p=list(ENTRY.values()))
    t, segs = t0, []
    while path != "EXIT":
        v = SPEED.get(path, DEFAULT_SPEED) * mult * rng.lognormal(0, c["segment_sigma"])
        dur = g.length[path] / v
        if rng.random() < c["dwell_prob"]:
            dur += rng.exponential(c["dwell_mean"])
        segs.append((pid, path, t, t + dur)); t += dur
        nx = TRANSITIONS[path]; path = rng.choice(list(nx), p=list(nx.values()))
    return segs


def _observe(segs, rng, g, c):
    out = []
    for pid, path, a, b in segs:
        if not g.has_camera[path]:
            continue
        for s in np.arange(a + rng.uniform(0, c["obs_period"]), b, c["obs_period"]):
            if rng.random() < c["p_detect"]:
                out.append((pid, path, s + rng.normal(0, c["jitter"])))
    return out


def generate(graph_path, preset="easy", n_persons=1500, duration_s=4 * 3600, n_incidents=200, seed=42, out_dir=None, window_s=900.0):
    g, c = _Graph(graph_path), PRESETS[preset]
    for p, nx in TRANSITIONS.items():
        for q in nx:
            assert q == "EXIT" or [p, q] in g.edges, f"simulator transition {p}->{q} not in graph"
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, duration_s, n_persons))
    segs, obs = [], []
    for pid, t0 in enumerate(starts):
        s = _person(pid, t0, rng, g, c); segs += s; obs += _observe(s, rng, g, c)
    seg = pd.DataFrame(segs, columns=["person_id", "path_id", "enter", "exit"])
    ob = pd.DataFrame(obs, columns=["person_id", "path_id", "timestamp"]).sort_values("timestamp").reset_index(drop=True)
    times = np.sort(rng.uniform(1800, duration_s - 1800, n_incidents))
    hid = seg[seg.path_id == g.hidden]
    ts, pids = ob.timestamp.to_numpy(), ob.person_id.to_numpy()
    inc, lab = [], []
    for k, T in enumerate(times):
        inside = set(hid[(hid.enter <= T) & (hid["exit"] >= T)].person_id)
        i0, i1 = np.searchsorted(ts, [T - window_s, T + window_s])
        for pid in np.unique(pids[i0:i1]):
            lab.append((k, int(pid), int(pid in inside)))
        inc.append((k, float(T), len(inside)))
    out = {"observations": ob, "segments": seg, "incidents": pd.DataFrame(inc, columns=["incident_id", "time", "n_inside_total"]),
           "labels": pd.DataFrame(lab, columns=["incident_id", "person_id", "in_hidden"])}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k, df in out.items():
            df.to_csv(os.path.join(out_dir, k + ".csv"), index=False)
        json.dump(dict(preset=preset, p_detect=c["p_detect"], obs_period=c["obs_period"], jitter=c["jitter"], n_persons=n_persons,
                       window_s=window_s, seed=seed), open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate a simulated dataset into data/<preset>/")
    ap.add_argument("--preset", default="easy", choices=list(PRESETS))
    ap.add_argument("--graph", default="configs/path_graph.json")
    ap.add_argument("--n-persons", type=int, default=1500)
    ap.add_argument("--n-incidents", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"data/{a.preset}"
    d = generate(a.graph, a.preset, a.n_persons, n_incidents=a.n_incidents, seed=a.seed, out_dir=out)
    print(f"{a.preset}: {len(d['observations'])} observations, {a.n_persons} persons, {len(d['incidents'])} incidents -> {out}/")
    print(f"mean people inside hidden path per incident: {d['incidents'].n_inside_total.mean():.1f}")
