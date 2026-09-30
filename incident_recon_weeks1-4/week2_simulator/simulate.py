"""Simulator: people walk the path graph; cameras only see observable paths.
Outputs (data/):
  observations.csv  person_id, path_id, timestamp        <- what the algorithm may use
  segments.csv      person_id, path_id, enter, exit      <- hidden ground truth
  incidents.csv     incident_id, time
  labels.csv        incident_id, person_id, in_hidden    <- ground truth for evaluation
"""
import numpy as np, pandas as pd
import config as C


def simulate_person(pid, t0, rng):
    speed_mult = rng.lognormal(0, C.SPEED_SIGMA)
    path = rng.choice(list(C.ENTRY), p=list(C.ENTRY.values()))
    t, segs = t0, []
    while path != "EXIT":
        p = C.PATHS[path]
        v = p["speed"] * speed_mult * rng.lognormal(0, C.SEGMENT_SIGMA)
        dur = p["length"] / v
        if rng.random() < C.DWELL_PROB:
            dur += rng.exponential(C.DWELL_MEAN_S)
        segs.append((pid, path, t, t + dur))
        t += dur
        nxt = C.TRANSITIONS[path]
        path = rng.choice(list(nxt), p=list(nxt.values()))
    return segs


def observe(segs, rng):
    obs = []
    for pid, path, a, b in segs:
        if not C.PATHS[path]["camera"]:
            continue
        for s in np.arange(a + rng.uniform(0, C.OBS_PERIOD_S), b, C.OBS_PERIOD_S):
            if rng.random() < C.P_DETECT:
                obs.append((pid, path, s + rng.normal(0, C.TIME_JITTER_S)))
    return obs


def main():
    rng = np.random.default_rng(C.SEED)
    starts = np.sort(rng.uniform(0, C.SIM_DURATION_S, C.N_PERSONS))
    segs, obs = [], []
    for pid, t0 in enumerate(starts):
        s = simulate_person(pid, t0, rng)
        segs += s
        obs += observe(s, rng)

    seg_df = pd.DataFrame(segs, columns=["person_id", "path_id", "enter", "exit"])
    obs_df = pd.DataFrame(obs, columns=["person_id", "path_id", "timestamp"]).sort_values("timestamp")

    # incidents away from the edges of the simulation
    times = np.sort(rng.uniform(1800, C.SIM_DURATION_S - 1800, C.N_INCIDENTS))
    inc_df = pd.DataFrame({"incident_id": range(len(times)), "time": times})

    # ground-truth labels: persons with any observation within the context window
    labels = []
    hid = seg_df[seg_df.path_id == C.HIDDEN_PATH]
    for _, inc in inc_df.iterrows():
        T = inc.time
        w = obs_df[(obs_df.timestamp > T - C.CONTEXT_WINDOW_S) & (obs_df.timestamp < T + C.CONTEXT_WINDOW_S)]
        inside = set(hid[(hid.enter <= T) & (hid["exit"] >= T)].person_id)
        for pid in w.person_id.unique():
            labels.append((int(inc.incident_id), pid, int(pid in inside)))
        # persons inside P4 but with no observation in window are unreachable: reported below
    lab_df = pd.DataFrame(labels, columns=["incident_id", "person_id", "in_hidden"])

    seg_df.to_csv("data/segments.csv", index=False)
    obs_df.to_csv("data/observations.csv", index=False)
    inc_df.to_csv("data/incidents.csv", index=False)
    lab_df.to_csv("data/labels.csv", index=False)

    print(f"persons={C.N_PERSONS} observations={len(obs_df)} incidents={len(inc_df)}")
    print(f"share of persons that ever enter {C.HIDDEN_PATH}: {seg_df[seg_df.path_id==C.HIDDEN_PATH].person_id.nunique()/C.N_PERSONS:.2f}")
    print(f"label rows={len(lab_df)}, positives={lab_df.in_hidden.sum()} ({lab_df.in_hidden.mean():.1%})")
    print(f"mean people inside {C.HIDDEN_PATH} per incident: {lab_df.groupby('incident_id').in_hidden.sum().mean():.2f}")


if __name__ == "__main__":
    main()
