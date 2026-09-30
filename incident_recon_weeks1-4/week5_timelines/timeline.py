"""Observations -> per-person timelines.

Input  "sightings": DataFrame(person_id, path_id, t_start, t_end)   (times in seconds; any common origin)
         - simulator observations / per-frame detections: one row per detection, t_start == t_end
         - real Re-ID output (Week 4 global_timeline.csv): one row per track, t_start < t_end
Output segments:  DataFrame(person_id, seg_idx, path_id, first_seen, last_seen, n_obs)
         consecutive sightings on the same path are merged into one segment; a new segment starts when the path changes
         or when the same path is unseen for longer than max_same_path_gap_s.
         NOTE first_seen/last_seen are when the person was SEEN, so true entry/exit are slightly earlier/later
         (by up to one sampling period) - Week 6 models that explicitly.
Output transitions: one row per pair of consecutive segments, annotated with graph feasibility (see PathGraph).
"""
import numpy as np
import pandas as pd


def to_sightings(obs, t_col="timestamp"):
    """Point observations (person_id, path_id, <t_col>) -> sightings."""
    s = obs[["person_id", "path_id"]].copy()
    s["t_start"] = obs[t_col].astype(float)
    s["t_end"] = s["t_start"]
    return s


def _label_and_aggregate(s, max_same_path_gap_s):
    new_person = s.person_id != s.person_id.shift()
    # running maximum of t_end inside a person, so overlapping sightings never create a fake gap
    prev_end = s.groupby("person_id").t_end.transform(lambda x: x.cummax().shift())
    path_change = s.path_id != s.path_id.shift()
    long_gap = (s.t_start - prev_end) > max_same_path_gap_s
    s["seg_key"] = (new_person | path_change | long_gap).cumsum()
    seg = (s.groupby("seg_key", sort=True)
             .agg(person_id=("person_id", "first"), path_id=("path_id", "first"),
                  first_seen=("t_start", "min"), last_seen=("t_end", "max"), n_obs=("t_start", "size"))
             .reset_index())
    return seg


def build_segments(sightings, max_same_path_gap_s=60.0, flicker_max_obs=2, flicker_max_s=8.0):
    """Merge sightings into segments.
    Flicker removal: timestamp jitter can swap the order of sightings at a path boundary (P2 P2 P3 P2 P3 P3), which would create
    tiny fake segments and impossible backward moves.  A segment with <= flicker_max_obs sightings lasting <= flicker_max_s that
    is sandwiched between two segments on the SAME path (A B A) is relabelled to A, then everything is re-merged.
    (The network is a DAG, so a genuine A -> B -> A walk cannot happen.)"""
    s = sightings.sort_values(["person_id", "t_start", "t_end"], kind="stable").reset_index(drop=True)
    for _ in range(5):
        seg = _label_and_aggregate(s, max_same_path_gap_s)
        g = seg.groupby("person_id")
        prev_p, next_p = g.path_id.shift(1), g.path_id.shift(-1)
        flick = ((seg.n_obs <= flicker_max_obs) & ((seg.last_seen - seg.first_seen) <= flicker_max_s)
                 & prev_p.notna() & (prev_p == next_p) & (seg.path_id != prev_p))
        if not flick.any():
            break
        fix = dict(zip(seg.seg_key[flick], prev_p[flick]))
        m = s.seg_key.isin(fix)
        s.loc[m, "path_id"] = s.loc[m, "seg_key"].map(fix)
    seg["seg_idx"] = seg.groupby("person_id").cumcount()
    return seg[["person_id", "seg_idx", "path_id", "first_seen", "last_seen", "n_obs"]]


def build_transitions(segments, graph):
    """Consecutive segment pairs of each person with gap and route feasibility.
    kind: 'direct'  - only routes that avoid the hidden path fit the gap
          'hidden'  - only routes through the hidden path fit the gap
          'either'  - both fit (ambiguous: Week 6 turns this into a probability)
          'impossible' - no route fits (Re-ID error, clock error or a missed segment in between)"""
    a = segments.copy()
    nxt = a.groupby("person_id").shift(-1)
    t = pd.DataFrame({
        "person_id": a.person_id, "seg_idx": a.seg_idx,
        "from_path": a.path_id, "to_path": nxt.path_id,
        "last_seen": a.last_seen, "next_first_seen": nxt.first_seen,
    }).dropna(subset=["to_path"]).reset_index(drop=True)
    t["gap_s"] = t.next_first_seen - t.last_seen
    ok = [graph.gap_feasibility(f, p, g) if f != p else (False, False)
          for f, p, g in zip(t.from_path, t.to_path, t.gap_s)]
    t["direct_ok"] = [o[0] for o in ok]
    t["hidden_ok"] = [o[1] for o in ok]
    t["kind"] = np.select([t.direct_ok & t.hidden_ok, t.hidden_ok, t.direct_ok], ["either", "hidden", "direct"], "impossible")
    return t
