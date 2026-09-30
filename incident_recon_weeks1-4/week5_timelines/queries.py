"""Timeline queries used by the incident engine (Week 6/7).  All take the `segments` table from timeline.build_segments."""
import pandas as pd


def person_timeline(segments, transitions, person_id):
    """Readable per-person timeline: observed segments, with the transition to the next one."""
    s = segments[segments.person_id == person_id].sort_values("seg_idx")
    t = transitions[transitions.person_id == person_id].set_index("seg_idx")
    out = s.merge(t[["to_path", "gap_s", "kind"]], left_on="seg_idx", right_index=True, how="left")
    return out.rename(columns={"to_path": "next_path", "gap_s": "gap_to_next_s", "kind": "next_kind"})


def seen_on_neighbors(segments, graph, path, t_lo, t_hi, direction="both"):
    """'Who was seen on neighbours of `path` in [t_lo, t_hi]?'   direction: 'in' (paths you enter `path` from),
    'out' (paths you leave `path` to) or 'both'.  A segment counts if its [first_seen, last_seen] overlaps the window."""
    nb = set(graph.neighbors(path, direction))
    s = segments[segments.path_id.isin(nb) & (segments.first_seen <= t_hi) & (segments.last_seen >= t_lo)].copy()
    role = {p: "in" for p in graph.predecessors(path)}
    role.update({p: ("both" if p in role else "out") for p in graph.successors(path)})
    s["role"] = s.path_id.map(role)
    return s.sort_values(["person_id", "first_seen"]).reset_index(drop=True)


def incident_context(segments, graph, path, T, window_s=900.0):
    """One row per person who has any sighting within T +- window_s: what was seen just before T and just after T,
    and whether a route through the hidden `path` can explain the gap.  No probabilities yet (that is Week 6).

    status:
      seen_at_T        a camera saw the person on an observable path at T  -> cannot be in the hidden path
      no_before        nothing seen before T in the window (cannot tell how they got here)
      no_after         nothing seen after T in the window (yet) -> undecided, keep watching
      hidden_feasible  last-seen-before -> first-seen-after can be explained by a route through `path`
      not_feasible     no route through `path` fits the time gap (e.g. P5 seen sooner than the fastest P3->P4->P5 walk)
    hidden_only = True when the gap is explained ONLY by routes through the hidden path (strongest evidence)."""
    lo, hi = T - window_s, T + window_s
    seg = segments[(segments.last_seen >= lo) & (segments.first_seen <= hi)]
    at = seg[(seg.first_seen <= T) & (seg.last_seen >= T)].set_index("person_id")
    b = (seg[seg.last_seen < T].sort_values("last_seen").groupby("person_id").tail(1)
         .set_index("person_id")[["path_id", "last_seen"]].rename(columns={"path_id": "before_path", "last_seen": "before_last_seen"}))
    a = (seg[seg.first_seen > T].sort_values("first_seen").groupby("person_id").head(1)
         .set_index("person_id")[["path_id", "first_seen"]].rename(columns={"path_id": "after_path", "first_seen": "after_first_seen"}))
    ctx = b.join(a, how="outer")
    ctx = ctx.join(at[["path_id"]].rename(columns={"path_id": "seen_at_T_path"}), how="outer")
    ctx["gap_s"] = ctx.after_first_seen - ctx.before_last_seen
    ctx["before_dt_s"] = T - ctx.before_last_seen            # seconds between last sighting and T
    ctx["after_dt_s"] = ctx.after_first_seen - T             # seconds between T and next sighting
    hid, direct = [], []
    for bp, ap, g in zip(ctx.before_path, ctx.after_path, ctx.gap_s):
        if isinstance(bp, str) and isinstance(ap, str) and bp != ap:
            d, h = graph.gap_feasibility(bp, ap, g)
        else:
            d, h = False, False
        direct.append(d); hid.append(h)
    ctx["direct_ok"], ctx["hidden_ok"] = direct, hid
    ctx["hidden_only"] = ctx.hidden_ok & ~ctx.direct_ok
    ctx["status"] = "not_feasible"
    ctx.loc[ctx.hidden_ok, "status"] = "hidden_feasible"
    ctx.loc[ctx.after_path.isna(), "status"] = "no_after"
    ctx.loc[ctx.before_path.isna(), "status"] = "no_before"
    ctx.loc[ctx.seen_at_T_path.notna(), "status"] = "seen_at_T"
    return ctx.reset_index().rename(columns={"index": "person_id"})
