"""Pandas reference implementation of the visit builder - mirrors sql/02_build_timelines.sql exactly:
all short 'flip' visits are relabelled simultaneously, equal neighbours are merged, repeated until stable (max 5 rounds).
Used only to cross-check the SQL result (check_timelines.py) and by the unit tests."""
import pandas as pd


def _islands(o, split_gap_s):
    new = (o.person_id != o.person_id.shift()) | (o.path_id != o.path_id.shift()) | ((o.timestamp - o.timestamp.shift()) > split_gap_s)
    o = o.assign(v=new.cumsum())
    return o.groupby("v").agg(person_id=("person_id", "first"), path_id=("path_id", "first"),
                              enter=("timestamp", "min"), exit=("timestamp", "max"), n_obs=("timestamp", "size")).reset_index(drop=True)


def _merge_equal(L, split_gap_s):
    out = []
    for p, e0, e1, n in L:
        if out and out[-1][0] == p and e0 - out[-1][2] <= split_gap_s:
            out[-1][2] = e1; out[-1][3] += n
        else:
            out.append([p, e0, e1, n])
    return out


def build_visits(obs, split_gap_s=300.0, flip_s=12.0):
    """obs: DataFrame(person_id, path_id, timestamp) -> DataFrame(person_id, path_id, enter, exit, n_obs)"""
    o = obs.sort_values(["person_id", "timestamp"], kind="stable").reset_index(drop=True)
    v = _islands(o, split_gap_s)
    rows = []
    for pid, g in v.groupby("person_id", sort=False):
        L = [list(r) for r in g[["path_id", "enter", "exit", "n_obs"]].itertuples(index=False)]
        for it in range(6):
            flips = {i: L[i - 1][0] for i in range(1, len(L) - 1)
                     if L[i - 1][0] == L[i + 1][0] != L[i][0] and L[i][2] - L[i][1] <= flip_s and L[i][3] <= 3
                     and L[i + 1][1] - L[i - 1][2] <= 2 * flip_s}
            if not flips or it >= 5:
                break
            for i, p in flips.items():
                L[i][0] = p
            L = _merge_equal(L, split_gap_s)
        rows += [(pid, *x) for x in L]
    return pd.DataFrame(rows, columns=["person_id", "path_id", "enter", "exit", "n_obs"])
