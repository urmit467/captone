"""Check the SQL-built timelines against the pandas reference implementation (and print data-quality numbers).
  python check_timelines.py
"""
import argparse
import numpy as np, pandas as pd
from dbutil import DEFAULT_DSN, connect, query_df
from timelines_ref import build_visits as ref_visits

ap = argparse.ArgumentParser(); ap.add_argument("--dsn", default=DEFAULT_DSN); a = ap.parse_args()
conn = connect(a.dsn)
obs = query_df(conn, "SELECT person_id, path_id, ts AS timestamp FROM observations")
sql_v = query_df(conn, "SELECT person_id, path_id, enter_ts AS enter, exit_ts AS exit, n_obs FROM visits")
ref_v = ref_visits(obs)
key = lambda d: d.round({"enter": 3, "exit": 3}).sort_values(["person_id", "enter"]).reset_index(drop=True)
s, r = key(sql_v), key(ref_v)
print(f"visits: SQL={len(s)}  pandas reference={len(r)}")
same = len(s) == len(r) and (s[["person_id", "path_id", "enter", "exit", "n_obs"]].to_numpy() == r[["person_id", "path_id", "enter", "exit", "n_obs"]].to_numpy()).all()
if same:
    print("OK - the SQL timeline is identical to the pandas reference")
else:
    m = s.merge(r, on=["person_id", "path_id", "enter", "exit"], how="outer", indicator=True)
    print(f"differences: {(m._merge != 'both').sum()} rows differ of {len(m)} ({(m._merge != 'both').mean():.2%})")
n_flip = int(query_df(conn, "SELECT COUNT(*) AS n FROM (SELECT person_id, path_id, visit_idx, "
                            "LEAD(path_id) OVER (PARTITION BY person_id ORDER BY visit_idx) AS nxt FROM visits) q WHERE path_id = nxt").n[0])
print(f"consecutive visits on the same path (should be 0): {n_flip}")
