"""Run any .sql file from sql/queries or sql/eval and print the result. ':name' variables are filled from --p name=value.

  python run_query.py sql/queries/q4_path_statistics.sql
  python run_query.py sql/queries/q1_seen_near_hidden_path.sql --p T=7200 --p W=900
  python run_query.py sql/queries/q6_incident_candidates.sql --p incident_id=5 --p W=900 --limit 20
The same files also run in psql / pgAdmin (set the variables with \\set first).
"""
import argparse
import pandas as pd
from dbutil import DEFAULT_DSN, connect, query_file

ap = argparse.ArgumentParser()
ap.add_argument("sqlfile")
ap.add_argument("--dsn", default=DEFAULT_DSN)
ap.add_argument("--p", action="append", default=[], help="name=value (repeatable)")
ap.add_argument("--limit", type=int, default=40, help="rows to print")
a = ap.parse_args()
params = {}
for kv in a.p:
    k, v = kv.split("=", 1)
    try: params[k] = float(v) if "." in v else int(v)
    except ValueError: params[k] = v
df = query_file(connect(a.dsn), a.sqlfile, params)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(df.head(a.limit).to_string(index=False))
if len(df) > a.limit:
    print(f"... {len(df)} rows total")
