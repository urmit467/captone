"""Create the database (if missing) and the Week 5 schema + SQL functions. No Docker needed - uses your local PostgreSQL.

  python setup_db.py --dsn postgresql://postgres:YOURPASSWORD@localhost:5432/recon
  python setup_db.py --dsn ... --reset      # wipe everything and start again
(You can also set the RECON_DSN environment variable once instead of passing --dsn.)
"""
import argparse
import psycopg2
from dbutil import DEFAULT_DSN, connect, parse_dsn_db, run_sql_file

ap = argparse.ArgumentParser()
ap.add_argument("--dsn", default=DEFAULT_DSN)
ap.add_argument("--reset", action="store_true", help="drop all tables and functions first")
a = ap.parse_args()

dbname, admin_dsn = parse_dsn_db(a.dsn)
admin = psycopg2.connect(admin_dsn); admin.autocommit = True
with admin.cursor() as cur:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{dbname}"'); print(f"created database {dbname}")
admin.close()

conn = connect(a.dsn)
if a.reset:
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    conn.commit(); print("schema reset")
for f in ("sql/01_schema.sql", "sql/02_build_timelines.sql", "sql/03_route_times.sql", "sql/04_incident_candidates.sql"):
    run_sql_file(conn, f); print("ran", f)
print("database ready")
