"""Small PostgreSQL helpers (psycopg2). No ORM: the logic lives in the .sql files."""
import os, re
from urllib.parse import urlparse
import pandas as pd
import psycopg2

DEFAULT_DSN = os.environ.get("RECON_DSN", "postgresql://postgres:postgres@localhost:5432/recon")
HERE = os.path.dirname(os.path.abspath(__file__))


def connect(dsn=None):
    return psycopg2.connect(dsn or DEFAULT_DSN)


def run_sql_file(conn, relpath):
    with open(os.path.join(HERE, relpath), encoding="utf-8") as f, conn.cursor() as cur:
        cur.execute(f.read())
    conn.commit()


def to_named_params(sql):
    """Turn psql-style ':name' variables into psycopg2 '%(name)s' (leaves '::casts' alone, escapes literal %)."""
    sql = sql.replace("%", "%%")
    return re.sub(r"(?<![:\w]):([A-Za-z_]\w*)", r"%(\1)s", sql)


def query_df(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    conn.rollback()
    return pd.DataFrame(rows, columns=cols)


def query_file(conn, relpath, params=None):
    sql = open(os.path.join(HERE, relpath), encoding="utf-8").read()
    # drop comment-only lines so psql-style hints (':' words inside comments) are never parsed as parameters
    sql = "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))
    return query_df(conn, to_named_params(sql), params or {})


def parse_dsn_db(dsn):
    u = urlparse(dsn)
    return u.path.lstrip("/"), dsn.replace(u.path, "/postgres")
