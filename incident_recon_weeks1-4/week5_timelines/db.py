"""Thin database layer: PostgreSQL (psycopg2) for real use, SQLite for tests / no-Docker runs.

  connect("postgresql://recon:recon@localhost:5432/recon")   # docker compose in week1_design
  connect("sqlite:///week5.db")                              # or sqlite:///:memory:

Times in the DB are absolute timestamps (TIMESTAMPTZ).  The simulator and the Week 3/4 CSVs use plain seconds, so every
loader/reader takes a time origin t0:   timestamp = t0 + seconds.
  simulator data : t0 = SIM_T0  (2026-01-01 08:00 UTC, arbitrary)
  real footage   : t0 = epoch 0 (Week 3 writes epoch seconds)
"""
import io, re, sqlite3
import numpy as np, pandas as pd

SIM_T0 = pd.Timestamp("2026-01-01T08:00:00", tz="UTC")
EPOCH0 = pd.Timestamp("1970-01-01T00:00:00", tz="UTC")


def secs_to_ts(seconds, t0):
    return t0 + pd.to_timedelta(np.asarray(seconds, float), unit="s")


def ts_to_secs(ts, t0):
    return (pd.to_datetime(ts, utc=True, format="ISO8601") - t0).dt.total_seconds()


def sqlite_ddl(pg_ddl):
    """Translate schema.sql to SQLite (only what the schema uses)."""
    s = pg_ddl
    s = re.sub(r"\bBIGSERIAL\s+PRIMARY\s+KEY", "INTEGER PRIMARY KEY", s)
    s = re.sub(r"\bSERIAL\s+PRIMARY\s+KEY", "INTEGER PRIMARY KEY", s)
    s = s.replace("TIMESTAMPTZ", "TEXT").replace("DEFAULT now()", "DEFAULT CURRENT_TIMESTAMP")
    return s


class DB:
    def __init__(self, url):
        self.url = url
        if url.startswith("sqlite:///"):
            self.kind = "sqlite"
            self.conn = sqlite3.connect(url[len("sqlite:///"):])
            self.conn.execute("PRAGMA foreign_keys = ON")
        elif url.startswith("postgres"):
            import psycopg2
            self.kind = "pg"
            self.conn = psycopg2.connect(url)
        else:
            raise ValueError("use postgresql://... or sqlite:///file.db")

    def _sql(self, sql):
        return sql.replace("?", "%s") if self.kind == "pg" else sql

    def run_script(self, ddl):
        if self.kind == "sqlite":
            self.conn.executescript(sqlite_ddl(ddl))
        else:
            with self.conn.cursor() as c:
                c.execute(ddl)
        self.conn.commit()

    def execute(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(self._sql(sql), params)
        self.conn.commit()
        return cur

    def fetch_df(self, sql, params=()):
        cur = self.conn.cursor()
        cur.execute(self._sql(sql), params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)

    def insert_df(self, table, df):
        """Bulk insert.  Datetime columns (tz-aware) are written as ISO-8601 UTC."""
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        df = df.astype(object).where(pd.notna(df), None)
        cols = list(df.columns)
        if self.kind == "pg":
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            with self.conn.cursor() as cur:
                cur.copy_expert(f"COPY {table} ({','.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)
        else:
            q = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
            self.conn.executemany(q, [tuple(None if v is None else (v.item() if hasattr(v, 'item') else v) for v in row)
                                      for row in df.itertuples(index=False, name=None)])
        self.conn.commit()

    def reset_serial(self, table, col="id"):
        if self.kind == "pg":
            self.execute(f"SELECT setval(pg_get_serial_sequence('{table}','{col}'), COALESCE(MAX({col}),1)) FROM {table}")


def connect(url):
    return DB(url)
