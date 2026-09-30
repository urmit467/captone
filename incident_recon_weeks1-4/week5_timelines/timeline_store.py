"""Glue between the database and timeline.py.  The `trajectories` table (source='observed') IS the stored timeline."""
import pandas as pd
from db import secs_to_ts, ts_to_secs


def load_sightings(db, t0):
    """Point sightings from `observations` -> (person_id, path_id, t_start, t_end) in seconds since t0."""
    df = db.fetch_df("SELECT person_id, path_id, ts FROM observations WHERE person_id IS NOT NULL")
    t = ts_to_secs(df.ts, t0)
    return pd.DataFrame({"person_id": df.person_id, "path_id": df.path_id, "t_start": t, "t_end": t})


def save_segments(db, seg, t0):
    db.execute("DELETE FROM trajectories WHERE source = 'observed'")
    db.insert_df("trajectories", pd.DataFrame({
        "person_id": seg.person_id, "path_id": seg.path_id,
        "start_time": secs_to_ts(seg.first_seen, t0), "end_time": secs_to_ts(seg.last_seen, t0),
        "source": "observed"}))                      # confidence stays NULL: Re-ID / detection uncertainty is modelled in Week 6


def load_segments(db, t0):
    df = db.fetch_df("SELECT person_id, path_id, start_time, end_time FROM trajectories WHERE source = 'observed' "
                     "ORDER BY person_id, start_time")
    seg = pd.DataFrame({"person_id": df.person_id, "path_id": df.path_id,
                        "first_seen": ts_to_secs(df.start_time, t0), "last_seen": ts_to_secs(df.end_time, t0)})
    seg["seg_idx"] = seg.groupby("person_id").cumcount()
    return seg


def parse_time(x, t0):
    """'1850.7' -> seconds since t0;  ISO string (naive = UTC) -> seconds since t0."""
    try:
        return float(x)
    except ValueError:
        ts = pd.Timestamp(x)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        return (ts - t0).total_seconds()


def iso(seconds, t0):
    return secs_to_ts([seconds], t0)[0].strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
