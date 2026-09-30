"""Turn per-camera track CSVs into one track summary table (input to Week 4 matching).

  python summarize_tracks.py output/tracks/*.csv --min-frames 15 --out output/track_summary.csv

track_summary.csv: camera_id, track_id, start_time, end_time, n_frames, mean_conf, entry_x, entry_y, exit_x, exit_y
Short / low-confidence tracks are dropped (usually false detections or fragments).
"""
import argparse, glob
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--min-frames", type=int, default=15)
ap.add_argument("--min-conf", type=float, default=0.4)
ap.add_argument("--out", default="output/track_summary.csv")
a = ap.parse_args()

files = [f for p in a.csvs for f in glob.glob(p)]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values(["camera_id", "track_id", "timestamp"])
rows = []
for (cam, tid), g in df.groupby(["camera_id", "track_id"]):
    rows.append(dict(camera_id=cam, track_id=tid, start_time=g.timestamp.iloc[0], end_time=g.timestamp.iloc[-1],
                     n_frames=len(g), mean_conf=g.confidence.mean(),
                     entry_x=g.foot_x.iloc[0], entry_y=g.foot_y.iloc[0], exit_x=g.foot_x.iloc[-1], exit_y=g.foot_y.iloc[-1]))
s = pd.DataFrame(rows)
kept = s[(s.n_frames >= a.min_frames) & (s.mean_conf >= a.min_conf)].sort_values("start_time")
kept.to_csv(a.out, index=False)
print(f"{len(s)} tracks -> kept {len(kept)} (dropped {len(s) - len(kept)} short/low-confidence) -> {a.out}")
