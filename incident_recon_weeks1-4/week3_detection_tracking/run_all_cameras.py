"""Run Week 3 on every camera video of a session, then build track_summary.csv.

Video names must follow the recording plan:  <CameraId>_<YYYY-MM-DD>T<HH-MM-SS>.mp4     e.g. C3_2026-10-14T10-00-00.mp4
  python run_all_cameras.py --videos D:\\session1 --offsets C1=0.4,C3=-0.2 --device cuda:0
  python run_all_cameras.py --videos D:\\session1 --dry-run          # only prints what it would run
--offsets: seconds to ADD to each camera's clock to match the reference clock (from your clap / phone-clock sync).
Outputs (in --out-dir): tracks/<cam>.csv, crops/, track_summary.csv  (inputs of Week 4)
"""
import argparse, glob, os, re, subprocess, sys

PAT = re.compile(r"^(C\d+)_(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_name(path):
    m = PAT.match(os.path.basename(path))
    if not m:
        return None
    cam, d, hh, mm, ss = m.groups()
    return cam, f"{d}T{hh}:{mm}:{ss}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with the camera videos")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--offsets", default="", help="C1=0.4,C3=-0.2")
    ap.add_argument("--min-frames", type=int, default=15)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--vid-stride", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    offsets = dict((k, float(v)) for k, v in (kv.split("=") for kv in a.offsets.split(",") if kv))
    vids = sorted(f for ext in ("mp4", "avi", "mov", "mkv") for f in glob.glob(os.path.join(a.videos, f"*.{ext}")))
    jobs, skipped = [], []
    for f in vids:
        p = parse_name(f)
        (jobs if p else skipped).append((f, p) if p else f)
    for s in skipped:
        print(f"skipping (name does not match C<id>_YYYY-MM-DDTHH-MM-SS): {s}")
    if not jobs:
        sys.exit("no matching videos found")
    cams = [p[0] for _, p in jobs]
    if len(set(cams)) != len(cams):
        print("WARNING: more than one video per camera - tracks of the same camera are written to the same CSV and will overwrite each other; process one session at a time")
    for f, (cam, start) in jobs:
        cmd = [sys.executable, os.path.join(HERE, "track_video.py"), "--video", f, "--camera-id", cam, "--start-time", start,
               "--clock-offset", str(offsets.get(cam, 0.0)), "--tracker", a.tracker, "--model", a.model, "--out-dir", a.out_dir,
               "--save-crops", "--vid-stride", str(a.vid_stride)]
        if a.device:
            cmd += ["--device", a.device]
        if a.save_video:
            cmd += ["--save-video"]
        print(" ".join(cmd))
        if not a.dry_run:
            subprocess.run(cmd, check=True)
    cmd = [sys.executable, os.path.join(HERE, "summarize_tracks.py"), os.path.join(a.out_dir, "tracks", "*.csv"),
           "--min-frames", str(a.min_frames), "--out", os.path.join(a.out_dir, "track_summary.csv")]
    print(" ".join(cmd))
    if not a.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
