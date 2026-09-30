"""Week 3: person detection (YOLO) + single-camera tracking (ByteTrack / BoT-SORT).

Example:
  python track_video.py --video C3_2026-10-14T10-00-00.mp4 --camera-id C3 \
        --start-time 2026-10-14T10:00:00 --tracker bytetrack --save-video --save-crops

Outputs (in --out-dir, default ./output):
  tracks/<camera_id>.csv        camera_id,track_id,frame,timestamp,x1,y1,x2,y2,foot_x,foot_y,confidence
  crops/<camera_id>/<track>/<frame>.jpg   sampled person crops (input for Week 4 Re-ID)
  videos/<camera_id>_tracked.mp4          annotated video (optional)
"""
import argparse, csv, os
from datetime import datetime, timezone
import cv2
from ultralytics import YOLO


def parse_start(s):
    """ISO string (naive = treated as UTC) or epoch seconds -> epoch seconds."""
    try:
        return float(s)
    except ValueError:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--start-time", default="0", help="wall-clock time of frame 0 (ISO or epoch seconds)")
    ap.add_argument("--clock-offset", type=float, default=0.0, help="seconds to add to correct this camera's clock")
    ap.add_argument("--model", default="yolo11n.pt", help="e.g. yolo11n.pt (fast) / yolo11s.pt / yolo11m.pt (accurate)")
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    ap.add_argument("--vid-stride", type=int, default=1, help="process every n-th frame")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crop-every", type=int, default=10, help="save a crop every n processed frames per track")
    ap.add_argument("--max-crops", type=int, default=25, help="max crops kept per track")
    ap.add_argument("--min-crop-h", type=int, default=64)
    a = ap.parse_args()

    os.makedirs(f"{a.out_dir}/tracks", exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    t0 = parse_start(a.start_time) + a.clock_offset

    writer = None
    if a.save_video:
        os.makedirs(f"{a.out_dir}/videos", exist_ok=True)
        writer = cv2.VideoWriter(f"{a.out_dir}/videos/{a.camera_id}_tracked.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps / a.vid_stride, (w, h))

    model = YOLO(a.model)
    stream = model.track(source=a.video, stream=True, persist=True, tracker=f"{a.tracker}.yaml",
                         classes=[0], conf=a.conf, imgsz=a.imgsz, device=a.device,
                         vid_stride=a.vid_stride, verbose=False)   # class 0 = person

    n_crops, n_rows, ids = {}, 0, set()
    with open(f"{a.out_dir}/tracks/{a.camera_id}.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["camera_id", "track_id", "frame", "timestamp", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "confidence"])
        for i, r in enumerate(stream):
            frame = i * a.vid_stride
            ts = t0 + frame / fps
            if r.boxes is not None and r.boxes.id is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                tid = r.boxes.id.int().cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), t, c in zip(xyxy, tid, conf):
                    wr.writerow([a.camera_id, int(t), frame, f"{ts:.3f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                                 f"{(x1 + x2) / 2:.1f}", f"{y2:.1f}", f"{c:.3f}"])
                    n_rows += 1; ids.add(int(t))
                    if a.save_crops and i % a.crop_every == 0 and n_crops.get(int(t), 0) < a.max_crops and (y2 - y1) >= a.min_crop_h:
                        d = f"{a.out_dir}/crops/{a.camera_id}/{int(t):04d}"
                        os.makedirs(d, exist_ok=True)
                        H, W = r.orig_img.shape[:2]
                        crop = r.orig_img[max(int(y1), 0):min(int(y2), H), max(int(x1), 0):min(int(x2), W)]
                        if crop.size:
                            cv2.imwrite(f"{d}/{frame:06d}.jpg", crop)
                            n_crops[int(t)] = n_crops.get(int(t), 0) + 1
            if writer is not None:
                writer.write(r.plot())
            if i % 200 == 0:
                print(f"frame {frame}  tracks so far: {len(ids)}", flush=True)
    if writer is not None:
        writer.release()
    print(f"done: {n_rows} detections, {len(ids)} tracks -> {a.out_dir}/tracks/{a.camera_id}.csv")


if __name__ == "__main__":
    main()
