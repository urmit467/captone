# Week 3 - Person detection and single-camera tracking

```
python track_video.py --video C3_2026-10-14T10-00-00.mp4 --camera-id C3 \
       --start-time 2026-10-14T10:00:00 --tracker bytetrack --save-video --save-crops
python summarize_tracks.py output/tracks/*.csv --out output/track_summary.csv
```
Run once per camera video. Use `--tracker botsort` to compare; BoT-SORT handles camera motion and is slightly slower.
Model choice: `yolo11n.pt` for CPU tests, `yolo11s/m.pt` on GPU for real footage. Weights download automatically the first time.

Checklist for the demo: annotated video (`--save-video`) with stable IDs; a note on ID switches you observed; timestamps that
match the wall clock (verify against the clock shown in the sync clip).

Tuning tips: lower `--conf` if people are missed; raise `--imgsz` for small/far people; `--vid-stride 2` to halve compute.
