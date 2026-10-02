# Incident-time person reconstruction - Weeks 1-4

```
incident_recon_weeks1-4/
├── requirements.txt
├── week1_design/            scope, path graph, DB schema, recording plan, consent template
├── week2_simulator/         simulator + labelled synthetic dataset (hidden ground truth)
├── week3_detection_tracking/  YOLO + ByteTrack/BoT-SORT on video -> track CSVs + person crops
└── week4_reid/              embeddings -> topology-gated matching -> global person IDs + evaluation
```

## Setup (Week 1)
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # for GPU install PyTorch first from pytorch.org (CUDA build), then the rest
pip install gdown                 # needed by torchreid
# PostgreSQL is installed directly on your computer (no Docker) - see the Week 5 package for the working schema and setup
python draw_graph.py
```

## Run order
1. `week2_simulator`:  `python simulate.py`  (already run; data included)
2. `week3_detection_tracking`:  `python track_video.py ...` per camera video, then `python summarize_tracks.py ...`
3. `week4_reid`:  `extract_embeddings.py` -> `match_tracks.py` -> `evaluate_reid.py`
   No footage yet?  `python synthetic_reid_test.py` exercises the matcher on simulator data.

## What was tested here, and what was not
| Part | Status |
|---|---|
| Week 1 SQL schema | parsed OK as PostgreSQL; the working version of the schema is in the Week 5 package, where it was run on a real PostgreSQL 16 server |
| Week 1 graph diagram | generated and checked |
| Week 2 simulator | run; 3000 people, 300 incidents, statistics checked |
| Week 3 YOLO + ByteTrack + BoT-SORT | run on CPU with the real yolo11n weights on a **synthetic moving-image video** (people from a sample photo), producing CSVs, annotated video and crops. **Not** tested on real camera footage or GPU |
| Week 4 histogram embeddings + matcher CLI | run on the real YOLO crops from that test |
| Week 4 matcher accuracy | tested on synthetic embeddings only |
| Week 4 OSNet backend | code path runs with random weights; **accuracy with real pretrained weights is untested** |

So the plumbing is verified end to end, but no accuracy claims can be made until you record real footage.
