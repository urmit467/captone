# Week 4 - Cross-camera Re-ID -> global person IDs

Pipeline: Week 3 crops -> **extract_embeddings.py** -> **match_tracks.py** -> `global_timeline.csv` -> **evaluate_reid.py**

```
python extract_embeddings.py --crops ../week3_detection_tracking/output/crops --backend hist            # baseline / distinct clothing
python extract_embeddings.py --crops ... --backend osnet --weights osnet_x1_0_market.pth --device cuda:0  # recommended
python sanity_check_embeddings.py --crops ...                     # no labels needed: split-half rank-1 accuracy
python match_tracks.py --tracks ../week3_detection_tracking/output/track_summary.csv \
       --embeddings track_embeddings.npz --graph ../week1_design/path_graph.json --out global_timeline.csv
python evaluate_reid.py --timeline global_timeline.csv --labels reid_labels.csv   # labels: camera_id,track_id,person_id
```
Camera IDs in `track_summary.csv` must match the ones in `path_graph.json` (C1, C2, C3, C5, C6, C7).

**Why the path graph matters:** `camera_topology.py` only lets a track join an identity if the camera transition and time gap are
physically possible (e.g. C3 -> C5 within ~0 s, or ~2-13 min if the person went through P4; never C3 -> C7 in 4 s).
Synthetic test (`python synthetic_reid_test.py`, 800 simulated people, ~2900 tracks):

| setting | noise | pair-F1 | link precision | merged IDs |
|---|---|---|---|---|
| appearance + time order only | 0.7 | 0.52 | 0.83 | 85 |
| appearance + topology gating | 0.7 | **0.90** | **0.999** | 2 |
| appearance + topology gating | 0.4 | 0.91 | 0.94 | 86 |

(Synthetic embeddings, so read this as "the matcher and gating work", not as real Re-ID accuracy.)

**Tuning `--max-dist`:** it is the main knob. Too small -> identities fragment (recall drops; at noise 1.0 with 0.35 nothing links);
too large -> different people merge. Choose it on your hand-labelled subset by maximising pair-F1.

**Fallback (from the plan):** if OSNet is weak on your footage, use `--backend hist` with participants in distinctly coloured clothing.

Getting OSNet weights: download an OSNet-x1.0 Market-1501 / MSMT17 checkpoint from the torchreid model zoo
(https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO) and pass it with `--weights`.

Deliverable for the week: global person IDs + the Re-ID accuracy number from `evaluate_reid.py` on ~50-100 hand-labelled tracks.
