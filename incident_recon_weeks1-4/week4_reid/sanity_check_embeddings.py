"""Quick Re-ID sanity check on your own crops (no labels needed):
split each track's crops into first half / second half, embed both halves, and check that the first half of a track is nearest
to the second half of the SAME track (rank-1 accuracy). Chance level = 1 / n_tracks.

  python sanity_check_embeddings.py --crops ../week3_detection_tracking/output/crops --backend hist
"""
import argparse
import cv2, numpy as np
from extract_embeddings import make_embedder, track_crops, l2

ap = argparse.ArgumentParser()
ap.add_argument("--crops", required=True)
ap.add_argument("--backend", default="hist")
ap.add_argument("--weights", default=None)
ap.add_argument("--device", default=None)
ap.add_argument("--min-crops", type=int, default=4)
a = ap.parse_args()
fn = make_embedder(a.backend, a.weights, a.device)
A, B, names = [], [], []
for cam, tid, files in track_crops(a.crops):
    if len(files) < a.min_crops:
        continue
    imgs = [cv2.imread(f) for f in files]
    h = len(imgs) // 2
    A.append(l2(l2(fn(imgs[:h])).mean(0))); B.append(l2(l2(fn(imgs[h:])).mean(0))); names.append(f"{cam}|{tid}")
A, B = np.stack(A), np.stack(B)
S = A @ B.T
top1 = (S.argmax(1) == np.arange(len(A))).mean()
print(f"{len(A)} tracks | rank-1 split-half accuracy = {top1:.2f} (chance = {1/len(A):.2f}) | "
      f"mean same-track sim = {np.diag(S).mean():.3f}, mean other-track sim = {(S.sum() - np.trace(S)) / (S.size - len(A)):.3f}")
