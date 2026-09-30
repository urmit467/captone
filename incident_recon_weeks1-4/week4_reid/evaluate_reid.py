"""Score a global timeline against manually labelled tracks.

Labels CSV: camera_id,track_id,person_id      (person_id = who the track really is; label a subset by hand)
  python evaluate_reid.py --timeline global_timeline.csv --labels reid_labels.csv

Metrics
  pairwise precision/recall/F1 : over all track pairs, "same predicted identity" vs "same true person"
  link precision : of the consecutive links inside each predicted identity, fraction joining tracks of the same person
  link recall    : of the consecutive tracks of each true person, fraction that ended up in the same predicted identity
  fragmentation  : predicted identities per true person (1.0 is perfect)
  merged ids     : predicted identities that contain more than one true person
"""
import argparse
import numpy as np, pandas as pd


def _pairs(x):
    return x * (x - 1) / 2


def evaluate(pred, true, times):
    """pred, true: aligned arrays of predicted / true identity per track; times: track start times."""
    df = pd.DataFrame(dict(p=pred, t=true, s=times))
    ct = df.groupby(["p", "t"]).size()
    tp = _pairs(ct).sum()
    pp = _pairs(df.groupby("p").size()).sum()
    tt = _pairs(df.groupby("t").size()).sum()
    prec, rec = tp / max(pp, 1), tp / max(tt, 1)

    def link_acc(key, other):
        ok = tot = 0
        for _, g in df.sort_values("s").groupby(key):
            a = g[other].to_numpy()
            ok += (a[1:] == a[:-1]).sum(); tot += max(len(a) - 1, 0)
        return ok / max(tot, 1)

    return dict(pair_precision=prec, pair_recall=rec, pair_f1=2 * prec * rec / max(prec + rec, 1e-9),
                link_precision=link_acc("p", "t"), link_recall=link_acc("t", "p"),
                n_pred_ids=df.p.nunique(), n_true_ids=df.t.nunique(),
                fragmentation=df.groupby("t").p.nunique().mean(),
                merged_ids=int((df.groupby("p").t.nunique() > 1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    tl = pd.read_csv(a.timeline).merge(pd.read_csv(a.labels), on=["camera_id", "track_id"])
    print(f"{len(tl)} labelled tracks")
    for k, v in evaluate(tl.global_person_id, tl.person_id, tl.start_time).items():
        print(f"  {k:16s} {v:.3f}" if isinstance(v, float) else f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
