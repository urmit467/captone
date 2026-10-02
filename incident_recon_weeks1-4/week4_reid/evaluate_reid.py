"""
Count predicted and true identities from a labelled Re-ID timeline.

Usage:
    python evaluate_reid.py --timeline global_timeline.csv --labels reid_labels.csv
"""

import argparse
import pandas as pd


def evaluate(timeline, labels):
    """
    Return only:
      - n_pred_ids: number of unique predicted identities
      - n_true_ids: number of unique true identities
    """

    tl = timeline.merge(
        labels,
        on=["camera_id", "track_id"]
    )

    return {
        "n_pred_ids": tl["global_person_id"].nunique(),
        "n_true_ids": tl["person_id"].nunique()
    }


def main():

    # ---------------------------------------------------------
    # Argument parser
    # ---------------------------------------------------------

    ap = argparse.ArgumentParser(
        description="Count predicted and true Re-ID identities"
    )

    ap.add_argument(
        "--timeline",
        required=True,
        help="Path to global_timeline.csv"
    )

    ap.add_argument(
        "--labels",
        required=True,
        help="Path to reid_labels.csv"
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Read CSV files
    # ---------------------------------------------------------

    timeline = pd.read_csv(args.timeline)
    labels = pd.read_csv(args.labels)

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    timeline_columns = [
        "camera_id",
        "track_id",
        "global_person_id"
    ]

    label_columns = [
        "camera_id",
        "track_id",
        "person_id"
    ]

    for column in timeline_columns:
        if column not in timeline.columns:
            raise ValueError(
                f"Missing column '{column}' in timeline CSV"
            )

    for column in label_columns:
        if column not in labels.columns:
            raise ValueError(
                f"Missing column '{column}' in labels CSV"
            )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    results = evaluate(
        timeline,
        labels
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print(f"{len(timeline.merge(labels, on=['camera_id', 'track_id']))} labelled tracks")
    print()
    print(f"n_pred_ids       {results['n_pred_ids']}")
    print(f"n_true_ids       {results['n_true_ids']}")


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------

if __name__ == "__main__":
    main()