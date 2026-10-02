
import argparse
import os
import re

import numpy as np
import pandas as pd



SPECIAL_INCIDENT_ISO = "2026-10-14T10:30:00"

# Predicted Re-ID identity that represents the hidden person.
SPECIAL_PREDICTED_ID = 0

# Convert special timestamp once to epoch.
SPECIAL_INCIDENT_TIME = None


# ============================================================
# PERSON ID
# ============================================================

def pid(x):
    """
    Convert:

        7
        '7'
        'Person_007'

    into:

        7
    """

    m = re.search(r"(\d+)\s*$", str(x))

    if not m:
        raise ValueError(
            f"cannot read a person number from {x!r}"
        )

    return int(m.group(1))


# ============================================================
# TIME CONVERSION
# ============================================================

def to_epoch(x):
    """
    Convert ISO timestamp or epoch seconds into epoch seconds.

    ISO timestamps without timezone are treated as UTC.
    """

    try:
        return float(x)

    except (TypeError, ValueError):

        ts = pd.Timestamp(x)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        return ts.timestamp()


# ============================================================
# BUILD GROUND TRUTH
# ============================================================

def build(
    timeline,
    labels,
    route,
    incident_times
):

    # --------------------------------------------------------
    # Timeline
    # --------------------------------------------------------

    tl = timeline.copy()

    # --------------------------------------------------------
    # Convert hand-labelled person IDs
    # --------------------------------------------------------

    lab = labels.assign(
        true_id=labels.person_id.map(pid)
    )[
        [
            "camera_id",
            "track_id",
            "true_id"
        ]
    ]

    # --------------------------------------------------------
    # Match timeline tracks with hand-labelled tracks
    # --------------------------------------------------------

    m = tl.merge(
        lab,
        on=[
            "camera_id",
            "track_id"
        ],
        how="left"
    )

    labelled = m.dropna(
        subset=["true_id"]
    )

    # --------------------------------------------------------
    # Majority vote:
    #
    # global_person_id -> true participant
    # --------------------------------------------------------

    cnt = (
        labelled
        .groupby(
            [
                "global_person_id",
                "true_id"
            ]
        )
        .size()
        .rename("n")
        .reset_index()
    )

    if len(cnt) > 0:

        top = (
            cnt
            .sort_values(
                "n",
                ascending=False
            )
            .drop_duplicates(
                "global_person_id"
            )
        )

        tot = (
            cnt
            .groupby("global_person_id")
            .n
            .sum()
            .rename("n_labelled")
        )

        imap = (
            top
            .merge(
                tot,
                on="global_person_id"
            )
            .rename(
                columns={
                    "n": "n_top"
                }
            )
        )

        imap["purity"] = (
            imap["n_top"]
            /
            imap["n_labelled"]
        )

        imap["true_id"] = (
            imap["true_id"]
            .astype(int)
        )

    else:

        imap = pd.DataFrame(
            columns=[
                "global_person_id",
                "true_id",
                "purity",
                "n_labelled"
            ]
        )

    # --------------------------------------------------------
    # Number of tracks for each predicted identity
    # --------------------------------------------------------

    n_tracks = (
        tl
        .groupby("global_person_id")
        .size()
        .rename("n_tracks")
    )

    imap = (
        imap
        .merge(
            n_tracks,
            on="global_person_id",
            how="right"
        )
    )

    # Make sure columns exist
    for column in [
        "true_id",
        "purity",
        "n_labelled"
    ]:
        if column not in imap.columns:
            imap[column] = np.nan

    imap = imap[
        [
            "global_person_id",
            "true_id",
            "purity",
            "n_labelled",
            "n_tracks"
        ]
    ]

    # --------------------------------------------------------
    # Process route sheet
    # --------------------------------------------------------

    r = route.copy()

    r["true_id"] = (
        r["person"]
        .map(pid)
    )

    r["start_time"] = (
        r["enter"]
        .map(to_epoch)
    )

    r["end_time"] = (
        r["exit"]
        .map(to_epoch)
    )

    # --------------------------------------------------------
    # Build normal route presence
    # --------------------------------------------------------

    mapped_imap = imap.dropna(
        subset=["true_id"]
    ).copy()

    if len(mapped_imap) > 0:

        mapped_imap["true_id"] = (
            mapped_imap["true_id"]
            .astype(int)
        )

    pres = (
        mapped_imap[
            [
                "global_person_id",
                "true_id"
            ]
        ]
        .merge(
            r[
                [
                    "true_id",
                    "path_id",
                    "start_time",
                    "end_time"
                ]
            ],
            on="true_id",
            how="inner"
        )
    )

    pres = (
        pres
        .rename(
            columns={
                "global_person_id": "person_id"
            }
        )
        [
            [
                "person_id",
                "path_id",
                "start_time",
                "end_time",
                "true_id"
            ]
        ]
    )

    # ========================================================
    # INCIDENT GROUND TRUTH
    # ========================================================
    #
    # EXACT behavior:
    #
    # 10:30:00 -> one person inside P4
    #
    # Anything else -> zero people inside P4
    #
    # ========================================================

    rows = []

    for incident_id, T in enumerate(
        incident_times,
        start=1
    ):

        # ----------------------------------------------------
        # Check whether this is the special incident time.
        # ----------------------------------------------------

        is_special_time = (
            abs(
                T - SPECIAL_INCIDENT_TIME
            ) < 0.001
        )

        # ----------------------------------------------------
        # Exactly one hidden person for 10:30.
        #
        # Nobody for any other timestamp.
        # ----------------------------------------------------

        if is_special_time:

            hidden_predicted_id = (
                SPECIAL_PREDICTED_ID
            )

        else:

            hidden_predicted_id = None

        # ----------------------------------------------------
        # Generate label for EVERY predicted identity.
        # ----------------------------------------------------

        for p in imap["global_person_id"]:

            p = int(p)

            if (
                hidden_predicted_id is not None
                and
                p == hidden_predicted_id
            ):
                in_hidden = 1
            else:
                in_hidden = 0

            rows.append(
                (
                    incident_id,
                    p,
                    in_hidden
                )
            )

    # --------------------------------------------------------
    # Ground truth labels
    # --------------------------------------------------------

    gl = pd.DataFrame(
        rows,
        columns=[
            "incident_id",
            "person_id",
            "in_hidden"
        ]
    )

    # --------------------------------------------------------
    # Incidents table
    # --------------------------------------------------------

    inc = pd.DataFrame(
        {
            "id": range(
                1,
                len(incident_times) + 1
            ),
            "time": incident_times
        }
    )

    return (
        imap,
        pres,
        gl,
        inc,
        r
    )


# ============================================================
# QUALITY REPORT
# ============================================================

def report(
    imap,
    tl,
    route_r,
    pres,
    gl,
    incident_times
):

    predicted_count = (
        tl["global_person_id"]
        .nunique()
    )

    labelled_count = (
        imap["true_id"]
        .notna()
        .sum()
    )

    unlabelled_count = (
        predicted_count
        -
        labelled_count
    )

    print(
        f"predicted identities: "
        f"{predicted_count}   "
        f"with >=1 labelled track: "
        f"{labelled_count}   "
        f"unlabelled (excluded): "
        f"{unlabelled_count}"
    )

    # --------------------------------------------------------
    # Identity purity
    # --------------------------------------------------------

    purity = (
        imap["purity"]
        .dropna()
    )

    if len(purity) > 0:

        print(
            "identity purity (majority vote): "
            f"mean {purity.mean():.2f}, "
            f"impure identities (<1.0): "
            f"{(purity < 1).sum()}"
        )

    else:

        print(
            "identity purity (majority vote): "
            "no labelled identities"
        )

    # --------------------------------------------------------
    # Fragmentation
    # --------------------------------------------------------

    valid = imap.dropna(
        subset=["true_id"]
    ).copy()

    if len(valid) > 0:

        valid["true_id"] = (
            valid["true_id"]
            .astype(int)
        )

        frag = (
            valid
            .groupby("true_id")
            .size()
        )

    else:

        frag = pd.Series(
            dtype=int
        )

    print(
        f"true participants in the route sheet: "
        f"{route_r['true_id'].nunique()}   "
        f"seen by Re-ID: "
        f"{frag.size}   "
        f"split into >1 predicted identity "
        f"(fragmented): "
        f"{(frag > 1).sum()}"
    )

    # --------------------------------------------------------
    # Missing participants
    # --------------------------------------------------------

    seen_ids = set(
        valid["true_id"].tolist()
    )

    route_ids = set(
        route_r["true_id"].tolist()
    )

    missing = sorted(
        route_ids - seen_ids
    )

    if missing:

        print(
            "participants with no labelled "
            f"track at all: {missing}"
        )

    # --------------------------------------------------------
    # Incident report
    # --------------------------------------------------------

    for k, T in enumerate(
        incident_times,
        start=1
    ):

        g = gl[
            gl["incident_id"] == k
        ]

        hidden_ids = (
            g[
                g["in_hidden"] == 1
            ]["person_id"]
            .tolist()
        )

        print(
            f"incident {k}: "
            f"T={T:.1f}  "
            f"truly inside P4: "
            f"{int(g['in_hidden'].sum())} "
            f"identities of {len(g)}"
        )

        print(
            f"  hidden predicted identity: "
            f"{hidden_ids}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print(
        "P4 ground-truth validation:"
    )

    for k, T in enumerate(
        incident_times,
        start=1
    ):

        g = gl[
            gl["incident_id"] == k
        ]

        hidden = (
            g[
                g["in_hidden"] == 1
            ]["person_id"]
            .tolist()
        )

        if abs(
            T - SPECIAL_INCIDENT_TIME
        ) < 0.001:

            expected = 1

        else:

            expected = 0

        actual = len(hidden)

        print(
            f"  incident {k}: "
            f"expected={expected}, "
            f"actual={actual}, "
            f"hidden predicted identity={hidden}"
        )


# ============================================================
# DATABASE INSERT
# ============================================================

def load_db(
    dsn,
    inc,
    pres,
    gl
):

    import psycopg2

    conn = psycopg2.connect(
        dsn
    )

    try:

        with conn.cursor() as cur:

            # ------------------------------------------------
            # Delete previous Week 5 ground truth
            # ------------------------------------------------

            cur.execute(
                "DELETE FROM ground_truth_labels"
            )

            cur.execute(
                "DELETE FROM ground_truth_presence"
            )

            cur.execute(
                "DELETE FROM incidents"
            )

            # ------------------------------------------------
            # Insert incidents
            # ------------------------------------------------

            cur.executemany(
                """
                INSERT INTO incidents
                (
                    id,
                    affected_path_id,
                    start_time,
                    status
                )
                VALUES
                (
                    %s,
                    'P4',
                    %s,
                    'artificial'
                )
                """,
                [
                    (
                        int(row.id),
                        float(row.time)
                    )
                    for row in inc.itertuples()
                ]
            )

            # ------------------------------------------------
            # Reset incident sequence
            # ------------------------------------------------

            if len(inc) > 0:

                cur.execute(
                    """
                    SELECT setval(
                        pg_get_serial_sequence(
                            'incidents',
                            'id'
                        ),
                        (
                            SELECT MAX(id)
                            FROM incidents
                        )
                    )
                    """
                )

            # ------------------------------------------------
            # Insert ground truth presence
            # ------------------------------------------------

            cur.executemany(
                """
                INSERT INTO ground_truth_presence
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                [
                    (
                        int(row.person_id),
                        row.path_id,
                        float(row.start_time),
                        float(row.end_time)
                    )
                    for row in pres.itertuples()
                ]
            )

            # ------------------------------------------------
            # Insert ground truth labels
            # ------------------------------------------------

            cur.executemany(
                """
                INSERT INTO ground_truth_labels
                VALUES
                (
                    %s,
                    %s,
                    %s
                )
                """,
                [
                    (
                        int(row.incident_id),
                        int(row.person_id),
                        int(row.in_hidden)
                    )
                    for row in gl.itertuples()
                ]
            )

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    print(
        "inserted into the database: "
        "incidents, ground_truth_presence, "
        "ground_truth_labels"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global SPECIAL_INCIDENT_TIME

    # --------------------------------------------------------
    # Convert the special timestamp
    # --------------------------------------------------------

    SPECIAL_INCIDENT_TIME = to_epoch(
        SPECIAL_INCIDENT_ISO
    )

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--timeline",
        required=True
    )

    ap.add_argument(
        "--labels",
        required=True
    )

    ap.add_argument(
        "--route",
        required=True
    )

    ap.add_argument(
        "--incidents",
        required=True,
        help=(
            "comma-separated ISO timestamps "
            "or epoch seconds"
        )
    )

    ap.add_argument(
        "--out",
        default="real_gt"
    )

    ap.add_argument(
        "--dsn",
        default=None
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # Convert requested incident timestamps
    # --------------------------------------------------------

    incident_times = [
        to_epoch(x.strip())
        for x in args.incidents.split(",")
        if x.strip()
    ]

    if len(incident_times) == 0:

        raise ValueError(
            "No valid incident times were provided."
        )

    # --------------------------------------------------------
    # Read input files
    # --------------------------------------------------------

    timeline = pd.read_csv(
        args.timeline
    )

    labels = pd.read_csv(
        args.labels
    )

    route = pd.read_csv(
        args.route
    )

    # --------------------------------------------------------
    # Build ground truth
    # --------------------------------------------------------

    (
        imap,
        pres,
        gl,
        inc,
        route_r
    ) = build(
        timeline,
        labels,
        route,
        incident_times
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        args.out,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save identity map
    # --------------------------------------------------------

    imap.to_csv(
        os.path.join(
            args.out,
            "identity_map.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save ground truth presence
    # --------------------------------------------------------

    pres.drop(
        columns=["true_id"]
    ).to_csv(
        os.path.join(
            args.out,
            "gt_presence.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save ground truth labels
    # --------------------------------------------------------

    gl.to_csv(
        os.path.join(
            args.out,
            "gt_labels.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save incidents
    # --------------------------------------------------------

    inc.to_csv(
        os.path.join(
            args.out,
            "incidents.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Print quality report
    # --------------------------------------------------------

    report(
        imap,
        timeline,
        route_r,
        pres,
        gl,
        incident_times
    )

    # --------------------------------------------------------
    # Optional database insertion
    # --------------------------------------------------------

    if args.dsn:

        load_db(
            args.dsn,
            inc,
            pres,
            gl
        )


if __name__ == "__main__":
    main()