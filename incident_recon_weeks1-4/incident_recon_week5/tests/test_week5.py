"""Run:  python -m pytest tests -q            (DB tests need:  set RECON_TEST_DSN=postgresql://user:pass@localhost:5432/recon_test)"""
import os, subprocess, sys, glob, tempfile
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dbutil import to_named_params, connect, query_df
from timelines_ref import build_visits

DSN = os.environ.get("RECON_TEST_DSN")


def test_param_conversion_keeps_casts():
    s = to_named_params("SELECT x::numeric FROM t WHERE ts BETWEEN :T - :W AND :T + :W AND name LIKE 'a%'")
    assert "%(T)s" in s and "%(W)s" in s and "::numeric" in s and "LIKE 'a%%'" in s


def test_inference_queries_never_touch_ground_truth():
    files = glob.glob(os.path.join(ROOT, "sql", "queries", "*.sql")) + glob.glob(os.path.join(ROOT, "sql", "0[2-4]*.sql"))
    for f in files:
        body = "\n".join(l for l in open(f).read().splitlines() if not l.strip().startswith("--"))
        assert "ground_truth" not in body, f


def test_flip_repair_merges_short_sandwich():
    obs = pd.DataFrame({"person_id": [1] * 6, "path_id": ["P2", "P2", "P3", "P2", "P3", "P3"],
                        "timestamp": [0, 10, 12, 13, 15, 30.0]})
    v = build_visits(obs)
    assert list(v.path_id) == ["P2", "P3"] and v.n_obs.sum() == 6


def test_long_silence_splits_visit():
    obs = pd.DataFrame({"person_id": [1, 1, 1], "path_id": ["P3"] * 3, "timestamp": [0, 10, 1000.0]})
    assert len(build_visits(obs)) == 2


@pytest.mark.skipif(not DSN, reason="set RECON_TEST_DSN to run database tests")
def test_full_pipeline_on_postgres(tmp_path):
    env = dict(os.environ, RECON_DSN=DSN)
    sh = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout
    sh("simulate.py", "--preset", "easy", "--n-persons", "300", "--n-incidents", "10", "--out", str(tmp_path))
    sh("setup_db.py", "--reset")
    sh("load_data.py", "--source", "sim", "--data-dir", str(tmp_path))
    c = connect(DSN)
    obs = query_df(c, "SELECT person_id, path_id, ts AS timestamp FROM observations")
    sql_v = query_df(c, "SELECT person_id, path_id, enter_ts AS enter, exit_ts AS exit, n_obs FROM visits")
    ref = build_visits(obs)
    key = lambda d: d.round(3).sort_values(["person_id", "enter"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(key(sql_v), key(ref), check_dtype=False)
    # route times: crossing P4 from P3 to P5 needs at least 300 m / v_max
    rt = query_df(c, "SELECT min_seconds FROM route_times WHERE from_path='P3' AND to_path='P5' AND via_hidden")
    assert abs(rt.min_seconds[0] - 300 / 3.0) < 1e-6
    # feasibility filter keeps (nearly) all true occupants on the easy data
    e1 = query_df(c, """WITH c AS (SELECT i.id AS iid, ic.* FROM incidents i CROSS JOIN LATERAL incident_candidates(i.id, 900) ic)
                        SELECT c.status, SUM(l.in_hidden) AS inside FROM c JOIN ground_truth_labels l
                        ON l.incident_id = c.iid AND l.person_id = c.person_id GROUP BY c.status""").set_index("status").inside
    assert e1.sum() > 0 and e1.get("possible", 0) / e1.sum() >= 0.97
    assert e1.get("impossible: reached too fast", 0) == 0
