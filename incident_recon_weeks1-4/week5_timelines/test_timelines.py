"""python -m unittest -v test_timelines   (run from this folder)"""
import os, sys, tempfile, unittest, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import PathGraph
from timeline import build_segments, build_transitions
from queries import incident_context, seen_on_neighbors
from db import connect, SIM_T0
import load_to_db
from timeline_store import load_sightings, save_segments, load_segments
from query_timelines import seen_on_neighbors_sql

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "..", "week1_design", "path_graph.json")
G = PathGraph(GRAPH)


def S(rows):
    """rows: (person, path, t) -> sightings"""
    df = pd.DataFrame(rows, columns=["person_id", "path_id", "t_start"])
    df["t_end"] = df.t_start
    return df


class TestSegments(unittest.TestCase):
    def test_merge_and_split_on_path_change(self):
        seg = build_segments(S([(1, "P2", 0), (1, "P2", 3), (1, "P3", 6), (1, "P3", 9)]))
        self.assertEqual(list(seg.path_id), ["P2", "P3"])
        self.assertEqual(list(seg.n_obs), [2, 2])

    def test_long_gap_on_same_path_splits(self):
        seg = build_segments(S([(1, "P3", 0), (1, "P3", 3), (1, "P3", 200)]), max_same_path_gap_s=60)
        self.assertEqual(len(seg), 2)

    def test_boundary_flicker_is_absorbed(self):
        seg = build_segments(S([(1, "P2", 0), (1, "P2", 3), (1, "P3", 5.8), (1, "P2", 6.1), (1, "P3", 6.5), (1, "P3", 9)]))
        self.assertEqual(list(seg.path_id), ["P2", "P3"])

    def test_persons_are_independent(self):
        seg = build_segments(S([(1, "P2", 0), (2, "P2", 1), (1, "P3", 3), (2, "P3", 4)]))
        self.assertEqual(len(seg), 4)


class TestGraph(unittest.TestCase):
    def test_neighbors(self):
        self.assertEqual(G.predecessors("P4"), ["P3"])
        self.assertEqual(G.successors("P4"), ["P5", "P6", "P7"])

    def test_min_hidden_time(self):
        self.assertAlmostEqual(G.min_hidden_time("P3", "P5"), 300 / 2.5 - 2)    # 118 s

    def test_gap_feasibility(self):
        self.assertEqual(G.gap_feasibility("P3", "P5", 3.0), (True, False))      # straight P3 -> P5
        self.assertEqual(G.gap_feasibility("P3", "P5", 100.0), (False, False))   # too slow for direct, too fast for P4
        self.assertEqual(G.gap_feasibility("P3", "P5", 200.0), (False, True))    # only via P4
        self.assertEqual(G.gap_feasibility("P3", "P6", 100.0), (True, False))    # via P5 fits, via P4 not yet
        self.assertEqual(G.gap_feasibility("P3", "P6", 250.0), (True, True))     # ambiguous


class TestIncidentContext(unittest.TestCase):
    """The worked example of the project description: incident in P4 at 10:30 (t = 1800 s after 10:00)."""
    T = 1800
    graph = G

    def ctx(self):
        rows = [(1, "P3", 1260), (1, "P3", 1263), (1, "P5", 2100),            # A: P3 10:21, P5 10:35            -> feasible, hidden-only (needs dwell allowance, see below)
                (2, "P3", 1735), (2, "P5", 1840),                             # B: P3 10:29:00 -> P5 10:30:40    -> P5 too soon for P4
                (3, "P3", 1790), (3, "P3", 1810),                             # C: on P3 at T                    -> seen_at_T
                (4, "P3", 1700)]                                              # D: nothing after                 -> no_after
        seg = build_segments(S(rows))
        return incident_context(seg, self.graph, "P4", self.T).set_index("person_id")

    def test_spec_example_needs_dwell_allowance(self):
        # 10:21 -> 10:35 is 840 s for a 300 m path = 0.36 m/s: slower than the 0.4 m/s bound of path_graph.json
        self.assertEqual(self.ctx().loc[1, "status"], "not_feasible")

    def test_statuses(self):
        self.graph = PathGraph(GRAPH, extra_dwell_s=120)              # allow people to linger in the hidden path
        c = self.ctx()
        self.assertEqual(c.loc[1, "status"], "hidden_feasible")
        self.assertTrue(c.loc[1, "hidden_only"])
        self.assertEqual(c.loc[2, "status"], "not_feasible")
        self.assertEqual(c.loc[3, "status"], "seen_at_T")
        self.assertEqual(c.loc[4, "status"], "no_after")

    def test_times(self):
        self.graph = PathGraph(GRAPH, extra_dwell_s=120)
        c = self.ctx()
        self.assertAlmostEqual(c.loc[1, "before_dt_s"], 1800 - 1263)
        self.assertAlmostEqual(c.loc[1, "after_dt_s"], 2100 - 1800)


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        data = os.path.join(cls.tmp, "data"); os.makedirs(data)
        src = os.path.join(HERE, "..", "week2_simulator", "data")
        obs = pd.read_csv(f"{src}/observations.csv"); keep = obs.person_id < 150
        obs[keep].to_csv(f"{data}/observations.csv", index=False)
        pd.read_csv(f"{src}/incidents.csv").head(3).to_csv(f"{data}/incidents.csv", index=False)
        cls.url = f"sqlite:///{cls.tmp}/t.db"
        load_to_db.cmd_sim(argparse.Namespace(db=cls.url, graph=GRAPH, reset=True, data=data, t0=str(SIM_T0), with_ground_truth=False))
        cls.db = connect(cls.url)
        cls.sight = load_sightings(cls.db, SIM_T0)
        cls.seg = build_segments(cls.sight)
        save_segments(cls.db, cls.seg, SIM_T0)

    def test_loaded(self):
        n = self.db.fetch_df("SELECT COUNT(*) n FROM observations").n[0]
        self.assertEqual(n, len(self.sight))
        self.assertEqual(self.db.fetch_df("SELECT COUNT(*) n FROM incidents").n[0], 3)

    def test_trajectory_roundtrip(self):
        back = load_segments(self.db, SIM_T0)
        a = self.seg.sort_values(["person_id", "first_seen"]).reset_index(drop=True)
        b = back.sort_values(["person_id", "first_seen"]).reset_index(drop=True)
        self.assertEqual(list(a.path_id), list(b.path_id))
        np.testing.assert_allclose(a.first_seen, b.first_seen, atol=1e-5)
        np.testing.assert_allclose(a.last_seen, b.last_seen, atol=1e-5)

    def test_sql_matches_pandas(self):
        back = load_segments(self.db, SIM_T0)
        total = 0
        for T in (2000.0, 5000.0, 8000.0):
            py = seen_on_neighbors(back, G, "P4", T - 900, T + 900, "both")
            sq = seen_on_neighbors_sql(self.db, "P4", T - 900, T + 900, SIM_T0)
            self.assertEqual(set(zip(py.person_id, py.path_id)), set(zip(sq.person_id, sq.path_id)))
            total += len(py)
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
