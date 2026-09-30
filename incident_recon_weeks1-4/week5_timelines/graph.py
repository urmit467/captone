"""Path network as a NetworkX DiGraph + travel-time bounds between paths.

Nodes are PATHS (attributes: length_m, camera, observed).  Edges are allowed movement (directed), from path_graph.json.

route_table(a, b) lists every simple route a -> ... -> b.  For each route the walking distance is the total length of the
INTERMEDIATE paths (a and b themselves are not counted: the person was already seen on a and is next seen on b):
    gap (last seen on a -> first seen on b)  in  [ L / v_max - jitter ,  L / v_min + slack (+ extra_dwell if L > 0) ]
extra_dwell_s (default 0 = same as Week 4) is an allowance for people who stop/linger inside unobserved paths; the upper bound is a
HARD exclusion, so if real participants pause, raise it (otherwise they are dropped as 'not_feasible').
This is the same rule Week 4 uses for Re-ID gating (camera_topology.py), so the two weeks stay consistent.
"""
import json
from functools import lru_cache
import networkx as nx


class PathGraph:
    def __init__(self, graph_json, jitter_s=2.0, slack_s=None, extra_dwell_s=0.0):
        with open(graph_json) as f:
            g = json.load(f)
        self.extra_dwell = extra_dwell_s
        self.hidden = g["hidden_path"]
        self.vmin, self.vmax = g["walking_speed_bounds_mps"]
        self.slack, self.jitter = (g["time_slack_s"] if slack_s is None else slack_s), jitter_s
        self.G = nx.DiGraph()
        for p, d in g["paths"].items():
            self.G.add_node(p, length_m=d["length_m"], camera=d["camera"], observed=d["camera"] is not None)
        self.G.add_edges_from(g["edges"])
        self.cam_path = {d["camera"]: p for p, d in g["paths"].items() if d["camera"]}
        self.entries = g.get("entries", [])

    # ---- simple topology queries -------------------------------------------------------------------------------
    def length(self, p):
        return self.G.nodes[p]["length_m"]

    def predecessors(self, p):
        return sorted(self.G.predecessors(p))

    def successors(self, p):
        return sorted(self.G.successors(p))

    def neighbors(self, p, direction="both"):
        """Paths from which p can be entered ('in'), that can be entered from p ('out'), or both."""
        s = set()
        if direction in ("in", "both"):
            s |= set(self.G.predecessors(p))
        if direction in ("out", "both"):
            s |= set(self.G.successors(p))
        return sorted(s)

    # ---- routes and travel-time windows ------------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def route_table(self, a, b):
        """[(route_tuple, L_metres, lo_s, hi_s, goes_through_hidden)] for every simple route a -> b (a != b)."""
        if a == b or a not in self.G or b not in self.G:
            return []
        out = []
        for r in nx.all_simple_paths(self.G, a, b):
            L = sum(self.length(q) for q in r[1:-1])
            out.append((tuple(r), L, L / self.vmax - self.jitter, L / self.vmin + self.slack + (self.extra_dwell if L > 0 else 0.0), self.hidden in r[1:-1]))
        return sorted(out, key=lambda x: (x[1], x[0]))

    def gap_feasibility(self, a, b, gap_s):
        """Which kinds of route could explain 'last seen on a, first seen on b, gap_s seconds later'?
        Returns (direct_ok, hidden_ok): direct_ok = a route avoiding the hidden path fits the gap,
        hidden_ok = a route through the hidden path fits the gap."""
        direct_ok = hidden_ok = False
        for _, _, lo, hi, thru in self.route_table(a, b):
            if lo <= gap_s <= hi:
                hidden_ok |= thru
                direct_ok |= not thru
        return direct_ok, hidden_ok

    def min_hidden_time(self, a, b):
        """Fastest possible a -> (through hidden) -> b, in seconds (None if no such route)."""
        t = [lo for _, _, lo, _, thru in self.route_table(a, b) if thru]
        return min(t) if t else None
