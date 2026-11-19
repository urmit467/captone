# Path Network Setup — Design, Operation, Engineering & NetworkX Usage

**Project:** AI-Based Incident-Time Person Reconstruction in an Unobservable Path  
**Location:** `D:\project\incident-reconstruction`  
**Primary module:** `modules/path_manager.py`  
**Supporting modules:** `database/schema.py`, `database/db_operations.py`, `modules/travel_time.py`, `modules/timeline_generator.py`, `data/synthetic_data.py`, `app.py`

---

## Table of Contents

1. [What the path network is](#what-the-path-network-is)
2. [What is NetworkX](#what-is-networkx)
3. [How NetworkX is used in this project](#how-networkx-is-used-in-this-project)
4. [Where the logic lives in the project](#where-the-logic-lives-in-the-project)
5. [How the path network works (step by step)](#how-the-path-network-works-step-by-step)
6. [Engineering design decisions](#engineering-design-decisions)
7. [Code walk-through with examples](#code-walk-through-with-examples)
8. [How the path network drives the rest of the system](#how-the-path-network-drives-the-rest-of-the-system)
9. [End-to-end demo example](#end-to-end-demo-example)

---

## 1. What the path network is

The **path network** is a connected, undirected graph that models the physical walkable layout of a facility (campus, mall, metro station, office complex). Each node in the graph is a *path segment* — a real place people can walk through, such as a lobby, corridor, stairwell, or service shaft. Each edge represents direct physical adjacency: if path A and path B are connected, a person can walk from A to B without leaving the covered walking area.

### Node attributes

Every path node stores:

| Attribute | Type | Meaning | Example |
|---|---|---|---|
| `path_id` | string | Unique identifier | `"P4"` |
| `name` | string | Human-readable name | `"Service shaft"` |
| `length` | float (meters) | Physical length of the path | `70.0` |
| `average_speed` | float (m/min) | Average walking speed along it | `84.0` (~1.4 m/s) |
| `is_observable` | bool | `true` if a camera covers it; `false` if camera-free/unobservable | `False` for P4 |
| `camera_ids` | list of strings | Cameras assigned to watch this path | `["CAM_P3"]` |
| `connected_paths` | list of strings | Path IDs directly adjacent to this one | `["P3", "P5"]` |

### Demo network topology

The demo dataset (`data/synthetic_data.py`) builds a simple line network:

```
P1 ⟷ P2 ⟷ P3 ⟷ P4* ⟷ P5 ⟷ P6
     observable   ^  unobservable   observable
                  |
            No camera on P4
```

P4 is deliberately **unobservable** (no camera). The core research question the project answers is:

> *"A person was seen on P3 before the incident and on P5 after the incident. No camera covers P4. What is the probability they were physically inside P4 at the exact incident timestamp?"*

Without the path network, this question is unanswerable: we know where the person was before and after, but not how they moved through the camera-free middle. The path network supplies the physical connectivity and travel-time model needed to reason about it.

---

## 2. What is NetworkX

**NetworkX** is a Python library for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks (graphs). It provides:

- **Graph data structures** — `Graph` (undirected), `DiGraph` (directed), `MultiGraph`, etc.
- **Algorithms** — shortest paths (Dijkstra, BFS), all simple paths, connected components, minimum spanning tree, centrality measures, traversal, and more.
- **Node/edge attributes** — arbitrary key-value metadata attached to nodes and edges.
- **Integration** — works with NumPy, SciPy, Pandas, Matplotlib, and Graphviz.

### Why NetworkX was chosen for this project

| Requirement | How NetworkX satisfies it |
|---|---|
| Represent path connectivity | `nx.Graph()` nodes = paths, edges = adjacency |
| Find shortest route between two observed paths | `nx.shortest_path(graph, source, target)` |
| Enumerate every feasible route (for route coverage) | `nx.all_simple_paths(graph, source, target, cutoff=8)` |
| Check whether the whole network is connected | `nx.is_connected(graph)` |
| Find connected components (isolated sub-networks) | `nx.connected_components(graph)` |
| Attach physics metadata (length, speed, cameras) | `graph.add_node(pid, length=..., average_speed=..., ...)` |
| Lightweight, pure Python, no GUI/server needed | Installed via pip, runs on CPU, no GPU |

NetworkX is a **pure graph algorithm library** — it does not persist data, render video, or run ML. That is exactly why it fits here: the project needs graph topology and pathfinding, and NetworkX provides it without adding a database, web server, or deep-learning dependency.

---

## 3. How NetworkX is used in this project

NetworkX is used exclusively inside **`modules/path_manager.py`**, inside the `PathManager` class. The class maintains an in-memory NetworkX graph (`self.graph`) that is synchronized with MongoDB.

### Graph instance creation

`modules/path_manager.py:26`
```python
self.graph = nx.Graph()
```

An **undirected** graph is used because movement through corridors is normally bidirectional: if a person can walk P3 → P4, they can normally also walk P4 → P3.

### Loading paths from MongoDB into the NetworkX graph

`modules/path_manager.py:32` — `load_from_db()`:

```python
def load_from_db(self) -> nx.Graph:
    self.graph.clear()
    paths = self.db.list_persons()  # actually db.list_paths()
    for p in paths:
        pid = p["path_id"]
        self.graph.add_node(
            pid,
            name=p.get("name", pid),
            length=float(p.get("length", 0.0)),
            average_speed=float(p.get("average_speed", 1.0)),
            is_observable=bool(p.get("is_observable", True)),
            camera_ids=list(p.get("camera_ids") or []),
        )
    for p in paths:
        pid = p["path_id"]
        for cp in (p.get("connected_paths") or []):
            if cp not in self.graph:
                self.graph.add_node(cp, name=cp, length=0.0,
                                    average_speed=1.0, is_observable=False,
                                    camera_ids=[])
            if not self.graph.has_edge(pid, cp):
                self.graph.add_edge(pid, cp)
    self._loaded = True
    return self.graph
```

Two passes are used: first all nodes are created (with physical attributes from MongoDB), then edges are added based on `connected_paths`. This guards against dangling references (a path listed as connected but not yet present as its own document).

### NetworkX graph queries exposed by PathManager

| Method | NetworkX call | Purpose |
|---|---|---|
| `shortest_path(source, target)` | `nx.shortest_path(...)` | Fewest-hop route between two paths (used for inferred transitions) |
| `all_simple_paths(source, target, cutoff=8)` | `nx.all_simple_paths(...)` | Every simple route up to 8 hops (used for route coverage in Case 3) |
| `is_connected(a, b)` | `graph.has_edge(...)` | Direct adjacency check |
| `neighbors(path_id)` | `graph.neighbors(...)` | List of directly-connected paths |
| `is_fully_connected()` | `nx.is_connected(...)` | Is the whole network one connected component? |
| `connected_components()` | `nx.connected_components(...)` | List of isolated sub-networks |
| `network_layout(seed=42)` | `nx.spring_layout(...)` | Node positions for the Plotly dashboard visualization |

The `cutoff=8` on `all_simple_paths` prevents combinatorial explosion on larger graphs while still covering realistic campus-scale networks.

---

## 4. Where the logic lives in the project

| Concern | File | Lines / symbols |
|---|---|---|
| Graph + CRUD + connections + camera assignment + NetworkX queries | `modules/path_manager.py` | `class PathManager`, all methods |
| Database schema / collection & index definitions | `database/schema.py` | `COLLECTIONS`, `INDEXES["paths"]` |
| DB CRUD (insert/get/list/update/delete path) | `database/db_operations.py` | `insert_path`, `get_path`, `list_paths`, `update_path`, `delete_path` |
| Travel time (length / speed, min/expected/max) | `modules/travel_time.py` | `class TravelTimeModel` |
| Demo network construction (P1–P6) | `data/synthetic_data.py` | `DEFAULT_PATHS`, `generate_demo()` |
| Dashboard UI for the network | `app.py` | `page_path_network()` |

### File: `modules/path_manager.py` — the heart of the path network

This single file owns:

1. **Graph lifecycle** — `__init__`, `load_from_db`, `refresh`, `is_loaded`
2. **Path CRUD** — `add_path`, `update_path`, `delete_path`, `get_path`, `list_paths`
3. **Connectivity** — `connect_paths`, `disconnect_paths`, `neighbors`, `is_connected`
4. **Cameras** — `assign_camera`, `unassign_camera`, `cameras_for_path`
5. **NetworkX algorithms** — `shortest_path`, `all_simple_paths`, `has_route`, `connected_components`, `is_fully_connected`
6. **Visualization data** — `network_layout` for Plotly rendering

### File: `database/schema.py` — schema contract

```python
COLLECTIONS = [
    "persons", "cameras", "paths", "observations", "trajectories",
    "incidents", "predictions", "ground_truth",
]

INDEXES = {
    "paths": [
        {"keys": [("path_id", 1)], "unique": True},
        {"keys": [("is_observable", 1)]},
    ],
    ...
}
```

The `paths` collection is the persistent store; `path_id` is unique, and `is_observable` is indexed so the dashboard can quickly split paths into observable vs unobservable.

### File: `database/db_operations.py` — persistent CRUD

`insert_path(...)` writes to MongoDB; `list_paths()` reads them back. `_serialize` converts datetime fields to ISO-8601 UTC strings so path documents (and related collections) store timestamps consistently.

---

## 5. How the path network works (step by step)

### Step A: Define paths with physical properties

Each path is registered with a length (meters) and average walking speed (m/min). These two numbers determine the **travel time** to cross the path:

```
travel_time_minutes = length / average_speed
```

### Step B: Connect paths into a graph

`connect_paths("P3", "P4")` adds an undirected edge between P3 and P4, both in MongoDB (`connected_paths` arrays on both path docs) and in the NetworkX graph (`graph.add_edge("P3", "P4")`).

### Step C: Assign cameras

`assign_camera("P3", "CAM_P3")` records that camera `CAM_P3` watches path P3. Paths without cameras (like P4) are unobservable — no direct sightings exist there.

### Step D: Load into memory

At app startup, `load_from_db()` pulls all paths and connections from MongoDB into the NetworkX graph. From then on, graph algorithms run in memory without hitting the database on every hop.

### Step E: Query and use

- **Shortest path:** `pm.shortest_path("P1", "P6")` → `["P1", "P2", "P3", "P4", "P5", "P6"]`
- **All routes:** `pm.all_simple_paths("P1", "P6")` → every simple route (for route coverage)
- **Connectivity:** `pm.is_fully_connected()` → `True` (or False if the network has isolated islands)

---

## 6. Engineering design decisions

### 6.1 Undirected graph, not directed

Corridor traversal is generally bidirectional. An undirected `nx.Graph()` keeps the domain model honest and avoids forcing one-way assumptions that may not hold in a real building.

### 6.2 Separation of concerns: graph vs database

- **MongoDB** is the source of truth and persists data across app restarts.
- **NetworkX** (in-memory) provides fast graph algorithms without repeated DB queries.

Mutations (`add_path`, `connect_paths`, `delete_path`) write to MongoDB **and** update the in-memory graph atomically, so both stay consistent.

### 6.3 Observability is a node attribute, not a separate node type

Every path is a node regardless of observability. The only difference is `is_observable`. This matters because:

- **Observable paths** produce camera observations (timestamps the reconstruction engine can anchor to).
- **Unobservable paths** (like P4) produce *no* observations — they can only be inferred as intermediates between two observed paths.

This keeps the graph uniform: the engine asks "is there a route from the last observed path to the first observed path passing through the affected path?" — observability does not change whether a node exists in the graph.

### 6.4 Cameras assigned to paths, not paths to cameras

Each camera document stores a `path_id`; each path document stores a list of `camera_ids`. `assign_camera()` keeps both sides in sync. The dashboard can answer either "which cameras watch this path?" or "which path does this camera watch?".

### 6.5 Why NetworkX instead of writing custom graph code

- Battle-tested shortest-path and all-simple-paths algorithms.
- Correct handling of disconnected graphs, cycles, and dangling references.
- No need to reinvent BFS/Dijkstra or path enumeration.
- Cutoff parameter bounds computation time on `all_simple_paths`.

---

## 7. Code walk-through with examples

### 7.1 Adding a path

`modules/path_manager.py:77` — `add_path`:

```python
def add_path(
    self, path_id: str, name: str, length: float,
    average_speed: float, is_observable: bool = True,
) -> bool:
    if length <= 0:
        raise ValueError("Path length must be positive")
    if average_speed <= 0:
        raise ValueError("Average walking speed must be positive")
    if self.path_exists(path_id):
        raise ValueError(f"Path already exists: {path_id}")
    self.db.insert_path(
        path_id=path_id, name=name, length=length,
        average_speed=average_speed, is_observable=is_observable,
        connected_paths=[],
    )
    self.graph.add_node(
        path_id, name=name,
        length=float(length),
        average_speed=float(average_speed),
        is_observable=bool(is_observable),
        camera_ids=[],
    )
    return True
```

Validation: length and speed must be positive because they feed directly into `length / speed` travel-time math. Duplicate IDs are rejected. The path is written to MongoDB **and** added to the NetworkX graph as a node with attributes.

### 7.2 Connecting two paths

`modules/path_manager.py:168` — `connect_paths`:

```python
def connect_paths(self, path_id_a: str, path_id_b: str) -> bool:
    if path_id_a == path_id_b:
        raise ValueError("Cannot connect a path to itself")
    if not self.path_exists(path_id_a) or not self.path_exists(path_id_b):
        raise ValueError("Both paths must exist before connecting")
    pa = self.db.get_path(path_id_a)
    pb = self.db.get_path(path_id_b)
    conns_a = list(pa.get("connected_paths") or [])
    conns_b = list(pb.get("connected_paths") or [])
    if path_id_b in conns_a and path_id_a in conns_b:
        return False  # already connected
    conns_a.append(path_id_b)
    conns_b.append(path_id_a)
    self.db.update_path(path_id_a, {"connected_paths": conns_a})
    self.db.update_path(path_id_b, {"connected_paths": conns_b})
    self.graph.add_edge(path_id_a, path_id_b)
    return True
```

Both sides of the edge are persisted (MongoDB `connected_paths` arrays) and held in memory (NetworkX `add_edge`).

### 7.3 Querying the NetworkX graph

```python
# Direct adjacency
pm.is_connected("P3", "P4")          # True / False
pm.neighbors("P4")                   # ["P3", "P5"]

# Shortest route (fewest hops, unweighted)
pm.shortest_path("P1", "P6")         # ["P1", "P2", "P3", "P4", "P5", "P6"]

# Every simple route up to 8 hops
pm.all_simple_paths("P1", "P6")      # [["P1","P2",...], ...]

# Network health
pm.is_fully_connected()              # bool
pm.connected_components()            # [["P1",...,"P6"], ...]
```

### 7.4 Deleting a path

`modules/path_manager.py:142` — `delete_path` removes all incident edges first (via `disconnect_paths`), then deletes the MongoDB document and the NetworkX node. This prevents dangling edges to a nonexistent node.

---

## 8. How the path network drives the rest of the system

The path network is not just storage — it is the **spatial backbone** consumed by several modules:

### 8.1 Travel-time estimation (`modules/travel_time.py`)

Travel time for a path = `length / average_speed`. Each path carries min/expected/max bounds (`0.75×` / `1.0×` / `1.25×` of expected). `route_times(route)` sums these across a route; `presence_window_on_path(route, path_id)` computes the earliest/latest a traveler could be on a path along a route.

This feeds the reconstruction engine's entry/exit estimates.

### 8.2 Timeline inference (`modules/timeline_generator.py`)

When a person has consecutive observations on non-adjacent paths, the generator inserts the shortest-path intermediates as **inferred** events:

```python
route = self.path_manager.shortest_path(src_path, dst_path)
if route and len(route) > 2:
    for intermediate in route[1:-1]:
        trajectory.append({..., "path_id": intermediate,
                           "source": "inferred", ...})
```

So the path network turns sparse camera sightings into a continuous reconstructed trajectory.

### 8.3 Reconstruction probability (`modules/reconstruction_engine.py`)

Case 3 (the core case) uses `all_simple_paths` to enumerate every route between the last-before and first-after observations, keeps routes that fit in the available time gap **and** pass through the affected path, and computes:

- `route_coverage = len(through) / len(feasible)` — what share of feasible routes go through the affected path.
- A presence window on the affected path bounded by travel times (entry constrained from the last observation; exit constrained so they reappear on the later-observed path on time).

Without the path network, none of this is computable.

---

## 9. End-to-end demo example

```python
from modules.path_manager import PathManager
from database.db_operations import DBOperations

db = DBOperations()
pm = PathManager(db_ops=db)

# --- Build the demo network (P1..P6, P4 unobservable) ---
pm.add_path("P1", "Lobby", 90.0, 84.0, is_observable=True)
pm.add_path("P2", "Corridor A", 110.0, 84.0, is_observable=True)
pm.add_path("P3", "Corridor B", 80.0, 84.0, is_observable=True)
pm.add_path("P4", "Service shaft", 70.0, 84.0, is_observable=False)  # affected
pm.add_path("P5", "Corridor C", 100.0, 84.0, is_observable=True)
pm.add_path("P6", "Exit lobby", 90.0, 84.0, is_observable=True)

pm.connect_paths("P1", "P2")
pm.connect_paths("P2", "P3")
pm.connect_paths("P3", "P4")
pm.connect_paths("P4", "P5")
pm.connect_paths("P5", "P6")

# --- Cameras ---
pm.assign_camera("P3", "CAM_P3")   # watches P3
pm.assign_camera("P5", "CAM_P5")   # watches P5
# P4 has NO camera → unobservable

# --- NetworkX queries ---
print(pm.shortest_path("P1", "P6"))                  # ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
print(pm.is_connected("P3", "P4"))                   # True
print(pm.neighbors("P4"))                            # ['P3', 'P5']
print(pm.is_fully_connected())                       # True
```

This network is exactly what **`Generate demo dataset`** builds in the Streamlit sidebar (`app.py` sidebar, lines 1324–1338), and it is what the reconstruction engine runs against when you click **`Run demo incident`**. The engine then answers the core question: given camera sightings on P3 (before) and P5 (after), what is the probability the person was inside unobservable P4 at incident time?

---

## Summary

- The path network is an **undirected NetworkX graph** persisted to MongoDB: nodes = paths, edges = physical adjacency.
- `PathManager` (`modules/path_manager.py`) owns graph CRUD, connections, camera assignment, and NetworkX queries; it keeps an in-memory graph synced with the database.
- Observable vs unobservable is a **node attribute** — unobservable paths (like P4) produce no camera observations but are still traversable nodes in routes.
- Travel time derives from `length / average_speed`; route coverage and presence windows derive from graph routes computed by NetworkX.
- NetworkX provides `shortest_path`, `all_simple_paths`, `is_connected`, `connected_components`, and `spring_layout` — the algorithms that make incident-time reconstruction possible.
- The network is the spatial foundation that lets the reconstruction engine answer "was this person in the camera-free path at incident time?" using only before/after camera sightings.