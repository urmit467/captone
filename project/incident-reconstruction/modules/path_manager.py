
from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from database.db_operations import DBOperations
from utils.helpers import clamp

logger = logging.getLogger(__name__)


class PathManager:
    """CRUD + graph operations for the connected path network."""

    def __init__(self, db_ops: Optional[DBOperations] = None) -> None:
        self.db = db_ops or DBOperations()
        self.graph = nx.Graph()
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Loading / syncing with MongoDB                                     #
    # ------------------------------------------------------------------ #
    def load_from_db(self) -> nx.Graph:
        """Load all paths and connections from MongoDB into the graph."""
        self.graph.clear()
        paths = self.db.list_paths()
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
                    self.graph.add_node(
                        cp,
                        name=cp,
                        length=0.0,
                        average_speed=1.0,
                        is_observable=False,
                        camera_ids=[],
                    )
                if not self.graph.has_edge(pid, cp):
                    self.graph.add_edge(pid, cp)
        self._loaded = True
        logger.info("Loaded path network: %d nodes, %d edges",
                    self.graph.number_of_nodes(), self.graph.number_of_edges())
        return self.graph

    def refresh(self) -> nx.Graph:
        return self.load_from_db()

    def is_loaded(self) -> bool:
        return self._loaded and self.graph.number_of_nodes() > 0

    # ------------------------------------------------------------------ #
    # Path CRUD                                                          #
    # ------------------------------------------------------------------ #
    def path_exists(self, path_id: str) -> bool:
        return self.db.get_path(path_id) is not None

    def add_path(
        self,
        path_id: str,
        name: str,
        length: float,
        average_speed: float,
        is_observable: bool = True,
    ) -> bool:
        if length <= 0:
            raise ValueError("Path length must be positive")
        if average_speed <= 0:
            raise ValueError("Average walking speed must be positive")
        if self.path_exists(path_id):
            raise ValueError(f"Path already exists: {path_id}")
        self.db.insert_path(
            path_id=path_id,
            name=name,
            length=length,
            average_speed=average_speed,
            is_observable=is_observable,
            connected_paths=[],
        )
        self.graph.add_node(
            path_id,
            name=name,
            length=float(length),
            average_speed=float(average_speed),
            is_observable=bool(is_observable),
            camera_ids=[],
        )
        logger.info("Added path %s (%s)", path_id, name)
        return True

    def update_path(
        self,
        path_id: str,
        name: Optional[str] = None,
        length: Optional[float] = None,
        average_speed: Optional[float] = None,
        is_observable: Optional[bool] = None,
    ) -> bool:
        if not self.path_exists(path_id):
            raise ValueError(f"Unknown path: {path_id}")
        fields: dict = {}
        if name is not None:
            fields["name"] = name
        if length is not None:
            if length <= 0:
                raise ValueError("Path length must be positive")
            fields["length"] = float(length)
        if average_speed is not None:
            if average_speed <= 0:
                raise ValueError("Average walking speed must be positive")
            fields["average_speed"] = float(average_speed)
        if is_observable is not None:
            fields["is_observable"] = bool(is_observable)
        if not fields:
            return True
        ok = self.db.update_path(path_id, fields)
        if ok:
            # Sync the in-memory graph attributes.
            for k, v in fields.items():
                self.graph.nodes[path_id][k] = v
        return ok

    def delete_path(self, path_id: str) -> bool:
        if not self.path_exists(path_id):
            return False
        # Remove all connections involving this path first.
        for neighbor in list(self.graph.neighbors(path_id)):
            self.disconnect_paths(path_id, neighbor)
        ok = self.db.delete_path(path_id)
        if ok and path_id in self.graph:
            self.graph.remove_node(path_id)
        return ok

    def get_path(self, path_id: str) -> Optional[dict]:
        return self.db.get_path(path_id)

    def list_paths(self) -> list[dict]:
        return self.db.list_paths()

    def observable_paths(self) -> list[dict]:
        return [p for p in self.list_paths() if p.get("is_observable")]

    def unobservable_paths(self) -> list[dict]:
        return [p for p in self.list_paths() if not p.get("is_observable")]

    # ------------------------------------------------------------------ #
    # Connections                                                        #
    # ------------------------------------------------------------------ #
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
        if path_id_b not in conns_a:
            conns_a.append(path_id_b)
        if path_id_a not in conns_b:
            conns_b.append(path_id_a)
        self.db.update_path(path_id_a, {"connected_paths": conns_a})
        self.db.update_path(path_id_b, {"connected_paths": conns_b})
        self.graph.add_edge(path_id_a, path_id_b)
        logger.info("Connected %s <-> %s", path_id_a, path_id_b)
        return True

    def disconnect_paths(self, path_id_a: str, path_id_b: str) -> bool:
        if not (self.path_exists(path_id_a) and self.path_exists(path_id_b)):
            return False
        pa = self.db.get_path(path_id_a)
        pb = self.db.get_path(path_id_b)
        conns_a = [c for c in (pa.get("connected_paths") or []) if c != path_id_b]
        conns_b = [c for c in (pb.get("connected_paths") or []) if c != path_id_a]
        self.db.update_path(path_id_a, {"connected_paths": conns_a})
        self.db.update_path(path_id_b, {"connected_paths": conns_b})
        if self.graph.has_edge(path_id_a, path_id_b):
            self.graph.remove_edge(path_id_a, path_id_b)
        return True

    # ------------------------------------------------------------------ #
    # Cameras                                                            #
    # ------------------------------------------------------------------ #
    def assign_camera(self, path_id: str, camera_id: str) -> bool:
        if not self.path_exists(path_id):
            raise ValueError(f"Unknown path: {path_id}")
        path = self.db.get_path(path_id)
        cam_ids = list(path.get("camera_ids") or [])
        if camera_id in cam_ids:
            return False
        cam_ids.append(camera_id)
        self.db.update_path(path_id, {"camera_ids": cam_ids})
        # Keep the camera's path_id reference in sync if the camera doc exists.
        cam = self.db.get_camera(camera_id)
        if cam:
            self.db.update_camera(camera_id, {"path_id": path_id})
        if path_id in self.graph:
            self.graph.nodes[path_id]["camera_ids"] = cam_ids
        return True

    def unassign_camera(self, path_id: str, camera_id: str) -> bool:
        path = self.db.get_path(path_id)
        if not path:
            return False
        cam_ids = [c for c in (path.get("camera_ids") or []) if c != camera_id]
        self.db.update_path(path_id, {"camera_ids": cam_ids})
        if path_id in self.graph:
            self.graph.nodes[path_id]["camera_ids"] = cam_ids
        return True

    def cameras_for_path(self, path_id: str) -> list[dict]:
        return self.db.cameras_on_path(path_id)

    # ------------------------------------------------------------------ #
    # Graph queries                                                      #
    # ------------------------------------------------------------------ #
    def neighbors(self, path_id: str) -> list[str]:
        if path_id not in self.graph:
            return []
        return sorted(self.graph.neighbors(path_id))

    def is_connected(self, path_id_a: str, path_id_b: str) -> bool:
        return self.graph.has_edge(path_id_a, path_id_b)

    def shortest_path(self, source: str, target: str) -> Optional[list[str]]:
        if source not in self.graph or target not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def all_simple_paths(self, source: str, target: str, cutoff: int = 8) -> list[list[str]]:
        if source not in self.graph or target not in self.graph:
            return []
        return list(nx.all_simple_paths(self.graph, source, target, cutoff=cutoff))

    def has_route(self, source: str, target: str) -> bool:
        return self.shortest_path(source, target) is not None

    def connected_components(self) -> list[list[str]]:
        return [sorted(c) for c in nx.connected_components(self.graph)]

    def is_fully_connected(self) -> bool:
        return nx.is_connected(self.graph)

    # ------------------------------------------------------------------ #
    # Visualization data                                                 #
    # ------------------------------------------------------------------ #
    def network_layout(self, seed: int = 42):
        """Return Plotly-friendly layout positions for the graph."""
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.spring_layout(self.graph, seed=seed)