"""Raw-data loading and coordinate projection for PathSnap trips.

Each trip folder <id>/ contains:
  <id>.track  lon lat t          (GPS observations, ~1 Hz)
  <id>.nodes  lon lat            (row index = node_id)
  <id>.arcs   from_node to_node  (row index = arc_id, directed)
  <id>.route  arc_id             (chronological reference route; optional at inference)

Every trip ships its own road network, so all processing is per trip.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

EARTH_R = 6_371_000.0  # metres


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (n,3) lon, lat, t
    nodes: np.ndarray            # (N,2) lon, lat
    arcs: np.ndarray             # (M,2) from, to
    route: np.ndarray | None     # (R,) arc ids or None
    # projected (filled by project())
    lat0: float = 0.0
    lon0: float = 0.0
    gps_xy: np.ndarray = field(default=None, repr=False)
    node_xy: np.ndarray = field(default=None, repr=False)

    # ---- projection -------------------------------------------------
    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        """Local equirectangular projection centred on the trip (metres).

        At city scale (< ~50 km) the distortion vs. a geodesic is < 0.1 %,
        which is far below GPS noise (~5-15 m), and it lets us use fast
        Euclidean geometry for point-to-segment distances.
        """
        k = np.pi / 180.0
        x = (lonlat[:, 0] - self.lon0) * k * EARTH_R * np.cos(self.lat0 * k)
        y = (lonlat[:, 1] - self.lat0) * k * EARTH_R
        return np.c_[x, y]

    def project(self) -> "Trip":
        self.lat0 = float(self.track[:, 1].mean())
        self.lon0 = float(self.track[:, 0].mean())
        self.gps_xy = self.to_xy(self.track[:, :2])
        self.node_xy = self.to_xy(self.nodes)
        return self

    # ---- arc geometry -----------------------------------------------
    @property
    def arc_a(self):
        return self.node_xy[self.arcs[:, 0]]

    @property
    def arc_b(self):
        return self.node_xy[self.arcs[:, 1]]


def _load(path, dtype=float):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2 if dtype is float else 1)
    return arr


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(os.path.normpath(trip_dir))
    track = np.loadtxt(os.path.join(trip_dir, f"{tid}.track"), ndmin=2)
    nodes = np.loadtxt(os.path.join(trip_dir, f"{tid}.nodes"), ndmin=2)
    arcs = np.loadtxt(os.path.join(trip_dir, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
    route = None
    rpath = os.path.join(trip_dir, f"{tid}.route")
    if with_route and os.path.exists(rpath):
        route = np.loadtxt(rpath, dtype=np.int64, ndmin=1)
    # make sure GPS is chronologically ordered
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    return Trip(tid, track, nodes, arcs, route).project()


def list_trips(partition_dir: str) -> list[str]:
    return sorted(
        os.path.join(partition_dir, d)
        for d in os.listdir(partition_dir)
        if os.path.isdir(os.path.join(partition_dir, d))
    )


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised distance from points p to segments a-b (all (k,2)).

    Returns (distance, t) where t in [0,1] is the projection fraction.
    """
    v = b - a
    w = p - a
    vv = np.maximum((v * v).sum(1), 1e-9)
    t = np.clip((w * v).sum(1) / vv, 0.0, 1.0)
    proj = a + t[:, None] * v
    return np.linalg.norm(p - proj, axis=1), t
