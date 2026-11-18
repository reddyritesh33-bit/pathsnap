"""Candidate road-segment generation and feature engineering.

For every GPS observation we retrieve up to K nearby directed arcs
(hidden-state candidates).  Each (observation, candidate) pair becomes one
row; the classifier predicts whether that candidate is the true arc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import Trip, point_segment

SAMPLE_STEP = 10.0   # m between sample points along an arc (spatial index)
RADIUS = 80.0        # m candidate search radius
K = 10               # max candidates per observation

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio", "n_cand",
    "head_diff", "head_cos", "head_diff_long", "speed",
    "arc_len", "t_proj", "t_clipped", "has_reverse",
    "dist_prev", "dist_next", "dist_win_mean", "dist_win_rank",
    "out_degree", "in_degree", "dt_prev",
]


class ArcIndex:
    """KD-tree over points sampled along every arc of one trip network."""

    def __init__(self, trip: Trip, step: float = SAMPLE_STEP):
        a, b = trip.arc_a, trip.arc_b
        L = np.linalg.norm(b - a, axis=1)
        nseg = np.maximum(np.ceil(L / step).astype(int), 1)
        arc_id = np.repeat(np.arange(len(L)), nseg + 1)
        offs = np.concatenate([np.arange(k + 1) / k for k in nseg])
        self.samples = a[arc_id] + offs[:, None] * (b - a)[arc_id]
        self.sample_arc = arc_id
        self.tree = cKDTree(self.samples)
        self.step = step
        self.arc_len = L
        self.a, self.b = a, b
        heading = np.arctan2((b - a)[:, 0], (b - a)[:, 1])  # bearing, 0 = north
        self.arc_heading = heading
        n_nodes = len(trip.node_xy)
        self.out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes)
        self.in_deg = np.bincount(trip.arcs[:, 1], minlength=n_nodes)
        # reverse-arc lookup (u,v) -> (v,u)
        key = trip.arcs[:, 0] * (n_nodes + 1) + trip.arcs[:, 1]
        rkey = trip.arcs[:, 1] * (n_nodes + 1) + trip.arcs[:, 0]
        self.has_reverse = np.isin(rkey, key)


def _gps_heading(xy: np.ndarray, w: int):
    n = len(xy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[i1] - xy[i0]
    return np.arctan2(d[:, 0], d[:, 1]), np.linalg.norm(d, axis=1)


def _angdiff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def generate_candidates(trip: Trip, index: ArcIndex | None = None,
                        radius: float = RADIUS, k: int = K):
    """Return DataFrame with one row per (obs, candidate arc) + features."""
    if index is None:
        index = ArcIndex(trip)
    g = trip.gps_xy
    n = len(g)
    lists = index.tree.query_ball_point(g, r=radius + index.step / 2)
    pi = np.repeat(np.arange(n), [len(l) for l in lists])
    arcs = index.sample_arc[np.concatenate([np.asarray(l, dtype=int) for l in lists])] if len(pi) else np.array([], int)
    # fallback for observations with nothing in range: nearest samples
    empty = np.where([len(l) == 0 for l in lists])[0]
    if len(empty):
        _, nn = index.tree.query(g[empty], k=40)
        pi = np.concatenate([pi, np.repeat(empty, 40)])
        arcs = np.concatenate([arcs, index.sample_arc[nn.ravel()]])
    key = np.unique(pi.astype(np.int64) * (len(index.arc_len) + 1) + arcs)
    pi = key // (len(index.arc_len) + 1)
    arcs = key % (len(index.arc_len) + 1)
    d, t = point_segment(g[pi], index.a[arcs], index.b[arcs])
    order = np.lexsort((d, pi))
    pi, arcs, d, t = pi[order], arcs[order], d[order], t[order]
    first = np.r_[True, pi[1:] != pi[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(pi)), 0))
    rank = np.arange(len(pi)) - grp_start
    keep = (rank < k) & ((d <= radius) | (rank == 0) | np.isin(pi, empty))
    keep &= rank < k
    pi, arcs, d, t, rank = pi[keep], arcs[keep], d[keep], t[keep], rank[keep]

    df = pd.DataFrame({"obs": pi, "arc": arcs, "dist": d, "t_proj": t, "rank": rank})
    return add_features(df, trip, index)


def add_features(df: pd.DataFrame, trip: Trip, index: ArcIndex) -> pd.DataFrame:
    g = trip.gps_xy
    n = len(g)
    pi = df["obs"].to_numpy()
    arcs = df["arc"].to_numpy()
    grp = df.groupby("obs")["dist"]
    df["n_cand"] = grp.transform("size")
    dmin = grp.transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio"] = df["dist"] / (dmin + 1.0)

    h_short, disp = _gps_heading(g, 1)
    h_long, _ = _gps_heading(g, 4)
    tt = trip.track[:, 2]
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    speed = disp / np.maximum(tt[i1] - tt[i0], 1.0)
    ah = index.arc_heading[arcs]
    df["head_diff"] = np.degrees(_angdiff(h_short[pi], ah))
    df["head_cos"] = np.cos(_angdiff(h_short[pi], ah))
    df["head_diff_long"] = np.degrees(_angdiff(h_long[pi], ah))
    df["speed"] = speed[pi]

    df["arc_len"] = index.arc_len[arcs]
    df["t_clipped"] = ((df["t_proj"] <= 0.0) | (df["t_proj"] >= 1.0)).astype(np.int8)
    df["has_reverse"] = index.has_reverse[arcs].astype(np.int8)
    a, b = index.a[arcs], index.b[arcs]
    dp, _ = point_segment(g[np.clip(pi - 1, 0, n - 1)], a, b)
    dn, _ = point_segment(g[np.clip(pi + 1, 0, n - 1)], a, b)
    win = [df["dist"].to_numpy()]
    for off in (-3, -2, -1, 1, 2, 3):
        dd, _ = point_segment(g[np.clip(pi + off, 0, n - 1)], a, b)
        win.append(dd)
    df["dist_prev"] = dp
    df["dist_next"] = dn
    df["dist_win_mean"] = np.mean(win, axis=0)
    df["dist_win_rank"] = df.groupby("obs")["dist_win_mean"].rank(method="first") - 1
    to_node = trip.arcs[arcs, 1]
    from_node = trip.arcs[arcs, 0]
    df["out_degree"] = index.out_deg[to_node]
    df["in_degree"] = index.in_deg[from_node]
    dtp = np.r_[1.0, np.diff(tt)]
    df["dt_prev"] = dtp[pi]
    return df


def build_trip_table(trip: Trip, labels: np.ndarray | None = None, **kw) -> pd.DataFrame:
    idx = ArcIndex(trip)
    df = generate_candidates(trip, idx, **kw)
    df.insert(0, "trip", trip.trip_id)
    if labels is not None:
        df["gt_arc"] = labels[df["obs"].to_numpy()]
        df["y"] = (df["arc"] == df["gt_arc"]).astype(np.int8)
    return df
