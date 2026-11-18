"""Derive one reference arc per GPS observation from the .route file.

The .route file is an ordered list of arcs, NOT one label per GPS row.
We align GPS points to route arcs with a monotone dynamic programme:

    D[i, j] = c(i, j) + min_{k <= j} D[i-1, k]

where c(i, j) is the (capped) distance of GPS point i to route arc j.
Monotonicity (the vehicle never goes back along its route) prevents a
point from being snapped to a later/earlier part of the route that
happens to pass nearby (loops, U-turns, parallel streets).

Points whose aligned arc is further than ``max_dist`` metres are labelled
-1 (no reliable reference) and are excluded from supervised metrics.
"""
from __future__ import annotations

import numpy as np

from .data import Trip, point_segment

LABEL_MAX_DIST = 75.0
COST_CAP = 100.0


def align_route(trip: Trip, max_dist: float = LABEL_MAX_DIST):
    route = trip.route
    n, m = len(trip.gps_xy), len(route)
    A = trip.node_xy[trip.arcs[route, 0]]
    B = trip.node_xy[trip.arcs[route, 1]]
    idx = np.arange(m)
    back = np.empty((n, m), dtype=np.int32)
    dist_all = np.empty((n, m), dtype=np.float32)
    prev = None
    for i in range(n):
        p = np.repeat(trip.gps_xy[i : i + 1], m, axis=0)
        d, _ = point_segment(p, A, B)
        dist_all[i] = d
        c = np.minimum(d, COST_CAP)
        if prev is None:
            cur = c
            back[i] = idx
        else:
            cm = np.minimum.accumulate(prev)
            arg = np.maximum.accumulate(np.where(prev <= cm, idx, 0))
            cur = c + cm
            back[i] = arg
        prev = cur
    # backtrack
    j = int(np.argmin(prev))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = int(back[i, j])
    d_assigned = dist_all[np.arange(n), path]
    labels = route[path].copy()
    labels[d_assigned > max_dist] = -1
    return labels, path, d_assigned
