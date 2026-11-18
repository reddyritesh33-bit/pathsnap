"""HMM + Viterbi route reconstruction.

Hidden states  : at time t, the candidate arcs C_t generated for GPS obs t.
Emissions      : log P(obs_t | arc) = alpha * log p_clf(arc | obs_t), where
                 p_clf is the classifier probability normalised over C_t.
Transitions    : Newson & Krumm style. For arcs a -> b between obs t, t+1
                     d_net = network driving distance from the projection of
                             obs t on a to the projection of obs t+1 on b
                     d_gc  = straight-line distance between the two GPS fixes
                 log P(b | a) = -|d_net - d_gc| / beta
                 Unreachable pairs (no path within the search limit) get
                 log P = -UNREACH.  A tiny backward move on the same arc is
                 treated as GPS jitter; a large one is penalised.
Gaps           : if dt > gap_s or d_gc > gap_m, the transition matrix is
                 flat (uniform), i.e. the chain restarts and each side of the
                 gap is decoded from its own evidence.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from .data import Trip

UNREACH = 50.0


class RoadGraph:
    def __init__(self, trip: Trip):
        a, b = trip.arcs[:, 0], trip.arcs[:, 1]
        self.arcs = trip.arcs
        self.L = np.linalg.norm(trip.node_xy[b] - trip.node_xy[a], axis=1)
        order = np.argsort(a, kind="stable")
        self.nbr = b[order]
        self.w = self.L[order]
        self.ptr = np.searchsorted(a[order], np.arange(len(trip.node_xy) + 1))
        self._cache: dict[int, tuple[float, dict]] = {}

    def sssp(self, src: int, limit: float) -> dict:
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        ptr, nbr, w = self.ptr, self.nbr, self.w
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for k in range(ptr[u], ptr[u + 1]):
                v = int(nbr[k]); nd = d + w[k]
                if nd < dist.get(v, np.inf) and nd <= limit:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._cache[src] = (limit, dist)
        return dist


def precompute_steps(trip: Trip, cand: pd.DataFrame, graph: RoadGraph | None = None,
                     gap_s: float = 60.0, gap_m: float = 500.0):
    """Per-step candidate lists + (d_net - d_gc) matrices (independent of
    alpha/beta, so the decoder can be tuned cheaply)."""
    if graph is None:
        graph = RoadGraph(trip)
    cand = cand.sort_values(["obs", "rank"])
    groups = {o: g for o, g in cand.groupby("obs", sort=True)}
    obs_ids = np.array(sorted(groups))
    steps = []
    g, tt = trip.gps_xy, trip.track[:, 2]
    for i, o in enumerate(obs_ids):
        c = groups[o]
        steps.append({"obs": int(o), "arcs": c["arc"].to_numpy(),
                      "t": c["t_proj"].to_numpy(), "p": c["p"].to_numpy()})
    deltas = [None]
    for i in range(1, len(steps)):
        s0, s1 = steps[i - 1], steps[i]
        o0, o1 = s0["obs"], s1["obs"]
        dgc = float(np.linalg.norm(g[o1] - g[o0]))
        dt = float(tt[o1] - tt[o0])
        if dt > gap_s or dgc > gap_m:
            deltas.append("gap")
            continue
        limit = 3.0 * dgc + 150.0
        A, B = s0["arcs"], s1["arcs"]
        La, Lb = graph.L[A], graph.L[B]
        D = np.full((len(A), len(B)), np.inf)
        for ia, a in enumerate(A):
            rem = (1.0 - s0["t"][ia]) * La[ia]
            sp = graph.sssp(int(graph.arcs[a, 1]), limit)
            for ib, b in enumerate(B):
                if a == b:
                    adv = (s1["t"][ib] - s0["t"][ia]) * La[ia]
                    D[ia, ib] = adv if adv >= -10.0 else -adv * 4.0
                else:
                    d0 = sp.get(int(graph.arcs[b, 0]))
                    if d0 is not None:
                        D[ia, ib] = rem + d0 + s1["t"][ib] * Lb[ib]
        deltas.append(np.abs(D - dgc))
    return steps, deltas


def viterbi(steps, deltas, alpha: float = 1.0, beta: float = 10.0,
            p_floor: float = 1e-4) -> np.ndarray:
    """Returns the chosen arc for every step."""
    n = len(steps)
    em = [alpha * np.log(np.clip(s["p"] / max(s["p"].sum(), 1e-12), p_floor, 1.0)) for s in steps]
    score = em[0]
    back = [None]
    for i in range(1, n):
        d = deltas[i]
        if isinstance(d, str):  # gap -> restart
            bp = np.full(len(em[i]), int(np.argmax(score)))
            score = em[i] + score.max()
        else:
            T = np.where(np.isfinite(d), -d / beta, -UNREACH)
            tot = score[:, None] + T
            bp = np.argmax(tot, axis=0)
            score = tot[bp, np.arange(tot.shape[1])] + em[i]
        back.append(bp)
        score = score - score.max()  # keep numbers bounded
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    return np.array([steps[i]["arcs"][path[i]] for i in range(n)])


def classifier_only(steps) -> np.ndarray:
    return np.array([s["arcs"][int(np.argmax(s["p"]))] for s in steps])
