"""End-to-end inference: raw trip folder -> predicted arc per GPS observation.

Used by the evaluation scripts and by the FastAPI backend. The reference
route is never read here; scoring happens separately (server side).
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .candidates import build_trip_table
from .config import ARTIFACTS
from .data import Trip, load_trip
from .hmm import RoadGraph, classifier_only, precompute_steps, viterbi


class PathSnapModel:
    def __init__(self, artifacts_dir: str = ARTIFACTS):
        bundle = joblib.load(os.path.join(artifacts_dir, "final_model.joblib"))
        self.model, self.features = bundle["model"], bundle["features"]
        cfg_path = os.path.join(artifacts_dir, "hmm_config.json")
        self.hmm_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"alpha": 1.0, "beta": 10.0}

    def predict_trip(self, trip: Trip) -> dict:
        t0 = time.time()
        cand = build_trip_table(trip)                         # candidates + features
        t1 = time.time()
        cand["p"] = self.model.predict_proba(cand[self.features].to_numpy(np.float32))[:, 1]
        t2 = time.time()
        steps, deltas = precompute_steps(trip, cand, RoadGraph(trip),
                                         gap_s=self.hmm_cfg.get("gap_s", 60.0), gap_m=self.hmm_cfg.get("gap_m", 500.0))
        clf_arcs = classifier_only(steps)
        hmm_arcs = viterbi(steps, deltas, alpha=self.hmm_cfg["alpha"], beta=self.hmm_cfg["beta"])
        t3 = time.time()
        return {
            "obs": np.array([s["obs"] for s in steps]),
            "classifier_arcs": clf_arcs,
            "hmm_arcs": hmm_arcs,
            "n_candidates": len(cand),
            "timing_s": {"candidates_features": t1 - t0, "classifier": t2 - t1, "hmm_viterbi": t3 - t2, "total": t3 - t0},
        }

    def predict_dir(self, trip_dir: str) -> tuple[Trip, dict]:
        trip = load_trip(trip_dir, with_route=False)
        return trip, self.predict_trip(trip)
