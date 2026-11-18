"""PathSnap FastAPI backend (Laptop 1).

Run:
    set PATHSNAP_EVAL_DIR=<folder containing the 10 evaluation trip folders>
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            liveness + loaded model info
    GET  /trips             evaluation trip ids (no labels)
    POST /predict           {"trip_id": "..."} -> predicted route + server-side scores

The reference .route is read ONLY after prediction, inside score_trip(), to
compute accuracy. It is never passed to the model and never returned to the UI.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import STEST_DIR  # noqa: E402
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import route_continuity, route_set_scores  # noqa: E402
from pathsnap.pipeline import PathSnapModel  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", STEST_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = PathSnapModel()
TRIPS = {os.path.basename(d): d for d in list_trips(EVAL_DIR)}
log.info("Loaded model (%s) | HMM config %s | %d evaluation trips from %s",
         type(MODEL.model).__name__, MODEL.hmm_cfg, len(TRIPS), EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def _lines(trip, arc_seq):
    """Collapse the per-observation arc sequence into lat/lon polylines."""
    lines, cur, prev = [], [], None
    for a in arc_seq:
        if a == prev:
            continue
        u, v = trip.arcs[a]
        pu = [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])]
        pv = [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]
        if cur and prev is not None and trip.arcs[prev, 1] == u:
            cur.append(pv)
        else:
            if cur:
                lines.append(cur)
            cur = [pu, pv]
        prev = a
    if cur:
        lines.append(cur)
    return lines


def score_trip(trip_dir, obs, pred):
    """Server-side scoring against the reference route (never sent to UI)."""
    ref_trip = load_trip(trip_dir, with_route=True)
    if ref_trip.route is None:
        return None
    labels, _, _ = align_route(ref_trip)
    gt = labels[obs]
    m = gt >= 0
    uniq = pred[np.r_[True, pred[1:] != pred[:-1]]]
    return {"top1_accuracy": float((pred[m] == gt[m]).mean()), "n_scored_obs": int(m.sum()),
            **route_set_scores(uniq, ref_trip.route)}


@app.get("/health")
def health():
    return {"status": "ok", "model": type(MODEL.model).__name__, "hmm": MODEL.hmm_cfg, "n_trips": len(TRIPS)}


@app.get("/trips")
def trips():
    log.info("GET /trips -> %d trips", len(TRIPS))
    return {"trips": sorted(TRIPS)}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in TRIPS:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    log.info("POST /predict trip=%s  -> running pipeline", req.trip_id)
    trip, res = MODEL.predict_dir(TRIPS[req.trip_id])          # route NOT loaded here
    obs, hmm, clf = res["obs"], res["hmm_arcs"], res["classifier_arcs"]
    s_hmm = score_trip(TRIPS[req.trip_id], obs, hmm)
    s_clf = score_trip(TRIPS[req.trip_id], obs, clf)
    step = max(1, len(trip.track) // 2000)
    out = {
        "trip_id": req.trip_id,
        "n_gps": int(len(trip.track)),
        "n_candidates": int(res["n_candidates"]),
        "gps": trip.track[::step, [1, 0]].round(7).tolist(),
        "predicted_route": _lines(trip, hmm),
        "predicted_arc_per_obs": [int(a) for a in hmm],
        "metrics": {
            "hmm": s_hmm,
            "classifier_only": s_clf,
            "route_continuity_hmm": route_continuity(hmm, trip.arcs),
            "route_continuity_classifier": route_continuity(clf, trip.arcs),
        },
        "timing_s": {**{k: round(v, 3) for k, v in res["timing_s"].items()}, "request_total": round(time.time() - t0, 3)},
    }
    acc = s_hmm["top1_accuracy"] if s_hmm else float("nan")
    log.info("POST /predict trip=%s  done | gps=%d | top1(HMM)=%.4f | top1(clf)=%.4f | continuity=%.4f | %.2fs",
             req.trip_id, len(trip.track), acc, s_clf["top1_accuracy"] if s_clf else float("nan"),
             out["metrics"]["route_continuity_hmm"], time.time() - t0)
    return out
