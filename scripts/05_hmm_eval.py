"""Tune and evaluate HMM/Viterbi route reconstruction.

1. Decoder parameters (alpha, beta) are grid-searched on OUT-OF-FOLD
   classifier probabilities of the training trips (no leakage).
2. The final tuned classifier + decoder is evaluated on the 10
   student-testing trips: classifier-only vs classifier + HMM.
3. Folium maps of student-testing trips are written to results/maps/.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import ARTIFACTS, CACHE_DIR, RESULTS, STEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.hmm import RoadGraph, classifier_only, precompute_steps, viterbi  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import route_continuity, route_set_scores  # noqa: E402
from pathsnap.pipeline import PathSnapModel  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

OUT = os.path.join(RESULTS, "hmm")
ALPHAS = [0.5, 1.0, 2.0]
BETAS = [2.0, 5.0, 10.0, 20.0, 50.0]


def score(trip, obs, pred, labels):
    gt = labels[obs]
    m = gt >= 0
    pred_unique = pred[np.r_[True, pred[1:] != pred[:-1]]]
    return {"top1_acc": float((pred[m] == gt[m]).mean()),
            "continuity": route_continuity(pred, trip.arcs),
            **route_set_scores(pred_unique, trip.route)}


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---------- 1. tune decoder on OOF train predictions ----------
    oof = pd.read_pickle(os.path.join(CACHE_DIR, "oof_train.pkl"))
    prepared = []
    t0 = time.time()
    for td in list_trips(TRAIN_DIR):
        trip = load_trip(td)
        labels, _, _ = align_route(trip)
        cand = oof[oof["trip"] == trip.trip_id]
        steps, deltas = precompute_steps(trip, cand, RoadGraph(trip))
        prepared.append((trip, steps, deltas, labels))
    prep_s = time.time() - t0
    print(f"precomputed transitions for {len(prepared)} train trips in {prep_s:.0f}s", flush=True)

    grid = []
    for a in ALPHAS:
        for b in BETAS:
            accs, conts = [], []
            for trip, steps, deltas, labels in prepared:
                obs = np.array([s["obs"] for s in steps])
                r = score(trip, obs, viterbi(steps, deltas, alpha=a, beta=b), labels)
                accs.append(r["top1_acc"]); conts.append(r["continuity"])
            grid.append(dict(alpha=a, beta=b, oof_top1=np.mean(accs), oof_continuity=np.mean(conts)))
            print(grid[-1], flush=True)
    base = [score(t, np.array([s["obs"] for s in st]), classifier_only(st), l)["top1_acc"] for t, st, d, l in prepared]
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(OUT, "decoder_grid_oof_train.csv"), index=False)
    best = grid.sort_values("oof_top1", ascending=False).iloc[0]
    cfg = {"alpha": float(best.alpha), "beta": float(best.beta), "gap_s": 60.0, "gap_m": 500.0,
           "tuned_on": "out-of-fold classifier probabilities, 80 training trips",
           "oof_top1_hmm": float(best.oof_top1), "oof_top1_classifier_only": float(np.mean(base))}
    json.dump(cfg, open(os.path.join(ARTIFACTS, "hmm_config.json"), "w"), indent=2)
    print("chosen decoder config", cfg, flush=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for a, g in grid.groupby("alpha"):
        ax.plot(g.beta, g.oof_top1, "o-", label=f"alpha={a}")
    ax.axhline(np.mean(base), color="k", ls="--", label="classifier only")
    ax.set_xscale("log"); ax.set_xlabel("beta (m)"); ax.set_ylabel("OOF Top-1 accuracy (train)")
    ax.set_title("HMM decoder tuning on out-of-fold train predictions"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "decoder_tuning.png"), dpi=120); plt.close(fig)

    # ---------- 2. student-test evaluation ----------
    model = PathSnapModel()
    rows = []
    for td in list_trips(STEST_DIR):
        trip = load_trip(td)
        labels, _, _ = align_route(trip)          # reference used ONLY for scoring
        res = model.predict_trip(trip)
        for kind in ("classifier_arcs", "hmm_arcs"):
            r = score(trip, res["obs"], res[kind], labels)
            rows.append(dict(trip=trip.trip_id, method="classifier only" if kind == "classifier_arcs" else "classifier + HMM",
                             n_obs=len(res["obs"]), seconds=res["timing_s"]["total"], **r))
        trip_map(trip, pred_arcs=res["hmm_arcs"], ref_route=trip.route, classifier_arcs=res["classifier_arcs"],
                 title=f"Student-test trip {trip.trip_id}").save(os.path.join(RESULTS, "maps", f"student_test_{trip.trip_id}.html"))
        print(rows[-2], "\n", rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "student_test_hmm_vs_classifier.csv"), index=False)
    summary = df.groupby("method")[["top1_acc", "continuity", "route_precision", "route_recall", "route_f1", "seconds"]].mean()
    summary.to_csv(os.path.join(OUT, "student_test_summary.csv"))
    print(summary)
    plot_per_trip(df)


def plot_per_trip(df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, col, title in [(ax[0], "top1_acc", "Student-test Top-1 accuracy per trip"),
                          (ax[1], "continuity", "Route continuity per trip")]:
        df.pivot(index="trip", columns="method", values=col).plot.bar(ax=a, legend=False)
        a.set_ylim(0.5, 1.01); a.set_title(title); a.grid(axis="y", alpha=.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "hmm_vs_classifier.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
