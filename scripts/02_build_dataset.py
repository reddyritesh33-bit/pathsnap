"""Build candidate/feature tables for the train and student_test partitions.

Output: cache/<partition>.pkl  (one row per (trip, obs, candidate arc))
        cache/labels_<partition>.pkl (per-observation reference arcs)
        results/eda/candidate_stats.csv
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.candidates import build_trip_table  # noqa: E402
from pathsnap.config import CACHE_DIR, RESULTS, STEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402


def process(trip_dir):
    t0 = time.time()
    trip = load_trip(trip_dir)
    labels, _, dist = align_route(trip)
    df = build_trip_table(trip, labels)
    lab = pd.DataFrame({"trip": trip.trip_id, "obs": np.arange(len(labels)),
                        "gt_arc": labels, "gt_dist": dist})
    covered = df.groupby("obs")["y"].max()
    valid = labels >= 0
    stats = dict(trip=trip.trip_id, n_obs=len(labels), n_valid=int(valid.sum()),
                 n_rows=len(df), cand_per_obs=len(df) / len(labels),
                 cand_miss_rate=float(1 - covered.reindex(np.where(valid)[0]).fillna(0).mean()),
                 secs=time.time() - t0)
    print(stats, flush=True)
    return df, lab, stats


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_stats = []
    for name, d in [("train", TRAIN_DIR), ("student_test", STEST_DIR)]:
        dirs = list_trips(d)
        with ProcessPoolExecutor(max_workers=max(os.cpu_count() - 1, 1)) as ex:
            out = list(ex.map(process, dirs))
        pd.concat([o[0] for o in out], ignore_index=True).to_pickle(os.path.join(CACHE_DIR, f"{name}.pkl"))
        pd.concat([o[1] for o in out], ignore_index=True).to_pickle(os.path.join(CACHE_DIR, f"labels_{name}.pkl"))
        for o in out:
            o[2]["partition"] = name
            all_stats.append(o[2])
    s = pd.DataFrame(all_stats)
    s.to_csv(os.path.join(RESULTS, "eda", "candidate_stats.csv"), index=False)
    print(s.groupby("partition")[["n_obs", "n_valid", "cand_per_obs", "cand_miss_rate"]].mean())


if __name__ == "__main__":
    main()
