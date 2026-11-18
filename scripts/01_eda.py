"""Exploratory data analysis of the raw PathSnap data (train + student_test).

Outputs in results/eda/: inventory CSV, summary stats, plots, Folium maps.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import RESULTS, STEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.data import list_trips, load_trip, point_segment  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

OUT = os.path.join(RESULTS, "eda")


def inspect(trip_dir, partition):
    tid = os.path.basename(trip_dir)
    files = {ext: os.path.join(trip_dir, f"{tid}.{ext}") for ext in ["track", "nodes", "arcs", "route"]}
    trip = load_trip(trip_dir)
    tr = trip.track
    dt = np.diff(tr[:, 2])
    seg = np.linalg.norm(np.diff(trip.gps_xy, axis=0), axis=1)
    speed = seg / np.maximum(dt, 1e-9)
    N, M = len(trip.nodes), len(trip.arcs)
    G = coo_matrix((np.ones(M), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(N, N))
    n_weak, lab_w = connected_components(G, directed=True, connection="weak")
    n_strong, _ = connected_components(G, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=N)
    r = trip.route
    route_conn = float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0
    labels, _, dist = align_route(trip)
    arc_len = np.linalg.norm(trip.arc_b - trip.arc_a, axis=1)
    return dict(
        partition=partition, trip=tid,
        **{f"size_kb_{k}": os.path.getsize(v) / 1024 for k, v in files.items() if os.path.exists(v)},
        n_gps=len(tr), duration_s=tr[-1, 2] - tr[0, 2],
        nan_values=int(np.isnan(tr).sum()), unsorted_ts=int((dt < 0).sum()), dup_ts=int((dt == 0).sum()),
        pct_dt_1s=float(np.mean(np.isclose(dt, 1.0))) if len(dt) else np.nan,
        max_gap_s=float(dt.max()) if len(dt) else 0, n_gaps_gt10s=int((dt > 10).sum()),
        max_jump_m=float(seg.max()) if len(seg) else 0, n_speed_gt60ms=int((speed > 60).sum()),
        track_len_km=float(seg.sum() / 1000),
        lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(), lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(),
        n_nodes=N, n_arcs=M, isolated_nodes=int((deg == 0).sum()),
        weak_components=n_weak, largest_weak_frac=float(np.bincount(lab_w).max() / N),
        strong_components=n_strong, median_arc_len_m=float(np.median(arc_len)),
        route_len=len(r), route_connected_frac=route_conn, route_unique_frac=len(set(r)) / len(r),
        gps_to_route_median_m=float(np.median(dist)), gps_to_route_p95_m=float(np.percentile(dist, 95)),
        unlabelled_frac=float(np.mean(labels < 0)),
    ), trip, dt


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, dts, trips = [], {"train": [], "student_test": []}, {}
    for part, d in [("train", TRAIN_DIR), ("student_test", STEST_DIR)]:
        for td in list_trips(d):
            row, trip, dt = inspect(td, part)
            rows.append(row); dts[part].append(dt)
            trips[(part, trip.trip_id)] = (trip.track[:, 0], trip.track[:, 1])
            print(part, trip.trip_id, flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    summ = inv.groupby("partition").agg(
        trips=("trip", "count"), gps_total=("n_gps", "sum"), gps_mean=("n_gps", "mean"),
        gps_min=("n_gps", "min"), gps_max=("n_gps", "max"), nan_values=("nan_values", "sum"),
        pct_dt_1s=("pct_dt_1s", "mean"), trips_gap_gt60s=("max_gap_s", lambda s: int((s > 60).sum())),
        arcs_mean=("n_arcs", "mean"), arcs_max=("n_arcs", "max"), isolated_nodes=("isolated_nodes", "sum"),
        weak_comp_mean=("weak_components", "mean"), route_connected=("route_connected_frac", "mean"),
        gps_to_route_median=("gps_to_route_median_m", "median"), unlabelled_frac=("unlabelled_frac", "mean"))
    summ.T.to_csv(os.path.join(OUT, "summary.csv"))
    print(summ.T)
    anomalies = inv[(inv.max_gap_s > 60) | (inv.unlabelled_frac > 0.05) | (inv.n_speed_gt60ms > 0) | (inv.route_connected_frac < 1)]
    anomalies[["partition", "trip", "n_gps", "max_gap_s", "max_jump_m", "n_speed_gt60ms", "route_connected_frac",
               "unlabelled_frac", "gps_to_route_p95_m"]].to_csv(os.path.join(OUT, "anomalies.csv"), index=False)

    # ---- plots ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for part, c in [("train", "C0"), ("student_test", "C1")]:
        s = inv[inv.partition == part]
        ax[0, 0].hist(s.n_gps, bins=30, alpha=.6, label=part, color=c)
        ax[0, 1].hist(np.log10(s.n_arcs), bins=30, alpha=.6, label=part, color=c)
        ax[0, 2].hist(s.gps_to_route_median_m, bins=30, alpha=.6, label=part, color=c)
        allg = np.concatenate(dts[part])
        ax[1, 0].hist(np.clip(allg, 0, 10), bins=40, alpha=.6, label=part, color=c, log=True)
        ax[1, 1].hist(s.track_len_km, bins=30, alpha=.6, label=part, color=c)
    ax[0, 0].set_title("GPS observations per trip"); ax[0, 1].set_title("log10(#arcs) in trip network")
    ax[0, 2].set_title("median GPS-to-route distance (m)"); ax[1, 0].set_title("sampling interval (s, clipped at 10, log count)")
    ax[1, 1].set_title("trajectory length (km)")
    for (part, tid), (lo, la) in trips.items():
        ax[1, 2].plot(lo, la, lw=.5, color="C0" if part == "train" else "C1")
    ax[1, 2].set_title("all trajectories (lon/lat) - blue train, orange student_test")
    for a in ax.ravel()[:5]:
        a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eda_overview.png"), dpi=120); plt.close(fig)

    # missing-value / data-quality plot
    fig, ax = plt.subplots(figsize=(10, 3.5))
    q = inv.set_index("trip")[["nan_values", "dup_ts", "n_gaps_gt10s", "n_speed_gt60ms"]]
    q.plot.bar(ax=ax, stacked=True, width=.9)
    ax.set_xticks([]); ax.set_title("Data-quality issues per trip (NaNs, duplicate timestamps, gaps>10s, speed>60 m/s)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "data_quality.png"), dpi=120); plt.close(fig)

    # one trip network + GPS + route (static) and folium
    tid = "00000001"
    trip = load_trip(os.path.join(TRAIN_DIR, tid))
    fig, ax = plt.subplots(figsize=(8, 8))
    from scipy.spatial import cKDTree
    mid = (trip.arc_a + trip.arc_b) / 2
    near = cKDTree(trip.gps_xy).query(mid, distance_upper_bound=600)[0] < np.inf
    for a, b in zip(trip.arc_a[near], trip.arc_b[near]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.75", lw=.6)
    for r in trip.route:
        a, b = trip.node_xy[trip.arcs[r]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="green", lw=3, alpha=.6)
    ax.scatter(*trip.gps_xy.T, s=2, color="C0", zorder=3)
    ax.set_aspect("equal"); ax.set_title(f"Train trip {tid}: road network (grey), reference route (green), GPS (blue) - metres")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"network_trip_{tid}.png"), dpi=120); plt.close(fig)
    trip_map(trip, ref_route=trip.route, title=f"EDA - train trip {tid}").save(os.path.join(RESULTS, "maps", f"eda_train_{tid}.html"))
    t2 = load_trip(os.path.join(TRAIN_DIR, "00000002"))
    trip_map(t2, ref_route=t2.route, title="EDA - train trip 00000002 (GPS outliers)").save(
        os.path.join(RESULTS, "maps", "eda_train_00000002.html"))


if __name__ == "__main__":
    main()
