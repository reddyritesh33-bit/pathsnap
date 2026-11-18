"""Folium map helpers."""
from __future__ import annotations

import folium
import numpy as np

from .data import Trip


def _arc_latlon(trip: Trip, arc_ids):
    a = trip.nodes[trip.arcs[arc_ids, 0]]
    b = trip.nodes[trip.arcs[arc_ids, 1]]
    return [[(p[1], p[0]), (q[1], q[0])] for p, q in zip(a, b)]


def _merge_chain(trip: Trip, arc_ids):
    """Collapse consecutive arcs into polylines for a lighter map."""
    lines, cur = [], []
    prev = None
    for a in arc_ids:
        u, v = trip.arcs[a]
        if prev is not None and prev == a:
            continue
        if cur and trip.arcs[prev, 1] == u:
            cur.append((trip.nodes[v, 1], trip.nodes[v, 0]))
        else:
            if cur:
                lines.append(cur)
            cur = [(trip.nodes[u, 1], trip.nodes[u, 0]), (trip.nodes[v, 1], trip.nodes[v, 0])]
        prev = a
    if cur:
        lines.append(cur)
    return lines


def trip_map(trip: Trip, pred_arcs=None, ref_route=None, classifier_arcs=None,
             network_margin_m: float = 150.0, max_network_arcs: int = 6000,
             title: str | None = None, focus: tuple[int, int] | None = None,
             show_classifier: bool = False) -> folium.Map:
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], tiles="OpenStreetMap", control_scale=True)
    s = slice(*focus) if focus else slice(None)
    m.fit_bounds([[lat[s].min(), lon[s].min()], [lat[s].max(), lon[s].max()]])

    # road network near the trajectory
    from scipy.spatial import cKDTree
    mid = (trip.arc_a + trip.arc_b) / 2
    near = cKDTree(trip.gps_xy).query(mid, distance_upper_bound=network_margin_m)[0] < np.inf
    ids = np.where(near)[0][:max_network_arcs]
    net = folium.FeatureGroup(name=f"Road network ({len(ids)} arcs near track)", show=True)
    for seg in _arc_latlon(trip, ids):
        folium.PolyLine(seg, color="#9e9e9e", weight=1.5, opacity=0.6).add_to(net)
    net.add_to(m)

    if ref_route is not None:
        fg = folium.FeatureGroup(name="Reference route (ground truth)")
        for line in _merge_chain(trip, ref_route):
            folium.PolyLine(line, color="#2e7d32", weight=7, opacity=0.55).add_to(fg)
        fg.add_to(m)
    if classifier_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only top-1 arcs", show=show_classifier)
        for line in _merge_chain(trip, classifier_arcs):
            folium.PolyLine(line, color="#ef6c00", weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)
    if pred_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)")
        for line in _merge_chain(trip, pred_arcs):
            folium.PolyLine(line, color="#d81b60", weight=3.5, opacity=0.95).add_to(fg)
        fg.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(lat) // 1500)
    for i in range(0, len(lat), step):
        folium.CircleMarker((lat[i], lon[i]), radius=1.8, color="#1565c0", fill=True, fill_opacity=0.6,
                            weight=0, tooltip=f"obs {i}  t={trip.track[i, 2]:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((lat[0], lon[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 10px;
         border:1px solid #999;border-radius:6px;font:12px sans-serif;">
      <b>{title or 'Trip ' + trip.trip_id}</b><br>
      <span style="color:#1565c0">&#9679;</span> raw GPS<br>
      <span style="color:#9e9e9e">&#9473;</span> road network<br>
      <span style="color:#2e7d32">&#9473;</span> reference route<br>
      <span style="color:#ef6c00">&#9473;</span> classifier-only<br>
      <span style="color:#d81b60">&#9473;</span> HMM/Viterbi prediction
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
