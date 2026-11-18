"""Architecture diagrams for the Phase 2 report (results/phase2/).

1. pipeline.png        – offline training pipeline and online inference pipeline
2. two_laptop_flow.png – request flow between Laptop 2 (UI) and Laptop 1 (API)
3. hmm_trellis.png     – schematic of the HMM trellis and Viterbi path
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import ARTIFACTS, RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "phase2")
C = {"data": "#e3f2fd", "proc": "#fff3e0", "model": "#f3e5f5", "out": "#e8f5e9", "gt": "#ffebee",
     "edge": "#37474f", "accent": "#d81b60"}


def box(ax, x, y, w, h, text, fc, fs=9, bold=False, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec or C["edge"], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, text=None, color=None, fs=8, style="-|>", ls="-", off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                                 color=color or C["edge"], lw=1.3, linestyle=ls))
    if text:
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, ha="center", va="bottom",
                fontsize=fs, color=color or C["edge"])


def pipeline(cfg):
    fig, ax = plt.subplots(figsize=(15, 7.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(0.2, 6.85, "OFFLINE – training & tuning (80 train trips; student-test only for comparison)",
            fontsize=11, fontweight="bold")
    xs = [0.2, 2.4, 4.6, 6.8, 9.0, 11.2]
    W = 1.9
    steps = [("Raw trips\n.track .nodes\n.arcs .route", C["data"]),
             ("Per-trip metric\nprojection +\nmonotone DP labels", C["proc"]),
             ("Candidates\nKD-tree, r=80 m, K≤10\n+ 20 features", C["proc"]),
             ("8 classifiers\ntrip-grouped 5-fold CV\n→ LightGBM", C["model"]),
             ("Bayesian tuning\nOptuna TPE, 30 trials\nobjective: CV Macro F1", C["model"]),
             (f"HMM decoder tuning\nOOF probabilities\nα={cfg['alpha']}, β={cfg['beta']:.0f} m", C["model"])]
    for (t, fc), x in zip(steps, xs):
        box(ax, x, 4.9, W, 1.55, t, fc, fs=8.5)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + W, 5.67, b, 5.67)
    box(ax, 13.4, 4.9, 1.45, 1.55, "artifacts/\nfinal_model.joblib\nhmm_config.json", C["out"], fs=8)
    arrow(ax, 11.2 + W, 5.67, 13.4, 5.67)

    ax.plot([0.1, 14.9], [4.45, 4.45], color="#b0bec5", lw=1, ls="--")
    ax.text(0.2, 4.0, "ONLINE – inference for one trip (inside the API, Laptop 1)", fontsize=11, fontweight="bold")
    ys = 1.9
    on = [("Trip folder\n(no .route read)", C["data"]),
          ("Projection\n+ candidates\n+ features", C["proc"]),
          ("LightGBM\nP(arc correct)\nnormalised per point", C["model"]),
          ("Road graph\nDijkstra distances\n(transitions)", C["proc"]),
          ("Viterbi\nmost likely\narc sequence", C["model"]),
          ("Route polylines\n+ arc per GPS point", C["out"])]
    for (t, fc), x in zip(on, xs):
        box(ax, x, ys, W, 1.55, t, fc, fs=8.5)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + W, ys + 0.77, b, ys + 0.77)
    box(ax, 9.0, 0.15, 4.1, 1.1, "score_trip(): reads .route ONLY after prediction\n→ Top-1, continuity, route P/R/F1 (server-side)",
        C["gt"], fs=8.5)
    arrow(ax, 12.15, ys, 11.6, 1.25)
    arrow(ax, 13.6, 4.9, 4.6 + W / 2, ys + 1.55, color=C["accent"], ls="--")
    arrow(ax, 13.8, 4.9, 9.0 + W / 2, ys + 1.55, color=C["accent"], ls="--")
    ax.text(11.0, 4.15, "final_model.joblib → LightGBM step;  hmm_config.json (α, β) → Viterbi step\n(loaded once at API start-up)",
            ha="center", va="center", fontsize=8, color=C["accent"],
            bbox=dict(fc="white", ec="none", pad=1))
    box(ax, 13.4, ys, 1.45, 1.55, "JSON\nresponse\n→ UI", C["out"], fs=9, bold=True)
    arrow(ax, 11.2 + W, ys + 0.77, 13.4, ys + 0.77)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=130); plt.close(fig)


def two_laptops():
    fig, ax = plt.subplots(figsize=(15, 8.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 8.2); ax.axis("off")
    # laptops
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 4.6, 7.4, boxstyle="round,pad=0.02", fc="#fafafa", ec=C["edge"], lw=1.5))
    ax.add_patch(FancyBboxPatch((8.0, 0.3), 6.8, 7.4, boxstyle="round,pad=0.02", fc="#fafafa", ec=C["edge"], lw=1.5))
    ax.text(2.5, 7.35, "LAPTOP 2 – Evaluator UI", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.5, 7.0, "run_ui.bat → python ui/serve.py (port 8501)\nbrowser: http://localhost:8501", ha="center", va="top", fontsize=8.5)
    ax.text(11.4, 7.35, "LAPTOP 1 – Model + API server", ha="center", fontsize=12, fontweight="bold")
    ax.text(11.4, 7.0, 'run_api.bat "<evaluator_test>" → uvicorn api.main:app --host 0.0.0.0 --port 8000\n'
            "holds: final_model.joblib, hmm_config.json, the 10 trip folders (incl. .route)",
            ha="center", va="top", fontsize=8.5)
    ax.text(6.4, 7.35, "same Wi-Fi / LAN\nHTTP + JSON\nport 8000", ha="center", va="top", fontsize=9, color=C["accent"])

    box(ax, 0.6, 5.0, 3.8, 1.0, "1. Evaluator enters API URL\nhttp://<laptop-1-IP>:8000 → Load trips", C["data"], fs=8.5)
    box(ax, 0.6, 3.4, 3.8, 1.0, "3. Evaluator picks a trip →\nReconstruct route", C["data"], fs=8.5)
    box(ax, 0.6, 0.8, 3.8, 1.5, "6. Draw GPS points + predicted route\n(Leaflet/OSM) and the metrics table\nclassifier vs + HMM", C["out"], fs=8.5)

    box(ax, 8.4, 5.0, 6.0, 1.0, "2. GET /health, GET /trips → 10 trip ids (no labels)\nlog: GET /trips -> 10 trips", C["proc"], fs=8.5)
    box(ax, 8.4, 2.95, 6.0, 1.75,
        "4. POST /predict {trip_id}\n  a) load_trip(with_route=False)\n  b) candidates + 20 features  c) LightGBM\n  d) HMM transitions + Viterbi",
        C["model"], fs=8.5)
    box(ax, 8.4, 0.8, 6.0, 1.75,
        "5. score_trip(): now read .route, align labels\n→ Top-1, continuity, route P/R/F1\nlog: POST /predict trip=… done | top1(HMM)=… | 2.4s\nresponse: route + arc per point + metrics + timing\n(reference route is NEVER returned)",
        C["gt"], fs=8.5)

    arrow(ax, 4.4, 5.5, 8.4, 5.5, "GET /trips", color=C["accent"])
    arrow(ax, 8.4, 5.2, 4.4, 5.2, "[\"00000015\", …]", off=(0, -0.32))
    arrow(ax, 4.4, 3.9, 8.4, 3.9, "POST /predict {\"trip_id\": …}", color=C["accent"])
    arrow(ax, 11.4, 2.95, 11.4, 2.55)
    arrow(ax, 8.4, 1.55, 4.4, 1.55, "200 OK  JSON (≈10–140 KB)", off=(0, 0.05))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "two_laptop_flow.png"), dpi=130); plt.close(fig)


def trellis():
    fig, ax = plt.subplots(figsize=(13, 6.4)); ax.set_xlim(-0.6, 5.6); ax.set_ylim(-1.3, 3.9); ax.axis("off")
    ax.set_aspect("equal")
    best = [1, 1, 0, 0, 2, 2]
    for t in range(6):
        ax.text(t, 3.55, f"GPS point t={t}", ha="center", fontsize=9, fontweight="bold")
        for k in range(3):
            on = best[t] == k
            ax.add_patch(plt.Circle((t, 2.6 - k * 1.1), 0.27, fc=C["accent"] if on else "white", ec=C["edge"], lw=1.3, zorder=3))
            ax.text(t, 2.6 - k * 1.1, f"arc\n{chr(65 + k)}{t}", ha="center", va="center", fontsize=7,
                    color="white" if on else "black", zorder=4)
        if t < 5:
            for k in range(3):
                for j in range(3):
                    ax.plot([t + 0.27, t + 1 - 0.27], [2.6 - k * 1.1, 2.6 - j * 1.1], color="#cfd8dc", lw=0.8, zorder=1)
            ax.plot([t + 0.27, t + 1 - 0.27], [2.6 - best[t] * 1.1, 2.6 - best[t + 1] * 1.1], color=C["accent"], lw=2.5, zorder=2)
    ax.text(3, -0.55, "Each column = candidate arcs (≤10) of one GPS point.  Emission: α·log p̂(arc | point) from LightGBM.\n"
            "Each grey edge = transition: −|d_network − d_GPS| / β (Dijkstra on the road graph).  "
            "Pink = Viterbi path (max total score).", ha="center", va="top", fontsize=9)
    ax.set_title("HMM trellis (schematic, 3 candidates shown per point)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "hmm_trellis.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(ARTIFACTS, "hmm_config.json")))
    pipeline(cfg); two_laptops(); trellis()
    print("written", os.listdir(OUT))
