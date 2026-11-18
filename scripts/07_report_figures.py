"""Extra report figures generated from saved results:
- confusion-matrix grid for all models (student-test candidates, threshold 0.5)
- LightGBM (final, tuned) feature importance
- screenshots of the running UI and a Folium map (headless Edge/Chrome)

Requires the API (port 8000) and UI (port 8501) to be running for the UI screenshot.
"""
import os
import subprocess
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import ARTIFACTS, RESULTS, ROOT  # noqa: E402
from pathsnap.modeling import load_partition  # noqa: E402

FIG = os.path.join(RESULTS, "report_figures")
NAMES = {"logreg": "Logistic Regression", "gnb": "Gaussian NB", "knn": "KNN", "dtree": "Decision Tree",
         "rf": "Random Forest", "hgb": "HistGradientBoosting", "lgbm": "LightGBM", "mlp": "MLP"}
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


def confusion_grid():
    test = load_partition("student_test")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (k, name) in zip(axes.ravel(), NAMES.items()):
        p = pd.read_pickle(os.path.join(RESULTS, "models", f"st_pred_{k}.pkl"))
        m = test[["trip", "obs", "arc", "y"]].merge(p, on=["trip", "obs", "arc"])
        cm = confusion_matrix(m["y"], (m["p"] >= 0.5).astype(int))
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > .5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred wrong", "pred true"]); ax.set_yticks([0, 1], ["wrong arc", "true arc"])
        ax.set_title(name)
    fig.suptitle("Confusion matrices – student-test candidates (threshold 0.5, row-normalised colour)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_matrices.png"), dpi=110); plt.close(fig)


def feature_importance():
    b = joblib.load(os.path.join(ARTIFACTS, "final_model.joblib"))
    imp = pd.Series(b["model"].booster_.feature_importance("gain"), index=b["features"]).sort_values()
    imp = imp / imp.sum()
    imp.to_csv(os.path.join(FIG, "lgbm_feature_importance.csv"))
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax, color="C0"); ax.set_xlabel("share of total split gain")
    ax.set_title("Final LightGBM – feature importance (gain)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "lgbm_feature_importance.png"), dpi=120); plt.close(fig)


def shot(url, out, w=1400, h=900, budget=20000):
    exe = next(b for b in BROWSERS if os.path.exists(b))
    subprocess.run([exe, "--headless", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                    f"--virtual-time-budget={budget}", f"--screenshot={out}", url],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def focused_map(tid="00000077", win=300):
    """Zoomed Folium map on the window where classifier-only and HMM disagree most."""
    from pathsnap.config import STEST_DIR
    from pathsnap.data import load_trip
    from pathsnap.pipeline import PathSnapModel
    from pathsnap.viz import trip_map
    trip = load_trip(os.path.join(STEST_DIR, tid))
    r = PathSnapModel().predict_trip(trip)
    diff = np.convolve((r["classifier_arcs"] != r["hmm_arcs"]).astype(float), np.ones(win), "valid")
    i0 = int(np.argmax(diff))
    path = os.path.join(RESULTS, "maps", f"focus_student_test_{tid}.html")
    trip_map(trip, pred_arcs=r["hmm_arcs"], ref_route=trip.route, classifier_arcs=r["classifier_arcs"],
             title=f"Trip {tid}, obs {i0}-{i0 + win}: classifier-only vs HMM", focus=(i0, i0 + win),
             show_classifier=True).save(path)
    print(f"focus window obs {i0}-{i0 + win}, disagreements {int(diff[i0])}")
    return path


def screenshots():
    shot("http://localhost:8501/?api=http://localhost:8000&trip=00000056", os.path.join(FIG, "ui_trip_00000056.png"))
    shot("http://localhost:8501/?api=http://localhost:8000&trip=00000007", os.path.join(FIG, "ui_trip_00000007.png"))
    mp = os.path.join(RESULTS, "maps", "student_test_00000076.html").replace("\\", "/")
    shot("file:///" + mp, os.path.join(FIG, "folium_trip_00000076.png"))
    shot("file:///" + focused_map().replace("\\", "/"), os.path.join(FIG, "folium_focus_trip_00000077.png"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    what = sys.argv[1:] or ["cm", "imp", "shots"]
    if "cm" in what:
        confusion_grid()
    if "imp" in what:
        feature_importance()
    if "shots" in what:
        screenshots()
    print("done", os.listdir(FIG))
