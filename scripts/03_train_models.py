"""Train and compare candidate classifiers.

* 5-fold GroupKFold over the 80 training trips (no trip split across folds)
* refit on all training trips, evaluate on the 10 student-testing trips
* records CV folds, student-test metrics, timings, confusion matrices,
  MLP loss curves -> results/models/
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import ARTIFACTS, RESULTS, SEED  # noqa: E402
from pathsnap.metrics import classifier_metrics, route_continuity, top1_predictions  # noqa: E402
from pathsnap.modeling import SUBSAMPLE_EVERY, grouped_cv, load_partition, make_model, predict_proba  # noqa: E402

MODELS = {
    "logreg": ("Linear / probabilistic", "StandardScaler + LogisticRegression C=1"),
    "gnb": ("Linear / probabilistic", "GaussianNB defaults"),
    "knn": ("Instance-based", "StandardScaler + KNN k=25 distance-weighted, 150k train rows"),
    "dtree": ("Tree-based", "DecisionTree max_depth=12 min_leaf=50"),
    "rf": ("Ensemble (bagging)", "RandomForest 200 trees depth 18 min_leaf=5"),
    "hgb": ("Boosting", "HistGradientBoosting 300 iters lr=0.1"),
    "lgbm": ("Boosting", "LightGBM 400 trees lr=0.05 leaves=63"),
    "mlp": ("Neural network", "StandardScaler + MLP (64,32) adam early-stopping"),
}
OUT = os.path.join(RESULTS, "models")


def continuity_by_trip(df, p, arcs_by_trip):
    top = top1_predictions(df.assign(gt_arc=df["gt_arc"]), p).sort_values(["trip", "obs"])
    vals = [route_continuity(g["pred_arc"].to_numpy(), arcs_by_trip[t]) for t, g in top.groupby("trip")]
    return float(np.mean(vals))


def main(only=None):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS, "models"), exist_ok=True)
    train = load_partition("train", subsample=SUBSAMPLE_EVERY)
    test = load_partition("student_test", labelled_only=False)
    test_l = test[test["gt_arc"] >= 0].reset_index(drop=True)
    from pathsnap.config import STEST_DIR
    from pathsnap.data import list_trips, load_trip
    arcs_by_trip = {os.path.basename(d): load_trip(d, with_route=False).arcs for d in list_trips(STEST_DIR)}
    print("train rows", len(train), "pos rate", train["y"].mean(), "| test rows", len(test_l), flush=True)

    rows = []
    for name, (family, cfg) in MODELS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        cv = grouped_cv(name, train)
        cv.to_csv(os.path.join(OUT, f"cv_{name}.csv"), index=False)
        X = train[FEATURES].to_numpy(np.float32)
        y = train["y"].to_numpy()
        idx = np.arange(len(X))
        if name == "knn":
            idx = np.random.default_rng(SEED).choice(idx, 150_000, replace=False)
        m = make_model(name)
        t1 = time.time(); m.fit(X[idx], y[idx]); fit_s = time.time() - t1
        Xt = test_l[FEATURES].to_numpy(np.float32)
        t1 = time.time(); p = predict_proba(m, Xt); inf_s = time.time() - t1
        st = classifier_metrics(test_l, p)
        cont = continuity_by_trip(test_l, p, arcs_by_trip)
        n_obs = test_l.groupby(["trip", "obs"]).ngroups
        row = dict(model=name, family=family, config=cfg,
                   cv_macro_f1=cv["macro_f1"].mean(), cv_macro_f1_std=cv["macro_f1"].std(),
                   cv_top1=cv["top1_acc"].mean(), cv_log_loss=cv["log_loss"].mean(),
                   cv_train_macro_f1=cv["train_macro_f1"].mean(),
                   st_macro_f1=st["macro_f1"], st_precision=st["precision"], st_recall=st["recall"],
                   st_top1=st["top1_acc"], st_log_loss=st["log_loss"], st_route_continuity=cont,
                   fit_s=fit_s, infer_ms_per_obs=1000 * inf_s / n_obs, total_s=time.time() - t0)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        pd.DataFrame({"trip": test_l["trip"], "obs": test_l["obs"], "arc": test_l["arc"], "p": p}).to_pickle(
            os.path.join(OUT, f"st_pred_{name}.pkl"))
        joblib.dump(m, os.path.join(ARTIFACTS, "models", f"{name}.joblib"), compress=3)

        cm = confusion_matrix(test_l["y"], (p >= 0.5).astype(int))
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm, display_labels=["wrong arc", "true arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{name} - student_test candidates")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"cm_{name}.png"), dpi=120); plt.close(fig)

        if name == "mlp":
            mlp = m[-1]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(mlp.loss_curve_, label="training loss (log loss)")
            ax2 = ax.twinx(); ax2.plot(mlp.validation_scores_, color="C1", label="validation accuracy (10% hold-out)")
            ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax2.set_ylabel("val accuracy")
            fig.legend(loc="upper right"); ax.set_title("MLP training history")
            fig.tight_layout(); fig.savefig(os.path.join(OUT, "mlp_training_curve.png"), dpi=120); plt.close(fig)
            pd.DataFrame({"epoch": range(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_acc": mlp.validation_scores_}).to_csv(os.path.join(OUT, "mlp_history.csv"), index=False)

    res = pd.DataFrame(rows)
    path = os.path.join(OUT, "model_comparison.csv")
    if only and os.path.exists(path):
        old = pd.read_csv(path)
        res = pd.concat([old[~old["model"].isin(res["model"])], res], ignore_index=True)
    res.to_csv(path, index=False)
    plot_comparison(res)


def plot_comparison(res):
    res = res.set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    res["cv_macro_f1"].plot.bar(ax=axes[0], yerr=res["cv_macro_f1_std"], color="C0", title="CV Macro F1 (5-fold, grouped by trip)")
    res[["st_top1", "st_macro_f1"]].plot.bar(ax=axes[1], title="Student-test Top-1 accuracy / Macro F1")
    res["st_log_loss"].plot.bar(ax=axes[2], color="C3", title="Student-test log loss (lower is better)")
    for a in axes:
        a.set_xlabel(""); a.grid(axis="y", alpha=.3)
    axes[0].set_ylim(0.5, 1); axes[1].set_ylim(0.5, 1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_comparison.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
