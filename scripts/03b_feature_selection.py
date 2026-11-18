"""Feature selection experiments (training trips only, trip-grouped 5-fold CV).

1. Feature correlation matrix (redundancy check).
2. Gain importance of LightGBM fitted on the training trips.
3. Group ablation: drop one feature group at a time and measure the CV change.
4. Importance-based selection: keep the top-k features (k = 5, 10, 15, 20)
   and measure CV Macro F1 / Top-1 for each k.

All decisions use CV on the 80 training trips; student-test is not used.
Outputs: results/feature_selection/
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pathsnap.modeling as modeling  # noqa: E402
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import RESULTS  # noqa: E402
from pathsnap.modeling import SUBSAMPLE_EVERY, grouped_cv, load_partition, make_model  # noqa: E402

OUT = os.path.join(RESULTS, "feature_selection")
GROUPS = {
    "distance": ["dist", "rank", "dist_minus_min", "dist_ratio", "n_cand"],
    "direction": ["head_diff", "head_cos", "head_diff_long", "speed"],
    "temporal_context": ["dist_prev", "dist_next", "dist_win_mean", "dist_win_rank"],
    "geometry_topology": ["arc_len", "t_proj", "t_clipped", "has_reverse", "out_degree", "in_degree"],
    "sampling": ["dt_prev"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES)


def cv_with(features, train):
    """Run the standard LightGBM grouped CV using only `features`."""
    old = modeling.FEATURES
    modeling.FEATURES = features            # grouped_cv reads the module-level list
    try:
        t0 = time.time()
        cv = grouped_cv("lgbm", train)
    finally:
        modeling.FEATURES = old
    return dict(n_features=len(features), cv_macro_f1=cv["macro_f1"].mean(), cv_macro_f1_std=cv["macro_f1"].std(),
                cv_top1=cv["top1_acc"].mean(), cv_log_loss=cv["log_loss"].mean(), seconds=time.time() - t0)


def main():
    os.makedirs(OUT, exist_ok=True)
    train = load_partition("train", subsample=SUBSAMPLE_EVERY)

    # 1. correlation
    corr = train[FEATURES].corr(method="spearman")
    corr.to_csv(os.path.join(OUT, "feature_correlation_spearman.csv"))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90); ax.set_yticks(range(len(FEATURES)), FEATURES)
    fig.colorbar(im, shrink=.8); ax.set_title("Spearman correlation between features (train)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "feature_correlation.png"), dpi=120); plt.close(fig)
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
    pairs[pairs.abs() > 0.8].sort_values(key=abs, ascending=False).to_csv(os.path.join(OUT, "highly_correlated_pairs.csv"))

    # 2. importance (fit on training trips only)
    m = make_model("lgbm").fit(train[FEATURES].to_numpy(np.float32), train["y"])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT, "gain_importance_train.csv"), header=["gain_share"])

    # 3. group ablation
    rows = [dict(experiment="all features", **cv_with(FEATURES, train))]
    print(rows[-1], flush=True)
    for g, cols in GROUPS.items():
        feats = [f for f in FEATURES if f not in cols]
        rows.append(dict(experiment=f"without {g}", **cv_with(feats, train)))
        print(rows[-1], flush=True)
    abl = pd.DataFrame(rows)
    abl["delta_macro_f1"] = abl["cv_macro_f1"] - abl.loc[0, "cv_macro_f1"]
    abl["delta_top1"] = abl["cv_top1"] - abl.loc[0, "cv_top1"]
    abl.to_csv(os.path.join(OUT, "group_ablation.csv"), index=False)

    # 4. top-k by importance
    topk = []
    for k in (5, 10, 15, 20):
        topk.append(dict(k=k, features=" ".join(imp.index[:k]), **cv_with(list(imp.index[:k]), train)))
        print(topk[-1], flush=True)
    topk = pd.DataFrame(topk)
    topk.to_csv(os.path.join(OUT, "topk_selection.csv"), index=False)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    imp[::-1].plot.barh(ax=ax[0], color="C0"); ax[0].set_title("LightGBM gain importance (train)")
    a = abl.iloc[1:].set_index("experiment")["delta_macro_f1"]
    a.plot.barh(ax=ax[1], color=["C3" if v < 0 else "C2" for v in a]); ax[1].axvline(0, color="k", lw=.8)
    ax[1].set_title("CV Macro F1 change when a feature group is removed")
    ax[2].errorbar(topk.k, topk.cv_macro_f1, yerr=topk.cv_macro_f1_std, marker="o", capsize=4)
    ax[2].set_xlabel("number of top features kept"); ax[2].set_ylabel("CV Macro F1")
    ax[2].set_title("Importance-based top-k selection"); ax[2].set_xticks(topk.k)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "feature_selection.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
