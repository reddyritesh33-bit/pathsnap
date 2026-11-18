"""Shared training utilities: data loading, model zoo, grouped CV."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .candidates import FEATURES
from .config import CACHE_DIR, SEED
from .metrics import classifier_metrics

SUBSAMPLE_EVERY = 3  # 1 Hz fixes are highly autocorrelated: keep every 3rd obs


def load_partition(name: str, subsample: int | None = None, labelled_only: bool = True) -> pd.DataFrame:
    df = pd.read_pickle(os.path.join(CACHE_DIR, f"{name}.pkl"))
    if labelled_only:
        df = df[df["gt_arc"] >= 0]
    if subsample:
        df = df[df["obs"] % subsample == 0]
    return df.reset_index(drop=True)


def make_model(name: str, params: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    params = params or {}
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=params.get("C", 1.0)))
    if name == "gnb":
        return GaussianNB()
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=params.get("k", 25), weights="distance", n_jobs=-1))
    if name == "dtree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth", 12), min_samples_leaf=50, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
    if name == "lgbm":
        import lightgbm as lgb
        base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                    subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0,
                    random_state=SEED, n_jobs=-1, verbose=-1)
        base.update(params)
        return lgb.LGBMClassifier(**base)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                             early_stopping=True, max_iter=60, random_state=SEED))
    raise ValueError(name)


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def grouped_cv(name: str, df: pd.DataFrame, params: dict | None = None, n_splits: int = 5,
               knn_cap: int | None = 150_000):
    X, y, g = df[FEATURES].to_numpy(np.float32), df["y"].to_numpy(), df["trip"].to_numpy()
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, g)):
        if name == "knn" and knn_cap and len(tr) > knn_cap:
            tr = np.random.default_rng(SEED + k).choice(tr, knn_cap, replace=False)
        m = make_model(name, params)
        t0 = time.time(); m.fit(X[tr], y[tr]); fit_s = time.time() - t0
        t0 = time.time(); p = predict_proba(m, X[va]); pred_s = time.time() - t0
        r = classifier_metrics(df.iloc[va], p)
        r.update(fold=k, fit_s=fit_s, pred_s=pred_s,
                 train_macro_f1=classifier_metrics(df.iloc[tr[:200_000]], predict_proba(m, X[tr[:200_000]]))["macro_f1"])
        folds.append(r)
    return pd.DataFrame(folds)
