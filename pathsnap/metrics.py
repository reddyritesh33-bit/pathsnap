"""Evaluation metrics for candidate classification and route reconstruction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def normalise_within_obs(df: pd.DataFrame, p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Turn independent candidate probabilities into a distribution per obs."""
    p = np.clip(p, eps, 1.0)
    s = pd.Series(p, index=df.index).groupby([df["trip"], df["obs"]]).transform("sum").to_numpy()
    return p / s


def top1_predictions(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    tmp = df[["trip", "obs", "arc", "gt_arc"]].copy()
    tmp["p"] = p
    idx = tmp.groupby(["trip", "obs"])["p"].idxmax()
    return tmp.loc[idx, ["trip", "obs", "arc", "gt_arc", "p"]].rename(columns={"arc": "pred_arc"})


def classifier_metrics(df: pd.DataFrame, p: np.ndarray, thr: float = 0.5) -> dict:
    """Metrics on observations that have a reference label (gt_arc >= 0)."""
    m = (df["gt_arc"] >= 0).to_numpy()
    d, pp = df[m], p[m]
    y = d["y"].to_numpy()
    yhat = (pp >= thr).astype(int)
    top = top1_predictions(d, pp)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, np.clip(pp, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "top1_acc": float((top["pred_arc"] == top["gt_arc"]).mean()),
    }


def route_continuity(pred_arcs: np.ndarray, arcs_tbl: np.ndarray) -> float:
    """Share of consecutive observation pairs whose predicted arcs are the same
    arc or directly connected (to-node of the first == from-node of the second)."""
    a, b = pred_arcs[:-1], pred_arcs[1:]
    if len(a) == 0:
        return 1.0
    ok = (a == b) | (arcs_tbl[a, 1] == arcs_tbl[b, 0])
    return float(ok.mean())


def route_set_scores(pred_arcs: np.ndarray, ref_route: np.ndarray) -> dict:
    """Sequence-level: precision/recall/F1 of the set of traversed arcs."""
    P, R = set(map(int, pred_arcs)), set(map(int, ref_route))
    tp = len(P & R)
    prec = tp / max(len(P), 1)
    rec = tp / max(len(R), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
