"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.

Objective : mean 5-fold GroupKFold (by trip) Macro F1 on the 80 training trips.
Outputs   : results/bayes/trials.csv, optimisation plots, tuned-vs-untuned table,
            artifacts/final_model.joblib, cache/oof_train.pkl (out-of-fold
            probabilities used to tune the HMM decoder without leakage).
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
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.candidates import FEATURES  # noqa: E402
from pathsnap.config import ARTIFACTS, CACHE_DIR, RESULTS, SEED  # noqa: E402
from pathsnap.metrics import classifier_metrics  # noqa: E402
from pathsnap.modeling import SUBSAMPLE_EVERY, grouped_cv, load_partition, make_model, predict_proba  # noqa: E402

MODEL = "lgbm"
N_TRIALS = int(os.environ.get("N_TRIALS", 40))
OUT = os.path.join(RESULTS, "bayes")


def space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def evaluate(params, train, test):
    cv = grouped_cv(MODEL, train, params)
    m = make_model(MODEL, params)
    X, y = train[FEATURES].to_numpy(np.float32), train["y"].to_numpy()
    t0 = time.time(); m.fit(X, y); fit_s = time.time() - t0
    Xt = test[FEATURES].to_numpy(np.float32)
    t0 = time.time(); p = predict_proba(m, Xt); inf_s = time.time() - t0
    st = classifier_metrics(test, p)
    n_obs = test.groupby(["trip", "obs"]).ngroups
    return m, dict(cv_macro_f1=cv["macro_f1"].mean(), cv_macro_f1_std=cv["macro_f1"].std(),
                   cv_top1=cv["top1_acc"].mean(), cv_log_loss=cv["log_loss"].mean(),
                   st_top1=st["top1_acc"], st_macro_f1=st["macro_f1"], st_log_loss=st["log_loss"],
                   fit_s=fit_s, infer_ms_per_obs=1000 * inf_s / n_obs)


def main():
    os.makedirs(OUT, exist_ok=True)
    train = load_partition("train", subsample=SUBSAMPLE_EVERY)
    test = load_partition("student_test")

    def objective(trial):
        cv = grouped_cv(MODEL, train, space(trial))
        trial.set_user_attr("cv_top1", float(cv["top1_acc"].mean()))
        trial.set_user_attr("cv_log_loss", float(cv["log_loss"].mean()))
        trial.set_user_attr("cv_std", float(cv["macro_f1"].std()))
        return float(cv["macro_f1"].mean())

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_lgbm")
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number} value={t.value:.5f} best={s.best_value:.5f}", flush=True)])
    search_s = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(OUT, "trials.csv"), index=False)
    best = {**space(optuna.trial.FixedTrial(study.best_params))}
    json.dump({"best_params": best, "best_cv_macro_f1": study.best_value, "n_trials": N_TRIALS,
               "search_seconds": search_s, "sampler": "TPE(seed=42, n_startup_trials=10)"},
              open(os.path.join(OUT, "best_params.json"), "w"), indent=2)

    # optimisation history plot
    v = trials["value"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(v, "o", ms=4, alpha=.6, label="trial"); ax[0].plot(np.maximum.accumulate(v), "r-", label="best so far")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV Macro F1"); ax[0].set_title("Optuna TPE optimisation history"); ax[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp)[::-1], list(imp.values())[::-1]); ax[1].set_title("Hyperparameter importance (fANOVA)")
        pd.Series(imp).to_csv(os.path.join(OUT, "param_importance.csv"))
    except Exception as e:  # noqa: BLE001
        ax[1].text(.1, .5, str(e))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "optimisation_history.png"), dpi=120); plt.close(fig)

    # tuned vs untuned
    _, untuned = evaluate({}, train, test)
    tuned_model, tuned = evaluate(best, train, test)
    cmp = pd.DataFrame([dict(model="LightGBM untuned (defaults used in comparison)", **untuned),
                        dict(model="LightGBM tuned (Optuna TPE)", **tuned)])
    cmp.to_csv(os.path.join(OUT, "tuned_vs_untuned.csv"), index=False)
    print(cmp.T)
    joblib.dump({"model": tuned_model, "features": FEATURES, "params": best},
                os.path.join(ARTIFACTS, "final_model.joblib"), compress=3)

    # out-of-fold probabilities on ALL labelled+unlabelled train rows (for HMM tuning)
    full = load_partition("train", labelled_only=False)
    oof = np.zeros(len(full))
    lab = full["gt_arc"] >= 0
    for tr_trips, va_trips in GroupKFold(5).split(full, groups=full["trip"]):
        va_set = set(full["trip"].iloc[va_trips])
        tr_mask = lab & ~full["trip"].isin(va_set) & (full["obs"] % SUBSAMPLE_EVERY == 0)
        m = make_model(MODEL, best).fit(full.loc[tr_mask, FEATURES].to_numpy(np.float32), full.loc[tr_mask, "y"])
        oof[va_trips] = predict_proba(m, full.iloc[va_trips][FEATURES].to_numpy(np.float32))
    full[["trip", "obs", "arc", "rank", "t_proj", "gt_arc", "y"]].assign(p=oof).to_pickle(os.path.join(CACHE_DIR, "oof_train.pkl"))


if __name__ == "__main__":
    main()
