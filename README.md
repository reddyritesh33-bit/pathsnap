# PathSnap – map-matching & route reconstruction (Phase 1)

Noisy 1 Hz GPS trajectories are matched to a directed road network. Phase 1 covers:

```
raw trip ─► per-trip projection ─► reference labels ─► candidate arcs (KD-tree, r=80 m, K≤10)
        ─► 20 features per (GPS point, arc) ─► feature selection ─► 8 classifiers compared (trip-grouped 5-fold CV)
        ─► Bayesian hyperparameter tuning of the selected model (Optuna TPE)
```

Next phase: HMM/Viterbi decoding, FastAPI backend and UI.

## Repository layout

| Folder / file | Content |
|---|---|
| `pathsnap/data.py` | raw-file loading, chronological sort, local equirectangular projection (metres) |
| `pathsnap/labels.py` | per-GPS-point reference arc from `.route` via monotone DP alignment |
| `pathsnap/candidates.py` | spatial index, candidate generation, feature engineering |
| `pathsnap/modeling.py` | model zoo, trip-grouped 5-fold CV |
| `pathsnap/metrics.py` | Macro F1, log loss, Top-1, route continuity |
| `pathsnap/viz.py` | Folium map generation |
| `scripts/01_eda.py` | dataset inventory, quality checks, plots, EDA Folium maps |
| `scripts/02_build_dataset.py` | builds candidate/feature tables → `cache/` |
| `scripts/03_train_models.py` | trains and compares 8 classifiers |
| `scripts/03b_feature_selection.py` | correlation, gain importance, group ablation, top-k selection |
| `scripts/04_bayes_opt.py` | Optuna TPE tuning of LightGBM, tuned vs untuned comparison |
| `scripts/md_to_pdf.py` | builds `PHASE1_REPORT.pdf` from the Markdown |
| `results/eda/` | inventory, summary, anomalies, candidate stats, plots |
| `results/models/` | per-model CV folds, comparison table and plots, confusion matrices, MLP history |
| `results/feature_selection/` | correlation, ablation and top-k results and plots |
| `results/bayes/` | full trial log, best parameters, optimisation history, tuned vs untuned |
| `results/maps/` | interactive Folium maps (EDA) |
| `PHASE1_REPORT.pdf` | Phase 1 progress report |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

The raw data is expected in `../Student folder/Student folder/{train,student_test}`.
Override with the `PATHSNAP_DATA` environment variable.

## Reproduce

```bash
python scripts/02_build_dataset.py     # ~1 min, writes cache/
python scripts/01_eda.py               # results/eda, results/maps
python scripts/03_train_models.py      # results/models, artifacts/models
python scripts/03b_feature_selection.py   # results/feature_selection (~5 min)
python scripts/04_bayes_opt.py         # results/bayes, artifacts/final_model.joblib (~30 min; N_TRIALS env, default 40)
```

Random seed 42 is used everywhere.

## Leakage controls

* Trips are never split across folds (GroupKFold by trip).
* Feature selection and hyperparameters are chosen on training-trip CV only.
* The evaluator partition is not used anywhere.
* Student-test trips are used only for development-stage comparison.
