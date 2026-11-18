# PathSnap – map-matching & route reconstruction

Noisy 1 Hz GPS trajectories are matched to a directed road network:

```
raw trip ─► candidate arcs (KD-tree, r=80 m, K≤10) ─► 20 features per (GPS point, arc)
        ─► classifier P(arc is correct) ─► HMM / Viterbi over candidates ─► route
```

**Reports:** `PHASE1_REPORT.pdf` (review 1) · `PHASE2_REPORT.pdf` (review 2: pipeline, tuning, HMM, two-laptop API) · `FINAL_REPORT.pdf` (final technical report).

**Student-test results (10 trips):** tuned LightGBM Top-1 0.895 → with HMM **0.935**; route continuity 0.953 → **0.979**.

## Repository layout

| Folder / file | Content |
|---|---|
| `pathsnap/data.py` | raw-file loading, chronological sort, local equirectangular projection (metres) |
| `pathsnap/labels.py` | per-GPS-point reference arc from `.route` via monotone DP alignment |
| `pathsnap/candidates.py` | spatial index, candidate generation, feature engineering |
| `pathsnap/modeling.py` | model zoo, trip-grouped 5-fold CV |
| `pathsnap/metrics.py` | Macro F1, log loss, Top-1, candidate miss rate, route continuity, route P/R/F1 |
| `pathsnap/hmm.py` | HMM transition model (network shortest paths) + Viterbi |
| `pathsnap/pipeline.py` | end-to-end inference used by evaluation and API |
| `pathsnap/viz.py` | Folium map generation |
| `scripts/01_eda.py` | dataset inventory, quality checks, plots, EDA Folium maps |
| `scripts/02_build_dataset.py` | builds candidate/feature tables → `cache/` |
| `scripts/03_train_models.py` | trains + compares 8 classifiers |
| `scripts/03b_feature_selection.py` | correlation, gain importance, feature-group ablation, top-k selection |
| `scripts/04_bayes_opt.py` | Optuna TPE Bayesian optimisation of LightGBM, final model, OOF predictions |
| `scripts/05_hmm_eval.py` | decoder tuning (OOF train), classifier vs classifier+HMM on student-test, maps |
| `scripts/06_collect_results.py` | collects all result tables into `results/RESULTS_SUMMARY.md` |
| `scripts/07_report_figures.py` | confusion-matrix grid, feature importance, UI and map screenshots |
| `scripts/md_to_pdf.py` | Markdown report → PDF |
| `scripts/08_phase2_diagrams.py` | pipeline, two-laptop flow and HMM trellis diagrams |
| `run_api.bat`, `run_ui.bat` | one-click start for Laptop 1 / Laptop 2 (Windows) |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/index.html`, `ui/serve.py` | evaluator UI (Laptop 2) |
| `artifacts/` | `final_model.joblib`, `hmm_config.json`, all compared models |
| `results/` | CSVs, logs, plots, Folium HTML maps |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

The raw data is expected in `../Student folder/Student folder/{train,student_test}`.
Override with the `PATHSNAP_DATA` environment variable.

## Reproduce all results

```bash
python scripts/02_build_dataset.py     # ~1 min, writes cache/
python scripts/01_eda.py               # results/eda, results/maps/eda_*.html
python scripts/03_train_models.py      # results/models, artifacts/models
python scripts/03b_feature_selection.py   # results/feature_selection
python scripts/04_bayes_opt.py         # results/bayes, artifacts/final_model.joblib  (N_TRIALS env, default 40)
python scripts/05_hmm_eval.py          # results/hmm, artifacts/hmm_config.json, results/maps/student_test_*.html
python scripts/06_collect_results.py   # results/RESULTS_SUMMARY.md
python scripts/07_report_figures.py    # results/report_figures (needs API + UI running for screenshots)
```

Random seed 42 is used everywhere.

## Deployment (two laptops)

**Laptop 1 – API**

```bash
run_api.bat "C:\path\to\evaluator_test"
```

or manually:

```bash
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_test      # folder with the 10 trip folders
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Allow port 8000 through the firewall. Find the IP with `ipconfig`. Every request is logged in this console, showing trip, accuracy and timing.

**Laptop 2 – UI**

```bash
run_ui.bat
```

(or `python ui/serve.py 8501`)

Open `http://localhost:8501`, enter `http://<laptop-1-ip>:8000`, click **Load trips**, pick a trip, and click **Reconstruct route**.
Deep link: `http://localhost:8501/?api=http://<laptop-1-ip>:8000&trip=<trip_id>` runs a trip directly.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | model + decoder config |
| GET | `/trips` | – | evaluation trip ids |
| POST | `/predict` | `{"trip_id": "00000000"}` | predicted route polylines, arc per GPS point, server-side metrics, timing |

Ground-truth routes are read only inside `score_trip()` after prediction. They are never passed to the model and never returned to the UI.

## Leakage controls

* Trips are never split across folds (GroupKFold by trip). The evaluator partition is not used anywhere in development.
* Feature selection and model hyperparameters were decided on training-trip CV only. Decoder parameters were selected on out-of-fold training predictions.
* Student-test trips are used only for development-stage comparison.
* The evaluator data is never stored in this repository; it is only read by the API at evaluation time.

## AI-use disclosure

An AI coding assistant (Claude) helped develop the code and draft the reports. All results come from executed code in this repository.
