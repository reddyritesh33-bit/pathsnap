# PathSnap – recorded results (auto-generated)

## Dataset summary (results/eda/summary.csv)

| Unnamed: 0          |    student_test |          train |
|:--------------------|----------------:|---------------:|
| trips               |    10           |     80         |
| gps_total           | 19160           | 208417         |
| gps_mean            |  1916           |   2605.21      |
| gps_min             |   262           |    361         |
| gps_max             |  3958           |  19313         |
| nan_values          |     0           |      0         |
| pct_dt_1s           |     0.98199     |      0.938334  |
| trips_gap_gt60s     |     1           |     22         |
| arcs_mean           | 18504.1         |  48461.3       |
| arcs_max            | 56027           | 307392         |
| isolated_nodes      |     0           |      0         |
| weak_comp_mean      |    22.7         |     39.0125    |
| route_connected     |     0.999732    |      0.999346  |
| gps_to_route_median |     4.74464     |      4.62926   |
| unlabelled_frac     |     0.000530571 |      0.0212057 |


## Candidate generation

| partition    |     n_obs |   n_valid |   cand_per_obs |   cand_miss_rate |
|:-------------|----------:|----------:|---------------:|-----------------:|
| student_test | 1916.0000 | 1913.9000 |         8.2435 |           0.0016 |
| train        | 2605.2125 | 2337.6625 |         8.3531 |           0.0120 |


## Model comparison (results/models/model_comparison.csv)

| model   | family                 | config                                                       |   cv_macro_f1 |   cv_macro_f1_std |   cv_top1 |   cv_train_macro_f1 |   st_top1 |   st_macro_f1 |   st_precision |   st_recall |   st_log_loss |   st_route_continuity |    fit_s |   infer_ms_per_obs |
|:--------|:-----------------------|:-------------------------------------------------------------|--------------:|------------------:|----------:|--------------------:|----------:|--------------:|---------------:|------------:|--------------:|----------------------:|---------:|-------------------:|
| logreg  | Linear / probabilistic | StandardScaler + LogisticRegression C=1                      |        0.9318 |            0.0105 |    0.8850 |              0.9304 |    0.8848 |        0.9286 |         0.8675 |      0.8835 |        0.0888 |                0.9499 |   1.1358 |             0.0012 |
| gnb     | Linear / probabilistic | GaussianNB defaults                                          |        0.8684 |            0.0202 |    0.8508 |              0.8543 |    0.8453 |        0.8637 |         0.6530 |      0.9352 |        0.5770 |                0.9299 |   0.1370 |             0.0014 |
| knn     | Instance-based         | StandardScaler + KNN k=25 distance-weighted, 150k train rows |        0.9289 |            0.0105 |    0.8784 |              1.0000 |    0.8831 |        0.9267 |         0.8655 |      0.8786 |        0.1177 |                0.9410 |   0.0672 |             1.1329 |
| dtree   | Tree-based             | DecisionTree max_depth=12 min_leaf=50                        |        0.9321 |            0.0108 |    0.8834 |              0.9413 |    0.8849 |        0.9312 |         0.8949 |      0.8646 |        0.0846 |                0.9453 |  10.7657 |             0.0005 |
| rf      | Ensemble (bagging)     | RandomForest 200 trees depth 18 min_leaf=5                   |        0.9376 |            0.0104 |    0.8889 |              0.9637 |    0.8909 |        0.9355 |         0.9053 |      0.8693 |        0.0693 |                0.9514 | 142.9961 |             0.0196 |
| hgb     | Boosting               | HistGradientBoosting 300 iters lr=0.1                        |        0.9353 |            0.0116 |    0.8873 |              0.9466 |    0.8927 |        0.9346 |         0.8949 |      0.8762 |        0.0706 |                0.9493 |   2.8327 |             0.0160 |
| lgbm    | Boosting               | LightGBM 400 trees lr=0.05 leaves=63                         |        0.9367 |            0.0128 |    0.8889 |              0.9666 |    0.8991 |        0.9378 |         0.9070 |      0.8757 |        0.0677 |                0.9541 |   4.2339 |             0.0268 |
| mlp     | Neural network         | StandardScaler + MLP (64,32) adam early-stopping             |        0.9297 |            0.0126 |    0.8818 |              0.9477 |    0.8947 |        0.9318 |         0.9011 |      0.8607 |        0.0732 |                0.9529 |  76.4057 |             0.0031 |


## Bayesian optimisation

```json
{
  "best_params": {
    "n_estimators": 600,
    "learning_rate": 0.014006991042428668,
    "num_leaves": 221,
    "min_child_samples": 12,
    "subsample": 0.6230590887084679,
    "subsample_freq": 1,
    "colsample_bytree": 0.48583383369193717,
    "reg_lambda": 0.17083764619132746
  },
  "best_cv_macro_f1": 0.9385512334113342,
  "n_trials": 30,
  "search_seconds": 1546.486221075058,
  "sampler": "TPE(seed=42, n_startup_trials=10)"
}
```

| model                                          |   cv_macro_f1 |   cv_macro_f1_std |   cv_top1 |   cv_log_loss |   st_top1 |   st_macro_f1 |   st_log_loss |   fit_s |   infer_ms_per_obs |
|:-----------------------------------------------|--------------:|------------------:|----------:|--------------:|----------:|--------------:|--------------:|--------:|-------------------:|
| LightGBM untuned (defaults used in comparison) |        0.9367 |            0.0128 |    0.8889 |        0.0774 |    0.8991 |        0.9378 |        0.0677 |  8.7004 |             0.0336 |
| LightGBM tuned (Optuna TPE)                    |        0.9386 |            0.0116 |    0.8908 |        0.0732 |    0.8945 |        0.9369 |        0.0677 | 12.5003 |             0.0874 |


## HMM decoder

```json
{
  "alpha": 0.5,
  "beta": 10.0,
  "gap_s": 60.0,
  "gap_m": 500.0,
  "tuned_on": "out-of-fold classifier probabilities, 80 training trips",
  "oof_top1_hmm": 0.9261612249945255,
  "oof_top1_classifier_only": 0.8971823150615899
}
```

### Decoder grid (OOF train)

|   alpha |    beta |   oof_top1 |   oof_continuity |
|--------:|--------:|-----------:|-----------------:|
|  0.5000 |  2.0000 |     0.9133 |           0.9682 |
|  0.5000 |  5.0000 |     0.9245 |           0.9715 |
|  0.5000 | 10.0000 |     0.9262 |           0.9724 |
|  0.5000 | 20.0000 |     0.9229 |           0.9728 |
|  0.5000 | 50.0000 |     0.9203 |           0.9719 |
|  1.0000 |  2.0000 |     0.9210 |           0.9708 |
|  1.0000 |  5.0000 |     0.9257 |           0.9725 |
|  1.0000 | 10.0000 |     0.9222 |           0.9729 |
|  1.0000 | 20.0000 |     0.9203 |           0.9722 |
|  1.0000 | 50.0000 |     0.9185 |           0.9702 |
|  2.0000 |  2.0000 |     0.9233 |           0.9720 |
|  2.0000 |  5.0000 |     0.9205 |           0.9728 |
|  2.0000 | 10.0000 |     0.9187 |           0.9721 |
|  2.0000 | 20.0000 |     0.9173 |           0.9706 |
|  2.0000 | 50.0000 |     0.9147 |           0.9684 |


### Student-test: classifier only vs classifier + HMM

| method           |   top1_acc |   continuity |   route_precision |   route_recall |   route_f1 |   seconds |
|:-----------------|-----------:|-------------:|------------------:|---------------:|-----------:|----------:|
| classifier + HMM |     0.9435 |       0.9794 |            0.9625 |         0.9056 |     0.9316 |    1.0208 |
| classifier only  |     0.9094 |       0.9535 |            0.9144 |         0.8957 |     0.9030 |    1.0208 |


### Per trip

|   trip | method           |   n_obs |   seconds |   top1_acc |   continuity |   route_precision |   route_recall |   route_f1 |
|-------:|:-----------------|--------:|----------:|-----------:|-------------:|------------------:|---------------:|-----------:|
|      0 | classifier only  |    2503 |    1.0875 |     0.9541 |       0.9820 |            0.8769 |         0.9553 |     0.9144 |
|      0 | classifier + HMM |    2503 |    1.0875 |     0.9632 |       0.9960 |            0.9454 |         0.9665 |     0.9558 |
|      7 | classifier only  |    1354 |    0.4517 |     0.9365 |       0.9985 |            0.9837 |         0.9837 |     0.9837 |
|      7 | classifier + HMM |    1354 |    0.4517 |     0.9941 |       1.0000 |            0.9837 |         0.9837 |     0.9837 |
|      9 | classifier only  |    1049 |    0.7259 |     0.8894 |       0.9122 |            0.9033 |         0.8275 |     0.8637 |
|      9 | classifier + HMM |    1049 |    0.7259 |     0.9180 |       0.9485 |            0.9396 |         0.8345 |     0.8840 |
|     40 | classifier only  |    2995 |    1.2098 |     0.9499 |       0.9803 |            0.9367 |         0.9579 |     0.9472 |
|     40 | classifier + HMM |    2995 |    1.2098 |     0.9639 |       0.9957 |            0.9834 |         0.9612 |     0.9722 |
|     52 | classifier only  |     262 |    0.1716 |     0.9809 |       0.9540 |            0.9512 |         0.8211 |     0.8814 |
|     52 | classifier + HMM |     262 |    0.1716 |     1.0000 |       0.9617 |            1.0000 |         0.8632 |     0.9266 |
|     56 | classifier only  |    3958 |    1.9384 |     0.9225 |       0.9227 |            0.9439 |         0.8820 |     0.9119 |
|     56 | classifier + HMM |    3958 |    1.9384 |     0.9672 |       0.9674 |            0.9775 |         0.8942 |     0.9340 |
|     65 | classifier only  |     768 |    0.5884 |     0.8711 |       0.9674 |            0.9220 |         0.8966 |     0.9091 |
|     65 | classifier + HMM |     768 |    0.5884 |     0.8919 |       0.9870 |            0.9710 |         0.9241 |     0.9470 |
|     76 | classifier only  |    1486 |    1.6977 |     0.9266 |       0.9279 |            0.8800 |         0.8824 |     0.8812 |
|     76 | classifier + HMM |    1486 |    1.6977 |     0.9711 |       0.9737 |            0.9456 |         0.8824 |     0.9129 |
|     77 | classifier only  |    3911 |    1.8045 |     0.7561 |       0.9639 |            0.8222 |         0.9664 |     0.8885 |
|     77 | classifier + HMM |    3911 |    1.8045 |     0.8369 |       0.9923 |            0.9059 |         0.9701 |     0.9369 |
|     91 | classifier only  |     874 |    0.5323 |     0.9073 |       0.9255 |            0.9239 |         0.7845 |     0.8485 |
|     91 | classifier + HMM |     874 |    0.5323 |     0.9291 |       0.9714 |            0.9730 |         0.7759 |     0.8633 |

