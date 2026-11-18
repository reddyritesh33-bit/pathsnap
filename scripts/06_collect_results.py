"""Collect all recorded results into results/RESULTS_SUMMARY.md (tables only,
generated from the CSV/JSON outputs of scripts 01-05; no numbers typed by hand)."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import ARTIFACTS, RESULTS  # noqa: E402


def md(df, floatfmt=".4f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    R = lambda *p: os.path.join(RESULTS, *p)  # noqa: E731
    parts = ["# PathSnap – recorded results (auto-generated)\n"]
    parts += ["## Dataset summary (results/eda/summary.csv)\n", pd.read_csv(R("eda", "summary.csv")).to_markdown(index=False), "\n"]
    cs = pd.read_csv(R("eda", "candidate_stats.csv"))
    parts += ["## Candidate generation\n", md(cs.groupby("partition")[["n_obs", "n_valid", "cand_per_obs", "cand_miss_rate"]].mean().reset_index()), "\n"]
    mc = pd.read_csv(R("models", "model_comparison.csv"))
    cols = ["model", "family", "config", "cv_macro_f1", "cv_macro_f1_std", "cv_top1", "cv_train_macro_f1", "st_top1",
            "st_macro_f1", "st_precision", "st_recall", "st_log_loss", "st_route_continuity", "fit_s", "infer_ms_per_obs"]
    parts += ["## Model comparison (results/models/model_comparison.csv)\n", md(mc[cols]), "\n"]
    if os.path.exists(R("bayes", "best_params.json")):
        bp = json.load(open(R("bayes", "best_params.json")))
        parts += ["## Bayesian optimisation\n", "```json\n" + json.dumps(bp, indent=2) + "\n```\n",
                  md(pd.read_csv(R("bayes", "tuned_vs_untuned.csv"))), "\n"]
    if os.path.exists(R("hmm", "student_test_summary.csv")):
        parts += ["## HMM decoder\n", "```json\n" + open(os.path.join(ARTIFACTS, "hmm_config.json")).read() + "\n```\n",
                  "### Decoder grid (OOF train)\n", md(pd.read_csv(R("hmm", "decoder_grid_oof_train.csv"))), "\n",
                  "### Student-test: classifier only vs classifier + HMM\n", md(pd.read_csv(R("hmm", "student_test_summary.csv"))), "\n",
                  "### Per trip\n", md(pd.read_csv(R("hmm", "student_test_hmm_vs_classifier.csv"))), "\n"]
    open(R("RESULTS_SUMMARY.md"), "w", encoding="utf-8").write("\n".join(parts))
    print("written", R("RESULTS_SUMMARY.md"))


if __name__ == "__main__":
    main()
