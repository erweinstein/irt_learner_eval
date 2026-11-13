# irt_learner_eval: evaluating machine-learning learners with item response theory

Eliot R. Weinstein · eliotw@uchicago.edu · erweinstein.com · November 2026

A first try at evaluating a group of machine-learning learners with item response models, both ways
round: learners as examinees (a multilevel Rasch model with the task bank's nesting as levels), and,
flipped, learners as items with the tasks as respondents (2PL, exploratory multidimensional, bifactor,
graded, multiple-group and two-level models), followed by bifactor adaptive testing of a new learner.
Following Bock and Gibbons (2021, *Item Response Theory*, sections 8, 11.5 and 13.4) and Gibbons and
Hedeker (1992). [`WALKTHROUGH.md`](WALKTHROUGH.md) goes through it step by step (also as
[Word](docs/WALKTHROUGH.docx) and [PDF](docs/WALKTHROUGH.pdf)); [`FLAGS.md`](FLAGS.md) lists possible
problems and open questions.

The MEPS tasks need `meps_ml.csv` from the MEPS walkthrough
([using-meps-data-for-simulations](https://github.com/erweinstein/using-meps-data-for-simulations),
`ml_00_extract`), placed at `../meps_for_simulation/data/meps_ml.csv` or named in `MEPS_ML_CSV`. It is
not included here: AHRQ's data use agreement does not allow MEPS public use files, or row-level data made
from them, to be redistributed. Nothing in `output/` is row-level MEPS data (scores and model output only).

## Running it

From this folder, in R:

```r
source("R/02_run.R")       # 301 task instances x 14 learners; ~50 min on 2 cores; resumable (output/instances/)
source("R/02b_combine.R")  # output/responses.csv
source("R/03_irt.R")       # all models, tables (output/irt_*.csv) and figures (output/fig_*.png)
source("R/05_cat.R")       # Rasch adaptive testing, leave one learner out
source("R/06_bifactor_cat.R")  # bifactor adaptive testing, leave one learner out (~10 min on 2 cores)
```

Added on 11 November:

- **More replicates.** `IRT_INSTANCE_DIR=output/instances_more IRT_REPS=10 IRT_FAMILIES=<the simulated
  families> Rscript R/02_run.R`, then `02b_combine.R` with `IRT_RESPONSES=output/responses_more.csv`
  (10 draws per simulated condition, 630 instances; the first 3 are the original ones).
- **Location-scale model.** `R/04_mels_volatility.R` (melsvmp; Wu and Hedeker 2026) and
  `R/04b_volatility_summary.R` (checks against moments and glmmTMB). The fit is degenerate here (FLAGS.md 16).
- **Five more learners.** `run_v19.sh FAMILIES REP_ONLY LOG` runs one chunk of `IRT_PLAN=v19` (LightGBM,
  CatBoost and BART on every instance; RealMLP and TabPFN on the first replicate of the prediction tasks;
  every learner rerun on real-data instances); then `R/02c_merge_v19.R`, `02b_combine.R` with
  `IRT_INSTANCE_DIR=output/instances_19 IRT_RESPONSES=output/responses_19.csv`, and `03_irt.R` with
  `IRT_RESPONSES=output/responses_19.csv IRT_OUT=output/v19`. Python: catboost 1.2.10, pytabkit 1.8.0,
  tabpfn 2.0.9 (its weights download from Hugging Face on first use), torch (CPU); R: lightgbm, dbarts,
  reticulate (1.47 or later, for NumPy 2).

Added on 12 November:

- **A seed bug, fixed.** `load_meps()` overwrote each instance's seed, so in the original run the three draws
  of each of the 16 MEPS-based conditions were one data set (FLAGS.md 19). `R/02d_check_meps_splits.R` checks
  every stored MEPS-based instance against its own split (`output/meps_split_check.csv`: the original run's
  MEPS-based instances fail, as they should; every 19-learner one passes); `R/02e_add_original_plasmode.R`
  adds the original learners to plasmode instances rerun with the new five only.
- **Queues.** `run_v19_queue.sh TAG SPEC...` runs chunks in order and stops after 112 minutes (rerun it to
  continue); `run_v19_fix.sh TAG` runs two queues side by side, the large TabPFN fits one at a time.
- **Final 19-learner results** are in `output/v19/` (from `output/responses_19.csv`, with `03_irt.R`,
  `06_bifactor_cat.R` and `05_cat.R` run with `IRT_RESPONSES=output/responses_19.csv IRT_OUT=output/v19`).
  The files directly in `output/` are the original 14-learner results, which count the MEPS-based task-items
  three times.

Added on 13 November:

- **Tuned LightGBM and CatBoost** (`lightgbm_tuned`, `catboost_tuned`; FLAGS.md 20 and 22), run with
  `PLAN=v21 ./run_v19_queue.sh TAG "||1|0|Inf"` and merged by `R/02f_merge_v21.R`; results with 21 learners
  in `output/responses_21.csv` and `output/v21/`.
- **Threshold sensitivity** (`R/08_threshold_sensitivity.R`; FLAGS.md 1): results in
  `output/v19/threshold_sensitivity_*.csv`.
- **Volatility with the new fast learners** (`R/02g_merge_more.R`; FLAGS.md 23): LightGBM, CatBoost and BART
  added to the 10-replicate run (`output/responses_more_19.csv`, `output/volatility_check_19.csv`,
  `output/mels_volatility_10reps_19.csv`). melsvmp must be on R's library path (`R_LIBS`).
- **A check of TabPFN's low means on spending and counts** (`R/07_tabpfn_check.R`, `py/tabpfn_check.py`,
  `run_tabpfn_check.sh`; FLAGS.md 21): results in `output/tabpfn_check.csv`.

`R/00_learners.R` defines the learners (the original 14, and five added on 11 November: LightGBM,
CatBoost, BART, RealMLP and TabPFN v2; the last three Python ones run through reticulate from
`py/learners_py.py`) and `R/01_tasks.R` the 95 task conditions. The MEPS tasks
read `../meps_for_simulation/data/meps_ml.csv` (from the MEPS walkthrough's `ml_00_extract`), which is
not shared under the MEPS data use agreement; nothing in `output/` is row-level MEPS data. Packages:
glmnet, mgcv, FNN, rpart, ranger, xgboost (3.x API), e1071, nnet, MASS, quadprog, AER, ISLR2, mlbench,
glmmTMB, mirt (needs Deriv; on R 4.3 install Deriv 4.1.6 from the CRAN archive), ggplot2, ggrepel.

## Files

| File | What it holds |
|---|---|
| `output/responses.csv` | one row per instance x criterion x learner: score and pass (6,223 rows) |
| `output/responses_19.csv`, `output/v19/` | the same for all 19 learners (7,904 rows), and the final results from them |
| `output/responses_21.csv`, `output/v21/` | the same with the two tuned learners added (8,842 rows) |
| `output/irt_prop_correct*.csv` | proportion of criteria met, overall and by task family |
| `output/irt_A_ability.csv`, `irt_A_profile.csv` | multilevel Rasch abilities; learner-by-family deviations |
| `output/irt_B_items_2pl.csv` | flipped 2PL slopes and difficulties (and `_without_ridge_enet`) |
| `output/irt_B_loadings2.csv`, `irt_B_loadings3.csv`, `irt_B_bifactor.csv` | factor loadings |
| `output/irt_B_multilevel.csv` | two-level 2PL slopes between and within task conditions |
| `output/irt_B_dif.csv` | differential functioning, real vs simulated tasks |
| `output/irt_B_task_scores.csv`, `irt_B_theta_by_family.csv` | task-item easiness scores |
| `output/irt_results.rds` | all fitted models |
| `output/cat_*.csv`, `output/bifactor_cat_*.csv` | adaptive testing (Rasch and bifactor) |
| `output/responses_more.csv`, `output/volatility_check.csv` | 10-replicate rerun; location-scale checks |
| `FLAGS.md` | possible problems and open questions |
