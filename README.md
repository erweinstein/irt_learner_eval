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
  `run_tabpfn_check.sh`; FLAGS.md 21): results in `output/tabpfn_check.csv`. The one-member fits on the
  instances with up to 1,000 training rows are done: fitting on log(1 + y) helps but does not remove the
  shortfall. Nor does TabPFN's default 8-member ensemble on the instances with up to 500 rows and 10-17
  features (14 November). Over all 26 instances, the log scale fixes the mean on spending (median ratio 0.76
  to 1.02) but not on counts (0.84 to 0.83). Added on 15 November: what goes with the shortfall
  (`R/07b_tabpfn_check_instances.R`), where in TabPFN's predictive distribution the mean on counts is lost
  (`R/07c_tabpfn_dist_check.R`, `py/tabpfn_dist_check.py`: too little in the upper tail, and sometimes too much
  at zero), and TabPFN's place among the 21 learners had it been fitted on log(1 + y)
  (`R/07d_tabpfn_log_sensitivity.R`: 2nd instead of 4th with log(1 + y) on spending). The check finished on
  15 November: the 8-member ensemble on log(1 + y) does little better than one member on log(1 + y).
  Added on 17 November: a recalibrated TabPFN (`R/07e_tabpfn_recalibrated.R`, which multiplies the test
  predictions by the training mean over the mean of the in-sample predictions; `output/tabpfn_recalibrated.csv`)
  and its place among the 21 learners (`R/07f_tabpfn_recal_sensitivity.R`: 2nd, at 3.03 against the stack's
  3.26, with the recalibrated responses on spending and counts). Run with
  `SCRIPT=R/07e_tabpfn_recalibrated.R LOGNAME_PREFIX=tabpfn_recal ./run_tabpfn_check.sh A` (resumable).

Added on 16 November (after Gibbons et al. 2024 and Hedeker 2015; FLAGS.md 24-25):

- **Graded bifactor adaptive test.** `BCAT_GRADED=1` with `R/06_bifactor_cat.R` scores responses 0/1/2 with a
  continuation-ratio logit (`output/v21/bifactor_cat_*_graded.*`); `R/06c_graded_step_test.R` tests its
  equal-effects assumption and reports ICCs.
- **Known-groups check.** `R/06b_known_groups.R`: does the adaptive score rank the tuned boosters above their
  defaults (`output/v21/known_groups.csv`)?
- **Flipped models with graded responses.** `R/03b_graded_bifactor.R` (`output/v21/irt_B_graded_*.csv`).
- **Laplace check.** `R/09_laplace_check.R`: a parametric bootstrap of the calibration model and Laplace against
  adaptive quadrature one component at a time; `R/09b_laplace_consistency.R`: TMB's joint consistency check;
  `R/10_partial_laplace.R`: an own solver that integrates the learner-side effects exactly by nested quadrature
  (`output/v21/laplace_*.csv`, `partial_laplace_*.csv`). `BCAT_SF_SCALE` and `BCAT_SC_SCALE` rerun the adaptive
  test with the bias-corrected SDs (`output/v21/bifactor_cat_*_sdadj.*`).

`R/00_learners.R` defines the learners (the original 14, and five added on 11 November: LightGBM,
CatBoost, BART, RealMLP and TabPFN v2; the last three Python ones run through reticulate from
`py/learners_py.py`) and `R/01_tasks.R` the 95 task conditions. The MEPS tasks
read `../meps_for_simulation/data/meps_ml.csv` (from the MEPS walkthrough's `ml_00_extract`), which is
not shared under the MEPS data use agreement; nothing in `output/` is row-level MEPS data. Packages:
glmnet, mgcv, FNN, rpart, ranger, xgboost (3.x API), e1071, nnet, MASS, quadprog, AER, ISLR2, mlbench,
glmmTMB, mirt (needs Deriv; on R 4.3 install Deriv 4.1.6 from the CRAN archive), ggplot2, ggrepel.

Added on 18 November (FLAGS.md 27-29):

- **Unanimous task-items** (`R/03d_unanimous_sensitivity.R`; `IRT_NCYCLES` sets mirt's EM cycle limit): refits the
  flipped 2PL and bifactor models and the multilevel Rasch model without the task-items every learner passed or
  failed. Results in `output/v21/unanimous_*.csv`. The flipped bifactor fit of `03_irt.R` does not converge with
  them.
- **A posterior-SD stopping rule** (`R/06d_cat_stopping_rule.R`; `CAT_STOP_PSD`, `CAT_CAP`, `CAT_RUNS`, `CAT_RAND`;
  about 10 minutes on 2 cores): adaptive tests with one task-item per family first, stopping at a posterior SD of
  0.9 or 0.7 logits, random selection under the same rule, and coverage of a predictive interval for the
  proportion of criteria met. Results in `output/v21/cat_stop_*.csv`.
- **The new learner calibrated as a new item** (`R/06e_online_calibration_cat.R`; `CAT_STOP_SDA`, `CAT_CAP`,
  `CAT_RUNS`, `CAT_RAND`, `CAT_NSIM`, `CAT_ONLY`; about 1 h 45 min on 2 cores, and it resumes from
  `output/v21/online_cat_parts/` if stopped): online calibration in the flipped orientation, with the learner's own
  intercept and slope estimated against the task-items' calibrated easiness, selection for the posterior variance
  of the bank proportion, a mixed rule that takes every second item at random, and the same coverage check.
  Results in `output/v21/online_cat_*.csv`.
- **The same with a cross-fitted prior and a stopping rule on the learner's own bank**
  (`R/06f_online_cat_crossfit.R`, added 24 November; `CAT_PRIOR` (`crossfit` or `insample`), `CAT_STOP_PSD`,
  `NSIM_STOP`, `CAT_STOP_EVERY` and 06e's settings; about 2 h 20 min on 2 cores, resuming from
  `output/v21/online_cat_cf_parts/`): each reference learner's slope is measured against an easiness fitted
  without it (210 pair calibrations, cached in `output/v21/crossfit_eta/`), and tests also stop on the predictive
  SD of the proportion met over the learner's own bank. Results in `output/v21/online_cat_cf_*.csv`; `FLAGS.md`
  item 30.

## Files

| File | What it holds |
|---|---|
| `rust/README.md` | the exact location-scale fit (`rust/mels_quad`): the model, what it showed against melsvmp, and what Rust contributed |
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
| `output/v21/unanimous_*.csv` | the unanimous task-items, and the fits with and without them |
| `output/v21/cat_stop_*.csv` | adaptive tests with a posterior-SD stopping rule; coverage of the proportion met |
| `output/v21/online_cat_*.csv` | adaptive tests that estimate the new learner's own slope (online calibration) |
| `FLAGS.md` | possible problems and open questions |
