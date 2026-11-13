# Flags and open questions (IRT evaluation of ML pipelines, 11 November 2026; items 12-17 added later that day, 18-21 on 12 November, 22-23 on 13 November, and the first TabPFN check results in 21 later that day)

1. **Thresholds are arbitrary.** Signal R^2 >= 0.8, AUC excess >= 90% of the oracle's, within 0.01 of the
   best learner on real data (0.02 for spending), calibration slope in [0.8, 1.25] and means within
   10-15%. The graded version agrees. A threshold sensitivity analysis (R/08_threshold_sensitivity.R, 13
   November, 19 learners; output/v19/threshold_sensitivity_*.csv) recomputes pass/fail from the stored scores
   under stricter and more lenient thresholds, one criterion at a time and all together, and refits the
   multilevel Rasch model with the learner-by-family term. The stack is first in every scenario, and the bottom
   seven (nnet, CatBoost and LightGBM at their defaults, svm, cart, knn, xgb_default) stay in the last seven places, but the order
   within the upper group is fragile: Spearman correlation with the baseline abilities 0.86-1.00, up to 6 places'
   change. Accuracy thresholds matter most: with the lenient ones (signal R^2 0.7, AUC excess 80%, twice the
   real-data tolerance) tuned xgboost moves from 8th to 2nd and TabPFN from 4th to 10th; with the strict ones
   (0.9, 95%, half the tolerance) the elastic net and lasso move to 2nd and 3rd. One limit: for calibration on
   yes/no outcomes the ratio of predicted to training mean is not stored, so that slope band could only be
   narrowed. Every importance response had all three signal predictors in its top three (item 8), so the
   lenient importance threshold changed nothing.
2. **Real-data accuracy is relative to the best learner on the split**, which makes responses within a
   task-item dependent; the real-vs-simulated DIF may partly reflect this difference in criteria.
3. **Near-duplicate learners** (lasso, ridge, enet agree on 90-95% of task-items with
   the original 14 learners, 89-94% with 19) inflate their slopes
   (12-38 in the one-factor 2PL with the original 14 learners; 11-23 with all 19). The refit without ridge and enet is in irt_B_items_2pl_without_ridge_enet.csv.
4. **Absolute fit is poor** (M2 RMSEA 0.17-0.21 for every flipped model with the original 14 learners, 0.16-0.20
   with all 19).
5. **lme4 failed** ("Downdated VtV is not positive definite") for any model with learner fixed effects and
   the loss-matched GLM present; glmmTMB was used instead (converged, positive-definite Hessians). Not
   investigated further: it may be an lme4/Matrix version issue on this machine.
6. **mirt::mixedmirt** with a condition-level random effect on theta stopped ("non-finite value supplied by
   'nlm'"); the two-level 2PL is fitted with glmmTMB's reduced-rank terms instead. The loadings at each level
   are taken from the first eigenvector of the estimated covariance (rank 1), signs set so they sum positive.
7. **Instance-level variance is zero.** With one criterion per instance for most instances, the instance and
   task-item levels are hard to separate; the estimate put all within-condition variation at the item level.
8. **The importance criterion fixed effect is huge** (about 21-22 logits): only 6 importance items, and every
   importance response passed (all 78 in the original run, all 96 with 19 learners). It does not affect the learner estimates much but is not well determined.
9. **Constant predictions.** 13 calibration responses (all from the pruned tree on yes/no tasks, pruned to one
   node) were scored as failures because a calibration slope cannot be computed.
10. **The tuned xgboost did poorly on spending in dollars here** (Tweedie loss), unlike in the MEPS walkthrough;
    worth checking (the variance power was fixed at 1.5, no tuning of it).
11. **Professor Gibbons's own method** (described in conversation) is not reproduced; this is my version from the
    textbook. Do not attribute these choices to him.
12. **Strong local dependence.** With learner-by-family and learner-by-condition random effects added to the
    multilevel Rasch model (all 14 learners), their SDs are 2.4 and 2.0 logits, larger than the spread of the
    learners' abilities. Answers within a task condition are far from independent given ability. Part of this was
    the MEPS duplication of item 19; on the final 19-learner responses the SDs are 2.2 and 1.7, against an
    ability SD of 1.4, so the conclusion stands.
13. **Rasch adaptive testing was no better than random selection** (05_cat.R), and its posterior SDs were too
    small: after 80-160 items the error relative to the full bank was within 1.96 posterior SDs for 79-83% of
    learners (final 19-learner run: 79% after 80 items and 94% after 160, the latter for the 16 learners with at least 160
    answers, against 97% and 99% at random). The cause is item 12: maximum-information selection draws items from one condition.
14. **Bifactor adaptive testing** (06_bifactor_cat.R): primary ability plus nested learner-by-family and
    learner-by-condition dimensions, loadings fixed at 1 (13 calibration learners cannot estimate loadings; a
    Rasch testlet version of the bifactor model, not Gibbons and Hedeker's full model with free loadings).
    Item parameters and the two SDs come from 13 learners and are treated as known; their uncertainty is
    ignored. The reference is the full-bank EAP with the same calibration, so "error" is distance from what
    the whole bank would say, not from the truth, and the posterior-SD check against it is lenient (both share
    the same answers). The candidate set for the preposterior step is screened (40 most Rasch-informative plus
    20 at random) for speed.
15. **The full bank says less than the Rasch model suggests.** Under the bifactor model the full-bank posterior
    SD of a learner's ability is 0.43-1.09 logits (14 learners; 0.39-1.01 in the final 19-learner run), against
    0.12-0.22 under the Rasch model without the interaction terms. The SEs in the Doc's ability table (0.67-1.02
    with 14 learners, 0.64-0.97 with 19) come from the model with the learner-by-family term and are of the
    right size.
16. **MELS volatility model was degenerate with 3 replicates per cell** (04_mels_volatility.R): between-cell
    variances collapsed for 12 of the 14 learners (alpha -17 to -65; not for xgb_default or nnet) and the scale random effect took everything (sigma_omega 2.5).
    melsvmp's sandwich stopped on a singular per-subject Hessian; the script patches it to skip such subjects
    (212 of 701) and does not use the sandwich for inference (the condition bootstrap is used). Rerun on 10
    replicates per condition. **Correction (13 November):** the condition bootstrap in output/mels_volatility.csv
    rests on 3 or 4 successful refits for every parameter (column boot_ok); the others stopped, apparently on
    a singular system ("reciprocal condition number 2.7e-24" in a reproduction of one replicate), and the
    error handler returned a vector of the wrong length (fixed). So its bootstrap SEs and intervals should not
    be used. No reported result depends on them: the Doc and the walkthrough give only the point estimates and
    the descriptive check. The 10-replicate fits (output/mels_volatility_10reps.csv, and with 19 learners
    output/mels_volatility_10reps_19.csv, MELS_B=0) have no bootstrap.
17. **Two (near-)constant cells dropped** from the MELS fit (pruned tree on condition 24; glm on condition 49).
18. **Five learners added** (LightGBM, CatBoost, BART, RealMLP, TabPFN v2), chosen from TabArena (Erickson et al.
    2025: TabM, LightGBM and RealMLP top its leaderboard with ensembling, CatBoost with tuning alone), TabPFN
    (Hollmann et al. 2025, best on datasets up to 10,000 rows) and causal inference (BART). Not added: TabM
    (similar to RealMLP), TabICL (classification only), EBM. Compromises: on one CPU thread RealMLP takes about
    45 s and TabPFN 45-95 s per fit at 1,000 training rows, so (a) TabPFN uses one ensemble member instead of
    its default 4 (classification) or 8 (regression), which handicaps it; (b) both run only on the first
    replicate of each prediction task, not on the inference (dml, procova) or importance tasks, so their
    responses are missing by design elsewhere (about a third of the prediction task-items); (c) real-data
    accuracy is now judged against the best of the 17 learners run on every instance, so the original 14's
    real-data responses can change where a new learner did best. The original learners' predictions are
    reproduced exactly when the data split is the same (each learner's seed is now tied to its place in
    the full list; on the 38 real-data instances outside the MEPS problem of item 19 that have a calibration
    criterion, their calibration scores are identical). At 2,000 training rows and 71 predictors (the larger
    real-data tasks) TabPFN took 20 minutes and 3.1 GB for one fit, and the session's memory limit is 5.8 GB:
    9 real-data runs and 2 simulated ones (3,000 training rows) were killed for lack of memory, without an error
    message, and rerun. TabPFN now predicts the
    test rows 1,000 at a time (the same predictions to about 1e-6, since each test row attends only to the
    training rows), and the first replicates run one at a time beside a queue of the fast learners. The last 14 large first
    replicates ran TabPFN on two CPU threads (`IRT_TABPFN_THREADS=2`); the predictions agree with one thread to
    floating-point rounding.
19. **The MEPS-based replicates were duplicates in the original run (bug, fixed 12 November).** `load_meps()` set
    the seed to 1 (for the lasso that defines the plasmode truth) the first time MEPS was loaded in an R process,
    which overwrote the instance's own seed. The original run forked a fresh process for each instance, so all
    three replicates of each of the 16 MEPS-based conditions (real c60-c69, plasmode c54-c59) were the same data
    set: glm, which is deterministic, has identical scores across them (except its real-data accuracy scores,
    which are relative to the best learner on the split), and the other learners differ only through
    their own seeds. So 48 of the 301 instances were 16 data sets, and the 14-learner results in the Doc count
    those task-items three times. This probably inflates the learner-by-condition SD (item 12) and the agreement
    of answers within a condition, and with them the adaptive-testing results (items 13-15); the 10-replicate
    MELS run did not use the MEPS conditions. The fix saves and restores the random-number state around the
    seed; R/02d_check_meps_splits.R regenerates every stored MEPS-based instance and checks it against its own
    split (output/meps_split_check.csv). In the 19-learner run, 12 instances drawn before the fix were set aside
    and rerun, and every learner is now run on the plasmode instances as on the real ones. The 14-learner results
    in the Doc have not been recomputed with this fix by themselves; the 19-learner refit replaces them. A check of
    the size of the effect, with the multilevel Rasch model of item 12 (14 learners): the learner-by-condition SD
    is 2.00 logits on the original responses, 1.51 with the MEPS-based conditions left out, and 1.74 on the final
    responses with the fix; the learner-by-family SD 2.36, 2.21 and 2.10. With all 19 learners the two SDs are
    1.72 and 2.18. So the duplication inflated both, but both stay larger than the SD of the learners' abilities
    in the Rasch model without the interaction terms (1.51 for the original 14 on the original responses; 1.50
    for the 14 and 1.42 for all 19 on the final ones), the spread item 12 compares them with. (With the
    interaction terms in the model the learners' fixed effects spread more: SD 3.15 on the original responses,
    2.89 and 2.74 on the final ones.)
20. **LightGBM and CatBoost run at their package defaults, untuned** (LightGBM 100 rounds at learning rate 0.1 with
    31 leaves; CatBoost 1,000 rounds), like xgb_default. In the final 19-learner fit they rank with xgb_default,
    knn, svm and cart, not with xgb_tuned: they pass 12% and 29% of the accuracy criteria and 28% and 10% of the
    calibration criteria (xgb_tuned 55% and 77%). On yes/no outcomes the failures look like overfitting at
    500-2,000 training rows: the median calibration slope is 0.36 for LightGBM and 0.71 for CatBoost, against 1.02
    for xgb_tuned and 1.04 for BART. (The interim figures reported earlier on 12 November, 0.52, 0.70, 0.98 and
    1.02, were wrong: they took the median over the yes/no slopes together with the ratios of predicted to
    training mean on counts and spending, which are stored in the same column.) TabArena's standings for both use
    tuning (and ensembling), so these results are about the defaults, not the methods. Tuned versions were run on
    13 November (item 22) and rank with xgb_tuned. Part of the default CatBoost's failures has a specific cause:
    CatBoost 1.2 does not start boosting from the training mean for log loss (boost_from_average is off by
    default) or for Poisson and Tweedie (where it is not allowed), so its predictions start from a logit of 0 or a
    mean of exp(0) = 1, and 1,000 rounds at its automatic learning rate do not get back: its predicted mean is a
    median 0.57 of the training mean on spending and 0.80 on counts, and it passes 7% of the yes/no calibration
    criteria.
21. **TabPFN underpredicts the mean on spending and counts.** On the 26 count and spending calibration items its
    predicted mean is a median 0.76 (spending) and 0.84 (counts) of the training mean, against 0.96-1.08 for
    BART, glm, lasso and RealMLP (xgb_tuned 0.85 on spending; item 10). So it meets the calibration-in-the-large
    criterion on only 38% and 33% of them, against 54% of the yes/no items, though its accuracy pass rate is second
    only to the stack's (73% against 76%). A possible cause, not checked: TabPFN's regressor predicts the mean of a binned
    distribution over a bounded range of the target, which may cut off long right tails, and here it uses one
    ensemble member (item 18). A log transform of the outcome, or the default ensemble, might change this.
    **Checked in part (13 November; `R/07_tabpfn_check.R`, output/tabpfn_check.csv).** The first of five phases
    refits TabPFN with one ensemble member on the 14 of these instances with up to 1,000 training rows (first
    replicates), and the refits reproduce the stored mean ratios and accuracy scores exactly. (a) The bounded
    range is not the cause: the mean of 999 quantiles of the predicted distribution is 0.96-1.00 of the point
    prediction. (b) Nor is the target transform that TabPFN's preprocessing draws for its one member: 7 of the 14
    fits drew a power transform and 7 none, with median ratios 0.76 and 0.77. (c) Fitted on log(1 + y), with the
    prediction taken as the mean of expm1 over the predicted distribution's quantiles, the median ratio moves
    from 0.77 to 0.91 (spending 0.77 to 1.02, counts 0.77 to 0.84). It is closer to 1 on 9 of the 14 instances,
    and the calibration criterion is met on 7 instead of 4. But the transform overshoots on one (c020, 1.01 to
    1.13) and lowers two (c016 from 1.00 to 0.91, c034 from 0.69 to 0.62). Accuracy rises on 9 of the 14 (c040
    from 0.36 to 0.75, c017 from 0.67 to 0.95), and the accuracy criterion is met on 5 instead of 4. So the log
    transform helps but is not a fix; on counts most of the shortfall remains, and I have not found its cause.
    The default 8-member ensemble (on the 12 instances with up to 500 rows, on y and on log(1 + y)) and the
    one-member fits on the 12 larger instances are still running.
22. **Tuned LightGBM and CatBoost (13 November; `IRT_PLAN=v21`, output/responses_21.csv, output/v21/).** Tuned as
    xgb_tuned is, not by a search: learning rate 0.05, small trees (LightGBM 8 leaves with 80% row and column
    subsampling; CatBoost depth 4), the number of rounds (at most 1,000) by 5-fold cross-validated early
    stopping, and, for CatBoost, boosting from the training mean (a constant baseline of log(mean y) for Poisson
    and Tweedie). They meet 61% (LightGBM) and 65% (CatBoost) of the criteria, against 24% and 28% at the
    defaults and 65% for xgb_tuned; in the 21-learner multilevel Rasch fit their abilities are 0.97 and 1.36
    (xgb_tuned 1.21; the defaults -2.93 and -1.82). The real-data accuracy reference stays the best of the 17
    learners of the v19 run, so no other learner's responses change; the tuned learners beat that reference on
    4 (LightGBM) and 12 (CatBoost) of the 78 real-data instances, by at most 0.026. On the real-data instances
    the 17 were rerun to rebuild the reference, and every one of their scores agrees with the stored one
    (R/02f_merge_v21.R). With 21 learners the earlier conclusions hold (the stack first, Spearman 0.94 between
    abilities and proportions met; three factors best by BIC, with the tuned boosters loading on the flexible
    factor at 0.85-0.89), except that bifactor adaptive selection is no better than random at 10 items (mean
    error 0.98 logits against 0.88); from 20 items on it is better (0.20 against 0.33 at 80).
23. **Draw-to-draw volatility with the new fast learners (13 November).** LightGBM and CatBoost (defaults and
    tuned) and BART were added to the 10-replicate run of the simulated tasks (R/02g_merge_more.R;
    output/responses_more_19.csv; RealMLP and TabPFN are too slow to run 10 times). By the moments of
    R/04b_volatility_summary.R (output/volatility_check_19.csv), the tuned LightGBM and CatBoost vary from draw to
    draw about as little as xgb_tuned (log average within-condition variance -6.0, -6.0 and -6.1, against about
    -6.5 for the three penalized regressions). BART, like the GAM, varies a lot on average (-2.2) but not in a
    typical condition (log median -7.0, in the middle of the 19): its average comes from spending in dollars,
    where its signal R^2 swung widely between draws (within-condition variance 4.3 on condition 20). The
    location-scale fit is as degenerate as with 14 learners: the between-condition variance collapsed (alpha
    -14 to -71) for 11 of the 19 learners, and the fit to scores floored at -1 stopped on a singular matrix and is
    left out. For the original 14 the within-condition moments are unchanged (the same responses); the
    between-condition moments move by -0.12 to +0.20, because each cell mean is centred on the condition's mean
    over all learners, which now includes five more.
