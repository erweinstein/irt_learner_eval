# Flags and open questions (IRT evaluation of ML pipelines, 11 November 2026; items 12-17 added later that day, 18-21 on 12 November)

1. **Thresholds are arbitrary.** Signal R^2 >= 0.8, AUC excess >= 90% of the oracle's, within 0.01 of the
   best learner on real data (0.02 for spending), calibration slope in [0.8, 1.25] and means within
   10-15%. The graded version agrees, but a threshold sensitivity analysis has not been run.
2. **Real-data accuracy is relative to the best learner on the split**, which makes responses within a
   task-item dependent; the real-vs-simulated DIF may partly reflect this difference in criteria.
3. **Near-duplicate learners** (lasso, ridge, enet agree on 90-95% of task-items) inflate their slopes
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
8. **The importance criterion fixed effect is huge** (about 21-22 logits): only 6 importance items, nearly all
   passed. It does not affect the learner estimates much but is not well determined.
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
    variances collapsed (alpha -17 to -65) and the scale random effect took everything (sigma_omega 2.5).
    melsvmp's sandwich stopped on a singular per-subject Hessian; the script patches it to skip such subjects
    (212 of 701) and does not use the sandwich for inference (the condition bootstrap is used). Rerun on 10
    replicates per condition.
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
    set: glm, which is deterministic, has identical scores across them, and the other learners differ only through
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
    tuning (and ensembling), so these results are about the defaults, not the methods; a tuned version, as for
    xgboost, would be the fairer comparison and has not been run.
21. **TabPFN underpredicts the mean on spending and counts.** On the 26 count and spending calibration items its
    predicted mean is a median 0.76 (spending) and 0.84 (counts) of the training mean, against 0.96-1.08 for
    BART, glm, lasso and RealMLP (xgb_tuned 0.85 on spending; item 10). So it meets the calibration-in-the-large
    criterion on only 38% and 33% of them, against 54% of the yes/no items, though its accuracy pass rate is second
    only to the stack's (73% against 76%). A possible cause, not checked: TabPFN's regressor predicts the mean of a binned
    distribution over a bounded range of the target, which may cut off long right tails, and here it uses one
    ensemble member (item 18). A log transform of the outcome, or the default ensemble, might change this.
