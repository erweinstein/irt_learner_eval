# Flags and open questions (IRT evaluation of ML pipelines, 11 November 2026; items 12-17 added later that day, 18-21 on 12 November, 22-23 on 13 November, and the first TabPFN check results in 21 later that day; the second and third phases of that check in 21 on 14 November, and (f)-(i) of 21 on 15 November; 24-25 on 16 November)

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
    (d) The ensemble is not the cause either (second phase, 14 November: TabPFN's default of 8 members, on the 8
    of these instances with up to 500 rows and 10-17 features). On every fit 4 members drew a power transform
    and 4 none. On y, 8 members move the median ratio only from 0.74 to 0.78 on these 8, and in both directions
    (c016 1.00 to 0.90, c019 0.76 to 0.67; c034 0.69 to 0.79, c070 0.72 to 0.83); the calibration criterion is
    met on 1 of the 8 either way. On log(1 + y), 8 members give a median ratio of 0.87 (one member 0.83): closer
    to 1 than the stored one-member fit on 6 of the 8, but closer than the one-member log fit on only 4. The
    calibration criterion is met on 3 of the 8 (as with one member on log(1 + y)), the accuracy criterion on 2
    (1 for the stored fit), and accuracy is higher than the stored fit's on 6. Two counts stay well short of the
    training mean in every variant (c036 0.73-0.82, c074 0.57-0.76). So of the four variants, 8 members on log(1
    + y) does best, but most of its gain comes from the log scale, it costs about ten times as long (30-45
    minutes against 4 on these instances), and it does not remove the shortfall on counts.
    (e) On the 12 larger instances (third phase, 14 November: one member, 2,000-3,000 training rows; 3 spending
    and 9 counts), the refits again reproduce the stored ratios exactly, the quantile mean is again 0.96-1.00 of
    the point prediction, and the drawn transform again does not explain the ratio (8 power transforms, median
    0.93; 4 none, 0.86). The shortfall on y is smaller here (median 0.93), but it splits by outcome: on spending
    the median is 0.75, and log(1 + y) brings it to 1.03 (c021 0.57 to 0.99, c063 0.75 to 1.03, c018 1.01 to
    1.07); on counts the median is 0.96, and log(1 + y) lowers it to 0.83 (c071 0.89 to 0.63, c059 0.96 to 0.83,
    c037 0.66 to 0.56; c035 and c075 rise a little, and c041's overshoot of 1.14 goes to 1.00). The calibration
    criterion is met on 7 of the 12 instead of 5, the accuracy criterion on 6 either way, and accuracy rises on
    8. Over all 26 instances, then, fitting one member on log(1 + y) moves the median ratio on spending from 0.76
    to 1.02 (calibration met on 7 of 8 instead of 3) and on counts from 0.84 to 0.83 (on 7 of 18 instead of 6).
    So the log scale fixes the mean on spending, which is strongly right-skewed, and does nothing for
    counts on the whole, where the shortfall varies from instance to instance.
    (f) The 8-member fits on y for the instances with 1,000 rows or 45-70 features (fourth and fifth phases, 14-15
    November) again move the ratio both ways: c017 0.77 to 0.80, c020 1.01 to 0.87, c058 0.82 to 0.80, c062 0.76
    to 0.87, c068 0.90 to 0.97, c082 1.00 to 1.01. On log(1 + y), 8 members on the last four (finished 15
    November) give c058 0.83 (one member on log(1 + y) 0.85), c062 0.99 (1.02), c068 1.02 (1.05) and c082 1.01
    (1.02): the spending instance c062 is fixed either way, and the count c058 stays short either way. Over the
    12 instances with all four variants (3 spending, 9 counts; up to 500 rows), the median ratio is 0.76 for the
    stored one-member fit on y, 0.88 for one member on log(1 + y), 0.82 for 8 members on y and 0.90 for 8 members
    on log(1 + y); the calibration criterion is met on 3, 6, 3 and 6, the accuracy criterion on 3, 3, 3 and 4.
    On the 9 counts alone the medians are 0.77, 0.84, 0.80 and 0.86, and five counts stay well short in the best
    variant (c036 0.73, c074 0.75, c070 0.82, c058 0.83, c040 0.86). So the conclusion of (d) holds on all 12: 8
    members on log(1 + y) does a little better than one member on log(1 + y), takes about nine times as long
    (a median of 45 minutes against 5 per fit on these instances), and does not remove the shortfall on counts.
    (g) What goes with the shortfall (15 November; `R/07b_tabpfn_check_instances.R`, which rebuilds the 26
    instances without fitting; output/tabpfn_check_instances.csv and tabpfn_check_correlates.csv). On spending
    (8 instances) the one-member ratio falls with the weight of the right tail: Spearman correlation -0.86 with
    the largest value over the mean and with the variance over the mean, -0.69 with skewness; and the gain from
    the log scale rises with them (+0.83 with the largest value over the mean). On counts (18) nothing about the
    tail goes with the ratio (|rho| <= 0.14 for skewness, the largest value over the mean and the largest
    standardized value); the ratio is lower where zeros are more common (-0.59) and, on the 10 simulated
    instances, where the true mean explains less of the outcome's variance (+0.62).
    (h) Where the mean is lost (15 November; `R/07c_tabpfn_dist_check.R`, `py/tabpfn_dist_check.py`,
    output/tabpfn_dist_check.csv). On six count instances with 500 training rows and 10-17 features, the one-member
    fit's predictive distribution, averaged over 2,000 test rows and set against the test outcomes, has far too
    little above the training 99th percentile on every one: 0.02-0.42% of the probability against 0.45-1.55% of
    the outcomes, about 4 to 35 times too little. On four of the six it also puts too much on zero (c040 0.56 against
    0.23; c038 0.32 against 0.24; c034, c070 about 0.04 too much). Values above the training 90th percentile
    account for 5-84% of the gap between the test mean and the mean prediction; the rest is the bulk of the
    distribution sitting too low. The training rows, predicted in-sample, fall short by as much (ratio 0.60-0.85
    against 0.55-0.86 on the test rows), so it is not overfitting or a difference between training and test
    data. (This does not contradict (a): the mean TabPFN reports is the mean of its own predictive distribution;
    that distribution is too light in its upper tail and, on some instances, too heavy at zero.) So on counts
    TabPFN's predictive distribution is too concentrated at the low end, most where zeros are common and the
    signal is weak, and neither the log scale nor the ensemble corrects it. The binning is continuous, so the
    shares above an integer threshold t are compared with the outcomes above t; that favors the predictions.
    (i) What it costs TabPFN among the 21 learners (15 November; `R/07d_tabpfn_log_sensitivity.R`,
    output/v21/tabpfn_log_sensitivity.csv). Putting the one-member log(1 + y) fit's responses in place of
    TabPFN's on the 8 spending instances changes 9 responses, raises its proportion of criteria met from 0.62 to
    0.67, and moves it in the multilevel Rasch model (A1, as in R/08) from 1.53 and 4th to 2.19 and 2nd, behind
    only the stack (3.28); no other learner moves more than 0.07. Doing the same on counts too (16 responses)
    gives 2.10, still 2nd. So TabPFN's place below gam and BART comes from its spending means, which a log scale
    would have fixed. This is what-if, not a fair comparison: I chose the log scale for spending after seeing
    these results, and no other learner was given a choice of scale (glm_matched, gam and the boosters fit
    spending with a log link; glm, lasso, ridge and enet fit dollars by squared error).
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
24. **Graded responses and a known-groups check (16 November; after Gibbons et al. 2024).**
    (a) A graded version of the bifactor adaptive test (BCAT_GRADED=1 R/06_bifactor_cat.R; 21 learners;
    output/v21/bifactor_cat_*_graded.*) scores each response 0 = clearly fails, 1 = near the criterion, 2 = meets
    it, with a continuation-ratio logit calibrated by glmmTMB on the stacked steps. A calibration slope that
    cannot be computed (14 responses, cart and gam) counts as 0. Against each scoring's own full bank: at 10
    items, mean error 0.70 logits graded against 0.98 binary (Pearson 0.89 against 0.84); at 20 items 0.53
    against 0.50; at 80 items 0.27 against 0.20. The full bank's mean posterior SD falls from 0.57 to 0.50
    (marginal reliability 0.92 to 0.94), most for the weakest learners (xgb_default 0.56 to 0.30, default
    CatBoost 0.64 to 0.32; Spearman 0.53 between ability and the ratio of posterior SDs). So graded scoring helps
    short tests and the low end, and is no more accurate from 20 items on. Coverage of the error by 1.96
    posterior SDs is 0.92-0.98 for graded bifactor selection, against 0.94-1.00 binary, so the graded posterior
    SDs may be a little small (see 25). Rasch-information selection is worse graded (Pearson 0.52 at 10 items).
    (b) The graded model assumes the same learner effect at both steps (the analogue of proportional odds).
    The likelihood ratio test of Hedeker (2015) against step-specific learner effects rejects it (chi-square
    172 on 20 df; R/06c_graded_step_test.R, output/v21/graded_step_test.csv), though BIC prefers the simpler
    model (11,299 against 11,320) and the two steps' effects correlate 0.93. TabPFN departs most (0.85 at step 1,
    2.36 at step 2, centred), then default CatBoost (-0.93, -2.81), the GAM and the neural network. ICCs on the
    logistic scale (pi^2/3 at level 1; output/v21/graded_icc.csv): task family 0.21, condition 0.19, item 0.12,
    learner by family 0.16, learner by condition 0.15; the instance variance is estimated at zero.
    (c) Flipped models with graded responses (R/03b_graded_bifactor.R; output/v21/irt_B_graded_fit.csv,
    irt_B_graded_bifactor.csv): three exploratory factors fit best by BIC in both scorings, and the bifactor with
    the learner groups fixed in advance is 112 (binary) and 294 (graded) BIC points behind. Unlike Gibbons et al.,
    whose domains were built into the item bank, my learner groups are a post hoc grouping. General-factor
    loadings correlate 0.94 between scorings (range 0.22-0.90 graded, 0.21-0.93 binary), general scores 0.95,
    and the general score's marginal reliability rises from 0.88 to 0.91 (scored by quasi-Monte Carlo). The
    elastic net is a near-Heywood case in both (communality 0.998; intercept 22 binary, 69 graded), and lasso and
    ridge are close (0.95-0.97), so their loadings are poorly determined.
    (d) Known groups (R/06b_known_groups.R, output/v21/known_groups.csv): in the binary leave-one-out runs, every
    run of every selection rule ranks tuned xgboost, LightGBM and CatBoost above their defaults from 10 items on,
    by 1.2-2.5 SD of the 21 full-bank abilities; with bifactor selection the difference exceeds 1.96 posterior SDs
    in every run from 20 items on. The differences are large, so this shows the score is not broken more than
    that it is sensitive.
25. **The calibrations' variance components are about 10% too small, mainly because of the Laplace approximation
    (checked 16 November; R/09_laplace_check.R, R/09b_laplace_consistency.R, R/10_partial_laplace.R;
    output/v21/laplace_*.csv, partial_laplace_*.csv).** Every glmmTMB fit here uses the Laplace approximation
    (adaptive quadrature with one point). Hedeker (2015) recommends more points, which glmmTMB and lme4 do not
    offer for crossed random effects. A learner has a median of only 6 task-items in a task condition, and a
    task-item about 19 responses. A first version of this item, written earlier on 16 November, said the
    approximation was not the problem; (a) shows why that was wrong.
    (a) One component at a time. Refitting each component alone on the real responses, with every other term of
    the glmmTMB fit in an offset, lme4 gives the same SD with Laplace and with 25-point adaptive quadrature, to
    within 0.4%. But holding the other effects at their estimates removes their uncertainty, so this check misses
    the joint error.
    (b) The estimates. In a parametric bootstrap (100 data sets simulated from the fitted calibration model with
    all 21 learners, refitted; 95 binary and 92 graded fits converged), the estimated SDs are on average this
    share of the ones simulated from: binary, learner by family 0.88, learner by condition 0.90, item 0.89,
    condition 0.95, task family 0.63 (16 families); graded, 0.90, 0.95, 0.95, 0.96 and 0.82. The learner SDs are
    too small in 97% of binary replicates (Monte Carlo SE of each ratio 0.006-0.03). The instance SD, simulated
    as zero, comes back at 0.36 (binary) and 0.22 (graded). The learner abilities are not distorted (their spread
    1.00 of the truth, correlation 0.97).
    (c) The joint approximation, by TMB's checkConsistency (200 data sets simulated from the fitted model, the
    gradient of the Laplace marginal likelihood at the true parameters; the importance criterion and the
    instance effect left out so that the Hessian is not singular). Under an exact likelihood that gradient
    averages zero whatever the small-sample bias; here the test rejects (p < 0.001). Its first-order estimate of
    the bias the approximation adds, as a factor on each SD: learner by family 0.90, learner by condition 0.88,
    item 0.91, condition 0.96, task family 0.98. These match the bootstrap's ratios in (b) except for task family,
    whose 0.63 is the small-sample bias of maximum likelihood with 16 families. So the approximation accounts for
    nearly all the shortfall in the item, learner-by-family and learner-by-condition SDs.
    (d) An own solver (R/10_partial_laplace.R). It integrates the learner-by-family and learner-by-condition
    effects exactly, by Gauss-Hermite quadrature nested in quadrature (the likelihood factors over those clusters
    once the task-side effects are given), and applies Laplace only to the task-side effects. Hand-written
    gradient and Hessian, checked against finite differences (relative error 10^-10); quadrature checked against
    adaptive integration on single clusters and 30 against 45 points (0.01 apart); agrees with glmmTMB to 0.008
    when the learner SDs are near zero. On a 5 x 5 grid around glmmTMB's estimates, with every other parameter
    held at its estimate, its log-likelihood is 29-41 higher than Laplace's and rises with the SDs, and its
    maximum lies at learner-by-family SD 2.15 against 2.07 (x 1.04) and learner-by-condition SD 1.75 against 1.64
    (x 1.07), the same on the inner 3 x 3. That is about half the shortfall in (c); the rest presumably comes
    from the Laplace approximation over the item effects, which rest on about 19 binary responses each and which
    this solver still approximates. Two first attempts failed: importance sampling around the Laplace mode (the
    weights collapsed onto one draw in 40, over 881 or 3,095 dimensions), and a TMB template for the nested
    quadrature (automatic differentiation of the nested integrals ran out of memory even with 3 points).
    (e) What it changes. Rerunning the binary bifactor adaptive test with the two learner SDs scaled up by the
    bootstrap's factors (1.14 and 1.12; BCAT_SF_SCALE, BCAT_SC_SCALE; output/v21/bifactor_cat_*_sdadj.*) corrects
    the total shortfall whatever its source. It leaves the full-bank abilities nearly unchanged (correlation
    0.999, largest change 0.23 logits) and raises their posterior SDs by about a tenth (mean 0.57 to 0.63 logits,
    range 0.36-1.06). The adaptive posterior SDs rise by 5-10%, and the share of errors within 1.96 posterior SDs
    at 5 and 10 items rises from 0.94 to 0.98 and 0.95; mean errors move both ways (bifactor selection at 10 items
    0.98 to 0.85 logits, at 20 items 0.50 to 0.58). So the uncorrected posterior SDs are slightly too small, but
    the abilities and the comparisons between rules stand. The multilevel Rasch models of R/03_irt.R use the same
    approximation, so their variance components are presumably too small by a similar amount; their abilities
    should be little affected. The graded calibration's shortfall is smaller (0.90 and 0.95) and was not
    corrected. A fully exact fit (all random effects integrated) would need MCMC or Monte Carlo EM; not done.
