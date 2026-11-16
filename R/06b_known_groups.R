# 06b_known_groups.R -- A known-groups check of the bifactor adaptive test (after Gibbons et al. 2024, who checked
# that their adaptive score separates people with mild cognitive impairment from those without). Here the groups
# are known by construction: the tuned version of a learner should do at least as well as the same learner at its
# package defaults. Three pairs: xgb_tuned vs xgb_default, lightgbm_tuned vs lightgbm, catboost_tuned vs catboost.
# For each pair, rule and test length k, from the leave-one-learner-out runs of R/06_bifactor_cat.R (each learner
# scored with the bank calibrated on the other learners): the difference in ability (tuned minus default) in SD
# units of the full-bank abilities of all learners, the share of pairs of runs in which the tuned version scores
# higher, and whether the difference exceeds 1.96 times its posterior SD. The full-bank difference is the
# reference; the multilevel Rasch abilities of 03_irt.R (model A1) are shown beside it.
# Input: output/v21/bifactor_cat_results.csv and bifactor_cat_results_graded.csv (if present),
# output/v21/irt_A_ability.csv. Output: output/v21/known_groups.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

OUT <- Sys.getenv("IRT_OUT", "output/v21")
PAIRS <- list(xgboost = c("xgb_default", "xgb_tuned"), lightgbm = c("lightgbm", "lightgbm_tuned"),
              catboost = c("catboost", "catboost_tuned"))
A <- read.csv(file.path(OUT, "irt_A_ability.csv"))
files <- c(binary = "bifactor_cat_results.csv", graded = "bifactor_cat_results_graded.csv")
files <- files[file.exists(file.path(OUT, files))]
rows <- list()
for (sc in names(files)) {
  R <- read.csv(file.path(OUT, files[[sc]]))
  full <- unique(R[, c("learner", "theta_full", "psd_full")]); sdf <- sd(full$theta_full)
  for (p in names(PAIRS)) {
    d0 <- PAIRS[[p]][1]; d1 <- PAIRS[[p]][2]
    f0 <- full[full$learner == d0, ]; f1 <- full[full$learner == d1, ]
    for (rule in unique(R$rule)) for (k in sort(unique(R$k))) {
      x0 <- R[R$learner == d0 & R$rule == rule & R$k == k, ]; x1 <- R[R$learner == d1 & R$rule == rule & R$k == k, ]
      diff <- outer(x1$theta, x0$theta, "-"); se <- sqrt(outer(x1$psd^2, x0$psd^2, "+"))
      rows[[length(rows) + 1]] <- data.frame(scoring = sc, pair = p, rule = rule, k = k, runs = length(diff),
        diff_sd = mean(diff) / sdf, share_tuned_higher = mean(diff > 0), share_sep_1.96 = mean(diff > 1.96 * se),
        full_diff_sd = (f1$theta_full - f0$theta_full) / sdf,
        full_z = (f1$theta_full - f0$theta_full) / sqrt(f1$psd_full^2 + f0$psd_full^2),
        rasch_A1_diff = A$theta[A$learner == d1] - A$theta[A$learner == d0],
        rasch_A1_z = (A$theta[A$learner == d1] - A$theta[A$learner == d0]) / sqrt(A$se[A$learner == d1]^2 + A$se[A$learner == d0]^2))
    }
  }
}
KG <- do.call(rbind, rows)
write.csv(KG, file.path(OUT, "known_groups.csv"), row.names = FALSE)
print(KG[KG$k %in% c(10, 20, 40, 80), ], digits = 2, row.names = FALSE)
