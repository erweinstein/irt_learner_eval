# 02f_merge_v21.R -- Add the tuned LightGBM and CatBoost of 13 November (IRT_PLAN = "v21", output/instances_v21) to
# the 19-learner instances (output/instances_19), writing output/instances_21. Only their rows are added. On the
# real-data instances the v21 run also reran the 17 learners that set the accuracy reference; their scores must
# agree with the stored ones (to 1e-8), and the script stops if any does not.
# Then: IRT_INSTANCE_DIR=output/instances_21 IRT_RESPONSES=output/responses_21.csv Rscript R/02b_combine.R
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

NEW <- c("lightgbm_tuned", "catboost_tuned")
unlink("output/instances_21", recursive = TRUE); dir.create("output/instances_21", showWarnings = FALSE)
f <- list.files("output/instances_19"); missing <- character(0); checked <- 0; worst <- 0
for (x in f) {
  old <- readRDS(file.path("output/instances_19", x)); nf <- file.path("output/instances_v21", x)
  if (!file.exists(nf)) { missing <- c(missing, x); saveRDS(old, file.path("output/instances_21", x)); next }
  new <- readRDS(nf)
  sh <- merge(old, new[!new$learner %in% NEW, ], by = c("learner", "criterion"), suffixes = c(".old", ".new"))
  if (nrow(sh)) {
    d <- max(abs(sh$score.old - sh$score.new), 0, na.rm = TRUE)
    if (d > 1e-8 || any(sh$pass.old != sh$pass.new, na.rm = TRUE)) stop("rerun learners disagree on ", x, ": ", d)
    checked <- checked + 1; worst <- max(worst, d)
  }
  add <- new[new$learner %in% NEW, names(old)]
  saveRDS(rbind(old, add), file.path("output/instances_21", x))
}
cat(length(f) - length(missing), "instances merged;", checked, "real-data instances checked (largest score difference",
    format(worst), ");", length(missing), "without the tuned learners:", paste(missing, collapse = " "), "\n")
