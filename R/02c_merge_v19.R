# 02c_merge_v19.R -- Merge the 19-learner run (IRT_PLAN = "v19", output/instances_v19) with the original
# instances (output/instances) into output/instances_19. For real-data instances, whose accuracy is judged
# against the best learner, and for plasmode instances (whose three replicates were the same data set in the
# original run, before the load_meps() fix of 12 November 2026), the v19 file replaces the original: it ran every
# learner. For every other instance, the new learners' rows are added to the original 14.
# An instance not yet rerun keeps its original 14 learners, except a MEPS-based one (real c60-c69, plasmode
# c54-c59), which is left out, since its original draw duplicated the other replicates.
# Then run R/02b_combine.R with IRT_INSTANCE_DIR=output/instances_19 IRT_RESPONSES=output/responses_19.csv.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/01_tasks.R"); C <- conditions()
unlink("output/instances_19", recursive = TRUE); dir.create("output/instances_19", showWarnings = FALSE)
f <- list.files("output/instances"); kept <- dropped <- 0
for (x in f) {
  old <- readRDS(file.path("output/instances", x)); nf <- file.path("output/instances_v19", x); i <- old$cond[1]
  replace <- C$source[C$cond == i] %in% c("real", "plasmode"); meps <- i %in% 54:69
  if (!file.exists(nf)) {
    if (meps) dropped <- dropped + 1 else { kept <- kept + 1; saveRDS(old, file.path("output/instances_19", x)) }
    next
  }
  new <- readRDS(nf)
  out <- if (replace) new else rbind(old, new[, names(old)])
  saveRDS(out, file.path("output/instances_19", x))
}
cat(length(f) - kept - dropped, "instances merged;", kept, "not yet rerun (original 14 learners kept);",
    dropped, "MEPS-based not yet rerun (left out)\n")
