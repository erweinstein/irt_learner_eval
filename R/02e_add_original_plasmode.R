# 02e_add_original_plasmode.R -- Add the original 14 learners to the plasmode instances of the 19-learner run
# (output/instances_v19) that were drawn from their own split but hold only the five new learners (the run began
# before the load_meps() fix of 12 November 2026; see 02d_check_meps_splits.R). Plasmode accuracy is judged
# against the true signal, not against the other learners, so the original learners' rows can be appended.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(source("R/02_run.R"))
for (i in 54:59) for (rep in 1:3) {
  f <- sprintf("output/instances_v19/c%03d_r%d.rds", i, rep); if (!file.exists(f)) next
  old <- readRDS(f); if ("glm" %in% old$learner) next
  cond <- CL[[i]]; seed <- 100000 + 1000 * i + rep; set.seed(seed); t0 <- Sys.time()
  inst <- plasmode_instance(cond)
  new <- score_pred(inst, run_learners(inst, learners = LEARNERS_ORIGINAL, seed = seed), cond)
  new$inst <- old$inst[1]; new$cond <- i; new$rep <- rep; new$secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  saveRDS(rbind(new[, names(old)], old), f); message("added original learners to c", i, " r", rep, " ", format(Sys.time()))
}
