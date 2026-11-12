# 02d_check_meps_splits.R -- Which stored MEPS-based instances (real c60-c69, plasmode c54-c59) were drawn from the
# split their own seed gives? Before 12 November 2026 load_meps() reset the seed, so the first MEPS instance in
# each R process got the split that follows seed 1. Each instance is regenerated with the fixed code and cheap,
# deterministic learners refitted; their scores are compared with the stored ones (for real data, where accuracy
# is relative to the best learner, the difference between glm and lasso; otherwise the scores themselves).
# Output: output/meps_split_check.csv (ok = TRUE: the stored instance used its own split)
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(source("R/02_run.R"))
out <- NULL
for (d in c("output/instances", "output/instances_v19")) for (i in 54:69) for (rep in 1:3) {
  f <- sprintf("%s/c%03d_r%d.rds", d, i, rep); if (!file.exists(f)) next
  old <- readRDS(f); cond <- CL[[i]]; seed <- 100000 + 1000 * i + rep; set.seed(seed)
  inst <- if (cond$source == "plasmode") plasmode_instance(cond) else real_instance(cond)
  use <- if ("glm" %in% old$learner) c("glm", "lasso") else "lightgbm"
  new <- score_pred(inst, run_learners(inst, learners = use, seed = seed), cond)
  m <- merge(old, new, by = c("learner", "criterion"))
  if (cond$source == "real") {      # accuracy relative to the best learner: compare glm minus lasso
    a <- m[m$criterion == "acc", ]; dd <- diff(a$score.x) - diff(a$score.y)
    dd <- c(dd, (m$score.x - m$score.y)[m$criterion != "acc"])
  } else dd <- m$score.x - m$score.y
  out <- rbind(out, data.frame(dir = basename(d), cond = i, rep = rep, learners = paste(use, collapse = "+"),
                               max_diff = max(abs(dd), na.rm = TRUE)))
}
out$ok <- out$max_diff < 1e-8
write.csv(out, "output/meps_split_check.csv", row.names = FALSE); print(out)
