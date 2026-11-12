# 02b_combine.R -- Combine the saved instances into output/responses.csv, with each condition's description.
# IRT_INSTANCE_DIR and IRT_RESPONSES change the input folder and the output file (for the 10-replicate rerun of
# the simulated tasks: output/instances_more and output/responses_more.csv).
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/01_tasks.R")
suppressMessages(library(glmnet))
C <- conditions()
f <- list.files(Sys.getenv("IRT_INSTANCE_DIR", "output/instances"), full.names = TRUE)
R <- do.call(rbind, lapply(f, function(x) { r <- readRDS(x); r[, c("inst", "cond", "rep", "criterion", "learner", "score", "pass", "secs")] }))
R <- merge(R, C, by = "cond")
R <- R[order(R$cond, R$rep, R$criterion, match(R$learner, c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn",
                                                             "cart", "rf", "xgb_default", "xgb_tuned", "svm", "nnet", "stack"))), ]
# a calibration slope cannot be computed when a learner predicts the same probability for everyone (a tree
# pruned to its root); that learner fails the calibration criterion rather than leaving it unanswered
fix <- is.na(R$pass) & R$criterion == "cal"; R$pass[fix] <- 0
cat(sum(fix), "calibration responses with constant predictions scored as failures (",
    paste(unique(R$learner[fix]), collapse = ", "), ")\n")
write.csv(R, Sys.getenv("IRT_RESPONSES", "output/responses.csv"), row.names = FALSE)
cat(length(f), "instances,", length(unique(paste(R$inst, R$criterion))), "task-items,", nrow(R), "responses;",
    sum(is.na(R$pass)), "missing (failed fits)\n")
print(table(R$family[!duplicated(paste(R$inst, R$criterion))], R$criterion[!duplicated(paste(R$inst, R$criterion))]))
