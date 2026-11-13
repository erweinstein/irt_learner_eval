# 04b_volatility_summary.R -- Side-by-side check of the location-scale fit on the 10-replicate rerun of the
# simulated tasks (output/responses_more.csv): melsvmp's estimates (04_mels_volatility.R run with
# IRT_RESPONSES=output/responses_more.csv MELS_OUT=output/mels_volatility_10reps), against
#   (a) moments: per learner, the log of the average within-cell variance across the 10 draws, and the log of
#       the variance over conditions of the learner's cell mean (minus the condition's mean over learners),
#       less the part the within-cell variance explains (average within-cell variance / 10);
#   (b) maximum likelihood (glmmTMB, Laplace) for the same model without the random scale effect:
#       score ~ learner + condition + a cell random intercept with a variance per learner, and a residual
#       variance per learner.
# Output: output/volatility_check.csv. From 13 November the inputs and output can be set with IRT_RESPONSES,
# MELS_OUT and VOL_OUT (the 19-learner version: output/responses_more_19.csv, output/mels_volatility_10reps_19,
# output/volatility_check_19.csv)
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(library(glmmTMB))
RESP <- Sys.getenv("IRT_RESPONSES", "output/responses_more.csv")
MELS <- Sys.getenv("MELS_OUT", "output/mels_volatility_10reps")
OUTF <- Sys.getenv("VOL_OUT", "output/volatility_check.csv")
r <- read.csv(RESP); a <- r[r$criterion == "acc" & r$source == "sim" & is.finite(r$score), ]
L <- intersect(c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default", "xgb_tuned", "svm", "nnet", "stack",
        "lightgbm", "catboost", "bart", "lightgbm_tuned", "catboost_tuned"), unique(a$learner))
a$learner <- factor(a$learner, L); a$cell <- factor(paste(a$cond, a$learner)); a$cond <- factor(a$cond)
cm <- aggregate(score ~ cell + learner + cond, a, mean); wv <- aggregate(score ~ cell + learner, a, var)
cm$dev <- cm$score - ave(cm$score, cm$cond)
mom <- data.frame(learner = L, log_within_var_moments = log(tapply(wv$score, wv$learner, mean)[L]),
                  log_between_var_moments = log(tapply(cm$dev, cm$learner, var)[L] - tapply(wv$score, wv$learner, mean)[L] / 10))
f <- glmmTMB(score ~ 0 + learner + cond + diag(0 + learner | cell), dispformula = ~ 0 + learner, data = a,
             control = glmmTMBControl(optCtrl = list(iter.max = 5000, eval.max = 5000)))
ml <- data.frame(learner = L, alpha_ML_no_scale_effect = 2 * log(attr(VarCorr(f)$cond$cell, "stddev")),
                 tau_ML_no_scale_effect = unname(glmmTMB::fixef(f)$disp))
v <- read.csv(paste0(MELS, ".csv")); v <- v[v$fit == "raw scores", ]
vm <- data.frame(learner = L, alpha_VMP = v$estimate[match(paste0("alpha_", L), v$parameter)],
                 tau_VMP = v$estimate[match(paste0("tau_", L), v$parameter)])
out <- Reduce(function(x, y) merge(x, y, by = "learner", sort = FALSE), list(mom, ml, vm))
out$sigma_omega_VMP <- v$estimate[v$parameter == "sigma_omega"]
out$tau_VMP_plus_half_sigma2 <- out$tau_VMP + out$sigma_omega_VMP^2 / 2   # log of the mean within variance under the MELS
write.csv(out, OUTF, row.names = FALSE)
print(out, digits = 3, row.names = FALSE)
