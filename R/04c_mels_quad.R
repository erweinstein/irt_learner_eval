# 04c_mels_quad.R -- The location-scale model of R/04_mels_volatility.R, fitted by exact maximum likelihood with an
# own solver (rust/mels_quad), with and without a fixed log-variance effect for each task condition (FLAGS.md items
# 16, 23 and 26). melsvmp's variational fit collapsed the between-condition variance (alpha_l) for 11 of the 19
# learners. Two questions: does an exact-likelihood fit of the same model collapse too (then it is the model and the
# data, not the algorithm), and does letting each condition have its own variance level (gamma_c) fix it?
#   Level 1 (replicate r of condition c, learner l): y_rcl = beta_l + beta_c + nu_cl + e_rcl,
#                                                     e_rcl ~ N(0, exp(tau_l + gamma_c + omega_cl))
#   Level 2 (cell = learner l on condition c):        nu_cl ~ N(0, exp(alpha_l)), omega_cl ~ N(0, sigma_omega^2)
# The combined model substitutes level 2 into level 1: y_rcl = beta_l + beta_c + nu_cl + exp{(tau_l + gamma_c +
# omega_cl)/2} u_rcl, u_rcl ~ N(0, 1). melsvmp's model is gamma_c = 0.
# Data as in R/04 (accuracy scores of the simulated tasks, 10 replicates per condition, 19 learners, cells with
# (near-)constant scores dropped). Output: output/mels_quad_10reps_19.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

BIN <- "rust/mels_quad/target/release/mels_quad"
if (!file.exists(BIN)) system("cd rust/mels_quad && cargo build --release")
r <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_more_19.csv"))
L <- intersect(c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default", "xgb_tuned", "svm", "nnet", "stack",
        "lightgbm", "catboost", "bart", "lightgbm_tuned", "catboost_tuned"), unique(r$learner))
a <- r[r$criterion == "acc" & r$source == "sim" & is.finite(r$score), ]
a$cell <- paste(a$cond, a$learner, sep = ":")
v0 <- tapply(a$score, a$cell, var); a <- a[!a$cell %in% names(v0)[v0 < 1e-8], ]
conds <- sort(unique(a$cond))
d <- data.frame(learner = match(a$learner, L) - 1, cond = match(a$cond, conds) - 1, cell = match(a$cell, unique(a$cell)) - 1, y = a$score)
B <- as.integer(Sys.getenv("MELS_QUAD_B", "0"))
tmp <- tempfile(fileext = ".csv"); write.csv(d, tmp, row.names = FALSE, quote = FALSE)
run <- function(args) {
  out <- system2(normalizePath(BIN), c(tmp, args), stdout = TRUE)
  ll <- as.numeric(sub("#loglik,", "", grep("^#loglik", out, value = TRUE)))
  f <- read.csv(text = out[!grepl("^#", out)])
  f$learner <- ifelse(grepl("_learner_", f$parameter), L[as.integer(sub(".*_learner_", "", f$parameter)) + 1], NA)
  f$cond <- ifelse(grepl("_cond_", f$parameter), conds[as.integer(sub(".*_cond_", "", f$parameter)) + 1], NA)
  f$loglik <- ll; f }
fits <- list()
for (g in c(FALSE, TRUE)) {
  t0 <- Sys.time()
  f <- run(c(if (!g) "--no-gamma", if (B > 0) c("--boot", B, "--seed", 2026)))
  f$model <- if (g) "with condition variance effects" else "melsvmp's model (no condition variance effects)"
  f$secs <- as.numeric(difftime(Sys.time(), t0, units = "secs")); fits[[length(fits) + 1]] <- f
  message(f$model[1], ": loglik ", round(f$loglik[1], 2), ", ", round(f$secs[1]), " s")
}
F <- do.call(rbind, lapply(fits, function(f) f[, intersect(c("model", "parameter", "learner", "cond", "estimate", "se", "boot_se", "boot_lo", "boot_hi", "boot_ok", "loglik", "secs"), names(f))]))
write.csv(F, "output/mels_quad_10reps_19.csv", row.names = FALSE)
M <- read.csv("output/mels_volatility_10reps_19.csv"); M <- M[M$fit == "raw scores", ]
cmp <- data.frame(learner = L,
  alpha_melsvmp = M$estimate[match(paste0("alpha_", L), M$parameter)],
  alpha_ml_same = F$estimate[F$model == fits[[1]]$model[1] & grepl("^alpha", F$parameter)][match(L, F$learner[F$model == fits[[1]]$model[1] & grepl("^alpha", F$parameter)])],
  alpha_ml_gamma = F$estimate[F$model == fits[[2]]$model[1] & grepl("^alpha", F$parameter)][match(L, F$learner[F$model == fits[[2]]$model[1] & grepl("^alpha", F$parameter)])],
  tau_melsvmp = M$estimate[match(paste0("tau_", L), M$parameter)],
  tau_ml_same = F$estimate[F$model == fits[[1]]$model[1] & grepl("^tau", F$parameter)][match(L, F$learner[F$model == fits[[1]]$model[1] & grepl("^tau", F$parameter)])],
  tau_ml_gamma = F$estimate[F$model == fits[[2]]$model[1] & grepl("^tau", F$parameter)][match(L, F$learner[F$model == fits[[2]]$model[1] & grepl("^tau", F$parameter)])])
print(cmp, digits = 3)
cat("sigma_omega: melsvmp", M$estimate[M$parameter == "sigma_omega"], " ML same model", exp(F$estimate[F$parameter == "log_sigma_omega"][1]),
    " ML with gamma", exp(F$estimate[F$parameter == "log_sigma_omega"][2]), "\n")
cat("log-likelihoods:", sapply(fits, function(f) round(f$loglik[1], 2)), " LR statistic", round(2 * diff(sapply(fits, function(f) f$loglik[1])), 1),
    "on", length(conds) - 1, "df\n")
