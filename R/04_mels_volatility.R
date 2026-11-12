# 04_mels_volatility.R -- How variable is each learner's accuracy from one draw of a task to the next? A
# mixed-effects location-scale model (Hedeker, Mermelstein and Demirtas 2008), fitted by variational message
# passing (Wu and Hedeker 2026, package melsvmp), on the continuous accuracy scores of the simulated tasks
# (fraction of the oracle's signal R^2, or of its AUC above 0.5). A "subject" is one learner on one task
# condition, observed on its replicate draws (3, or 10 in the rerun):
#   Level 1 (draw r of condition c, learner l):  y_rcl = x'_cl beta + nu_cl + e_rcl,
#                                                 e_rcl ~ N(0, exp(tau_l + omega_cl))
#   Level 2 (condition c, learner l):             nu_cl ~ N(0, exp(alpha_l)),  omega_cl ~ N(0, sigma_omega^2)
# with x'beta = a learner effect plus a condition effect, so nu_cl is the learner-by-condition interaction.
# exp(tau_l) is learner l's typical variance across replicate draws of the same condition (its volatility, or
# instability); exp(alpha_l) is how much its average accuracy varies across conditions beyond what the
# condition does to every learner (its sensitivity to the task); sigma_omega is how much the volatility itself
# varies across conditions. Inference: the robust sandwich standard errors
# melsvmp reports, and a nonparametric bootstrap (B = 500), which Wu and Hedeker found kept its coverage for the
# variance parameters where the sandwich did not. melsvmp's own bootstrap resamples subjects; here the 14 cells
# of one condition share its draws, so the bootstrap resamples conditions, each with all its cells.
# The scores have a long lower tail (the default xgboost and the neural net sometimes do much worse than predicting the mean), so the model is
# also refitted with scores floored at -1.
# Input: IRT_RESPONSES (default output/responses.csv). Output: MELS_OUT.csv and MELS_OUT_fig.png (default
# output/mels_volatility).
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(melsvmp); library(ggplot2) })
L <- c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default", "xgb_tuned", "svm", "nnet", "stack")
r <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses.csv"))
a <- r[r$criterion == "acc" & r$source == "sim" & is.finite(r$score), ]
a$learner <- factor(a$learner, L); a$family <- factor(a$family)
a$cell <- paste(a$cond, a$learner, sep = ":")
a <- a[order(a$cell, a$rep), ]
# a cell whose 3 scores are identical (the pruned tree predicting a constant) has a within-cell variance of
# zero, which sends its scale random effect to minus infinity and makes the sandwich singular; such cells are
# dropped and counted
v0 <- tapply(a$score, a$cell, var); drop <- names(v0)[v0 < 1e-8]
cat("dropped cells with (near-)constant scores:", paste(drop, collapse = ", "), "\n")
a <- a[!a$cell %in% drop, ]
cat(nrow(a), "scores,", length(unique(a$cell)), "learner-condition cells,", length(unique(a$cond)), "conditions\n")

# melsvmp's sandwich inverts a 2 x 2 Hessian for each subject's (nu, omega); for a cell whose 3 scores are nearly
# equal that matrix is numerically singular and the fit stops. Here such a subject is left out of the bread
# (its score still enters the meat), and the number left out is counted; the bootstrap does not use the sandwich.
ns <- asNamespace("melsvmp"); se_fun <- get("san_est", ns); b <- body(se_fun)
txt <- deparse(se_fun); i <- grep("H_local_inv <- solve(H_local)", txt, fixed = TRUE)
txt[i] <- "H_local_inv <- tryCatch(solve(H_local), error = function(e) NULL); if (is.null(H_local_inv)) { SKIPPED <<- SKIPPED + 1; next }"
se_new <- eval(parse(text = txt)); environment(se_new) <- ns
unlockBinding("san_est", ns); assign("san_est", se_new, ns); lockBinding("san_est", ns)
SKIPPED <- 0
suppressMessages(library(parallel)); RNGkind("L'Ecuyer-CMRG"); set.seed(4004)
# melsvmp rebuilds a formula from the design's column names, so the dummies are made here with plain names
fit_mels <- function(d) {
  X <- model.matrix(~ 0 + learner, d); colnames(X) <- paste0("L_", L)
  C <- model.matrix(~ factor(cond), d)[, -1, drop = FALSE]; colnames(C) <- paste0("C_", seq_len(ncol(C)))
  d <- cbind(d, X, C)
  mels_vmp(y = "y", beta_formula = reformulate(c("0", colnames(X), colnames(C))), alpha_formula = reformulate(c("0", colnames(X))),
           tau_formula = reformulate(c("0", colnames(X))), id = "cell", data = d, verbose = FALSE)
}
est <- function(f) c(setNames(f$beta$params[1:14], paste0("beta_", L)), setNames(f$alpha$params, paste0("alpha_", L)),
                     setNames(f$tau$params, paste0("tau_", L)), sigma_omega = f$omega$std_dev)
fit_one <- function(d, label, B = as.integer(Sys.getenv("MELS_B", "500"))) {
  SKIPPED <<- 0; f <- fit_mels(d); e <- est(f); cat(label, ": subjects left out of the sandwich's bread:", SKIPPED, "\n")
  sw <- c(sqrt(diag(f$beta$cov_mat))[1:14], sqrt(diag(f$alpha$cov_mat)), sqrt(diag(f$tau$cov_mat)), NA)
  conds <- unique(d$cond)
  bt <- do.call(rbind, mclapply(seq_len(B), function(b) {
    s <- sample(conds, replace = TRUE)
    db <- do.call(rbind, lapply(seq_along(s), function(j) { x <- d[d$cond == s[j], ]
      x$cell <- paste(j, x$cell); x$cond <- j; x }))
    tryCatch(est(fit_mels(db)), error = function(e) rep(NA, length(e)))
  }, mc.cores = 2, mc.set.seed = TRUE))
  data.frame(fit = label, parameter = names(e), estimate = unname(e), sandwich_se = unname(sw),
             boot_se = apply(bt, 2, sd, na.rm = TRUE), boot_lo = apply(bt, 2, quantile, 0.025, na.rm = TRUE),
             boot_hi = apply(bt, 2, quantile, 0.975, na.rm = TRUE), boot_ok = colSums(is.finite(bt)),
             runtime = f$runtime, row.names = NULL)
}
a$y <- a$score; F1 <- fit_one(a, "raw scores")
a$y <- pmax(a$score, -1); F2 <- fit_one(a, "scores floored at -1")
out <- rbind(F1, F2)
out$learner <- sub("^(beta|alpha|tau)_", "", out$parameter); out$learner[out$parameter == "sigma_omega"] <- NA
OUT <- Sys.getenv("MELS_OUT", "output/mels_volatility")
write.csv(out, paste0(OUT, ".csv"), row.names = FALSE)
# descriptive check: the average, over conditions, of each learner's variance across its replicate draws
a$y <- a$score
dv <- aggregate(y ~ cell + learner, a, var); dv <- aggregate(y ~ learner, dv, function(v) log(mean(v)))
names(dv)[2] <- "log_mean_within_variance"; print(dv)
w <- out[grepl("^tau_", out$parameter), ]
w$learner <- factor(w$learner, rev(L))
g <- ggplot(w, aes(estimate, learner, colour = fit)) +
  geom_pointrange(aes(xmin = boot_lo, xmax = boot_hi), position = position_dodge(width = 0.5)) +
  labs(x = "log variance across replicate draws of the same task (tau; 95% bootstrap percentile interval)",
       y = NULL, colour = NULL) + theme_minimal(base_size = 11) + theme(legend.position = "bottom")
ggsave(paste0(OUT, "_fig.png"), g, width = 7, height = 5, dpi = 200)
print(out[, c("fit", "parameter", "estimate", "sandwich_se", "boot_se", "boot_lo", "boot_hi", "boot_ok")], digits = 3)
