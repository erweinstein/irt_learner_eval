# 09b_laplace_consistency.R -- A joint check of the Laplace approximation for the adaptive tests' calibration model
# (FLAGS.md item 25), with all the crossed random effects at once. TMB::checkConsistency (Kristensen et al. 2016,
# TMB) simulates data sets from the fitted model and evaluates the gradient of the Laplace-approximated marginal
# log-likelihood at the parameters they were simulated from. If the approximation were exact, that gradient would
# average zero (the score is unbiased at the true parameters), whatever the small-sample bias of the estimates;
# a nonzero average, converted to the parameter scale with the Hessian, is the bias the approximation adds. So it
# separates the approximation's share of the bias in R/09_laplace_check.R from that of maximum likelihood.
# Output: output/v21/laplace_consistency.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(TMB) })
set.seed(915)
OUT <- Sys.getenv("IRT_OUT", "output/v21"); N <- as.integer(Sys.getenv("CONSIST_N", "200"))
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_21.csv"), stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":"); R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$lf <- paste(R$learner, R$task_family); R$lc <- paste(R$learner, R$cond)
# Two changes from the calibration model, so that the marginal test's Hessian is not singular: the importance
# criterion is left out (all 108 of its responses pass, so its fixed effect is infinite), and so is the instance
# effect (its SD is estimated at zero, on the boundary).
R <- R[R$criterion != "imp", ]
fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | item) + (1 | lf) + (1 | lc), R, binomial)
cc <- TMB::checkConsistency(fit$obj, n = N, hessian = TRUE, observation.name = "yobs")
s <- summary(cc); print(s)
par <- fit$obj$env$last.par.best[-fit$obj$env$random]
nm <- names(par); nm[nm == "theta"] <- paste0("log_sd_", c("task_family", "cond", "item", "lf", "lc"))
B <- data.frame(param = nm, estimate = unname(par), laplace_bias = unname(s$marginal$bias),
                p_marginal = s$marginal$p.value, p_joint = s$joint$p.value,
                sd = exp(unname(par)), sd_corrected = exp(unname(par) - unname(s$marginal$bias)))
write.csv(B, file.path(OUT, "laplace_consistency.csv"), row.names = FALSE)
print(B[grepl("log_sd", B$param), ], digits = 3)
