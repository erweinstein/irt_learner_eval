# 08_threshold_sensitivity.R -- How much do the learners' abilities depend on the pass thresholds (FLAGS.md item 1)?
# Each response's score is stored, so pass/fail can be recomputed under other thresholds, one criterion at a time
# and all together, and the multilevel Rasch model with the learner-by-family interaction (model A1 of
# R/03_irt.R) refitted under each. The one exception: on yes/no outcomes the calibration criterion also needs the
# predicted mean within 15% of the training mean, and that ratio is not stored, so its slope band can only be
# narrowed (a response passes the narrower band if it passed before and its slope is inside it); the lenient
# scenarios leave it as it was.
# Input: IRT_RESPONSES (default output/responses_19.csv). Output in IRT_OUT (default output/v19):
#   threshold_sensitivity_abilities.csv (learner x scenario), threshold_sensitivity_summary.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(library(glmmTMB))
RESP <- Sys.getenv("IRT_RESPONSES", "output/responses_19.csv"); OUT <- Sys.getenv("IRT_OUT", "output/v19")
R <- read.csv(RESP, stringsAsFactors = FALSE)
R <- R[!is.na(R$pass), ]
R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$item <- paste(R$inst, R$criterion, sep = ":")
R$lf <- paste(R$learner, R$task_family, sep = "|")
truth <- R$source != "real"

# pass under a set of thresholds; th = the baseline unless changed
base <- list(auc = 0.9, r2 = 0.8, tol = 0.01, tol_cost = 0.02, slope = c(0.8, 1.25), mean_band = 0.10, eff = 1.05, imp = 3)
pass_under <- function(th) {
  p <- R$pass; s <- R$score; a <- R$criterion == "acc"; ca <- R$criterion == "cal"
  i <- a & truth & R$type == "binary"; p[i] <- as.integer(s[i] >= th$auc)
  i <- a & truth & R$type != "binary"; p[i] <- as.integer(s[i] >= th$r2)
  i <- a & !truth; tl <- ifelse(R$type[i] == "cost", th$tol_cost, th$tol); p[i] <- as.integer(s[i] >= -tl - 1e-12)
  i <- ca & R$type %in% c("count", "cost"); p[i] <- as.integer(abs(s[i] - 1) <= th$mean_band + 1e-12)
  i <- ca & R$type == "binary"                       # narrowing only (see the header)
  inb <- s[i] >= th$slope[1] & s[i] <= th$slope[2]
  p[i] <- if (th$slope[1] >= base$slope[1] && th$slope[2] <= base$slope[2]) as.integer(R$pass[i] == 1 & inb) else R$pass[i]
  i <- R$criterion == "eff"; p[i] <- as.integer(s[i] <= th$eff)
  i <- R$criterion == "imp"; p[i] <- as.integer(s[i] >= th$imp)
  p
}
mod <- function(...) modifyList(base, list(...))
acc_strict <- list(auc = 0.95, r2 = 0.9, tol = 0.005, tol_cost = 0.01)
acc_lenient <- list(auc = 0.8, r2 = 0.7, tol = 0.02, tol_cost = 0.04)
cal_strict <- list(slope = c(0.9, 1.11), mean_band = 0.05); cal_lenient <- list(mean_band = 0.20)
S <- list(
  baseline = base,
  acc_strict = do.call(mod, acc_strict), acc_lenient = do.call(mod, acc_lenient),
  cal_strict = do.call(mod, cal_strict), cal_lenient = do.call(mod, cal_lenient),
  eff_strict = mod(eff = 1.02), eff_lenient = mod(eff = 1.10), imp_lenient = mod(imp = 2),
  all_strict = do.call(mod, c(acc_strict, cal_strict, list(eff = 1.02))),
  all_lenient = do.call(mod, c(acc_lenient, cal_lenient, list(eff = 1.10, imp = 2))))

stopifnot(all(pass_under(base) == R$pass))        # the baseline reproduces the stored pass/fail
ab <- list(); sm <- list()
for (k in names(S)) {
  D <- R; D$pass <- pass_under(S[[k]])
  fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) + (1 | lf), D, binomial)
  fe <- fixef(fit)$cond; fe <- fe[grepl("^learner", names(fe))]; names(fe) <- sub("^learner", "", names(fe))
  th <- fe - mean(fe)
  ab[[k]] <- data.frame(learner = names(th), scenario = k, theta = unname(th), prop_met = unname(tapply(D$pass, D$learner, mean)[names(th)]))
  lf_sd <- attr(VarCorr(fit)$cond$lf, "stddev")
  sm[[k]] <- data.frame(scenario = k, changed = sum(D$pass != R$pass), prop_met = mean(D$pass), lf_sd = unname(lf_sd), sd_theta = sd(th))
  message(k, ": ", sm[[k]]$changed, " responses changed")
}
A <- do.call(rbind, ab); Sm <- do.call(rbind, sm)
b <- A[A$scenario == "baseline", ]
Sm$spearman <- sapply(Sm$scenario, function(k) { a <- A[A$scenario == k, ]; cor(a$theta, b$theta[match(a$learner, b$learner)], method = "spearman") })
Sm$max_rank_change <- sapply(Sm$scenario, function(k) { a <- A[A$scenario == k, ]; max(abs(rank(-a$theta) - rank(-b$theta[match(a$learner, b$learner)]))) })
Sm$top <- sapply(Sm$scenario, function(k) { a <- A[A$scenario == k, ]; a$learner[which.max(a$theta)] })
write.csv(A, file.path(OUT, "threshold_sensitivity_abilities.csv"), row.names = FALSE)
write.csv(Sm, file.path(OUT, "threshold_sensitivity_summary.csv"), row.names = FALSE)
print(Sm, digits = 3)
W <- reshape(A[, c("learner", "scenario", "theta")], idvar = "learner", timevar = "scenario", direction = "wide")
names(W) <- sub("theta.", "", names(W)); print(W[order(-W$baseline), ], digits = 2, row.names = FALSE)
