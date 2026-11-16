# 09_laplace_check.R -- Does the Laplace approximation understate the variance components of the adaptive tests'
# calibration model (FLAGS.md item 25)? Hedeker (2015) recommends adaptive quadrature with several points; glmmTMB
# uses the Laplace approximation (adaptive quadrature with one point), and with binary responses and small
# clusters (a learner has a median of 6 task-items in a task condition) that can bias variance components
# toward zero. Two checks, on the calibration model of R/06_bifactor_cat.R fitted to all 21 learners:
#
# A. Parametric bootstrap. Simulate new responses from the fitted model (new random effects drawn from their
#    estimated distributions; fixed effects as estimated), refit by glmmTMB, and compare each estimated SD with
#    the one it was simulated from. This measures the bias of the whole procedure in this design (the Laplace
#    approximation plus the small-sample bias of maximum likelihood), for binary and for graded responses (the
#    continuation-ratio model, stacked steps).
# B. Laplace against adaptive quadrature on the real responses. lme4 allows more than one quadrature point only
#    for a single scalar random effect, so each component is refitted alone, with every other term of the
#    glmmTMB fit held in an offset: glmer(pass ~ 1 + (1 | g) + offset, nAGQ = 1) against nAGQ = 25, for
#    g = learner by condition, learner by family, item and condition. The difference isolates the approximation.
#
# Output: output/v21/laplace_bootstrap.csv (one row per replicate and component), output/v21/laplace_check.csv
# (summary), output/v21/laplace_agq.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(lme4); library(parallel) })
RNGkind("L'Ecuyer-CMRG"); set.seed(909)
OUT <- Sys.getenv("IRT_OUT", "output/v21"); B <- as.integer(Sys.getenv("LAPLACE_B", "100"))
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_21.csv"), stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":"); R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$lf <- paste(R$learner, R$task_family); R$lc <- paste(R$learner, R$cond)
L <- readLines("R/03_irt.R")
eval(parse(text = L[grep("^grade <- function", L):(grep("^R\\$grade <- grade", L) - 1)]))
R$grade <- grade(R); R$grade[is.na(R$grade) & R$pass == 0] <- 0
TERMS <- c("task_family", "cond", "inst", "item", "lf", "lc")
RE <- paste(sprintf("(1 | %s)", TERMS), collapse = " + ")
stacked <- function(D) {
  S <- rbind(transform(D, y = as.integer(grade >= 1), crit = paste(criterion, "s1")),
             transform(D[D$grade >= 1, ], y = as.integer(grade == 2), crit = paste(criterion, "s2")))
  S }
fit_one <- function(D, graded) {
  if (graded) { D <- stacked(D) } else { D$y <- D$pass; D$crit <- D$criterion }
  glmmTMB(as.formula(paste("y ~ 0 + learner + crit +", RE)), D, binomial)
}
sds <- function(fit) { v <- VarCorr(fit)$cond; sapply(TERMS, function(t) unname(attr(v[[t]], "stddev"))) }

# simulate from a fitted model: new random effects, fixed effects as estimated
sim_from <- function(fit, D, graded) {
  sd0 <- sds(fit); fe <- fixef(fit)$cond
  u <- lapply(TERMS, function(t) { lv <- unique(D[[t]]); setNames(rnorm(length(lv), 0, sd0[[t]]), lv) }); names(u) <- TERMS
  eta_re <- Reduce(`+`, lapply(TERMS, function(t) u[[t]][D[[t]]]))
  lfe <- function(crit) unname(fe[paste0("learner", D$learner)]) + ifelse(is.na(fe[paste0("crit", crit)]), 0, fe[paste0("crit", crit)])
  if (!graded) { D$pass <- rbinom(nrow(D), 1, plogis(lfe(D$criterion) + eta_re)); return(D) }
  s1 <- rbinom(nrow(D), 1, plogis(lfe(paste(D$criterion, "s1")) + eta_re))
  s2 <- rbinom(nrow(D), 1, plogis(lfe(paste(D$criterion, "s2")) + eta_re))
  D$grade <- ifelse(s1 == 0, 0, ifelse(s2 == 0, 1, 2)); D
}

# ---- A. parametric bootstrap ------------------------------------------------------------------------------------
boot <- list()
for (graded in c(FALSE, TRUE)) {
  sc <- if (graded) "graded" else "binary"
  F0 <- fit_one(R, graded); truth <- sds(F0)
  th0 <- fixef(F0)$cond; th0 <- th0[grepl("^learner", names(th0))]
  message(sc, " fit: ", paste(names(truth), round(truth, 3), collapse = ", "))
  rows <- mclapply(seq_len(B), function(b) {
    Db <- sim_from(F0, R, graded)
    Fb <- tryCatch(fit_one(Db, graded), error = function(e) NULL)
    if (is.null(Fb)) return(NULL)
    thb <- fixef(Fb)$cond; thb <- thb[names(th0)]
    data.frame(scoring = sc, rep = b, term = TERMS, truth = unname(truth), est = unname(sds(Fb)),
               conv = Fb$fit$convergence == 0 && isTRUE(Fb$sdr$pdHess),
               learner_sd_ratio = sd(thb) / sd(th0), learner_cor = cor(thb, th0))
  }, mc.cores = 2, mc.set.seed = TRUE)
  boot[[sc]] <- do.call(rbind, rows)
}
BT <- do.call(rbind, boot)
write.csv(BT, file.path(OUT, "laplace_bootstrap.csv"), row.names = FALSE)
summ <- do.call(rbind, lapply(split(BT, list(BT$scoring, BT$term), drop = TRUE), function(x) {
  x <- x[x$conv, ]
  data.frame(scoring = x$scoring[1], term = x$term[1], reps = nrow(x), truth = x$truth[1], mean_est = mean(x$est),
             median_est = median(x$est), ratio_mean = mean(x$est) / x$truth[1], ratio_median = median(x$est) / x$truth[1],
             mcse_ratio = sd(x$est) / sqrt(nrow(x)) / x$truth[1], share_below = mean(x$est < x$truth[1]),
             learner_sd_ratio = mean(x$learner_sd_ratio), learner_cor = mean(x$learner_cor)) }))
summ <- summ[order(summ$scoring, match(summ$term, TERMS)), ]
write.csv(summ, file.path(OUT, "laplace_check.csv"), row.names = FALSE)
print(summ, digits = 3, row.names = FALSE)

# ---- B. Laplace against adaptive quadrature, one component at a time ------------------------------------------------
F0 <- fit_one(R, FALSE); R$y <- R$pass
eta <- predict(F0, type = "link")
reb <- ranef(F0)$cond
blup <- function(t) { v <- reb[[t]][[1]]; out <- v[match(R[[t]], rownames(reb[[t]]))]; out[is.na(out)] <- 0; out }
agq <- do.call(rbind, lapply(c("lc", "lf", "item", "cond"), function(t) {
  R$off <- eta - blup(t); R$g <- R[[t]]
  r <- lapply(c(1, 25), function(q) { f <- glmer(y ~ 1 + (1 | g) + offset(off), R, binomial, nAGQ = q)
    c(sd = unname(attr(VarCorr(f)$g, "stddev")), intercept = unname(fixef(f)[1]), deviance = -2 * as.numeric(logLik(f))) })
  data.frame(term = t, glmmTMB_sd = unname(sds(F0)[t]), laplace_sd = r[[1]][["sd"]], agq25_sd = r[[2]][["sd"]],
             ratio = r[[1]][["sd"]] / r[[2]][["sd"]], laplace_int = r[[1]][["intercept"]], agq25_int = r[[2]][["intercept"]])
}))
write.csv(agq, file.path(OUT, "laplace_agq.csv"), row.names = FALSE)
print(agq, digits = 3, row.names = FALSE)
