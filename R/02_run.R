# 02_run.R -- Run every learner on every task instance and score the responses. Each instance is saved as
# output/instances/c<cond>_r<rep>.rds as it finishes, so an interrupted run resumes where it stopped.
# Environment variables: IRT_FAMILIES (comma-separated families to run; default all), IRT_REPS (replicates
# per condition; default 3, 4 for procova, 5 for dml), IRT_CORES (default 2), IRT_INSTANCE_DIR (where instances are
# saved; default output/instances).
# Output (after all instances): output/responses.csv, one row per instance x criterion x learner:
#   inst, cond, rep, family, source, type, n, detail, criterion, learner, score, pass
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/00_learners.R"); source("R/01_tasks.R")
suppressMessages(library(parallel))
C <- conditions(); CL <- attr(C, "list")
FAM <- Sys.getenv("IRT_FAMILIES", ""); CORES <- as.integer(Sys.getenv("IRT_CORES", "2"))
IDIR <- Sys.getenv("IRT_INSTANCE_DIR", "output/instances")
# IRT_LEARNERS: run only these learners (comma-separated), e.g. the five added on 11 November
LEARNERS_ALL <- LEARNERS
if (Sys.getenv("IRT_LEARNERS") != "") LEARNERS <- strsplit(Sys.getenv("IRT_LEARNERS"), ",")[[1]]
# IRT_PLAN = "v19": the run that adds the five learners of 11 November to the original 301 instances. LightGBM,
# CatBoost and BART are run on every instance; RealMLP and TabPFN (slow on a CPU) on the first replicate of the
# prediction tasks only; on real-data instances, whose accuracy is judged against the best learner, all the
# learners are rerun so that the reference includes the new ones (the original 14 reproduce their earlier
# predictions, since each learner's seed is fixed by its place in the full list)
PLAN <- Sys.getenv("IRT_PLAN", "")
plan_learners <- function(cond, rep) {
  fast_new <- c("lightgbm", "catboost", "bart")
  slow <- if (rep == 1 && !cond$family %in% c("dml", "procova", "import")) c("realmlp", "tabpfn") else character(0)
  # plasmode instances too: before the fix of 12 November in load_meps() their three replicates were the same
  # data set in the original run, so every learner is rerun on them
  if (cond$source %in% c("real", "plasmode")) c(LEARNERS_ORIGINAL, fast_new, slow) else c(fast_new, slow)
}
# IRT_PLAN = "v21" (13 November 2026): tuned LightGBM and CatBoost on every instance (output/instances_v21). The
# real-data accuracy reference stays the best of the 17 learners of the v19 run (REF_OUT below), so adding them
# changes no other learner's responses; on real-data instances those 17 are rerun to rebuild the reference (they
# reproduce their v19 predictions; R/02f_merge_v21.R checks that their scores agree)
TUNED_NEW <- c("lightgbm_tuned", "catboost_tuned")
plan_learners21 <- function(cond, rep)
  if (cond$source == "real") c(setdiff(LEARNERS_ALL[1:19], c("realmlp", "tabpfn")), TUNED_NEW) else TUNED_NEW
# the slow learners are not run on the importance tasks, whose test set is the predictors' permuted copies
# (11 times the usual test rows); their importance responses are missing by design
SLOW <- c("realmlp", "tabpfn")
REF_OUT <- c(SLOW, "lightgbm_tuned", "catboost_tuned")   # not in the real-data accuracy reference
reps_for <- function(f) as.integer(Sys.getenv("IRT_REPS", c(procova = "4", dml = "5")[f] %||% "3"))
`%||%` <- function(a, b) if (is.null(a) || is.na(a)) b else a

# ---- scoring --------------------------------------------------------------------------------------------------
signal_r2 <- function(p, truth) 1 - mean((p - truth)^2) / mean((truth - mean(truth))^2)
score_pred <- function(inst, P, cond) {
  y <- inst$yte; type <- inst$type; out <- list()
  ok <- !vapply(P, is.null, TRUE); P <- P[ok]
  good <- vapply(P, function(p) all(is.finite(p)), TRUE)
  if (!is.null(inst$truth)) {                                   # simulated or plasmode: against the oracle
    if (type == "binary") {
      ao <- auc(y, inst$truth)
      s <- vapply(P, function(p) if (all(is.finite(p))) (auc(y, p) - 0.5) / (ao - 0.5) else NA, 0)
      out$acc <- data.frame(learner = names(P), score = s, pass = as.integer(s >= 0.9))
    } else {
      s <- vapply(P, function(p) if (all(is.finite(p))) signal_r2(p, inst$truth) else NA, 0)
      out$acc <- data.frame(learner = names(P), score = s, pass = as.integer(s >= 0.8))
    }
  } else {                                                      # real data: against the best learner
    m <- vapply(P, function(p) if (all(is.finite(p))) switch(type, binary = auc(y, p), count = d2(y, p), r2(y, p)) else NA, 0)
    tol <- if (type == "cost") 0.02 else 0.01
    # the reference is the best learner among those run on every real-data instance (the slow learners, run on
    # the first replicate only, are scored against it but do not set it)
    best <- max(m[!names(m) %in% REF_OUT], na.rm = TRUE)
    out$acc <- data.frame(learner = names(P), score = m - best, pass = as.integer(m >= best - tol))
  }
  # calibration in the large is judged against the training mean, so that it reflects the learner (shrinkage,
  # a mismatched loss) rather than the sampling error of the training sample's mean
  ref_mean <- mean(inst$ytr)
  if (type == "binary") {
    s <- vapply(P, function(p) if (all(is.finite(p))) cal_slope(y, p) else NA, 0)
    r <- vapply(P, function(p) mean(p) / ref_mean, 0)
    out$cal <- data.frame(learner = names(P), score = s, pass = as.integer(s >= 0.8 & s <= 1.25 & abs(r - 1) <= 0.15))
  }
  if (type %in% c("count", "cost")) {
    r <- vapply(P, function(p) mean(p) / ref_mean, 0)
    out$cal <- data.frame(learner = names(P), score = r, pass = as.integer(abs(r - 1) <= 0.10))
  }
  for (k in names(out)) { out[[k]]$pass[is.na(out[[k]]$score)] <- NA; out[[k]]$criterion <- k }
  do.call(rbind, out)
}

run_learners <- function(inst, learners = LEARNERS, seed = 1) {
  P <- list()
  # each learner's seed depends on its place in the full list, so a learner gives the same predictions
  # whichever others are run with it
  for (j in seq_along(learners)) { set.seed(seed + match(learners[j], LEARNERS_ALL))
    P[learners[j]] <- list(fit_predict(learners[j], inst$type, inst$xtr, inst$ytr, inst$xte)) }
  P
}

# ---- the inference families -----------------------------------------------------------------------------------
procova_instance <- function(cond, seed) {
  f <- if (cond$kind == "linear") f_lin else function(x) f_nonlin(x, "threshold")
  xh <- sim_x(cond$n); fh <- f(xh); s <- noise_for(fh, 0.5); yh <- fh + rnorm(cond$n, 0, s)
  xt <- sim_x(300); ft <- f(xt); A <- rep(0:1, 150); yt <- ft + 0.3 * A + rnorm(300, 0, s)
  se_with <- function(score) summary(lm(yt ~ A + score))$coefficients["A", 2]
  se0 <- se_with(ft)
  rows <- list()
  for (j in seq_along(LEARNERS)) { set.seed(seed + match(LEARNERS[j], LEARNERS_ALL))
    p <- fit_predict(LEARNERS[j], "gaussian", xh, yh, xt)
    if (is.null(p)) next
    sc <- if (all(is.finite(p))) se_with(p) / se0 else NA
    rows[[j]] <- data.frame(learner = LEARNERS[j], score = sc, pass = as.integer(sc <= 1.05), criterion = "eff") }
  do.call(rbind, rows)
}
dml_instance <- function(cond, seed) {
  n <- cond$n; x <- sim_x(n); tau <- 0.5
  ps <- switch(cond$kind, nonlin_both = plogis(0.9 * as.numeric(scale(f_nonlin(x, "threshold")))), plogis(0.9 * as.numeric(scale(f_lin(x)))))
  ps <- pmin(pmax(ps, 0.05), 0.95); A <- rbinom(n, 1, ps)
  mu0 <- if (cond$kind == "linear") f_lin(x) else f_nonlin(x, "threshold")
  y <- mu0 + tau * A + rnorm(n, 0, noise_for(mu0, 0.5))
  fo <- sample(rep_len(1:2, n)); rows <- list()
  for (j in seq_along(LEARNERS)) {
    l <- LEARNERS[j]; if (l %in% c("stack", "glm_matched")) next
    psi <- numeric(n); okk <- TRUE
    for (k in 1:2) { tr <- fo != k; te <- fo == k; set.seed(seed + 10 * match(l, LEARNERS_ALL) + k)
      xa <- cbind(x, A = A)
      m <- fit_predict(l, "gaussian", xa[tr, ], y[tr], rbind(cbind(x[te, ], A = 1), cbind(x[te, ], A = 0)))
      e <- fit_predict(l, "binary", x[tr, ], A[tr], x[te, ])
      if (is.null(m) || is.null(e) || any(!is.finite(m)) || any(!is.finite(e))) { okk <- FALSE; break }
      nt <- sum(te); m1 <- m[seq_len(nt)]; m0 <- m[nt + seq_len(nt)]; e <- pmin(pmax(e, 0.02), 0.98)
      psi[te] <- m1 - m0 + A[te] * (y[te] - m1) / e - (1 - A[te]) * (y[te] - m0) / (1 - e) }
    if (!okk) { rows[[j]] <- data.frame(learner = l, score = NA, pass = NA, criterion = "cover"); next }
    est <- mean(psi); se <- sd(psi) / sqrt(n)
    rows[[j]] <- data.frame(learner = l, score = (est - tau) / se, pass = as.integer(abs(est - tau) <= 1.96 * se), criterion = "cover") }
  do.call(rbind, rows)
}

# ---- one instance ----------------------------------------------------------------------------------------------
run_instance <- function(i, rep) {
  f <- sprintf("%s/c%03d_r%d.rds", IDIR, i, rep)
  if (file.exists(f)) return(invisible(NULL))
  cond <- CL[[i]]; seed <- 100000 + 1000 * i + rep; set.seed(seed); t0 <- Sys.time()
  if (PLAN == "v19") LEARNERS <<- plan_learners(cond, rep)
  if (PLAN == "v21") LEARNERS <<- plan_learners21(cond, rep)
  res <- if (cond$family == "procova") procova_instance(cond, seed) else if (cond$family == "dml") dml_instance(cond, seed) else {
    inst <- switch(cond$source, sim = sim_instance(cond), plasmode = plasmode_instance(cond), real = real_instance(cond))
    if (cond$family == "import") {               # importance: predict the test data and its column-permuted copies
      pp <- ncol(inst$xte); xa <- inst$xte
      for (k in seq_len(pp)) { xk <- inst$xte; xk[, k] <- sample(xk[, k]); xa <- rbind(xa, xk) }
      n0 <- nrow(inst$xte); inst2 <- inst; inst2$xte <- xa
      Pa <- run_learners(inst2, learners = setdiff(LEARNERS, SLOW), seed = seed)
      P <- lapply(Pa, function(p) if (is.null(p)) NULL else p[seq_len(n0)])
      r <- score_pred(inst, P, cond)
      imp <- do.call(rbind, lapply(names(Pa), function(l) { p <- Pa[[l]]; if (is.null(p) || any(!is.finite(p))) return(NULL)
        base <- mean((inst$yte - p[seq_len(n0)])^2)
        vi <- vapply(seq_len(pp), function(k) mean((inst$yte - p[k * n0 + seq_len(n0)])^2) - base, 0)
        top <- colnames(inst$xte)[order(-vi)][1:3]
        data.frame(learner = l, score = sum(top %in% inst$signal), pass = as.integer(setequal(top, inst$signal)), criterion = "imp") }))
      rbind(r, imp)
    } else score_pred(inst, run_learners(inst, seed = seed), cond)
  }
  res$inst <- sprintf("c%03d_r%d", i, rep); res$cond <- i; res$rep <- rep
  res$secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  saveRDS(res, f); invisible(res)
}

if (sys.nframe() == 0) {
  dir.create(IDIR, showWarnings = FALSE, recursive = TRUE)
  todo <- do.call(rbind, lapply(seq_len(nrow(C)), function(i) if (FAM == "" || C$family[i] %in% strsplit(FAM, ",")[[1]])
    data.frame(i = i, rep = seq_len(reps_for(C$family[i])))))
  todo <- todo[!file.exists(sprintf("%s/c%03d_r%d.rds", IDIR, todo$i, todo$rep)), ]
  # IRT_REP_ONLY = "1" runs only the first replicate; "2+" only the others (to split long runs into chunks)
  ro <- Sys.getenv("IRT_REP_ONLY", ""); if (ro == "1") todo <- todo[todo$rep == 1, ]; if (ro == "2+") todo <- todo[todo$rep > 1, ]
  # IRT_N_MIN / IRT_N_MAX restrict to conditions by training size (to run the most memory-hungry ones on one core)
  nn <- C$n[match(todo$i, C$cond)]
  todo <- todo[nn >= as.numeric(Sys.getenv("IRT_N_MIN", "0")) & nn <= as.numeric(Sys.getenv("IRT_N_MAX", "Inf")), ]
  message(nrow(todo), " instances to run, ", format(Sys.time()))
  invisible(mclapply(seq_len(nrow(todo)), function(k) { r <- try(run_instance(todo$i[k], todo$rep[k]))
    if (inherits(r, "try-error")) message("FAILED ", todo$i[k], "/", todo$rep[k], ": ", r)
    else message("done c", todo$i[k], " r", todo$rep[k], " ", format(Sys.time())) },
    mc.cores = CORES, mc.preschedule = FALSE))
}
