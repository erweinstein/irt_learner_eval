# 01_tasks.R -- The task bank. A "condition" is a kind of task (a family, an outcome type, a sample size and
# so on); an "instance" is one draw of it (one simulated data set, or one random split of a real one). Each
# instance is scored on one or more criteria, and each (instance, criterion) pair is one row of responses:
# for every learner, whether it met the criterion (1) or not (0), plus the continuous score behind it.
#
# Families (resembling the prediction problems of the MEPS machine-learning walkthrough):
#   Simulated, with a known E[y | x] (the "oracle"), so a learner is judged against the best possible:
#     lin      linear signal with one strong "last year's outcome" predictor (oracle R^2 = 0.5)
#     nonlin   Friedman's function, thresholds and interactions, or smooth curves
#     cost     spending-like: a lump at zero and a log-normal tail (two-part truth)
#     binary   rare to common yes/no outcomes (prevalence 5%, 15%, 35%), linear or with interactions
#     count    negative binomial counts, very or mildly overdispersed
#     sparse   n = 150, 100 predictors, 5 of them real
#     shift    the test data lie partly outside the training range (extrapolation)
#     missing  a predictor missing more often for people with higher outcomes (indicator informative)
#     import   explanation: does permutation importance put the three real predictors on top?
#   Plasmode: MEPS adults' real predictors with outcomes simulated from a known model fitted to MEPS
#   Real data (no oracle; judged against the best learner on the same split, and on calibration):
#     MEPS Panel 27 (log spending, spending, top 10%, any ER visit, office visits), AER's NMES1988 (the 1987-88
#     National Medical Expenditure Survey, MEPS's predecessor: office visits, any hospital stay),
#     DoctorVisits, HealthInsurance, ISLR2's Wage, Default, Bikeshare and Hitters, mlbench's Pima diabetes
#   Inference:
#     procova  a trial adjusted for a prognostic score learned from historical data: is the adjusted
#              estimate within 5% of the standard error the true prognostic function would give?
#     dml      the average treatment effect from observational data by cross-fitted AIPW with the learner
#              for both nuisance functions: does the 95% interval cover the true effect?
#
# Criteria: "acc" (accuracy: for simulated data, signal R^2 = 1 - mean((prediction - E[y|x])^2) / var(E[y|x])
# of at least 0.8, or for yes/no outcomes an AUC whose excess over 0.5 is at least 90% of the oracle's; for
# real data, within 0.01 of the best learner's R^2, AUC or explained Poisson deviance, 0.02 for spending),
# "cal" (calibration: for yes/no outcomes a calibration slope in [0.8, 1.25] and a mean test prediction within
# 15% of the training share; for counts and spending a mean test prediction within 10% of the training mean),
# "imp", "eff" and "cover" as above.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(AER); library(ISLR2); library(mlbench) })

# ---- simulated predictors: a "last year's outcome" z, correlated normals and binaries -----------------------
sim_x <- function(n, p = 10, rho = 0.3, shift = 0) {
  k <- p - 1; nb <- floor(k / 3); nn <- k - nb
  S <- matrix(rho, nn, nn); diag(S) <- 1
  xn <- matrix(rnorm(n * nn), n) %*% chol(S); xn[, 1] <- xn[, 1] + shift
  xb <- matrix(rbinom(n * nb, 1, 0.3), n)
  z <- 0.6 * xn[, 2] + 0.5 * xb[, 1] + rnorm(n, 0, 0.8)           # "last year's outcome"
  x <- cbind(z = z, xn, xb); colnames(x) <- c("z", paste0("x", 1:nn), paste0("b", 1:nb)); x
}
f_lin <- function(x) 1.2 * x[, "z"] + 0.5 * x[, "x1"] - 0.4 * x[, "x3"] + 0.3 * x[, "b1"] + 0.3 * x[, "x4"]
f_nonlin <- function(x, kind) switch(kind,
  friedman = { u <- pnorm(x[, c("x1", "x2", "x3", "x4", "x5")]); (10 * sin(pi * u[, 1] * u[, 2]) + 20 * (u[, 3] - 0.5)^2 + 10 * u[, 4] + 5 * u[, 5]) / 4 },
  threshold = x[, "z"] + 1.5 * (x[, "x1"] > 0) * x[, "x2"] + 1.2 * (x[, "x3"] > 0.8) - 0.8 * x[, "b1"] * x[, "b2"],
  smooth = x[, "z"] + 1.2 * sin(2 * x[, "x1"]) + 0.6 * x[, "x2"]^2 - 0.6)
noise_for <- function(f, r2) sqrt(var(f) * (1 - r2) / r2)

# one simulated instance: returns training and test data and the oracle on the test data
sim_instance <- function(cond, n_test = 5000) {
  n <- cond$n; ntot <- n + n_test; type <- cond$type
  p <- if (is.null(cond[["p"]])) 10 else cond[["p"]]        # [[ ]]: $ would match "prev" partially
  x <- sim_x(ntot, p, rho = if (is.null(cond[["rho"]])) 0.3 else cond[["rho"]])
  if (cond$family == "shift") x[n + seq_len(n_test), "x1"] <- x[n + seq_len(n_test), "x1"] + 1.5
  sig <- switch(cond$family, nonlin = f_nonlin(x, cond$kind),
    shift = if (cond$kind == "linear") f_lin(x) else x[, "z"] + 0.5 * x[, "x1"]^2,
    sparse = x[, "z"] + 0.8 * x[, "x1"] - 0.6 * x[, "x2"] + 0.5 * x[, "x3"] + 0.4 * x[, "b1"],
    binary = , count = if (cond$kind == "linear") f_lin(x) else f_nonlin(x, "threshold"),
    import = 1.0 * x[, "x1"] + 0.8 * x[, "x2"] + 0.6 * x[, "x3"],
    f_lin(x))
  if (cond$family == "missing") {   # x5 (independent of everything else) hidden more often when z is high
    x[, "x5"] <- rnorm(ntot); m <- rbinom(ntot, 1, plogis(-1.2 + 0.8 * x[, "z"]))
    sig <- f_lin(x) + 0.6 * x[, "x5"] + 0.7 * m
  }
  ok <- function(v) v[n + seq_len(n_test)]
  if (type == "gaussian") {
    s <- noise_for(sig[seq_len(n)], 0.5); y <- sig + rnorm(ntot, 0, s); truth <- sig
    if (cond$family == "missing") truth <- f_lin(x) + 0.6 * x[, "x5"] * (1 - m) + 0.7 * m   # E[x5] = 0 when hidden
  } else if (type == "binary") {
    mu0 <- mean(sig[seq_len(n)]); s0 <- sd(sig[seq_len(n)]); eta <- 1.2 * (sig - mu0) / s0
    a <- uniroot(function(a) mean(plogis(a + eta)) - cond$prev, c(-10, 5))$root
    pr <- plogis(a + eta); y <- rbinom(ntot, 1, pr); truth <- pr
    if (cond$family == "missing") {    # when x5 is hidden, average the probability over x5 ~ N(0, 1)
      nodes <- qnorm(ppoints(40)); base <- eta - 1.2 * 0.6 * x[, "x5"] / s0
      pm <- rowMeans(plogis(a + outer(base, 1.2 * 0.6 * nodes / s0, "+")))
      truth <- ifelse(m == 1, pm, pr)
    }
  } else if (type == "count") {
    mu <- exp(0.6 + 0.5 * as.numeric(scale(sig))); y <- rnbinom(ntot, size = cond$theta, mu = mu); truth <- mu
  } else if (type == "cost") {
    e <- as.numeric(scale(sig)); pz <- plogis(1.3 + 0.8 * e); s <- cond$sigma
    mpos <- exp(7.5 + 0.8 * e)                         # mean of the positive part (dollars)
    y <- rbinom(ntot, 1, pz) * mpos * exp(rnorm(ntot, -s^2 / 2, s)); truth <- pz * mpos
  }
  if (cond$family == "missing") { x[m == 1, "x5"] <- 0; x <- cbind(x, x5_miss = m) }
  list(xtr = x[seq_len(n), , drop = FALSE], ytr = y[seq_len(n)], xte = x[n + seq_len(n_test), , drop = FALSE],
       yte = ok(y), truth = ok(truth), type = type, signal = c("x1", "x2", "x3"))
}

# ---- plasmode: MEPS predictors, outcomes from a known model ---------------------------------------------------
meps_env <- new.env()
load_meps <- function() {
  if (!is.null(meps_env$d)) return(invisible())
  f <- Sys.getenv("MEPS_ML_CSV", "../meps_for_simulation/data/meps_ml.csv")   # from the MEPS walkthrough; not shared
  d <- read.csv(f, stringsAsFactors = FALSE)
  cat_vars <- c("race_eth", "region", "marital", "education", "poverty", "insurance")
  skewed <- c("cost_2022", "oop_2022", "ob_visits_2022", "er_visits_2022", "stays_2022", "rx_2022")
  for (v in skewed) d[[paste0("log_", v)]] <- log1p(d[[v]])
  ids <- c("person", "household", "weight", "stratum", "psu", "test", "fold")
  outs <- c("cost", "high_cost", "log_cost", "ob_visits", "any_er", "any_ip")
  pred <- c(setdiff(names(d), c(ids, outs, skewed)))
  for (v in cat_vars) d[[v]] <- factor(d[[v]])
  X <- model.matrix(~ . - 1, d[, pred], contrasts.arg = lapply(d[, cat_vars], contrasts, contrasts = FALSE))
  meps_env$d <- d; meps_env$X <- X
  # the plasmode truth: a lasso fit to log spending (linear part) plus a curve in age and an interaction.
  # Its seed is set here, so the caller's random-number state is saved and restored: without this the first
  # MEPS instance drawn in each R process got the split that follows seed 1, whatever its own seed (found
  # 12 November 2026; replicates of the MEPS-based conditions were duplicated)
  old_seed <- get0(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (is.null(old_seed)) rm(".Random.seed", envir = globalenv()) else assign(".Random.seed", old_seed, envir = globalenv()))
  set.seed(1); fl <- cv.glmnet(X, d$log_cost, alpha = 1, nfolds = 5)
  lin <- as.numeric(predict(fl, X, s = "lambda.1se"))
  a <- (d$age - 50) / 15; ph <- as.numeric(scale(d$pcs))
  meps_env$sig <- lin + 0.6 * sin(1.5 * a) - 0.5 * a * ph
}
plasmode_instance <- function(cond, n_test = 3000) {
  load_meps(); X <- meps_env$X; sig <- meps_env$sig; N <- nrow(X)
  i_tr <- sample(N, cond$n); i_te <- sample(setdiff(seq_len(N), i_tr), min(n_test, N - cond$n))
  e <- as.numeric(scale(sig))
  if (cond$type == "gaussian") { y <- sig + rnorm(N, 0, noise_for(sig, 0.5)); truth <- sig }
  if (cond$type == "binary") { a <- uniroot(function(a) mean(plogis(a + 1.5 * e)) - 0.10, c(-10, 5))$root
    truth <- plogis(a + 1.5 * e); y <- rbinom(N, 1, truth) }
  if (cond$type == "count") { truth <- exp(1.2 + 0.7 * e); y <- rnbinom(N, size = 1.2, mu = truth) }
  list(xtr = X[i_tr, ], ytr = y[i_tr], xte = X[i_te, ], yte = y[i_te], truth = truth[i_te], type = cond$type)
}

# ---- real data sets ------------------------------------------------------------------------------------------
mm <- function(d) { d <- d[, sapply(d, function(z) !all(is.na(z))), drop = FALSE]
  for (v in names(d)) if (is.numeric(d[[v]]) && anyNA(d[[v]])) { d[[paste0(v, "_miss")]] <- as.integer(is.na(d[[v]]))
    d[[v]][is.na(d[[v]])] <- median(d[[v]], na.rm = TRUE) }
  for (v in names(d)) if (is.character(d[[v]])) d[[v]] <- factor(d[[v]])
  model.matrix(~ . - 1, d) }
real_data <- function(name, outcome) {
  switch(name,
    meps = { load_meps(); d <- meps_env$d
      y <- switch(outcome, log_cost = d$log_cost, cost = d$cost, high_cost = d$high_cost, any_er = d$any_er, ob_visits = d$ob_visits)
      list(X = meps_env$X, y = y, group = d$household) },
    nmes = { data("NMES1988", package = "AER", envir = environment()); d <- NMES1988
      y <- if (outcome == "visits") d$visits else as.integer(d$hospital > 0)
      list(X = mm(d[, setdiff(names(d), c("visits", "nvisits", "ovisits", "novisits", "emergency", "hospital"))]), y = y) },
    doctor = { data("DoctorVisits", package = "AER", envir = environment()); d <- DoctorVisits
      list(X = mm(d[, setdiff(names(d), "visits")]), y = d$visits) },
    insurance = { data("HealthInsurance", package = "AER", envir = environment()); d <- HealthInsurance
      list(X = mm(d[, setdiff(names(d), "insurance")]), y = as.integer(d$insurance == "yes")) },
    wage = { d <- ISLR2::Wage; list(X = mm(d[, c("year", "age", "maritl", "race", "education", "jobclass", "health", "health_ins")]), y = d$logwage) },
    default = { d <- ISLR2::Default; list(X = mm(d[, c("student", "balance", "income")]), y = as.integer(d$default == "Yes")) },
    bike = { d <- ISLR2::Bikeshare; list(X = mm(d[, c("season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed")]),
                                          y = d$bikers) },
    hitters = { d <- na.omit(ISLR2::Hitters); list(X = mm(d[, setdiff(names(d), "Salary")]), y = log(d$Salary)) },
    pima = { data("PimaIndiansDiabetes2", package = "mlbench", envir = environment()); d <- PimaIndiansDiabetes2
      list(X = mm(d[, setdiff(names(d), "diabetes")]), y = as.integer(d$diabetes == "pos")) })
}
real_instance <- function(cond, n_test = 3000) {
  r <- real_data(cond$data, cond$outcome); N <- nrow(r$X)
  if (!is.null(r$group)) {              # keep households together
    g <- unique(r$group); gs <- sample(g); cs <- cumsum(table(factor(r$group, levels = gs)))
    gtr <- gs[seq_len(which(cs >= cond$n)[1])]; itr <- which(r$group %in% gtr); ite <- setdiff(seq_len(N), itr)
  } else { itr <- sample(N, cond$n); ite <- setdiff(seq_len(N), itr) }
  if (length(ite) > n_test) ite <- sample(ite, n_test)
  list(xtr = r$X[itr, ], ytr = r$y[itr], xte = r$X[ite, ], yte = r$y[ite], truth = NULL, type = cond$type)
}

# ---- the conditions ------------------------------------------------------------------------------------------
conditions <- function() {
  L <- list(); add <- function(...) L[[length(L) + 1]] <<- list(...)
  for (p in c(10, 40)) for (n in c(250, 1000, 3000)) add(family = "lin", source = "sim", type = "gaussian", p = p, n = n)
  for (k in c("friedman", "threshold", "smooth")) for (n in c(250, 1000, 3000)) add(family = "nonlin", source = "sim", type = "gaussian", kind = k, n = n)
  for (s in c(0.8, 1.4)) for (n in c(250, 1000, 3000)) add(family = "cost", source = "sim", type = "cost", sigma = s, n = n)
  for (pv in c(0.05, 0.15, 0.35)) for (k in c("linear", "interaction")) for (n in c(500, 2000))
    add(family = "binary", source = "sim", type = "binary", prev = pv, kind = k, n = n)
  for (th in c(1, 5)) for (k in c("linear", "interaction")) for (n in c(500, 2000)) add(family = "count", source = "sim", type = "count", theta = th, kind = k, n = n)
  add(family = "sparse", source = "sim", type = "gaussian", p = 100, n = 150)
  add(family = "sparse", source = "sim", type = "binary", p = 100, n = 150, prev = 0.3)
  for (k in c("linear", "quadratic")) for (n in c(500, 2000)) add(family = "shift", source = "sim", type = "gaussian", kind = k, n = n)
  for (t in c("gaussian", "binary")) for (n in c(500, 2000)) add(family = "missing", source = "sim", type = t, n = n, prev = 0.2)
  for (r in c(0, 0.6)) add(family = "import", source = "sim", type = "gaussian", rho = r, n = 1000)
  for (t in c("gaussian", "binary", "count")) for (n in c(500, 2000)) add(family = "plasmode", source = "plasmode", type = t, n = n)
  rd <- list(c("meps", "log_cost", "gaussian"), c("meps", "cost", "cost"), c("meps", "high_cost", "binary"),
             c("meps", "any_er", "binary"), c("meps", "ob_visits", "count"), c("nmes", "visits", "count"),
             c("nmes", "hospital", "binary"), c("doctor", "visits", "count"), c("insurance", "insurance", "binary"),
             c("wage", "logwage", "gaussian"), c("default", "default", "binary"), c("bike", "bikers", "count"))
  for (r in rd) for (n in c(500, 2000)) add(family = "real", source = "real", data = r[1], outcome = r[2], type = r[3], n = n)
  add(family = "real", source = "real", data = "hitters", outcome = "logsalary", type = "gaussian", n = 150)
  add(family = "real", source = "real", data = "pima", outcome = "diabetes", type = "binary", n = 300)
  for (k in c("linear", "nonlinear")) for (nh in c(300, 1500)) add(family = "procova", source = "sim", type = "gaussian", kind = k, n = nh)
  for (k in c("linear", "nonlin_outcome", "nonlin_both")) for (n in c(500, 2000)) add(family = "dml", source = "sim", type = "gaussian", kind = k, n = n)
  C <- do.call(rbind, lapply(seq_along(L), function(i) { z <- L[[i]]
    data.frame(cond = i, family = z$family, source = z$source, type = z$type, n = z$n,
               detail = paste(unlist(z[setdiff(names(z), c("family", "source", "type", "n"))]), collapse = "/"), stringsAsFactors = FALSE) }))
  attr(C, "list") <- L; C
}
