# 00_learners.R -- The learners ("examinees") and the scoring helpers for the IRT evaluation of machine
# learning pipelines. Every learner has the same interface:
#   fit_predict(name, type, xtr, ytr, xte) -> predictions for xte
# where type is "gaussian" (a continuous outcome), "binary" (0/1; predictions are probabilities), "count"
# (a non-negative integer outcome; predictions are means) or "cost" (non-negative, with a lump at zero and a
# long right tail, like health care spending; predictions are means), xtr and xte are numeric matrices with
# the same columns (dummies already made), and ytr is numeric. A learner that does not apply to a type
# returns NULL (the response is then missing by design).
#
# The 14 learners follow the MEPS machine-learning walkthrough (chapters 1-8 and 13):
#   glm          OLS / logistic / Poisson regression; for "cost", OLS on dollars (squared error)
#   glm_matched  the loss-matched GLM: quasi-Poisson (log link) for cost, negative binomial for counts
#   lasso, ridge, enet   glmnet, penalty by 5-fold CV (gaussian for gaussian and cost, binomial, poisson)
#   gam          mgcv, penalized regression splines for up to 8 continuous columns (REML); Tweedie for cost
#   knn          k nearest neighbours on standardized columns, k (5, 15, 40) by 5-fold CV
#   cart         rpart, pruned by cross-validated error (a regression tree on 0/1 for binary; Poisson for counts)
#   rf           ranger, 300 trees (a probability forest for binary)
#   xgb_default  xgboost with its defaults (100 rounds, depth 6, eta 0.3), squared error unless binary
#   xgb_tuned    xgboost, eta 0.05, depth 3, subsample 0.8, rounds by 5-fold early stopping, loss matched
#                (Tweedie for cost, Poisson for counts)
#   svm          e1071, radial kernel, defaults (Platt probabilities for binary); at most 1,500 training rows
#   nnet         one hidden layer of 5 units, decay 0.1, standardized inputs
#   stack        non-negative weights summing to one on 5-fold out-of-fold predictions of glm, lasso, knn, rf
#                and xgb_tuned (squared error; on probabilities for binary)
# Five added on 11 November 2026, from recent benchmarks of learners for tabular data (TabArena, Erickson et
# al. 2025; TabPFN, Hollmann et al. 2025) and from causal inference (BART; Hill 2011, Dorie et al. 2019):
#   lightgbm     LightGBM with its defaults (100 rounds, 31 leaves, learning rate 0.1), loss matched to the
#                outcome (binary, Poisson, Tweedie 1.5, squared error)
#   catboost     CatBoost with its defaults (1,000 rounds), loss matched (py/learners_py.py)
#   bart         dbarts::bart2, 200 trees, 2 chains, defaults otherwise (probit BART for binary); the posterior
#                mean of E[y | x]
#   realmlp      RealMLP with its tuned defaults (pytabkit), a modern multilayer perceptron (py/learners_py.py)
#   tabpfn       TabPFN v2, a pretrained transformer that predicts by in-context learning (py/learners_py.py)
# The Python learners run through reticulate (catboost 1.2.10, pytabkit 1.8.0, tabpfn 2.0.9, torch 2.15 CPU).
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({
  library(glmnet); library(mgcv); library(FNN); library(rpart); library(ranger); library(xgboost)
  library(e1071); library(nnet); library(MASS); library(quadprog)
})

LEARNERS <- c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default",
              "xgb_tuned", "svm", "nnet", "stack", "lightgbm", "catboost", "bart", "realmlp", "tabpfn")
LEARNERS_ORIGINAL <- LEARNERS[1:14]
PY_LEARNERS <- c("catboost", "realmlp", "tabpfn")
.py <- NULL
py_fit <- function(name, type, xtr, ytr, xte) {
  if (is.null(.py)) { Sys.setenv(RETICULATE_PYTHON = Sys.getenv("RETICULATE_PYTHON", "/usr/local/bin/python3"))
    .py <<- reticulate::import_from_path("learners_py", path = normalizePath("py")) }
  as.numeric(.py$fit_predict(name, type, xtr, ytr, xte, seed = sample.int(1e6, 1)))
}
STACK_BASE <- c("glm", "lasso", "knn", "rf", "xgb_tuned")

scaler <- function(xtr) { mu <- colMeans(xtr); s <- apply(xtr, 2, sd); s[!is.finite(s) | s == 0] <- 1
  function(x) sweep(sweep(x, 2, mu, "-"), 2, s, "/") }
clip01 <- function(p) pmin(pmax(p, 1e-4), 1 - 1e-4)
folds_of <- function(n, k = 5) sample(rep_len(seq_len(k), n))
drop_const <- function(xtr, xte) { keep <- apply(xtr, 2, function(z) length(unique(z)) > 1)
  list(xtr = xtr[, keep, drop = FALSE], xte = xte[, keep, drop = FALSE]) }

fit_predict <- function(name, type, xtr, ytr, xte) {
  dc <- drop_const(xtr, xte); xtr <- dc$xtr; xte <- dc$xte
  colnames(xtr) <- colnames(xte) <- paste0("v", seq_len(ncol(xtr)))
  out <- tryCatch(switch(name,
    glm = {
      d <- data.frame(y = ytr, xtr); nd <- data.frame(xte)
      if (type == "binary") predict(suppressWarnings(glm(y ~ ., binomial, d)), nd, type = "response")
      else if (type == "count") predict(suppressWarnings(glm(y ~ ., poisson, d)), nd, type = "response")
      else predict(lm(y ~ ., d), nd)
    },
    glm_matched = {
      d <- data.frame(y = ytr, xtr); nd <- data.frame(xte)
      if (type == "cost") predict(suppressWarnings(glm(y ~ ., quasipoisson, d, control = list(maxit = 100))), nd, type = "response")
      else if (type == "count") predict(suppressWarnings(glm.nb(y ~ ., d)), nd, type = "response")
      else NULL
    },
    lasso = , ridge = , enet = {
      a <- c(lasso = 1, ridge = 0, enet = 0.5)[[name]]
      fam <- switch(type, binary = "binomial", count = "poisson", "gaussian")
      f <- cv.glmnet(xtr, ytr, family = fam, alpha = a, nfolds = 5)
      as.numeric(predict(f, xte, s = "lambda.min", type = "response"))
    },
    gam = {
      nu <- apply(xtr, 2, function(z) length(unique(z)))
      sm <- names(sort(nu[nu > 10], decreasing = TRUE))[seq_len(min(8, sum(nu > 10)))]
      lin <- setdiff(colnames(xtr), sm)
      fo <- as.formula(paste("y ~", paste(c(if (length(sm)) paste0("s(", sm, ", k = 6, bs = 'cr')"), lin, "1"), collapse = " + ")))
      fam <- switch(type, binary = binomial(), count = poisson(), cost = tw(), gaussian())
      g <- gam(fo, family = fam, data = data.frame(y = ytr, xtr), method = "REML")
      as.numeric(predict(g, data.frame(xte), type = "response"))
    },
    knn = {
      sc <- scaler(xtr); a <- sc(xtr); b <- sc(xte); ks <- c(5, 15, 40); fo <- folds_of(nrow(a))
      cvk <- sapply(ks, function(k) mean(unlist(lapply(1:5, function(f) {
        p <- knn.reg(a[fo != f, , drop = FALSE], a[fo == f, , drop = FALSE], ytr[fo != f], k = k)$pred
        (ytr[fo == f] - p)^2 }))))
      knn.reg(a, b, ytr, k = ks[which.min(cvk)])$pred
    },
    cart = {
      d <- data.frame(y = ytr, xtr)
      t0 <- rpart(y ~ ., d, method = if (type == "count") "poisson" else "anova", control = rpart.control(cp = 0.001, xval = 5))
      cp <- t0$cptable[which.min(t0$cptable[, "xerror"]), "CP"]
      p <- predict(prune(t0, cp = cp), data.frame(xte)); if (is.matrix(p)) p[, 1] else p
    },
    rf = {
      d <- data.frame(y = if (type == "binary") factor(ytr) else ytr, xtr)
      f <- ranger(y ~ ., d, num.trees = 300, probability = type == "binary", min.node.size = if (type == "binary") 10 else 5,
                  num.threads = 1)
      p <- predict(f, data.frame(xte), num.threads = 1)$predictions; if (is.matrix(p)) p[, "1"] else p
    },
    xgb_default = {
      obj <- if (type == "binary") "binary:logistic" else "reg:squarederror"
      f <- xgb.train(list(objective = obj, nthread = 1), xgb.DMatrix(xtr, label = ytr), nrounds = 100, verbose = 0)
      predict(f, xte)
    },
    xgb_tuned = {
      obj <- switch(type, binary = "binary:logistic", count = "count:poisson", cost = "reg:tweedie", "reg:squarederror")
      pr <- list(objective = obj, eta = 0.05, max_depth = 3, subsample = 0.8, colsample_bytree = 0.8, nthread = 1)
      if (type == "cost") pr$tweedie_variance_power <- 1.5
      dm <- xgb.DMatrix(xtr, label = ytr)
      cv <- xgb.cv(pr, dm, nrounds = 1000, nfold = 5, early_stopping_rounds = 30, verbose = 0)
      bi <- if (!is.null(cv$early_stop$best_iteration)) cv$early_stop$best_iteration else cv$best_iteration
      f <- xgb.train(pr, dm, nrounds = max(10, bi), verbose = 0)
      predict(f, xte)
    },
    svm = {
      sc <- scaler(xtr); a <- sc(xtr); b <- sc(xte)
      if (nrow(a) > 1500) { i <- sample(nrow(a), 1500); a <- a[i, , drop = FALSE]; yy <- ytr[i] } else yy <- ytr
      if (type == "binary") {
        f <- svm(a, factor(yy), probability = TRUE)
        attr(predict(f, b, probability = TRUE), "probabilities")[, "1"]
      } else as.numeric(predict(svm(a, yy), b))
    },
    nnet = {
      sc <- scaler(xtr); a <- sc(xtr); b <- sc(xte)
      if (type == "binary") as.numeric(predict(nnet(a, ytr, size = 5, decay = 0.1, maxit = 300, entropy = TRUE, trace = FALSE, MaxNWts = 5000), b))
      else { m <- mean(ytr); s <- sd(ytr); if (!is.finite(s) || s == 0) s <- 1
        m + s * as.numeric(predict(nnet(a, (ytr - m) / s, size = 5, decay = 0.1, maxit = 300, linout = TRUE, trace = FALSE, MaxNWts = 5000), b)) }
    },
    lightgbm = {
      obj <- switch(type, binary = "binary", count = "poisson", cost = "tweedie", "regression")
      f <- lightgbm::lgb.train(list(objective = obj, num_threads = 1, verbosity = -1, seed = sample.int(1e6, 1)),
                               lightgbm::lgb.Dataset(xtr, label = ytr), nrounds = 100)
      predict(f, xte)
    },
    catboost = , realmlp = , tabpfn = py_fit(name, type, xtr, ytr, xte),
    bart = {
      if (type == "binary") {
        f <- dbarts::bart2(xtr, ytr, xte, n.chains = 2, n.threads = 1, verbose = FALSE, keepTrees = FALSE)
        colMeans(pnorm(rbind(f$yhat.test[1, , ], f$yhat.test[2, , ])))
      } else {
        f <- dbarts::bart2(xtr, ytr, xte, n.chains = 2, n.threads = 1, verbose = FALSE, keepTrees = FALSE)
        colMeans(rbind(f$yhat.test[1, , ], f$yhat.test[2, , ]))
      }
    },
    stack = {
      fo <- folds_of(nrow(xtr)); Z <- matrix(NA_real_, nrow(xtr), length(STACK_BASE))
      for (f in 1:5) for (j in seq_along(STACK_BASE))
        Z[fo == f, j] <- fit_predict(STACK_BASE[j], type, xtr[fo != f, , drop = FALSE], ytr[fo != f], xtr[fo == f, , drop = FALSE])
      ok <- colSums(is.na(Z)) == 0; Z <- Z[, ok, drop = FALSE]
      w <- nnls_simplex(Z, ytr)
      P <- sapply(STACK_BASE[ok], function(l) fit_predict(l, type, xtr, ytr, xte))
      as.numeric(P %*% w)
    }), error = function(e) NA)
  if (is.null(out)) return(NULL)
  out <- as.numeric(out)
  if (length(out) != nrow(xte) || all(is.na(out))) return(rep(NA_real_, nrow(xte)))
  if (type == "binary") out <- clip01(out)
  if (type %in% c("count", "cost")) out <- pmax(out, 0)
  out
}

# non-negative weights summing to one minimizing squared error (quadprog)
nnls_simplex <- function(Z, y) {
  sc <- sd(y); if (!is.finite(sc) || sc == 0) sc <- 1; Z <- Z / sc; y <- y / sc   # dollars make the QP ill-conditioned
  k <- ncol(Z); D <- crossprod(Z) + diag(1e-8 * mean(diag(crossprod(Z))), k)
  sol <- solve.QP(D, crossprod(Z, y), cbind(rep(1, k), diag(k)), c(1, rep(0, k)), meq = 1)$solution
  pmax(sol, 0) / sum(pmax(sol, 0))
}

# ---- metrics ------------------------------------------------------------------------------------------------
r2 <- function(y, p) 1 - mean((y - p)^2) / mean((y - mean(y))^2)
auc <- function(y, p) { r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0); (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
pois_dev <- function(y, p) { p <- pmax(p, 1e-8); 2 * mean(ifelse(y > 0, y * log(y / p), 0) - (y - p)) }
# "explained deviance" for counts, relative to the mean
d2 <- function(y, p) 1 - pois_dev(y, p) / pois_dev(y, rep(mean(y), length(y)))
cal_slope <- function(y, p) { z <- qlogis(clip01(p)); unname(coef(suppressWarnings(glm(y ~ z, binomial)))[2]) }
