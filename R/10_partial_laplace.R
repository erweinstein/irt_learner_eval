# 10_partial_laplace.R -- A joint check of the Laplace approximation in the adaptive tests' calibration model
# (FLAGS.md item 25), with all the crossed random effects present, by a solver of my own: the learner-side random
# effects are integrated exactly by nested Gauss-Hermite quadrature, and only the task-side ones by the Laplace
# approximation ("partial Laplace").
#
# Model (as in R/06_bifactor_cat.R; binary responses, all 21 learners):
#   Level 1 (learner l, task-item i in condition c of task family f):
#     logit P(Y_li = 1) = x_li'beta + u_f + u_c + u_inst + u_i + a_lf + b_lc
#   Level 2: task side u_* ~ N(0, s_*^2) (crossed with learners); a_lf ~ N(0, s_lf^2); b_lc ~ N(0, s_lc^2), nested
# Given the task-side effects u, the likelihood factors over the 318 learner-by-family clusters, and within them
# over learner-by-condition clusters (a median of 6 responses each), so
#   I_lf(u) = sum_q w_q prod_{c in lf} sum_r w_r prod_{i in c} P(y_i | eta_i(u) + s_lf z_q + s_lc z_r)
# is computed by quadrature. The marginal log-likelihood is then approximated by Laplace over u alone:
#   h(u) = log p(u) + sum_lf log I_lf(u),  log L ~ h(u^) + (d/2) log(2 pi) - (1/2) log det(-h''(u^)),
# with u^ found by Newton's method. The gradient and Hessian of log I_lf with respect to the linear predictor are
# posterior moments over the quadrature grid: d/d eta_i = E[y_i - p_i], d2/(d eta_i d eta_j) =
# Cov(y_i - p_i, y_j - p_j) - 1(i = j) E[p_i (1 - p_i)], where the posterior over (a_lf, b_lc) given u is discrete
# on the grid (a shared within lf, b independent across lc given a). Each task-side effect is shared by all 21
# learners (an item has about 19 responses, a condition about 90), so Laplace over u is the safer part.
# glmmTMB applies Laplace to the learner-side effects as well. The two log-likelihoods are compared on a 5 x 5 grid
# of (log s_lf, log s_lc) around glmmTMB's estimates, with every other parameter held at its estimate; the
# difference between the maximizers of quadratic surfaces fitted to each is the bias the learner-side Laplace
# approximation adds to the two SDs (conditional on the other parameters). Checks: quadrature against adaptive
# numerical integration on single clusters; the analytic gradient against
# finite differences; Q against 1.5 Q points; and agreement with glmmTMB when the learner-side SDs are near zero (when both methods reduce
# to Laplace over u).
# Output: output/v21/partial_laplace_grid.csv, output/v21/partial_laplace_summary.csv
#
# Plain importance sampling (around glmmTMB's joint mode, or over the task side only with the learner side by
# quadrature) was tried first and failed: the weights collapsed onto one draw in 40 (881-3,095 dimensions).
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(Matrix); library(statmod) })
set.seed(1010)
OUT <- Sys.getenv("IRT_OUT", "output/v21"); Q <- as.integer(Sys.getenv("PL_Q", "30"))
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_21.csv"), stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":"); R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$lf <- paste(R$learner, R$task_family); R$lc <- paste(R$learner, R$cond)
fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) + (1 | lf) + (1 | lc), R, binomial)
obj <- fit$obj; env <- obj$env; rt <- fit$modelInfo$reTrms$cond
nlev <- sapply(rt$flist, nlevels); stopifnot(names(nlev) == c("task_family", "cond", "inst", "item", "lf", "lc"))
blk <- rep(names(nlev), nlev); iu <- which(blk %in% c("task_family", "cond", "inst", "item"))
pb <- env$last.par.best; par0 <- pb[-env$random]; th0 <- par0[names(par0) == "theta"]; names(th0) <- names(nlev)
y <- env$data$yobs; n <- length(y); Zu <- as(env$data$Z, "CsparseMatrix")[, iu]; d <- length(iu)
eta_fix <- as.vector(env$data$X %*% pb[names(pb) == "beta"]); sd_u <- exp(th0[blk[iu]])
lc_code <- as.integer(rt$flist$lc); lf_code <- as.integer(rt$flist$lf)
lf_of_lc <- lf_code[match(seq_len(nlev[["lc"]]), lc_code)]
stopifnot(all(tapply(lf_code, lc_code, function(v) length(unique(v))) == 1))
obs_by_lf <- split(seq_len(n), lf_code)

# ---- the learner-side integrals and their derivatives ---------------------------------------------------------------
lse_rows <- function(M) { m <- apply(M, 1, max); m + log(rowSums(exp(M - m))) }
# returns log I (sum over clusters), gradient and Hessian (sparse, block diagonal by lf) w.r.t. eta
learner_side <- function(eta, sf, sc, Q, derivs = TRUE) {
  gh <- gauss.quad.prob(Q, "normal"); lw <- log(gh$weights)
  s <- as.vector(outer(sc * gh$nodes, sf * gh$nodes, "+"))             # column (q - 1) * Q + r
  E <- outer(eta, s, "+"); P <- plogis(E)
  LL <- y * E - (pmax(E, 0) + log1p(exp(-abs(E))))
  S <- rowsum(LL, lc_code, reorder = TRUE)                             # lc x Q^2
  A <- matrix(0, nrow(S), Q); PR <- vector("list", Q)                  # A[c, q] = log sum_r w_r exp(S[c, q, r])
  for (q in seq_len(Q)) { Sq <- sweep(S[, (q - 1) * Q + seq_len(Q), drop = FALSE], 2, lw, "+")
    A[, q] <- lse_rows(Sq); PR[[q]] <- exp(Sq - A[, q]) }               # PR[[q]][c, r] = pi(r | q, c)
  Tq <- sweep(rowsum(A, lf_of_lc, reorder = TRUE), 2, lw, "+")          # lf x Q
  logI <- lse_rows(Tq); PQ <- exp(Tq - logI)                           # PQ[f, q] = pi(q | f)
  if (!derivs) return(list(logI = sum(logI)))
  Sres <- y - P                                                        # n x Q^2 scores
  Wr <- do.call(cbind, lapply(seq_len(Q), function(q) PR[[q]][lc_code, , drop = FALSE]))   # n x Q^2: pi(r | q, c(i))
  M <- sapply(seq_len(Q), function(q) rowSums(Wr[, (q - 1) * Q + seq_len(Q)] * Sres[, (q - 1) * Q + seq_len(Q)]))  # n x Q
  pq <- PQ[lf_code, , drop = FALSE]                                    # n x Q: pi(q | f(i))
  grad <- rowSums(pq * M)
  W <- Wr * pq[, rep(seq_len(Q), each = Q)]                            # n x Q^2: joint posterior weight
  dpp <- rowSums(W * P * (1 - P))
  blocks <- lapply(obs_by_lf, function(ii) {
    Mf <- M[ii, , drop = FALSE]; Hb <- tcrossprod(Mf * sqrt(pq[ii, , drop = FALSE]))   # E[m_i m_j] across lc
    cc <- lc_code[ii]
    for (cl in unique(cc)) { jj <- which(cc == cl)                      # same lc: E[s_i s_j] on the full grid
      Hb[jj, jj] <- tcrossprod(Sres[ii[jj], , drop = FALSE] * sqrt(W[ii[jj], , drop = FALSE])) }
    Hb - tcrossprod(grad[ii]) - diag(dpp[ii], length(ii)) })
  Heta <- bdiag(blocks); ord <- unlist(obs_by_lf)
  Heta <- Heta[order(ord), order(ord)]                                 # back to observation order
  list(logI = sum(logI), grad = grad, Heta = Heta)
}

# h(u), its Newton maximization and the partial-Laplace log-likelihood
h_val <- function(u, sf, sc, Q) learner_side(eta_fix + as.vector(Zu %*% u), sf, sc, Q, FALSE)$logI + sum(dnorm(u, 0, sd_u, log = TRUE))
partial_laplace <- function(sf, sc, Q, u_start, tol = 1e-8, maxit = 50) {
  u <- u_start
  for (it in seq_len(maxit)) {
    ls <- learner_side(eta_fix + as.vector(Zu %*% u), sf, sc, Q)
    g <- as.vector(crossprod(Zu, ls$grad)) - u / sd_u^2
    Hh <- as.matrix(crossprod(Zu, ls$Heta %*% Zu)) - diag(1 / sd_u^2)
    step <- -solve(Hh, g); hu <- ls$logI + sum(dnorm(u, 0, sd_u, log = TRUE)); t <- 1
    repeat { un <- u + t * step; if (h_val(un, sf, sc, Q) >= hu - 1e-10 || t < 1e-4) break; t <- t / 2 }
    u <- un; if (max(abs(t * step)) < tol) break
  }
  ls <- learner_side(eta_fix + as.vector(Zu %*% u), sf, sc, Q)
  Hh <- as.matrix(crossprod(Zu, ls$Heta %*% Zu)) - diag(1 / sd_u^2)
  ld <- determinant(-Hh, logarithm = TRUE)$modulus
  list(loglik = ls$logI + sum(dnorm(u, 0, sd_u, log = TRUE)) + d / 2 * log(2 * pi) - as.numeric(ld) / 2, u = u, iter = it)
}

sf0 <- exp(th0[["lf"]]); sc0 <- exp(th0[["lc"]]); u0 <- pb[env$random][iu]

# ---- checks -----------------------------------------------------------------------------------------------------------
# (1) analytic gradient of sum log I against finite differences in eta (random direction)
eta_m <- eta_fix + as.vector(Zu %*% u0); ls <- learner_side(eta_m, sf0, sc0, Q); v <- rnorm(n); e <- 1e-5
fd <- (learner_side(eta_m + e * v, sf0, sc0, Q, FALSE)$logI - learner_side(eta_m - e * v, sf0, sc0, Q, FALSE)$logI) / (2 * e)
fd2 <- (learner_side(eta_m + e * v, sf0, sc0, Q)$grad - learner_side(eta_m - e * v, sf0, sc0, Q)$grad) / (2 * e)
chk_grad <- c(analytic = sum(ls$grad * v), finite_diff = fd)
chk_hess <- c(max_abs_diff = max(abs(as.vector(ls$Heta %*% v) - fd2)), scale = max(abs(fd2)))
print(chk_grad, digits = 10); print(chk_hess)
# (2) quadrature against adaptive integration on single learner-by-family clusters
check_cluster <- function(f, Qn) {
  rows <- which(lf_code == f); cls <- split(rows, lc_code[rows])
  inner <- function(a) sapply(a, function(aa) prod(sapply(cls, function(ii)
    integrate(function(b) sapply(b, function(bb) prod(plogis((2 * y[ii] - 1) * (eta_m[ii] + aa + bb)))) * dnorm(b, 0, sc0),
              -Inf, Inf, rel.tol = 1e-10)$value)))
  exact <- log(integrate(function(a) inner(a) * dnorm(a, 0, sf0), -Inf, Inf, rel.tol = 1e-10)$value)
  gh <- gauss.quad.prob(Qn, "normal")
  quad <- log(sum(gh$weights * sapply(sf0 * gh$nodes, function(aa) prod(sapply(cls, function(ii)
      sum(gh$weights * sapply(sc0 * gh$nodes, function(bb) prod(plogis((2 * y[ii] - 1) * (eta_m[ii] + aa + bb))))))))))
  c(n = length(rows), err = quad - exact) }
big <- order(-tabulate(lf_code))[1:4]; CC <- t(sapply(c(big, sample(nlev[["lf"]], 6)), check_cluster, Qn = Q))
print(CC)
# (3) near-zero learner-side SDs: partial Laplace and glmmTMB's Laplace should agree
p_small <- par0; p_small[names(p_small) == "theta"][5:6] <- log(c(0.02, 0.02))
small <- c(glmmTMB = -obj$fn(p_small), partial = partial_laplace(0.02, 0.02, Q, u0)$loglik)
print(small, digits = 10)

# ---- the grid --------------------------------------------------------------------------------------------------------
grid <- expand.grid(df = c(-0.2, -0.1, 0, 0.1, 0.2), dc = c(-0.2, -0.1, 0, 0.1, 0.2))
res <- list(); ustart <- u0
for (g in seq_len(nrow(grid))) {
  sf <- sf0 * exp(grid$df[g]); sc <- sc0 * exp(grid$dc[g])
  p <- par0; p[names(p) == "theta"][5:6] <- log(c(sf, sc)); lap <- -obj$fn(p)
  pl <- partial_laplace(sf, sc, Q, u0)
  res[[g]] <- data.frame(grid[g, ], sd_lf = sf, sd_lc = sc, loglik_laplace = lap, loglik_partial = pl$loglik, newton_iter = pl$iter)
  message(g, ": Laplace ", round(lap, 3), "  partial Laplace ", round(pl$loglik, 3), "  (", pl$iter, " Newton steps)")
}
G <- do.call(rbind, res); write.csv(G, file.path(OUT, "partial_laplace_grid.csv"), row.names = FALSE)
argmax <- function(v) { f <- lm(v ~ df + dc + I(df^2) + I(dc^2) + I(df * dc), G); b <- coef(f)
  Hm <- matrix(c(2 * b[4], b[6], b[6], 2 * b[5]), 2); setNames(-solve(Hm, b[2:3]), c("lf", "lc")) }
a_lap <- argmax(G$loglik_laplace); a_pl <- argmax(G$loglik_partial)
# (4) quadrature points at the estimates: Q against 1.5 Q
Q2 <- round(1.5 * Q); q_hi <- partial_laplace(sf0, sc0, Q2, u0)$loglik; q_lo <- G$loglik_partial[G$df == 0 & G$dc == 0]
S <- data.frame(Q = Q, loglik_laplace = -obj$fn(par0), loglik_partial = q_lo, difference = q_lo - (-obj$fn(par0)),
                Q_check = Q2, loglik_partial_Q_check = q_hi,
                sd_lf_glmmTMB = sf0, sd_lf_laplace_grid = sf0 * exp(a_lap[["lf"]]), sd_lf_partial = sf0 * exp(a_pl[["lf"]]),
                sd_lc_glmmTMB = sc0, sd_lc_laplace_grid = sc0 * exp(a_lap[["lc"]]), sd_lc_partial = sc0 * exp(a_pl[["lc"]]),
                ratio_lf = exp(a_pl[["lf"]] - a_lap[["lf"]]), ratio_lc = exp(a_pl[["lc"]] - a_lap[["lc"]]),
                grad_check_rel = abs(diff(chk_grad)) / abs(chk_grad[[2]]), hess_check = chk_hess[[1]] / chk_hess[[2]],
                cluster_check_max_err = max(abs(CC[, "err"])), small_sd_diff = diff(small))
write.csv(S, file.path(OUT, "partial_laplace_summary.csv"), row.names = FALSE)
print(G, digits = 7); print(t(S), digits = 6)
