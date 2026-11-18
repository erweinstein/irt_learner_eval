# 06e_online_calibration_cat.R -- Adaptive testing of a new learner as a new item (online calibration).
# (Added 18 November 2026; FLAGS.md item 29. Adds to 06_bifactor_cat.R and 06d_cat_stopping_rule.R.)
#
# 06d found that adaptive tests scored with a common slope predict a learner's proportion of criteria met over
# its bank with errors that follow the learner's own flipped-2PL slope, so their 95% intervals cover in only
# 81-84% of tests. Here the new learner is treated as a new item in the flipped orientation: the task-items are
# respondents with known easiness, and the learner's own intercept and slope are estimated from its answers, as
# in online calibration of new items in adaptive testing (Stocking 1988; Ban et al. 2001).
#
# Model for the left-out learner l, given the calibration of the other learners:
#   Level 1 (task-item i in condition c of family f): logit P(Y_li = 1) = d_l + a_l eta_i + u_lf + w_lc
#   Level 2 (learner): (d_l, a_l) ~ N(m0, S0); u_lf ~ N(0, s_f^2); w_lc ~ N(0, s_c^2)
#   Combined: logit P(Y_li = 1) = d_l + a_l eta_i + u_lf + w_lc,  with eta_i known
# eta_i is the task-item's easiness from the multilevel Rasch calibration on the other learners (the sum of its
# family, condition, instance and item effects and its criterion's fixed effect, as in 06), so a_l = 1 for every
# learner gives back the Rasch testlet model of 06/06d. m0 and S0 come from the calibration learners: each one's
# (d, a) from a logistic regression of its own answers on eta (an empirical prior; S0 is their sample covariance,
# doubled to allow for the noise in eta). s_f and s_c are the calibration's learner-by-family and
# learner-by-condition SDs.
# Variants: "2pl" (s_f = s_c = 0: no testlet effects) and "2pl_testlet".
# Posterior: a 41 x 25 grid over (d, a) (+-4 prior SDs), with the testlet effects integrated by nested
# Gauss-Hermite quadrature (7 points each) as in 06.
# Target: the learner's proportion of criteria met over its bank. A(d, a) = mean over the bank of
# P(Y = 1 | d, a) with the testlet effects at their prior; selection takes the item that minimizes the expected
# posterior variance of A (exposure 0.8/0.2 as in 06), after one item from each family (content balancing); the
# candidates for that calculation are the 20 items nearest a predicted 0.5, the 10 easiest, the 10 hardest and 20
# at random.
# Rules: "online" (that selection), "mixed" (every second item at random, the others as "online": a test built
# on the current estimate learns little about a slope the prior does not expect, since items far from the
# learner's 50% point then look uninformative), and "random", each under the same stopping rule: posterior SD of A <=
# CAT_STOP_SDA (default 0.04 and 0.025), cap CAT_CAP (150); the state at k = 25, 50 and 100 is also recorded.
# At each stop: a 95% predictive interval for the full-bank proportion met (answers given plus simulated answers
# to the rest, from the posterior of (d, a, u, w)), its coverage of the observed proportion, its width and
# signed error; and the posterior mean of the slope a against the learner's full-bank estimate.
#
# Input: IRT_RESPONSES (default output/responses_21.csv). Output (IRT_OUT, default output/v21):
#   online_cat_parts/<learner>.csv, online_cat_results.csv, online_cat_summary.csv.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(statmod); library(parallel); library(MASS) })
RNGkind("L'Ecuyer-CMRG"); set.seed(20261119)
OUT <- Sys.getenv("IRT_OUT", "output/v21"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_21.csv"), stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":")
R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$lf <- paste(R$learner, R$task_family); R$lc <- paste(R$learner, R$cond)
LEARNERS <- c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default",
              "xgb_tuned", "svm", "nnet", "stack", "lightgbm", "catboost", "bart", "realmlp", "tabpfn",
              "lightgbm_tuned", "catboost_tuned")
LEARNERS <- LEARNERS[LEARNERS %in% unique(R$learner)]
TARGETS <- sort(as.numeric(strsplit(Sys.getenv("CAT_STOP_SDA", "0.04,0.025"), ",")[[1]]), decreasing = TRUE)
KFIX <- c(25, 50, 100)
CAP <- as.integer(Sys.getenv("CAT_CAP", "150"))
RUNS <- as.integer(Sys.getenv("CAT_RUNS", "3")); RAND <- as.integer(Sys.getenv("CAT_RAND", "10"))
NSIM <- as.integer(Sys.getenv("CAT_NSIM", "400"))
gh <- gauss.quad.prob(7, "normal"); Q <- length(gh$nodes); lq <- log(gh$weights)

lse_last <- function(x) { d <- dim(x); m <- matrix(x, prod(d[-length(d)]), d[length(d)])
  mx <- m[, 1]; for (j in seq_len(ncol(m))[-1]) mx <- pmax(mx, m[, j])
  r <- mx + log(rowSums(exp(m - mx))); if (length(d) == 3) matrix(r, d[1], d[2]) else r }

# ---- one learner -------------------------------------------------------------------------------------------------
one_learner <- function(lo) {
  Rc <- R[R$learner != lo, ]; Rc$learner <- factor(Rc$learner)
  fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) +
                   (1 | lf) + (1 | lc), Rc, binomial)
  fe <- glmmTMB::fixef(fit)$cond; re <- glmmTMB::ranef(fit)$cond; vc <- glmmTMB::VarCorr(fit)$cond
  sf0 <- unname(attr(vc$lf, "stddev")); sc0 <- unname(attr(vc$lc, "stddev"))
  cn <- sub("^criterion", "", names(fe)[grepl("^criterion", names(fe))])
  ce <- setNames(c(0, fe[grepl("^criterion", names(fe))]), c(setdiff(sort(unique(Rc$criterion)), cn)[1], cn))
  pick <- function(tab, key) { v <- tab[[1]]; out <- v[match(key, rownames(tab))]; out[is.na(out)] <- 0; out }
  eta_of <- function(D) unname(pick(re$task_family, D$task_family) + pick(re$cond, as.character(D$cond)) +
                               pick(re$inst, D$inst) + pick(re$item, D$item)) + unname(ce[D$criterion])
  # empirical prior for (d, a) from the calibration learners
  Rc$eta <- eta_of(Rc)
  ref <- do.call(rbind, lapply(split(Rc, Rc$learner, drop = TRUE), function(z)
    coef(glm(pass ~ eta, binomial, data = z))))
  m0 <- colMeans(ref); S0 <- 2 * cov(ref)
  it <- unique(R[R$learner == lo, c("item", "inst", "cond", "task_family", "criterion")])
  it$eta <- eta_of(it); names(it)[names(it) == "task_family"] <- "fam"
  y <- R$pass[R$learner == lo][match(it$item, R$item[R$learner == lo])]
  N <- nrow(it)
  sdv <- sqrt(diag(S0))
  dg <- seq(m0[1] - 4 * sdv[1], m0[1] + 4 * sdv[1], length.out = 41)
  ag <- seq(max(-0.5, m0[2] - 4 * sdv[2]), m0[2] + 4 * sdv[2], length.out = 25)
  grid <- expand.grid(d = dg, a = ag); M <- nrow(grid)
  Si <- solve(S0); dz <- cbind(grid$d - m0[1], grid$a - m0[2])
  lprior <- -0.5 * rowSums((dz %*% Si) * dz)
  full_a <- coef(glm(y ~ it$eta, binomial))[2]                         # the learner's full-bank slope (no testlets)
  out <- list()
  for (variant in c("2pl", "2pl_testlet")) {
    sf <- if (variant == "2pl") 0 else sf0; sc <- if (variant == "2pl") 0 else sc0
    lq3 <- array(rep(lq, each = M * Q), c(M, Q, Q)); lq2 <- matrix(rep(lq, each = M), M, Q)
    UW <- outer(gh$nodes * sf, gh$nodes * sc, "+")                     # Q x Q
    lin_of <- function(i) grid$d + grid$a * it$eta[i]                 # M
    S_of <- function(i) array(lin_of(i), c(M, Q, Q)) + array(rep(UW, each = M), c(M, Q, Q))
    # A(d, a): bank proportion with the testlet effects at their prior
    Abank <- rowMeans(vapply(seq_len(N), function(i) {
      l <- lin_of(i); s <- 0; for (q in seq_len(Q)) for (r in seq_len(Q))
        s <- s + gh$weights[q] * gh$weights[r] * plogis(l + UW[q, r]); s }, numeric(M)))
    make_state <- function() list(
      logB = setNames(lapply(unique(as.character(it$cond)), function(z) array(0, c(M, Q, Q))), unique(as.character(it$cond))),
      A = setNames(lapply(unique(as.character(it$cond)), function(z) matrix(0, M, Q)), unique(as.character(it$cond))),
      logP = setNames(lapply(unique(it$fam), function(z) matrix(0, M, Q)), unique(it$fam)),
      logF = setNames(lapply(unique(it$fam), function(z) rep(0, M)), unique(it$fam)), logL = rep(0, M))
    item_lp <- function(i, yy) { a <- S_of(i); if (yy == 1) plogis(a, log.p = TRUE) else plogis(-a, log.p = TRUE) }
    add_item <- function(st, i, yy) {
      c0 <- as.character(it$cond[i]); f0 <- it$fam[i]
      st$logB[[c0]] <- st$logB[[c0]] + item_lp(i, yy)
      An <- lse_last(st$logB[[c0]] + lq3)
      st$logP[[f0]] <- st$logP[[f0]] - st$A[[c0]] + An; st$A[[c0]] <- An
      Fn <- lse_last(st$logP[[f0]] + lq2)
      st$logL <- st$logL - st$logF[[f0]] + Fn; st$logF[[f0]] <- Fn
      st
    }
    post <- function(st) { lp <- st$logL + lprior; w <- exp(lp - max(lp)); w / sum(w) }
    wvar <- function(w, v) { m <- sum(w * v); sum(w * (v - m)^2) }
    exp_var <- function(st, cand) {
      w0 <- post(st)
      vapply(cand, function(i) { c0 <- as.character(it$cond[i]); f0 <- it$fam[i]; ev <- 0
        for (yy in 0:1) {
          An <- lse_last(st$logB[[c0]] + item_lp(i, yy) + lq3)
          Fn <- lse_last(st$logP[[f0]] - st$A[[c0]] + An + lq2)
          ratio <- exp(Fn - st$logF[[f0]]); py <- sum(w0 * ratio); ev <- ev + py * wvar(w0 * ratio / py, Abank)
        }; ev }, 0)
    }
    predict_bank <- function(st, used) {
      w0 <- post(st); t <- sample.int(M, NSIM, replace = TRUE, prob = w0)
      lin <- outer(grid$d[t], rep(1, N)) + outer(grid$a[t], it$eta)
      if (sf > 0 || sc > 0) for (f0 in unique(it$fam)) {
        pu <- exp(st$logP[[f0]][t, , drop = FALSE] + matrix(lq, NSIM, Q, byrow = TRUE)); pu <- pu / rowSums(pu)
        q <- apply(pu, 1, function(p) sample.int(Q, 1, prob = p))
        for (c0 in unique(as.character(it$cond[it$fam == f0]))) {
          lb <- st$logB[[c0]]; pw <- exp(t(vapply(seq_len(NSIM), function(s) lb[t[s], q[s], ], numeric(Q))) +
                                         matrix(lq, NSIM, Q, byrow = TRUE)); pw <- pw / rowSums(pw)
          r <- apply(pw, 1, function(p) sample.int(Q, 1, prob = p))
          idx <- which(as.character(it$cond) == c0)
          lin[, idx] <- lin[, idx] + gh$nodes[q] * sf + gh$nodes[r] * sc
        }
      }
      rest <- setdiff(seq_len(N), used)
      p <- plogis(lin[, rest, drop = FALSE])
      sims <- (sum(y[used]) + rowSums(matrix(rbinom(length(p), 1, p), NSIM))) / N
      c(pred = mean(sims), lo = unname(quantile(sims, 0.025)), hi = unname(quantile(sims, 0.975)),
        a_post = sum(w0 * grid$a), sda = sqrt(wvar(w0, Abank)))
    }
    run_rule <- function(rule) {
      st <- make_state(); used <- integer(0); rec <- list(); ti <- 1; fams_all <- unique(it$fam)
      record <- function(k, what, target, stopped) { pb <- predict_bank(st, used)
        data.frame(what = what, target = target, stopped = stopped, k = k, pred_prop = pb[["pred"]], pred_lo = pb[["lo"]],
                   pred_hi = pb[["hi"]], sd_A = pb[["sda"]], a_post = pb[["a_post"]],
                   max_cond_share = max(table(as.character(it$cond[used]))) / length(used)) }
      for (k in seq_len(min(CAP, N))) {
        free <- setdiff(seq_len(N), used); need <- setdiff(fams_all, unique(it$fam[used]))
        if (length(need)) free <- free[it$fam[free] %in% need]
        if (rule == "random" || (rule == "mixed" && k %% 2 == 0)) nxt <- free[sample.int(length(free), 1)] else {
          w0 <- post(st); pr <- vapply(free, function(i) sum(w0 * plogis(lin_of(i))), 0)
          # screen before the exact calculation: the 20 items nearest a predicted 0.5, the 10 easiest and 10 hardest
          # (an item far from the learner's 50% point is what informs its slope), and 20 at random
          oe <- order(it$eta[free])
          cand <- unique(c(free[order(abs(pr - 0.5))][seq_len(min(20, length(free)))], free[head(oe, 10)], free[tail(oe, 10)],
                           free[sample.int(length(free), min(20, length(free)))]))
          ev <- exp_var(st, cand); o <- order(ev)
          nxt <- if (length(o) > 1 && runif(1) < 0.2) cand[o[2]] else cand[o[1]]
        }
        st <- add_item(st, nxt, y[nxt]); used <- c(used, nxt)
        if (k %in% KFIX) rec[[length(rec) + 1]] <- record(k, "fixed", NA, NA)
        if (k >= length(fams_all) && ti <= length(TARGETS)) {
          sda <- sqrt(wvar(post(st), Abank))
          while (ti <= length(TARGETS) && sda <= TARGETS[ti]) { rec[[length(rec) + 1]] <- record(k, "stop", TARGETS[ti], TRUE); ti <- ti + 1 }
        }
        if (ti > length(TARGETS) && k >= max(KFIX)) break
      }
      while (ti <= length(TARGETS)) { rec[[length(rec) + 1]] <- record(length(used), "stop", TARGETS[ti], FALSE); ti <- ti + 1 }
      do.call(rbind, rec)
    }
    for (rule in c("online", "mixed", "random")) for (r in seq_len(if (rule == "random") RAND else RUNS))
      out[[length(out) + 1]] <- cbind(data.frame(learner = lo, variant = variant, rule = rule, run = r, items_in_bank = N),
                                      run_rule(rule), prop_full = mean(y), a_full = unname(full_a),
                                      a_prior_mean = unname(m0[2]), a_prior_sd = unname(sdv[2]), s_family = sf, s_condition = sc)
  }
  message(lo, " done ", format(Sys.time()))
  do.call(rbind, out)
}

ONLY <- Sys.getenv("CAT_ONLY", "")
todo <- if (ONLY == "") LEARNERS else strsplit(ONLY, ",")[[1]]
PARTS <- file.path(OUT, "online_cat_parts"); dir.create(PARTS, showWarnings = FALSE)
invisible(mclapply(todo, function(lo) { f <- file.path(PARTS, paste0(lo, ".csv"))
  if (!file.exists(f)) write.csv(one_learner(lo), f, row.names = FALSE) }, mc.cores = 2, mc.preschedule = FALSE))
parts <- list.files(PARTS, full.names = TRUE)
if (length(parts) == length(LEARNERS)) {
  res <- do.call(rbind, lapply(parts, read.csv))
  res$prop_err <- res$pred_prop - res$prop_full
  res$covers <- res$pred_lo <= res$prop_full & res$prop_full <= res$pred_hi
  write.csv(res, file.path(OUT, "online_cat_results.csv"), row.names = FALSE)
  res$key <- ifelse(res$what == "stop", paste0("stop at sd(A) ", res$target), paste0("k = ", res$k))
  summ <- do.call(rbind, lapply(split(res, list(res$variant, res$rule, res$key), drop = TRUE), function(x)
    data.frame(variant = x$variant[1], rule = x$rule[1], when = x$key[1], tests = nrow(x),
               share_stopped = if (x$what[1] == "stop") mean(x$stopped) else NA, mean_items = mean(x$k),
               median_items = median(x$k), prop_mean_abs_err_pts = 100 * mean(abs(x$prop_err)),
               prop_signed_err_pts = 100 * mean(x$prop_err), prop_coverage_95 = mean(x$covers),
               prop_width_pts = 100 * mean(x$pred_hi - x$pred_lo),
               cor_err_with_slope = cor(tapply(x$prop_err, x$learner, mean), tapply(x$a_full, x$learner, mean)),
               slope_mean_abs_err = mean(abs(x$a_post - x$a_full)), max_cond_share = mean(x$max_cond_share))))
  write.csv(summ, file.path(OUT, "online_cat_summary.csv"), row.names = FALSE)
  print(summ, digits = 3, row.names = FALSE)
}
