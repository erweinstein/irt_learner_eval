# 06d_cat_stopping_rule.R -- Adaptive tests that stop by a posterior-SD rule, judged by coverage.
# (Added 18 November 2026; FLAGS.md item 28. 06_bifactor_cat.R is unchanged; this adds to it.)
#
# 06_bifactor_cat.R gives every rule the same fixed lengths (5-80 items) and judges it by distance from the
# full-bank ability. Operational adaptive tests instead stop when the posterior SD of the score falls below a
# threshold, with a minimum that makes the test cover every content area (content balancing), and a cap. Here:
#   - Minimum: one task-item from every task family the learner has items in (16 families) before any other item.
#   - Stop: posterior SD of the primary ability <= each threshold in CAT_STOP_PSD (default 0.9 and 0.7 logits),
#     or the cap CAT_CAP (default 150 items) if it is not reached first. The thresholds are absolute, as in
#     deployment; with the local dependence of this bank the full bank's own posterior SD is 0.39-1.01 logits,
#     so some learners cannot reach 0.7 at all, and the share that stops is part of the result.
#   - Rules, all scored by the bifactor EAP of 06_bifactor_cat.R (Rasch testlet bifactor, specific dimensions
#     for family and condition): "bifactor" (least expected posterior variance, exposure 0.8/0.2), "rasch_max_info"
#     and "random", each under the same minimum and stopping rule. A random arm that stops by the same rule is the
#     fair comparison for an adaptive rule; the fixed-length comparison of 06 favours whichever rule is faster.
#   - Pools: "all" task-items, and "no_unanimous": without the task-items that every calibration learner passed
#     or every one failed (decided from the calibration learners only, so no information from the left-out
#     learner is used). Random selection from the reduced pool is the same-pool control for the adaptive rules.
# What is recorded at each stop:
#   - the error against the full-bank ability (same calibration), and whether it is within 1.96 posterior SDs
#     (06's check; lenient, since the full-bank reference shares the answers);
#   - a stricter check on a quantity readers know: the learner's proportion of criteria met over its whole bank.
#     The prediction combines the answers given with simulated answers to the rest, drawn from the posterior of
#     (theta, the family effects, the condition effects); the 95% predictive interval's coverage of the observed
#     full-bank proportion, its width and the signed error are reported;
#   - the mean residual on the administered items, observed minus predicted from the full-bank ability with the
#     specific dimensions at their prior (a check of whether selection lands on items the model misfits), and the
#     largest share of the administered items in one task condition (how much selection bunches).
#
# Input: IRT_RESPONSES (default output/responses_21.csv). Output (IRT_OUT, default output/v21):
#   cat_stop_parts/<learner>.csv, cat_stop_results.csv, cat_stop_summary.csv.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(statmod); library(parallel) })
RNGkind("L'Ecuyer-CMRG"); set.seed(20261118)
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
TARGETS <- sort(as.numeric(strsplit(Sys.getenv("CAT_STOP_PSD", "0.9,0.7"), ",")[[1]]), decreasing = TRUE)
CAP <- as.integer(Sys.getenv("CAT_CAP", "150"))
RUNS <- as.integer(Sys.getenv("CAT_RUNS", "3")); RAND <- as.integer(Sys.getenv("CAT_RAND", "10"))
NSIM <- as.integer(Sys.getenv("CAT_NSIM", "400"))
g <- seq(-7, 9, by = 0.15); G <- length(g)
gh <- gauss.quad.prob(7, "normal"); Q <- length(gh$nodes); lq <- log(gh$weights)

# ---- posterior machinery: as in 06_bifactor_cat.R (binary responses) --------------------------------------------
lse_last <- function(x) { d <- dim(x); m <- matrix(x, prod(d[-length(d)]), d[length(d)])
  mx <- m[, 1]; for (j in seq_len(ncol(m))[-1]) mx <- pmax(mx, m[, j])
  r <- mx + log(rowSums(exp(m - mx))); if (length(d) == 3) matrix(r, d[1], d[2]) else r }
lq3 <- array(rep(lq, each = G * Q), c(G, Q, Q)); lq2 <- matrix(rep(lq, each = G), G, Q)
make_state <- function(it, sf, sc) {
  S <- array(g, c(G, Q, Q)) + array(rep(gh$nodes * sf, each = G), c(G, Q, Q)) +
       array(rep(gh$nodes * sc, each = G * Q), c(G, Q, Q))
  conds <- unique(as.character(it$cond)); fams <- unique(it$fam)
  list(S = S, logB = setNames(lapply(conds, function(z) array(0, c(G, Q, Q))), conds),
       A = setNames(lapply(conds, function(z) matrix(0, G, Q)), conds),
       logP = setNames(lapply(fams, function(z) matrix(0, G, Q)), fams),
       logF = setNames(lapply(fams, function(z) rep(0, G)), fams), logL = rep(0, G))
}
post_of <- function(logL, lprior) { lp <- logL + lprior; w <- exp(lp - max(lp)); w / sum(w) }
mv <- function(w) { m <- sum(w * g); c(eap = m, psd = sqrt(sum(w * (g - m)^2))) }
item_lp <- function(S, it, i, y) { a <- S + it$e1[i]; if (y == 1) plogis(a, log.p = TRUE) else plogis(-a, log.p = TRUE) }
info_at <- function(th, it, idx) { p1 <- plogis(th + it$e1[idx]); p1 * (1 - p1) }
add_item <- function(st, it, i, y) {
  c0 <- as.character(it$cond[i]); f0 <- it$fam[i]
  st$logB[[c0]] <- st$logB[[c0]] + item_lp(st$S, it, i, y)
  Anew <- lse_last(st$logB[[c0]] + lq3)
  st$logP[[f0]] <- st$logP[[f0]] - st$A[[c0]] + Anew; st$A[[c0]] <- Anew
  Fnew <- lse_last(st$logP[[f0]] + lq2)
  st$logL <- st$logL - st$logF[[f0]] + Fnew; st$logF[[f0]] <- Fnew
  st
}
exp_var <- function(st, it, cand, lprior) {
  w0 <- post_of(st$logL, lprior); out <- rep(NA_real_, length(cand))
  for (c0 in unique(as.character(it$cond[cand]))) {
    ii <- cand[as.character(it$cond[cand]) == c0]; f0 <- it$fam[ii[1]]
    for (j in seq_along(ii)) {
      ev <- 0
      for (y in 0:1) {
        An <- lse_last(st$logB[[c0]] + item_lp(st$S, it, ii[j], y) + lq3)
        Fn <- lse_last(st$logP[[f0]] - st$A[[c0]] + An + lq2)
        ratio <- exp(Fn - st$logF[[f0]]); py <- sum(w0 * ratio); wy <- w0 * ratio / py
        ev <- ev + py * sum(wy * (g - sum(wy * g))^2)
      }
      out[match(ii[j], cand)] <- ev
    }
  }
  out
}

# ---- new: predictive distribution of the full-bank proportion met ----------------------------------------------
# Draw theta from its posterior; for each family, u from its posterior given theta and the family's answers; for
# each condition, w given theta, u and its answers (on the quadrature nodes); then the unanswered items.
predict_bank <- function(st, it, y, used, lprior, sf, sc) {
  w0 <- post_of(st$logL, lprior); t <- sample.int(G, NSIM, replace = TRUE, prob = w0)
  lin <- matrix(g[t], NSIM, nrow(it))
  for (f0 in unique(it$fam)) {
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
  rest <- setdiff(seq_len(nrow(it)), used)
  p <- plogis(lin[, rest, drop = FALSE] + matrix(it$e1[rest], NSIM, length(rest), byrow = TRUE))
  sims <- (sum(y[used]) + rowSums(matrix(rbinom(length(p), 1, p), NSIM))) / nrow(it)
  c(pred = mean(sims), lo = unname(quantile(sims, 0.025)), hi = unname(quantile(sims, 0.975)))
}
# P(y = 1) for each item at ability th, with the specific dimensions at their prior
p_marg <- function(th, it, idx, sf, sc) {
  out <- 0; for (a in seq_len(Q)) for (b in seq_len(Q))
    out <- out + gh$weights[a] * gh$weights[b] * plogis(th + it$e1[idx] + gh$nodes[a] * sf + gh$nodes[b] * sc)
  out }

run_rule <- function(rule, it, y, pool, sf, sc, lprior, th_full) {
  st <- make_state(it, sf, sc); used <- integer(0); stops <- list(); ti <- 1
  fams_all <- unique(it$fam[pool])
  record <- function(k, target, stopped) {
    e <- mv(post_of(st$logL, lprior)); pb <- predict_bank(st, it, y, used, lprior, sf, sc)
    data.frame(target = target, stopped = stopped, k = k, theta = e[["eap"]], psd = e[["psd"]],
               pred_prop = pb[["pred"]], pred_lo = pb[["lo"]], pred_hi = pb[["hi"]],
               resid_admin = mean(y[used] - p_marg(th_full, it, used, sf, sc)),
               max_cond_share = max(table(as.character(it$cond[used]))) / length(used))
  }
  for (k in seq_len(min(CAP, length(pool)))) {
    free <- setdiff(pool, used)
    need <- setdiff(fams_all, unique(it$fam[used]))
    if (length(need)) free <- free[it$fam[free] %in% need]           # content balancing: one item per family first
    if (rule == "random") nxt <- free[sample.int(length(free), 1)]
    if (rule == "rasch_max_info") {
      th <- mv(post_of(st$logL, lprior))[["eap"]]; inf <- info_at(th, it, free)
      cand <- free[inf >= max(inf) - 1e-12]; nxt <- cand[sample.int(length(cand), 1)]
    }
    if (rule == "bifactor") {
      th <- mv(post_of(st$logL, lprior))[["eap"]]; inf <- info_at(th, it, free)
      cand <- unique(c(free[order(-inf)][seq_len(min(40, length(free)))], free[sample.int(length(free), min(20, length(free)))]))
      ev <- exp_var(st, it, cand, lprior); o <- order(ev)
      nxt <- if (length(o) > 1 && runif(1) < 0.2) cand[o[2]] else cand[o[1]]
    }
    st <- add_item(st, it, nxt, y[nxt]); used <- c(used, nxt)
    if (k >= length(fams_all)) {
      psd <- mv(post_of(st$logL, lprior))[["psd"]]
      while (ti <= length(TARGETS) && psd <= TARGETS[ti]) { stops[[ti]] <- record(k, TARGETS[ti], TRUE); ti <- ti + 1 }
    }
    if (ti > length(TARGETS)) break
  }
  while (ti <= length(TARGETS)) { stops[[ti]] <- record(length(used), TARGETS[ti], FALSE); ti <- ti + 1 }
  do.call(rbind, stops)
}

one_learner <- function(lo) {
  Rc <- R[R$learner != lo, ]; Rc$learner <- factor(Rc$learner)
  fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) +
                   (1 | lf) + (1 | lc), Rc, binomial)
  fe <- glmmTMB::fixef(fit)$cond; re <- glmmTMB::ranef(fit)$cond; vc <- glmmTMB::VarCorr(fit)$cond
  sf <- unname(attr(vc$lf, "stddev")); sc <- unname(attr(vc$lc, "stddev"))
  th <- fe[grepl("^learner", names(fe))]
  it <- unique(R[R$learner == lo, c("item", "inst", "cond", "task_family", "criterion")]); names(it)[4] <- "fam"
  cn <- sub("^criterion", "", names(fe)[grepl("^criterion", names(fe))])
  ce <- setNames(c(0, fe[grepl("^criterion", names(fe))]), c(setdiff(sort(unique(Rc$criterion)), cn)[1], cn))
  pick <- function(tab, key) { v <- tab[[1]]; out <- v[match(key, rownames(tab))]; out[is.na(out)] <- 0; out }
  it$e1 <- unname(pick(re$task_family, it$fam) + pick(re$cond, as.character(it$cond)) + pick(re$inst, it$inst) +
                  pick(re$item, it$item)) + unname(ce[it$criterion])
  stopifnot(all(is.finite(it$e1)))
  y <- R$pass[R$learner == lo][match(it$item, R$item[R$learner == lo])]
  lprior <- dnorm(g, mean(th), sd(th), log = TRUE)
  st <- make_state(it, sf, sc); for (i in seq_len(nrow(it))) st <- add_item(st, it, i, y[i])
  full <- mv(post_of(st$logL, lprior))
  pc <- tapply(Rc$pass, Rc$item, mean)                                 # unanimous among the calibration learners
  unan <- it$item %in% names(pc)[pc %in% c(0, 1)]
  pools <- list(all = seq_len(nrow(it)), no_unanimous = which(!unan))
  rows <- list()
  for (pn in names(pools)) for (rule in c("bifactor", "rasch_max_info", "random")) {
    nr <- if (rule == "random") RAND else RUNS
    for (r in seq_len(nr)) {
      s <- run_rule(rule, it, y, pools[[pn]], sf, sc, lprior, full[["eap"]])
      rows[[length(rows) + 1]] <- cbind(data.frame(learner = lo, pool = pn, rule = rule, run = r,
        items_in_bank = nrow(it), items_in_pool = length(pools[[pn]]), unanimous_in_bank = sum(unan)), s,
        theta_full = full[["eap"]], psd_full = full[["psd"]], prop_full = mean(y), s_family = sf, s_condition = sc)
    }
  }
  message(lo, " done ", format(Sys.time()))
  do.call(rbind, rows)
}

ONLY <- Sys.getenv("CAT_ONLY", "")
todo <- if (ONLY == "") LEARNERS else strsplit(ONLY, ",")[[1]]
PARTS <- file.path(OUT, "cat_stop_parts"); dir.create(PARTS, showWarnings = FALSE)
invisible(mclapply(todo, function(lo) { f <- file.path(PARTS, paste0(lo, ".csv"))
  if (!file.exists(f)) write.csv(one_learner(lo), f, row.names = FALSE) }, mc.cores = 2, mc.preschedule = FALSE))
parts <- list.files(PARTS, full.names = TRUE)
if (length(parts) == length(LEARNERS)) {
  res <- do.call(rbind, lapply(parts, read.csv))
  res$abs_err <- abs(res$theta - res$theta_full)
  res$within_1.96 <- res$abs_err <= 1.96 * res$psd
  res$prop_err <- res$pred_prop - res$prop_full
  res$covers <- res$pred_lo <= res$prop_full & res$prop_full <= res$pred_hi
  write.csv(res, file.path(OUT, "cat_stop_results.csv"), row.names = FALSE)
  summ <- do.call(rbind, lapply(split(res, list(res$pool, res$rule, res$target), drop = TRUE), function(x)
    data.frame(pool = x$pool[1], rule = x$rule[1], target_psd = x$target[1], tests = nrow(x),
               share_stopped = mean(x$stopped), mean_items = mean(x$k), median_items = median(x$k),
               mean_abs_err = mean(x$abs_err), within_1.96_psd = mean(x$within_1.96),
               prop_mean_abs_err_pts = 100 * mean(abs(x$prop_err)), prop_signed_err_pts = 100 * mean(x$prop_err),
               prop_coverage_95 = mean(x$covers), prop_width_pts = 100 * mean(x$pred_hi - x$pred_lo),
               resid_admin_pts = 100 * mean(x$resid_admin), max_cond_share = mean(x$max_cond_share))))
  summ <- summ[order(summ$target_psd, summ$pool, summ$rule, decreasing = c(TRUE, FALSE, FALSE), method = "radix"), ]
  write.csv(summ, file.path(OUT, "cat_stop_summary.csv"), row.names = FALSE)
  print(summ, digits = 3, row.names = FALSE)
}
