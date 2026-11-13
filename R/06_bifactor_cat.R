# 06_bifactor_cat.R -- Adaptive testing of a new learner when the task-items are locally dependent.
# 05_cat.R found that maximum-information selection under the Rasch model was no more accurate than random
# selection, and that its posterior SDs were too small (the error was within 1.96 posterior SDs for only
# 79-83% of learners after 80-160 items). The likely reason is local dependence: a learner's answers to items
# from the same task family, and still more the same task condition, go together beyond its overall ability
# (learner-by-family SD 2.4 and learner-by-condition SD 2.0 logits, on 14 learners; larger than the spread of
# the abilities themselves), and maximum-information selection draws many items from one condition.
# This is the situation the bifactor model was made for (Gibbons and Hedeker 1992; Gibbons et al. 2007): each
# item measures the primary dimension and one specific dimension, and the primary score is the EAP with the
# specific dimensions integrated out (their eqs. 17-18), which is what the CAT-MH, K-CAT (Gibbons et al. 2020)
# and CAT-Autism (Tseng et al. 2026) tests report.
# Model here, with loadings fixed at 1 (13 calibration learners cannot estimate loadings), i.e. a Rasch
# testlet version of the bifactor model with the specific dimensions nested:
#   logit P(Y_li = 1) = theta_l + u_l,f(i) + w_l,c(i) + e_i
#   theta_l ~ N(mu, s^2) (primary: the learner's ability), u_lf ~ N(0, s_f^2) (learner by task family),
#   w_lc ~ N(0, s_c^2) (learner by task condition, c within f), e_i the item's easiness
# so the likelihood for the primary dimension is a product over families of a double integral (over u_f, and
# over each w_c within f), evaluated by Gauss-Hermite quadrature.
# Calibration: glmmTMB on the 13 other learners (the multilevel Rasch model of 03_irt.R plus learner-by-family
# and learner-by-condition random effects); e_i, s_f and s_c are taken as known.
# Selection rules, all scored by the bifactor EAP:
#   "bifactor": the item that minimizes the expected posterior variance of theta (preposterior analysis under
#     the bifactor model), taken with probability 0.8, the second best with 0.2 (CAT-Autism's exposure rule);
#   "rasch_max_info": 05_cat.R's rule (the item with most Rasch information at the current estimate);
#   "random".
# Reference: the bifactor EAP from every item the learner answered (same calibration). Lengths k = 5-80;
# also the number of items each rule needs to bring the posterior SD within 25% and 10% of the full bank's (with
# the local dependence the full bank's posterior SD is about 0.75, so K-CAT's absolute criterion of 0.3 cannot
# be reached; NA = not reached in 80 items).
# Output: output/bifactor_cat_results.csv, output/bifactor_cat_summary.csv, output/fig_bifactor_cat.png
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(ggplot2); library(statmod); library(parallel) })
RNGkind("L'Ecuyer-CMRG"); set.seed(806)
# IRT_RESPONSES and IRT_OUT change the input and the output folder (the 19-learner run:
# IRT_RESPONSES=output/responses_19.csv IRT_OUT=output/v19)
OUT <- Sys.getenv("IRT_OUT", "output"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses.csv"), stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":")
R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$lf <- paste(R$learner, R$task_family); R$lc <- paste(R$learner, R$cond)
LEARNERS <- c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default",
              "xgb_tuned", "svm", "nnet", "stack", "lightgbm", "catboost", "bart", "realmlp", "tabpfn",
              "lightgbm_tuned", "catboost_tuned")
LEARNERS <- LEARNERS[LEARNERS %in% unique(R$learner)]
K <- c(5, 10, 20, 40, 80); KMAX <- max(K)
RUNS <- as.integer(Sys.getenv("BCAT_RUNS", "3")); RAND <- as.integer(Sys.getenv("BCAT_RAND", "10"))
g <- seq(-7, 9, by = 0.15); G <- length(g)
gh <- gauss.quad.prob(7, "normal"); Q <- length(gh$nodes); lq <- log(gh$weights)
# log-sum-exp over the last dimension (of an array G x Q x Q, or a matrix G x Q)
lse_last <- function(x) { d <- dim(x); m <- matrix(x, prod(d[-length(d)]), d[length(d)])
  mx <- m[, 1]; for (j in seq_len(ncol(m))[-1]) mx <- pmax(mx, m[, j])
  r <- mx + log(rowSums(exp(m - mx))); if (length(d) == 3) matrix(r, d[1], d[2]) else r }
lq3 <- array(rep(lq, each = G * Q), c(G, Q, Q)); lq2 <- matrix(rep(lq, each = G), G, Q)

# ---- posterior machinery for one learner ---------------------------------------------------------------------
# state: for each condition, logB[G, Q(u), Q(w)] (log-likelihood of its answered items) and A = log of its
# w-integral [G, Q(u)]; for each family, logP = sum of its conditions' A, logF = log of its u-integral [G]
make_state <- function(it, sf, sc) {
  S <- array(g, c(G, Q, Q)) + array(rep(gh$nodes * sf, each = G), c(G, Q, Q)) +
       array(rep(gh$nodes * sc, each = G * Q), c(G, Q, Q))
  conds <- unique(it$cond); fams <- unique(it$fam)
  list(S = S, logB = setNames(lapply(conds, function(z) array(0, c(G, Q, Q))), conds),
       A = setNames(lapply(conds, function(z) matrix(0, G, Q)), conds),
       logP = setNames(lapply(fams, function(z) matrix(0, G, Q)), fams),
       logF = setNames(lapply(fams, function(z) rep(0, G)), fams), logL = rep(0, G))
}
post_of <- function(logL, lprior) { lp <- logL + lprior; w <- exp(lp - max(lp)); w / sum(w) }
mv <- function(w) { m <- sum(w * g); c(eap = m, psd = sqrt(sum(w * (g - m)^2))) }
add_item <- function(st, it, i, y) {
  c0 <- as.character(it$cond[i]); f0 <- it$fam[i]
  lp <- if (y == 1) plogis(st$S + it$e[i], log.p = TRUE) else plogis(-(st$S + it$e[i]), log.p = TRUE)
  st$logB[[c0]] <- st$logB[[c0]] + lp
  Anew <- lse_last(st$logB[[c0]] + lq3)
  st$logP[[f0]] <- st$logP[[f0]] - st$A[[c0]] + Anew; st$A[[c0]] <- Anew
  Fnew <- lse_last(st$logP[[f0]] + lq2)
  st$logL <- st$logL - st$logF[[f0]] + Fnew; st$logF[[f0]] <- Fnew
  st
}
# expected posterior variance of theta after answering each candidate item (vectorized within a condition)
exp_var <- function(st, it, cand, lprior) {
  w0 <- post_of(st$logL, lprior); out <- rep(NA_real_, length(cand))
  for (c0 in unique(as.character(it$cond[cand]))) {
    ii <- cand[as.character(it$cond[cand]) == c0]; f0 <- it$fam[ii[1]]
    for (j in seq_along(ii)) {
      ev <- 0
      for (y in 0:1) {
        lp <- if (y == 1) plogis(st$S + it$e[ii[j]], log.p = TRUE) else plogis(-(st$S + it$e[ii[j]]), log.p = TRUE)
        An <- lse_last(st$logB[[c0]] + lp + lq3)
        Fn <- lse_last(st$logP[[f0]] - st$A[[c0]] + An + lq2)
        ratio <- exp(Fn - st$logF[[f0]])                       # P(y | theta, answers so far)
        py <- sum(w0 * ratio); wy <- w0 * ratio / py
        ev <- ev + py * sum(wy * (g - sum(wy * g))^2)
      }
      out[match(ii[j], cand)] <- ev
    }
  }
  out
}

run_rule <- function(rule, it, y, sf, sc, lprior, psd_full) {
  st <- make_state(it, sf, sc); used <- integer(0); path <- list(); n05 <- NA; n03 <- NA
  for (k in seq_len(min(KMAX, nrow(it)))) {
    free <- setdiff(seq_len(nrow(it)), used)
    if (rule == "random") nxt <- free[sample.int(length(free), 1)]
    if (rule == "rasch_max_info") {
      th <- mv(post_of(st$logL, lprior))[["eap"]]; inf <- plogis(th + it$e[free]); inf <- inf * (1 - inf)
      cand <- free[inf >= max(inf) - 1e-12]; nxt <- cand[sample.int(length(cand), 1)]
    }
    if (rule == "bifactor") {
      # screen to the 40 items with most Rasch information at the current estimate, plus 20 at random, before
      # the exact (slower) preposterior calculation
      th <- mv(post_of(st$logL, lprior))[["eap"]]; inf <- plogis(th + it$e[free]); inf <- inf * (1 - inf)
      cand <- unique(c(free[order(-inf)][seq_len(min(40, length(free)))], free[sample.int(length(free), min(20, length(free)))]))
      ev <- exp_var(st, it, cand, lprior); o <- order(ev)
      nxt <- if (length(o) > 1 && runif(1) < 0.2) cand[o[2]] else cand[o[1]]
    }
    st <- add_item(st, it, nxt, y[nxt]); used <- c(used, nxt)
    e <- mv(post_of(st$logL, lprior))
    if (is.na(n05) && e[["psd"]] <= 1.25 * psd_full) n05 <- k
    if (is.na(n03) && e[["psd"]] <= 1.1 * psd_full) n03 <- k
    if (k %in% K) path[[as.character(k)]] <- e
  }
  list(path = path, n05 = n05, n03 = n03)
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
  it$e <- unname(ce[it$criterion] + pick(re$task_family, it$fam) + pick(re$cond, as.character(it$cond)) +
                 pick(re$inst, it$inst) + pick(re$item, it$item))
  y <- R$pass[R$learner == lo][match(it$item, R$item[R$learner == lo])]
  lprior <- dnorm(g, mean(th), sd(th), log = TRUE)
  st <- make_state(it, sf, sc); for (i in seq_len(nrow(it))) st <- add_item(st, it, i, y[i])
  full <- mv(post_of(st$logL, lprior))
  rows <- list()
  for (rule in c("bifactor", "rasch_max_info", "random")) {
    nr <- if (rule == "random") RAND else RUNS
    for (r in seq_len(nr)) {
      p <- run_rule(rule, it, y, sf, sc, lprior, full[["psd"]])
      for (k in names(p$path)) rows[[length(rows) + 1]] <- data.frame(learner = lo, items_in_bank = nrow(it), rule = rule,
        run = r, k = as.integer(k), theta = p$path[[k]][["eap"]], psd = p$path[[k]][["psd"]],
        theta_full = full[["eap"]], psd_full = full[["psd"]], n_to_1.25_full_psd = p$n05, n_to_1.1_full_psd = p$n03,
        s_family = sf, s_condition = sc)
    }
  }
  message(lo, " done ", format(Sys.time()))
  do.call(rbind, rows)
}

ONLY <- Sys.getenv("BCAT_ONLY", "")
todo <- if (ONLY == "") LEARNERS else strsplit(ONLY, ",")[[1]]
dir.create(file.path(OUT, "bcat_parts"), showWarnings = FALSE)
invisible(mclapply(todo, function(lo) { f <- file.path(OUT, "bcat_parts", paste0(lo, ".csv"))
  if (!file.exists(f)) write.csv(one_learner(lo), f, row.names = FALSE) }, mc.cores = 2, mc.preschedule = FALSE))
parts <- list.files(file.path(OUT, "bcat_parts"), full.names = TRUE)
if (length(parts) == length(LEARNERS)) {
  res <- do.call(rbind, lapply(parts, read.csv)); res$abs_err <- abs(res$theta - res$theta_full)
  write.csv(res, file.path(OUT, "bifactor_cat_results.csv"), row.names = FALSE)
  summ <- do.call(rbind, lapply(split(res, list(res$k, res$rule), drop = TRUE), function(x)
    data.frame(k = x$k[1], rule = x$rule[1], learners = length(unique(x$learner)), mean_abs_err = mean(x$abs_err),
               mean_psd = mean(x$psd), within_1.96_psd = mean(x$abs_err <= 1.96 * x$psd),
               spearman_vs_full = mean(vapply(split(x, x$run), function(z) cor(z$theta, z$theta_full, method = "spearman"), 0)))))
  summ <- summ[order(summ$k, summ$rule), ]
  n_to <- aggregate(cbind(n_to_1.25_full_psd, n_to_1.1_full_psd) ~ rule, unique(res[, c("learner", "rule", "run", "n_to_1.25_full_psd", "n_to_1.1_full_psd")]),
                    function(v) median(v, na.rm = TRUE), na.action = na.pass)
  write.csv(summ, file.path(OUT, "bifactor_cat_summary.csv"), row.names = FALSE)
  print(summ, digits = 3, row.names = FALSE); print(n_to)
  lab <- c(bifactor = "adaptive, bifactor (expected posterior variance)", rasch_max_info = "adaptive, Rasch information",
           random = "random")
  pl <- aggregate(abs_err ~ learner + k + rule, res, mean); pl$how <- lab[pl$rule]
  gg <- ggplot(pl, aes(k, abs_err, colour = how)) + geom_line(aes(group = interaction(learner, how)), alpha = 0.2) +
    stat_summary(fun = mean, geom = "line", linewidth = 1.2) + scale_x_log10(breaks = K) +
    labs(x = "task-items given (log scale)", y = "|ability after k items - full-bank ability| (logits)", colour = NULL) +
    theme_minimal(base_size = 11) + theme(legend.position = "bottom", legend.direction = "vertical")
  ggsave(file.path(OUT, "fig_bifactor_cat.png"), gg, width = 7, height = 4.8, dpi = 200)
}
