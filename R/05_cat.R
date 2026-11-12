# 05_cat.R -- Computerized adaptive testing of a new learner (Bock and Gibbons 2021, ch. 8), with
# leave-one-learner-out calibration: the task bank is calibrated on 13 learners, and the 14th is then tested
# adaptively, as a new pipeline would be, using its actual answers.
# Calibration: the multilevel Rasch model of 03_irt.R (learners as examinees),
#   logit P(Y_li = 1) = theta_l + c_crit(i) + g_family(i) + v_cond(i) + u_inst(i) + r_i,
# fitted to the 13 learners with glmmTMB; a task-item's easiness e_i is its criterion effect plus the
# predicted random effects (shrunken toward their family and condition, so every item gets a finite easiness,
# even one that all 13 passed). Their uncertainty is ignored, as in most operational CAT.
# Testing: ability prior N(mean, SD) of the 13 calibration learners' abilities; after each answer, the
# expected a posteriori (EAP) ability on a grid; the next task-item is the unused one with the most Fisher
# information at the current EAP (for a Rasch item, p(1 - p): the item whose easiness is nearest -theta),
# ties broken at random; and a version that takes at most one item per task condition in each round. Compared with the same number of items chosen at random (50 orders), each adaptive rule run 20 times (ties), and with the
# EAP from every item the learner answered ("full bank", same calibration).
# Input: output/responses.csv. Output: output/cat_results.csv, output/cat_summary.csv, output/fig_cat.png
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(ggplot2) })
set.seed(805)
R <- read.csv("output/responses.csv", stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":")
R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
LEARNERS <- c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default",
              "xgb_tuned", "svm", "nnet", "stack")
grid <- seq(-8, 12, by = 0.02)
eap <- function(e, y, mu, s) {
  lp <- dnorm(grid, mu, s, log = TRUE)
  for (j in seq_along(e)) lp <- lp + dbinom(y[j], 1, plogis(grid + e[j]), log = TRUE)
  w <- exp(lp - max(lp)); w <- w / sum(w); m <- sum(w * grid); c(eap = m, psd = sqrt(sum(w * (grid - m)^2)))
}
K <- c(5, 10, 20, 40, 80, 160)
res <- list()
for (lo in LEARNERS) {
  Rc <- R[R$learner != lo, ]; Rc$learner <- factor(Rc$learner)
  fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item), Rc, binomial)
  fe <- glmmTMB::fixef(fit)$cond; re <- glmmTMB::ranef(fit)$cond
  th <- fe[grepl("^learner", names(fe))]
  # easiness = criterion effect (the first criterion is absorbed in the learner effects) + random effects
  it <- unique(R[R$learner == lo, c("item", "inst", "cond", "task_family", "criterion")])
  ce <- c(0, fe[grepl("^criterion", names(fe))]); names(ce) <- c(sort(unique(Rc$criterion))[1], sub("^criterion", "", names(fe)[grepl("^criterion", names(fe))]))
  pick <- function(tab, key) { v <- tab[[1]]; out <- v[match(key, rownames(tab))]; out[is.na(out)] <- 0; out }
  it$e <- ce[it$criterion] + pick(re$task_family, it$task_family) + pick(re$cond, as.character(it$cond)) +
          pick(re$inst, it$inst) + pick(re$item, it$item)
  y <- R$pass[R$learner == lo][match(it$item, R$item[R$learner == lo])]
  mu <- mean(th); s <- sd(th)
  full <- eap(it$e, y, mu, s)
  # adaptive: "max_info" takes the most informative unused item; "one_per_condition" takes the most informative
  # unused item from a task condition not yet used (all conditions are then reopened), since items from the same
  # condition share its draws and its learner-by-condition interaction (local dependence) and add less
  # information than the Rasch model credits them with
  run_cat <- function(rule) {
    used <- integer(0); est <- c(eap = mu, psd = s); path <- list()
    for (k in seq_len(min(max(K), nrow(it)))) {
      info <- plogis(est["eap"] + it$e); info <- info * (1 - info); info[used] <- -1
      if (rule == "one_per_condition") {
        # conditions used in the current round (rounds of one item per condition)
        r0 <- length(used) - length(used) %% length(unique(it$cond))
        cur <- if (length(used) > r0) used[(r0 + 1):length(used)] else integer(0)
        info[it$cond %in% it$cond[cur]] <- -1
      }
      cand <- which(info >= max(info) - 1e-12); nxt <- cand[sample.int(length(cand), 1)]
      used <- c(used, nxt); est <- eap(it$e[used], y[used], mu, s)
      if (k %in% K) path[[as.character(k)]] <- est
    }
    path
  }
  # selection is random among tied items, so each adaptive rule is run 20 times; random selection 50 times
  runs <- list(max_info = replicate(20, run_cat("max_info"), simplify = FALSE),
               one_per_condition = replicate(20, run_cat("one_per_condition"), simplify = FALSE))
  for (k in K[K <= nrow(it)]) {
    rnd <- lapply(1:50, function(z) { u <- sample.int(nrow(it), k); eap(it$e[u], y[u], mu, s) })
    for (rule in c(names(runs), "random")) {
      ests <- if (rule == "random") rnd else lapply(runs[[rule]], function(p) p[[as.character(k)]])
      res[[length(res) + 1]] <- data.frame(learner = lo, items_in_bank = nrow(it), k = k, rule = rule, run = seq_along(ests),
        theta = vapply(ests, `[[`, 0, "eap"), psd = vapply(ests, `[[`, 0, "psd"),
        theta_full = full[["eap"]], psd_full = full[["psd"]], prior_mean = mu, prior_sd = s)
    }
  }
  message(lo, " done")
}
res <- do.call(rbind, res); res$abs_err <- abs(res$theta - res$theta_full)
write.csv(res, "output/cat_results.csv", row.names = FALSE)
# per k and rule: mean absolute error over learners and runs; the share of learners whose error is within
# 1.96 posterior SDs (is the reported precision honest?); the Spearman correlation with the full-bank
# ranking, averaged over runs (runs 1-20)
summ <- do.call(rbind, lapply(split(res, list(res$k, res$rule), drop = TRUE), function(x) {
  rr <- x[x$run <= 20, ]
  data.frame(k = x$k[1], rule = x$rule[1], learners = length(unique(x$learner)), mean_abs_err = mean(x$abs_err),
             mean_psd = mean(x$psd), within_1.96_psd = mean(x$abs_err <= 1.96 * x$psd),
             spearman_vs_full = mean(vapply(split(rr, rr$run), function(z) cor(z$theta, z$theta_full, method = "spearman"), 0))) }))
summ <- summ[order(summ$k, summ$rule), ]
write.csv(summ, "output/cat_summary.csv", row.names = FALSE)
print(summ, digits = 3, row.names = FALSE)
lab <- c(max_info = "adaptive (most informative item)", one_per_condition = "adaptive, one item per condition per round",
         random = "random")
pl <- aggregate(abs_err ~ learner + k + rule, res, mean); pl$how <- lab[pl$rule]
g <- ggplot(pl, aes(k, abs_err, colour = how)) + geom_line(aes(group = interaction(learner, how)), alpha = 0.25) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) + scale_x_log10(breaks = K) +
  labs(x = "task-items given (log scale)", y = "|ability after k items - full-bank ability| (logits)", colour = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = "bottom", legend.direction = "vertical")
ggsave("output/fig_cat.png", g, width = 7, height = 4.5, dpi = 200)
