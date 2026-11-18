# 03d_unanimous_sensitivity.R -- Do the task-items that every learner passes, or every learner fails, change
# the fits? (Added 18 November 2026; FLAGS.md item 27.)
#
# In the flipped orientation a unanimous task-item is a respondent with a perfect or a zero score: it says
# nothing about how the learners differ, but it still enters the latent distribution of task easiness, which
# fixes the scale of the slopes and intercepts. Dropping such respondents is a common step when a fit has
# trouble converging. In the usual orientation the same items carry no information about the learners'
# abilities under the Rasch model, but they inform the variance of the task difficulties.
# This script refits, with and without the unanimous task-items:
#   B. the flipped one-factor 2PL and the bifactor model of 03_irt.R part B (mirt);
#   A. the multilevel Rasch model A1 of 03_irt.R part A (glmmTMB).
# A task-item is unanimous if every learner that answered it passed, or every one failed.
#
# Input: IRT_RESPONSES (default output/responses_21.csv). Output (in IRT_OUT, default output/v21):
#   unanimous_items.csv, unanimous_sensitivity_flipped.csv (each learner's slope and intercept, both fits),
#   unanimous_sensitivity_fit.csv (convergence, iterations, logLik, BIC, M2 RMSEA), unanimous_sensitivity_A1.csv
#   (abilities and variance components, both fits).
# IRT_NCYCLES (default 500, mirt's default and what 03_irt.R uses) sets the EM cycle limit for the flipped fits;
# with another value the two flipped output files end in _ncyc<value>. At 500 cycles the bifactor fit of all
# task-items, as in 03_irt.R, stops without converging (FLAGS.md item 27).
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(mirt) })
set.seed(20261118)
OUT <- Sys.getenv("IRT_OUT", "output/v21"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
NC <- as.integer(Sys.getenv("IRT_NCYCLES", "500")); SUF <- if (NC == 500) "" else paste0("_ncyc", NC)
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_21.csv"), stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":")
LEARNERS <- c("glm", "glm_matched", "lasso", "ridge", "enet", "gam", "knn", "cart", "rf", "xgb_default",
              "xgb_tuned", "svm", "nnet", "stack", "lightgbm", "catboost", "bart", "realmlp", "tabpfn",
              "lightgbm_tuned", "catboost_tuned")
LEARNERS <- LEARNERS[LEARNERS %in% unique(R$learner)]
fam_of <- c(glm = "linear", glm_matched = "linear", lasso = "linear", ridge = "linear", enet = "linear", gam = "linear",
            knn = "local", svm = "local", nnet = "local", cart = "trees", rf = "trees", xgb_default = "trees",
            xgb_tuned = "trees", stack = "ensemble", lightgbm = "trees", catboost = "trees", bart = "trees",
            realmlp = "local", tabpfn = "pretrained", lightgbm_tuned = "trees", catboost_tuned = "trees")
R$learner <- factor(R$learner, levels = LEARNERS)
R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)

pm <- tapply(R$pass, R$item, mean); nresp <- tapply(R$pass, R$item, length)
un <- data.frame(item = names(pm), answered_by = as.integer(nresp), prop_pass = as.numeric(pm),
                 unanimous = ifelse(pm == 1, "all pass", ifelse(pm == 0, "all fail", "")))
un$task_family <- R$task_family[match(un$item, R$item)]
write.csv(un[un$unanimous != "", ], file.path(OUT, "unanimous_items.csv"), row.names = FALSE)
drop_items <- un$item[un$unanimous != ""]
cat(sprintf("%d task-items; %d unanimous (%d all pass, %d all fail)\n", nrow(un), length(drop_items),
            sum(un$unanimous == "all pass"), sum(un$unanimous == "all fail")))
print(table(un$task_family[un$unanimous != ""], un$unanimous[un$unanimous != ""]))

# ---- B. flipped fits, as in 03_irt.R part B -------------------------------------------------------------------
flip_matrix <- function(R) {
  W <- reshape(R[, c("item", "learner", "pass")], idvar = "item", timevar = "learner", direction = "wide")
  names(W) <- sub("^pass\\.", "", names(W)); rownames(W) <- W$item
  Y <- W[, intersect(LEARNERS, names(W))]
  Y <- Y[, apply(Y, 2, function(z) length(unique(na.omit(z))) > 1)]
  Y[rowSums(!is.na(Y)) >= 4, ]
}
fit_flipped <- function(Y, label) {
  B1 <- mirt(Y, 1, itemtype = "2PL", verbose = FALSE, SE = TRUE, technical = list(NCYCLES = NC))
  spec <- c(linear = 1, trees = 2, local = 3, ensemble = NA, pretrained = NA)[fam_of[colnames(Y)]]
  BB <- bfactor(Y, spec, verbose = FALSE, SE = TRUE, technical = list(NCYCLES = NC))
  fitrow <- function(m, nm) { M2 <- tryCatch(M2(m, na.rm = TRUE), error = function(e) NULL)
    data.frame(bank = label, model = nm, respondents = nrow(Y), converged = extract.mirt(m, "converged"),
               iterations = extract.mirt(m, "iterations"), logLik = extract.mirt(m, "logLik"),
               BIC = extract.mirt(m, "BIC"), RMSEA = if (is.null(M2)) NA else M2$RMSEA,
               secondordertest = extract.mirt(m, "secondordertest")) }
  cf <- coef(B1, IRTpars = TRUE, simplify = TRUE)$items
  cb <- coef(BB, simplify = TRUE)$items
  items <- data.frame(bank = label, learner = rownames(cf), a_2pl = cf[, "a"], b_2pl = cf[, "b"],
                      a_general = cb[rownames(cf), "a1"],
                      a_group = apply(cb[rownames(cf), grep("^a[2-9]", colnames(cb)), drop = FALSE], 1, function(z) sum(z)),
                      d_bifactor = cb[rownames(cf), "d"], row.names = NULL)
  list(fit = rbind(fitrow(B1, "2PL, one factor"), fitrow(BB, "bifactor (general + linear, trees, local)")), items = items)
}
Yall <- flip_matrix(R); Yno <- Yall[!rownames(Yall) %in% drop_items, ]
F1 <- fit_flipped(Yall, "all task-items"); F2 <- fit_flipped(Yno, "without unanimous")
fit <- rbind(F1$fit, F2$fit); items <- merge(F1$items, F2$items, by = "learner", suffixes = c("_all", "_without"))
items <- items[, setdiff(names(items), c("bank_all", "bank_without"))]
write.csv(fit, file.path(OUT, paste0("unanimous_sensitivity_fit", SUF, ".csv")), row.names = FALSE)
write.csv(items, file.path(OUT, paste0("unanimous_sensitivity_flipped", SUF, ".csv")), row.names = FALSE)
print(fit, digits = 4, row.names = FALSE)
cat(sprintf("2PL slopes: Pearson %.3f, median ratio without/all %.3f; intercept-scale b: Pearson %.3f\n",
            cor(items$a_2pl_all, items$a_2pl_without), median(items$a_2pl_without / items$a_2pl_all),
            cor(items$b_2pl_all, items$b_2pl_without)))
cat(sprintf("bifactor general slopes: Pearson %.3f, median ratio %.3f\n",
            cor(items$a_general_all, items$a_general_without), median(items$a_general_without / items$a_general_all)))
print(items[order(-items$a_2pl_all), c("learner", "a_2pl_all", "a_2pl_without", "b_2pl_all", "b_2pl_without")],
      digits = 3, row.names = FALSE)

# ---- A. multilevel Rasch A1, as in 03_irt.R part A -------------------------------------------------------------
fit_A1 <- function(D, label) {
  D$lf <- interaction(D$learner, D$task_family, drop = TRUE, sep = "|")
  m <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) + (1 | lf),
               D, binomial)
  fe <- glmmTMB::fixef(m)$cond; il <- grep("^learner", names(fe)); se <- sqrt(diag(vcov(m)$cond))[il]
  v <- sapply(VarCorr(m)$cond, function(z) unname(attr(z, "stddev")))
  list(ab = data.frame(learner = sub("^learner", "", names(fe)[il]), theta = fe[il] - mean(fe[il]), se = se, row.names = NULL),
       sd = data.frame(bank = label, t(v), pdHess = m$sdr$pdHess))
}
A_all <- fit_A1(R, "all task-items"); A_no <- fit_A1(R[!R$item %in% drop_items, ], "without unanimous")
ab <- merge(A_all$ab, A_no$ab, by = "learner", suffixes = c("_all", "_without"))
write.csv(ab, file.path(OUT, "unanimous_sensitivity_A1.csv"), row.names = FALSE)
write.csv(rbind(A_all$sd, A_no$sd), file.path(OUT, "unanimous_sensitivity_A1_sd.csv"), row.names = FALSE)
cat(sprintf("A1 abilities: Pearson %.4f, Spearman %.3f, largest change %.3f logits; median SE ratio %.3f\n",
            cor(ab$theta_all, ab$theta_without), cor(ab$theta_all, ab$theta_without, method = "spearman"),
            max(abs(ab$theta_all - ab$theta_without)), median(ab$se_without / ab$se_all)))
print(rbind(A_all$sd, A_no$sd), digits = 3, row.names = FALSE)
