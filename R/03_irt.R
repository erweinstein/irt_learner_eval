# 03_irt.R -- Item response models for the learners' responses to the task bank, both ways round.
#
#   A. Learners as examinees, tasks as items (the usual framing in machine learning; Martinez-Plumed et al.
#      2016). With 14 examinees, item slopes cannot be estimated, so this side uses the multilevel Rasch
#      model of Bock and Gibbons (2021, sec. 11.5) as a mixed-effects logistic regression: learner
#      "abilities" are fixed effects, and task difficulty is decomposed over the task bank's nesting
#      (task-items in instances in conditions in families). A random learner-by-family interaction tests
#      whether one ability is enough (a learner may be good at some kinds of task and not others).
#   B. Flipped (Bock and Gibbons 2021, sec. 13.4: "it would be possible to treat the classifiers as the items
#      and the instances as subjects"): the learners are items and the task-items are respondents, so each
#      learner gets an intercept (how often it succeeds on a task of average easiness) and a slope
#      (discrimination: how sharply its success depends on the task's easiness). Fitted with mirt: the
#      unidimensional 2PL, exploratory two- and three-factor models, a confirmatory bifactor model (a general
#      factor plus one group factor per learner family; Gibbons and Hedeker 1992), a graded version on three
#      categories, a multiple-group model for differential functioning between simulated and real-data tasks
#      (Bock and Gibbons ch. 9), and a two-level version with a random effect for the task condition.
#
# Input: output/responses.csv (from 02_run.R). Output: output/irt_*.csv, output/fig_*.png and
# output/irt_results.rds.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages({ library(glmmTMB); library(mirt); library(ggplot2) })
set.seed(2026)
# IRT_RESPONSES and IRT_OUT change the input and the output folder (the 19-learner run:
# IRT_RESPONSES=output/responses_19.csv IRT_OUT=output/v19)
OUT <- Sys.getenv("IRT_OUT", "output"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses.csv"), stringsAsFactors = FALSE)
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
out <- list()

# ---- descriptive: proportion correct ----------------------------------------------------------------------------
pc <- aggregate(pass ~ learner, R, mean); names(pc)[2] <- "prop_correct"
pc_fam <- aggregate(pass ~ learner + family, R, mean)
out$prop_correct <- pc; write.csv(pc, file.path(OUT, "irt_prop_correct.csv"), row.names = FALSE)
write.csv(reshape(pc_fam, idvar = "learner", timevar = "family", direction = "wide"), file.path(OUT, "irt_prop_correct_by_family.csv"), row.names = FALSE)

# ---- A. learners as examinees: multilevel Rasch as a GLMM ---------------------------------------------------
# Level 1 (response of learner l to task-item i): logit P(Y_li = 1) = theta_l - b_i
# Level 2 (task-item i in instance j):  b_i = pi_0j + r_i,          r_i ~ N(0, s2_item)
# Level 3 (instance j in condition k):  pi_0j = beta_0k + u_j,      u_j ~ N(0, s2_inst)
# Level 4 (condition k in family f):    beta_0k = gamma_f + v_k,    v_k ~ N(0, s2_cond), gamma_f ~ N(0, s2_fam)
# Combined: logit P(Y_li = 1) = theta_l - gamma_f - v_k - u_j - r_i  (+ w_lf, a learner-by-family interaction)
# lme4's Laplace fit stopped with "Downdated VtV is not positive definite" on these data, so glmmTMB (also a
# Laplace approximation) fits the same models
R$lf <- interaction(R$learner, R$task_family, drop = TRUE, sep = "|")
A0 <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item), R, binomial)
A1 <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) + (1 | lf), R, binomial)
vc <- function(m) { v <- VarCorr(m)$cond; sapply(v, function(z) unname(attr(z, "stddev"))^2) }
v0 <- vc(A0); v1 <- vc(A1)
out$A_varcomp <- rbind(rasch = v0[c("task_family", "cond", "inst", "item")][c(1:4)], rasch_lxf = v1[c("task_family", "cond", "inst", "item")])
out$A_varcomp <- cbind(out$A_varcomp, learner_x_family = c(NA, v1[["lf"]]))
out$A_compare <- data.frame(model = c("Rasch, multilevel tasks", "+ learner x family"), AIC = c(AIC(A0), AIC(A1)),
                            BIC = c(BIC(A0), BIC(A1)), logLik = c(logLik(A0), logLik(A1)))
out$A_lrt <- anova(A0, A1)
fe <- glmmTMB::fixef(A1)$cond; V <- vcov(A1)$cond; il <- grep("^learner", names(fe))
ab <- fe[il]; se <- sqrt(diag(V))[il]
out$A_ability <- data.frame(learner = sub("^learner", "", names(ab)), theta = ab - mean(ab), se = se, row.names = NULL)
out$A_criterion <- fe[-il]
re <- glmmTMB::ranef(A1)$cond$lf; key <- rownames(re)
out$A_profile <- data.frame(learner = sub("\\|.*", "", key), task_family = sub("^[^|]*\\|", "", key), w = re[, 1])
write.csv(out$A_ability, file.path(OUT, "irt_A_ability.csv"), row.names = FALSE)
write.csv(out$A_profile, file.path(OUT, "irt_A_profile.csv"), row.names = FALSE)

# ---- B. flipped: learners as items --------------------------------------------------------------------------
W <- reshape(R[, c("item", "learner", "pass")], idvar = "item", timevar = "learner", direction = "wide")
names(W) <- sub("^pass\\.", "", names(W)); rownames(W) <- W$item
Y <- W[, intersect(LEARNERS, names(W))]
Y <- Y[, apply(Y, 2, function(z) length(unique(na.omit(z))) > 1)]       # a learner must both pass and fail somewhere
ITEMS <- colnames(Y)
meta <- unique(R[, c("item", "inst", "cond", "family", "task_family", "source", "type", "n", "detail", "criterion")])
meta <- meta[match(rownames(Y), meta$item), ]
keep <- rowSums(!is.na(Y)) >= 4; Y <- Y[keep, ]; meta <- meta[keep, ]
out$B_n <- c(respondents = nrow(Y), items = ncol(Y), missing = mean(is.na(Y)))

B1 <- mirt(Y, 1, itemtype = "2PL", verbose = FALSE, SE = TRUE)
B2 <- mirt(Y, 2, itemtype = "2PL", verbose = FALSE)
B3 <- mirt(Y, 3, itemtype = "2PL", verbose = FALSE, method = "QMCEM")
spec <- c(linear = 1, trees = 2, local = 3, ensemble = NA, pretrained = NA)[fam_of[ITEMS]]
BB <- bfactor(Y, spec, verbose = FALSE, SE = TRUE)
fitstat <- function(m, nm) { M2 <- tryCatch(M2(m, na.rm = TRUE), error = function(e) NULL)
  data.frame(model = nm, logLik = extract.mirt(m, "logLik"), AIC = extract.mirt(m, "AIC"), BIC = extract.mirt(m, "BIC"),
             M2 = if (is.null(M2)) NA else M2$M2, df = if (is.null(M2)) NA else M2$df, RMSEA = if (is.null(M2)) NA else M2$RMSEA) }
out$B_fit <- rbind(fitstat(B1, "2PL, one factor"), fitstat(B2, "2PL, two factors (exploratory)"),
                   fitstat(B3, "2PL, three factors (exploratory)"), fitstat(BB, "bifactor (general + linear, trees, local)"))
cf <- coef(B1, IRTpars = TRUE, simplify = TRUE)$items
cf_se <- coef(B1, IRTpars = TRUE, printSE = TRUE)
out$B_items <- data.frame(learner = rownames(cf), family = fam_of[rownames(cf)], a = cf[, "a"], b = cf[, "b"],
                          a_se = sapply(rownames(cf), function(l) cf_se[[l]]["SE", "a"]),
                          b_se = sapply(rownames(cf), function(l) cf_se[[l]]["SE", "b"]),
                          prop_correct = colMeans(Y, na.rm = TRUE)[rownames(cf)])
out$B_loadings2 <- summary(B2, rotate = "oblimin", verbose = FALSE)
out$B_loadings3 <- summary(B3, rotate = "oblimin", verbose = FALSE)
out$B_bifactor <- summary(BB, verbose = FALSE)
write.csv(out$B_items, file.path(OUT, "irt_B_items_2pl.csv"), row.names = FALSE)
L2 <- out$B_loadings2$rotF; L3 <- out$B_loadings3$rotF; Lb <- out$B_bifactor$rotF
write.csv(data.frame(learner = rownames(L2), round(L2, 3), h2 = round(out$B_loadings2$h2, 3)), file.path(OUT, "irt_B_loadings2.csv"), row.names = FALSE)
write.csv(data.frame(learner = rownames(L3), round(L3, 3), h2 = round(out$B_loadings3$h2, 3)), file.path(OUT, "irt_B_loadings3.csv"), row.names = FALSE)
write.csv(data.frame(learner = rownames(Lb), round(Lb, 3), h2 = round(out$B_bifactor$h2, 3)), file.path(OUT, "irt_B_bifactor.csv"), row.names = FALSE)

# task scores: EAP easiness (one factor) and bifactor scores, related to the tasks' features
th1 <- fscores(B1, method = "EAP", full.scores.SE = TRUE)
thb <- fscores(BB, method = "EAP")
meta$theta <- th1[, 1]; meta$theta_se <- th1[, 2]
colnames(thb) <- c("G", "S_linear", "S_trees", "S_local"); meta <- cbind(meta, thb)
meta$logn <- log2(as.numeric(meta$n) / 500)
out$B_theta_by_family <- aggregate(cbind(theta, G, S_linear, S_trees, S_local) ~ task_family + criterion, meta, mean)
fo <- if (length(unique(meta$criterion)) > 1) theta ~ task_family + criterion + logn else theta ~ task_family + logn
out$B_theta_model <- summary(lm(fo, meta))$coefficients
write.csv(meta, file.path(OUT, "irt_B_task_scores.csv"), row.names = FALSE)
write.csv(out$B_theta_by_family, file.path(OUT, "irt_B_theta_by_family.csv"), row.names = FALSE)

# graded version: 0 = clearly fails, 1 = near the criterion, 2 = meets it (from the continuous scores)
grade <- function(r) { s <- r$score; c0 <- r$criterion
  ifelse(is.na(r$pass), NA, ifelse(r$pass == 1, 2,
    ifelse(c0 == "acc" & r$source != "real" & r$type == "binary", as.integer(s >= 0.75),
    ifelse(c0 == "acc" & r$source != "real", as.integer(s >= 0.6),
    ifelse(c0 == "acc", as.integer(s >= -0.03),
    ifelse(c0 == "cal" & r$type == "binary", as.integer(s >= 0.65 & s <= 1.5),
    ifelse(c0 == "cal", as.integer(abs(s - 1) <= 0.2),
    ifelse(c0 == "imp", as.integer(s >= 2), ifelse(c0 == "eff", as.integer(s <= 1.10), as.integer(abs(s) <= 2.58)))))))))) }
R$grade <- grade(R)
G <- reshape(R[, c("item", "learner", "grade")], idvar = "item", timevar = "learner", direction = "wide")
names(G) <- sub("^grade\\.", "", names(G)); rownames(G) <- G$item; G <- G[rownames(Y), ITEMS]
BG <- mirt(G, 1, itemtype = "graded", verbose = FALSE)
cg <- coef(BG, IRTpars = TRUE, simplify = TRUE)$items
out$B_graded <- data.frame(learner = rownames(cg), a = cg[, "a"], b1 = cg[, "b1"], b2 = cg[, "b2"])
write.csv(out$B_graded, file.path(OUT, "irt_B_graded.csv"), row.names = FALSE)

# differential functioning: simulated (incl. plasmode) vs real-data tasks, accuracy criterion only
acc <- meta$criterion == "acc"; grp <- ifelse(meta$source[acc] == "real", "real", "simulated")
if (length(unique(grp)) == 2) {
  Ya <- Y[acc, ]; Ya <- Ya[, sapply(split(seq_len(nrow(Ya)), grp), function(r) apply(Ya[r, , drop = FALSE], 2,
                                    function(z) length(unique(na.omit(z))) > 1)) |> apply(1, all)]
  MG <- multipleGroup(Ya, 1, group = grp, invariance = c("slopes", "intercepts", "free_means", "free_var"), verbose = FALSE)
  dif <- DIF(MG, which.par = c("a1", "d"), scheme = "drop", Wald = FALSE, verbose = FALSE, p.adjust = "holm")
  out$B_dif <- data.frame(learner = rownames(dif), dif, row.names = NULL)
  MGf <- multipleGroup(Ya, 1, group = grp, verbose = FALSE)
  out$B_dif_pars <- lapply(coef(MGf, IRTpars = TRUE, simplify = TRUE), function(z) z$items)
  write.csv(out$B_dif, file.path(OUT, "irt_B_dif.csv"), row.names = FALSE)
} else MGf <- NULL

# two-level flipped 2PL as a mixed model (Bock and Gibbons eqs. 11.60-11.61, lambda_i = X_i beta + X_i' T theta_i):
# each learner's slope is the loading of a one-dimensional reduced-rank random effect, at the level of the task
# condition (between conditions) and of the task-item (within conditions); glmmTMB's rr() terms. mirt's
# mixedmirt() with a condition-level random effect on theta stopped ("non-finite value supplied by 'nlm'").
R$condf <- factor(R$cond); R$itemf <- factor(R$item)
M1 <- glmmTMB(pass ~ 0 + learner + criterion + rr(0 + learner | itemf, d = 1), R, binomial)
M2 <- glmmTMB(pass ~ 0 + learner + criterion + rr(0 + learner | condf, d = 1) + rr(0 + learner | itemf, d = 1), R, binomial)
load1 <- function(S) { e <- eigen(S); l <- sqrt(max(e$values[1], 0)) * e$vectors[, 1]; if (sum(l) < 0) l <- -l
  setNames(l, sub("^learner", "", rownames(S))) }
vcm <- VarCorr(M2)$cond
out$B_multilevel <- data.frame(learner = names(load1(vcm$condf)), a_between_conditions = load1(vcm$condf),
                               a_within_conditions = load1(vcm$itemf)[names(load1(vcm$condf))],
                               intercept = glmmTMB::fixef(M2)$cond[paste0("learner", names(load1(vcm$condf)))], row.names = NULL)
out$B_multilevel_fit <- data.frame(model = c("2PL, task-items only", "2PL, conditions and task-items"),
  logLik = c(logLik(M1), logLik(M2)), AIC = c(AIC(M1), AIC(M2)), BIC = c(BIC(M1), BIC(M2)),
  converged = c(M1$fit$convergence == 0, M2$fit$convergence == 0), pdHess = c(M1$sdr$pdHess, M2$sdr$pdHess))
write.csv(out$B_multilevel, file.path(OUT, "irt_B_multilevel.csv"), row.names = FALSE)

# sensitivity: the three penalized regressions answer almost identically (local dependence inflates their
# slopes); refit the one-factor 2PL with lasso standing in for all three
Yr <- Y[, setdiff(ITEMS, c("ridge", "enet"))]
Br <- mirt(Yr, 1, itemtype = "2PL", verbose = FALSE)
cr <- coef(Br, IRTpars = TRUE, simplify = TRUE)$items
out$B_items_reduced <- data.frame(learner = rownames(cr), a = cr[, "a"], b = cr[, "b"])
write.csv(out$B_items_reduced, file.path(OUT, "irt_B_items_2pl_without_ridge_enet.csv"), row.names = FALSE)

saveRDS(list(out = out, A0 = A0, A1 = A1, B1 = B1, B2 = B2, B3 = B3, BB = BB, BG = BG, MG = MGf, M1 = M1, M2 = M2, Br = Br, meta = meta, Y = Y),
        file.path(OUT, "irt_results.rds"))

# ---- figures ----------------------------------------------------------------------------------------------
pal <- c(linear = "#2a6f97", trees = "#c1502e", local = "#6a994e", ensemble = "#7b2cbf")
th <- seq(-3, 3, length.out = 200)
icc <- do.call(rbind, lapply(ITEMS, function(l) data.frame(learner = l, family = fam_of[l], theta = th,
  p = plogis(out$B_items$a[out$B_items$learner == l] * (th - out$B_items$b[out$B_items$learner == l])))))
g1 <- ggplot(icc, aes(theta, p, group = learner, colour = family)) + geom_line(linewidth = 0.7) +
  ggrepel::geom_text_repel(data = data.frame(learner = out$B_items$learner, family = out$B_items$family, theta = out$B_items$b, p = 0.5),
    aes(label = learner), size = 3, show.legend = FALSE, min.segment.length = 0, seed = 1, max.overlaps = 30,
    nudge_y = rep(c(0.12, -0.12), length.out = nrow(out$B_items))) +
  scale_colour_manual(values = pal) +
  labs(x = expression("Task easiness " * theta * " (one-factor 2PL, learners as items)"), y = "Probability the learner meets the criterion",
       colour = NULL, title = "Learner characteristic curves", subtitle = "Labels at each learner's difficulty b (50% point)") + theme_minimal(base_size = 11) + theme(legend.position = "bottom")
ggsave(file.path(OUT, "fig_learner_curves.png"), g1, width = 7.5, height = 5, dpi = 200)
# empirical version (Bock and Gibbons Fig. 13.6): proportion correct by bins of the task's easiness
meta$bin <- cut(meta$theta, quantile(meta$theta, 0:6 / 6), include.lowest = TRUE)
emp <- do.call(rbind, lapply(ITEMS, function(l) { a <- aggregate(Y[[l]], list(bin = meta$bin), mean, na.rm = TRUE)
  data.frame(learner = l, family = fam_of[l], bin = a$bin, mid = tapply(meta$theta, meta$bin, median)[as.character(a$bin)], p = a$x) }))
emp$learner <- factor(emp$learner, levels = ITEMS)
g2 <- ggplot(emp, aes(mid, p, group = learner, colour = family)) + geom_line() + geom_point(size = 1) +
  facet_wrap(~ learner, ncol = 7) + scale_colour_manual(values = pal) +
  labs(x = "Task easiness (median of sextile)", y = "Proportion meeting the criterion", colour = NULL) +
  theme_minimal(base_size = 9) + theme(legend.position = "bottom")
ggsave(file.path(OUT, "fig_learner_curves_empirical.png"), g2, width = 9, height = 4, dpi = 200)
g3 <- ggplot(out$B_items, aes(b, a, colour = family, label = learner)) + geom_point(size = 2) +
  geom_errorbar(aes(ymin = a - 1.96 * a_se, ymax = a + 1.96 * a_se), width = 0, alpha = 0.4) +
  geom_errorbar(aes(xmin = b - 1.96 * b_se, xmax = b + 1.96 * b_se), width = 0, alpha = 0.4, orientation = "y") +
  geom_text(vjust = -0.8, size = 3, show.legend = FALSE) + scale_colour_manual(values = pal) +
  labs(x = "Difficulty b (task easiness needed for a 50% chance)", y = "Discrimination a", colour = NULL,
       title = "Learners as items: difficulty and discrimination (95% intervals)") + theme_minimal(base_size = 11)
ggsave(file.path(OUT, "fig_learner_a_b.png"), g3, width = 7.5, height = 5, dpi = 200)
prof <- out$A_profile; prof$learner <- factor(prof$learner, levels = LEARNERS)
g4 <- ggplot(prof, aes(task_family, learner, fill = w)) + geom_tile() +
  scale_fill_gradient2(low = "#b2182b", mid = "white", high = "#2166ac", name = "logit") +
  labs(x = NULL, y = NULL, title = "Learner-by-task-family deviations (multilevel Rasch, learners as examinees)") +
  theme_minimal(base_size = 10) + theme(axis.text.x = element_text(angle = 40, hjust = 1))
ggsave(file.path(OUT, "fig_learner_family_profile.png"), g4, width = 8.5, height = 5, dpi = 200)
ml <- out$B_multilevel; ml$family <- fam_of[ml$learner]
g5 <- ggplot(ml, aes(a_between_conditions, a_within_conditions, colour = family, label = learner)) + geom_point(size = 2) +
  geom_text(vjust = -0.8, size = 3, show.legend = FALSE) + scale_colour_manual(values = pal) + scale_y_sqrt(expand = expansion(mult = 0.08)) + scale_x_sqrt(expand = expansion(mult = 0.08)) +
  labs(x = "Discrimination between task conditions", y = "Discrimination within conditions (task-items)", colour = NULL,
       title = "Two-level 2PL, learners as items (square-root axes)") + theme_minimal(base_size = 11)
ggsave(file.path(OUT, "fig_two_level_discrimination.png"), g5, width = 7.5, height = 5, dpi = 200)
message("done")
