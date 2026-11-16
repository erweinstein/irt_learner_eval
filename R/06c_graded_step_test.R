# 06c_graded_step_test.R -- Tests the assumption behind the graded adaptive test of R/06_bifactor_cat.R that a
# learner's effect is the same at both steps of the continuation-ratio model (from "clearly fails" to "near or
# meets", and from "near" to "meets"). This is the analogue of the proportional odds assumption, and the test is
# the one Hedeker (2015, Prevention Science 16, 997-1006; Hedeker and Mermelstein 1998) describes: fit the model
# with and without step-specific learner effects and compare deviances by a likelihood ratio test. Both fits use
# all 21 learners and the calibration model of 06_bifactor_cat.R (task family, condition, instance and item
# random effects, plus learner-by-family and learner-by-condition), with step thresholds for each criterion.
# Also reports the intraclass correlations of the random effects on the logistic scale (level-1 variance pi^2/3).
# Input: output/responses_21.csv. Output: output/v21/graded_step_test.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(library(glmmTMB))
OUT <- Sys.getenv("IRT_OUT", "output/v21")
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_21.csv"), stringsAsFactors = FALSE)
R <- R[is.finite(R$pass), ]
R$item <- paste(R$inst, R$criterion, sep = ":")
R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$lf <- paste(R$learner, R$task_family); R$lc <- paste(R$learner, R$cond)
L <- readLines("R/03_irt.R")
eval(parse(text = L[grep("^grade <- function", L):(grep("^R\\$grade <- grade", L) - 1)]))
R$grade <- grade(R); R$grade[is.na(R$grade) & R$pass == 0] <- 0
S <- rbind(transform(R, y = as.integer(grade >= 1), step = "s1"),
           transform(R[R$grade >= 1, ], y = as.integer(grade == 2), step = "s2"))
S$crit <- paste(S$criterion, S$step); S$s2 <- as.integer(S$step == "s2")
S$learner <- factor(S$learner)
re <- "+ (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) + (1 | lf) + (1 | lc)"
ctl <- glmmTMBControl(rank_check = "adjust")
M0 <- glmmTMB(as.formula(paste("y ~ 0 + learner + crit", re)), S, binomial, control = ctl)
M1 <- glmmTMB(as.formula(paste("y ~ 0 + learner + crit + learner:s2", re)), S, binomial, control = ctl)
lrt <- anova(M0, M1); print(lrt)
fe0 <- fixef(M0)$cond; fe1 <- fixef(M1)$cond
b1 <- fe1[grepl("^learner[^:]*$", names(fe1))]; d2 <- fe1[grepl(":s2$", names(fe1))]
names(b1) <- sub("^learner", "", names(b1)); names(d2) <- sub("^learner(.*):s2$", "\\1", names(d2))
th0 <- fe0[grepl("^learner", names(fe0))]; names(th0) <- sub("^learner", "", names(th0))
lv <- names(th0)
step1 <- b1[lv]; step2 <- b1[lv] + ifelse(is.na(d2[lv]), 0, d2[lv])
cc <- function(x) x - mean(x)
T <- data.frame(learner = lv, theta_same = round(cc(th0), 3), theta_step1 = round(cc(step1), 3),
                theta_step2 = round(cc(step2), 3))
cat("correlation of the step-1 and step-2 learner effects:", round(cor(step1, step2), 3), "\n")
print(T, row.names = FALSE)
vc <- VarCorr(M0)$cond; v <- sapply(vc, function(m) attr(m, "stddev")^2)
icc <- v / (sum(v) + pi^2 / 3)
print(round(cbind(variance = v, icc = icc), 3))
out <- data.frame(T, deviance_same = -2 * as.numeric(logLik(M0)), deviance_step = -2 * as.numeric(logLik(M1)),
                  df = lrt$`Chi Df`[2], chisq = lrt$Chisq[2], p = lrt$`Pr(>Chisq)`[2],
                  cor_steps = cor(step1, step2))
write.csv(out, file.path(OUT, "graded_step_test.csv"), row.names = FALSE)
write.csv(data.frame(term = names(v), variance = v, icc = icc), file.path(OUT, "graded_icc.csv"), row.names = FALSE)
