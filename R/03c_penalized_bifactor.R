# 03c_penalized_bifactor.R -- The flipped bifactor model of R/03b_graded_bifactor.R (learners as items) with a ridge
# penalty on the slopes, as a check on the near-Heywood elastic net (communality 0.998; FLAGS.md item 24(c)) and the
# other near-duplicate penalized regressions. With 21 "items" and their loadings poorly determined, maximum likelihood
# pushes three nearly identical learners (lasso, ridge, elastic net) to communalities near 1 and extreme intercepts.
# A normal(0, 2) penalty on every slope (maximum a posteriori in mirt; equivalent to penalized ML with a ridge
# term) keeps them finite. This is reported beside the unpenalized fit, not in place of it: it is a different
# estimator, and the penalty's scale is a choice (2 is wide on the slope scale, where the other learners' slopes
# are 0.5-3).
# Output: output/v21/irt_B_bifactor_penalized.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(library(mirt))
OUT <- Sys.getenv("IRT_OUT", "output/v21"); PSD <- as.numeric(Sys.getenv("SLOPE_PRIOR_SD", "2"))
src <- readLines("R/03b_graded_bifactor.R")
stop_at <- grep("^fits <- list\\(\\); fitrows", src) - 1
eval(parse(text = src[grep("^suppressMessages", src):stop_at]))        # Y, G, spec, ITEMS, fam_of as in 03b
fit_pen <- function(D, it) {
  sv <- bfactor(D, spec, itemtype = it, pars = "values", verbose = FALSE)
  sl <- grepl("^a[0-9]", sv$name) & sv$est
  sv$prior.type[sl] <- "norm"; sv$prior_1[sl] <- 0; sv$prior_2[sl] <- PSD
  bfactor(D, spec, itemtype = it, verbose = FALSE, pars = sv)
}
loads <- function(m) { s <- summary(m, verbose = FALSE); data.frame(G = s$rotF[, 1], S = rowSums(s$rotF[, -1, drop = FALSE]), h2 = s$h2) }
rows <- list()
for (sc in c("binary", "graded")) {
  D <- if (sc == "binary") Y else G; it <- if (sc == "binary") "2PL" else "graded"
  m0 <- bfactor(D, spec, itemtype = it, verbose = FALSE); m1 <- fit_pen(D, it)
  l0 <- loads(m0); l1 <- loads(m1)
  f0 <- fscores(m0, method = "EAP", QMC = TRUE)[, 1]; f1 <- fscores(m1, method = "EAP", QMC = TRUE)[, 1]
  rows[[sc]] <- data.frame(scoring = sc, learner = ITEMS, family = fam_of[ITEMS],
    G_ml = round(l0$G, 3), S_ml = round(l0$S, 3), h2_ml = round(l0$h2, 3),
    G_pen = round(l1$G, 3), S_pen = round(l1$S, 3), h2_pen = round(l1$h2, 3),
    cor_general_scores = round(cor(f0, f1), 4), loglik_ml = extract.mirt(m0, "logLik"), loglik_pen = extract.mirt(m1, "logLik"))
  message(sc, ": general scores correlate ", round(cor(f0, f1), 4), "; loglik ", round(extract.mirt(m0, "logLik"), 1), " vs ", round(extract.mirt(m1, "logLik"), 1))
}
P <- do.call(rbind, rows)
write.csv(P, file.path(OUT, "irt_B_bifactor_penalized.csv"), row.names = FALSE)
print(P[P$learner %in% c("lasso", "ridge", "enet", "glm", "tabpfn", "stack"), ], row.names = FALSE)
cat("max |change in general loading| over the other learners:",
    round(max(abs((P$G_pen - P$G_ml)[!P$learner %in% c("lasso", "ridge", "enet")])), 3), "\n")
