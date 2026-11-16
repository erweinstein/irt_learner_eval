# 03b_graded_bifactor.R -- The flipped models of R/03_irt.R (learners as items, task-items as respondents) with
# graded instead of pass/fail responses, after Gibbons et al. (2024), who found that scoring each item 0-3 instead
# of as three binary trials widened the range of thresholds, made the loadings more even and added precision at the
# low end. Responses: 0 = clearly fails, 1 = near the criterion, 2 = meets it, as in the graded model of 03_irt.R
# (here a calibration slope that cannot be computed counts as 0 instead of missing; 14 responses).
# For each scoring (binary, graded): one factor, two and three exploratory factors, and the bifactor model with the
# learner groups fixed in advance (general + linear, trees, local; the stack and TabPFN on the general factor only),
# compared by BIC within scoring, as Gibbons et al. compare a unidimensional model, the best exploratory one and the
# bifactor; then the bifactor loadings under both scorings, and the marginal reliability of the general score.
# Input: IRT_RESPONSES (default output/responses_21.csv). Output (IRT_OUT, default output/v21):
# irt_B_graded_fit.csv, irt_B_graded_bifactor.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(library(mirt))
OUT <- Sys.getenv("IRT_OUT", "output/v21")
R <- read.csv(Sys.getenv("IRT_RESPONSES", "output/responses_21.csv"), stringsAsFactors = FALSE)
R$item <- paste(R$inst, R$criterion, sep = ":")
L <- readLines("R/03_irt.R")                       # the learner list, families and graded scoring of 03_irt.R
eval(parse(text = L[grep("^LEARNERS <- c", L):(grep("^LEARNERS <- c", L) + 2)]))
eval(parse(text = L[grep("^fam_of <- c", L):(grep("^fam_of <- c", L) + 3)]))
eval(parse(text = L[grep("^grade <- function", L):(grep("^R\\$grade <- grade", L) - 1)]))
LEARNERS <- LEARNERS[LEARNERS %in% unique(R$learner)]
R$grade <- grade(R); R$grade[is.na(R$grade) & R$pass %in% 0] <- 0
wide <- function(v) { W <- reshape(R[, c("item", "learner", v)], idvar = "item", timevar = "learner", direction = "wide")
  names(W) <- sub(paste0("^", v, "\\."), "", names(W)); rownames(W) <- W$item; W[, intersect(LEARNERS, names(W))] }
Y <- wide("pass")
Y <- Y[, apply(Y, 2, function(z) length(unique(na.omit(z))) > 1)]       # as in 03_irt.R
ITEMS <- colnames(Y); keep <- rowSums(!is.na(Y)) >= 4; Y <- Y[keep, ]
G <- wide("grade")[rownames(Y), ITEMS]
spec <- c(linear = 1, trees = 2, local = 3, ensemble = NA, pretrained = NA)[fam_of[ITEMS]]
cat("task-items", nrow(Y), " learners", ncol(Y), "\n"); print(apply(G, 2, function(z) table(factor(z, 0:2))))

fits <- list(); fitrows <- list()
for (sc in c("binary", "graded")) {
  D <- if (sc == "binary") Y else G; it <- if (sc == "binary") "2PL" else "graded"
  m <- list(one = mirt(D, 1, itemtype = it, verbose = FALSE),
            two = mirt(D, 2, itemtype = it, verbose = FALSE),
            three = mirt(D, 3, itemtype = it, verbose = FALSE, method = "QMCEM"),
            bifactor = bfactor(D, spec, itemtype = it, verbose = FALSE))
  fits[[sc]] <- m
  for (nm in names(m)) {
    rel <- if (nm %in% c("one", "bifactor")) { f <- fscores(m[[nm]], method = "EAP", full.scores.SE = TRUE, QMC = nm == "bifactor")
      empirical_rxx(f)[1] } else NA
    fitrows[[length(fitrows) + 1]] <- data.frame(scoring = sc, model = nm, npar = extract.mirt(m[[nm]], "nest"),
      logLik = extract.mirt(m[[nm]], "logLik"), AIC = extract.mirt(m[[nm]], "AIC"), BIC = extract.mirt(m[[nm]], "BIC"),
      reliability_general = unname(rel))
  }
  message(sc, " done ", format(Sys.time()))
}
FIT <- do.call(rbind, fitrows); FIT$dBIC <- FIT$BIC - ave(FIT$BIC, FIT$scoring, FUN = min)
write.csv(FIT, file.path(OUT, "irt_B_graded_fit.csv"), row.names = FALSE)
print(FIT, digits = 6, row.names = FALSE)

lb <- summary(fits$binary$bifactor, verbose = FALSE); lg <- summary(fits$graded$bifactor, verbose = FALSE)
cg <- coef(fits$graded$bifactor, simplify = TRUE)$items; cb <- coef(fits$binary$bifactor, simplify = TRUE)$items
gb <- fscores(fits$binary$bifactor, method = "EAP", QMC = TRUE)[, 1]; gg <- fscores(fits$graded$bifactor, method = "EAP", QMC = TRUE)[, 1]
B <- data.frame(learner = ITEMS, family = fam_of[ITEMS],
                G_binary = round(lb$rotF[, 1], 3), S_binary = round(apply(lb$rotF[, -1, drop = FALSE], 1, sum), 3),
                G_graded = round(lg$rotF[, 1], 3), S_graded = round(apply(lg$rotF[, -1, drop = FALSE], 1, sum), 3),
                d_binary = round(cb[, "d"], 3), d1_graded = round(cg[, "d1"], 3), d2_graded = round(cg[, "d2"], 3))
write.csv(B, file.path(OUT, "irt_B_graded_bifactor.csv"), row.names = FALSE)
print(B, row.names = FALSE)
cat("correlation of the general scores (binary, graded):", round(cor(gb, gg), 3), "\n")
cat("range of loadings on G: binary", round(range(B$G_binary), 2), " graded", round(range(B$G_graded), 2), "\n")
cat("correlation of the loadings on G (binary, graded):", round(cor(B$G_binary, B$G_graded), 3), "\n")
cat("range of intercepts: binary", round(range(B$d_binary), 2), " graded", round(range(c(B$d1_graded, B$d2_graded)), 2), "\n")
