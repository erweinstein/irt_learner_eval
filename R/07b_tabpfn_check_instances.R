# 07b_tabpfn_check_instances.R -- What do the count and spending instances on which TabPFN falls short of the
# training mean have in common (FLAGS.md item 21)? Rebuilds each of the 26 instances of R/07_tabpfn_check.R exactly
# as the IRT run drew it (no fitting) and summarizes its training outcome: share of zeros, mean over median,
# skewness, largest value over the mean, variance over the mean, the largest standardized value, and (simulated
# instances) the share of the test outcome's variance that the true mean explains. Then relates them to TabPFN's
# ratio of predicted to training mean (output/tabpfn_check.csv) by Spearman correlation, within counts and
# spending. Output: output/tabpfn_check_instances.csv (one row per instance; summaries only, no row-level data)
# and output/tabpfn_check_correlates.csv.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/02_run.R")
CHK <- read.csv("output/tabpfn_check.csv")
R19 <- read.csv("output/responses_19.csv")
items <- R19[R19$learner == "tabpfn" & R19$criterion == "cal" & R19$type %in% c("count", "cost"), c("inst", "cond", "rep", "n", "type")]
skew <- function(y) mean((y - mean(y))^3) / sd(y)^3
rows <- lapply(seq_len(nrow(items)), function(k) {
  it <- items[k, ]; i <- it$cond; cond <- CL[[i]]
  seed <- 100000 + 1000 * i + it$rep; set.seed(seed)
  inst <- switch(cond$source, sim = sim_instance(cond), plasmode = plasmode_instance(cond), real = real_instance(cond))
  y <- inst$ytr
  data.frame(inst = it$inst, type = it$type, source = cond$source, n = length(y), p = ncol(drop_const(inst$xtr, inst$xte)$xtr),
             zero = mean(y == 0), mean_med = mean(y) / max(median(y), 1e-8), skew = skew(y), max_mean = max(y) / mean(y),
             disp = var(y) / mean(y), zmax = (max(y) - mean(y)) / sd(y), test_train = mean(inst$yte) / mean(y),
             signal = if (!is.null(inst$truth)) var(inst$truth) / var(inst$yte) else NA)
})
D <- do.call(rbind, rows)
for (v in c("raw1", "log1", "raw8", "log8")) D[[paste0("ratio_", v)]] <- CHK$ratio[CHK$variant == v][match(D$inst, CHK$inst[CHK$variant == v])]
# the largest value is one record's: left out of the written file for the instances built from real data (MEPS among
# them), and used only through the correlations below
Dw <- D; Dw[Dw$source != "sim", c("max_mean", "zmax")] <- NA
write.csv(Dw, "output/tabpfn_check_instances.csv", row.names = FALSE)
vars <- c("n", "p", "zero", "mean_med", "skew", "max_mean", "disp", "zmax", "signal")
cors <- do.call(rbind, lapply(c("count", "cost"), function(t) { d <- D[D$type == t, ]
  do.call(rbind, lapply(vars, function(x) { ok <- is.finite(d[[x]])
    data.frame(type = t, var = x, n_inst = sum(ok),
               rho_raw1 = if (sum(ok) > 3) cor(d[[x]][ok], d$ratio_raw1[ok], method = "spearman") else NA,
               rho_log_gain = if (sum(ok) > 3) cor(d[[x]][ok], (d$ratio_log1 - d$ratio_raw1)[ok], method = "spearman") else NA) })) }))
write.csv(cors, "output/tabpfn_check_correlates.csv", row.names = FALSE)
print(D, digits = 3); print(cors, digits = 2)
