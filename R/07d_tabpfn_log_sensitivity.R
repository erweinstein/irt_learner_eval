# 07d_tabpfn_log_sensitivity.R -- How much of TabPFN's place among the learners comes from its low means on spending
# and counts (FLAGS.md item 21)? Replaces TabPFN's calibration and accuracy responses on the count and spending
# instances with those of the one-member fit on log(1 + y) from R/07_tabpfn_check.R (output/tabpfn_check.csv), on
# spending only and then on spending and counts, and refits the multilevel Rasch model with the learner-by-family
# term (model A1 of R/03_irt.R, as in R/08_threshold_sensitivity.R) with all 21 learners. This is what TabPFN would
# have scored had it been run on the log scale for those outcomes; no other learner's responses change.
# Input: output/responses_21.csv, output/tabpfn_check.csv. Output: output/v21/tabpfn_log_sensitivity.csv
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

suppressMessages(library(glmmTMB))
R <- read.csv("output/responses_21.csv", stringsAsFactors = FALSE)
R <- R[!is.na(R$pass), ]
R$task_family <- ifelse(R$family == "real", paste0("real_", R$type), R$family)
R$item <- paste(R$inst, R$criterion, sep = ":")
R$lf <- paste(R$learner, R$task_family, sep = "|")
CHK <- read.csv("output/tabpfn_check.csv")
L1 <- CHK[CHK$variant == "log1", ]; R1 <- CHK[CHK$variant == "raw1", ]
tp <- R$learner == "tabpfn" & R$type %in% c("count", "cost") & R$criterion %in% c("cal", "acc")
stopifnot(setequal(unique(R$inst[tp]), L1$inst))
# the raw1 refits reproduce the stored pass/fail
k <- match(R$inst[tp], R1$inst)
stopifnot(all(R$pass[tp] == ifelse(R$criterion[tp] == "cal", R1$cal_pass[k], R1$acc_pass[k])))
swap <- function(types) {
  p <- R$pass; i <- which(tp & R$type %in% types); k <- match(R$inst[i], L1$inst)
  p[i] <- ifelse(R$criterion[i] == "cal", L1$cal_pass[k], L1$acc_pass[k]); p
}
S <- list(baseline = R$pass, log_spending = swap("cost"), log_spending_counts = swap(c("cost", "count")))
out <- list()
for (s in names(S)) {
  D <- R; D$pass <- S[[s]]
  fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) + (1 | lf), D, binomial)
  fe <- fixef(fit)$cond; fe <- fe[grepl("^learner", names(fe))]; names(fe) <- sub("^learner", "", names(fe))
  th <- fe - mean(fe)
  out[[s]] <- data.frame(scenario = s, learner = names(th), theta = unname(th), rank = rank(-th),
                         prop_met = unname(tapply(D$pass, D$learner, mean)[names(th)]), changed = sum(D$pass != R$pass))
  message(s, ": TabPFN theta ", round(th["tabpfn"], 2), ", rank ", rank(-th)["tabpfn"], " of ", length(th))
}
O <- do.call(rbind, out)
write.csv(O, "output/v21/tabpfn_log_sensitivity.csv", row.names = FALSE)
W <- reshape(O[, c("scenario", "learner", "theta", "rank")], idvar = "learner", timevar = "scenario", direction = "wide")
print(W[order(W$rank.baseline), ], digits = 3)
