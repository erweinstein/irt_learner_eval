# 07f_tabpfn_recal_sensitivity.R -- TabPFN's place among the 21 learners when its count and spending responses are
# those of the recalibrated TabPFN of R/07e (FLAGS.md item 21). Replaces TabPFN's calibration and accuracy responses
# on the 26 count and spending instances with the recalibrated ones (output/tabpfn_recalibrated.csv), on spending only
# and then on spending and counts, and refits the multilevel Rasch model with the learner-by-family term (model A1
# of R/03_irt.R, as in R/07d and R/08). No other learner's responses change. The other learners get no comparable
# rescaling, so this asks how much of TabPFN's place comes from the shortfall in its means, not which learner is best.
# Input: output/responses_21.csv, output/tabpfn_recalibrated.csv. Output: output/v21/tabpfn_recal_sensitivity.csv
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
RC <- read.csv("output/tabpfn_recalibrated.csv")
tp <- R$learner == "tabpfn" & R$type %in% c("count", "cost") & R$criterion %in% c("cal", "acc")
stopifnot(setequal(unique(R$inst[tp]), RC$inst))
# the unscaled predictions reproduce the stored pass/fail
k <- match(R$inst[tp], RC$inst)
agree <- R$pass[tp] == ifelse(R$criterion[tp] == "cal", RC$cal_pass_raw[k], RC$acc_pass_raw[k])
message("stored responses reproduced: ", sum(agree), " of ", length(agree))
swap <- function(types) {
  p <- R$pass; i <- which(tp & R$type %in% types); k <- match(R$inst[i], RC$inst)
  p[i] <- ifelse(R$criterion[i] == "cal", RC$cal_pass[k], RC$acc_pass[k]); p
}
S <- list(baseline = R$pass, recal_spending = swap("cost"), recal_spending_counts = swap(c("cost", "count")))
out <- list()
for (s in names(S)) {
  D <- R; D$pass <- S[[s]]
  fit <- glmmTMB(pass ~ 0 + learner + criterion + (1 | task_family) + (1 | cond) + (1 | inst) + (1 | item) + (1 | lf), D, binomial)
  fe <- fixef(fit)$cond; fe <- fe[grepl("^learner", names(fe))]; names(fe) <- sub("^learner", "", names(fe))
  th <- fe - mean(fe)
  out[[s]] <- data.frame(scenario = s, learner = names(th), theta = unname(th), rank = rank(-th),
                         prop_met = unname(tapply(D$pass, D$learner, mean)[names(th)]), changed = sum(D$pass != R$pass))
  message(s, ": TabPFN theta ", round(th["tabpfn"], 2), ", rank ", rank(-th)["tabpfn"], " of ", length(th),
          " (", sum(D$pass != R$pass), " responses changed)")
}
O <- do.call(rbind, out)
write.csv(O, "output/v21/tabpfn_recal_sensitivity.csv", row.names = FALSE)
W <- reshape(O[, c("scenario", "learner", "theta", "rank")], idvar = "learner", timevar = "scenario", direction = "wide")
print(W[order(W$rank.baseline), ], digits = 3)
