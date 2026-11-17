# 07e_tabpfn_recalibrated.R -- A recalibrated TabPFN for counts and spending (FLAGS.md item 21; the "own procedure"
# of the 16 November plan). TabPFN's predicted mean on these outcomes falls short of the training mean, in-sample as
# well as on test rows (R/07c). The rule, fixed before it was run and the same on every instance: fit TabPFN as the
# IRT run did (one member, raw outcome), predict the training rows in-sample, and multiply the test predictions by
# s = mean(y_train) / mean(in-sample predictions). Applied to count and spending outcomes only (non-negative, with
# means far from zero); binary and continuous outcomes are untouched. The rule was chosen after seeing the shortfall,
# so this is a check of a remedy, not a blind comparison.
# For each of the 26 count and spending instances of R/07_tabpfn_check.R: the test predictions (the cached raw1
# blocks, refitted where no cache exists), the in-sample predictions, s, and the IRT criteria for the rescaled
# predictions (calibration: predicted mean within 10% of the training mean; accuracy as in R/07). Resumable.
# Output: output/tabpfn_recalibrated.csv (no row-level data)
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/02_run.R")
OUT <- "output/tabpfn_recalibrated.csv"
CHK <- read.csv("output/tabpfn_check.csv"); R1 <- CHK[CHK$variant == "raw1", ]
R19 <- read.csv("output/responses_19.csv")
items <- R19[R19$learner == "tabpfn" & R19$criterion == "cal" & R19$type %in% c("count", "cost"), c("inst", "cond", "rep", "n", "type")]
items <- items[order(items$n), ]
done <- if (file.exists(OUT)) read.csv(OUT) else NULL
Sys.setenv(RETICULATE_PYTHON = Sys.getenv("RETICULATE_PYTHON", "/usr/local/bin/python3"))
py <- reticulate::import_from_path("tabpfn_check", path = normalizePath("py"))
dir.create("cache/tabpfn_check", recursive = TRUE, showWarnings = FALSE)
for (k in seq_len(nrow(items))) {
  it <- items[k, ]; ii <- it$inst
  if (!is.null(done) && ii %in% done$inst) next
  i <- it$cond; cond <- CL[[i]]; seed <- 100000 + 1000 * i + it$rep; set.seed(seed)
  inst <- switch(cond$source, sim = sim_instance(cond), plasmode = plasmode_instance(cond), real = real_instance(cond))
  dc <- drop_const(inst$xtr, inst$xte)
  set.seed(seed + match("tabpfn", LEARNERS_ALL)); s <- sample.int(1e6, 1)
  te <- py$fit_variant(dc$xtr, inst$ytr, dc$xte, s, n_est = 1L, log = FALSE, cache = file.path("cache/tabpfn_check", paste0(ii, "_raw1")))
  tr <- py$predict_train(dc$xtr, inst$ytr, s, cache = file.path("cache/tabpfn_check", paste0(ii, "_raw1train")))
  p <- as.numeric(te$mean); ptr <- as.numeric(tr$mean)
  ratio0 <- mean(p) / mean(inst$ytr); stopifnot(abs(ratio0 - R1$ratio[R1$inst == ii]) < 1e-6)   # the refit reproduces raw1
  sc <- mean(inst$ytr) / mean(ptr); pr <- sc * p; ratio <- mean(pr) / mean(inst$ytr)
  if (!is.null(inst$truth)) { acc <- signal_r2(pr, inst$truth); acc_pass <- as.integer(acc >= 0.8); best <- NA
  } else {
    m <- switch(inst$type, count = d2(inst$yte, pr), r2(inst$yte, pr)); tol <- if (inst$type == "cost") 0.02 else 0.01
    best <- R1$best[R1$inst == ii]; acc <- m - best; acc_pass <- as.integer(m >= best - tol)
  }
  row <- data.frame(inst = ii, cond = i, n = it$n, type = inst$type, source = cond$source, scale = sc,
                    ratio_raw = ratio0, ratio_insample = mean(ptr) / mean(inst$ytr), ratio = ratio,
                    cal_pass = as.integer(abs(ratio - 1) <= 0.10), acc = acc, acc_pass = acc_pass,
                    cal_pass_raw = R1$cal_pass[R1$inst == ii], acc_pass_raw = R1$acc_pass[R1$inst == ii],
                    secs = round(te$secs + tr$secs, 1))
  write.table(row, OUT, sep = ",", append = file.exists(OUT), col.names = !file.exists(OUT), row.names = FALSE)
  message(sprintf("done %s scale %.3f ratio %.3f -> %.3f %s", ii, sc, ratio0, ratio, format(Sys.time())))
}
message("recalibration finished")
