# 07_tabpfn_check.R -- Why does TabPFN underpredict the mean of spending and counts (FLAGS.md item 21)? Refits
# TabPFN on the 26 count and spending instances it answered in the 19-learner run (first replicates), in four
# variants: one ensemble member or TabPFN's default of 8, on the outcome or on log1p(outcome) (py/tabpfn_check.py),
# and scores each variant with the IRT run's criteria (calibration in the large: predicted mean within 10% of the
# training mean; accuracy: signal R^2 >= 0.8 against the oracle, or on real data within the tolerance of the best
# of the 17 learners run on every instance, recovered from the stored scores).
# Resumable: each variant x instance is appended to output/tabpfn_check.csv as it finishes. The order is set so
# that a run stopped early still answers the main question: the one-member variants on the instances with up to
# 1,000 training rows first, then 8 members on those with up to 500 (an 8-member fit takes about 30-40 minutes
# there on one CPU thread, with 5,000 test rows), then the larger instances.
# IRT_CHECK_MAX_N (default Inf) skips instances with more training rows.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/02_run.R")
OUT <- "output/tabpfn_check.csv"
R19 <- read.csv("output/responses_19.csv")
items <- R19[R19$learner == "tabpfn" & R19$criterion == "cal" & R19$type %in% c("count", "cost"), c("inst", "cond", "rep", "n", "type", "score")]
acc0 <- R19[R19$learner == "tabpfn" & R19$criterion == "acc", c("inst", "score")]
items$acc_stored <- acc0$score[match(items$inst, acc0$inst)]
names(items)[names(items) == "score"] <- "cal_stored"
maxn <- as.numeric(Sys.getenv("IRT_CHECK_MAX_N", "Inf"))
ph <- function(sel, v) expand.grid(inst = items$inst[sel], variant = v, stringsAsFactors = FALSE)
plan <- rbind(ph(items$n <= 1000, c("raw1", "log1")),   # 1: the main comparison, smaller instances
              ph(items$n <= 500, "raw8"),               # 2: TabPFN's default ensemble (8 times as slow)
              ph(items$n <= 500, "log8"),
              ph(items$n > 1000, c("raw1", "log1")),    # 3: the larger instances
              ph(items$n > 500 & items$n <= 1000, "raw8"))
plan$phase <- rep(1:5, c(sum(items$n <= 1000) * 2, sum(items$n <= 500), sum(items$n <= 500), sum(items$n > 1000) * 2,
                         sum(items$n > 500 & items$n <= 1000)))
plan$n <- items$n[match(plan$inst, items$inst)]
plan <- plan[order(plan$phase, plan$n), ]
plan <- plan[plan$n <= maxn, ]
done <- if (file.exists(OUT)) read.csv(OUT) else NULL
Sys.setenv(RETICULATE_PYTHON = Sys.getenv("RETICULATE_PYTHON", "/usr/local/bin/python3"))
py <- reticulate::import_from_path("tabpfn_check", path = normalizePath("py"))
cache <- list()
for (k in seq_len(nrow(plan))) {
  ii <- plan$inst[k]; v <- plan$variant[k]
  if (!is.null(done) && any(done$inst == ii & done$variant == v)) next
  it <- items[items$inst == ii, ]; i <- it$cond; rep <- it$rep; cond <- CL[[i]]
  if (is.null(cache[[ii]])) {                       # the instance exactly as the IRT run drew it
    seed <- 100000 + 1000 * i + rep; set.seed(seed)
    inst <- switch(cond$source, sim = sim_instance(cond), plasmode = plasmode_instance(cond), real = real_instance(cond))
    dc <- drop_const(inst$xtr, inst$xte)
    set.seed(seed + match("tabpfn", LEARNERS_ALL)); s <- sample.int(1e6, 1)
    cache <- list(); cache[[ii]] <- list(inst = inst, xtr = dc$xtr, xte = dc$xte, s = s)
  }
  ci <- cache[[ii]]; inst <- ci$inst
  r <- py$fit_variant(ci$xtr, inst$ytr, ci$xte, ci$s, n_est = if (grepl("8", v)) 8L else 1L, log = grepl("log", v))
  p <- as.numeric(r$mean); ratio <- mean(p) / mean(inst$ytr)
  if (!is.null(inst$truth)) { acc <- signal_r2(p, inst$truth); acc_pass <- as.integer(acc >= 0.8); best <- NA
  } else {
    m <- switch(inst$type, count = d2(inst$yte, p), r2(inst$yte, p)); tol <- if (inst$type == "cost") 0.02 else 0.01
    # the reference: the stored score is m(as run) - best, and raw1 reproduces the run
    best <- if (v == "raw1") m - it$acc_stored else read.csv(OUT)$best[read.csv(OUT)$inst == ii & read.csv(OUT)$variant == "raw1"][1]
    acc <- m - best; acc_pass <- as.integer(m >= best - tol)
  }
  row <- data.frame(inst = ii, cond = i, n = it$n, p = ncol(ci$xtr), type = inst$type, source = cond$source,
                    variant = v, secs = round(r$secs, 1), ratio = ratio, cal_pass = as.integer(abs(ratio - 1) <= 0.10),
                    qmean_ratio = mean(as.numeric(r$qmean)) / mean(inst$ytr), acc = acc, acc_pass = acc_pass, best = best,
                    cal_stored = it$cal_stored, acc_stored = it$acc_stored, transforms = r$transforms)
  write.table(row, OUT, sep = ",", append = file.exists(OUT), col.names = !file.exists(OUT), row.names = FALSE)
  message(sprintf("done %s %s %.0fs ratio %.3f (stored %.3f) %s", ii, v, r$secs, ratio, it$cal_stored, format(Sys.time())))
}
message("check finished")
