# 07c_tabpfn_dist_check.R -- Where in its predictive distribution does TabPFN lose the mean on counts (FLAGS.md
# item 21)? On the count instances with 500 training rows and 10-17 features, refits the one-member TabPFN of the IRT
# run (same seed and settings as R/07_tabpfn_check.R's raw1) and compares, averaged over the first 2,000 test rows,
# its predictive distribution with the test outcomes: the predicted and observed share of zeros, and of values
# above the training 90th and 99th percentiles, and those values' contributions to the mean (E[Y 1(Y > t)]); the mean of the predicted medians; and the predicted mean over the
# test mean. The same for the training rows predicted in-sample, to see whether the shortfall is there too.
# Output: output/tabpfn_dist_check.csv (one row per instance and set; summaries only, no row-level data).
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/02_run.R")
OUT <- "output/tabpfn_dist_check.csv"
R19 <- read.csv("output/responses_19.csv")
items <- R19[R19$learner == "tabpfn" & R19$criterion == "cal" & R19$type == "count", c("inst", "cond", "rep", "n")]
INST <- strsplit(Sys.getenv("IRT_DIST_INST", "c034_r1,c036_r1,c038_r1,c040_r1,c070_r1,c074_r1"), ",")[[1]]
Sys.setenv(RETICULATE_PYTHON = Sys.getenv("RETICULATE_PYTHON", "/usr/local/bin/python3"))
py <- reticulate::import_from_path("tabpfn_dist_check", path = normalizePath("py"))
done <- if (file.exists(OUT)) read.csv(OUT)$inst else character(0)
for (ii in setdiff(INST, done)) {
  it <- items[items$inst == ii, ]; i <- it$cond; cond <- CL[[i]]
  seed <- 100000 + 1000 * i + it$rep; set.seed(seed)
  inst <- switch(cond$source, sim = sim_instance(cond), plasmode = plasmode_instance(cond), real = real_instance(cond))
  dc <- drop_const(inst$xtr, inst$xte)
  set.seed(seed + match("tabpfn", LEARNERS_ALL)); s <- sample.int(1e6, 1)
  th <- unname(quantile(inst$ytr, c(0.9, 0.99), type = 1))
  t0 <- Sys.time()
  r <- py$dist_check(dc$xtr, inst$ytr, dc$xte, s, th, max_test = 2000L)
  rows <- lapply(c("test", "train"), function(set) {
    z <- r[[set]]; y <- if (set == "test") inst$yte[seq_along(z$mean)] else inst$ytr
    data.frame(inst = ii, set = set, n_rows = length(y), mean_ratio = mean(z$mean) / mean(y),
               mean_vs_train = mean(z$mean) / mean(inst$ytr),
               zero_obs = mean(y == 0), zero_pred = mean(z$p0),
               above90_obs = mean(y > th[1]), above90_pred = mean(z$exceed0),
               above99_obs = mean(y > th[2]), above99_pred = mean(z$exceed1),
               pmean90_obs = mean(y * (y > th[1])), pmean90_pred = mean(z$pmean0),
               pmean99_obs = mean(y * (y > th[2])), pmean99_pred = mean(z$pmean1),
               median_pred = mean(z$median), median_obs = median(y), q90 = th[1], q99 = th[2],
               mean_obs = mean(y), mean_pred = mean(z$mean))
  })
  row <- do.call(rbind, rows); row$secs <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")))
  write.table(row, OUT, sep = ",", append = file.exists(OUT), col.names = !file.exists(OUT), row.names = FALSE)
  message(sprintf("done %s %s", ii, format(Sys.time())))
}
print(read.csv(OUT), digits = 3)
