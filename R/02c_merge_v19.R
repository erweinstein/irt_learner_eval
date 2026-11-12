# 02c_merge_v19.R -- Merge the 19-learner run (IRT_PLAN = "v19", output/instances_v19) with the original
# instances (output/instances) into output/instances_19: for real-data instances, whose accuracy is judged
# against the best learner, the v19 file replaces the original (it reran every learner, and its reference
# includes the new ones); for every other instance, the new learners' rows are added to the original 14.
# Then run R/02b_combine.R with IRT_INSTANCE_DIR=output/instances_19 IRT_RESPONSES=output/responses_19.csv.
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

source("R/01_tasks.R"); C <- conditions()
dir.create("output/instances_19", showWarnings = FALSE)
f <- list.files("output/instances"); miss <- 0
for (x in f) {
  old <- readRDS(file.path("output/instances", x)); nf <- file.path("output/instances_v19", x)
  if (!file.exists(nf)) { miss <- miss + 1; next }
  new <- readRDS(nf); i <- old$cond[1]
  out <- if (C$source[C$cond == i] == "real") new else rbind(old, new[, names(old)])
  saveRDS(out, file.path("output/instances_19", x))
}
cat(length(f) - miss, "instances merged;", miss, "not yet run\n")
