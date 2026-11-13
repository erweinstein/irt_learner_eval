# 02g_merge_more.R -- Add the fast learners of 12-13 November (LightGBM and CatBoost, at their defaults and tuned,
# and BART; run with IRT_LEARNERS on the 10-replicate simulated tasks into output/instances_more_new) to the
# 14 learners' 10-replicate run (output/instances_more), writing output/instances_more_19. (RealMLP and TabPFN
# are too slow to run 10 times; the name keeps the "19" of the main run's other learners.)
# Then: IRT_INSTANCE_DIR=output/instances_more_19 IRT_RESPONSES=output/responses_more_19.csv Rscript R/02b_combine.R
#
# Eliot R. Weinstein
# eliotw@uchicago.edu
# erweinstein.com

unlink("output/instances_more_19", recursive = TRUE); dir.create("output/instances_more_19", showWarnings = FALSE)
f <- list.files("output/instances_more"); miss <- character(0)
for (x in f) {
  old <- readRDS(file.path("output/instances_more", x)); nf <- file.path("output/instances_more_new", x)
  if (!file.exists(nf)) { miss <- c(miss, x); saveRDS(old, file.path("output/instances_more_19", x)); next }
  new <- readRDS(nf); new <- new[!new$learner %in% old$learner, names(old)]
  saveRDS(rbind(old, new), file.path("output/instances_more_19", x))
}
cat(length(f) - length(miss), "of", length(f), "instances merged; without the new learners:", paste(miss, collapse = " "), "\n")
