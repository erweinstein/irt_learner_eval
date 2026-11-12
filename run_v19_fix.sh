#!/bin/bash
# After the load_meps() fix (12 November 2026): two queues side by side, the large first draws one at a time
# (TabPFN at 2,000-3,000 training rows needs about 3.7 GB) and, on the other core, the original learners added to
# the plasmode instances that are on their own split, then the small MEPS-based draws.
cd "$(dirname "$0")"; TAG=$1
./run_v19_queue.sh "${TAG}s" "real,plasmode|1|1|501|Inf" "shift,missing|1|1|501|Inf" "lin,nonlin|1|1|0|Inf" &
( LANG=C.UTF-8 LC_ALL=C.UTF-8 IRT_PLAN=v19 IRT_INSTANCE_DIR=output/instances_v19 Rscript R/02e_add_original_plasmode.R > logs/v19_${TAG}_02e.log 2>&1
  ./run_v19_queue.sh "${TAG}t" "real|2+|1|0|500" "real,plasmode|1|1|0|500" ) &
wait
