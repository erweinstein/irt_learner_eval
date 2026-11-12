#!/bin/bash
# run_v19.sh FAMILIES REP_ONLY LOG -- one chunk of the 19-learner run (see R/02_run.R, IRT_PLAN = "v19")
cd "$(dirname "$0")"
LANG=C.UTF-8 LC_ALL=C.UTF-8 IRT_PLAN=v19 IRT_INSTANCE_DIR=output/instances_v19 IRT_FAMILIES="$1" IRT_REP_ONLY="$2" \
  Rscript R/02_run.R > "logs/$3" 2>&1
echo "EXIT $?" >> "logs/$3"
