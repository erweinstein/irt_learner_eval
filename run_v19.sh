#!/bin/bash
# run_v19.sh FAMILIES REP_ONLY LOG -- one chunk of the 19-learner run (see R/02_run.R, IRT_PLAN = "v19"), or of
# another plan with PLAN set (PLAN=v21: the tuned LightGBM and CatBoost of 13 November, in output/instances_v21)
cd "$(dirname "$0")"; P=${PLAN:-v19}
LANG=C.UTF-8 LC_ALL=C.UTF-8 IRT_PLAN=$P IRT_INSTANCE_DIR=output/instances_$P IRT_FAMILIES="$1" IRT_REP_ONLY="$2" \
  Rscript R/02_run.R > "logs/$3" 2>&1
echo "EXIT $?" >> "logs/$3"
