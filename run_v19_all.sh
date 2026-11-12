#!/bin/bash
# run_v19_all.sh TAG -- the two v19 queues side by side (see run_v19_queue.sh); rerun with a new TAG to resume
cd "$(dirname "$0")"
SIM=binary,cost,count,dml,import,lin,missing,nonlin,plasmode,procova,shift,sparse
./run_v19_queue.sh "${1}s" "real|1|1|0|Inf" "lin,nonlin|1|1|0|Inf" "missing,plasmode,shift,sparse|1|1|0|Inf" &
./run_v19_queue.sh "${1}f" "dml,procova,import||1|0|Inf" "$SIM|2+|1|0|Inf" "real|2+|1|0|Inf" &
wait
