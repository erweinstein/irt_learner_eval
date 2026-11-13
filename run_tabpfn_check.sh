#!/bin/bash
# run_tabpfn_check.sh TAG -- runs R/07_tabpfn_check.R (resumable) for at most ~112 minutes, with a memory watchdog
# (the session's limit is 5.8 GB; it stops the check if R and Python together pass 5.3 GB)
cd "$(dirname "$0")"; LOG=logs/tabpfn_check_$1.log
LANG=C.UTF-8 LC_ALL=C.UTF-8 timeout $(( ${QUEUE_MIN:-112} * 60 ))s Rscript R/07_tabpfn_check.R >> $LOG 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  used=$(ps -eo rss,cmd | grep -E "[e]xec/R|[p]ython" | awk '{s+=$1} END {print s*1024}')
  if [ -n "$used" ] && [ "$used" -gt 5300000000 ]; then pkill -P $P; kill $P; echo "killed for memory $(date) used $used" >> $LOG; fi
  sleep 5
done
wait $P; echo "EXIT $? $(date)" >> $LOG
