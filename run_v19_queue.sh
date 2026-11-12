#!/bin/bash
# run_v19_queue.sh TAG SPEC... -- runs v19 chunks in order, resumably, stopping after ~112 minutes (each chunk
# skips instances already saved in output/instances_v19). SPEC: families|replicates|cores|n_min|n_max.
# Two queues run side by side on one core each: one for the first replicates with RealMLP and TabPFN (TabPFN
# alone took 3.1 GB and 20 minutes on a 2,000-row real-data task, so two at once ran out of memory), one for
# the fast learners on the inference tasks and the later replicates.
cd "$(dirname "$0")"; TAG=$1; shift
END=$(( $(date +%s) + ${QUEUE_MIN:-112}*60 ))
for spec in "$@"; do
  IFS='|' read -r fam rep cores nmin nmax <<< "$spec"; left=$(( END - $(date +%s) ))
  [ $left -le 60 ] && { echo "queue stopped: time" >> logs/v19_queue_$TAG.log; exit 0; }
  echo "start [$spec] $(date)" >> logs/v19_queue_$TAG.log
  IRT_CORES=$cores IRT_N_MIN=$nmin IRT_N_MAX=$nmax timeout ${left}s ./run_v19.sh "$fam" "$rep" \
    "v19_${TAG}_$(echo ${fam:0:30} | tr ',' '_')_${rep:-all}_n${nmin}.log"
  echo "end [$spec] status $? $(date)" >> logs/v19_queue_$TAG.log
done
echo "queue finished" >> logs/v19_queue_$TAG.log
