#!/bin/bash
# r5_sampled wave worker: claims jobs atomically from one or more queue files
# and runs them sequentially on the assigned GPU.
# Usage: worker.sh <GPU_ID> <queue1.txt> [queue2.txt ...]
set -u
GPU="$1"; shift
cd /home/fjiang4/tournament_experiment
MANIFEST=results/r5_sampled/manifest.csv

for QUEUE in "$@"; do
  QNAME=$(basename "$QUEUE" .txt)
  LOCK="results/r5_sampled/.${QNAME}.lock"
  CURSOR="results/r5_sampled/.${QNAME}.cursor"
  TOTAL=$(wc -l < "$QUEUE")
  while true; do
    IDX=$(flock "$LOCK" bash -c "i=\$(cat '$CURSOR' 2>/dev/null || echo 0); echo \$((i+1)) > '$CURSOR'; echo \$i")
    if [ "$IDX" -ge "$TOTAL" ]; then break; fi
    CMD=$(sed -n "$((IDX+1))p" "$QUEUE")
    LOGF="results/r5_sampled/logs/${QNAME}_$(printf %03d "$IDX")_gpu${GPU}.log"
    echo "$(date -Is),START,${QNAME},${IDX},gpu${GPU},\"${CMD}\"" >> "$MANIFEST"
    if [ "$QNAME" = "gradient_jobs" ]; then
      CUDA_VISIBLE_DEVICES="" bash -c "$CMD" > "$LOGF" 2>&1
    else
      CUDA_VISIBLE_DEVICES=$GPU bash -c "$CMD" > "$LOGF" 2>&1
    fi
    RC=$?
    echo "$(date -Is),END,${QNAME},${IDX},gpu${GPU},rc=${RC},\"${CMD}\"" >> "$MANIFEST"
  done
done
echo "$(date -Is),WORKER_DONE,gpu${GPU}" >> "$MANIFEST"
echo "[worker $GPU] all queues empty"
