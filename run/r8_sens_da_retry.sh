#!/bin/bash
# Retry of the 5 da jobs from r8_sens_eps001 that crashed at the first
# exploitability call (quadratic (M,M) broadcast OOM at M=65536, surfacing
# as an NVML assert on this host — fixed in utils/exploit_asymmetric.py by
# the linear-memory CPU evaluation branch for M > 16384).
# Cmdlines identical to run/r8_sens_eps001_wave.sh. Own wave dir/manifest.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="$PWD/.venv/bin/python"
EP=6144000
WAVE_DIR="results/r8_sens_eps001_da"
SEEDS="42 43 44 45 46"

declare -a JOBS TARGETS
for s in $SEEDS; do
  JOBS+=("$PY run/run_different_ability.py --method ppo --q 55 --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --exploit-eps 0.01 --exploit-M 65536 --episodes $EP --ablation-name r8_sens_eps001")
  TARGETS+=("results/different_ability/convergence/different_ability_ppo_q55.0_seed${s}_r8_sens_eps001_convergence.json")
done

for t in "${TARGETS[@]}"; do
  [ -e "$t" ] && { echo "REFUSING: exists: $t"; exit 2; }
done

mkdir -p "$WAVE_DIR/logs"
printf '%s\n' "${JOBS[@]}" > "$WAVE_DIR/jobs.txt"
echo "ts,event,queue,idx,gpu,info,cmd" > "$WAVE_DIR/manifest.csv"
rm -f "$WAVE_DIR/.jobs.cursor"
git diff > "$WAVE_DIR/code_state.diff"

# GPUs 2,4,6 are free plus whichever frees first; use 5 fixed free-ish GPUs
# (3P jobs occupy 0,1,3,5,7 right now) — workers 2,4,6 start immediately,
# 8elast two workers reuse 2,4 after... simpler: give the 5 jobs GPUs 2,4,6,2,4
# via a 3-worker queue (jobs are ~hours; 5 jobs on 3 GPUs is fine) — but
# cleanest is 5 workers pinned to 2,4,6 and the two that finish first pick up
# the queue remainder. Use 3 workers on the free GPUs.
for g in 2 4 6; do
  tmux new-session -d -s "r8sda_gpu$g" \
    "bash -c '
      QUEUE=$WAVE_DIR/jobs.txt; LOCK=$WAVE_DIR/.jobs.lock; CURSOR=$WAVE_DIR/.jobs.cursor
      TOTAL=\$(wc -l < \$QUEUE)
      while true; do
        IDX=\$(flock \$LOCK bash -c \"i=\\\$(cat \$CURSOR 2>/dev/null || echo 0); echo \\\$((i+1)) > \$CURSOR; echo \\\$i\")
        [ \$IDX -ge \$TOTAL ] && break
        CMD=\$(sed -n \$((IDX+1))p \$QUEUE)
        echo \"\$(date -Is),START,jobs,\$IDX,gpu$g,,\\\"\$CMD\\\"\" >> $WAVE_DIR/manifest.csv
        CUDA_VISIBLE_DEVICES=$g bash -c \"\$CMD\" > $WAVE_DIR/logs/job\$(printf %02d \$IDX)_gpu$g.log 2>&1
        RC=\$?
        echo \"\$(date -Is),END,jobs,\$IDX,gpu$g,rc=\$RC,\\\"\$CMD\\\"\" >> $WAVE_DIR/manifest.csv
      done'"
done
echo "LAUNCHED: 3 workers (tmux r8sda_gpu2/4/6) over 5 jobs"
