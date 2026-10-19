#!/usr/bin/env bash
# full suite in 12 shards (round-robin over the test files), all concurrently, single-threaded each
cd /home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-audit-6f3a38
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=results/t3_r2/preflight/tests
mkdir -p $OUT
files=(tests/test_*.py)
n=12
for ((i=0;i<n;i++)); do
  sel=()
  for ((j=i;j<${#files[@]};j+=n)); do sel+=("${files[$j]}"); done
  ( echo "shard $i: ${sel[*]}"; date -u +"start %Y-%m-%dT%H:%M:%SZ"; /home/fjiang4/tournament_experiment/.venv/bin/python -m pytest "${sel[@]}" -p no:cacheprovider -q -rf 2>&1 | tail -25; echo "exit=${PIPESTATUS[0]}"; date -u +"end %Y-%m-%dT%H:%M:%SZ" ) > $OUT/shard_$(printf %02d $i).txt 2>&1 &
done
wait
echo all_done > $OUT/shards.done
