#!/usr/bin/env bash
# T3-R3 P0.3: the two seed inventories (70501-70520, then 80501-80540), then the full hit listing.
# Run from the repository (worktree) root, inside tmux.
set -u
PY=/home/fjiang4/tournament_experiment/.venv/bin/python
OUT=results/t3_r3/preflight
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for BLOCK in "70501 70520" "80501 80540"; do
  TAG=$(echo "$BLOCK" | tr ' ' '_')
  {
    echo "\$ $PY tools/v2/seed_inventory.py --block $BLOCK --out $OUT/seed_inventory_${TAG}.csv"
    date -u +"start %Y-%m-%dT%H:%M:%SZ"
    "$PY" tools/v2/seed_inventory.py --block $BLOCK --out "$OUT/seed_inventory_${TAG}.csv"
    echo "exit=$?"
    date -u +"end %Y-%m-%dT%H:%M:%SZ"
  } > "$OUT/seed_inventory_${TAG}.out.txt" 2>&1
done

{
  echo "\$ $PY tools/t3/seed_hits.py --out-dir $OUT --block 70501 70520 --block 80501 80540"
  date -u +"start %Y-%m-%dT%H:%M:%SZ"
  "$PY" tools/t3/seed_hits.py --out-dir "$OUT" --block 70501 70520 --block 80501 80540
  echo "exit=$?"
  date -u +"end %Y-%m-%dT%H:%M:%SZ"
} > "$OUT/seed_hits.out.txt" 2>&1
echo done > "$OUT/preflight_seeds.done"
