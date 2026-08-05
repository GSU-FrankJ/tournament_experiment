#!/bin/bash
# =============================================================================
# r9_cert001 — adopts the tightened certificate (eps=0.01, M=65,536) as THE
# configuration for the full matrix (owner decision 2026-08-03, following the
# sensitivity result: err 3.22->1.12 / 3.42->0.71 on the two probe cells).
#
# Everything else is byte-identical to the unified r7/r8 cmdlines. 65 new runs:
#   2P Set1 15, Set2 15, no_stability 15, 3P q35 5, dc q35/55 10, da q35 5
# REUSED without rerun (documented in docs/tables34_raw_r7r8.md successor):
#   - 3P q55 & da q55: r8_sens_eps001 ARE this configuration (10 runs)
#   - no_exploit arm: verifier disabled, eps/M never enter training ->
#     r7_fig7_no_exploit remains valid bit-for-bit (15 runs)
#
# Prefix jobs are ordered longest-first (dc ~35 min, 2P q55/45 ~30 min, ...).
# Default: dry run. Pass --launch to start.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry-run}"
NGPU="${NGPU:-8}"
PY="$PWD/.venv/bin/python"
EP=6144000
CERT="--exploit-eps 0.01 --exploit-M 65536"
WAVE_DIR="results/r9_cert001"
SEEDS="42 43 44 45 46"

declare -a JOBS TARGETS
add() { JOBS+=("$1"); TARGETS+=("$2"); }

# 1) het-cost q35/55 (~35 min each; CPU verifier path)
for q in 35 55; do for s in $SEEDS; do
  add "$PY run/run_different_cost.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 $CERT --episodes $EP --ablation-name r9_cert001" \
      "results/different_cost/convergence/different_cost_ppo_q${q}.0_seed${s}_r9_cert001_convergence.json"
done; done

# 2) 2P families, q55 then q45 then q35 (longest first within each family)
for q in 55 45 35; do for s in $SEEDS; do
  add "$PY run/run_two_players.py --method ppo --q $q --seed $s --override-conc-ramp-warmup 200 $CERT --episodes $EP --ablation-name r9_cert001" \
      "results/two_players/convergence/ppo_q${q}.0_seed${s}_r9_cert001_convergence.json"
  add "$PY run/run_two_players.py --method ppo --q $q --seed $s --k 0.0006 --w_h 8 --w_l 4 --variant-name wh8_wl4 --override-conc-ramp-warmup 200 $CERT --episodes $EP --ablation-name r9_cert001" \
      "results/two_players/convergence/ppo_q${q}.0_seed${s}_wh8_wl4_r9_cert001_convergence.json"
  add "$PY run/run_two_players.py --method ppo --q $q --seed $s --disable-cheap-gate --override-conc-ramp-warmup 200 $CERT --episodes $EP --ablation-name r9_fig7_no_stability" \
      "results/two_players/convergence/ppo_q${q}.0_seed${s}_r9_fig7_no_stability_convergence.json"
done; done

# 3) 3P q35 (~23 min each)
for s in $SEEDS; do
  add "$PY run/run_three_players.py --method ppo --q 35 --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 $CERT --episodes $EP --ablation-name r9_cert001" \
      "results/three_players/convergence/ppo_3p_q35.0_seed${s}_r9_cert001_convergence.json"
done

# 4) da q35 (~11 min each)
for s in $SEEDS; do
  add "$PY run/run_different_ability.py --method ppo --q 35 --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 $CERT --episodes $EP --ablation-name r9_cert001" \
      "results/different_ability/convergence/different_ability_ppo_q35.0_seed${s}_r9_cert001_convergence.json"
done

echo "r9_cert001 wave: ${#JOBS[@]} runs planned"

COLLIDE=0
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then echo "REFUSING: target already exists: $t"; COLLIDE=1; fi
done
[ "$COLLIDE" = "1" ] && { echo "Aborting: resolve collisions first."; exit 2; }

if [ "$MODE" != "--launch" ]; then
  echo "DRY RUN (pass --launch to start). Matrix:"
  for i in "${!JOBS[@]}"; do printf '  [%2d] %s\n' "$i" "${JOBS[$i]}"; done
  exit 0
fi

mkdir -p "$WAVE_DIR/logs"
printf '%s\n' "${JOBS[@]}" > "$WAVE_DIR/jobs.txt"
echo "ts,event,queue,idx,gpu,info,cmd" > "$WAVE_DIR/manifest.csv"
rm -f "$WAVE_DIR/.jobs.cursor"
git rev-parse HEAD > "$WAVE_DIR/code_state.txt"
git status --short >> "$WAVE_DIR/code_state.txt"
git diff > "$WAVE_DIR/code_state.diff"

for (( g=0; g<NGPU; g++ )); do
  tmux new-session -d -s "r9w_gpu$g" \
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
echo "LAUNCHED: $NGPU workers (tmux r9w_gpu0..$((NGPU-1)))"
echo "status:   $WAVE_DIR/manifest.csv"
