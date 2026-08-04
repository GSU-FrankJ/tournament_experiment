#!/bin/bash
# =============================================================================
# r8_sens_eps001 — sensitivity analysis for the two wide-certificate cells
# (docs/Figures&Tables080226.docx red item: 3P q=55 and Het. ability q=55,
# rerun with a tighter certificate: eps=0.01, verifier M=65,536).
#
# Everything else is byte-identical to the r8_unified cmdlines (unified
# config: v2 head, entropy 0, 4-dim state, no min-updates floor, budget 1500).
# 10 runs (2 cells x 5 seeds). Expected: substantially longer training than
# r8 (eps 0.03 stops were 55-93 updates at exploit ~0.017/0.009); budget cap
# protects the worst case (~5-7 h/run).
#
# Default: dry run. Pass --launch to start.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry-run}"
NGPU="${NGPU:-8}"
PY="$PWD/.venv/bin/python"
EP=6144000
WAVE_DIR="results/r8_sens_eps001"
SEEDS="42 43 44 45 46"

declare -a JOBS TARGETS
add() { JOBS+=("$1"); TARGETS+=("$2"); }

for s in $SEEDS; do
  add "$PY run/run_different_ability.py --method ppo --q 55 --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --exploit-eps 0.01 --exploit-M 65536 --episodes $EP --ablation-name r8_sens_eps001" \
      "results/different_ability/convergence/different_ability_ppo_q55.0_seed${s}_r8_sens_eps001_convergence.json"
  add "$PY run/run_three_players.py --method ppo --q 55 --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --exploit-eps 0.01 --exploit-M 65536 --episodes $EP --ablation-name r8_sens_eps001" \
      "results/three_players/convergence/ppo_3p_q55.0_seed${s}_r8_sens_eps001_convergence.json"
done

echo "r8_sens_eps001 wave: ${#JOBS[@]} runs planned"

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
  tmux new-session -d -s "r8s_gpu$g" \
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
echo "LAUNCHED: $NGPU workers (tmux r8s_gpu0..$((NGPU-1)))"
echo "status:   $WAVE_DIR/manifest.csv"
