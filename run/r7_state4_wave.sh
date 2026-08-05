#!/bin/bash
# =============================================================================
# r7_state4 wave — full baseline + Table-3 ablation rerun under the 4-dim state
#   s_i = [q/60, k_i/1e-3, (w_H − w_L)/10, (l_i − l̄_{−i})/10]
# and the unified training verifier M=16384 (config/one_stage_two_players.py).
#
# Owner request (docs/Figures&Tables073026.docx, red items 1–4):
#   1) 4-dim state so all scenarios share one input architecture
#   3) redo the ablation arms
#   4) training verifier 8192 -> 16384 (two-player was the only 8192)
#
# Cmdlines are byte-identical to the canonical r5 templates recovered from
# results/r5_sampled/manifest.csv, except:
#   - tag r5_* -> r7_*   (never overwrite r5: it is the 3-dim comparison arm)
#   - the 4-dim state + verifier M come from the code/config edits, not flags
#
# 100 runs, ~182 GPU-h, ~24 h on 8 V100s (LPT order):
#   15 no_exploit (4.9 h)  20 da std+v2 (3.5 h)  10 3P (1.5 h)  10 dc (1.1 h)
#   15 Set1 + 15 Set2 + 15 no_stability (~0.25 h)
#
# Default is a dry run printing the matrix; pass --launch to start.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry-run}"
NGPU="${NGPU:-8}"
PY="$PWD/.venv/bin/python"
EP=6144000
WAVE_DIR="results/r7_state4"
SEEDS="42 43 44 45 46"

declare -a JOBS TARGETS
add() { JOBS+=("$1"); TARGETS+=("$2"); }

# --- LPT order: longest jobs first ---------------------------------------
# 1) 2P no-exploit ablation (q 35/45/55, ~4.9 h each)
for q in 35 45 55; do for s in $SEEDS; do
  add "$PY run/run_two_players.py --method ppo --q $q --seed $s --disable-exploitability --override-conc-ramp-warmup 200 --episodes $EP --ablation-name r7_fig7_no_exploit" \
      "results/two_players/convergence/ppo_q${q}.0_seed${s}_r7_fig7_no_exploit_convergence.json"
done; done

# 2) het-ability std + v2 (q 35/55, ~3.5 h each)
for q in 35 55; do for s in $SEEDS; do
  add "$PY run/run_different_ability.py --method ppo --q $q --seed $s --min-updates 1000 --episodes $EP --ablation-name r7_state4_std" \
      "results/different_ability/convergence/different_ability_ppo_q${q}.0_seed${s}_r7_state4_std_convergence.json"
  add "$PY run/run_different_ability.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --min-updates 1000 --episodes $EP --ablation-name r7_state4_v2" \
      "results/different_ability/convergence/different_ability_ppo_q${q}.0_seed${s}_r7_state4_v2_convergence.json"
done; done

# 3) three-player (q 35/55, ~1.5 h each)
for q in 35 55; do for s in $SEEDS; do
  add "$PY run/run_three_players.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --min-updates 300 --episodes $EP --ablation-name r7_state4" \
      "results/three_players/convergence/ppo_3p_q${q}.0_seed${s}_r7_state4_convergence.json"
done; done

# 4) het-cost (q 35/55, ~1.1 h each)
for q in 35 55; do for s in $SEEDS; do
  add "$PY run/run_different_cost.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --min-updates 300 --episodes $EP --ablation-name r7_state4" \
      "results/different_cost/convergence/different_cost_ppo_q${q}.0_seed${s}_r7_state4_convergence.json"
done; done

# 5) 2P Set 1, Set 2, no-stability (q 35/45/55, ~0.25 h each)
for q in 35 45 55; do for s in $SEEDS; do
  add "$PY run/run_two_players.py --method ppo --q $q --seed $s --override-conc-ramp-warmup 200 --episodes $EP --ablation-name r7_state4" \
      "results/two_players/convergence/ppo_q${q}.0_seed${s}_r7_state4_convergence.json"
  add "$PY run/run_two_players.py --method ppo --q $q --seed $s --k 0.0006 --w_h 8 --w_l 4 --variant-name wh8_wl4 --override-conc-ramp-warmup 200 --episodes $EP --ablation-name r7_state4" \
      "results/two_players/convergence/ppo_q${q}.0_seed${s}_wh8_wl4_r7_state4_convergence.json"
  add "$PY run/run_two_players.py --method ppo --q $q --seed $s --disable-cheap-gate --override-conc-ramp-warmup 200 --episodes $EP --ablation-name r7_fig7_no_stability" \
      "results/two_players/convergence/ppo_q${q}.0_seed${s}_r7_fig7_no_stability_convergence.json"
done; done

echo "r7_state4 wave: ${#JOBS[@]} runs planned"

# Preflight: refuse any pre-existing target (PPO runs have no overwrite guard)
COLLIDE=0
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then echo "REFUSING: target already exists: $t"; COLLIDE=1; fi
done
[ "$COLLIDE" = "1" ] && { echo "Aborting: resolve collisions first (rename/retag)."; exit 2; }

if [ "$MODE" != "--launch" ]; then
  echo "DRY RUN (pass --launch to start). Matrix:"
  for i in "${!JOBS[@]}"; do printf '  [%3d] %s\n' "$i" "${JOBS[$i]}"; done
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
  tmux new-session -d -s "r7w_gpu$g" \
    "bash -c '
      QUEUE=$WAVE_DIR/jobs.txt; LOCK=$WAVE_DIR/.jobs.lock; CURSOR=$WAVE_DIR/.jobs.cursor
      TOTAL=\$(wc -l < \$QUEUE)
      while true; do
        IDX=\$(flock \$LOCK bash -c \"i=\\\$(cat \$CURSOR 2>/dev/null || echo 0); echo \\\$((i+1)) > \$CURSOR; echo \\\$i\")
        [ \$IDX -ge \$TOTAL ] && break
        CMD=\$(sed -n \$((IDX+1))p \$QUEUE)
        echo \"\$(date -Is),START,jobs,\$IDX,gpu$g,,\\\"\$CMD\\\"\" >> $WAVE_DIR/manifest.csv
        CUDA_VISIBLE_DEVICES=$g bash -c \"\$CMD\" > $WAVE_DIR/logs/job\$(printf %03d \$IDX)_gpu$g.log 2>&1
        RC=\$?
        echo \"\$(date -Is),END,jobs,\$IDX,gpu$g,rc=\$RC,\\\"\$CMD\\\"\" >> $WAVE_DIR/manifest.csv
      done'"
done
echo "LAUNCHED: $NGPU workers (tmux r7w_gpu0..$((NGPU-1)))"
echo "status:   $WAVE_DIR/manifest.csv   logs: $WAVE_DIR/logs/"
