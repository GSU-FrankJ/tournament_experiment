#!/bin/bash
# =============================================================================
# r8_unified — completes the SINGLE-CONFIGURATION policy across all four
# scenarios (owner directive 2026-08-01: one policy head, one entropy/conc/var
# preset, one verifier, one stopping principle — no per-scenario selection).
#
# Unified configuration (identical for all four scenarios):
#   state          s_i = [q/60, k_i/1e-3, Δw/10, (l_i − l̄_{−i})/10]  (4-dim)
#   policy head    mean × concentration (theory-align-v2)
#   entropy        0 for the entire run (v2 preset)
#   conc/var       one preset schedule: warmup 200, ramp 50,
#                  conc_min 100→1000, conc_scale 100→10000, var_coef 0→0.05
#   verifier       M=16384, eps=0.03, patience=5, every 10 updates
#   stopping       verification-triggered stop, NO minimum-update floor,
#                  budget cap 1500 updates
#
# The two-player group (Set1/Set2/no_stab/no_exploit) already trained under
# exactly this configuration in the r7 wave (2P has no floor flag at all), so
# it is REUSED, not rerun. This wave adds the three groups whose r5/r7 runs
# carried scenario-specific deviations:
#   3P  : r7 had --min-updates 300      -> dropped here
#   dc  : r7 had --min-updates 300      -> dropped here
#   da  : r7 shipped std head + entropy schedule + --min-updates 1000
#         -> v2 head, entropy 0, no floor here
#
# NOTE the honest consequence, to be stated in Table 2: with one preset and
# verification-triggered stopping, the REALIZED late-schedule values depend
# only on when each scenario verifies (3P/dc likely stop pre-ramp like 2P;
# da stops post-ramp). Same preset, stop-time-dependent realization.
#
# 30 runs ≈ 36 GPU-h ≈ 6.5 h on 8 V100s. Default: dry run; --launch to start.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry-run}"
NGPU="${NGPU:-8}"
PY="$PWD/.venv/bin/python"
EP=6144000
WAVE_DIR="results/r8_unified"
SEEDS="42 43 44 45 46"

declare -a JOBS TARGETS
add() { JOBS+=("$1"); TARGETS+=("$2"); }

# LPT: da (~2.5-3.2 h) first, then 3P (~0.3-0.5 h), dc (~0.4-0.5 h)
for q in 35 55; do for s in $SEEDS; do
  add "$PY run/run_different_ability.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --episodes $EP --ablation-name r8_unified" \
      "results/different_ability/convergence/different_ability_ppo_q${q}.0_seed${s}_r8_unified_convergence.json"
done; done
for q in 35 55; do for s in $SEEDS; do
  add "$PY run/run_three_players.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --episodes $EP --ablation-name r8_unified" \
      "results/three_players/convergence/ppo_3p_q${q}.0_seed${s}_r8_unified_convergence.json"
done; done
for q in 35 55; do for s in $SEEDS; do
  add "$PY run/run_different_cost.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --episodes $EP --ablation-name r8_unified" \
      "results/different_cost/convergence/different_cost_ppo_q${q}.0_seed${s}_r8_unified_convergence.json"
done; done

echo "r8_unified wave: ${#JOBS[@]} runs planned"

COLLIDE=0
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then echo "REFUSING: target already exists: $t"; COLLIDE=1; fi
done
[ "$COLLIDE" = "1" ] && { echo "Aborting: resolve collisions first (rename/retag)."; exit 2; }

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
  tmux new-session -d -s "r8w_gpu$g" \
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
echo "LAUNCHED: $NGPU workers (tmux r8w_gpu0..$((NGPU-1)))"
echo "status:   $WAVE_DIR/manifest.csv   logs: $WAVE_DIR/logs/"
