#!/bin/bash
# =============================================================================
# r6_floor_ablation — isolates the `--min-updates` floor and the concentration
# ramp for the three-player and heterogeneous-cost scenarios.
# PREPARED, NOT LAUNCHED: default mode is a dry run that prints the matrix;
# pass --launch to start (owner approval required).
#
# WHY THIS WAVE EXISTS
#   The shipped r5_sampled 3P/dc runs stop at update 300-309 with
#   joint_exploit_ok_streak = 73-208 (patience is 5). Their exploitability
#   condition was first satisfied at update ~48 (3P q=35), ~83 (3P q=55),
#   ~108 (dc q=35), ~126 (dc q=55). They did NOT stop because verification
#   fired -- they stopped because `--min-updates 300` expired. Those extra
#   ~200 updates are exactly the window in which the concentration ramp runs
#   (warmup 200 -> complete 250), so in the shipped data
#       "3P/dc trained longer"  and  "3P/dc got conc_min=1000 + var_coef=0.05"
#   are the same event and cannot be separated. 2P, which stops at 49-109,
#   never reaches the ramp: its realized conc_min is 100 and var_coef is 0
#   for the whole run, from the SAME config block.
#
#   This wave breaks that confound with two new arms that, together with the
#   existing r5_sampled runs, form a 2x2:
#
#                        | ramp NOT reached      | ramp applied
#     ---------------------------------------------------------------------
#     stop at verif.     | A  r6_nofloor  (new)  | (unreachable by design)
#     hold to update 300 | B  r6_noramp   (new)  | r5_sampled (on disk)
#
#   A answers "what does the unified 2P protocol give on 3P/dc?" (Q4/Q5).
#   B answers "is the variance penalty + conc floor doing the work, or is it
#   just the extra training?" (Q2). A alone changes two things at once, so B
#   is what makes A interpretable.
#
# ARMS
#   A  r6_nofloor  (default stage, 20 runs)
#        r5_sampled flags with `--min-updates 300` REMOVED. The runner default
#        is 0 = no floor (run_three_players.py:1727, run_different_cost.py:1286),
#        so the run stops the moment the 5-check exploitability streak
#        completes -- identical stopping rule to 2P.
#        Expected stop: ~48/~83 (3P q=35/55), ~108/~126 (dc q=35/55).
#        Realized ramp state at those updates: conc_min=100, var_coef=0.
#
#   B  r6_noramp   (opt-in via --include-noramp, 20 runs)
#        r5_sampled flags with `--override-conc-ramp-warmup 100000`. Warmup
#        beyond the 1500-update cap means ramp_t == 0 for the entire run
#        (run_two_players.py:868-880 logic, ported verbatim to 3P/dc), so
#        conc_min stays 100 and var_coef stays 0 while `--min-updates 300`
#        still holds the run to update 300+. Same length as r5_sampled,
#        no ramp.
#
# EVERYTHING ELSE IS BYTE-IDENTICAL to the canonical r5_sampled cmdlines
# recovered from results/r5_sampled/manifest.csv:
#   3P: --method ppo --theory-align-v2 --override-conc-ramp-warmup 200 \
#       --min-updates 300 --episodes 6144000 --ablation-name r5_sampled
#   dc: (same)
# Sampled training, eps_eq=0.03, patience=5, exploit_every_updates=10,
# episodes 6144000 = 1500-update cap, seeds 42-46, q in {35,55}.
#
# WHAT TO READ OUT (per arm x scenario x q, across the 5 seeds)
#   stopped_at_update, joint_exploit_ok_streak, stop_reason  <- did verification
#     actually fire, or did a floor expire?
#   final.policy_mean_effort  -> mean, sample SD, |mean - e*|
#   final_exploit_max
#   A vs r5_sampled  -> does the floor change the answer at all?
#   B vs r5_sampled  -> does the ramp change the answer at fixed length?
#
# SAFETY
#   - PPO outputs have NO runner-level overwrite guard (`--force` only guards
#     gradient results: run_three_players.py:1896, run_different_cost.py:1361).
#     This script refuses to launch if ANY target file already exists.
#   - Does not touch results/*/convergence/*r5_sampled* or any existing file.
#   - Launch uses the r5 worker pattern: one run per GPU, tmux sessions,
#     atomic flock queue claims, per-job logs.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry-run}"
INCLUDE_NORAMP=0
for a in "$@"; do [ "$a" = "--include-noramp" ] && INCLUDE_NORAMP=1; done

QS="35 55"
SEEDS="42 43 44 45 46"
NGPU="${NGPU:-8}"
EPISODES=6144000
WAVE_DIR="results/r6_floor_ablation"

THREEP_OUT="results/three_players/convergence"
DC_OUT="results/different_cost/convergence"

declare -a JOBS TARGETS
add_job() { JOBS+=("$1"); TARGETS+=("$2"); }

for q in $QS; do for s in $SEEDS; do
  # --- Arm A: r6_nofloor (no --min-updates) ---
  add_job "python3 run/run_three_players.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --episodes $EPISODES --ablation-name r6_nofloor" \
          "$THREEP_OUT/ppo_3p_q${q}.0_seed${s}_r6_nofloor_convergence.json"
  add_job "python3 run/run_different_cost.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 200 --episodes $EPISODES --ablation-name r6_nofloor" \
          "$DC_OUT/different_cost_ppo_q${q}.0_seed${s}_r6_nofloor_convergence.json"

  # --- Arm B: r6_noramp (floor kept, ramp pushed past the budget) ---
  if [ "$INCLUDE_NORAMP" = "1" ]; then
    add_job "python3 run/run_three_players.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 100000 --min-updates 300 --episodes $EPISODES --ablation-name r6_noramp" \
            "$THREEP_OUT/ppo_3p_q${q}.0_seed${s}_r6_noramp_convergence.json"
    add_job "python3 run/run_different_cost.py --method ppo --q $q --seed $s --theory-align-v2 --override-conc-ramp-warmup 100000 --min-updates 300 --episodes $EPISODES --ablation-name r6_noramp" \
            "$DC_OUT/different_cost_ppo_q${q}.0_seed${s}_r6_noramp_convergence.json"
  fi
done; done

echo "r6_floor_ablation: ${#JOBS[@]} runs planned (A r6_nofloor 20$([ "$INCLUDE_NORAMP" = "1" ] && echo ' + B r6_noramp 20'))"

# Preflight: refuse any pre-existing target (PPO runs have no overwrite guard)
COLLIDE=0
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then echo "REFUSING: target already exists: $t"; COLLIDE=1; fi
done
[ "$COLLIDE" = "1" ] && { echo "Aborting: resolve collisions first (rename/retag)."; exit 2; }

if [ "$MODE" != "--launch" ]; then
  echo "DRY RUN (pass --launch to start). Matrix:"
  for i in "${!JOBS[@]}"; do printf '  [%3d] %s\n        -> %s\n' "$i" "${JOBS[$i]}" "${TARGETS[$i]}"; done
  exit 0
fi

mkdir -p "$WAVE_DIR/logs"
printf '%s\n' "${JOBS[@]}" > "$WAVE_DIR/jobs.txt"
echo "ts,event,queue,idx,gpu,info,cmd" > "$WAVE_DIR/manifest.csv"
rm -f "$WAVE_DIR/.jobs.cursor"
for (( g=0; g<NGPU; g++ )); do
  tmux new-session -d -s "r6floor_gpu$g" \
    "bash -c '
      QUEUE=$WAVE_DIR/jobs.txt; LOCK=$WAVE_DIR/.jobs.lock; CURSOR=$WAVE_DIR/.jobs.cursor
      TOTAL=\$(wc -l < \$QUEUE)
      while true; do
        IDX=\$(flock \$LOCK bash -c \"i=\\\$(cat \$CURSOR 2>/dev/null || echo 0); echo \\\$((i+1)) > \$CURSOR; echo \\\$i\")
        [ \$IDX -ge \$TOTAL ] && break
        CMD=\$(sed -n \$((IDX+1))p \$QUEUE)
        echo \"\$(date -Is),START,jobs,\$IDX,gpu$g,\\\"\$CMD\\\"\" >> $WAVE_DIR/manifest.csv
        CUDA_VISIBLE_DEVICES=$g bash -c \"\$CMD\" > $WAVE_DIR/logs/job\$(printf %03d \$IDX)_gpu$g.log 2>&1
        echo \"\$(date -Is),END,jobs,\$IDX,gpu$g,rc=\$?,\\\"\$CMD\\\"\" >> $WAVE_DIR/manifest.csv
      done'"
done
echo "LAUNCHED: $NGPU workers (tmux r6floor_gpu0..$((NGPU-1))); status: $WAVE_DIR/manifest.csv"
