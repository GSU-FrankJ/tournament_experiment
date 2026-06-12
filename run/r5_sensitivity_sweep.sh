#!/bin/bash
# =============================================================================
# r5_sensitivity sweep — refills paper/figures/hyperparam_sensitivity under
# SAMPLED training. PREPARED, NOT LAUNCHED: default mode is a dry run that
# prints the matrix; pass --launch to start (owner approval required).
#
# WHAT THE FIGURE CONSUMES (paper/generator/plots.py:plot_hyperparam_sensitivity):
#   - two_players TEL-PPO runs, weight_variant=baseline, q in {35,45,55}
#   - row 1: exploitability-tolerance sweep, ablation names EXACTLY
#       eps_001, eps_003, eps_010, eps_020   (legend: eps=0.01/0.03/0.10/0.20)
#   - row 2: verification-patience sweep, ablation names EXACTLY
#       pat_01, pat_03, pat_10               (legend: p=1/3/10)
#   - the black "baseline" curve = the promoted r5_sampled runs (already on
#     disk; NOT re-run here)
#   - per-seed traces + across-seed mean/CI are drawn automatically when a
#     variant has multiple seeds; series plotted = policy_mean_effort vs step.
#   Outputs slot straight into `python -m paper.generator make_all` once the
#   runs land — no plot-code changes needed.
#
# NOTE ON SCOPE: the owner's original request mentioned entropy_end/lr_end/
# clip_end variants; those belong to the figure's OLD design (the archived
# closed-form-era files in _archive_pre_warmup_fix/). The CURRENT figure
# (post paper-figures-tables-revision redesign) sweeps the VERIFICATION
# hyperparameters (eps_eq, patience) instead — so that grid is what slots in
# without changing the figure's format. The training-hparam grid is still
# available behind --include-training-hparams (it produces valid tagged runs
# but is NOT consumed by the current figure; a new plot would be needed).
#
# PRIMARY RUN MATRIX (105 runs; ~10-20 min each on 1 GPU; ~4-6 h on 8 GPUs):
#   variants x q in {35,45,55} x seeds {42..46}:
#     eps_001: --exploit-eps 0.01      -> ppo_q{Q}.0_seed{S}_eps_001_convergence.json
#     eps_003: --exploit-eps 0.03      -> ppo_q{Q}.0_seed{S}_eps_003_convergence.json
#              (duplicates the unified baseline gate; consistency arm)
#     eps_010: --exploit-eps 0.10      -> ppo_q{Q}.0_seed{S}_eps_010_convergence.json
#     eps_020: --exploit-eps 0.20      -> ppo_q{Q}.0_seed{S}_eps_020_convergence.json
#     pat_01:  --exploit-patience 1    -> ppo_q{Q}.0_seed{S}_pat_01_convergence.json
#     pat_03:  --exploit-patience 3    -> ppo_q{Q}.0_seed{S}_pat_03_convergence.json
#     pat_10:  --exploit-patience 10   -> ppo_q{Q}.0_seed{S}_pat_10_convergence.json
#   all under results/two_players/convergence/, each with a sibling
#   *_metadata.json (cmdline + git sha). Base flags identical to the canonical
#   r5_sampled A1 runs: sampled training, theory-align-v2 (PPO default),
#   --override-conc-ramp-warmup 200, --episodes 6144000 (max 1500 updates),
#   method's own stopping; --run-id r5_sensitivity for wave provenance.
#
# OPTIONAL STAGE (--include-training-hparams; 90 runs; NOT figure-consumed):
#   one mutually-exclusive override per run (runner enforces this):
#     --override-entropy-end {0.0005, 0.002}  -> variant entropy_end_{v}
#     --override-lr-end      {1e-05, 5e-05}   -> variant lr_end_{v}
#     --override-clip-end    {0.1, 0.2}       -> variant clip_end_{v}
#   x q {35,45,55} x seeds {42..46};
#   -> ppo_q{Q}.0_seed{S}_{variant}_convergence.json
#
# SAFETY:
#   - PPO outputs have no runner-level --force guard, so this script REFUSES
#     to launch if ANY target output file already exists (no overwrites).
#   - Launch uses the committed r5 worker/queue pattern (one run per GPU,
#     tmux, atomic claims): results/r5_sampled/worker.sh.
#   - After the wave: `python -m paper.generator make_all` refills the figure;
#     then run the before/after format check (same layout/axes/legend; only
#     data and the corrected baseline-gate label differ).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-dry-run}"
INCLUDE_HPARAMS=0
for a in "$@"; do [ "$a" = "--include-training-hparams" ] && INCLUDE_HPARAMS=1; done

QS="35 45 55"
SEEDS="42 43 44 45 46"
BASE="--method ppo --override-conc-ramp-warmup 200 --episodes 6144000 --run-id r5_sensitivity"
OUTDIR="results/two_players/convergence"
WAVE_DIR="results/r5_sensitivity"

declare -a JOBS TARGETS
add_job() { JOBS+=("$1"); TARGETS+=("$2"); }

for q in $QS; do for s in $SEEDS; do
  for pair in "eps_001:0.01" "eps_003:0.03" "eps_010:0.10" "eps_020:0.20"; do
    name="${pair%%:*}"; eps="${pair##*:}"
    add_job "python3 run/run_two_players.py $BASE --q $q --seed $s --exploit-eps $eps --ablation-name $name" \
            "$OUTDIR/ppo_q${q}.0_seed${s}_${name}_convergence.json"
  done
  for pair in "pat_01:1" "pat_03:3" "pat_10:10"; do
    name="${pair%%:*}"; pat="${pair##*:}"
    add_job "python3 run/run_two_players.py $BASE --q $q --seed $s --exploit-patience $pat --ablation-name $name" \
            "$OUTDIR/ppo_q${q}.0_seed${s}_${name}_convergence.json"
  done
  if [ "$INCLUDE_HPARAMS" = "1" ]; then
    for pair in "--override-entropy-end:0.0005" "--override-entropy-end:0.002" \
                "--override-lr-end:1e-05" "--override-lr-end:5e-05" \
                "--override-clip-end:0.1" "--override-clip-end:0.2"; do
      flag="${pair%%:*}"; val="${pair##*:}"
      vname="$(echo "${flag#--override-}" | tr '-' '_')_${val}"
      add_job "python3 run/run_two_players.py $BASE --q $q --seed $s $flag $val" \
              "$OUTDIR/ppo_q${q}.0_seed${s}_${vname}_convergence.json"
    done
  fi
done; done

echo "r5_sensitivity sweep: ${#JOBS[@]} runs planned (primary 105$([ "$INCLUDE_HPARAMS" = "1" ] && echo ' + training-hparams 90'))"

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
# Reuse the committed r5 worker (atomic queue claims, one run per GPU).
# Point it at this wave's queue/manifest via a thin wrapper per GPU.
for g in 0 1 2 3 4 5 6 7; do
  tmux new-session -d -s "r5sens_gpu$g" \
    "MANIFEST=$WAVE_DIR/manifest.csv bash -c '
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
echo "LAUNCHED: 8 workers (tmux r5sens_gpu0..7); status: $WAVE_DIR/manifest.csv"
