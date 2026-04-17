#!/bin/bash
# Round 3 monitor: writes status to results/round3_status.txt
# Usage: bash results/round3_monitor.sh

STATUS_FILE="results/round3_status.txt"

{
  echo "=== Round 3 Status — $(date) ==="
  echo ""

  # 3P runs
  echo "--- 3P Batch ---"
  for q in 35 55; do
    for seed in 42 43 44 45 46; do
      log="results/round2_conc_fix/log_3p_r3_q${q}_s${seed}.txt"
      json="results/three_players/convergence/ppo_3p_q${q}.0_seed${seed}_round3_baseline_convergence.json"

      if [ ! -f "$log" ]; then
        printf "  q=%s s=%s: NOT STARTED\n" "$q" "$seed"
        continue
      fi

      # Check if done
      final=$(grep 'Final q=' "$log" 2>/dev/null | tail -1)
      if [ -n "$final" ]; then
        # Extract gap from final line
        gap=$(echo "$final" | grep -oP 'abs_err=[\d.]+' | head -1)
        printf "  q=%s s=%s: DONE  %s\n" "$q" "$seed" "$gap"
      else
        # Still running — get latest update
        latest=$(grep 'update=' "$log" 2>/dev/null | tail -1)
        if [ -n "$latest" ]; then
          upd=$(echo "$latest" | grep -oP 'update=\d+' | head -1)
          effort=$(echo "$latest" | grep -oP 'policy_mean=[\d.]+' | head -1)
          err=$(echo "$latest" | grep -oP 'abs_err=[\d.]+' | head -1)
          printf "  q=%s s=%s: RUNNING  %s %s %s\n" "$q" "$seed" "$upd" "$effort" "$err"
        else
          printf "  q=%s s=%s: STARTING (no updates yet)\n" "$q" "$seed"
        fi
      fi
    done
  done

  echo ""
  echo "--- dc/da Diagnostic ---"
  for scenario in dc da; do
    log="results/round2_conc_fix/log_${scenario}_diag_q35_s42.txt"
    if [ ! -f "$log" ]; then
      printf "  %s: NOT STARTED\n" "$scenario"
    else
      final=$(grep 'Final' "$log" 2>/dev/null | tail -1)
      if [ -n "$final" ]; then
        printf "  %s: DONE  %s\n" "$scenario" "$final"
      else
        latest=$(grep 'update=' "$log" 2>/dev/null | tail -1)
        upd=$(echo "$latest" | grep -oP 'update=\d+' 2>/dev/null | head -1)
        printf "  %s: RUNNING  %s\n" "$scenario" "${upd:-starting}"
      fi
    fi
  done

  echo ""
  # Count completed
  done_3p=$(ls results/three_players/convergence/ppo_3p_*_round3_baseline_convergence.json 2>/dev/null | wc -l)
  echo "Completed: 3P=$done_3p/10"

  # Check for errors
  errors=$(grep -l 'Error\|Traceback\|OOM\|Killed' results/round2_conc_fix/log_3p_r3_*.txt results/round2_conc_fix/log_*_diag_*.txt 2>/dev/null)
  if [ -n "$errors" ]; then
    echo ""
    echo "!!! ERRORS DETECTED in: $errors"
  fi
} > "$STATUS_FILE"

cat "$STATUS_FILE"
