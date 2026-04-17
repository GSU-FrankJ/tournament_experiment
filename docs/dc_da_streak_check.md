# dc/da Streak Reset Bug Check

## Question
Do `run_different_cost.py` and `run_different_ability.py` have the same streak reset bug as `run_three_players.py:1119` (resetting `exploit_ok_streak` on non-eval updates)?

## Answer: NO — neither has the bug.

### run_different_cost.py

`joint_exploit_ok_streak` is referenced at 8 locations:
- Line 574: init to 0
- Line 826: `+= 1` (eval passes)
- Line 828: `= 0` (eval fails)
- Lines 809, 821, 831, 835, 882, 920: logging/recording only

**No non-eval reset.** The streak only resets when an eval actually runs and fails (line 828). Non-eval updates leave the streak unchanged.

Eval trigger (line 767): `should_eval_exploit = gate_pass or (updates_since_exploit_eval >= exploit_every_updates)`. Periodic eval is a fallback that always fires, so evals happen at minimum every `exploit_every_updates` updates regardless of the gate.

### run_different_ability.py

Same pattern. `joint_exploit_ok_streak` only reset on eval failure (line 809). No non-eval reset. Eval trigger (line 747) identical to dc.

### Why 3P was different

The 3P runner was written separately (different author/time) and used a more aggressive convergence logic where the `else` branch (non-eval) reset everything including the streak. This was likely intended to ensure "consecutive" meant "without gaps," but in practice it made stopping impossible with periodic-only evals.

### Implication for batch plan

dc and da do NOT need the streak fix. Their stopping logic is already correct. If they have convergence issues, the root cause is elsewhere (early stopping, concentration ramp, or training dynamics).
