"""Verify metric B is being used for all paper-reported gaps (Step F)."""
import numpy as np
from paper.generator.run_registry import discover_runs, Run
from paper.generator.extract import load_convergence_json, get_final_values
import inspect
from paper.generator import extract as extract_module

# --- Find a two_players flat run ---
all_runs = discover_runs()
flat_run = None
for r in all_runs:
    if (r.method in ("TEL-PPO", "PPO") and not r.is_nested_format
            and r.experiment == "two_players" and r.q == 55.0 and r.seed == 42):
        flat_run = r
        break

assert flat_run is not None, "Cannot find two_players q=55 seed=42 run"
print(f"Using flat run: {flat_run.path}")
df = load_convergence_json(flat_run)

# Assertion 1: sample_effort_mean and policy_mean_effort are different columns
assert "sample_effort_mean" in df.columns
assert "policy_mean_effort" in df.columns
assert not np.allclose(
    df["sample_effort_mean"].values, df["policy_mean_effort"].values
), "sample and policy means should differ!"
print("Assertion 1 passed: sample and policy columns differ")

# Assertion 2: final gap uses metric B
final = get_final_values(df)
gap_b = abs(final["policy_mean_effort"].iloc[0] - final["theoretical_effort"].iloc[0])
# For q=55 seed=42, metric B gap should be ~0.36
# (from docs/round2_metric_decision.md Section 2)
assert abs(gap_b - 0.36) < 0.1, f"q=55 seed=42 gap B expected ~0.36, got {gap_b:.4f}"
print(f"Assertion 2 passed: gap_b = {gap_b:.4f} (expected ~0.36)")

# Assertion 3: effort_error in final uses policy_mean_effort (metric B)
effort_error = final["effort_error"].iloc[0]
assert abs(effort_error - gap_b) < 1e-6, \
    f"effort_error ({effort_error}) should match gap_b ({gap_b})"
print(f"Assertion 3 passed: effort_error matches gap_b")

# Assertion 4: flat loader raises on missing policy_mean_effort
src = inspect.getsource(extract_module)
assert "missing 'policy_mean_effort'" in src, \
    "Expected explicit raise for missing policy_mean_effort in flat loader"
print("Assertion 4 passed: flat loader has raise for missing policy_mean_effort")

# --- Nested format sanity check ---
nested_run = None
for r in all_runs:
    if r.is_nested_format and r.experiment == "different_cost" and r.seed == 42:
        nested_run = r
        break

if nested_run is None:
    # Try different_cost with any seed
    for r in all_runs:
        if r.is_nested_format and r.experiment == "different_cost":
            nested_run = r
            break

assert nested_run is not None, "Cannot find a nested-format different_cost run"
print(f"\nUsing nested run: {nested_run.path}")

df_nested = load_convergence_json(nested_run)
final_nested = get_final_values(df_nested)

assert "policy_mean_effort" in final_nested.columns
assert not final_nested["policy_mean_effort"].isna().any(), \
    "nested final policy_mean_effort should not be NaN"

# Nested sample_effort_mean should be NaN
assert final_nested["sample_effort_mean"].isna().all(), \
    "nested sample_effort_mean should be NaN"
print("Assertion 5 passed: nested format has non-NaN policy, NaN sample")

# Het cost gap should be in reasonable range (< 10 effort units)
gap_nested = abs(
    final_nested["policy_mean_effort"].iloc[0]
    - final_nested["theoretical_effort"].iloc[0]
)
assert gap_nested < 10, f"nested gap suspiciously large: {gap_nested}"
print(f"Assertion 6 passed: nested gap = {gap_nested:.3f} (< 10)")

print("\nAll metric B assertions passed.")
