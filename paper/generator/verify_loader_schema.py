"""Verify both loaders produce the same DataFrame schema (Step C)."""
import numpy as np
from paper.generator.run_registry import discover_runs

from paper.generator.extract import load_convergence_json

# Find one flat two_players run (where sample != policy) and one nested run
all_runs = discover_runs()
flat_run = None
nested_run = None
for r in all_runs:
    # Use two_players specifically — three_players has shared policy where
    # agent efforts == policy mean, so sample/policy columns are identical.
    if (r.method in ("TEL-PPO", "PPO") and not r.is_nested_format
            and r.experiment == "two_players" and flat_run is None):
        flat_run = r
    if r.is_nested_format and nested_run is None:
        nested_run = r
    if flat_run and nested_run:
        break

assert flat_run is not None, "No flat-format run found"
assert nested_run is not None, "No nested-format run found"
print(f"Flat run:   {flat_run.path}")
print(f"Nested run: {nested_run.path}")

df_flat = load_convergence_json(flat_run)
df_nested = load_convergence_json(nested_run)

# Assertion 1: both have the two columns
for col in ["policy_mean_effort", "sample_effort_mean"]:
    assert col in df_flat.columns, f"flat missing {col}"
    assert col in df_nested.columns, f"nested missing {col}"
print("Assertion 1 passed: both formats have policy_mean_effort and sample_effort_mean")

# Assertion 2: flat has non-NaN sample_effort_mean
assert not df_flat["sample_effort_mean"].isna().all(), \
    "flat format must record sample means"
print("Assertion 2 passed: flat sample_effort_mean is non-NaN")

# Assertion 3: nested has NaN sample_effort_mean (by design)
assert df_nested["sample_effort_mean"].isna().all(), \
    "nested format should have NaN sample_effort_mean"
print("Assertion 3 passed: nested sample_effort_mean is all NaN")

# Assertion 4: both have non-NaN policy_mean_effort
assert not df_flat["policy_mean_effort"].isna().any(), \
    "flat policy_mean_effort has NaN"
assert not df_nested["policy_mean_effort"].isna().any(), \
    "nested policy_mean_effort has NaN"
print("Assertion 4 passed: policy_mean_effort is non-NaN in both formats")

# Assertion 5: flat sample mean and policy mean differ
assert not np.allclose(
    df_flat["sample_effort_mean"].values, df_flat["policy_mean_effort"].values
), "sample and policy means should differ in flat format"
print("Assertion 5 passed: flat sample_effort_mean != policy_mean_effort")

# Assertion 6: nested policy_mean_effort matches (agent1+agent2)/2 exactly
nested_computed = (
    df_nested["agent1_effort"].values + df_nested["agent2_effort"].values
) / 2.0
assert np.allclose(
    df_nested["policy_mean_effort"].values, nested_computed
), "nested policy_mean_effort should equal (agent1+agent2)/2"
print("Assertion 6 passed: nested policy_mean_effort == (agent1+agent2)/2")

print("\nSchema verification passed.")
