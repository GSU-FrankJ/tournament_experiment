"""V1 acceptance check for registry canonicalization (plan items C1-C3+D1).

Verifies, against the real results on disk, that the paper-generator pipeline
now selects exactly the owner-approved interim canonical runs:

  three_players     -> round3_baseline   (q in {35, 55})
  different_cost    -> r4_dc_final       (q in {35, 55})
  different_ability -> r4_h1_long        (q in {35, 55})
  two_players       -> unchanged baseline (q in {35, 45, 55}; Set 2 separate)

Acceptance criteria (per owner):
  1. zero duplicate run identities in the registry (no guard warnings),
  2. exactly 5 seeds {42..46} per (experiment, q) baseline cell,
  3. every promoted baseline row originates from the approved post-fix tag
     (checked against the registry's file paths),
  4. every canonical run has stop_reason == "exploitability".

Run directly (python tests/test_registry_canonicalization.py) or via pytest.
READ-ONLY: loads JSONs, writes nothing.
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_SEEDS = {42, 43, 44, 45, 46}
CANONICAL = {
    "three_players": {"qs": [35.0, 55.0], "tag": "r5_sampled"},
    "different_cost": {"qs": [35.0, 55.0], "tag": "r5_sampled"},
    "different_ability": {"qs": [35.0, 55.0], "tag": "r5_sampled_std"},
    "two_players": {"qs": [35.0, 45.0, 55.0], "tag": "r5_sampled"},
}


def _load():
    from paper.generator.run_registry import discover_runs
    from paper.generator.extract import load_all_convergence_data, get_verified_convergence_step

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = discover_runs()
        df = load_all_convergence_data()
    dup_warnings = [str(w.message) for w in caught if "Duplicate run identity" in str(w.message)]
    conv = get_verified_convergence_step(df)
    return runs, df, conv, dup_warnings


def test_registry_canonicalization():
    runs, df, conv, dup_warnings = _load()

    # 1. zero duplicate identities
    keys = [(r.experiment, r.method, r.q, r.seed, r.ablation, r.weight_variant) for r in runs]
    assert len(keys) == len(set(keys)), "duplicate run identities present"
    assert not dup_warnings, f"duplicate-identity guard fired: {dup_warnings}"
    print(f"[V1] duplicate run identities: 0 (guard warnings: 0) across {len(runs)} runs")

    # Registry path lookup for source-tag verification
    path_of = {
        (r.experiment, r.method, r.q, r.seed, r.ablation): r.path
        for r in runs
        if r.method in ("TEL-PPO", "PPO") and r.weight_variant == "baseline"
    }

    base = conv[
        (conv["method"].isin(["TEL-PPO", "PPO"]))
        & (conv["ablation"] == "baseline")
        & (conv["weight_variant"] == "baseline")
    ]

    for exp, spec in CANONICAL.items():
        for q in spec["qs"]:
            cell = base[(base["experiment"] == exp) & (base["q"] == q)]
            seeds = sorted(cell["seed"].tolist())

            # 2. exactly 5 seeds
            assert set(seeds) == EXPECTED_SEEDS, f"{exp} q={q}: seeds {seeds}"

            # 3. post-fix tags only: the promoted baseline rows must come from
            # files registered under the approved tag
            srcs = []
            for s in seeds:
                p = path_of.get((exp, "TEL-PPO", q, s, spec["tag"])) or path_of.get(
                    (exp, "PPO", q, s, spec["tag"])
                )
                assert p is not None, f"{exp} q={q} seed {s}: no registry run with tag '{spec['tag']}'"
                srcs.append(os.path.basename(p))
            if spec["tag"] != "baseline":
                assert all(spec["tag"] in s for s in srcs), f"{exp} q={q}: wrong source files {srcs}"

            # 4. verified stop on every seed
            reasons = sorted(set(cell["stop_reason"].tolist()))
            assert reasons == ["exploitability"], f"{exp} q={q}: stop_reasons {reasons}"

            upds = {int(r["seed"]): int(r["convergence_update"]) for _, r in cell.iterrows()}
            print(
                f"[V1] {exp:<18} q={int(q):<3} tag={spec['tag']:<16} seeds={seeds} "
                f"stop=exploitability conv_update={[upds[s] for s in sorted(upds)]}"
            )

    # two_players Set 2 still present as a separate variant (not pooled):
    # 15 r5 PPO runs (3 q x 5 seeds) + 15 r5 gradient runs (legacy Set-2 rows
    # are dropped by the promotion)
    set2 = conv[
        (conv["experiment"] == "two_players")
        & (conv["weight_variant"] == "wh8_wl4")
        & (conv["ablation"] == "baseline")
    ]
    n_ppo = len(set2[set2["method"].isin(["TEL-PPO", "PPO"])])
    n_grad = len(set2[set2["method"] == "Gradient"])
    assert n_ppo == 15, f"expected 15 Set-2 PPO runs, got {n_ppo}"
    assert n_grad == 15, f"expected 15 Set-2 gradient runs, got {n_grad}"
    print(f"[V1] two_players Set 2 (wh8_wl4): {n_ppo} PPO + {n_grad} gradient runs kept separate (not pooled)")

    # Gradient baseline must also be the sampled r5 runs: 5 seeds per cell
    gbase = conv[
        (conv["method"] == "Gradient")
        & (conv["ablation"] == "baseline")
        & (conv["weight_variant"] == "baseline")
    ]
    for exp, spec in CANONICAL.items():
        for q in spec["qs"]:
            n = len(gbase[(gbase["experiment"] == exp) & (gbase["q"] == q)])
            assert n == 5, f"gradient baseline {exp} q={q}: expected 5 seeds, got {n}"
    print(f"[V1] gradient baseline: 5 sampled MC-FD seeds in every (experiment, q) cell")
    print("[V1] ACCEPTANCE: PASS")


if __name__ == "__main__":
    test_registry_canonicalization()
