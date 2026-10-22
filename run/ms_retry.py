"""Candidate A of round T3-R3: the verifier-gated restart of the terminal stage (config key ``stage3_retry``).

``run/run_ms_stagewise.py`` calls these helpers only when the key is present; without it nothing here runs and the pipeline is the
``bw`` pipeline bit for bit.

Specification (``reports/t3/r3/02_proposal.md`` section 2.1, accepted at gate G1, ``reports/t3/r3/pi_record/28b_g1_reply.md``):

* the trigger is read at the development check of the last local update of the stage-3 phase. It fires iff ``Delta`` > ``delta_gt``,
  or ``Delta``, ``s`` or ``C`` is not finite, or ``s`` <= 0. No other column is read (in particular no closed-form quantity and not
  ``valid``, which also depends on stages 1 and 2 of the composite candidate);
* at most one retry; attempt 2 is frozen whatever its own check says;
* attempt 2 re-initialises the actor and the critic **in place** (a new ``nn.Linear`` would draw from the process-global torch
  generator, and a new optimiser object would trip the ``optimizer object replaced`` guard of ``run_stage``) with the generator calls
  of the original construction from ``SeedSequence([seed, q, ns_init, 2])``, clears both Adam states in place, and draws from five new
  streams ``default_rng(SeedSequence([seed, q, ns_name, 2]))``;
* the artefacts of attempt 1 are moved to ``attempt1/`` (never overwritten, never deleted), its in-memory histories are dumped there,
  and the counters of the run are reset, so that the frozen path (attempt 2, then stages 2 and 1) writes exactly the files, columns
  and rows of ``bw``; ``retry.json`` carries the cost of both attempts.
"""

from __future__ import annotations

import copy
import csv
import glob
import json
import math
import os
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.ppo_curriculum import _orthogonal

SCHEMA = "t3_retry/1"
KEYS = ("delta_gt", "max_retries")
ATTEMPT_DIR = "attempt1"
ATTEMPT = 2                                    # the derived-seed tag of the second attempt
STREAM_NAMES = {"env": "env_noise", "learn": "learner_action", "opp": "opponent_action", "start": "starts_roles",
                "minibatch": "minibatch"}      # run stream -> key of ``protocol.rng_namespaces``
CHECK_FIELDS = ("valid", "Delta", "s", "R", "R_tail", "C")


def validate(sr: Any, T: int) -> Optional[str]:
    """Reason why ``stage3_retry`` is invalid (``None`` if it is valid)."""
    if T != 3:
        return f"stage3_retry is defined for T = 3 (the terminal stage is stage 3); got T = {T}"
    if not isinstance(sr, dict) or set(sr) != set(KEYS):
        return f"stage3_retry must have exactly the keys {KEYS}"
    d, m = sr["delta_gt"], sr["max_retries"]
    if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0.0:
        return "stage3_retry.delta_gt must be a positive finite number"
    if isinstance(m, bool) or m != 1 or not isinstance(m, int):
        return "stage3_retry.max_retries must be the integer 1 (at most one retry)"
    return None


def trigger(check: Optional[Mapping[str, Any]], delta_gt: float, expected_local: int) -> Tuple[bool, List[str]]:
    """Whether the retry fires, and why.

    Fires iff ``Delta`` > ``delta_gt``, or ``Delta``, ``s`` or ``C`` is not finite, or ``s`` <= 0 (strict, float64). Only these three
    columns of the check are read.

    Args:
        check: The last development check of the phase (``MSRun.last_check``).
        delta_gt: The limit (0.005).
        expected_local: The local update the check must belong to (the stage-3 budget).

    Raises:
        RuntimeError: There is no check, or it is not the check of local update ``expected_local`` of stage 3.
    """
    if check is None or int(check["stage"]) != 3 or int(check["local"]) != int(expected_local):
        raise RuntimeError(f"the trigger needs the stage-3 check of local update {expected_local}; got {check!r}")
    delta, s, c = (float(check[k]) for k in ("Delta", "s", "C"))
    reasons: List[str] = []
    if delta > float(delta_gt):
        reasons.append("delta_gt")
    if not (math.isfinite(delta) and math.isfinite(s) and math.isfinite(c)):
        reasons.append("non_finite")
    if s <= 0.0:
        reasons.append("s_le_0")
    return bool(reasons), reasons


def derived_streams(seed: int, q: float, namespaces: Mapping[str, int], attempt: int = ATTEMPT
                    ) -> Tuple[torch.Generator, Dict[str, np.random.Generator]]:
    """The init generator and the five streams of attempt ``attempt`` (the ``SeedSequence`` entropy of the original streams with
    ``attempt`` appended)."""
    init_state = np.random.SeedSequence([int(seed), int(q), int(namespaces["init"]), int(attempt)]).generate_state(1)[0]
    gen = torch.Generator().manual_seed(int(init_state))
    rngs = {k: np.random.default_rng(np.random.SeedSequence([int(seed), int(q), int(namespaces[v]), int(attempt)]))
            for k, v in STREAM_NAMES.items()}
    return gen, rngs


def reinit_networks(agent: Any, gen: torch.Generator) -> None:
    """Re-initialise the actor and the critic in place, with the generator calls (and their order) of the original construction
    (``BetaActor.__init__`` then ``Critic.__init__``): orthogonal gain sqrt 2 on the hidden layers, zero biases, the actor output
    head 0 / 0, the critic output orthogonal gain 1. Draws nothing from any process-global generator."""
    with torch.no_grad():
        _orthogonal(agent.actor.l1, math.sqrt(2.0), gen)
        _orthogonal(agent.actor.l2, math.sqrt(2.0), gen)
        nn.init.zeros_(agent.actor.out.weight)
        nn.init.zeros_(agent.actor.out.bias)
        _orthogonal(agent.critic.l1, math.sqrt(2.0), gen)
        _orthogonal(agent.critic.l2, math.sqrt(2.0), gen)
        _orthogonal(agent.critic.out, 1.0, gen)


def write_rows_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """The writer of ``write_histories`` (header = union of the keys in first-seen order, empty string for a missing value)."""
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def _dump_json(path: str, obj: Any) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "x") as f:
        json.dump(obj, f, indent=1, default=default)


def phase_counts(run: Any) -> Dict[str, Any]:
    """Updates, episodes, transitions (policy rows), minibatch steps and wall time of what the run has done so far."""
    st = run.phase_timing.get("stage3", {})
    return {"updates": int(run.global_u), "episodes": int(run.total_episodes), "transitions": int(run.total_transitions),
            "minibatch_steps": int(sum(h["n_minibatch_steps"] for h in run.history)),
            "wall_sec": float(st.get("wall_sec", 0.0)), "process_cpu_sec": float(st.get("process_cpu_sec", 0.0)),
            "dev_verifier_sec": float(run.costs.t.get("dev_verifier", 0.0)), "n_dev_checks": int(run.costs.n.get("dev_verifier", 0))}


def restart_stage3(run: Any, rec1: Mapping[str, Any], reasons: Sequence[str]) -> Dict[str, Any]:
    """Archive attempt 1 and reset the run for attempt 2 (see the module docstring). Returns the record of attempt 1.

    Everything is moved or dumped, nothing is overwritten or deleted. Raises ``RuntimeError`` if ``attempt1/`` exists.
    """
    out = run.out_dir
    adir = os.path.join(out, ATTEMPT_DIR)
    if os.path.exists(adir):
        raise RuntimeError(f"{adir} already exists; refusing to overwrite")
    os.makedirs(adir)
    chk = {k: run.last_check[k] for k in ("stage", "local", "update", *CHECK_FIELDS)}
    rec = {"schema": SCHEMA, "attempt": 1, "trigger_check": chk, "reasons": list(reasons), "delta_gt": float(run.retry_cfg["delta_gt"]),
           "counts": phase_counts(run), "costs": run.costs.as_dict(), "rule_record": copy.deepcopy(dict(rec1)),
           "init_state_sha256": run.init_state_sha256, "seed_namespaces": dict(run.seed_namespaces)}
    # in-memory state that write_histories would write at the end of the run: dump it now
    write_rows_csv(os.path.join(adir, "ms_updates_stage3.csv"), run.rows)
    _dump_json(os.path.join(adir, "train_history_stage3.json"),
               {"run": run.cfg["run"], "arm": run.cfg["arm"], "attempt": 1, "history": run.history, "snapshots": run.snapshot_log,
                "weight_checkpoints": run.weights_log, "phase_timing": run.phase_timing})
    torch.save(run.full_state("stage3_attempt1"), os.path.join(adir, "state_attempt1_end.pt"))
    # files the phase wrote
    os.makedirs(os.path.join(adir, "weights"))
    for f in sorted(glob.glob(os.path.join(run.weights_dir, "u*.npz"))):
        shutil.move(f, os.path.join(adir, "weights", os.path.basename(f)))
    for name in ("ms_checks_stage3.csv", "ms_binmaps_stage3.npz"):
        shutil.move(os.path.join(out, name), os.path.join(adir, name))
    for f in sorted(glob.glob(os.path.join(out, "state_u*.pt"))):
        shutil.move(f, os.path.join(adir, os.path.basename(f)))
    # attempt 2: fresh networks (in place), optimiser states, streams, counters
    gen, rngs = derived_streams(run.seed, run.spec.q, run.P["rng_namespaces"])
    reinit_networks(run.agent, gen)
    rec["attempt2_init_state_sha256"] = run._init_digest(run.agent)
    _dump_json(os.path.join(adir, "attempt1_record.json"), rec)
    run.torch_gen = gen
    run.agent.opt_actor.state.clear()
    run.agent.opt_critic.state.clear()
    run.rngs = {k: rngs[k] for k in ("env", "learn", "opp", "start")}
    run.agent.rng_mb = rngs["minibatch"]
    run.agent.snapshot_refreshes = 1                      # as after the constructor; run_stage refreshes at the phase entry
    run.history, run.rows, run.weights_log = [], [], []
    run.snapshot_log = [{"update": 0, "reason": "init"}]
    run.phase_timing = {}
    run.costs = type(run.costs)()
    run.total_episodes = run.total_transitions = run.global_u = 0
    run.last_check = None
    return rec


def attempt2_init_digest(run: Any) -> str:
    """The init digest of attempt 2 for a freshly constructed run (the planned value of the launch plan): the networks are
    re-initialised in place with the derived generator and hashed. Draws nothing from any process-global generator."""
    gen, _ = derived_streams(run.seed, run.spec.q, run.P["rng_namespaces"])
    reinit_networks(run.agent, gen)
    return run._init_digest(run.agent)


def final_record(run: Any, rec1: Mapping[str, Any], a2_check: Mapping[str, Any], a2_digest: str) -> Dict[str, Any]:
    """``retry.json``: the trigger, both attempts and the total cost (written at the end of the run)."""
    a2 = phase_counts(run)
    st3 = run.phase_timing.get("stage3", {})
    a2_stage3 = {"updates": int(st3.get("updates", 0)), "wall_sec": float(st3.get("wall_sec", 0.0)),
                 "process_cpu_sec": float(st3.get("process_cpu_sec", 0.0))}
    c1 = rec1["counts"]
    totals = {"updates": c1["updates"] + a2["updates"], "transitions": c1["transitions"] + a2["transitions"],
              "episodes": c1["episodes"] + a2["episodes"], "minibatch_steps": c1["minibatch_steps"] + a2["minibatch_steps"],
              "dev_checks": c1["n_dev_checks"] + a2["n_dev_checks"], "dev_verifier_sec": c1["dev_verifier_sec"] + a2["dev_verifier_sec"],
              "stage3_wall_sec": c1["wall_sec"] + a2_stage3["wall_sec"]}
    return {"schema": SCHEMA, "fired": True, "config": dict(run.retry_cfg), "frozen_attempt": 2,
            "attempt1": {k: rec1[k] for k in ("trigger_check", "reasons", "counts", "init_state_sha256")},
            "attempt2": {"final_check": dict(a2_check), "init_state_sha256": a2_digest, "stage3": a2_stage3,
                         "entropy": {"init": [int(run.seed), int(run.spec.q), int(run.P["rng_namespaces"]["init"]), ATTEMPT],
                                     "streams": {k: [int(run.seed), int(run.spec.q), int(run.P["rng_namespaces"][v]), ATTEMPT]
                                                 for k, v in STREAM_NAMES.items()}}},
            "frozen_path_counts": a2, "totals": totals}
