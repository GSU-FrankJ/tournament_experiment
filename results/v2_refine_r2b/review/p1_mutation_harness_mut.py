"""Mutation harness (scratch copy only). Usage: python mut.py [--k EXPR] [--log FILE] ID [ID ...]

Applies ONE mutant (a list of (file, old, new) string edits) to the scratch COPY, runs pytest, restores the
file from the pristine worktree (copy2, original mtime), prints KILLED / SURVIVED and the first failing test.
Never touches the worktree (read-only source of the pristine text).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

PRISTINE = "/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-r2b"
SCR = "/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-r7-canonical-reference-checks-bd0e75/c3e11635-f436-49ae-b267-74fbf597c7c2/scratchpad/review_mutation"
COPY = SCR + "/repo"
PY = "/home/fjiang4/tournament_experiment/.venv/bin/python"
DEFAULT_K = "not v20 and not c7"
TESTFILE = "tests/test_v2_r2b.py"

BT = "utils/beta_tail.py"
V2 = "agents/ppo_curriculum_v2.py"
PW = "agents/ppo_pathwise.py"
CE = "envs/curriculum_env.py"
RO = "run/v2_rollout.py"
SW = "run/run_v2_stagewise.py"
LR = "tools/v2/launch_refine.py"
LC = "tools/v2/r2b_launch_checks.py"

M = {}


def mut(mid, desc, *edits):
    M[mid] = {"desc": desc, "edits": list(edits)}


# ---------------------------------------------------------------- required 1..25
mut("M01", "beta_tail row_log_prob: upper side uses (alpha, beta) instead of (beta, alpha)",
    (BT, "log_betainc_small_x(clamp_c, beta[hi], alpha[hi])", "log_betainc_small_x(clamp_c, alpha[hi], beta[hi])"))
mut("M02", "beta_tail row_log_prob: lower mask side<=0 instead of side<0",
    (BT, "lo = torch.nonzero(side < 0).squeeze(-1)", "lo = torch.nonzero(side <= 0).squeeze(-1)"))
mut("M03", "beta_tail: series truncated at n_terms=3 (n=1,2 only)",
    (BT, "N_TERMS = 12 ", "N_TERMS = 3 "))
mut("M04", "beta_tail: DOMAIN_X_B check removed",
    (BT, "if b64.numel() and float(b64.detach().max()) * x > DOMAIN_X_B:", "if False:"))
mut("M05", "v2.update KL block uses plain density instead of row_log_prob",
    (V2, """                logp = (torch.distributions.Beta(alpha, beta).log_prob(ac[prow]) if side is None
                        else row_log_prob(alpha, beta, ac[prow], side[prow], cfg.action_clamp))""",
     """                logp = torch.distributions.Beta(alpha, beta).log_prob(ac[prow])"""))
mut("M06", "masked_actor_loss ignores side (plain density)",
    (V2, "logp = dist.log_prob(ac[rows]) if side is None else row_log_prob(alpha, beta, ac[rows], side[rows], clamp_c)",
     "logp = dist.log_prob(ac[rows])"))
mut("M07", "update dispatch ignores side_np (flagged buffer takes the base path)",
    (V2, "if policy_mask is None and norm_mask is None and side_np is None:",
     "if policy_mask is None and norm_mask is None:"))
mut("M08", "update dispatch always takes the masked path (even with no side, no masks)",
    (V2, "if policy_mask is None and norm_mask is None and side_np is None:", "if False:"))
mut("M09", "rollout: side flags of mean-mode rows not suppressed",
    (RO, 'if not (use_frozen and continuation_action_mode == "mean"):', "if True:"))
mut("M10", "rollout: stored logp of censored mode uses the density",
    (RO, 'if clamp_likelihood == "censored":\n            logp = agent.log_prob(a_L, b_L, act_L, clamp_side=side_L)',
     'if False:\n            logp = agent.log_prob(a_L, b_L, act_L, clamp_side=side_L)'))
mut("M11", "rollout: raw > 1-c flagged as -1",
    (RO, "side_L[hi_L] = 1", "side_L[hi_L] = -1"))
mut("M12", "env peak_set uses <= / >= (closed interval)",
    (CE, "return (edges[:-1] < half_width) & (edges[1:] > -half_width)",
     "return (edges[:-1] <= half_width) & (edges[1:] >= -half_width)"))
mut("M13", "env peak_bin_probs: swap share and 1-share",
    (CE, "return np.where(m, share / n_peak, (1.0 - share) / (n - n_peak))",
     "return np.where(m, (1.0 - share) / (n - n_peak), share / n_peak)"))
mut("M14", "env peak_focused: searchsorted side='left'",
    (CE, 'rng.random(n), side="right")', 'rng.random(n), side="left")'))
mut("M15", "env peak_focused: within-bin position drawn before the bin draw",
    (CE, """        b = np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), edges.size - 2)
        u = rng.random(n)""",
     """        u = rng.random(n)
        b = np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), edges.size - 2)"""))
mut("M16", "env balanced(): consumes one extra rng.random",
    (CE, """        b = rng.integers(0, edges.size - 1, size=n)
        u = rng.random(n)""",
     """        b = rng.integers(0, edges.size - 1, size=n)
        rng.random(1)
        u = rng.random(n)"""))
mut("M17", "pathwise_update: permutation drawn once instead of once per epoch",
    (PW, """    for _ in range(epochs):
        perm = agent.rng_mb.permutation(n)
""", """    perm = agent.rng_mb.permutation(n)
    for _ in range(epochs):
"""))
mut("M18", "pathwise_update: slice of minibatch+1 rows (range step unchanged)",
    (PW, "d[perm[start:start + minibatch]]", "d[perm[start:start + minibatch + 1]]"))
mut("M18b", "pathwise_update: effective minibatch size M+1 (range step and slice)",
    (PW, "d[perm[start:start + minibatch]]", "d[perm[start:start + minibatch + 1]]"),
    (PW, "for start in range(0, n, minibatch):", "for start in range(0, n, minibatch + 1):"))
mut("M19", "pathwise_update: no permutation draw (arange)",
    (PW, "perm = agent.rng_mb.permutation(n)", "perm = np.arange(n)"))
mut("M20", "pathwise_update: foc/e0 reported from the first (pre-step) step instead of after the last",
    (PW, "    losses, norms = [], []\n", "    losses, norms = [], []\n    first = None\n"),
    (PW, '            losses.append(out["loss"])', '            first = first or out\n            losses.append(out["loss"])'),
    (PW, '''"grad_norm_pre_clip_max": float(np.max(norms)), "foc_abs_mean": float(foc.mean().item()),
            "foc_abs_max": float(foc.max().item()), "e0": float(e0.item()), "n_steps": len(losses)}''',
     '''"grad_norm_pre_clip_max": float(np.max(norms)), "foc_abs_mean": first["foc_abs_mean"],
            "foc_abs_max": first["foc_abs_max"], "e0": first["e0"], "n_steps": len(losses)}'''))
_FOC_BLOCK = """    with torch.no_grad():
        dev = agent.device
        obs_l = torch.as_tensor(spec.encode_obs(stage, d), device=dev)
        obs_o = torch.as_tensor(spec.encode_obs(stage, -d), device=dev)
        d_t = torch.as_tensor(d, dtype=torch.float64, device=dev)
        foc = foc_residual(spec, d_t, effort_mean(agent.actor, obs_l, spec),
                           effort_mean(agent.opponent, obs_o, spec)).abs()
        e0 = effort_mean(agent.actor, torch.as_tensor(spec.encode_obs(stage, np.zeros(1)), device=dev), spec)[0]
"""
mut("M20b", "pathwise_update: foc/e0 evaluated BEFORE the first step (pre-update weights), robust to the mock",
    (PW, _FOC_BLOCK, ""),
    (PW, "    losses, norms = [], []\n", _FOC_BLOCK + "    losses, norms = [], []\n"))
mut("M21", "runner: legacy-branch condition uses 'or' instead of 'and'",
    (SW, "if self.pathwise_epochs == 1 and self.pathwise_minibatch is None:",
     "if self.pathwise_epochs == 1 or self.pathwise_minibatch is None:"))
mut("M22", "Run.draw_starts always uses balanced",
    (SW, 'if sw is None or sw["scheme"] == "bin_balanced":', "if True:"))
mut("M23", "manifest drops pathwise_epochs",
    (SW, '"pathwise_epochs": run.pathwise_epochs, "pathwise_minibatch": run.pathwise_minibatch,',
     '"pathwise_minibatch": run.pathwise_minibatch,'))
mut("M24", "validate_config accepts pathwise_epochs/minibatch outside phase_P",
    (SW, 'if (pe != 1 or pm is not None) and mode != "phase_P":', "if False:"))
mut("M25", "launcher: P20_lr3e-4 arm uses LR 3e-5 (LR1)",
    (LR, '"P20_lr3e-4": {"kind": "pathwise", "lr": [(LR0, LR0, 1, 200)],',
     '"P20_lr3e-4": {"kind": "pathwise", "lr": [(LR1, LR1, 1, 200)],'))
# ---------------------------------------------------------------- own mutants
mut("M26", "pathwise_update: grad_norm_pre_clip_max = mean instead of max",
    (PW, '"grad_norm_pre_clip_max": float(np.max(norms))', '"grad_norm_pre_clip_max": float(np.mean(norms))'))
mut("M27", "pathwise_update: loss = last step instead of mean over steps",
    (PW, '"loss": float(np.mean(losses))', '"loss": float(losses[-1])'))
mut("M28", "pathwise_update: final foc evaluated with the opponent seeing +d instead of -d",
    (PW, """        obs_l = torch.as_tensor(spec.encode_obs(stage, d), device=dev)
        obs_o = torch.as_tensor(spec.encode_obs(stage, -d), device=dev)""",
     """        obs_l = torch.as_tensor(spec.encode_obs(stage, d), device=dev)
        obs_o = torch.as_tensor(spec.encode_obs(stage, d), device=dev)"""))
mut("M29", "row_log_prob: censored log-mass detached (no gradient from clamped rows)",
    (BT, "log_betainc_small_x(clamp_c, alpha[lo], beta[lo]).to(lp.dtype)",
     "log_betainc_small_x(clamp_c, alpha[lo], beta[lo]).to(lp.dtype).detach()"),
    (BT, "log_betainc_small_x(clamp_c, beta[hi], alpha[hi]).to(lp.dtype)",
     "log_betainc_small_x(clamp_c, beta[hi], alpha[hi]).to(lp.dtype).detach()"))
mut("M30", "v2.update: actor loss gets a different clamp c (1e-5) than the rollout (cfg.action_clamp)",
    (V2, "cfg.clip_eps, side, cfg.action_clamp)", "cfg.clip_eps, side, 1e-5)"))
mut("M31", "runner: update receives clamp_side even in density mode",
    (SW, 'side = batch["clamp_side"] if self.clamp_likelihood == "censored" else None',
     'side = batch["clamp_side"]'))
mut("M32", "v2.update KL block: side[:n_prow] instead of side[prow] (misaligned under a policy mask)",
    (V2, "row_log_prob(alpha, beta, ac[prow], side[prow], cfg.action_clamp))",
     "row_log_prob(alpha, beta, ac[prow], side[:prow.numel()], cfg.action_clamp))"))
mut("M33", "CLAMP_MODES drops phase_B (censored refused in phase B)",
    (SW, 'CLAMP_MODES = ("phase_A", "phase_A_continue", "phase_B")', 'CLAMP_MODES = ("phase_A", "phase_A_continue")'))
mut("M34", "START_MODES drops phase_A_continue and phase_P (peak_focused refused there)",
    (SW, 'START_MODES = ("phase_A", "phase_A_continue", "phase_P")', 'START_MODES = ("phase_A",)'))
mut("M35", "phase A call site draws balanced starts instead of Run.draw_starts",
    (SW, "d0 = self.draw_starts(spec.T, n_ep)", "d0 = self.sampler.balanced(spec.T, n_ep, rng_start)"))
mut("M36", "phase P call site draws balanced starts instead of Run.draw_starts",
    (SW, "d0 = self.draw_starts(stage, n_ep)", "d0 = self.sampler.balanced(stage, n_ep, rng_start)"))
mut("M37", "launch checks: 'files changed since code commit' violation branch disabled",
    (LC, "elif commit_cache[c]:", "elif False:"))
mut("M38", "phase P: minibatch config ignored (n_steps_batch = n_ep) with E=10",
    (SW, "n_steps_batch = n_ep if self.pathwise_minibatch is None else int(self.pathwise_minibatch)",
     "n_steps_batch = n_ep"))
mut("M39", "update(): np.all instead of np.any in the flagged-buffer test",
    (V2, "np.any(np.asarray(clamp_side) != 0)", "np.all(np.asarray(clamp_side) != 0)"))
mut("M40", "peak_focused: cdf[-1]=1.0 and np.minimum guard both removed (float-sum safety)",
    (CE, "        cdf[-1] = 1.0\n", ""),
    (CE, 'b = np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), edges.size - 2)',
     'b = np.searchsorted(cdf, rng.random(n), side="right")'))
mut("M41", "log_betainc_small_x: float32 internals (no float64 promotion)",
    (BT, "a64 = a.to(torch.float64)", "a64 = a.to(torch.float32)"),
    (BT, "b64 = b.to(torch.float64)", "b64 = b.to(torch.float32)"))
mut("M42", "update(): n_censored_rows counts only the lower side",
    (V2, 'out["n_censored_rows"] = int((side != 0).sum().item())', 'out["n_censored_rows"] = int((side < 0).sum().item())'))
mut("M43", "run: v2row n_clamped_rows_learner / censored columns recorded only for censored; counts hi only",
    (SW, 'v2row["n_clamped_rows_learner"] = int((batch["clamp_side"] != 0).sum())',
     'v2row["n_clamped_rows_learner"] = int((batch["clamp_side"] > 0).sum())'))
mut("M44", "pathwise_update: epochs ignored in the runner call (always 1 epoch)",
    (SW, "diag = pathwise_update(agent, spec, d_learner, stage, self.pathwise_epochs, n_steps_batch,",
     "diag = pathwise_update(agent, spec, d_learner, stage, 1, n_steps_batch,"))
mut("M45", "series: first-order sign error (n + b) instead of (n - b)",
    (BT, "term = term * ((n - b64) * (x / n))", "term = term * ((n + b64) * (x / n))"))
mut("M46", "rollout: stage-ordering of clamp_side reversed vs. actions",
    (RO, '"clamp_side": np.concatenate([per_stage[t]["side"] for t in sorted(per_stage)])',
     '"clamp_side": np.concatenate([per_stage[t]["side"] for t in sorted(per_stage, reverse=True)])'))
mut("M00", "CONTROL: no-op edit (the suite must pass, harness sanity)",
    (CE, "edges = self.bin_edges(t)\n        cdf = np.cumsum", "edges = self.bin_edges(t)\n        cdf = np.cumsum"))
mut("M48", "pathwise_update: ValueError guard for epochs/minibatch removed",
    (PW, "if epochs < 1 or minibatch < 1:", "if False:"))
mut("M49", "peak_bin_probs: share validity guard removed (share==1 allowed)",
    (CE, "if not 0.0 < share < 1.0:", "if False:"))
mut("M50", "Run: peak-set proper-subset check removed from Run.__init__",
    (SW, 'if self.start_weights is not None and self.start_weights["scheme"] == "peak_focused":',
     'if False:'))


mut("M51", "env peak_set: bins whose CENTRE lies in (-h, h) instead of bins intersecting (-h, h)",
    (CE, "return (edges[:-1] < half_width) & (edges[1:] > -half_width)",
     "return np.abs(0.5 * (edges[:-1] + edges[1:])) < half_width"))
mut("M52", "rollout: clamp_likelihood validity check removed",
    (RO, "    if clamp_likelihood not in CLAMP_LIKELIHOODS:\n        raise ValueError(f\"clamp_likelihood {clamp_likelihood!r} not in {CLAMP_LIKELIHOODS}\")\n", ""))
mut("M53", "Run.draw_starts: peak_focused called with (share, half_width) swapped",
    (SW, 'float(sw["peak_half_width"]), float(sw["peak_share"]))\n\n    # ----', 'float(sw["peak_share"]), float(sw["peak_half_width"]))\n\n    # ----'))
mut("M54", "pathwise_update: grad_norm_pre_clip (mean) reported as the max",
    (PW, '"grad_norm_pre_clip": float(np.mean(norms))', '"grad_norm_pre_clip": float(np.max(norms))'))


mut("L1", "launch checks: dirty flag accepted when missing/None (only truthy dirty refused)",
    (LC, 'if git.get("dirty") is not False:', 'if git.get("dirty"):'))
mut("L2", "launch checks: manifest parent_checkpoint vs parents.csv comparison disabled",
    (LC, 'if man.get("parent_checkpoint") != p["waveP_parent_path"]:', 'if False:'))
mut("L3", "launch checks: parent FILE sha256 vs parents.csv comparison disabled",
    (LC, 'elif sha256_file(Path(p["waveP_parent_path"])) != p["waveP_parent_sha256"]:', 'elif False:'))
mut("L4", "launch checks: manifest R2b keys vs arm table loop disabled",
    (LC, "    for k in R2B_KEYS:\n        want_v", "    for k in ():\n        want_v"))
mut("L5", "launcher: A_peak25 share 0.25 -> 0.5 (control; pinned)",
    (LR, '"peak_share": 0.25}},\n                 "definition": "mechanism 1: peak-focused exploring starts, peak set = bins intersecting',
     '"peak_share": 0.5}},\n                 "definition": "mechanism 1: peak-focused exploring starts, peak set = bins intersecting'))
mut("L6", "launcher: PEAK_HALF_WIDTH 20 -> 25",
    (LR, "PEAK_HALF_WIDTH = 20", "PEAK_HALF_WIDTH = 25"))
mut("L7", "launcher: wave-P pathwise arms use minibatch 128 (steps 40 per update)",
    (LR, "PATHWISE_EPOCHS, PATHWISE_MINIBATCH = 10, 256", "PATHWISE_EPOCHS, PATHWISE_MINIBATCH = 10, 128"))
mut("L8", "launcher: control A_ctrl200_lr3e-4 keeps LR1 (3e-5) -> duplicates R1 A_ctrl200",
    (LR, '"A_ctrl200_lr3e-4": {"kind": "ctrl200", "lr": [(LR0, LR0, 1, 200)]', '"A_ctrl200_lr3e-4": {"kind": "ctrl200", "lr": [(LR1, LR1, 1, 200)]'))
mut("L9", "launcher: pathwise arms cap 200 -> 100 via lr window local_last",
    (LR, '"P20_lr3e-5": {"kind": "pathwise", "lr": [(LR1, LR1, 1, 200)]', '"P20_lr3e-5": {"kind": "pathwise", "lr": [(LR1, LR1, 1, 100)]'))
mut("L10", "launcher: _with_r2b_keys writes arm values BEFORE defaults (defaults overwrite the arm)",
    (LR, "    cfg.update(copy.deepcopy(R2B_DEFAULTS))\n    cfg.update(copy.deepcopy(overrides))", "    cfg.update(copy.deepcopy(overrides))\n    cfg.update(copy.deepcopy(R2B_DEFAULTS))"))


mut("M55", "runner: rollout always called with clamp_likelihood='censored' (update still gets side=None in density mode)",
    (SW, "cont_table=cont_table, clamp_likelihood=self.clamp_likelihood)", 'cont_table=cont_table, clamp_likelihood="censored")'))
mut("M56", "runner: rollout never gets the configured clamp_likelihood (always density) -> censored arm stores density",
    (SW, "cont_table=cont_table, clamp_likelihood=self.clamp_likelihood)", 'cont_table=cont_table, clamp_likelihood="density")'))


mut("M57", "runner: masked-branch update (phase B) drops clamp_side",
    (SW, "policy_mask=s1, norm_mask=norm_mask, clamp_side=side)", "policy_mask=s1, norm_mask=norm_mask)"))
mut("M58", "runner: unmasked censored branch drops clamp_side (falls back to the base update)",
    (SW, "                                    batch[\"advantages\"], clamp_side=side)", "                                    batch[\"advantages\"])"))


def run_one(mid, k):
    desc = M[mid]
    originals = {}
    t0 = time.time()
    try:
        for f, old, new in desc["edits"]:
            if f not in originals:
                originals[f] = open(f"{PRISTINE}/{f}").read()
        cur = dict(originals)
        for f, old, new in desc["edits"]:
            assert cur[f].count(old) == 1, f"{mid}: pattern count {cur[f].count(old)} in {f}: {old[:60]!r}"
            cur[f] = cur[f].replace(old, new)
        for f, txt in cur.items():
            with open(f"{COPY}/{f}", "w") as fh:
                fh.write(txt)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1",
                   GIT_DIR="/home/fjiang4/tournament_experiment/.git/worktrees/v2-t2-r2b")
        env.pop("GIT_WORK_TREE", None)
        cmd = [PY, "-m", "pytest", TESTFILE, "-p", "no:cacheprovider", "-q", "-x", "-k", k]
        r = subprocess.run(cmd, cwd=COPY, capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        with open(f"{SCR}/out_{mid}.txt", "w") as fh:
            fh.write(out)
        failed = re.findall(r"^FAILED (\S+)", out, re.M)
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        status = "SURVIVED" if r.returncode == 0 else ("KILLED" if failed else "ERROR")
        return {"id": mid, "status": status, "rc": r.returncode, "failed": failed, "tail": tail,
                "sec": round(time.time() - t0, 1), "desc": desc["desc"], "out_tail": out[-1500:]}
    finally:
        for f in originals:
            shutil.copy2(f"{PRISTINE}/{f}", f"{COPY}/{f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    k, log = DEFAULT_K, SCR + "/log.jsonl"
    while args and args[0] in ("--k", "--log", "--file"):
        if args[0] == "--k":
            k = args[1]
        elif args[0] == "--file":
            TESTFILE = args[1]
        else:
            log = args[1]
        args = args[2:]
    ids = args or list(M)
    for mid in ids:
        res = run_one(mid, k)
        res["k"] = k
        print(f"{res['id']:5s} {res['status']:9s} {res['sec']:6.1f}s  first_fail={res['failed'][:1]}  | {res['tail']} | {res['desc']}", flush=True)
        with open(log, "a") as fh:
            fh.write(json.dumps(res) + "\n")
