#!/usr/bin/env python3
"""D1 (T57 issue 7): likelihood consistency of clipped Beta samples, per q and per phase.

The v2 rollout draws a = rng.beta(alpha, beta) and stores clip(a, 1e-6, 1 - 1e-6); the PPO ratio
uses the Beta DENSITY at the stored clipped value, whereas a clipped draw is a censored
observation whose likelihood is the tail MASS P(A <= c) (or P(A >= 1 - c)). This tool measures
how often that matters, from the ``d1_*`` columns of ``v2_updates.csv`` and the saved buffers
``d1_buffers/u*.npz`` of finished runs (read-only; nothing is written next to the runs).

Run layout (one or more roots, each ``[LABEL=]PATH``)::

    <root>/q{q}/seed{seed}/v2_updates.csv  (+ d1_buffers/, weights/)           arm = ""
    <root>/q{q}/seed{seed}/<arm>/v2_updates.csv                                 arm = <arm>

A "group" is ``LABEL`` (default: the root's directory name) plus ``/<arm>`` if there is an arm.
All aggregation is per (group, q, phase). Policy rows are the learner rows that enter the actor
loss: every row in phase A, the stage-1 rows of a frozen phase B (stage-2 rows are reported but
labelled non-policy). The policy-row set of an update is read off the counts
(``d1_pol_n_rows`` equals the number of all learner rows, or of the stage-1 rows).

Tables written to ``--out`` (CSV; every number of the report comes from one of them):
  d1_runs, d1_per_update, d1_per_run_phase, d1_by_group, d1_alpha_beta_lt1,
  d1_clamp_fraction_by_local, d1_buffers, d1_clamped_rows, d1_logdiff_by_group,
  d1_gradient_share, d1_flags, d1_summary.json; figures/d1_clamp_fraction_*.{png,pdf}.

Materiality flags (pre-registered, report-only):
  M1: median over runs of the per-phase clamp fraction among learner policy rows > 1e-3
  M2: gradient share of the clamped rows > 1% at any saved update in MORE THAN 2 runs
      (evaluated literally with however many runs exist).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python tools/v2/d1_clamp_analysis.py --roots results/v2_refine/v11_reproduction \
      --out results/v2_refine/d1_clamp --report-path reports/v2/refine/02_d1_clamp.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import special
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import BetaActor  # noqa: E402
from agents.ppo_curriculum_v2 import masked_actor_loss  # noqa: E402

CLAMP = 1e-6                 # PPOConfig.action_clamp: raw draws below / above 1 - CLAMP are clipped
CLIP_EPS = 0.2               # PPOConfig.clip_eps
ADV_NORM_EPS = 1e-8          # PPOConfig.adv_norm_eps
C_MIN, MU_CLAMP = 100.0, 1e-6    # PPOConfig.c_min / mu_clamp; hidden size is read from the npz
M1_THRESHOLD = 1e-3          # M1: median over runs of the per-phase clamp fraction, strict >
M2_SHARE_THRESHOLD = 0.01    # M2: gradient share of the clamped rows, strict >
M2_MAX_RUNS = 2              # M2 fires when MORE THAN this many runs exceed the share threshold
RUN_KEYS = ["group", "arm", "q", "seed"]
PHASE_KEYS = RUN_KEYS + ["phase"]
D1_FILES = ("d1_runs", "d1_per_update", "d1_per_run_phase", "d1_by_group", "d1_alpha_beta_lt1",
            "d1_clamp_fraction_by_local", "d1_buffers", "d1_clamped_rows", "d1_logdiff_by_group",
            "d1_gradient_share", "d1_flags")


# ============================================================================ run discovery
@dataclass(frozen=True)
class RunInfo:
    """One finished run directory."""

    group: str
    arm: str
    q: float
    seed: int
    path: Path


def _num(x: float):
    """int when integral (so q prints as 50, not 50.0), else float."""
    return int(x) if float(x) == int(float(x)) else float(x)


def parse_root_spec(spec: str) -> Tuple[str, Path]:
    """``LABEL=PATH`` or ``PATH`` (label = directory name)."""
    if "=" in spec:
        label, p = spec.split("=", 1)
        return label, Path(p)
    return Path(spec).name, Path(spec)


def discover_runs(roots: Sequence[Tuple[str, Path]], arms: Optional[Sequence[str]] = None
                  ) -> Tuple[List[RunInfo], List[Dict[str, str]]]:
    """Find every ``<root>/q*/seed*[/<arm>]`` directory with a ``v2_updates.csv``.

    Args:
        roots: ``(label, path)`` pairs.
        arms: If given, keep only runs whose arm name is in this list.

    Returns:
        ``(runs, skipped)``; ``skipped`` lists run-like directories without ``v2_updates.csv``.
    """
    runs: List[RunInfo] = []
    skipped: List[Dict[str, str]] = []
    for label, root in roots:
        for qdir in sorted(Path(root).glob("q*")):
            mq = re.fullmatch(r"q(\d+(?:\.\d+)?)", qdir.name)
            if not mq or not qdir.is_dir():
                continue
            for sdir in sorted(qdir.glob("seed*")):
                ms = re.fullmatch(r"seed(\d+)", sdir.name)
                if not ms or not sdir.is_dir():
                    continue
                cands = [("", sdir)] + [(c.name, c) for c in sorted(sdir.iterdir()) if c.is_dir()]
                for arm, p in cands:
                    if (p / "v2_updates.csv").exists():
                        if arms is None or arm in arms:
                            group = label if not arm else f"{label}/{arm}"
                            runs.append(RunInfo(group, arm, _num(float(mq.group(1))),
                                                int(ms.group(1)), p))
                    elif arms is None and ((p / "status.json").exists()
                                           or (p / "run_config.json").exists()):
                        skipped.append({"path": str(p), "reason": "no v2_updates.csv"})
    return runs, skipped


# ============================================================================ per-update tables
def stage_indices(columns: Sequence[str]) -> List[int]:
    """Stage indices t with a ``d1_L_s{t}_n`` column."""
    return sorted({int(m.group(1)) for c in columns if (m := re.fullmatch(r"d1_L_s(\d+)_n", c))})


def derive_update_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add the policy-row columns and the category columns ``c_<cat>_{n,lo,hi,pol}``.

    Rows without d1 data (e.g. phase P) are dropped. Raises ValueError when the policy-row count
    of an update equals neither the number of all learner rows nor the number of stage-1 rows.
    """
    df = df[df["d1_pol_n_rows"].notna()].copy()
    stages = stage_indices(df.columns)
    if df.empty or not stages:
        return df.assign(policy_scope=pd.Series(dtype=str))
    T = max(stages)
    tot = sum(df[f"d1_L_s{t}_n"] for t in stages)
    npol = df["d1_pol_n_rows"]
    scope = np.where(npol == tot, "all_stages",
                     np.where(npol == df["d1_L_s1_n"], "stage1_only", "?"))
    if (scope == "?").any():
        raise ValueError("policy-row count matches neither all learner rows nor the stage-1 rows "
                         f"at updates {df.loc[scope == '?', 'update'].tolist()[:5]}")
    df["policy_scope"] = scope
    allp = scope == "all_stages"

    def put(name: str, n, lo, hi, pol) -> None:
        df[f"c_{name}_n"], df[f"c_{name}_lo"], df[f"c_{name}_hi"] = n, lo, hi
        df[f"c_{name}_pol"] = pol

    lo_all = sum(df[f"d1_L_s{t}_lo"] for t in stages)
    hi_all = sum(df[f"d1_L_s{t}_hi"] for t in stages)
    put("learner_policy", npol, np.where(allp, lo_all, df["d1_L_s1_lo"]),
        np.where(allp, hi_all, df["d1_L_s1_hi"]), True)
    put("learner_all", tot, lo_all, hi_all, allp)
    for t in stages:
        put(f"learner_s{t}", df[f"d1_L_s{t}_n"], df[f"d1_L_s{t}_lo"], df[f"d1_L_s{t}_hi"],
            allp | (t == 1))
        put(f"opponent_s{t}", df[f"d1_O_s{t}_n"], df[f"d1_O_s{t}_lo"], df[f"d1_O_s{t}_hi"], False)
    put("opponent_all", sum(df[f"d1_O_s{t}_n"] for t in stages),
        sum(df[f"d1_O_s{t}_lo"] for t in stages), sum(df[f"d1_O_s{t}_hi"] for t in stages), False)
    for reg in ("in", "out"):
        if f"d1_L_s{T}_{reg}_n" in df.columns:
            put(f"learner_final_{reg}", df[f"d1_L_s{T}_{reg}_n"], df[f"d1_L_s{T}_{reg}_lo"],
                df[f"d1_L_s{T}_{reg}_hi"], allp | (T == 1))
    df["pol_n"] = npol
    df["pol_lo"], df["pol_hi"] = df["c_learner_policy_lo"], df["c_learner_policy_hi"]
    df["pol_hit"] = df["pol_lo"] + df["pol_hi"]
    df["pol_frac"] = np.where(npol > 0, df["pol_hit"] / npol.where(npol > 0, 1), np.nan)
    return df


def load_run_updates(run: RunInfo) -> Tuple[pd.DataFrame, int]:
    """Derived per-update table of one run (with run keys) and the number of rows dropped."""
    df = pd.read_csv(run.path / "v2_updates.csv",
                     usecols=lambda c: c in ("update", "phase", "local") or c.startswith("d1_"))
    if "d1_pol_n_rows" not in df.columns:
        return pd.DataFrame(), len(df)
    out = derive_update_table(df)
    for k, v in (("seed", run.seed), ("q", run.q), ("arm", run.arm), ("group", run.group)):
        out.insert(0, k, v)
    return out, len(df) - len(out)


def category_names(u: pd.DataFrame) -> List[str]:
    """Category names present in a derived update table, in report order."""
    names = [c[2:-2] for c in u.columns if c.startswith("c_") and c.endswith("_n")]
    first = ["learner_policy", "learner_all"]
    rest = sorted(n for n in names if n not in first)
    return [n for n in first + rest if n in names]


def _quant(x: np.ndarray) -> Dict[str, float]:
    """min / median / p90 / max (numpy linear-interpolation percentiles)."""
    if x.size == 0:
        return {"min": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    return {"min": float(x.min()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def category_stats(sub: pd.DataFrame, cat: str) -> Optional[Dict[str, object]]:
    """Clamp-hit statistics of one category over one (group, q, phase) block of updates."""
    n, lo, hi = (sub[f"c_{cat}_{s}"].to_numpy(float) for s in ("n", "lo", "hi"))
    if n.sum() == 0:
        return None
    pol = sub[f"c_{cat}_pol"].to_numpy(bool)
    status = ("policy" if pol.all() else "mixed (stage-1 rows are policy)" if pol.any() else
              "opponent (not trained)" if cat.startswith("opponent") else "non-policy (masked)")
    if cat == "learner_all" and not pol.all():
        status = "mixed (stage-1 rows are policy)"
    m = n > 0
    upd = (lo + hi)[m] / n[m]
    run_sum = sub.assign(_n=n, _h=lo + hi).groupby(["arm", "seed"])[["_n", "_h"]].sum()
    run_frac = (run_sum["_h"] / run_sum["_n"].where(run_sum["_n"] > 0)).dropna().to_numpy()
    out: Dict[str, object] = {
        "category": cat, "policy_status": status, "n_runs": int(len(run_sum)),
        "n_updates": int(m.sum()), "rows_sum": int(n.sum()), "lo_sum": int(lo.sum()),
        "hi_sum": int(hi.sum()), "hit_sum": int((lo + hi).sum()),
        "frac_sum": float((lo + hi).sum() / n.sum()), "n_updates_with_hit": int((upd > 0).sum())}
    for k, v in _quant(upd).items():
        out[f"upd_frac_{k}"] = v
    for k, v in _quant(run_frac).items():
        out[f"run_frac_{k}"] = v
    return out


def per_run_phase_table(u: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, phase): policy-row clamp statistics, category sums, alpha / beta < 1."""
    cats = category_names(u)
    rows = []
    for key, g in u.groupby(PHASE_KEYS, sort=True):
        row: Dict[str, object] = dict(zip(PHASE_KEYS, key))
        scopes = sorted(set(g["policy_scope"]))
        row["policy_scope"] = scopes[0] if len(scopes) == 1 else "mixed"
        row["n_updates"] = int(len(g))
        row["first_update"], row["last_update"] = int(g["update"].min()), int(g["update"].max())
        for s in ("n", "lo", "hi"):
            row[f"pol_{s}"] = int(g[f"pol_{s}"].sum())
        row["pol_hit"] = row["pol_lo"] + row["pol_hi"]
        row["pol_frac"] = row["pol_hit"] / row["pol_n"] if row["pol_n"] else np.nan
        for k, v in _quant(g["pol_frac"].dropna().to_numpy()).items():
            row[f"upd_frac_{k}"] = v
        row["n_updates_with_hit"] = int((g["pol_hit"] > 0).sum())
        for c in cats:
            for s in ("n", "lo", "hi"):
                row[f"{c}_{s}"] = int(g[f"c_{c}_{s}"].sum())
        row["pol_alpha_min"], row["pol_beta_min"] = (float(g[f"d1_pol_{s}_min"].min())
                                                     for s in ("alpha", "beta"))
        row["pol_n_alpha_lt1"], row["pol_n_beta_lt1"] = (int(g[f"d1_pol_n_{s}_lt1"].sum())
                                                         for s in ("alpha", "beta"))
        row["n_updates_alpha_lt1"] = int((g["d1_pol_n_alpha_lt1"] > 0).sum())
        row["n_updates_beta_lt1"] = int((g["d1_pol_n_beta_lt1"] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def by_group_table(u: pd.DataFrame) -> pd.DataFrame:
    """Per (group, q, phase, category): pooled per-update and per-run clamp statistics."""
    rows = []
    for key, g in u.groupby(["group", "q", "phase"], sort=True):
        for cat in category_names(g):
            st = category_stats(g, cat)
            if st is not None:
                rows.append(dict(zip(["group", "q", "phase"], key), **st))
    return pd.DataFrame(rows)


def alpha_beta_table(prp: pd.DataFrame) -> pd.DataFrame:
    """Per (group, q, phase): rows with alpha < 1 or beta < 1 among the learner policy rows."""
    rows = []
    for key, g in prp.groupby(["group", "q", "phase"], sort=True):
        rows.append(dict(zip(["group", "q", "phase"], key), n_runs=int(len(g)),
                         pol_rows_sum=int(g["pol_n"].sum()),
                         n_alpha_lt1_sum=int(g["pol_n_alpha_lt1"].sum()),
                         n_beta_lt1_sum=int(g["pol_n_beta_lt1"].sum()),
                         n_updates_alpha_lt1=int(g["n_updates_alpha_lt1"].sum()),
                         n_updates_beta_lt1=int(g["n_updates_beta_lt1"].sum()),
                         alpha_min=float(g["pol_alpha_min"].min()),
                         beta_min=float(g["pol_beta_min"].min()),
                         alpha_min_median_over_runs=float(g["pol_alpha_min"].median()),
                         beta_min_median_over_runs=float(g["pol_beta_min"].median())))
    return pd.DataFrame(rows)


def by_local_table(u: pd.DataFrame) -> pd.DataFrame:
    """Per (group, q, phase, local update): policy-row clamp fraction over runs (the plot data)."""
    g = u.groupby(["group", "q", "phase", "local"], sort=True)["pol_frac"]
    return pd.DataFrame({"n_runs": g.count(), "median": g.median(),
                         "p10": g.quantile(0.10), "p90": g.quantile(0.90),
                         "mean": g.mean(), "max": g.max()}).reset_index()


# ============================================================================ censored mass
def _log_cdf_small_x(x: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log I_x(a, b) from I_x = x^a / (a B(a, b)) * 2F1(a, 1-b; a+1; x), for tiny x (float64)."""
    term = np.ones_like(a)
    total = np.ones_like(a)
    for n in range(200):
        term = term * (a + n) * (n + 1.0 - b) / ((a + 1.0 + n) * (n + 1.0)) * x
        total = total + term
        if np.all(np.abs(term) <= 1e-17 * np.abs(total)):
            break
    with np.errstate(invalid="ignore", divide="ignore"):
        return a * np.log(x) - np.log(a) - special.betaln(a, b) + np.log(total)


def log_censored_mass(side: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                      clamp: float = CLAMP) -> np.ndarray:
    """log P(A <= c) for side "lo" and log P(A >= 1 - c) for side "hi", A ~ Beta(alpha, beta).

    scipy.stats.beta ``logcdf`` / ``logsf`` in float64 from the float32 parameters. Where the
    mass underflows (scipy returns -inf) the leading series of the incomplete beta function is
    used (for "hi": the same series for Beta(beta, alpha), since 1 - A ~ Beta(beta, alpha)).
    """
    a, b = np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    lo = np.asarray(side) == "lo"
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(lo, beta_dist.logcdf(clamp, a, b), beta_dist.logsf(1.0 - clamp, a, b))
    bad = ~np.isfinite(out)
    if bad.any():
        sa, sb = np.where(lo, a, b)[bad], np.where(lo, b, a)[bad]
        out[bad] = _log_cdf_small_x(clamp, sa, sb)
    return out


# ============================================================================ buffer analysis
def _torch_logp(alpha: np.ndarray, beta: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Beta log-density as the agent computes it (torch float32), returned as float64."""
    with torch.no_grad():
        dist = torch.distributions.Beta(torch.as_tensor(np.asarray(alpha, dtype=np.float32)),
                                        torch.as_tensor(np.asarray(beta, dtype=np.float32)))
        return dist.log_prob(torch.as_tensor(np.asarray(actions, dtype=np.float32))
                             ).numpy().astype(np.float64)


def clamped_row_table(buf: Dict[str, np.ndarray], clamp: float = CLAMP) -> pd.DataFrame:
    """Every clamped POLICY row of a buffer with its training log-prob and censored log-mass.

    ``logp_stored`` is the buffer's ``old_logp``; ``logp_torch`` recomputes it with
    ``torch.distributions.Beta`` in float32 at the stored (clipped) action. The differences are
    log density - log censored mass (stored and recomputed).
    """
    raw = np.asarray(buf["raw"], dtype=np.float64)
    pm = np.asarray(buf["policy_mask"], dtype=bool)
    lo, hi = raw < clamp, raw > 1.0 - clamp
    idx = np.nonzero((lo | hi) & pm)[0]
    cols = ["row", "side", "stage", "alpha", "beta", "raw", "action", "logp_stored", "logp_torch",
            "log_mass", "diff_stored", "diff_torch"]
    if idx.size == 0:
        return pd.DataFrame(columns=cols)
    a32 = np.asarray(buf["alpha"], dtype=np.float32)[idx]
    b32 = np.asarray(buf["beta"], dtype=np.float32)[idx]
    act = np.asarray(buf["actions"], dtype=np.float32)[idx]
    lp = _torch_logp(a32, b32, act)
    side = np.where(lo[idx], "lo", "hi")
    mass = log_censored_mass(side, a32, b32, clamp)
    stored = np.asarray(buf["old_logp"], dtype=np.float64)[idx]
    return pd.DataFrame({"row": idx, "side": side, "stage": np.asarray(buf["stage"])[idx],
                         "alpha": a32.astype(np.float64), "beta": b32.astype(np.float64),
                         "raw": raw[idx], "action": act.astype(np.float64), "logp_stored": stored,
                         "logp_torch": lp, "log_mass": mass, "diff_stored": stored - mass,
                         "diff_torch": lp - mass})


def load_actor(npz_path: Path) -> BetaActor:
    """BetaActor (hidden from the file, c_min 100, mu_clamp 1e-6) from ``weights/u*.npz``."""
    with np.load(npz_path) as w:
        hidden = int(w["actor.l1.weight"].shape[0])
        actor = BetaActor(hidden, C_MIN, MU_CLAMP, torch.Generator().manual_seed(0))
        actor.load_state_dict({k[len("actor."):]: torch.as_tensor(np.array(w[k]))
                               for k in w.files if k.startswith("actor.")})
        if "conc_scale" in w.files:
            actor.conc_scale = float(w["conc_scale"])
    actor.eval()
    return actor


def load_actor_pre(buf: Dict[str, np.ndarray]) -> Optional[BetaActor]:
    """BetaActor of the policy that GENERATED the buffer (``actor_pre.*`` arrays), or None."""
    keys = [k for k in buf if k.startswith("actor_pre.")]
    if not keys:
        return None
    hidden = int(buf["actor_pre.l1.weight"].shape[0])
    actor = BetaActor(hidden, C_MIN, MU_CLAMP, torch.Generator().manual_seed(0))
    actor.load_state_dict({k[len("actor_pre."):]: torch.as_tensor(np.array(buf[k])) for k in keys})
    actor.conc_scale = float(buf["conc_scale_pre"]) if "conc_scale_pre" in buf else 1.0
    actor.eval()
    return actor


def _flat_grad(actor: BetaActor) -> torch.Tensor:
    """Flattened float64 gradient of all actor parameters (zeros where no gradient)."""
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1).double()
                      for p in actor.parameters()])


def gradient_share(actor: BetaActor, buf: Dict[str, np.ndarray], clamped: np.ndarray
                   ) -> Dict[str, object]:
    """Actor-loss gradient of ONE whole-buffer PPO pass, all policy rows vs clamped rows removed.

    The loss is the clipped surrogate ``-min(r A, clip(r, 1 - eps, 1 + eps) A)`` of
    ``masked_actor_loss`` over the policy rows, with the stored advantages normalised as
    ``(adv_raw - adv_norm_mean) / (adv_norm_std + 1e-8)`` and r = exp(logp_new - old_logp) at the
    given weights. Primary quantities use the common denominator n_policy, so that
    g_all = g_without + g_clamped: ``share = 1 - |g_without| / |g_all|`` and
    ``clamped_over_all = |g_clamped| / |g_all|``. ``share_renorm`` is the variant whose
    "without" loss is the mean over the remaining rows (denominator n_policy - n_clamped).
    """
    st = torch.as_tensor(np.asarray(buf["states"], dtype=np.float32))
    ac = torch.as_tensor(np.asarray(buf["actions"], dtype=np.float32))
    olp = torch.as_tensor(np.asarray(buf["old_logp"], dtype=np.float32))
    adv_raw = torch.as_tensor(np.asarray(buf["adv_raw"], dtype=np.float32))
    mean_t = torch.tensor(float(buf["adv_norm_mean"]), dtype=torch.float32)
    std_t = torch.tensor(float(buf["adv_norm_std"]), dtype=torch.float32)
    adv = (adv_raw - mean_t) / (std_t + ADV_NORM_EPS)
    pm = np.asarray(buf["policy_mask"], dtype=bool)
    cl = np.asarray(clamped, dtype=bool) & pm
    rows_all = torch.as_tensor(np.nonzero(pm)[0])
    rows_wo = torch.as_tensor(np.nonzero(pm & ~cl)[0])
    rows_cl = torch.as_tensor(np.nonzero(cl)[0])
    n_pol, n_cl = int(rows_all.numel()), int(rows_cl.numel())

    def grad(rows: torch.Tensor, scale: float):
        if rows.numel() == 0:
            return torch.zeros_like(_flat_grad(actor)), None
        actor.zero_grad(set_to_none=True)
        loss, ratio, _ = masked_actor_loss(actor, st, ac, olp, adv, rows, CLIP_EPS)
        (loss * scale).backward()
        return _flat_grad(actor), ratio.detach()

    g_all, ratio = grad(rows_all, 1.0)
    g_wo, _ = grad(rows_wo, float(rows_wo.numel()) / n_pol)
    g_cl, ratio_cl = grad(rows_cl, float(n_cl) / n_pol)
    g_rm, _ = grad(rows_wo, 1.0)
    na = float(g_all.norm())
    nw, nc, nr = float(g_wo.norm()), float(g_cl.norm()), float(g_rm.norm())
    dev = (ratio.double() - 1.0).abs()
    out: Dict[str, object] = {
        "n_policy_rows": n_pol, "n_clamped": n_cl, "grad_norm_all": na, "grad_norm_without": nw,
        "grad_norm_clamped": nc, "share": (1.0 - nw / na) if na > 0 else np.nan,
        "clamped_over_all": (nc / na) if na > 0 else np.nan,
        "grad_norm_without_renorm": nr, "share_renorm": (1.0 - nr / na) if na > 0 else np.nan,
        "ratio_dev_mean": float(dev.mean()), "ratio_dev_max": float(dev.max()),
        "ratio_dev_clamped_mean": (float((ratio_cl.double() - 1.0).abs().mean())
                                   if ratio_cl is not None else np.nan)}
    return out


def _open_buffer(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def analyse_run_buffers(run: RunInfo, phase_of_update: Dict[int, str],
                        csv_policy_hits: Dict[int, int]
                        ) -> Tuple[List[Dict], List[pd.DataFrame], List[Dict]]:
    """Buffer summaries, clamped-row tables and gradient-share rows of one run."""
    summ: List[Dict] = []
    rows_tabs: List[pd.DataFrame] = []
    grads: List[Dict] = []
    for path in sorted((run.path / "d1_buffers").glob("u*.npz")):
        buf = _open_buffer(path)
        gu, local = int(buf["global_update"]), int(buf["local"])
        key = dict(zip(RUN_KEYS, (run.group, run.arm, run.q, run.seed)),
                   phase=phase_of_update.get(gu, "?"), update=gu, local=local)
        raw = np.asarray(buf["raw"], dtype=np.float64)
        pm = np.asarray(buf["policy_mask"], dtype=bool)
        clamped = (raw < CLAMP) | (raw > 1.0 - CLAMP)
        tab = clamped_row_table(buf)
        n_cl = int(len(tab))
        s: Dict[str, object] = dict(key, n_rows=int(raw.size), n_policy_rows=int(pm.sum()),
                                    n_clamped_policy=n_cl,
                                    n_clamped_policy_lo=int((tab["side"] == "lo").sum()),
                                    n_clamped_policy_hi=int((tab["side"] == "hi").sum()),
                                    n_clamped_nonpolicy=int((clamped & ~pm).sum()),
                                    csv_policy_hits=csv_policy_hits.get(gu, -1))
        s["buffer_equals_csv"] = bool(csv_policy_hits.get(gu, -1) == n_cl)
        pol_idx = np.nonzero(pm)[0]
        lp = _torch_logp(buf["alpha"][pol_idx], buf["beta"][pol_idx], buf["actions"][pol_idx])
        stored_pol = np.asarray(buf["old_logp"], dtype=np.float64)[pol_idx]
        s["max_abs_logp_recompute_diff_policy_rows"] = float(np.max(np.abs(lp - stored_pol)))
        if n_cl:
            d = tab["diff_stored"].to_numpy()
            s.update(max_abs_logp_recompute_diff_clamped=float(
                np.max(np.abs(tab["logp_torch"] - tab["logp_stored"]))),
                diff_min=float(d.min()), diff_median=float(np.median(d)),
                diff_mean=float(d.mean()), diff_max=float(d.max()), note="")
            rows_tabs.append(pd.DataFrame(dict(key, **{c: tab[c] for c in tab.columns})))
        else:
            s["note"] = "no clamped rows"
        summ.append(s)
        wpath = run.path / "weights" / f"u{gu:05d}.npz"
        g: Dict[str, object] = dict(key, weights_file=str(wpath.relative_to(run.path))
                                    if wpath.exists() else "")
        pre = load_actor_pre(buf)
        if pre is not None:   # primary: the policy that generated the buffer (ratio = 1 at the start)
            g.update(gradient_share(pre, buf, clamped))
            g["weights_source"] = "pre-update actor stored in the buffer (generated this buffer)"
            if wpath.exists():   # secondary: the post-update export of the same update
                post = gradient_share(load_actor(wpath), buf, clamped)
                g.update({f"postexport_{k}": v for k, v in post.items()})
            g["note"] = "no clamped rows (share = 0 by construction)" if n_cl == 0 else ""
        elif wpath.exists():
            g.update(gradient_share(load_actor(wpath), buf, clamped))
            g["weights_source"] = "post-update weights export (buffer has no actor_pre; off-policy)"
            g["note"] = "no clamped rows (share = 0 by construction)" if n_cl == 0 else ""
        else:
            g["note"] = "weights file missing; gradient share not computed"
        grads.append(g)
    return summ, rows_tabs, grads


def logdiff_by_group(buffers: pd.DataFrame, clamped: pd.DataFrame) -> pd.DataFrame:
    """Per (group, q, phase): distribution of log density - log censored mass, clamped rows."""
    rows = []
    for key, g in buffers.groupby(["group", "q", "phase"], sort=True):
        row = dict(zip(["group", "q", "phase"], key), n_buffers=int(len(g)),
                   n_clamped_policy_rows=int(g["n_clamped_policy"].sum()),
                   n_clamped_lo=int(g["n_clamped_policy_lo"].sum()),
                   n_clamped_hi=int(g["n_clamped_policy_hi"].sum()),
                   n_clamped_nonpolicy_rows=int(g["n_clamped_nonpolicy"].sum()))
        if len(clamped):
            c = clamped[(clamped["group"] == key[0]) & (clamped["q"] == key[1])
                        & (clamped["phase"] == key[2])]
        else:
            c = clamped
        if len(c):
            d = c["diff_stored"].to_numpy(float)
            row.update(diff_min=float(d.min()), diff_p10=float(np.percentile(d, 10)),
                       diff_median=float(np.median(d)), diff_mean=float(d.mean()),
                       diff_p90=float(np.percentile(d, 90)), diff_max=float(d.max()),
                       max_abs_logp_recompute_diff=float(
                           np.max(np.abs(c["logp_torch"] - c["logp_stored"]))), note="")
        else:
            row["note"] = "no clamped rows"
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ flags
def evaluate_flags(prp: pd.DataFrame, grad: pd.DataFrame) -> pd.DataFrame:
    """M1 and M2 per (group, q, phase) from the per-run-phase and gradient-share tables.

    M1: median over runs of ``pol_frac`` > 1e-3 (strict). M2: number of runs whose maximum
    gradient share over the saved updates is > 1% (strict), compared with ``> 2`` (strict).
    Runs without gradient data are not counted as exceeding; their number is reported.
    """
    rows = []
    # per (group, q, phase), then the same flags pooled over q ("all": the literal "of the 20 runs")
    parts = [(key, g, (grad["q"] == key[1]) if len(grad) and "q" in grad.columns else None)
             for key, g in prp.groupby(["group", "q", "phase"], sort=True)]
    for key, g in prp.groupby(["group", "phase"], sort=True):
        parts.append(((key[0], "all", key[1]), g, None))
    for key, g, qmask in parts:
        k3 = dict(zip(["group", "q", "phase"], key))
        n_runs = int(len(g))
        med = float(g["pol_frac"].median())
        row = dict(k3, n_runs=n_runs, M1_median_over_runs=med, M1_threshold=M1_THRESHOLD,
                   M1_runs_above_threshold=int((g["pol_frac"] > M1_THRESHOLD).sum()),
                   M1_max_run_fraction=float(g["pol_frac"].max()),
                   M1_outcome="exceeded" if med > M1_THRESHOLD else "not exceeded")
        if len(grad) and "share" in grad.columns:
            gg = grad[(grad["group"] == key[0]) & (grad["phase"] == key[2])]
            if key[1] != "all":
                gg = gg[gg["q"] == key[1]]
            per_run = gg.dropna(subset=["share"]).groupby(["arm", "q", "seed"])["share"].max()
            cl_all = gg.get("clamped_over_all", pd.Series(dtype=float))
            max_cl = float(cl_all.max()) if cl_all.notna().any() else np.nan
        else:
            per_run, max_cl = pd.Series(dtype=float), np.nan
        hits = int((per_run > M2_SHARE_THRESHOLD).sum())
        outcome = ("not evaluable (no gradient data)" if len(per_run) == 0 else
                   "exceeded" if hits > M2_MAX_RUNS else "not exceeded")
        note = ("" if n_runs >= 20 else f"{n_runs} runs (< 20); M2 evaluated as "
                f"'more than {M2_MAX_RUNS} runs' literally")
        row.update(M2_n_runs_with_gradient_data=int(len(per_run)),
                   M2_runs_share_above_threshold=hits, M2_share_threshold=M2_SHARE_THRESHOLD,
                   M2_runs_needed_more_than=M2_MAX_RUNS,
                   M2_max_share=float(per_run.max()) if len(per_run) else np.nan,
                   M2_max_clamped_over_all=max_cl, M2_outcome=outcome, note=note)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================ figure
def make_figures(by_local: pd.DataFrame, prp: pd.DataFrame, fig_dir: Path) -> List[str]:
    """Clamp-hit fraction against the local update: a panel per (q, phase), a figure per group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                                "axes.spines.top": False, "axes.spines.right": False})
    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e1e0dc", "#2a78d6"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for group, gb in by_local.groupby("group", sort=True):
        qs, phases = sorted(gb["q"].unique()), sorted(gb["phase"].unique())
        top = max(2.0 * M1_THRESHOLD, 1.05 * float(gb["p90"].max()))
        fig, axes = plt.subplots(len(qs), len(phases), sharey=True, squeeze=False,
                                 figsize=(4.0 * len(phases) + 0.6, 2.7 * len(qs) + 0.6))
        for i, q in enumerate(qs):
            for j, ph in enumerate(phases):
                ax = axes[i][j]
                d = gb[(gb["q"] == q) & (gb["phase"] == ph)]
                if d.empty:
                    ax.axis("off")
                    continue
                ax.fill_between(d["local"], d["p10"], d["p90"], color=blue, alpha=0.22, lw=0)
                ax.plot(d["local"], d["median"], color=blue, lw=1.2)
                ax.axhline(M1_THRESHOLD, color=muted, lw=0.8, ls="--")
                r = prp[(prp["group"] == group) & (prp["q"] == q) & (prp["phase"] == ph)]
                ax.set_title(f"q = {q:g}, phase {ph}  (n = {len(r)} runs)", color=ink, fontsize=9,
                             loc="left")
                note = (f"median over runs of the phase fraction: {r['pol_frac'].median():.3g}\n"
                        f"max over runs: {r['pol_frac'].max():.3g}")
                ax.text(0.01, 0.97, note, transform=ax.transAxes, ha="left", va="top",
                        fontsize=7.5, color=muted)
                ax.text(0.99, M1_THRESHOLD, "M1 threshold 1e-3 ",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=7.5, color=muted)
                ax.set_ylim(-0.03 * top, top)
                ax.grid(True, color=grid, lw=0.5)
                ax.tick_params(colors=muted)
                if i == len(qs) - 1:
                    ax.set_xlabel("local update", color=muted)
                if j == 0:
                    ax.set_ylabel("clamp-hit fraction", color=muted)
        fig.suptitle(f"{group}: raw draws at a clamp / learner policy rows per update "
                     "(line: median over runs, band: p10-p90)", fontsize=9, color=ink,
                     x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        stem = "d1_clamp_fraction_" + re.sub(r"[^A-Za-z0-9_.-]+", "__", group)
        for ext in ("png", "pdf"):
            p = fig_dir / f"{stem}.{ext}"
            fig.savefig(p, dpi=150)
            written.append(p.name)
        plt.close(fig)
    return written


# ============================================================================ report
def _f(x) -> str:
    """Compact number formatting for the report (blank for missing)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return "0" if x == 0 else f"{x:.4g}"


def _md(df: pd.DataFrame, cols: Sequence[str], heads: Optional[Sequence[str]] = None) -> List[str]:
    """Markdown table of ``cols`` (a missing note is blank, any other missing value is n/a)."""
    out = ["| " + " | ".join(heads or cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = [r.get(c, np.nan) for c in cols]
        cells = [v if isinstance(v, str) else ("" if c == "note" and pd.isna(v) else _f(v))
                 for c, v in zip(cols, vals)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _read_table(path: Path) -> pd.DataFrame:
    """Read a table back from its CSV (an empty file gives an empty frame)."""
    try:
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_report(out: Path, report_path: Path, command: str, roots: Sequence[Tuple[str, Path]],
                 figures: Sequence[str], skipped: Sequence[Dict[str, str]]) -> None:
    """Write the markdown report from the CSV tables in ``out`` (no recommendations)."""
    t = {n: _read_table(out / f"{n}.csv") for n in D1_FILES}
    try:
        rel = str(Path(out).resolve().relative_to(ROOT))
    except ValueError:
        rel = str(out)
    runs, bg, bufs, ld = t["d1_runs"], t["d1_by_group"], t["d1_buffers"], t["d1_logdiff_by_group"]
    L: List[str] = [
        "# D1: likelihood consistency of clipped Beta samples (T57 issue 7)", "",
        "Descriptive report generated by `tools/v2/d1_clamp_analysis.py`; no fix is applied and no",
        f"recommendation is made. Every number below is transcribed from a CSV in `{rel}/`",
        "(the source file is named under each table).", "",
        "## 1. Data and definitions", "",
        "- Roots: " + "; ".join(f"`{lab}` = `{p}`" for lab, p in roots) + ".",
        f"- Runs analysed: {len(runs)} (`{rel}/d1_runs.csv`); run-like directories skipped",
        f"  for lack of `v2_updates.csv`: {len(skipped)}.",
        *([f"- Runs without D1 columns by construction (a pathwise phase P draws no action): "
           f"{int((runs['n_rows_without_d1'] > 0).sum())} of {len(runs)} runs "
           f"(arms: {', '.join(sorted(set(runs.loc[runs['n_rows_without_d1'] > 0, 'arm'].astype(str))))}); "
           "they are counted in 'Runs analysed' but contribute no rows to any table."]
          if (runs["n_rows_without_d1"] > 0).any() else []),
        f"- A raw draw is a clamp hit when it is below {CLAMP:g} (lo) or above 1 - {CLAMP:g} (hi).",
        "  Counts come from the `d1_*` columns of `v2_updates.csv` (raw `rng.beta` output before",
        "  the clip).",
        "- Learner policy rows are the learner rows that enter the actor loss: all rows in phase A",
        "  (stage 2) and the stage-1 rows of a frozen phase B. The stage-2 rows of a frozen",
        "  phase B are reported separately and labelled non-policy; the opponent's draws are",
        "  never trained on.",
        "- Per-update statistics: min / median / p90 / max (numpy linear percentiles) are taken",
        "  over (i) all (run, update) pairs pooled, columns `upd_frac_*`, and (ii) over runs",
        "  of the per-run phase-summed fraction, columns `run_frac_*`. `frac_sum` is hits /",
        "  rows summed over all runs and updates of the phase.", "",
        "## 2. Clamp-hit fraction of learner policy rows, per q and phase", "",
        f"Source: `{rel}/d1_by_group.csv`, category `learner_policy`.", ""]
    L += _md(bg[bg["category"] == "learner_policy"],
             ["group", "q", "phase", "n_runs", "n_updates", "rows_sum", "lo_sum", "hi_sum",
              "frac_sum", "upd_frac_min", "upd_frac_median", "upd_frac_p90", "upd_frac_max",
              "run_frac_min", "run_frac_median", "run_frac_p90", "run_frac_max"])
    L += ["", "## 3. By stage, by region at the final stage, and the opponent", "",
          f"Source: `{rel}/d1_by_group.csv` (all other categories). `policy_status` states whether",
          "the rows enter the actor loss. `learner_final_in` / `learner_final_out` split the",
          "final-stage learner rows by |d| < 2q vs |d| >= 2q.", ""]
    L += _md(bg[bg["category"] != "learner_policy"],
             ["group", "q", "phase", "category", "policy_status", "rows_sum", "lo_sum", "hi_sum",
              "frac_sum", "upd_frac_max", "run_frac_max"])
    L += ["", "## 4. Rows with alpha < 1 or beta < 1 (learner policy rows)", "",
          f"Source: `{rel}/d1_alpha_beta_lt1.csv`.", ""]
    L += _md(t["d1_alpha_beta_lt1"], ["group", "q", "phase", "n_runs", "pol_rows_sum",
                                      "n_alpha_lt1_sum", "n_beta_lt1_sum", "alpha_min", "beta_min"])
    L += ["", "## 5. Saved buffers: training log-density against the censored log-mass", "",
          f"Source: `{rel}/d1_buffers.csv` (per buffer), `{rel}/d1_clamped_rows.csv` (per clamped",
          f"row), `{rel}/d1_logdiff_by_group.csv` (distribution of log density - log censored",
          "mass). Log mass = log P(A <= c) for a low clamp and log P(A >= 1 - c) for a high clamp",
          "(scipy.stats.beta in float64 from the float32 alpha, beta). Only rows in `policy_mask`",
          "are considered.", ""]
    if len(bufs):
        n_eq = int(bufs["buffer_equals_csv"].astype(bool).sum())
        L += [f"- Buffers read: {len(bufs)}; clamped policy rows in them: "
              f"{int(bufs['n_clamped_policy'].sum())}; clamped non-policy rows: "
              f"{int(bufs['n_clamped_nonpolicy'].sum())}.",
              "- The buffer's clamped policy-row count equals the CSV policy-row hit count of the",
              f"  same update in {n_eq} of {len(bufs)} buffers (column `buffer_equals_csv`).", ""]
        L += _md(ld, ["group", "q", "phase", "n_buffers", "n_clamped_policy_rows", "n_clamped_lo",
                      "n_clamped_hi", "n_clamped_nonpolicy_rows", "diff_min", "diff_median",
                      "diff_max", "note"])
    else:
        L += ["No `d1_buffers/u*.npz` were found for the analysed runs."]
    gr = t["d1_gradient_share"]
    L += ["", "## 6. Gradient share of the clamped rows", "",
          f"Source: `{rel}/d1_gradient_share.csv`. At the actor that GENERATED the buffer (the",
          "pre-update weights stored in the buffer file as `actor_pre.*`, so the ratio is exactly 1",
          "and `ratio_dev_mean` is 0 up to float rounding; the post-update export of the same",
          "global update is kept as a secondary comparison in the `postexport_*` columns, where the",
          "ratio deviates), the actor-loss gradient norm of ONE whole-buffer clipped-surrogate pass (no",
          "minibatching, stored advantages normalised with the stored mean / SD) with all policy",
          "rows against the clamped rows removed. `share` = 1 - |g_without| / |g_all| (common",
          "denominator n_policy); `clamped_over_all` = |g_clamped| / |g_all|.", ""]
    if len(gr):
        agg = []
        for key, g in gr.groupby(["group", "q", "phase"], sort=True):
            agg.append(dict(
                zip(["group", "q", "phase"], key), n_buffers=len(g),
                n_with_gradient=int(g["share"].notna().sum()),
                n_clamped=int(g["n_clamped"].fillna(0).sum()), max_share=g["share"].max(),
                max_clamped_over_all=g["clamped_over_all"].max(),
                grad_norm_all_median=g["grad_norm_all"].median(),
                ratio_dev_mean_median=g["ratio_dev_mean"].median(),
                ratio_dev_max_max=g["ratio_dev_max"].max()))
        L += _md(pd.DataFrame(agg),
                 ["group", "q", "phase", "n_buffers", "n_with_gradient", "n_clamped", "max_share",
                  "max_clamped_over_all", "grad_norm_all_median", "ratio_dev_mean_median",
                  "ratio_dev_max_max"])
    L += ["", "## 7. Materiality flags (pre-registered, report-only)", "",
          f"Source: `{rel}/d1_flags.csv`. M1: the median over runs of the per-phase clamp fraction",
          f"among learner policy rows exceeds {M1_THRESHOLD:g} (strict). M2: the gradient share of",
          f"the clamped rows exceeds {M2_SHARE_THRESHOLD:.0%} at any saved update in more than",
          f"{M2_MAX_RUNS} of the runs (strict; evaluated with the number of runs available).", ""]
    headline = t["d1_flags"][t["d1_flags"]["group"] == "v11_reproduction"]
    if len(headline):
        hl = ["## 0. Headline: the locked pipeline (the C-R1 runs, group `v11_reproduction`)", "",
              f"Source: `{rel}/d1_flags.csv`, `{rel}/d1_by_group.csv` (descriptive; no fix applied). Rows for q = `all` "
              "pool both q (the pre-registered reading 'of the 20 runs'); the rows per q are shown beside them.", ""]
        for _, r in headline.iterrows():
            hl.append(f"- q = {r['q']}, phase {r['phase']} ({int(r['n_runs'])} runs): M1 {r['M1_outcome']} "
                      f"(median over runs of the clamp fraction among learner policy rows = "
                      f"{_f(r['M1_median_over_runs'])} against {_f(r['M1_threshold'])}); M2 {r['M2_outcome']} "
                      f"({int(r['M2_runs_share_above_threshold'])} of {int(r['M2_n_runs_with_gradient_data'])} "
                      f"runs have a clamped-row gradient share above {_f(r['M2_share_threshold'])} at a saved "
                      f"update, needs more than {int(r['M2_runs_needed_more_than'])}; max share "
                      f"{_f(r['M2_max_share'])}).")
        hl.append("")
        k = L.index("## 1. Data and definitions")
        L[k:k] = hl
    for _, r in t["d1_flags"].iterrows():
        note = f" Note: {r['note']}." if isinstance(r["note"], str) and r["note"] else ""
        L.append(
            f"- {r['group']}, q = {r['q']}, phase {r['phase']} ({int(r['n_runs'])} runs): "
            f"M1 {r['M1_outcome']} (median over runs = {_f(r['M1_median_over_runs'])}, threshold "
            f"{_f(r['M1_threshold'])}, {int(r['M1_runs_above_threshold'])} run(s) above, max run "
            f"fraction {_f(r['M1_max_run_fraction'])}); M2 {r['M2_outcome']} "
            f"({int(r['M2_runs_share_above_threshold'])} of "
            f"{int(r['M2_n_runs_with_gradient_data'])} runs with gradient data above "
            f"{_f(r['M2_share_threshold'])}, max share {_f(r['M2_max_share'])}, needs more than "
            f"{int(r['M2_runs_needed_more_than'])}; largest |g_clamped| / |g_all| over the saved "
            f"buffers {_f(r['M2_max_clamped_over_all'])}, information only)." + note)
    L += ["", "## 8. Figures", ""]
    L += [f"- `{rel}/figures/{n}`" for n in figures] or ["- none"]
    L += ["", "## 9. Reproduce", "", "```", command, "```", "",
          "## 10. Limitations", "",
          "Counts are of raw draws of the rollout policy; a clamp hit needs a draw in the extreme",
          f"{CLAMP:g} tail of Beta(alpha, beta), so a zero count says nothing about policies with",
          "other (alpha, beta). The gradient share is computed at the pre-update actor stored in the",
          "buffer (collected before update u; three buffers per phase and run), with one",
          "whole-buffer pass instead of the minibatched epochs of the real update, and with the",
          "stored advantages, so it is a snapshot measure and not the cumulative effect of clamped",
          "rows on training; the share is signed (it is negative when the clamped rows' gradient",
          "opposes the rest). The log-density of a clamped row is of order -500 (alpha, beta of",
          "order 50), so its ratio at the post-update export can be far from 1 (columns",
          "`postexport_ratio_dev_clamped_mean`) and the clipped surrogate then removes its gradient on one",
          "side. In a frozen phase B the stage-2 draws may be discarded (continuation",
          "by the Beta mean), in which case their counts describe unused draws. The stored float32",
          "action at a high clamp is not exactly 1 - c (float32 spacing near 1 is 6e-8). The M2",
          "count uses the runs and saved updates present; the architecture constants (c_min 100,",
          "mu_clamp 1e-6) are those of the locked record and are not read from the run."]
    if skipped:
        L += ["", "Skipped directories: "
              + "; ".join(f"`{s['path']}` ({s['reason']})" for s in skipped)]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L) + "\n")


# ============================================================================ driver
CLAMPED_ROW_COLUMNS = RUN_KEYS + ["phase", "update", "local", "row", "side", "stage", "alpha",
                                  "beta", "raw", "action", "logp_stored", "logp_torch",
                                  "log_mass", "diff_stored", "diff_torch"]


def run_analysis(roots: Sequence[Tuple[str, Path]], out: Path,
                 arms: Optional[Sequence[str]] = None, command: str = "",
                 report_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Analyse every run under ``roots``; write the tables, the figures and (optionally) the report.

    Args:
        roots: ``(label, path)`` run roots.
        out: Output directory (``d1_*.csv``, ``d1_summary.json``, ``figures/``).
        arms: Optional arm-directory filter.
        command: Command line recorded in the summary and in the report.
        report_path: If given, the markdown report is written there.

    Returns:
        The tables by file stem.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs, skipped = discover_runs(roots, arms)
    ups, inv, bsum, btabs, grads = [], [], [], [], []
    for run in runs:
        u, n_drop = load_run_updates(run)
        buf_dir = run.path / "d1_buffers"
        n_buf = len(list(buf_dir.glob("u*.npz"))) if buf_dir.exists() else 0
        inv.append({"group": run.group, "arm": run.arm, "q": run.q, "seed": run.seed,
                    "path": str(run.path), "n_update_rows": int(len(u)),
                    "n_rows_without_d1": n_drop,
                    "phases": ",".join(sorted(set(u["phase"]))) if len(u) else "",
                    "n_buffers": n_buf})
        if u.empty:
            continue
        ups.append(u)
        if n_buf:
            pmap = dict(zip(u["update"].astype(int), u["phase"]))
            hits = dict(zip(u["update"].astype(int), u["pol_hit"].astype(int)))
            s, c, g = analyse_run_buffers(run, pmap, hits)
            bsum += s
            btabs += c
            grads += g
    if not ups:
        raise SystemExit(f"no runs with d1 columns found under {[str(p) for _, p in roots]}")
    u = pd.concat(ups, ignore_index=True)
    prp = per_run_phase_table(u)
    by_local = by_local_table(u)
    buffers, grad = pd.DataFrame(bsum), pd.DataFrame(grads)
    clamped = (pd.concat(btabs, ignore_index=True) if btabs
               else pd.DataFrame(columns=CLAMPED_ROW_COLUMNS))
    t = {"d1_runs": pd.DataFrame(inv),
         "d1_per_update": u[[c for c in u.columns if not c.startswith("c_")]],
         "d1_per_run_phase": prp, "d1_by_group": by_group_table(u),
         "d1_alpha_beta_lt1": alpha_beta_table(prp), "d1_clamp_fraction_by_local": by_local,
         "d1_buffers": buffers, "d1_clamped_rows": clamped,
         "d1_logdiff_by_group": (logdiff_by_group(buffers, clamped) if len(buffers)
                                 else pd.DataFrame()),
         "d1_gradient_share": grad, "d1_flags": evaluate_flags(prp, grad)}
    for name, df in t.items():
        df.to_csv(out / f"{name}.csv", index=False)
    figs = make_figures(by_local, prp, out / "figures")
    meta = {"roots": {lab: str(p) for lab, p in roots},
            "arms_filter": None if arms is None else list(arms),
            "n_runs": len(runs), "skipped": skipped, "figures": figs, "command": command,
            "thresholds": {"M1": M1_THRESHOLD, "M2_share": M2_SHARE_THRESHOLD,
                           "M2_more_than_runs": M2_MAX_RUNS, "clamp": CLAMP,
                           "clip_eps": CLIP_EPS}}
    (out / "d1_summary.json").write_text(json.dumps(meta, indent=1) + "\n")
    if report_path is not None:
        write_report(out, Path(report_path), command, roots, figs, skipped)
    return t


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roots", nargs="+", required=True, help="[LABEL=]PATH run roots")
    ap.add_argument("--out", required=True, help="output directory (d1_*.csv, figures/)")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="keep only these arm directory names")
    ap.add_argument("--report-path", default=None, help="write the markdown report here")
    a = ap.parse_args(argv)
    roots = [parse_root_spec(s) for s in a.roots]
    cmd = ("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "
           "tools/v2/d1_clamp_analysis.py " + " ".join(sys.argv[1:] if argv is None else argv))
    report = Path(a.report_path) if a.report_path else None
    t = run_analysis(roots, Path(a.out), a.arms, cmd, report)
    cols = ["group", "q", "phase", "n_runs", "M1_median_over_runs", "M1_outcome",
            "M2_runs_share_above_threshold", "M2_n_runs_with_gradient_data", "M2_outcome"]
    print(t["d1_flags"][cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
