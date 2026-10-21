#!/usr/bin/env python3
"""Tables of ``reports/t3/r3/02_proposal.md``: the model projection of the development target, the operating characteristics
of the development target and of the later confirmation, the budgets, and the gradient of the clamped-draw likelihood.
Writes ``reports/t3/r3/tables/<name>.md`` (same convention as ``diagnosis_tables.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from diagnosis_tables import DIAG, OUT, fmt, md, rd  # noqa: E402


# ---------------------------------------------------------------------------------- operating characteristics
def target_prob(p50: float, p60: float, n: int = 20, total: int = 38, per_q: int = 18) -> float:
    """P(at least ``total`` of 2n pass and at least ``per_q`` of n at each q) for independent runs with pass probabilities p50, p60."""
    k50 = np.arange(n + 1)
    k60 = np.arange(n + 1)
    a, b = binom.pmf(k50, n, p50), binom.pmf(k60, n, p60)
    tot = 0.0
    for i in k50:
        for j in k60:
            if i >= per_q and j >= per_q and i + j >= total:
                tot += a[i] * b[j]
    return float(tot)


def confirm_prob(f50: float, f60: float, n: int = 40, total: int = 6, per_q: int = 4) -> float:
    """P(at most ``total`` of 2n fail and at most ``per_q`` of n at each q) for independent runs with failure probabilities f50, f60."""
    a, b = binom.pmf(np.arange(n + 1), n, f50), binom.pmf(np.arange(n + 1), n, f60)
    tot = 0.0
    for i in range(per_q + 1):
        for j in range(per_q + 1):
            if i + j <= total:
                tot += a[i] * b[j]
    return float(tot)


def t_oc() -> str:
    rows = [{"per-run pass probability p (both q)": p, "P(development target: >= 38 of 40 and >= 18 of 20 per q)": target_prob(p, p)}
            for p in (0.99, 0.97, 0.95, 0.90, 0.875)]
    rows2 = [{"per-run failure probability f (both q)": f, "P(confirmation: <= 6 of 80 and <= 4 of 40 per q)": confirm_prob(f, f)}
             for f in (0.01, 0.03, 0.05, 0.10, 0.15)]
    return (md(rows, nd=4) + "\n\nThe later confirmation (D6):\n\n" + md(rows2, nd=4))


# ---------------------------------------------------------------------------------- projection
def cp(k: int, n: int, a: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided interval of k / n."""
    lo = 0.0 if k == 0 else beta_dist.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def project(f: Dict[int, float], g: Dict[int, float], mode: str = "gate") -> Dict[int, float]:
    """Failure probability per q after one fresh re-draw of stage 3 for the runs that fire.

    ``f`` = probability that the stage-3 check fires, ``g`` = probability that a run that does not fire fails (stage 2 / stage 1 /
    shape parts). A fired attempt that fires again fails; a second attempt that does not fire fails with probability ``g``.
    Assumes independent fresh draws from the same distribution as a first attempt (the model of the table).
    """
    out = {}
    for q in (50, 60):
        out[q] = f[q] * (f[q] + (1.0 - f[q]) * g[q]) + (1.0 - f[q]) * g[q]
    return out


def t_projection() -> str:
    c = rd("retro_checks_local1600.csv")
    fail = {q: int((~c[c.q == q]["pass"]).sum()) for q in (50, 60)}
    fire = {q: int((c[c.q == q].Delta > 0.005).sum()) for q in (50, 60)}
    nonfire_fail = {q: int(((c.q == q) & ~c["pass"] & (c.Delta <= 0.005)).sum()) for q in (50, 60)}
    form_fire = {q: int(((c.q == q) & (c.R > 0.7)).sum()) for q in (50, 60)}
    form_nonfire_fail = {q: int(((c.q == q) & ~c["pass"] & (c.R <= 0.7)).sum()) for q in (50, 60)}

    def scen(name: str, trig_fire: Dict[int, int], trig_nf_fail: Dict[int, int], bound: str) -> Dict[str, object]:
        f, g = {}, {}
        for q in (50, 60):
            nf = 40 - trig_fire[q]
            if bound == "central":
                f[q], g[q] = trig_fire[q] / 40, trig_nf_fail[q] / nf
            elif bound == "pessimistic":
                f[q], g[q] = cp(trig_fire[q], 40)[1], cp(trig_nf_fail[q], nf)[1]
            else:
                f[q], g[q] = cp(trig_fire[q], 40)[0], cp(trig_nf_fail[q], nf)[0]
        fl = project(f, g)
        # a formation-level trigger fires on runs whose second draw fails with the unconditional failure rate of the q
        if name.startswith("formation"):
            fl = {q: f[q] * (fail[q] / 40) + (1.0 - f[q]) * g[q] for q in (50, 60)}
        return {"arm / trigger": name, "assumed rates": bound, "fail q50": fl[50], "fail q60": fl[60],
                "expected failures in 40 runs": 20 * (fl[50] + fl[60]), "P(development target)": target_prob(1 - fl[50], 1 - fl[60])}
    rows = []
    base = {q: fail[q] / 40 for q in (50, 60)}
    rows.append({"arm / trigger": "bw (control, observed rates)", "assumed rates": "central", "fail q50": base[50], "fail q60": base[60],
                 "expected failures in 40 runs": 20 * (base[50] + base[60]), "P(development target)": target_prob(1 - base[50], 1 - base[60])})
    for bound in ("central", "pessimistic", "optimistic"):
        rows.append(scen("gate-level: Delta > 0.005", fire, nonfire_fail, bound))
    for bound in ("central",):
        rows.append(scen("formation-level: R > 0.7", form_fire, form_nonfire_fail, bound))
    note = (f"\n\nRates from the 80 recorded runs (`retro_checks_local1600.csv`): failures q50 {fail[50]} of 40, q60 {fail[60]} of 40; the gate-level trigger fires on "
            f"{fire[50]} (q50) and {fire[60]} (q60) runs and misses {nonfire_fail[50]} and {nonfire_fail[60]} failures; the formation-level trigger fires on "
            f"{form_fire[50]} and {form_fire[60]}. 'pessimistic' / 'optimistic' use the upper / lower limit of the exact 95 % interval of every rate. "
            f"**This is a model, not a result**: it assumes independent runs, a re-drawn stage 3 distributed as a first attempt, a run that fires again failing, "
            f"and rates estimated from 40 runs per q; it is shown to size the target, not to predict the wave.")
    return md(rows) + note


# ---------------------------------------------------------------------------------- budgets
def t_budgets() -> str:
    P, ep, mb = 512, 20, 1
    rows = []
    stages = {"stage 3": 1600, "stage 2": 1600, "stage 1": 600}
    tot = sum(stages.values())
    pol = {k: v * 512 for k, v in stages.items()}
    steps = {"stage 3": 1600 * 20, "stage 2": 1600 * 20, "stage 1": 600 * 20}
    chk = {"stage 3": 64, "stage 2": 64, "stage 1": 24}
    rows.append({"case": "bw, and bwA / bwAB when the trigger does not fire", "updates": tot, "policy rows": sum(pol.values()), "minibatch steps": sum(steps.values()),
                 "development checks": sum(chk.values()), "exports (frozen path)": 152, "relative to bw": 1.0})
    rows.append({"case": "bwA / bwAB when the trigger fires (stage 3 trained twice)", "updates": tot + 1600, "policy rows": sum(pol.values()) + pol["stage 3"],
                 "minibatch steps": sum(steps.values()) + steps["stage 3"], "development checks": sum(chk.values()) + 64,
                 "exports (frozen path)": "152 (+ 64 in attempt1/)", "relative to bw": (tot + 1600) / tot})
    c = rd("retro_checks_local1600.csv")
    k = int((c.Delta > 0.005).sum())
    lo, hi = cp(k, 80)
    for nm, p in (("expected, fire rate 7 / 80", k / 80), ("expected, lower limit of the 95 % interval", lo), ("expected, upper limit", hi)):
        rows.append({"case": f"bwA / bwAB {nm} = {p:.4f}", "updates": tot + 1600 * p, "policy rows": sum(pol.values()) + pol["stage 3"] * p,
                     "minibatch steps": sum(steps.values()) + steps["stage 3"] * p, "development checks": sum(chk.values()) + 64 * p,
                     "exports (frozen path)": 152, "relative to bw": (tot + 1600 * p) / tot})
    import glob
    w, s3 = [], []
    for p in glob.glob(str(ROOT / "results/t3_r1/runs/q*/seed*/bw/status.json")) + glob.glob(str(ROOT / "results/t3_r2/confirmation/q*/seed*/bw/status.json")):
        w.append(json.load(open(p))["total_wall_sec"])
        s3.append(json.load(open(p.replace("status.json", "ms_run_summary.json")))["phase_timing"]["stage3"]["wall_sec"])
    note = (f"\n\nWall time of the 80 recorded `bw` runs (`status.json` `total_wall_sec`): median {np.median(w):.0f} s, range {min(w):.0f}-{max(w):.0f} s; the stage-3 phase "
            f"(`ms_run_summary.json`) median {np.median(s3):.0f} s, i.e. {np.median(s3) / np.median(w):.0%} of a run. One retraining therefore costs about "
            f"{np.median(s3):.0f} s; a wave of 40 `bw` runs costs about {40 * np.median(w) / 3600:.1f} CPU-hours, of 40 runs of an arm with the expected extra stage-3 work "
            f"about {40 * (np.median(w) + (k / 80) * np.median(s3)) / 3600:.1f}. The censored update measured on a synthetic batch of 512 rows with 466 flagged rows "
            f"(`agent.update` with `clamp_side`, mean of 3 calls, one thread) took 1.17 times the plain update (0.1355 against 0.1161 s); a regime run with about 57 % flagged rows "
            f"costs less.")
    return md(rows) + note


# ---------------------------------------------------------------------------------- the clamped-draw likelihood
def t_clamp_grad() -> str:
    import torch
    from utils.beta_tail import log_betainc_small_x
    c = 1e-6
    beta = torch.tensor(150.0, dtype=torch.float64)
    rows = []
    for al in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0, 3.0):
        a = torch.tensor(al, dtype=torch.float64, requires_grad=True)
        gd, = torch.autograd.grad(torch.distributions.Beta(a, beta).log_prob(torch.tensor(c, dtype=torch.float64)), a)
        a2 = torch.tensor(al, dtype=torch.float64, requires_grad=True)
        lm = log_betainc_small_x(c, a2, beta)
        gm, = torch.autograd.grad(lm, a2)
        rows.append({"alpha": al, "d log density at a = 1e-6 / d alpha": gd.item(), "d log P(A <= 1e-6) / d alpha": gm.item(), "P(A <= 1e-6)": lm.exp().item()})
    return md(rows)


TABLES = {"oc": t_oc, "projection": t_projection, "budgets": t_budgets, "clamp_grad": t_clamp_grad}


def main() -> int:
    """Write the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (OUT / f"{name}.md").write_text(fn() + "\n")
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
