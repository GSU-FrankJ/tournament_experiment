#!/usr/bin/env python3
"""Why the first-order residual is larger on the d < 0 side: one stored v2.0 export, read-only.

For the stage-2 policy of a stored export (default: rehearsal_v2_0, q = 50, seed 10501, u1600) the development-tier
verifier gives the policy mean e_hat(d) and the one-step best-response effort a_dev(d); the residual is
r(d) = |e_hat(d) - a_dev(d)|. The policy itself is nearly symmetric in d, the best response is not: linearising the
first-order condition 2 k e = DW f_xi(d + e - e_opp) around a symmetric profile, a deviation c of the opponent from
equilibrium is answered by -c a / (2k - a) for d < 0 (the density f_xi is increasing there) and by +c a / (2k + a)
for d > 0, a = DW / (4 q^2). A policy that is off by c everywhere (and plays against itself) therefore has the
residual c 2k/(2k - a) at d < 0 and c 2k/(2k + a) at d > 0: 3.33 c and 0.59 c at q = 50, 1.95 c and 0.67 c at q = 60.

Usage:
    python tools/ms/residual_asymmetry_example.py --root-v2 <.../v2_T2_locked> [--q 50 --seed 10501 --u 1600]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ppo_curriculum import BetaActor  # noqa: E402
from envs.curriculum_env import GameSpec  # noqa: E402
from run.run_final_dp_br import make_policy_fns  # noqa: E402
from utils.dp_br_verifier import DEV_CONFIG, verify  # noqa: E402


class _Shim:
    """Exposes ``beta_params`` of a bare actor for ``make_policy_fns``."""

    def __init__(self, net: BetaActor):
        self.actor = net

    def beta_params(self, obs, net=None):
        net = self.actor if net is None else net
        with torch.no_grad():
            a, b = net(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
        return a.numpy(), b.numpy()


def main() -> int:
    """Print e_hat(+-d), a_dev(+-d), r(+-d) at selected d and the analytic amplification factors."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-v2", required=True)
    p.add_argument("--run", default="rehearsal_v2_0")
    p.add_argument("--q", type=int, nargs="+", default=[50, 60])
    p.add_argument("--seed", type=int, nargs="+", default=[10501, 10503])
    p.add_argument("--u", type=int, default=1600)
    a = p.parse_args()
    proto = json.load(open(ROOT / "protocols" / "v2_T2_locked_v2_0.json"))
    for q, seed in zip(a.q, a.seed):
        game = proto["records"][str(q)]["game"]
        spec = GameSpec(**{k: game[k] for k in ("w_h", "w_l", "k", "q", "T", "e_min", "e_max")})
        z = np.load(Path(a.root_v2) / a.run / f"q{q}" / f"seed{seed}" / "weights" / f"u{a.u:05d}.npz")
        net = BetaActor(64, 100.0, 1e-6, torch.Generator().manual_seed(0))
        if "actor_variant" in z.files:    # MS-R3: this MS-R1 example rebuilds the tanh d / B actor only
            raise ValueError(f"export of actor variant {str(z['actor_variant'])!r}: this tool reads t1 exports only")
        net.load_state_dict({k[6:]: torch.as_tensor(z[k]) for k in z.files if k.startswith("actor.")})
        net.eval()
        mf, bf = make_policy_fns(_Shim(net), spec)
        res = verify(mf, w_h=spec.w_h, w_l=spec.w_l, k=spec.k, q=spec.q, T=2, cfg=DEV_CONFIG, beta_fn=bf)
        s2 = res.stages[2]
        d, e, ad = s2.d_grid, s2.e_hat, s2.a_dev
        mid = len(d) // 2
        r = np.abs(e - ad)
        pos = np.abs(d) < 2 * q
        dw, k = spec.dw, spec.k
        amp = dw / (4.0 * q * q)
        print(f"q={q} seed={seed} {a.run} u{a.u}: e_hat(0)={e[mid]:.2f} a_dev(0)={ad[mid]:.2f}  "
              f"a = DW/(4q^2) = {amp:.4g}, 2k = {2 * k:.4g}; amplification 2k/(2k-a) = {2 * k / (2 * k - amp):.3f} "
              f"(d<0), 2k/(2k+a) = {2 * k / (2 * k + amp):.3f} (d>0)")
        for dd in (4, 8, 16, 24, 40, 60, 80, 96):
            i, j = mid + dd // 4, mid - dd // 4
            print(f"  d=+-{dd:3d}: e_hat(+d)={e[i]:6.2f} e_hat(-d)={e[j]:6.2f} | a_dev(+d)={ad[i]:6.2f} "
                  f"a_dev(-d)={ad[j]:6.2f} | r(+d)={r[i]:5.2f} r(-d)={r[j]:5.2f}")
        jm = int(np.argmax(np.where(pos, r, -1.0)))
        print(f"  max sym err of e_hat on |d|<2q: {np.abs(e[pos] - e[pos][::-1]).max():.3f}; "
              f"argmax of r at d = {d[jm]:g} (r = {r[jm]:.2f}, R = r/s = {r[jm] / ad[mid]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
