#!/usr/bin/env python3
"""Stage-1 error decomposition with residual-minimization bands (decision D2).

e1 - e1* = (e1 - e~1) + (e~1 - e1*), with e~1 and its band from tools/v2/induced_band.py:
  learning term  point e1 - e~1,  interval [e1 - band_hi, e1 - band_lo]
  inherited term point e~1 - e1*, interval [band_lo - e1*, band_hi - e1*]
Frozen arms use the band of their parent (their stage 2 IS the parent's); the joint arm uses
the band of the matching weight export (live stage 2). Rows: every training-time verifier
checkpoint (frozen arms; e1 from v2_checkpoints.csv) and every weight export (all arms).

Usage: python tools/v2/decomposition.py --pilot pilot2|pilot3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "v2"))

from common import spec_for  # noqa: E402
from induced_band import OUT as BAND_DIR, actor_policy  # noqa: E402


def rows_for(e1, band, g1):
    lo, hi, et = band["band_lo"], band["band_hi"], band["e_tilde"]
    L = (e1 - et, e1 - hi, e1 - lo)
    I = (et - g1, lo - g1, hi - g1)
    return {"e1_at_0": e1, "e_tilde": et, "band_lo": lo, "band_hi": hi,
            "learning_rel": L[0] / g1, "learning_rel_lo": L[1] / g1, "learning_rel_hi": L[2] / g1,
            "learning_contains_0": bool(L[1] <= 0.0 <= L[2]),
            "inherited_rel": I[0] / g1, "inherited_rel_lo": I[1] / g1, "inherited_rel_hi": I[2] / g1,
            "inherited_contains_0": bool(I[1] <= 0.0 <= I[2]),
            "total_rel": (e1 - g1) / g1,
            "e1_inside_sweep": bool(band["sweep_lo"] <= e1 <= band["sweep_hi"])}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", required=True)
    a = p.parse_args()
    pb = pd.read_csv(BAND_DIR / "parent_bands.csv").set_index(["q", "seed"])
    jb = None
    if os.path.exists(BAND_DIR / "joint_export_bands.csv"):
        jb = pd.read_csv(BAND_DIR / "joint_export_bands.csv").set_index(["q", "seed", "update"])
    rows = []
    for d in sorted(glob.glob(str(ROOT / "results" / "v2_pilots" / a.pilot / "q*" / "seed*" / "*"))):
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        man = json.load(open(os.path.join(d, "manifest.json")))
        q, seed, arm = int(man["q"]), man["seed"], man["arm"]
        spec = spec_for(q)
        frozen = man["stage2_update_mode"] == "frozen"
        if frozen:
            band = pb.loc[(q, seed)]
            g1 = float(band["g1"])
            ck = pd.read_csv(os.path.join(d, "v2_checkpoints.csv"))
            for r in ck.itertuples():
                rows.append({"q": q, "seed": seed, "arm": arm, "update": int(r.update), "source": "checkpoint",
                             **rows_for(float(r.e1_at_0), band, g1)})
        for f in sorted(glob.glob(os.path.join(d, "weights", "u*.npz"))):
            u = int(os.path.basename(f)[1:6])
            if u <= 400:
                continue
            if frozen:
                band = pb.loc[(q, seed)]
            else:
                if jb is None or (q, seed, u) not in jb.index:
                    continue
                band = jb.loc[(q, seed, u)]
            g1 = float(band["g1"])
            e1 = float(actor_policy(f, spec)(1, np.zeros(1))[0])
            rows.append({"q": q, "seed": seed, "arm": arm, "update": u, "source": "weights",
                         **rows_for(e1, band, g1)})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "v2_pilots" / a.pilot / "analysis"
    os.makedirs(out, exist_ok=True)
    df.to_csv(out / "decomposition_residual_band.csv", index=False)
    print(len(df), "rows; e1 outside sweep:", int((~df.e1_inside_sweep).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
