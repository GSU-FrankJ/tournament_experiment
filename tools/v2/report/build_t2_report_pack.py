#!/usr/bin/env python
"""Rebuild the T=2 v2 report pack (``reports/v2/t2_report/``) from ``results/``.

Usage (single-threaded, venv Python)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /home/fjiang4/tournament_experiment/.venv/bin/python tools/v2/report/build_t2_report_pack.py \
      --results-root /home/fjiang4/tournament_experiment/.claude/worktrees/pilot-4-stabilization-fb99a2/results

* no option: clean build of every module, then ``finalize`` (manifest, key numbers, dictionary, gaps,
  consistency, re-evaluations, README);
* ``--only sec_pilots,sec_locked``: run only those modules (their fragments are replaced; no finalize unless ``--finalize``);
* ``--finalize``: merge the existing fragments only.

No training is run. Forward passes / verifier evaluations that a module needs are logged by the
module in ``reevaluations.csv``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# build order matters only for modules that read pack tables written by earlier ones
MODULES = ["sec_front", "sec_p0p1", "sec_method", "sec_pilot1", "sec_pilot23", "sec_locked_a", "sec_locked_b",
           "sec_ext", "sec_stage1", "sec_appendix"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated module names to run")
    ap.add_argument("--results-root", default=None, help="directory that plays the role of results/")
    ap.add_argument("--finalize", action="store_true", help="merge fragments into the pack-level files")
    ap.add_argument("--no-clean", action="store_true", help="full build without removing the previous outputs")
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if args.results_root:
        os.environ["T2_REPORT_RESULTS_ROOT"] = args.results_root

    import common as C  # noqa: E402  (after the environment is set)

    only = [m for m in args.only.split(",") if m]
    mods = only or MODULES
    if not only and not args.no_clean and not args.finalize:
        for sub in ("tables", "figures", "data", "provenance", "_build"):
            shutil.rmtree(C.PACK_DIR / sub, ignore_errors=True)
        for f in ("manifest.csv", "key_numbers.csv", "data_dictionary.csv", "gaps.md", "consistency.md",
                  "reevaluations.csv", "README.md"):
            (C.PACK_DIR / f).unlink(missing_ok=True)
    print(f"results root: {C.RESULTS_ROOT}\npack dir: {C.PACK_DIR}\nbase commit: {C.BASE_COMMIT}", flush=True)

    if not args.finalize or only:
        for name in mods:
            t0 = time.time()
            try:
                mod = importlib.import_module(name)
            except ModuleNotFoundError as e:
                if e.name == name and not only:
                    print(f"[skip] {name}: module not present", flush=True)
                    continue
                raise
            mod.build()
            print(f"[done] {name} in {time.time() - t0:.1f} s", flush=True)

    if args.finalize or not only:
        summary = C.finalize()
        print("finalize:", summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
