"""Re-run the R7 block (full test suite + C7) of tools/v2/v2_0_rehearsal_checks.py at the fix commit.

Calls the tool's own run_check_7 unchanged. The only thing set from outside is the tool's module global LK, which
run_check_7 uses solely for the pytest log, so that the committed rehearsal_v2_0_pytest.txt is not overwritten.
Usage: python -B driver.py <commit sha of the fix>   (OMP/MKL/OPENBLAS_NUM_THREADS are set below, as the tool sets
them for its subprocesses)
"""
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/v2-t2-refine")
HERE = Path(__file__).resolve().parent
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "v2"))
import v2_0_rehearsal_checks as T  # noqa: E402

commit = sys.argv[1]
T.LK = HERE                                   # the pytest log lands here as rehearsal_v2_0_pytest.txt
res = T.run_check_7(commit)
res["driver"] = "results/v2_T2_locked/rehearsal_v2_0_checks/tool_fix_rerun/driver.py"
res["launch_commit_argument"] = commit
(HERE / "r7_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
