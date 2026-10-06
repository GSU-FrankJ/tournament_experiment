"""Read the last lines of the pytest logs named in the round reports (read-only)."""
import os

WT = "/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack"
FILES = [
    "results/v2_refine/code_pytest_full.txt",
    "results/v2_T2_locked/rehearsal_v2_0_pytest.txt",
    "results/v2_T2_locked/rehearsal_v2_0_checks/tool_fix_rerun/rehearsal_v2_0_pytest.txt",
    "results/v2_T2_locked/v2_0/fullsuite_prelock.txt",
    "results/v2_refine_r2b/code_pytest_full.txt",
    "results/v2_refine_r2b/final_pytest_full.txt",
    "results/v2_refine_r2b/post_audit_pytest_full.txt",
    "results/v2_refine_r2c/code_pytest_full.txt",
]
for f in FILES:
    p = os.path.join(WT, f)
    print("==", f, "exists" if os.path.exists(p) else "MISSING")
    if os.path.exists(p):
        with open(p, errors="replace") as fh:
            lines = fh.read().splitlines()
        for ln in lines[-4:]:
            print("   ", ln[:250])
        fails = [l for l in lines if l.startswith("FAILED") or l.startswith("ERROR")]
        print("    FAILED/ERROR lines:", fails)
