"""More read-only git facts: manifest commits of the run families, remote heads, code drift."""
import json
import subprocess

WT = "/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack"
OUT = ("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
       "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/"
       "report_parts/pi_readings/git_facts2.json")


def git(*args):
    r = subprocess.run(["git", "-C", WT] + list(args), capture_output=True, text=True)
    return (r.stdout.strip() if r.returncode == 0 else "ERR: " + r.stderr.strip())


facts = {}
for h in ["655b14c", "6e99e01", "3492cac", "6a8f449", "4cdf60a", "66fa551", "db78f75", "e89b61d"]:
    facts[h] = {
        "type": git("cat-file", "-t", h),
        "full": git("rev-parse", h),
        "subject": git("show", "-s", "--format=%s", h),
        "date": git("show", "-s", "--format=%ad", "--date=iso", h),
    }
for r in ["origin/v2-t2-refine", "origin/v2-t2-r2b", "origin/v2-t2-r2c", "origin/main"]:
    facts[r] = git("rev-parse", r)
facts["is_ancestor_3ad1b07_in_r2c"] = git("merge-base", "--is-ancestor", "3ad1b07", "v2-t2-r2c") or "yes(0)"
facts["drift_32a8c21_to_6e99e01_code_files"] = git(
    "diff", "--name-only", "32a8c21", "6e99e01", "--", "agents", "run", "utils", "envs", "tools",
    "tests").splitlines()
facts["drift_32a8c21_to_655b14c_code_files"] = git(
    "diff", "--name-only", "32a8c21", "655b14c", "--", "agents", "run", "utils", "envs", "tools",
    "tests").splitlines()
facts["log_32a8c21_to_155cdec"] = git("log", "--format=%h %ad %s", "--date=iso",
                                     "32a8c21^..155cdec").splitlines()
facts["log_r2b_1ff99bd_to_62ecc43"] = git("log", "--format=%h %ad %s", "--date=iso",
                                         "1ff99bd^..62ecc43").splitlines()
facts["log_r2c_58c2671_to_head"] = git("log", "--format=%h %ad %s", "--date=iso",
                                      "58c2671^..v2-t2-r2c").splitlines()
facts["log_lock_to_pubhead"] = git("log", "--format=%h %ad %s", "--date=iso",
                                  "1d6d4d0^..b55d389").splitlines()
json.dump(facts, open(OUT, "w"), indent=1)
print(json.dumps(facts, indent=1))
