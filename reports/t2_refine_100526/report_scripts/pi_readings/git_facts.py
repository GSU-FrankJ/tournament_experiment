"""Read-only git facts for the commit hashes listed by the PI. Writes git_facts.json."""
import json
import subprocess

WT = "/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack"
OUT = ("/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
       "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/"
       "report_parts/pi_readings/git_facts.json")

HASHES = ["32a8c21", "6c902db", "155cdec", "1d6d4d0", "f2d616c", "d2e377d", "85c294e",
          "6e216e2", "b55d389", "1ff99bd", "d581b3c", "62ecc43", "58c2671", "3ad1b07"]
BRANCHES = ["v2-t2-refine", "v2-t2-r2b", "v2-t2-r2c", "t2-refine-pack", "main"]


def git(*args):
    r = subprocess.run(["git", "-C", WT] + list(args), capture_output=True, text=True)
    return (r.stdout.strip() if r.returncode == 0 else "ERR: " + r.stderr.strip())


facts = {"commits": {}, "branches": {}, "tags": {}, "contains": {}, "files": {}}
for h in HASHES:
    t = git("cat-file", "-t", h)
    rec = {"type": t}
    if t == "commit":
        rec["full"] = git("rev-parse", h)
        rec["subject"] = git("show", "-s", "--format=%s", h)
        rec["date"] = git("show", "-s", "--format=%ad", "--date=iso", h)
        rec["parents"] = git("show", "-s", "--format=%P", h)
        rec["files"] = git("show", "--stat", "--format=", "--name-only", h).splitlines()[:40]
        rec["branches_containing"] = git("branch", "-a", "--contains", h).splitlines()
    facts["commits"][h] = rec
for b in BRANCHES:
    facts["branches"][b] = git("rev-parse", b)
for t in git("tag", "-l", "t2-v2-*").splitlines():
    facts["tags"][t] = git("rev-parse", t + "^{commit}")
json.dump(facts, open(OUT, "w"), indent=1)
print(json.dumps(facts, indent=1))
