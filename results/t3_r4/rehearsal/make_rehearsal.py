"""Rehearsal root for the T3-R4 post-launch tools: 80 run directories in the confirmation layout, each a real directory whose files are
symlinks into one of T3-R3's 40 bwA development runs (read-only originals). Seeds 80501-80540 map to T3-R3 seeds 70501 + (i % 20).
Nothing is written into the originals."""
import os
import sys
from pathlib import Path

R3 = Path("/home/fjiang4/tournament_experiment/.claude/worktrees/t3-bw-failures-diagnosis-25cb13/results/t3_r3/dev")
ROOT = Path(sys.argv[1])


def link_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.iterdir()):
        if p.is_dir():
            link_tree(p, dst / p.name)
        else:
            os.symlink(p, dst / p.name)


n = 0
for q in (50, 60):
    for i, s in enumerate(range(80501, 80541)):
        src = R3 / f"q{q}" / f"seed{70501 + (i % 20)}" / "bwA"
        dst = ROOT / f"q{q}" / f"seed{s}" / "bwA"
        link_tree(src, dst)
        n += 1
print(n, "run directories under", ROOT)
