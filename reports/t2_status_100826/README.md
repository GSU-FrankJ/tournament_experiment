# T=2 status pack (2026-10-08): index

A self-contained folder that lets a coworker who did not take part decide whether T=2 should be closed now or improved further. It gathers the locked solver v2.0 and the earlier rounds (already published in `../t2_refine_100526/`) and the three multistage rounds of the current PI session (MS-R1, MS-R2, MS-R3). **It writes material and runs nothing:** no training, no verifier evaluation of weight exports, no launcher, no re-run of a round analysis tool. Derived numbers come from tracked records through the scripts in `report_scripts/`.

## Status

**Decision pending.** The decision is the coworker's. No experiment of any kind starts in this round or before their reply. Nothing was sent to anyone (no email, message, issue, pull request, mention or comment); the recipient and the channel have not been named.

The question the folder has to let them answer, verbatim:

> 基于现有结果，我们应该在此阶段收口，还是继续提高精度？如果继续，优先改进什么、目标是什么？
>
> (Based on the current results, should we close T=2 at this stage, or continue improving its accuracy? If we continue, what should be improved first, and what is the target?)

## Reading order

1. [report.md](report.md): the report. A one-page Chinese summary (section 0), then the English body. About fifteen minutes are enough to decide from sections 0, 7.2 (current accuracy), 9 (unresolved issues) and 10 (the decision: Path A close now, Path B continue, the PI-side recommendation labelled as an input, the question and what to send back). The other sections support those four.
2. [handoff.md](handoff.md): the message to the coworker, in Chinese then English, with the links.
3. The round reports that the report cites, in this order: [../t2_refine_100526/README.md](../t2_refine_100526/README.md) (v2.0 and the rounds before it), [../ms/r1/summary.md](../ms/r1/summary.md), [../ms/r2/summary.md](../ms/r2/summary.md), [../ms/r3/summary.md](../ms/r3/summary.md).
4. [evidence/manifest.csv](evidence/manifest.csv): every evidence item (id, source path and commit, SHA-256, status, copy path, the report sections that cite it); [pi_record/01_factcheck.md](pi_record/01_factcheck.md): the independent fact-check ledger.

## Folder map

| path | what it is |
|---|---|
| `README.md` | this index |
| `report.md` | the report (D4 of the prompt): Chinese summary + English body; every table between `<!-- TBL:name -->` markers is produced by `report_scripts/tables.py` |
| `handoff.md` | the message to the coworker (Chinese, then English) |
| `evidence/` | copies of the tracked files the report cites, under their original paths (`manifest.csv`, `SHA256SUMS`); items of the 100526 pack are cited in place, not copied; `evidence/dot-claude/CLAUDE.md.txt` is the project instruction file stored under a neutral name |
| `figures/` | the figures the report embeds: `FIG-01` to `FIG-11` are copies, `FIG-12` (F1) and `FIG-13` (F2) are drawn by `report_scripts/figures.py` from evidence copies (`SHA256SUMS`) |
| `report_scripts/` | `build_pack.py`, `tables.py`, `figures.py`, `check_numbers.py`, `check_links.py`, `README.md` |
| `pi_record/21_t2_status_pack_prompt.md` | the prompt of this round, verbatim (Appendix A = the PI's plan note, Appendix B = the PI-side reading after MS-R3; both are inputs, not evidence) |
| `pi_record/00_build_log.md` | commands, hashes and verbatim outputs of P0-P4 |
| `pi_record/01_factcheck.md` | the fact-check ledger (one row per checked statement, with a verdict and the disposition of every row that is not OK) |

## Branch and commits

Branch `t2-status-pack`, created from `origin/ms-r3` (`be4fd2021e5aee52625a994b3566dc18726e0586`) in the session worktree `t2-status-pack-0c452c`. The head of the branch is the commit that contains this file; `git rev-parse origin/t2-status-pack` gives it, and the report permalink with its full SHA is in `handoff.md`. `main`, the round branches (`ms-r1`, `ms-r2`, `ms-r3`, `t2-refine-pack`) and every tag are untouched; no pull request was opened. The only change outside this folder is a dated pointer section at the top of `docs/STATE.md`.

## What is in the folder that the prompt did not name

- `TBL-` rows in the manifest for the tables produced by `tables.py` (status `generated (table text)`; the SHA-256 is that of the rendered text), and `BG-` rows for three background files (`reports/v2/pilot1_reward_estimator.md`, `.claude/CLAUDE.md`, `agents/ppo_curriculum.py`).
- `T2R:100526report` for the 100526 folder report, which is not an evidence item of that pack.
- Anything in the report that differs from the inputs of the prompt is listed in `report.md` section 11 and in the ledger.

## Integrity

```
(cd reports/t2_status_100826/evidence && sha256sum -c SHA256SUMS)
(cd reports/t2_status_100826/figures  && sha256sum -c SHA256SUMS)
python reports/t2_status_100826/report_scripts/build_pack.py --check      # rebuild in a temp dir, compare byte for byte
python reports/t2_status_100826/report_scripts/tables.py --verify reports/t2_status_100826/report.md
python reports/t2_status_100826/report_scripts/check_numbers.py            # tables equal the script output; every numeric sentence cites an item
python reports/t2_status_100826/report_scripts/check_links.py [--ref origin/t2-status-pack]
```

Python is `/home/fjiang4/tournament_experiment/.venv/bin/python` (pandas, numpy, matplotlib); the scripts read only tracked CSV/JSON/PNG files.
