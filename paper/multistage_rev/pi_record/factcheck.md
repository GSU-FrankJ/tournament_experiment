# Fact-check record (round P-FIG)

Independent, read-only checks by agents that did not produce the items. Two rounds; every finding was corrected at its origin (the builder script) and the corrected items were re-checked in round 2. The checkers recomputed values from the primary sources (stored per-run records, protocols, run configs, code at the run commits, the stored weights) and did not use the producers' notes.

## Round 1 (ten checkers)

| checker | scope | checked | correct | problems |
|---|---|---|---|---|
| T1a | T1 groups A, E, F | 116 | 116 | 2 |
| T1b | T1 group B | 122 | 102 | 2 |
| T1c | T1 group C | 79 | 75 | 2 |
| T1d | T1 groups D, G, H | 145 | 142 | 3 |
| T23 | T2 and T3, every value | 320 | 317 | 2 |
| T4 | T4, every value | 457 | 447 | 4 |
| PROVF | PROVENANCE global facts, F1-F5 | 285 | 277 | 8 |
| PROVT | PROVENANCE T1-T4 | 1500 | 1494 | 5 |
| FIG12 | data and figures F1, F2, F5 | 35670 | 35669 | 1 |
| FIG34 | data and figures F3, F4 | 2050 | 2049 | 1 |

No wrong number was found. Findings (one major, rest minor) and their corrections:
1. T1 group B labelled the T3-R1 comparison block (seeds 50501-50520) a "development block" (major, label only; values correct). Corrected: rows read "development or comparison block", a row "role of the block (spec terms)" was added (T3-R1 = comparison block, with the protocol's own wording quoted; T3-R3 = T3-1.1 development block), and every group-B statistic carries the round of its numbers.
2. T1 citation and label slips: `t3_eval.py:96` (now 98-100), "schedule before the window" (now "schedule in phase A"), the per-stage policy-row source wording, `ms_rollout.py` "9-3" (now 3-9), `gates.G-A.all_must_hold[1]`/`[2]`, no "tables built" row for the T3 columns (14 rows added). Values unchanged.
3. T3.md: two rows with unescaped pipes (escaped); Sources item G said `headline.json` lists the failing quantities and stage (only `failures.csv` does; reworded).
4. T4: the label "start-gap distribution" (now "distribution of the start score gap d"); source wording of "policy rows per update"; non-verbatim code quotes (now verbatim with file:line); freeze statement of bw; single-threaded source (now also cites `threads_per_process`).
5. F3: the effort panels had no "score gap d" x label (added; figure 7 x 10.1 in).
6. PROVENANCE: imprecise statements (F1 runs no verifier evaluation; `__pycache__`; palette; F2 reads only `gates.json`; `retry.json` exists only in 5 runs; helpers "unchanged"; T1 does not read the protocol `.md` files; unstored "645 assertions" and "260 values" claims). All reworded.

## Round 2 (four checkers: T1 after correction; T2-T4 and PROVENANCE T2-T4; figures and PROVENANCE figures/global; full reproduction in a scratch copy)

| checker | checked | correct | problems |
|---|---|---|---|
| T1 (all 476 rows, old/new diff, every new/changed row, 60 random unchanged rows) | 476 | 476 | 2 minor |
| T34 (T2, T3, T4, changed rows, PROVENANCE T2-T4) | 1019 | 1015 | 4 minor |
| FIGPROV (F1-F5 figures, PDF/PNG measurements, F3/F4 re-derivation, PROVENANCE figures/global) | 205 | 203 | 2 minor |
| REPRO (every extraction, table and plot script re-run in a scratch copy) | 181 | 178 | 3 minor |

No wrong value and no regression. The old-vs-new diff of T1 showed 0 value changes (14 rows added, 135 non-value cells changed, all intended). Reproduction: all 23 regenerated outputs (data F1-F5, figures F1-F5 pdf and png, tables T1-T4 csv and md) were byte-identical to the committed files; PNG pixel arrays identical; smallest glyph run 8.0 pt in all five PDFs; every figure is 7 in wide; `F{n}.py` read only `data/F{n}.csv` and `_style.py`; no file outside the scratch copy was written. `pi_record/30_paper_figures_prompt.md` equals the original prompt text (191 lines, 14,795 bytes) except for the transport wrapper tags.

Round-2 findings and dispositions:
- T1.md introduction said every group-B statistic ends with a round tag; the three "q values" rows do not. Corrected in `scripts/T1.py`; T1.csv unchanged, T1.md regenerated.
- T4.md footnote 6 omitted four config keys (`record.obs_encoding`, `record.action_mapping`, `record.action_sampling`, `record.mean_extraction`) and footnote 7 omitted `BetaActor.__init__`. Corrected in `scripts/T4.py`; T4.csv unchanged, T4.md regenerated.
- F1 legend lacked "at the terminal stage" (now verbatim as the spec's terms). F1 figure regenerated; `data/F1.csv` unchanged.
- PROVENANCE: T4 "Runs used"/"Commit" claimed reads of R4 `manifest.json`/`gates.json` and of `gates.json` for jt/jt_es that `T4.py` does not do; "similar directories" for `__pycache__` (only `envs/` and `utils/`); the Re-running note (scripts need a git checkout: `git rev-parse HEAD` in `extract_F2.py`, `git show <run commit>` in `T1.py`) was added. All reworded.
- Accepted as is (informational): the T = 2 column of T1 states the quadrature-weight property inside its "z-integral (quoted)" row and has no separate "weights (quoted)" / "sum of the quadrature weights" rows (an asymmetry across columns, not an error); `data/F2.csv` has CRLF line endings (csv.DictWriter default); the pre-existing `__pycache__` directories are git-ignored and are not committed.

After the round-2 corrections only `scripts/T1.py`, `scripts/T4.py`, `scripts/F1.py`, `tables/T1.md`, `tables/T4.md`, `figures/F1.pdf`, `figures/F1.png` and `PROVENANCE.md` changed; the regenerated T1.csv and T4.csv are byte-identical to the checked versions, and the diffs of T1.md and T4.md are exactly the two sentences above. These last edits were verified by diff by the lead, not by a third independent round.
