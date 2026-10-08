# T=2 status pack for a coworker's decision: report, evidence pack and handoff (no new experiments)

This round writes material and runs nothing. It gathers the state of the T=2 work into one self-contained folder that a coworker who did not take part can review and decide on. The folder covers two things:

- **Background:** the locked solver v2.0 and the earlier rounds already published in `reports/t2_refine_100526/`.
- **The substance:** the three multistage rounds of the current PI session, MS-R1, MS-R2 and MS-R3 (prompts 17-20; branches `ms-r1`, `ms-r2`, `ms-r3`; `origin/ms-r3` at `be4fd202`).

The coworker is asked one question, verbatim:

> 基于现有结果，我们应该在此阶段收口，还是继续提高精度？如果继续，优先改进什么、目标是什么？
>
> (Based on the current results, should we close T=2 at this stage, or continue improving its accuracy? If we continue, what should be improved first, and what is the target?)

The folder must let them answer it. It has to say what T=2 is for and what was done, what the results establish and how accurate the solver is, what is still open, and how the two paths compare. The decision is the coworker's. No new experiment of any kind starts in this round or before their reply.

Steps:

1. **P0:** housekeeping (§1).
2. **P1:** the evidence pack (§2).
3. **P2:** `report.md`, `README.md` and `handoff.md` (§3).
4. **P3:** the checks and an independent fact-check (§4).
5. **P4:** commit, push, return the links and the handoff text, ask me for the recipient and the channel, then **STOP** (§5).

**Setup.**
- Repository `/home/fjiang4/tournament_experiment`.
- Python `/home/fjiang4/tournament_experiment/.venv/bin/python` only.
- Work in your session's own worktree, on a new branch `t2-status-pack` created from `origin/ms-r3` (`be4fd202`). Record the worktree path.
- New files go only under `reports/t2_status_100826/`. The one change outside that folder is a dated pointer section at the top of `docs/STATE.md`.

**Provenance**, as in every round:
- Every number in the folder cites an evidence item.
- Nothing is fabricated or rounded differently from its source without saying so.
- Nothing is carried over from this prompt without checking it against its record.

**Nothing is run:**
- no RL or supervised training;
- no verifier evaluation of weight exports;
- no launcher;
- no re-run of the round analysis tools.

Derived numbers come only from tracked records (CSV/JSON), through scripts committed in the new folder.

**Nothing existing is changed:** no round report, result, protocol, code, test, other branch, tag or `main`.

Right after creating the branch (P0), and before anything else, save this prompt verbatim as `reports/t2_status_100826/pi_record/21_t2_status_pack_prompt.md`.

**If any step fails, or anything does not match this prompt, stop at that point and report. Do not work around it.**

---

## 0. Decisions (binding)

### D1. Sources, and what the numbers of this prompt are

- **Sources, in reading order:**
  1. `reports/t2_refine_100526/README.md` and `100526report.md` (sections 0, 3 and 6).
  2. For each MS round: `reports/ms/r{1,2,3}/summary.md`, `05_decision_inputs.md`, `04_pilot.md` and `02_preregistration.md`.
  3. The PI record: `reports/ms/r1/pi_record/17_ms_r1_prompt.md`, `18_g1_reply.md`, `reports/ms/r2/pi_record/19_ms_r2_prompt.md` and `reports/ms/r3/pi_record/20_ms_r3_prompt.md`.
  4. The fact-check ledgers of MS-R2 and MS-R3.
  5. The tracked records under `results/ms_r1/`, `results/ms_r2/` and `results/ms_r3/`.
  6. `protocols/v2_T2_locked_v2_0.{json,md}`.
  7. Appendices A and B of this prompt.
- **What the appendices are.**
  - Appendix A is the PI's plan note for this session; it states the T=2 goals.
  - Appendix B is the PI-side reading after MS-R3: interpretation, two derived computations, the PI's hypotheses and the PI's recommendation.
- **Every number in this prompt is an input to verify, not evidence.**
  - Where a record differs, the record wins, and the difference goes in the fact-check ledger.
  - A number that cannot be traced to a record is left out.
- **Earlier rounds** (R1, v2.0, R2b, R2c) are summarised from the 100526 publication and cited through its evidence items (`T2R:<id>`). They are not re-analysed.
- **T=3** enters only where a choice has a consequence for it, and then only as "not evaluated".

### D2. Folder layout

```
reports/t2_status_100826/
  README.md           index: what this is, status (decision pending), reading order, folder map,
                      branch and commit, integrity commands
  report.md           the report (D4): English body, one-page Chinese summary (section 0) at the top
  handoff.md          the message to the coworker (D7), Chinese first, then English
  evidence/           copies of the small tracked files the report cites, under their original paths
    manifest.csv
    SHA256SUMS
  figures/            the figures the report embeds (copies with FIG-ids; at most two new ones, D5)
    SHA256SUMS
  report_scripts/     build_pack.py, tables.py, figures.py, check_links.py, check_numbers.py, README.md
  pi_record/
    21_t2_status_pack_prompt.md    this prompt, verbatim
    00_build_log.md                commands, hashes and verbatim outputs of P0-P4
    01_factcheck.md                the fact-check ledger (P3)
```

### D3. Status labels

Every claim in report sections 7-10 carries one label:

- **[Verified].** Established by one of:
  - a pre-registered criterion;
  - a gate;
  - a deterministic check (bit-identity or reproduction);
  - an exact formula check.

  Each comes with its evidence item. A plain measurement, such as "the median gap of `relu` is lower in all eight cells", is **[Verified, descriptive]** when the sentence claims no more than what was measured.
- **[Hypothesis].** An interpretation or a mechanism, including every PI reading. Each comes with the experiment that would test it, and whether that experiment was run.
- **[Insufficient evidence].** A question the records cannot answer, said plainly. Examples:
  - an interval that contains 0 with ten seeds, which is not "no effect";
  - a failure rate estimated from two (q, seed) cases;
  - any fresh-seed claim about an MS configuration (none was confirmed).

Two further rules:
- Post-hoc tables are marked post hoc.
- Development-seed and fresh-seed numbers are never mixed in one statistic; every accuracy number names its seed set and n.

### D4. `report.md`: required content

**Reading time.** A reader should be able to read sections 0, 7.2, 9 and 10 in about fifteen minutes and decide. Everything else supports those sections, and details stay in the linked round reports.

**0. 中文摘要.** One page, with the same numbers and evidence IDs as the body. It covers:
- what T=2 is and what "solved" means here;
- the current accuracy;
- what is settled and what is open;
- the two paths, in a few lines each;
- the PI-side recommendation, attributed to the PI;
- the question, verbatim.

**1. Purpose and reading guide.** Who decides what; the question; how the labels work.

**2. The T=2 problem.**
- The game and its parameters: q in {50, 60}, k, DW, B = 100 + 2q, and the closed-form tent e2*(d), with e2*(0) = 70 at q = 50 and 58.33 at q = 60.
- What is learned: the stage-2 and stage-1 Beta policies, trained by self-play against a lagged copy.
- Where the closed form enters: evaluation only.
- Where the game model enters training:
  - the reward estimator `reward_mode = expected` in `protocols/v2_T2_locked_v2_0.json`, i.e. the conditional expectation over the shock given the sampled actions (`reports/v2/pilot1_reward_estimator.md`);
  - the expected-continuation table.
- State plainly that `.claude/CLAUDE.md` lists a sampled-reward invariant for the original runners, that the v2 pipeline deliberately differs from it, and what this means for the claims a paper can make.

**3. Goals.**
- The role of T=2 in the project: a base case with a closed-form ground truth.
- The session goals of Appendix A, each with its status after MS-R1..R3: implemented, tested, outcome.

**4. Definitions.** Every term used, in one place:
- d, q, k, DW, B, e2*(d) and effort units;
- |peak error| and signed peak error (relative to e2*(0), at d = 0, final tier, terminal freeze), and the location-free peak error;
- the gap, e2*(0) − e_hat_2(0), in effort units;
- sigma_2(0); the smoothing part, e2*(0) sigma_2(0) / (sqrt(pi) q); the remainder; the additive and quadrature models; w_eff;
- R0, R and Delta_2;
- RMSE_pos, tail mean and eta_2/DW;
- the gates G-A, G-F, G-N and G-S, with their limits;
- criterion parts (a) and (b), and the bootstrap;
- the seed blocks: development 10501-10510, v2.0 confirmation 30501-30520, reserved 40501-40520;
- the starts: bin-balanced, and stratified with lambda_P, lambda_T and alpha;
- the noise landing and s;
- the arms of each round: `MS_*`, `NL_*`, `{t1,relu,t10}_{bb,st}_s{1,16}`;
- `parents_A` and `rehearsal_v2_0`, which share the same terminal stage (MS-R1 check C-MS1).

**5. The current approach.**
- (a) The locked solver v2.0: pipeline, gates and confirmation design.
- (b) The MS stage-wise runner. What it added, all behind keys:
  - the stop rule with a first-order residual;
  - coverage-constrained stratified starts;
  - the stage-wise freeze;
  - the noise landing;
  - the actor variants.

  Also the checks that keep v2.0 intact: the C-R4 and C-R6 reproductions, and C-MS1..C-MS5.

**6. Completed work.**
- One table with one row per round (R1, v2.0, R2b, R2c, MS-R1, MS-R2, MS-R3). Columns:
  - dates; branch and head; prompt file;
  - question;
  - design: arms, seeds, q and budget;
  - pre-registered criterion and its outcome;
  - main finding;
  - report path.
- Then the PI decisions at each gate of this session, read from the prompts and from `18_g1_reply.md`.

**7. Key results** (all labelled):
- **7.1** v2.0 as the certified T=2 solver: the confirmation, 19/20 and 20/20; the failed run; the gates.
- **7.2 Current accuracy.** One table.
  - **Rows:**
    - v2.0 on the fresh seeds, q = 50 and 60;
    - v2.0 on the development seeds (`parents_A`);
    - the development-seed arms the decision needs, at least `MS_base2400`, `MS_s35a5`, `t1_bb_s1`, `t1_st_s16`, `t10_st_s16`, `relu_st_s16` and `relu_bb_s16`.
  - **Columns:**
    - seed set and n;
    - mean and median |peak error|; median signed peak error; runs with |peak error| <= 0.05;
    - median gap; smoothing part; remainder;
    - RMSE_pos/e2*(0); tail mean/e2*(0); max eta_2/DW;
    - gate failures; stage-1 error where recorded.
  - Mark which rows are confirmed and which are development-only.
  - Mark that the MS-R3 `t1` arms are bit-identical re-runs of MS-R2's `NL_*` arms (C-MS5), not replications.
- **7.3** The stage-2 tip deficit:
  - size and sign;
  - the decomposition and the formula check;
  - additive against quadrature.
- **7.4** What moved the tip and what did not. One table over all rounds, with columns: intervention, round, paired |peak| change with its interval at each q, guard-rail effects, gate failures, verdict. The failures and the unmet criteria are included in it.
- **7.5** MS-R3 in detail:
  - the premise check;
  - RL against the supervised screen;
  - the `relu` typical-run improvement **and** its five failed runs, with what is known and what is not;
  - `t10`'s non-transfer;
  - the noise landing under each actor.
- **7.6** The plan's stop rule and polishing at T=2 (MS-R1), and R0 as a closed-form-free tie monitor.
- **7.7** Global accuracy and stage 1 across arms.

**8. What the evidence does not show.** A labelled list; Appendix B10 is the minimum.

**9. Unresolved issues.** Each with its label, what is known, and what would resolve it.

**10. Decision: close now or continue** (D6).

**11. Evidence index and reproduction.** The manifest, the figure list, the scripts and the commands.

### D5. Evidence pack and figures

**Copies.**
- Each tracked file the report cites that is smaller than 1 MiB is copied to `evidence/<original path>`.
- Larger tracked files, and untracked files, are listed as referenced (path, SHA-256, size, location) and not copied.
- Items already in the 100526 pack are cited in place by their ID, with status `cited in t2_refine_100526`, and not copied again.

**Manifest.**
- `manifest.csv` columns: `item_id, title, round, source_branch, source_commit, source_path, sha256, size_bytes, status, copy_path, conditions, used_in`.
- ID prefixes:
  - `M1-..`, `M2-..`, `M3-..` for the MS rounds;
  - `PI-..` for the prompts, the replies and the plan note;
  - `FIG-..` for figures;
  - `T2R:<id>` for the 100526 items.

**Build script.** `build_pack.py`:
- builds `evidence/` and `figures/` from the sources at their recorded commits;
- writes the manifest and `SHA256SUMS`;
- with `--check`, rebuilds in a temporary directory and compares byte for byte.

**Figures to embed (copies):**
- the MS-R3 learned tie profiles (`results/ms_r3/analysis/figures/tie_profile_runs.png`);
- the MS-R3 decomposition along the run, for both starts;
- the MS-R3 paired |peak| differences against `t1`;
- the MS-R3 noise-landing figure;
- MS-R2's `trajectory_decomposition.png` and `scatter_remainder_vs_smoothing_change.png`;
- the MS-R1 calibration figure that shows R against the peak;
- the relevant 100526 figures: the end-of-Phase-A profile and the peak trajectory.

**New figures:** at most two, drawn by `figures.py` only from evidence copies (of this pack or of the 100526 pack):
- **F1:** |peak error| per run, for every arm of MS-R1..MS-R3 and for the v2.0 confirmation, with the 0.05 line and with development and fresh seeds visibly separated.
- **F2:** the d = 0 gap split into the smoothing part and the remainder, per arm (MS-R2, MS-R3, `rehearsal_v2_0`).

Every figure has a caption with seeds, q, arms, budget and evidence IDs.

**Tables.**
- Every table of `report.md` is produced by `tables.py` from evidence copies, not typed by hand.
- `check_numbers.py` confirms that:
  - the tables in `report.md` equal the script output;
  - every numeric sentence outside the tables cites an evidence ID.

### D6. Report section 10: the decision

One comparison table, then prose, covering two paths.

**Path A, close now.**
- Which conclusions the current results support, each labelled, and which they do not.
- What closing consists of:
  - no new runs;
  - v2.0 remains the T=2 solver;
  - MS-R1..R3 produce no protocol change;
  - what would be written up.
- The limitations that a reader of any write-up would have to be told.

**Path B, continue improving accuracy.**
- **The metric to improve:**
  - Primary: the stage-2 |peak error| at d = 0, and the share of runs within 0.05.
  - Guard rails: RMSE_pos, tail mean, eta_2, gate pass rate and stage-1 error.
- **Reference points for choosing a target:**
  - v2.0 on the fresh seeds;
  - the best development-seed arms;
  - the smoothing floor sigma_2(0)/(sqrt(pi) q) at s = 1 and at s = 16. This is the error a policy with that noise would have if it played the noisy game's equilibrium exactly. It is not a strict bound: single runs can overshoot it (negative remainder);
  - the supervised-fit deficits.
- **For each candidate measure:**
  - the evidence for it (IDs);
  - the expected gain, **only from records**; otherwise write "cannot be estimated from the current evidence" and say what would be needed to estimate it;
  - the workload in recorded units: runs, rounds, and wall times **read from** the launch records and budget tables of comparable rounds. Person-time is "cannot be estimated";
  - the uncertainty;
  - the risks to the guard rails;
  - the consequences. A protocol change needs a lock and a fresh-seed confirmation; an actor change carries into T=3, which is not evaluated.
- **Measures to cover**, at least:
  1. `relu` with robustness fixes;
  2. the non-actor combination of stratified starts, the noise landing and 2800 updates;
  3. more near-tie samples (share or batch) within the tail constraint;
  4. the untested levers of Appendix B9;
  5. a mechanism round, which has scientific value but is not expected to raise accuracy by itself.

**Recommendations.**
- Include the PI-side recommendation of Appendix B11, labelled "PI-side recommendation (input; the decision is the coworker's)".
- You may add an observation of your own that the evidence supports, labelled as yours.
- Neither one decides.
- Nothing in this section invents a gain, a cost or a probability.

**End of the section.** The question, verbatim in Chinese and English, and what the coworker is asked to return:
- (i) close or continue;
- (ii) if continue: the measure to start with, and the target (metric, value, seed set).

### D7. Handoff

**`handoff.md`.** The message to the coworker, in Chinese then in English, each at most about 200 words. It contains:
- two sentences of context;
- the links: the folder, the report permalink and the compare view;
- three lines of status: what is settled, the current accuracy, what is open;
- the question, verbatim;
- what to send back;
- that no experiment starts before their reply.

**The recipient and the channel have not been named.**
- Send nothing to anyone: no email, Slack message, issue, pull request, mention or comment.
- Prepare the text, and ask me in your final message:
  1. who receives it;
  2. through which channel;
  3. their GitHub handle, if a mention or a pull request is wanted;
  4. whether they have read access to `GSU-FrankJ/tournament_experiment`.

### D8. Git and delivery

**Branch and commits.**
- Branch `t2-status-pack` from `origin/ms-r3` (`be4fd202`).
- Conventional commits (`docs:` or `chore:`), subjects under 72 characters.
- One logical change per commit: the pack; the report and the handoff; the STATE pointer; the fact-check corrections.
- No `.pt` files, weight exports, freeze arrays, run logs or `__pycache__`. Nothing under `results/` is moved or edited.

**Push.**
- Push `t2-status-pack` as a new branch, without force.
- Do not touch `main`, the round branches or the tags.
- Do not open a pull request, and do not merge.

**Verify after the push.**
- `git ls-remote origin t2-status-pack` equals the local HEAD.
- Every file of the folder appears in `git ls-tree -r origin/t2-status-pack`.
- Every relative link and image in `report.md`, `README.md` and `handoff.md` resolves in that tree (`check_links.py`, run against the pushed commit).

**Links to return:**
- the folder: `https://github.com/GSU-FrankJ/tournament_experiment/tree/t2-status-pack/reports/t2_status_100826`;
- the report permalink: `https://github.com/GSU-FrankJ/tournament_experiment/blob/<full sha>/reports/t2_status_100826/report.md`;
- the commit URL;
- the compare view: `https://github.com/GSU-FrankJ/tournament_experiment/compare/ms-r3...t2-status-pack`.

---

## 1. P0: housekeeping (log in `pi_record/00_build_log.md`)

1. Run `git ls-remote origin` and record `main`, `ms-r1`, `ms-r2`, `ms-r3`, `t2-refine-pack` and the tags. Verify that `origin/ms-r3` is `be4fd202` and that `t2-status-pack` does not exist on `origin`. Then create the branch in your worktree.
2. Save this prompt verbatim, as stated above.
3. Inventory the sources of D1 at `be4fd202`: for each, whether it is present, whether it is tracked, and its size. A missing source is a stop-and-report.

## 2. P1: the evidence pack

1. Read the sources in the order given in D1 before selecting any items.
2. Build the pack with `build_pack.py`.
3. Run `build_pack.py --check`.
4. Commit the pack (`chore:`).

## 3. P2: the report, the index and the handoff

1. Write `report.md` to D4 and D6, `README.md`, and `handoff.md` to D7. Tables come from `tables.py`; figures come from `figures.py` or are copies.
2. Write for a reader who did not take part:
   - define each term before using it;
   - give the conditions with every table and figure;
   - keep the failures and the unmet criteria in view. Appendices B5 and B10 list the ones that must appear.
3. Add the dated pointer section at the top of `docs/STATE.md`: what the folder is, the branch, and that the decision is pending.
4. Commit (`docs:`).

## 4. P3: checks and the independent fact-check

1. Run `check_numbers.py`, `check_links.py` and `build_pack.py --check`. All must pass.
2. Have an agent that did not write the folder do an independent, read-only fact-check of:
   - every number, against its evidence item;
   - every label: is each [Verified] claim actually established, and is any hypothesis written as fact?
   - the coverage of D4, D6 and Appendix B10;
   - that no gain, cost or probability is invented.
3. The ledger is `pi_record/01_factcheck.md`: one row per checked statement, with a verdict (OK / WRONG / OVERCLAIM / UNSUPPORTED / MISSING), and the disposition of every row that is not OK.
4. Apply the corrections before the push, and commit them (`docs:`).

## 5. P4: push, deliver, STOP

1. Push (D8) and verify.
2. Send me one message containing:
   - the four links;
   - the head commit;
   - the handoff text in both languages;
   - the fact-check summary (counts by verdict);
   - anything that differs from this prompt;
   - the four questions of D7.
3. Then **STOP**: no experiment, no new round, no pull request, no message to anyone. Wait for my reply.

---

## Appendix A. The PI's plan note for this session

Source: `Multistage100526.docx`, which is not in the repository. Below is its text as converted from the .docx; the layout of the formula is approximate.

> 我们现在可以确定Baseline: conditional expected reward + expected continuation + backward freeze，在这个基础上，我想我们现在可以把训练stop rule恢复到最初的设计：development DP-BR（达到predefined threshold or budget）+ stopping + targeted polishing，同时着重解决报告里所说的系统性偏差。
>
> 1. Verifier- guided prioritized state sampling：现在是peak-focused sampling，可以尝试让development DP-BR告诉我们哪些states还有比较大的deviation，然后下阶段training更频繁采样。但要保留global/tail coverage，不能全部集中到高的部分，产生peak好了，tail变差的情况。查到的一个办法是Peak + tail constrained stratified sampling（coverage-constrained stratified sampling）。可以把state sampling分成三部分，$p(d) = \lambda_{P}p_{\text{near-tie}} + \lambda_{M}p_{\text{middle}} + \lambda_{T}p_{\text{tail}}$，具体的比例需要development seed 预先测试，目标是increase peak resolution without reducing tail coverage
>
> 2. development DP-BR+ stopping + training/polishing，可以不光用在global，也可以在state阶段：
>    $\Delta_{t,\max}^{dev} \le \epsilon_t$ for $M$ checks → Freeze stage；residual broad → Continue global training；residual localized → Targeted polishing

**English rendering** (the assistant's, for the English report):

- **Baseline and aim.** The baseline is now settled: conditional expected reward + expected continuation + backward freeze. On that basis, restore the training stop rule to its original design: development DP-BR (until a predefined threshold or a budget is reached) + stopping + targeted polishing. At the same time, focus on the systematic bias described in the report.
- **(1) Verifier-guided prioritized state sampling.**
  - The current sampling is peak-focused. Let the development DP-BR tell us which states still have large deviations, and sample those more often in the next stage of training.
  - Global and tail coverage must be kept: sampling may not concentrate on the high part, which would give a better peak and a worse tail.
  - One method is peak + tail constrained (coverage-constrained) stratified sampling. Split the state sampling into three parts, p(d) = lambda_P p_near-tie + lambda_M p_middle + lambda_T p_tail.
  - The proportions are to be pre-tested on development seeds. The goal is to increase peak resolution without reducing tail coverage.
- **(2) Per-stage stop rule.** Development DP-BR + stopping + training/polishing can be used per stage, not only globally:
  - if Delta^dev_{t,max} <= eps_t for M checks → freeze the stage;
  - if the residual is broad → continue global training;
  - if the residual is localized → targeted polishing.

## Appendix B. PI-side reading after MS-R3

This appendix is input, not evidence. Verify every number, and keep the labels as given.

### B1. Overall

- **[Verified]** T=2 is solved as an equilibrium-finding problem by protocol v2.0: gates, and a fresh-seed confirmation.
- **Not solved:** the stage-2 tip deficit at the tie, d = 0. The learned effort there is below the closed form in every run of every round.
- **[Verified]** Six rounds tested stage-2 (tip) interventions: R1, R2b, R2c, MS-R1, MS-R2 and MS-R3. None of those interventions met its pre-registered criterion with part (b) holding. Hence there is no admissible tip fix and no v2.1.
- R1's method 6 (expected continuation) did meet its criterion, but for stage 1; it became part of v2.0. Do not count it as a tip result.

### B2. Size and sign of the tip deficit

**v2.0 on the fresh seeds** (30501-30520, n = 20 per q). Source: `reports/t2_refine_100526/100526report.md` section 6.1, table 6.1b, item `T2R:R2B-18`.

| | q = 50 | q = 60 |
|---|---|---|
| median signed peak error | −0.06155 | −0.06778 |
| mean \|peak error\| | 0.06305 | 0.0678 |
| runs within 0.05 | 5 of 20 | 4 of 20 |
| runs with a negative signed error | 20 of 20 | 20 of 20 |
| most negative signed error | −0.1458 (seed 30510) | −0.1237 |

**v2.0 confirmation** (`T2R:CF-02`, `T2R:CF-13`):
- 19/20 at q = 50 and 20/20 at q = 60.
- The one failure is q = 50, seed 30510: eta_2/DW = 0.005803 against the limit 0.005.

**v2.0 on the development seeds** (`parents_A`, n = 10 per q; 100526 table 6.1a; MS-R3 `05_decision_inputs.md` section 1):
- mean |peak error|: 0.0656 at q = 50, 0.0533 at q = 60;
- runs within 0.05: 2 of 10 at q = 50, 5 of 10 at q = 60.

### B3. Equilibrium quality despite the tip

- **[Verified, descriptive; derived by the PI]** In the 160 MS-R3 `t1` and `t10` runs:
  - eta_2/DW lies between 0.00013 and 0.00226, against the limit 0.005 (`results/ms_r3/analysis/per_run.csv`, column `eta_T_over_dw`);
  - all 160 runs pass every gate (`04_pilot.md` section 3).
- **PI reasoning, to be stated as such:** the deficit costs little payoff, because the learner's payoff is flat on the under-effort side of the tie.

### B4. Decomposition of the gap

**Smoothing part.**
- **[Verified]** The smoothing part equals e2*(0) sigma_2(0) / (sqrt(pi) q):
  - MS-R3 ratio 0.9992-0.9995 outside the collapsed run (`04_pilot.md` section 10, block `smoothing_formula`);
  - MS-R2: within 0.083 %.
- **[Verified, descriptive]** At s = 1 the smoothing part is 48-62 % of the gap in the four MS-R3 `t1` cells (two arms × two q), and 50 % / 53 % in `rehearsal_v2_0` (`05_decision_inputs.md` section 1).

**The noise landing.**
- MS-R2:
  - the landing lowered the smoothing part by 0.64-1.39 effort units;
  - the remainder rose by 0.30-1.40 effort units;
  - transmission ratios ranged from −1.04 to +0.54.
- MS-R3 transmission (`transmission.csv`):
  - `t1`: −0.43 to +0.28;
  - `t10`: 0.11, 0.60, 0.72, 0.69 (two intervals exclude 0);
  - `relu`: 0.06, 1.53, 0.14, 0.33 (none excludes 0).

**Model fit.**
- **[Verified, descriptive]** The quadrature model is closer to the observed gap at s = 16 than the additive model in 10 of 12 MS-R3 cells (`quadrature_check.csv`).
- For MS-R2's four cells, see the preamble of prompt 20, reproduced in MS-R3's `01_supervised_screen.md` section 4.

**Smoothing floor relative to e2*(0)** (`04_pilot.md` section 3.1):
- 2.3-2.7 % at s = 1;
- 0.6-0.7 % at s = 16.

### B5. What was tried, including the failures and unmet results that must appear

| Round | Item | Result |
|---|---|---|
| R1 | LR polish | Not met. |
| R1 | Larger batch | Not met. `A_batch_mb256` meets part (a) at q = 50 only. |
| R1 | Target-KL | Not met. |
| R1 | Concentration annealing | Not met. `A_anneal4` is worse at q = 60: +0.0118 [0.0051, 0.0195]. |
| R1 | Pathwise single-step | Not met. |
| R2b | Pathwise P20 | Not better than the matched PPO controls. `P20_lr3e-4` is worse at q = 50: +0.0127 [6.1e-05, 0.0268]. |
| R2b | Censored likelihood | Not separated from its baseline. |
| R2b | Peak share 0.5 | Meets (a) at both q, but violates (b): 3 runs at q = 60 cross the tail-mean limit. |
| R2c | Four start-distribution arms | No arm selected. Three meet (a) at q = 50; none at q = 60. |
| MS-R1 | Criterion | No arm meets it. The sampler arms meet (a) at q = 50 only (−0.023 to −0.031). `MS_s35a0` violates (b) once (eta_2/DW 0.005054). |
| MS-R1 | Budget control | 1600 → 2400 updates: −0.0126 [−0.0285, +0.0014] at q = 50 and −0.0085 [−0.0193, +0.0041] at q = 60. It reproduces 40-55 % (q = 50) and 52-112 % (q = 60) of the sampler arms' mean improvement. |
| MS-R1 | Stop rule at rho_2 = 0.05 | 0 of 100 rule-arm runs stopped before the cap. |
| MS-R1 | Polishing | Reached at q = 60, rarely at q = 50. Its effect is not separated from the sampler's. |
| MS-R2 | Criterion | Not met in any of four rows. `NL_bb_s4` at q = 60 lies above 0: +0.0114 [+0.0034, +0.0185]. |
| MS-R3 | Criterion | Not met in any of eight rows. Five `relu` failures (B6). `t10` does not transfer from the supervised screen. |

Sources:
- R1, R2b and R2c: 100526 sections 6.2-6.5.
- MS rounds: `reports/ms/r{1,2,3}/summary.md` and the criterion tables.

Also include the PI's withdrawn and half-supported readings (B7), and the sandbox discrepancy (B7).

### B6. MS-R3 in detail

All MS-R3 numbers are from development seeds 10501-10510, n = 10 per arm and q.

**Premise check** (supervised fit, bin-balanced, 56,000 steps; `results/ms_r3/supervised_screen/premise_check.json`). Median tip deficit, effort units, q = 50 / 60:

| Actor | q = 50 | q = 60 |
|---|---|---|
| `t1` | 1.64 | 6.32 |
| `relu` | 0.48 | 0.03 |
| `t10` | 0.53 | 0.48 |

- At q = 60, 6 of the 10 `t1` seeds plateau at 6.18-6.74.
- With four times the budget (224,000 steps), the `t1` deficit falls to 0.65 / 0.87 (`01_supervised_screen.md` sections 2.3 and 5).
- Pilot 4's least-squares fit of the same actor class (300,000 steps) reached a median of 0.00133 effort units at d = 0 (`T2R:R2B-17`, 100526 section 6.1).

**RL against the screen.** The RL median gap exceeds the screen's median deficit in 11 of 12 cells, by factors of 2.1-59 (`05_decision_inputs.md` section 6).

**`relu`: the typical run** (post-hoc robust table):
- median gap 0.84-2.36 effort units, against 2.11-3.77 for `t1`;
- in each q = 60 arm, 3-6 of 10 runs have a gap of at most 1 effort unit (`t1`: 0-1);
- the smoothing part, sigma_2(0) and the tail mean are lower in all eight cells.

**`relu`: the failures.** Five of the 40 runs at q = 50 fail G-A, from two (q, seed) cases:
- **Seed 10504 (collapse).**
  - The tie effort collapsed to the mean clamp after local update 825: e_hat_2(0) = 1e-4, eta_2/DW = 0.259.
  - Two arms share this run up to update 2001.
- **Seed 10506 (dead region).**
  - A dead middle-stratum region: three of four arms fail.
  - Symmetry error up to 0.547 of e2*(0).

At q = 60: 0 of 40 runs fail. Criterion part (b) is violated in all four `relu` rows; part (a) holds at both q only for `relu_st_s16`.

**`t10`.**
- Its first-layer units are about six times sharper: they bend over 24-30 units of d, against 159-173 for `t1`.
- Mean |peak error| is within 0.015 of `t1`'s in every arm.
- The tail mean is lower in all eight cells.
- RMSE_pos is lower in 6 of 8 cells, two with intervals below 0.
- No gate failures.

**Smoothing floor.** No arm reaches it at s = 16 (|peak| / floor):
- `relu` and `t10`: 3.1-7.4, outside the collapsed cell;
- `t1`: 5.6-7.6.

**R0.** Spearman correlation of R0 with |peak error| at the freeze (`r0_spearman.csv`):
- 1.000 for `t1` and `t10`;
- 0.993 at q = 50 and 0.970 at q = 60 for `relu`.

**Checks.** C-R6 20/20; C-MS5 80/80; C-NL 120/120; C-INIT 20/20 (`launch_checks.json`, `v20_reproduction_checks.json`).

**Trajectory.** Describe the trajectory of e_hat_2(0) over local updates 1800-2800 from block `trajectory` of `04_pilot.md` section 10. PI reading: mostly a plateau, with moves of about one effort unit in the seed mean between checks, and a few arms drifting up by up to about 1.7.

**Best development-seed arms** (`05_decision_inputs.md` section 1; `criterion_vs_parents_A.csv`). Mean |peak error| and runs within 0.05, q = 50 / 60:

| Arm | Mean \|peak error\| | Runs within 0.05 | Gate failures |
|---|---|---|---|
| `t10_st_s16` | 0.0363 / 0.0279 | 8 / 9 of 10 | none |
| `relu_st_s16` | 0.0297 / 0.0209 | 10 / 10 of 10 | one G-A failure at q = 50 (seed 10506) |
| `t1_st_s16` | 0.0401 / 0.0415 | 8 / 7 of 10 | none |

Against `parents_A` (1600 updates), `t1_st_s16` and `t10_st_s16` meet MS-R1's criterion. That comparison confounds budget, sampler and landing.

### B7. History of the PI's readings

- **After MS-R1:** the PI read the remaining deficit as a policy-noise floor (prompt 19).
  - MS-R2 found that the tie effort did not follow the lower noise. That observation is [Verified, descriptive]: no primary row was met, and the seed means of e_hat_2(0) are nearly the same at s = 1, 4 and 16.
  - The reading was withdrawn in the preamble of prompt 20.
- **Before MS-R3:** the PI read it as the actor's resolution at the kink (prompt 20). MS-R3 found:
  - it holds for the supervised fit (the premise check);
  - it is not sufficient in RL (`t10`'s non-transfer; the RL gap is much larger than the screen deficit);
  - `relu`'s typical run improved, with failures.
- **The PI's sandbox** (3 seeds; `reports/ms/r3/pi_record/sandbox_fit_tip*`) gave a `t1` median of 1.49 at q = 60. The repository screen gave 6.32, with 6 of 10 seeds at 6.18-6.74. The premise check is unaffected.

### B8. A derived quantity: the non-smoothing rounding width F_d

Label: [Verified, descriptive], derived by the PI, post hoc.

**Definition.** F_d = sqrt(w_eff² − (2 sigma_2(0)/sqrt(pi))²), using arm means of `w_eff` and `sigma_effort_at_0_t2` from `results/ms_r3/analysis/per_run.csv`. The term 2 sigma/sqrt(pi) is the smoothing part in units of d, i.e. smoothing / (e2*(0)/2q).

**Values, in units of d:**
- `t1`: 3.4-5.6;
- `t10`: 3.3-6.0;
- `relu`: 1.6-3.6, excluding `relu_bb_s1` and `relu_bb_s16` at q = 50, which contain the collapsed run.

**What it suggests.** The tanh actors keep a rounding of about 3.5-6 units of d whatever their input scale and noise level.

Recompute these values with a script, and report the per-run medians next to the arm-mean values.

### B9. Hypotheses and untested levers

All labelled [Hypothesis]; none was tested.

**H1: in RL, the cusp is estimation-limited, not capacity-limited, for tanh actors.**
- Consistent observations:
  - `t10`'s non-transfer;
  - F_d similar for `t1` and `t10`, and at s = 1 and s = 16;
  - tie weighting and budget help a little;
  - the plateau over updates 1800-2800.
- Test (not run): vary the number of near-tie samples per update (batch or near-tie share), with everything else fixed, for `t1` and `t10`; see whether F_d falls.

**H2: `relu` forms the cusp from the two well-sampled side slopes.**
- Test (not run): `relu`'s response to the near-tie share should be weaker than that of the tanh actors.

**H3: the `relu` failure modes.**
- **(a) The collapse.**
  - The collapsed run's tie mean sits at the hard mean clamp: mean = clamp(sigmoid(z0), mu_clamp, 1 − mu_clamp) in `agents/ppo_curriculum.py`. `torch.clamp` passes no gradient outside its bounds, which can keep a collapsed state from recovering.
  - The trigger is unknown: RMSE_pos/e2*(0) was already 0.35-0.41 at local updates 750-825, before the tie collapsed.
- **(b) The dead region.**
  - In every `relu` run, good and failed alike, 14-28 of the 64 first-layer units are never active on D_2 (`relu_units.csv`). Dead units are therefore only a candidate cause.
- Candidate fixes (untested): leaky ReLU; a mean map without a hard clamp.

**Untested levers.** The gain of each cannot be estimated.
- **(i) The critic.** It is unchanged: tanh on d/B. Under a tent-shaped policy, the value function has a kink at d = 0 from the effort cost. Whether the critic's rounding of that kink matters is unknown.
- **(ii) The opponent refresh interval.** The opponent is a lagged copy refreshed every 20 updates. Whether the interval matters is unknown.

### B10. What the evidence does not show (minimum list)

- **`relu`'s failure rate.** Two (q, seed) cases among 40 runs at q = 50, 0 of 40 at q = 60. Nor whether the fixes remove the failures.
- **Fresh-seed performance of any MS configuration.** None was confirmed; every MS number is from development seeds 10501-10510.
- **The mechanism** of the remainder, and of `t10`'s non-transfer.
- **Whether the noise landing helps under a kink-capable actor.** There is partial transmission under `t10` only.
- **Small effects.** Effects smaller than the intervals (half-widths 0.006-0.16 in MS-R3's primary rows) cannot be seen; an interval that contains 0 is not "no effect".
- **Other stop thresholds.** Whether another stop metric or threshold would make the plan's stop rule work at T=2. Only rho_2 = 0.05 was run, and R0 was studied as a monitor only.
- **Polishing alone.** The effect of polishing separately from the sampler (MS-R1).
- **Longer budgets.** RL budgets beyond 2800 terminal updates.
- **T=3.** Anything about it.

### B11. PI-side recommendation

Label in the report: "PI-side recommendation (input; the decision is the coworker's)".

**Path A.**
- Keep v2.0 as the T=2 solver. MS-R1..R3 produce no protocol change.
- Report the tip deficit as a characterised limitation:
  - its size and sign on the fresh seeds;
  - the exact smoothing part;
  - the unexplained remainder;
  - its small effect on eta_2;
  - the `relu` result as an ablation.

**Reasons:**
- the tanh-side levers tried so far (B5) met no criterion;
- the remaining effects are about one effort unit or less. That is the size of the seed spread: the seed SD of e_hat_2(0) within an arm is 0.45-1.56 effort units (MS-R2 `05_decision_inputs.md` section 2.2);
- the PI's two mechanism readings did not hold in RL, so a third, untested reading is a weak basis for a larger programme.

**If the coworker chooses Path B, the PI would start with `relu` robustness only:**
- leaky ReLU and a mean map without a hard clamp;
- stratified starts and the noise landing;
- 20 seeds per q (ten more development seeds) to estimate the failure rate;
- a pre-registered target on |peak error|, and a zero-failure requirement on the gates.

Adoption would still need a lock and a fresh-seed confirmation, and the actor change would carry into T=3.

**Not recommended now:** more tanh-side knobs; a mechanism round.

### B12. Workload references

Read the values from the records; do not estimate them.

| Round | Runs | Where to read the wall time |
|---|---|---|
| MS-R3 | 240 RL runs, 140 supervised fits, 20 C-R6 runs | Pilot launched 2026-10-08 00:24 with 40 workers (`results/ms_r3/pilot/launch_20261008_002423.json`). Per-run wall times are in `results/ms_r3/analysis/budget.csv`; the pilot's end is in the logs. |
| MS-R2 | 120 runs | The round's launch records and budget table. |
| MS-R1 | 120 pilot runs and 20 base runs | The round's launch records and budget table. |
| v2.0 confirmation | 40 runs | The confirmation records. |

A protocol change needs a lock, a re-rehearsal and a fresh-seed confirmation; the precedent is the v2.0 round. Person-time cannot be estimated.
