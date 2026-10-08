# T=2 status report: close now, or continue improving accuracy?

Folder `reports/t2_status_100826/`, branch `t2-status-pack` (from `origin/ms-r3` `be4fd202`), written 2026-10-08 for a coworker who did not take part in the work and who decides. **Status: decision pending; no experiment of any kind starts before the coworker's reply.** English body with a Chinese one-page summary (section 0). A tag such as `[M3-08]` is an item of `evidence/manifest.csv` (`M1-`..`M3-` the MS rounds, `PI-` the PI's prompts, reply and plan note, `BG-` background files, `T2R:` an item of the 100526 pack that is cited in place, `FIG-` a figure, `TBL-` a table produced by `report_scripts/tables.py`). Every number cites one of them.

## 0. 中文摘要

**问题（原文）：** 基于现有结果，我们应该在此阶段收口，还是继续提高精度？如果继续，优先改进什么、目标是什么？

**T=2 是什么，“解决”指什么。** T=2 是锦标赛博弈中最小的两阶段基准，有闭式均衡可作真值：两个 PPO 学习者自博弈（对手是滞后副本），学习 stage-2 与 stage-1 的 Beta 策略，q 取 50 和 60；闭式均衡只用于评估，不进入训练 [T2R:PL-01][PI-05]。训练回报用博弈模型给出的条件期望回报和期望续值表，不是原始采样回报；这与 `.claude/CLAUDE.md` 为原有 runner 写的“只用采样奖励”不变量不同，是 v2 管线有意为之，论文里能声称的内容要相应收窄 [BG-01][BG-02]。“解决”指协议 v2.0：门槛 G-A、G-F、G-N、G-S 加 fresh-seed 确认；确认通过，q=50 为 19/20，q=60 为 20/20 [T2R:CF-02]。

**当前精度（v2.0，fresh seeds 30501-30520，每个 q 20 个 run）[T2R:R2B-18]。** stage-2 平局点 d=0 的 |peak 误差| 均值为 0.0630（q=50）和 0.0678（q=60）；误差不超过 0.05 的 run 为 5/20 和 4/20；40 个 run 的符号全为负，即学到的努力低于闭式解；缺口中位数 4.31 和 3.95 个努力单位 [T2R:R2B-18][TBL-accuracy-a][TBL-accuracy-b]。这个缺口对支付影响很小：MS-R3 的 160 个 `t1`/`t10` run 中 eta_2/DW 在 0.00013 到 0.00226 之间，门槛为 0.005，全部通过 [TBL-eta]（“为什么影响小”是 PI 的解释，属 [Hypothesis]）。开发种子（10501-10510，每个 arm 和 q 10 个 run）上最好的 arm `relu_st_s16` 均值 0.0297 和 0.0209，两个 q 上 10/10 个 run 不超过 0.05，但 q=50 有一个 G-A 失败；所有 MS 数字都是开发种子，没有任何 MS 配置经过 fresh-seed 确认 [TBL-accuracy-a][M3-01]。

**已确定（[Verified]）。** (1) v2.0 通过门槛和 fresh-seed 确认 [T2R:CF-02]。(2) 缺口的平滑部分严格等于 e2*(0)·σ_2(0)/(√π q)，比值 0.99917 到 0.99971 [TBL-formula]；v2.0 开发种子上平滑部分占缺口 50% 和 53% [TBL-share]。(3) 六轮针对 tip 的干预（R1、R2b、R2c、MS-R1、MS-R2、MS-R3）没有一个满足预先登记的判据 (a) 和 (b)，所以没有可采纳的 tip 修复，也没有 v2.1 [TBL-interventions]。(4) 噪声着陆把平滑部分降到预期值，但余项上升、缺口不降 [M2-01]。

**未决。** 余项的机制（[Hypothesis]）；`relu` 的失败率（80 个 run 中 5 个失败，来自两个 (q, seed) 案例，[Insufficient evidence]）[TBL-relufail]；任何 MS 配置在 fresh seeds 上的表现（没有数据）；第三个未经检验的机制读法不足以支撑大规模新项目。

**路径 A（现在收口）。** 不再新跑；v2.0 仍是 T=2 求解器；MS-R1 到 MS-R3 不产生协议改动；把 tip 缺口作为已刻画的局限写出（大小、符号、精确的平滑部分、未解释的余项、对 eta_2 的小影响、`relu` 作为消融）。**路径 B（继续提精度）。** 主指标是 stage-2 d=0 的 |peak 误差| 及误差不超过 0.05 的 run 占比；护栏是 RMSE_pos、tail mean、eta_2、门槛通过率、stage-1 误差；候选措施和各自的证据、可记录的工作量、风险见第 10 节；任何采纳都需要锁定协议、重新排练和 fresh-seed 确认，改 actor 还会带入未评估的 T=3 [T2R:RR-03]。

**PI 侧建议（输入；决定权在同事）。** 倾向路径 A：保留 v2.0，把 tip 缺口作为受限结论报告；理由是已试的 tanh 侧手段均未达判据，剩余效应约一个努力单位或更小，与种子间的离散度同量级（arm 内 e_hat_2(0) 的种子 SD 为 0.45 到 1.56 个努力单位 [M2-02]），而且 PI 的两个机制读法在 RL 中都没有成立。若选路径 B，PI 会只从 `relu` 的稳健性开始（leaky ReLU、不带硬 clamp 的均值映射、分层起点加噪声着陆、每个 q 20 个种子估失败率、预先登记 |peak 误差| 目标并要求门槛零失败）[PI-06]。

**请同事回复：** (i) 收口，还是继续；(ii) 若继续，先做哪项措施，目标是什么（指标、数值、种子集）。

## 1. Purpose and reading guide

**Who decides what.** The coworker decides. This report assembles the state of the T=2 work so that they can answer one question, verbatim:

> 基于现有结果，我们应该在此阶段收口，还是继续提高精度？如果继续，优先改进什么、目标是什么？
>
> (Based on the current results, should we close T=2 at this stage, or continue improving its accuracy? If we continue, what should be improved first, and what is the target?)

Nothing here selects a path. Section 10 contains the PI's recommendation, labelled as an input, and one observation of the assistant that wrote this folder, labelled as such; neither decides. No experiment, lock, confirmation or message to anyone follows from this report.

**Reading time.** Sections 0, 7.2, 9 and 10 are enough to decide (about fifteen minutes); everything else supports them, and the details stay in the linked round reports (`../ms/r1/summary.md`, `../ms/r2/summary.md`, `../ms/r3/summary.md`, `../t2_refine_100526/README.md`).

**Labels (sections 7-10).** Every claim carries one of three labels.
- **[Verified]**: established by a pre-registered criterion, a gate, a deterministic check (bit-identity or reproduction) or an exact formula check; the evidence item is named. A plain measurement that claims no more than what was measured is **[Verified, descriptive]**.
- **[Hypothesis]**: an interpretation or a mechanism, including every PI reading; each comes with the experiment that would test it and whether it was run.
- **[Insufficient evidence]**: a question the records cannot answer, for example an interval that contains 0 with ten seeds (which is not "no effect"), a failure rate estimated from two (q, seed) cases, or any fresh-seed claim about an MS configuration.

Post hoc tables are marked post hoc. Development-seed and fresh-seed numbers are never mixed in one statistic: every accuracy number names its seed set and n. An asterisk `*` in a table marks a 95% percentile-bootstrap interval that excludes 0 (definition in section 4) [M3-04].

**Inputs that are not evidence.** Appendix A (the PI's plan note) and Appendix B (the PI-side reading after MS-R3) of the prompt of this round are inputs [PI-05][PI-06]. Every number taken from them was checked against its record; where a record differs, the record is used and the difference is listed in `pi_record/01_factcheck.md` and section 11.

## 2. The T=2 problem

**The game and its parameters.** Two players choose an effort in [0, 100] at each of two stages; the lead d entering stage 2 (a state) and the final shock decide the win probability through F_ξ, the distribution of the difference of two shocks, which has the triangular density (2q - |z|)/(4q²) on [-2q, 2q] [T2R:100526report section 3.1]. The terminal reward of the learner is w_L + DW F_ξ(d + e - e_opp) - k e² [PI-01]. The parameters are q in {50, 60}, k = 1/3500, w_H = 6, w_L = 2, DW = w_H - w_L = 4 and B = 100 + 2q (the scale of the state input d/B of the actor) [TBL-params][PI-04].

<!-- TBL:params -->
<!-- /TBL:params -->

The closed-form equilibrium of the last stage, e2*(d), is a tent: it peaks at d = 0, where e2*(0) = DW/(4qk) (70 at q = 50 and 58.33 at q = 60 [TBL-params]), because the triangular density has its kink at the tie, and it is 0 in the tail |d| >= 2q [PI-01]. "Effort units" below are units of the effort variable (range 0 to 100), so a gap of 4.3 effort units at q = 50 is 6.1% of e2*(0) [T2R:R2B-18].

**What is learned.** For each stage a Beta policy: a 2-64-64 tanh network whose inputs are a stage feature and d/B outputs the Beta mean (sigmoid, clamped to [1e-6, 1 - 1e-6], times the effort range) and a concentration; the learner trains by self-play against a lagged copy of itself that is refreshed every 20 updates, with a critic of the same size [BG-03][PI-01][PI-04][PI-06]. The learned effort at a state is the Beta mean (effort = 100 times the mean) and evaluation uses the mean, not the mode [PI-04][BG-02].

**Where the closed form enters: evaluation only.** The closed-form equilibrium is used to report errors (peak error, RMSE_pos, tail mean, stage-1 error) and in the offline supervised screen of MS-R3; it never enters the rollout, the sampler, the schedule, the stop rule or any criterion of the runs [PI-01][PI-04]. A test asserts that the stop rule and the sampler make identical decisions when the closed-form functions are replaced by stubs [M1-04].

**Where the game model enters training.** The v2 pipeline does not train on sampled one-step outcomes alone.
- The terminal-stage return uses the conditional expectation over the shock given the sampled actions, `reward_mode = expected` [T2R:PL-01][BG-01]; the pilot that chose the estimator is `reports/v2/pilot1_reward_estimator.md` [BG-01].
- The stage-1 return replaces the sampled continuation by a table value Ṽ2(y) = E_z[g_2(y + z)] built once from the frozen stage-2 Beta mean (expected continuation, in v2.0 since the R1 round) [T2R:RR-03].

**The sampled-reward invariant, plainly.** `.claude/CLAUDE.md` lists "sampled training rewards only; closed-form win probability and expected payoff are evaluation-only" as a critical invariant of the original runners [BG-02]. The v2 pipeline deliberately differs from it: the shock distribution, i.e. the game model, enters the training return twice (the conditional-expectation reward and the continuation table) [T2R:PL-01][BG-01]. What does not enter is the closed-form equilibrium effort e*. A claim that "PPO agents learn the equilibrium from sampled tournament outcomes" therefore cannot rest on the v2.0/MS results; what they support is "PPO with an expected-reward, expected-continuation estimator, which uses the game's shock distribution but not the equilibrium, recovers the equilibrium to the accuracy of section 7". This is a statement about the scope of the claims, not a defect that this report found.

## 3. Goals

**The role of T=2.** T=2 is the base case of the project with a closed-form ground truth, so that the learned policies can be scored against the equilibrium [PI-05][BG-02]. Later horizons (T=3) are out of scope here and enter only where a choice has a consequence for them, as "not evaluated".

**The session goals (Appendix A, the PI's plan note) and their status after MS-R1..R3.** Baseline: conditional expected reward + expected continuation + backward freeze, settled (protocol v2.0) [PI-05][T2R:PL-01].

<!-- TBL:goals -->
<!-- /TBL:goals -->

## 4. Definitions

- **d, q, k, DW, B, e2*(d), effort units.** d is the state entering a stage (the lead); q is the half-width of the uniform shock; k = 1/3500 the quadratic effort cost coefficient; DW = w_H - w_L = 4; B = 100 + 2q; e2*(d) the closed-form stage-2 equilibrium effort (a tent with e2*(0) = DW/(4qk)); effort units are the units of the effort variable, 0 to 100 [TBL-params][T2R:PL-02].
- **Peak error.** The signed peak error is (ê2(0) - e2*(0))/e2*(0), with ê2(0) the Beta mean of the stage-2 policy at d = 0, on the final verifier tier, at the terminal freeze of the stage; |peak error| is its absolute value; negative means the learned effort is below the closed form [T2R:100526report section 6]. The location-free peak error compares the maximum over the recovery grid of ê2(d), wherever it lies, with e2*(0) [M3-04].
- **The gap** is e2*(0) - ê2(0) in effort units. Where the signed peak error is negative, |peak error| = gap/e2*(0) [M3-03].
- **sigma_2(0), the smoothing part, the remainder, w_eff.** sigma_2(0) is the standard deviation of the learned effort at d = 0 (the policy's own noise). The smoothing part is e2*(0) - e_σ(0), where e_σ(0) is the tie effort of the game both players actually play when their actions carry that noise; it equals e2*(0) σ_2(0)/(√π q) [M2-05][TBL-formula]. The remainder is e_σ(0) - ê2(0), so gap = smoothing part + remainder. w_eff = gap/(e2*(0)/2q) is the gap in units of d (the rounding width of the tent's tip) [M3-04]. The additive model predicts gap(s=16) = gap(s=1) - (smoothing(s=1) - smoothing(s=16)); the quadrature model predicts sqrt(F² + smoothing(s=16)²) with F² = gap(s=1)² - smoothing(s=1)² [M3-04][TBL-quad].
- **R0, R, Delta_2.** From the development DP-BR verifier, with no closed form: r_2(d) = |ê2(d) - ẽ2(d)| is the distance of the policy mean from the one-step best response ẽ2(d); R0 = r_2(0)/s_2 with s_2 = ẽ2(0) (the tie residual); R is the maximum of r_2(d)/s_2 over the non-tail region; Delta_2 is the maximum one-step deviation gain, so Delta_2/DW is eta_2/DW at the terminal stage [PI-01].
- **RMSE_pos, tail mean, eta_2/DW.** RMSE_pos/e2*(0): the root-mean-square error of ê2 against e2* over recovery-grid nodes with |d| < 2q, divided by e2*(0); tail mean/e2*(0): the mean of ê2 over nodes with |d| >= 2q, divided by e2*(0); eta_2/DW: the largest one-step deviation gain over the final-tier stage-2 grid, divided by DW [T2R:PL-02][PI-01].
- **Gates and their limits** [T2R:PL-02]. G-A (end of the terminal stage, final tier): eta_2/DW <= 0.005, RMSE_pos/e2*(0) <= 0.05 and tail mean/e2*(0) <= 0.02 [T2R:PL-02]. G-F (end of stage 1): the maximum one-step gain of the full policy, over DW, <= 0.01 [T2R:PL-02]. G-N: the development-tier and final-tier values of eta_2/DW and of the stage-1 gain differ by at most 0.001 [T2R:PL-02]. G-S (new in v2.0): |ê1(0) - e1*(0)|/e1*(0) <= 0.05 [T2R:PL-02]. A run passes if all four hold. In the MS tables "stage-2 gate failures" means G-A or the eta part of G-N, and "stage-1 gate failures" means G-F, the Gmax part of G-N or G-S.
- **Criterion parts (a) and (b), and the bootstrap.** For an arm against its comparator, paired by (q, seed): [M3-04] (a) the 95% percentile bootstrap interval of the mean paired difference of |peak error| (arm minus comparator; negative is better) lies below 0 at both q; (b) no run that passes G-A with its G-N(eta) part under the comparator fails it under the arm. The bootstrap uses 10,000 resamples of the ten paired seeds with a fresh generator per (q, statistic); the generator seeds are 20261003 (R1), 20261004 (R2b), 20261005 (R2c), 20261006 (MS-R1), 20261007 (MS-R2) and 20261008 (MS-R3) [T2R:RR-01][T2R:RR-04][T2R:RR-06][M1-04][M2-04][M3-04]. The criterion is descriptive: it is not a gate and it selects nothing [M1-02].
- **Seed blocks.** Development 10501-10510 (all arms of all rounds, ten seeds, both q); the v2.0 confirmation block 30501-30520 (twenty fresh seeds, both q); 40501-40520 is reserved and was never used [T2R:PL-01][M3-01].
- **Starts.** Bin-balanced: the locked sampler that draws the stage-2 starting state uniformly over bins of width 10 [PI-01]. Stratified (`stratified_priority`): the starts are split into near-tie bins (|d| < 20), middle bins and tail bins with shares lambda_P, lambda_M and lambda_T; lambda_T is fixed at the bin-balanced tail share (0.5000 at q = 50, 0.4545 at q = 60) so that tail coverage cannot fall; alpha mixes a verifier-guided focus distribution (proportional to the residual map) into the non-tail part; in the MS-R2/R3 arms lambda_P = 0.35 and alpha = 0.5 [PI-01][PI-03].
- **The noise landing and s.** The terminal stage trains 2800 updates; at local updates 2001-2200 the Beta concentration scale is ramped from 1 to s, held to 2400, and the learning rate decays from 3e-4 to 3e-5 over 2401-2800; s = 1 means no landing [PI-03]. A larger s lowers sigma_2(0) by about 1/sqrt(s) [M2-01].
- **Arms.** `MS_*` (MS-R1): `MS_base` (= `parents_A`, 1600 updates), `MS_base2400` (budget control, bin-balanced, 2400 updates), `MS_rule` (rule with polishing, global sampler bin-balanced in distribution), `MS_s25a0`, `MS_s25a5`, `MS_s35a0`, `MS_s35a5` (rule with stratified sampler, lambda_P 0.25 or 0.35, alpha 0 or 0.5) [M1-01]. `NL_{bb,st}_s{1,4,16}` (MS-R2): bin-balanced or stratified starts, scale s [M2-01]. `{t1,relu,t10}_{bb,st}_s{1,16}` (MS-R3): actor variant `t1` (the current tanh actor), `relu` (ReLU hidden units) or `t10` (tanh with the d input multiplied by 10), bin-balanced (`bb`) or stratified (`st`) starts, scale s in {1, 16}; the critic is unchanged [M3-01].
- **`parents_A` and `rehearsal_v2_0`.** `parents_A` is the v2.0 terminal-stage state of the twenty development runs (seeds 10501-10510, both q) and `rehearsal_v2_0` the full v2.0 re-rehearsal of the same seeds; they share the same terminal stage (MS-R1 check C-MS1: the new runner with legacy settings reproduces it bit for bit, 20 of 20) [M1-06][M1-22][M1-23].

## 5. The current approach

**(a) The locked solver v2.0** [T2R:PL-02][T2R:RR-03].
- Pipeline: terminal stage first (Phase A, 1600 updates: learning rate 3e-4 constant to local update 1200, then linear to 3e-5 over 1201-1600), then stage 1 (Phase B, 600 updates, learning rate linear 3e-4 to 3e-5), with the expected continuation table in Phase B; 512 episodes per update; weight exports every 25 updates [T2R:PL-01].
- Gates G-A, G-F, G-N and G-S as in section 4; a run passes if all hold [T2R:PL-02].
- Confirmation design: fresh seeds 30501-30520 for both q (40 runs), pass rule at least 18 of 20 per q, exact Clopper-Pearson 95% intervals; a re-rehearsal on the development seeds had to pass first [T2R:PL-01][T2R:CF-01].

**(b) The MS stage-wise runner** (`run/run_ms_stagewise.py`; entry points of the line in `reports/ms/README.md`). It generalises the pipeline to T stages, one stage per phase from the last to the first, with exact nested continuation tables, and adds the following, all behind keys whose absence leaves every existing path bit-identical [PI-01][M1-01].
- The stop rule with a first-order residual: a check is eligible when Delta_t <= 0.005, R_t <= rho_t, the tail residual <= 0.02 and the concentration <= 0.04; M = 3 consecutive eligible checks stop the block; otherwise the residual is classified as broad (continue global training) or localised (a targeted polishing block) [PI-01]. rho_2 = 0.05 and rho_1 = 0.03 were pre-registered [PI-02].
- Coverage-constrained stratified starts, with lambda_P, lambda_T and alpha as in section 4 [PI-01].
- The stage-wise freeze: each stage ends in a landing window (LR 3e-4 to 3e-5), is evaluated on both verifier tiers, and is frozen as a snapshot before the next phase starts [PI-01].
- The noise landing (MS-R2) and the actor variants `relu` and `t10` (MS-R3, `BetaActor.variant`) [PI-03][PI-04].

**The checks that keep v2.0 intact** (each is a stop-and-report if it fails): C-R4, the unchanged v2.0 entry point reproduces `rehearsal_v2_0` on the MS-R1 code, 20 of 20 [M1-20]; C-R6, the same on the MS-R3 code, 20 of 20 [M3-22]; C-MS1, the new runner with legacy settings equals `parents_A`, 20 of 20 [M1-22]; C-MS2, `MS_base2400` equals `parents_A` through update 1201, 20 of 20 [M1-01]; C-MS3 and C-MS4, the MS-R2 arms equal their MS-R1 references through update 2001, 20 of 20 each, and C-NL, the s > 1 arms equal the s = 1 arm through update 2001, 80 of 80 [M2-01]; in MS-R3, C-INIT (identical initial weights across actors) 20 of 20, C-NL 120 of 120 and C-MS5 80 of 80 [M3-01]. C-MS5 says the four MS-R3 `t1` arms are bit-identical re-runs of MS-R2's `NL_*` arms: they are not replications [M3-03].

## 6. Completed work

<!-- TBL:rounds -->
<!-- /TBL:rounds -->

Prompts 12-14 were delivered to the earlier sessions but are not available in the repository; prompt 15 is a transcription [T2R:100526report]. Dates, heads and prompt files of the MS rounds are those of their summaries and of `git ls-remote` at the start of this round (`pi_record/00_build_log.md`).

**The PI's decisions at each gate of this session**, as the prompts and the G1 reply record them:

<!-- TBL:gates_pi -->
<!-- /TBL:gates_pi -->

## 7. Key results

### 7.1 v2.0 as the certified T=2 solver

- **[Verified]** (pre-registered rule, fresh seeds 30501-30520) The confirmation passed: 19 of 20 runs at q = 50 (exact 95% interval [0.7513, 0.9987]) and 20 of 20 at q = 60 ([0.8316, 1.0000]) against the rule of at least 18 of 20 [T2R:CF-02].
- **[Verified]** (gate) The one failed run is q = 50 seed 30510: it fails G-A through its eta_2 part, eta_2/DW = 0.005803 against the limit 0.005; its RMSE_pos/e2*(0) (0.0451 against 0.05) and tail mean (0.008822 against 0.02) pass, and G-F, G-N and G-S pass [T2R:CF-13][T2R:100526report section 3.5]. G-S passes in 20 of 20 runs at both q [T2R:CF-02]. A read-only diagnostic of the failed run is descriptive and did not decide a cause [T2R:RR-11].
- **[Verified, descriptive]** Stage-1 error (fresh seeds): median |ê1(0) - e1*(0)|/e1* 0.0131 (q = 50) and 0.0117 (q = 60), maximum 0.0464 and 0.0324, no run above the 0.05 limit [TBL-stage1]; the development re-rehearsal gives 0.0120 and 0.0101 [TBL-stage1].

<!-- TBL:stage1_ref -->
<!-- /TBL:stage1_ref -->

### 7.2 Current accuracy

Table 1 gives the peak error; table 2 the decomposition and the guard rails. **Confirmed rows** are those with fresh seeds (v2.0 only); every other row is **development-only** (seeds 10501-10510, n = 10 per q), not confirmed. `parents_A` and `rehearsal_v2_0` are the same terminal stage (C-MS1), so their peak columns agree; `parents_A` has no stage-1 phase. The MS-R3 `t1` arms are bit-identical re-runs of MS-R2's `NL_*` arms (C-MS5), not replications [M3-01]. `relu_bb_s16` at q = 50 contains the collapsed run (seed 10504), which dominates its mean and its eta_2 maximum [M3-03].

**Table 1. Peak accuracy at d = 0, terminal freeze, final tier** [TBL-accuracy-a].

<!-- TBL:accuracy_a -->
<!-- /TBL:accuracy_a -->

**Table 2. Decomposition and guard rails** (gap in effort units; the smoothing part and the remainder of the fresh-seed rows are those of the pack's smoothed-game decomposition [T2R:R2B-18]; `parents_A` carries no decomposition columns, see `rehearsal_v2_0`) [TBL-accuracy-b].

<!-- TBL:accuracy_b -->
<!-- /TBL:accuracy_b -->

- **[Verified, descriptive]** On the fresh seeds v2.0 has mean |peak error| 0.0630 and 0.0678 with 5 and 4 of 20 runs within 0.05, and 39 of 40 runs pass G-A [T2R:R2B-18][TBL-accuracy-a][TBL-accuracy-b].
- **[Verified, descriptive]** On the development seeds the three arms named by the decision lie lower: `relu_st_s16` 0.0297 and 0.0209 (10 and 10 of 10 runs within 0.05, one G-A failure at q = 50), `t10_st_s16` 0.0363 and 0.0279, `t1_st_s16` 0.0401 and 0.0415, against 0.0656 and 0.0533 for `parents_A` [TBL-accuracy-a]. Against `parents_A` with MS-R1's criterion, `t1_st_s16` and `t10_st_s16` are "met"; that comparison confounds budget, sampler and landing [M3-10][M2-02]. The matched-budget comparison is `MS_base2400`: 0.0530 and 0.0449 [TBL-accuracy-a].
- **[Insufficient evidence]** Whether any of these development-seed gains survive on fresh seeds: no MS configuration was confirmed [M3-01].

### 7.3 The stage-2 tip deficit

**Size and sign.**
- **[Verified, descriptive]** On the fresh seeds the median gap is 4.31 effort units at q = 50 and 3.95 at q = 60, i.e. mean |peak error| 0.0630 and 0.0678 of e2*(0); the signed error is negative in 40 of 40 runs, the most negative -0.1458 (seed 30510) [T2R:R2B-18][TBL-accuracy-b][TBL-signs].
- **[Verified, descriptive]** The deficit is below the closed form in every v2.0 run (40 of 40 fresh, 20 of 20 development) and in all 160 MS-R3 `t1` and `t10` runs, but not in every run of every round: eight runs have a non-negative signed error, one each in R2b, R2c and MS-R1 (at most +0.0009) and five in MS-R3, all `relu` (up to +0.0123) [TBL-signs][TBL-nonneg]. This corrects the statement of the PI-side reading (Appendix B1) that the learned effort is below the closed form "in every run of every round" [PI-06].

<!-- TBL:signs -->
<!-- /TBL:signs -->

<!-- TBL:nonneg -->
<!-- /TBL:nonneg -->

**Decomposition and the formula check.**
- **[Verified]** (exact formula check) The smoothing part equals e2*(0) σ_2(0)/(√π q): the ratio of the recorded smoothing part to the formula lies between 0.99917 and 0.99971 in every MS-arm run (largest deviation 0.083%) except the two runs of the collapsed `relu` policy, whose tie effort is 1e-4 (ratios 0.0000 and 0.0472) [TBL-formula].

<!-- TBL:formula -->
<!-- /TBL:formula -->

- **[Verified, descriptive]** At s = 1 the smoothing part is 48-62% of the gap in the four `t1` cells of MS-R3 and 50% and 53% in `rehearsal_v2_0`; the rest is the remainder [TBL-share]. At s = 16 the smoothing part is 13-18% of the gap [TBL-share].

<!-- TBL:share -->
<!-- /TBL:share -->

- **[Insufficient evidence]** What the remainder is. It is the part of the gap that the policy's own noise does not explain; the records do not give its mechanism (sections 7.4, 7.5 and 9).

**Additive against quadrature.**
- **[Verified, descriptive]** The quadrature model is closer to the observed gap at s = 16 than the additive model in 10 of 12 MS-R3 cells (the exceptions are `relu_bb` at q = 60 and `t10_st` at q = 50) [TBL-quad]. For MS-R2's four `t1` cells see the preamble of prompt 20 [PI-04].
- **[Hypothesis]** The reading behind the quadrature model is that the fit rounding and the noise smoothing combine like two kernel widths. It was written down before MS-R3 and tested only through this fit; a test that would separate it from other readings (for example a run with F varied at fixed s) was not run [M3-04][M3-03].

<!-- TBL:quad -->
<!-- /TBL:quad -->

**The effect on payoff.**
- **[Verified, descriptive]** In the 160 MS-R3 `t1` and `t10` runs eta_2/DW lies between 0.00013 and 0.00226 against the limit 0.005 and every run passes every gate [TBL-eta][M3-03]; on the fresh seeds eta_2/DW is at most 0.0058 and 39 of 40 runs pass G-A [TBL-accuracy-b].
- **[Hypothesis]** The PI's reading is that the deficit costs little payoff because the learner's payoff is flat on the under-effort side of the tie [PI-06]. A test would compare the learner's payoff along effort offsets at the tie; it was not run.

<!-- TBL:eta -->
<!-- /TBL:eta -->

### 7.4 What moved the tip and what did not

One row per arm and comparison, all rounds. Paired change of |peak error| (arm minus comparator; negative is better) with its 95% interval at each q (development seeds, ten pairs per q) [TBL-interventions]. The arm's guard rails are the maximum tail mean over both q against the 0.02 limit and the arm runs failing G-A or G-N(eta), of 20 [T2R:PL-02]; then part (b), and the verdict as each round pre-registered it. `A_ctrl200`, `A_ctrl200_lr3e-4` and `MS_base2400` are controls, not candidates; the MS-R1 secondary rows compare the arms with the matched-budget control `MS_base2400` [TBL-interventions].

<!-- TBL:interventions -->
<!-- /TBL:interventions -->

- **[Verified]** (pre-registered criteria) No intervention of any of the six tip rounds met its criterion with part (b) holding. Part (a) holds at both q for `A_peak50` (R2b) and for `relu_st_s16` (MS-R3), and for each of them part (b) is violated; the pre-registered rule of R2c selected no arm [TBL-interventions][T2R:R2C-03].
- **[Verified]** (criterion) The failures and unmet results the table contains: R1: polishing, larger batch (`A_batch_mb256` meets (a) at q = 50 only), target-KL, annealing (`A_anneal4` is worse at q = 60: +0.0118 [+0.0051, +0.0195]) and the one-step pathwise arm; R2b: pathwise P20 (`P20_lr3e-4` worse than its control at q = 50: +0.0127 [+0.0001, +0.0268]), the censored likelihood, and peak share 0.5, which meets (a) at both q but has three q = 60 runs above the tail limit; R2c: three of four arms meet (a) at q = 50, none at q = 60; MS-R1: sampler arms meet (a) at q = 50 only (-0.0227 to -0.0313) and `MS_s35a0` has one run with eta_2/DW 0.005054; MS-R2: no row, one cell above 0 (`NL_bb_s4`, q = 60: +0.0114 [+0.0034, +0.0185]); MS-R3: no row, five `relu` failures [TBL-interventions][T2R:R2B-02][M1-09][M2-09][M3-09].
- **[Verified]** (exception that does not count) R1's expected continuation (method 6) met its pre-registered criterion, but for the stage-1 error; it became part of v2.0 and is not a tip result [T2R:RR-01].
- **[Verified, descriptive]** The budget control: 1600 to 2400 updates changes |peak error| by -0.0126 [-0.0285, +0.0014] at q = 50 and -0.0085 [-0.0193, +0.0041] at q = 60, and reproduces 40-55% (q = 50) and 52-112% (q = 60) of the sampler arms' mean improvement; no matched-budget interval of the secondary table excludes 0 on |peak error| [TBL-budget][M1-01][M1-02].

<!-- TBL:budget -->
<!-- /TBL:budget -->

- **[Verified]** (MS-R2, reading withdrawn) The noise landing lowered the smoothing part by 0.64-1.39 effort units in all eight cells, as predicted, while the remainder rose by 0.30-1.40; the seed means of ê2(0) are nearly the same at s = 1, 4 and 16, so the tie effort did not follow the lower noise; the PI's noise-floor reading was withdrawn in prompt 20 [M2-01][M2-02][PI-04].
- **[Insufficient evidence]** Effects smaller than the intervals (half-widths 0.006-0.16 in MS-R3's primary rows, 0.008-0.017 in MS-R2's) cannot be seen with ten seeds; an interval that contains 0 is not "no effect" [M3-02][M2-01].

### 7.5 MS-R3 in detail

All numbers of this section are from the development seeds 10501-10510, n = 10 per arm and q [M3-08].

**The premise check.**
- **[Verified]** (pre-registered check, PASS) In the offline supervised fit at the RL budget (56,000 steps, bin-balanced) the median tip deficit is 1.64 (q = 50) and 6.32 (q = 60) effort units for `t1`, 0.48 and 0.03 for `relu`, 0.53 and 0.48 for `t10`; at q = 60 six of ten `t1` seeds plateau at 6.18-6.74 [TBL-premise][M3-23].
- **[Verified, descriptive]** With four times the budget the `t1` deficit falls to 0.65 and 0.87; the least-squares fit of the same actor class of pilot 4 (300,000 steps) reached a median of 0.00133 effort units at d = 0 [TBL-premise][M3-36][T2R:R2B-17].

<!-- TBL:premise -->
<!-- /TBL:premise -->

**RL against the screen.**
- **[Verified, descriptive]** The RL median gap at s = 1 exceeds the screen's median deficit in 11 of 12 cells, by factors of 2.1 to 59 (the exception is `t1`, bin-balanced, q = 60, where the screen's actors often plateau) [TBL-rlscreen][M3-02].

<!-- TBL:rl_vs_screen -->
<!-- /TBL:rl_vs_screen -->

**`relu`: the typical run improves (post hoc robust table).**
- **[Verified, descriptive]** The median gap of `relu` is 0.84-2.36 effort units against 2.11-3.77 for `t1` (lower in all eight cells); in each q = 60 arm 3-6 of 10 `relu` runs have a gap of at most 1 effort unit (`t1`: 0-1); the smoothing part, sigma_2(0) and the tail mean are lower in all eight cells [TBL-relutyp][M3-02].

<!-- TBL:relu_typical -->
<!-- /TBL:relu_typical -->

**`relu`: the five failed runs.**
- **[Verified, descriptive]** Five of the 40 `relu` runs at q = 50 fail G-A (none of 40 at q = 60, none of the 160 `t1` and `t10` runs); they come from two (q, seed) cases, and all five also fail a stage-1 gate [TBL-relufail]. Part (b) is violated in all four `relu` rows and holds in all four `t10` rows; part (a) holds at both q only for `relu_st_s16` [M3-01].
- **Seed 10504 (collapse), known:** the tie effort was learned normally to local update 825, then R0 went from 0.070 to 0.978 at 850; the tie effort ended at the mean clamp, ê2(0) = 1e-4, eta_2/DW 0.259, stage-1 error -1.0000; RMSE_pos/e2*(0) was already 0.35-0.41 at local updates 750-825; the same seed under the stratified arms passes; two arms share the run up to update 2001 [M3-03][TBL-relufail].
- **Seed 10506 (dead region), known:** a dead middle-stratum region with a good tie; three of the four arms fail G-A; the symmetry error is up to 0.547 of e2*(0) (38.3 effort units at |d| = 35 in `relu_st_s1`) [M3-03].
- **[Insufficient evidence]** The failure rate (2 (q, seed) cases among 40 runs at q = 50, 0 of 40 at q = 60) and the mechanism of the failures: 14-28 of the 64 first-layer `relu` units are never active on D_2 in good and failed runs alike (post hoc), so dead units are a candidate cause, not an established one [TBL-reluunits][M3-02].
- **[Hypothesis]** (H3, [PI-06]; not tested) The collapsed run's tie mean sits at the hard clamp `mu = clamp(sigmoid(z0), 1e-6, 1 - 1e-6)` (line 99 of `agents/ppo_curriculum.py`), and `torch.clamp` passes no gradient outside its bounds, which could keep a collapsed state from recovering [BG-03]; the trigger is unknown. Candidate fixes: a leaky ReLU, a mean map without a hard clamp; neither was run.

<!-- TBL:relu_fail -->
<!-- /TBL:relu_fail -->

<!-- TBL:relu_units -->
<!-- /TBL:relu_units -->

**`t10`: no transfer.**
- **[Verified, descriptive]** The first-layer units of the RL `t10` actors are about six times sharper than `t1`'s (the sharpest bends over 24-30 units of d against 159-173), yet mean |peak error| is within 0.015 of `t1`'s in every arm; the tail mean is lower in all eight cells, RMSE_pos in six of eight (two intervals below 0), and no run fails a gate [TBL-t10][M3-02].
- **[Verified, descriptive]** The supervised screen's advantage of `t10` (0.33-0.53 against 0.71-6.32 effort units for `t1`) does not appear in the RL runs [M3-01].
- **[Hypothesis]** (H1, [PI-06]; test not run) In RL the cusp is estimation-limited, not capacity-limited, for tanh actors; consistent observations are `t10`'s non-transfer, similar F_d for `t1` and `t10` at s = 1 and 16 (post hoc, below), small gains from tie weighting and budget, and a plateau over local updates 1800-2800. The test: vary the number of near-tie samples per update with all else fixed, for `t1` and `t10`, and see whether F_d falls.

<!-- TBL:t10 -->
<!-- /TBL:t10 -->

**The non-smoothing rounding width F_d (post hoc, derived).** F_d = sqrt(w_eff² - (2σ_2(0)/√π)²) in units of d, from arm means of `w_eff` and sigma_2(0), with the per-run median next to it [M3-08].
- **[Verified, descriptive]** From arm means F_d is 3.4-5.6 for `t1`, 3.3-6.0 for `t10` and 1.6-3.6 for `relu` (excluding the two `relu_bb` cells at q = 50 that contain the collapsed run) [TBL-fd]. The per-run median of the `relu` cell `relu_st_s1` at q = 60 is 0.00 because in 6 of its 10 runs w_eff is below 2σ_2(0)/√π (F_d is then set to 0); the arm-mean value is therefore not representative of that cell [TBL-fd].
- **[Hypothesis]** The tanh actors keep a rounding of about 3.5-6 units of d whatever their input scale and noise level (H1 above) [TBL-fd].

<!-- TBL:fd -->
<!-- /TBL:fd -->

**The noise landing under each actor.** Transmission ratio = (mean change of the gap)/(mean change of the smoothing part) between s = 16 and s = 1 of the same actor and starts; 1 means the whole smoothing reduction reaches the gap (MS-R2's prompt had written it with a minus sign that contradicted its own endpoints; the records and this report use the form without it) [M2-01].
- **[Verified, descriptive]** `t1`: -0.43 to +0.28 (as in MS-R2, ranging -1.04 to +0.54 over its eight cells); `t10`: 0.11, 0.60, 0.72, 0.69, with two intervals excluding 0; `relu`: 0.06, 1.53, 0.14, 0.33, none excluding 0 [TBL-transmission]. No arm reaches the smoothing floor at s = 16 (|peak error|/floor 3.1-7.4 for `relu` and `t10` outside the collapsed cell, 5.6-7.6 for `t1`) [M3-02].
- **[Insufficient evidence]** Whether the landing helps under a kink-capable actor: partial transmission under `t10` only, and the two transmission intervals that exclude 0 are two of the twelve in the table, both under `t10` [TBL-transmission].

<!-- TBL:transmission -->
<!-- /TBL:transmission -->

**The tie effort along the run.**
- **[Verified, descriptive]** Over local updates 1800-2800 the seed-mean ê2(0) is mostly a plateau: its largest move between two consecutive checks (25 updates) is 0.77-2.08 effort units depending on the arm, and its net change from 1800 to 2800 is between -0.59 and +1.83 (the Appendix B6 reading of "up to about 1.7" is the rounded `t10` q = 50 values +1.69 and +1.83) [TBL-trajectory][PI-06].
- **[Insufficient evidence]** Whether a longer budget would keep helping: no RL run of MS-R3 used more than 2800 updates [M3-02].

<!-- TBL:trajectory -->
<!-- /TBL:trajectory -->

### 7.6 The plan's stop rule and polishing at T=2, and R0 as a closed-form-free tie monitor

- **[Verified]** (MS-R1 pilot) At the pre-registered rho_2 = 0.05 the terminal-stage stop fired in 0 of 100 rule-arm runs (all ran 2000 training updates plus the 400-update landing); R_2 <= 0.05 held at 1 of 4000 checks at q = 50 and 91 of 4000 at q = 60, with at most 2 consecutive eligible checks; the calibration had expected about 1 in 60 [TBL-stoprule][M1-02].
- **[Verified, descriptive]** The localised branch (polishing) was reached at q = 60 (26-34 of 50 classifications per arm; polishing in 9-10 of 10 runs) and rarely at q = 50 (1-6 of 50) [M1-01][TBL-stoprule].
- **[Insufficient evidence]** Whether another stop metric or threshold would make the plan's stop rule work at T=2 (only rho_2 = 0.05 was run); the effect of polishing alone (no arm has the stratified sampler without the polishing branch); both are listed in section 9 [M1-02].
- **[Verified, descriptive]** R0 = r_2(0)/s_2, which needs no closed form, ranks |peak error| at the freeze with Spearman correlation 1.000 for `t1` and `t10` and 0.993 and 0.970 for `relu` (q = 50, 60) [TBL-r0]; in MS-R1 |peak|/R0 has median 1.66-1.68 (q = 50) and 1.46-1.47 (q = 60), close to the linearised factor 1.700 and 1.486 [M1-01][FIG-09].
- **[Verified, descriptive]** R0 does not see failures in the middle stratum: the three dead-region runs that fail G-A have R0 of 0.017-0.030 and gaps of 2.0-3.5 effort units [M3-02]. In MS-R3 R0 was reported as a monitor, not a rule [M3-04].

<!-- TBL:stoprule -->
<!-- /TBL:stoprule -->

<!-- TBL:r0 -->
<!-- /TBL:r0 -->

![MS-R1 calibration: R and Delta against the peak error, and the residual at d = 0 (R0)](figures/FIG-09_cal_fig1_scatter_R_Delta_vs_peak.png)

*FIG-09. MS-R1 calibration (`01_calibration.md`): the v2.0 weight exports, development seeds 10501-10510, both q [M1-05]. R (maximum over the non-tail region) ranks the peak error poorly, the residual at d = 0 tracks it [M1-02].*

### 7.7 Global accuracy and stage 1 across arms

- **[Verified]** (gate) The tail mean is below the 0.02 limit in every MS run (largest 0.0125 in MS-R1, 0.0134 in MS-R2 and MS-R3); all MS-R1 and MS-R2 runs pass the stage-1 gates; in MS-R3 every `t1` and `t10` run passes every gate and the only gate failures of the pilot are the five `relu` runs of section 7.5 [TBL-gates][M3-03].
- **[Verified, descriptive]** The stratified starts leave the tail mean slightly higher than the comparators in MS-R1 (+0.0005 to +0.0010 at q = 50), and the noise landing raised it by at most +0.0007 [M1-02][M2-01]. RMSE_pos/e2*(0) is lower under `t10` in six of eight cells and under `relu` in four; `relu` and `t10` have lower tail mean than `t1` in all eight cells each [M3-02].

<!-- TBL:gates -->
<!-- /TBL:gates -->

<!-- TBL:arms_all -->
<!-- /TBL:arms_all -->

![F1: |peak error| per run, every arm of MS-R1..MS-R3 and the v2.0 confirmation](figures/FIG-12_F1_abs_peak_per_run.png)

*FIG-12 (F1, drawn by `report_scripts/figures.py`). |peak error| at d = 0 per run (dots), with the arm mean (black bar) and the 0.05 line [M3-08]. Grey band: v2.0 confirmation on FRESH seeds 30501-30520 (n = 20 per q) [T2R:R2B-18]. Coloured: DEVELOPMENT seeds 10501-10510 (n = 10 per arm and q): blue MS-R1 (`parents_A` = v2.0 on the development seeds, budget control, rule and sampler arms; budgets 1600 or 2400 updates), orange MS-R2 (`NL_*`, 2800 updates), green MS-R3 (`relu_*`, `t10_*`; the `t1` arms equal the `NL_*` arms and are not drawn twice) [M1-08][M2-08][M3-08]. Triangles: runs clipped at 0.20 (the collapsed `relu` run at q = 50) [M3-03].*

![F2: the d = 0 gap split into the smoothing part and the remainder](figures/FIG-13_F2_gap_decomposition.png)

*FIG-13 (F2, drawn by `report_scripts/figures.py`). Mean smoothing part (blue) plus mean remainder (orange) per arm, effort units at d = 0, development seeds 10501-10510, n = 10 per arm and q; the black bar is the median gap. `rehearsal_v2_0` is v2.0; MS-R1 arms 1600/2400 updates, MS-R2/MS-R3 arms 2800 updates; bars above 6 effort units are clipped and labelled with their total [M1-08][M2-08][M3-08].*

![MS-R3 learned tie profiles](figures/FIG-01_tie_profile_runs.png)

*FIG-01. The learned policy near the tie, every run and the seed median, 12 MS-R3 arms, development seeds, terminal freeze [M3-03].*

![MS-R3 decomposition along the run, bin-balanced](figures/FIG-02_trajectory_decomposition_bb.png)

*FIG-02. MS-R3 decomposition along local updates 1800-2800, bin-balanced starts, development seeds [M3-03].*

![MS-R3 decomposition along the run, stratified](figures/FIG-03_trajectory_decomposition_st.png)

*FIG-03. The same, stratified starts [M3-03].*

![MS-R3 paired differences against t1](figures/FIG-04_paired_abs_peak_vs_t1.png)

*FIG-04. MS-R3 paired |peak error| differences of `relu` and `t10` against `t1` (primary criterion, part (a)), development seeds, ten pairs per q [M3-09].*

![MS-R3 noise landing](figures/FIG-05_paired_noise_landing.png)

*FIG-05. MS-R3 noise-landing paired differences (s = 16 against s = 1) under each actor, development seeds [M3-03].*

![MS-R2 decomposition along the run](figures/FIG-07_trajectory_decomposition.png)

*FIG-07. MS-R2 decomposition along the run, six arms, development seeds [M2-03].*

![MS-R2 remainder change against smoothing change](figures/FIG-08_scatter_remainder_vs_smoothing_change.png)

*FIG-08. MS-R2: per-run change of the remainder against the change of the smoothing part under the noise landing, development seeds [M2-03].*

![End-of-Phase-A profile, seed 30510](figures/FIG-10_FG-13_fig3_endA_profile.png)

*FIG-10. 100526 pack figure FG-13: the end-of-Phase-A profile of the failed v2.0 confirmation run (q = 50, seed 30510) against the pack [T2R:RR-11].*

![Stage-2 peak trajectory, q = 50](figures/FIG-11_FG-12_fig1_peak_trajectory_q50.png)

*FIG-11. 100526 pack figure FG-12: the stage-2 peak trajectory of seed 30510 against the other 19 q = 50 confirmation seeds [T2R:RR-11].*

## 8. What the evidence does not show

All items are **[Insufficient evidence]** unless another label is given.

1. **`relu`'s failure rate.** Two (q, seed) cases among 40 runs at q = 50 and none among 40 at q = 60; nor whether the proposed fixes remove the failures [TBL-relufail][M3-02].
2. **Fresh-seed performance of any MS configuration.** None was confirmed; every MS number is from the development seeds 10501-10510 [M3-01].
3. **The mechanism of the remainder, and of `t10`'s non-transfer.** Several readings were tested and two were withdrawn or not sufficient (section 9); none is established [M3-02][PI-04].
4. **Whether the noise landing helps under a kink-capable actor.** Partial transmission under `t10` only [TBL-transmission].
5. **Small effects.** Effects smaller than the intervals (half-widths 0.006-0.16 in MS-R3's primary rows) cannot be seen; an interval that contains 0 with ten seeds is not "no effect" [M3-02].
6. **Other stop thresholds.** Whether another stop metric or threshold would make the plan's stop rule work at T=2: only rho_2 = 0.05 was run, and R0 was studied as a monitor only [M1-02][M2-01].
7. **Polishing alone.** The effect of polishing separately from the sampler (MS-R1) [M1-02].
8. **Longer budgets.** RL budgets beyond 2800 terminal updates [M3-02].
9. **T=3.** Anything about it: not evaluated; the T=3 smoke tests of MS-R1 and MS-R3 show only that the pipeline runs [M3-02].
10. **The payoff cost of the deficit as a mechanism.** The small eta_2 is measured; why it is small is a [Hypothesis] (section 7.3) [TBL-eta].
11. **The reason for the sandbox discrepancy.** The PI's three-seed sandbox gave a `t1` median tip deficit of 1.49 effort units at q = 60 (bin-balanced) [PI-04][M3-33]. The repository screen with ten seeds gave 6.32, six of ten seeds at 6.18-6.74 [TBL-premise]. The premise check is unaffected; the cause of the difference was not investigated.

## 9. Unresolved issues

<!-- TBL:issues -->
<!-- /TBL:issues -->

## 10. Decision: close now or continue

<!-- TBL:compare -->
<!-- /TBL:compare -->

### Path A: close now

**Conclusions the current results support** (labels as in section 7).
- **[Verified]** v2.0 passes its gates and its fresh-seed confirmation (19/20 and 20/20) [T2R:CF-02]; the one failed run is a recorded G-A failure through eta_2 [T2R:CF-13].
- **[Verified, descriptive]** The tip deficit is characterised: its size (mean |peak error| 0.0630 and 0.0678 on the fresh seeds), its sign (negative in 40 of 40 fresh runs; non-negative only in eight runs of the MS and earlier rounds, at most +0.0009 outside `relu`), the exact smoothing part (ratio 0.99917-0.99971 to the formula) and its small effect on eta_2 (at most 0.0058 on the fresh seeds, at most 0.00226 in MS-R3 `t1` and `t10`) [T2R:R2B-18][TBL-signs][TBL-formula][TBL-eta].
- **[Verified]** (pre-registered criteria) No tested intervention is admissible: none of the six tip rounds met its criterion with part (b) holding, so there is no v2.1 and MS-R1..R3 produce no protocol change [TBL-interventions].
- **[Verified, descriptive]** `relu` is an ablation result: the typical run has the smaller tie deficit and the tail mean is lower, at the price of five failed runs [TBL-relutyp][TBL-relufail].

**What the results do not support.** Any statement about fresh-seed performance of an MS configuration; any estimate of the failure rate of `relu`; any claim that the tip deficit is removable by the levers tried; any statement about T=3 [M3-01][TBL-relufail].

**What closing consists of.** No new runs. v2.0 (`protocols/v2_T2_locked_v2_0.json`, tags `t2-v2-lock-v2.0`, `t2-v2-confirmation-v2.0`) remains the T=2 solver [T2R:PL-01]. The MS runner and MS-R1..R3 results are kept as recorded and produce no protocol change. What would be written up: the solver and its confirmation [T2R:CF-02]; the characterised tip deficit (section 7.3); the stop-rule and polishing study (section 7.6); the `relu` and `t10` results as ablations (section 7.5); the list of untested levers (section 9).

**Limitations a reader of any write-up would have to be told.** (1) The training return uses the game's shock distribution (conditional-expectation reward and an expected-continuation table), unlike the sampled-reward invariant of the original runners, so the claims are about this estimator (section 2) [BG-01][BG-02]. (2) The peak error is reported, not gated: 5 of 20 and 4 of 20 fresh runs are within 0.05 [T2R:R2B-18]. (3) Every MS number is from ten development seeds [M3-01]. (4) Nothing is evaluated at T=3 [M3-02].

### Path B: continue improving accuracy

**The metric to improve.** Primary: the stage-2 |peak error| at d = 0 (final tier, terminal freeze), and the share of runs within 0.05 [T2R:PL-02]. Guard rails: RMSE_pos/e2*(0) (limit 0.05), tail mean/e2*(0) (limit 0.02), eta_2/DW (limit 0.005), the gate pass rate (all gates, both stages), and the stage-1 error (limit 0.05) [T2R:PL-02].

**Reference points for choosing a target** (not a recommendation of a value):

<!-- TBL:refpoints -->
<!-- /TBL:refpoints -->

- v2.0 on the fresh seeds, and the best development-seed arms, are the first two reference points; the third is the smoothing floor σ_2(0)/(√π q), the error that a policy with that noise would have if it played the noisy game's equilibrium exactly. It is not a strict bound: single runs can overshoot it (a negative remainder; non-negative signed errors exist in eight runs) [TBL-nonneg]. No arm reaches the floor at s = 16: |peak error|/floor is 3.1-7.4 for `relu` and `t10` outside the collapsed cell and 5.6-7.6 for `t1` [M3-02]. The fourth reference point is the supervised-fit deficits, an offline reference for what an actor class can represent with exact targets and no RL noise [TBL-premise].

**Candidate measures** (workload is read from the launch records and budget tables of comparable waves; person-time cannot be estimated; nothing in this table invents a gain, a cost or a probability):

<!-- TBL:measures -->
<!-- /TBL:measures -->

<!-- TBL:workload -->
<!-- /TBL:workload -->

The workload table gives, per wave, the number of runs, the workers, the per-run wall time (minimum, median, maximum) and its sum; the elapsed time of a wave is not recorded in the launch records, so the sum divided by the workers is shown as a derived lower bound [TBL-workload]. A protocol change needs a lock, a re-rehearsal and a fresh-seed confirmation; the precedent is the v2.0 round (20 re-rehearsal runs on the development seeds and 40 confirmation runs) [T2R:RR-03].

### Recommendations

**PI-side recommendation (input; the decision is the coworker's).** [PI-06]
- Path A: keep v2.0 as the T=2 solver; MS-R1..R3 produce no protocol change; report the tip deficit as a characterised limitation (its size and sign on the fresh seeds, the exact smoothing part, the unexplained remainder, its small effect on eta_2, the `relu` result as an ablation).
- Reasons given: the tanh-side levers tried so far met no criterion; the remaining effects are about one effort unit or less, the size of the seed spread (seed SD of ê2(0) within an arm 0.45-1.56 effort units [M2-02]); the PI's two mechanism readings did not hold in RL, so a third, untested reading is a weak basis for a larger programme [PI-06].
- If the coworker chooses Path B, the PI would start with `relu` robustness only: a leaky ReLU and a mean map without a hard clamp; stratified starts and the noise landing; 20 seeds per q (ten more development seeds) to estimate the failure rate [PI-06]; a pre-registered target on |peak error| and a zero-failure requirement on the gates. Adoption would still need a lock and a fresh-seed confirmation, and the actor change would carry into T=3 [PI-06].
- Not recommended by the PI now: more tanh-side knobs; a mechanism round [PI-06].

**Observation of the assistant that wrote this folder (mine; it decides nothing).** The decision depends on a requirement that the records do not contain: whether a claim that needs the peak (rather than the gates) must hold on most runs. The gates that v2.0 meets (39 of 40 fresh runs pass G-A; eta_2/DW at most 0.0058) do not include the peak, which is reported but not gated, and only 5 and 4 of 20 fresh runs are within 0.05 [T2R:R2B-18][TBL-accuracy-b]. The development-seed arm that already puts 10 of 10 runs within 0.05 at both q (`relu_st_s16`) is also the one with failures in part (b), and the arms without failures (`t10_st_s16`, `t1_st_s16`) reach 8 and 9, or 8 and 7, of 10 [TBL-accuracy-a]. So a target stated as a share of runs within 0.05 [TBL-accuracy-a] has to say whether it accepts a failure risk that the current runs cannot measure.

**What the coworker is asked to return.**
- (i) Close or continue.
- (ii) If continue: the measure to start with, and the target (metric, value, seed set).

The question, verbatim:

> 基于现有结果，我们应该在此阶段收口，还是继续提高精度？如果继续，优先改进什么、目标是什么？
>
> (Based on the current results, should we close T=2 at this stage, or continue improving its accuracy? If we continue, what should be improved first, and what is the target?)

No experiment of any kind starts before the reply.

## 11. Evidence index and reproduction

**Manifest and integrity.** `evidence/manifest.csv` lists every item (id, title, round, source branch and commit, source path, SHA-256, size, status, copy path, conditions, sections that cite it); the copies sit under `evidence/<original path>`; `evidence/SHA256SUMS` and `figures/SHA256SUMS` are in `sha256sum -c` format. Items of the 100526 pack are cited in place (status `cited in t2_refine_100526`). Two extensions to the prefixes of the prompt: `TBL-` rows for the tables produced by `tables.py` (status `generated (table text)`) and `BG-` for three background files (the instruction file is stored as `evidence/dot-claude/CLAUDE.md.txt` so that no tool reads it as an instruction file).

**Figures.** FIG-01 to FIG-11 are copies (the MS-R3 tie profiles, decompositions along the run, paired differences against `t1`, the noise-landing figure; MS-R2's decomposition and scatter; the MS-R1 calibration figure; the 100526 end-of-Phase-A profile and peak trajectory); FIG-12 (F1) and FIG-13 (F2) are drawn by `report_scripts/figures.py` only from evidence copies of this pack and of the 100526 pack.

**Scripts** (`report_scripts/README.md`). `build_pack.py` builds `evidence/` and `figures/` from git objects at recorded commits and writes the manifest and the sums; `tables.py` produces every table of this report from the evidence copies; `figures.py` draws F1 and F2; `check_numbers.py` and `check_links.py` check the report. Nothing is run on weights, no verifier is evaluated, no launcher and no round analysis tool is re-run.

**Commands.**
```
python reports/t2_status_100826/report_scripts/build_pack.py            # (re)build evidence/ and figures/
python reports/t2_status_100826/report_scripts/build_pack.py --check    # rebuild in a temp dir, compare byte for byte
python reports/t2_status_100826/report_scripts/tables.py --verify reports/t2_status_100826/report.md
python reports/t2_status_100826/report_scripts/check_numbers.py
python reports/t2_status_100826/report_scripts/check_links.py [--ref origin/t2-status-pack]
(cd reports/t2_status_100826/evidence && sha256sum -c SHA256SUMS)
```

**Differences between the inputs of the prompt and the records** (the full ledger is `pi_record/01_factcheck.md`): the statement that the learned effort is below the closed form in every run of every round does not hold for eight runs (section 7.3); the smoothing-floor range "2.3-2.7% at s = 1" of Appendix B4 is the `t1` range, while with the `t10` arms the range is 2.2-2.7% (the reference-point table of section 10); the relation of each other number to its record is in the ledger [PI-06][TBL-refpoints].
