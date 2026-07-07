# Phase-0 Response to Revision Plan

Prepared 2026-07-02, vector2. 单份交付物，未提交 — 供 owner review。
所有数字的权威来源在文末"Provenance index"里逐条列出；本文不重复推导，只引用。

---

## 0. 背景：原始 revision plan 提出了什么

原始 revision plan（来自 `.docx`：`06192026TEL-PPO_one_stage_revision_plan_1_.docx`，历史
Claude.ai 会话，非本仓库文件）提出五个**串联、非互斥**的组件，目标是压低 3P/dc/da 在
q∈{35,55} 上相对 analytic e* 的 effort 误差（当时的 r5 headline：3P 8.03%/6.63%，
da standard 5.25%/3.26%，dc 2.88%/4.69% — 见 `SESSION_STATE.md` "Adopted canonical state"）：

1. **MC reward averaging（K）**：训练奖励从单次采样改为 K 次采样均值，降方差。
2. **Mode-concentration head + conditional ramp**：把策略参数化从 mean-conc 换成
   mode-conc，配合"exploitability 触发式"的浓度斜坡，让确定性 effort（mode）更早锁定
   在高浓度、低方差的位置。
3. **Mode extraction**：报告用 Beta mode 而非 mean。
4. **Post-hoc MC best-response polishing**：在已有 r5 数据上，用 sampled-only 的 MC-BR
   对每个 cell 的最终 effort 做零 GPU 的精修。
5. **Exploitability + FOC-based stopping**：用 sampled exploitability 而非 closed-form
   FOC 作为收敛判据。

2026-06-23 的会话做了风险评估，拦下了"直接全跑五个组件"，转而先执行 **Phase 0**：只用
已有的 r5 3P/dc/da 数据，跑组件 4（post-hoc MC-BR polish）+ 组件 5（exploitability/FOC
验收），零 GPU，作为其余组件是否值得投入 GPU 的前置判据。

---

## 1. Phase 0 做了什么，以及为什么脚本/文档会丢失

Phase 0 的三个脚本（`utils/mc_br_polish.py`、`tools/phase0_verify.py`、
`tools/phase0_decomposition.py`）在原服务器上于 `8df54d0` 提交（2026-06-24），2P
do-no-harm 的 Set-1 收窄在 `7e593b4`（2026-06-25）。这两个 commit **已推送到远端**
`origin/fix/audit-remediation`，但原服务器在完整同步/合并进 main 之前损坏。vector2 是
一次全新 clone（`git reflog` 只有两条记录：clone + checkout 分支，无本地残留文件），
脚本靠 `git checkout fix/audit-remediation` 完整恢复，**没有按规格重写**。

两项 uncommitted 的工作树改动确认永久丢失，本文档和 `SESSION_STATE.md` 的整合就是补上
这两项缺口：
- `SESSION_STATE.md` 的 Phase-0 整合段（verdict + provenance + superseded 映射 +
  两处 self-correction）——已于本轮补上，见 `SESSION_STATE.md` "PHASE 0" 系列章节。
- 本文档 `docs/phase0_response_to_revision_plan.md` ——原稿据历史会话记录曾在 6/25
  起草过，owner 已确认找不到原稿，属永久丢失，本文档是**重新起草**，不是"找回"。

`tools/phase0_mc_br_polish.py`（组件 4 最早的一次性脚本）已被 decomposition/verify 取代，
在任何 git 历史中都查无踪迹，不需要、也没有恢复。

---

## 2. Phase 0 三个核心实证发现

### Finding A — mode extraction（组件 3）对 3P 完全 inert

收敛时 3P Beta 的 concentration α+β ≈ 25,000–33,000（near-spike，由 3P/dc 的 r5 用了
`--theory-align-v2 --override-conc-ramp-warmup 200` 造成），因此 mean ≈ mode 精确到
小数点后 3 位；10 runs 上 mean-extraction 与 mode-extraction 的 mean|err| 几乎相同
（1.531 vs 1.532）。全轨迹 max|mean−mode| 按 run 为 0.19–0.30 effort units（出现在
训练早期 conc≈210–270 段；3P q35 五个 seed 为 0.19–0.25，全 10 runs 最大 0.30 在
q55 seed 44），对比 ~2.0 的 undershoot gap，mode 修不了误差。**组件 3 因此降级为纯 diagnostic 列，不作为主报告口径**（这也是
本次 Component-2 retrain 仍然按 CLAUDE.md 不变量报告 Beta mean、而非组件 2 原规格里的
mode 的原因之一）。

### Finding B — undershoot 是真实的 best-response gap，e* 是对的目标

复用已有 sampled MC-FD 梯度（CRN）核对：3P q35 在 e_conv=22.99 处 sampled FOC=+0.00398
（上坡），在 e*=25 处 sampled FOC≈0（−0.00023）；q55 同理（e_conv=15.31 处 +0.00150，
e*=15.91 处 −0.00011）。即 **analytic e* ≈ sampled game 的均衡**，PPO policy 冻结在
低约 2 units 处——这正是组件 4（MC-BR polish）能起作用、且**必须零 GPU**的条件。

### Attribution finding — MC-BR polish 是 global solver，直接决定了 Claim A/B 这个 framing 问题

测试 polish 是否依赖 PPO 的 warm start：**不依赖**。从任意初值都收敛到 e*
（3P q35：inits 10/25/40/60 → 24.75/24.81/25.03/24.78；dc q35：inits (15,15)/(50,50)/(38,28)
→ 全部 → [≈38.1, 27.66]；da q35：inits 20/46/70 → 全部 → ≈46.5）。

**这直接决定了论文不能只说"polish 把误差压到 <1%"——那句话对 polish 这个 global solver
而言是同义反复，不是对 PPO 学习能力的陈述。** 这就是第 5 节 Claim A/B 决策的由来。

---

## 3. 定义性结果：6-cell 表 + 2P do-no-harm（2026-07-01 重跑，全部已核实）

方法：`tools/phase0_verify.py`，debiased（quadratic-vertex）BR polish + 三条
acceptance leg（(a) fresh-seed exploitability < τ_E=0.005；(b) fresh-seed/different-step
(δ=0.75)/larger-M(1e6) 的 interior FOC < τ_g=0.001；(c) polish 收敛：per-seed 轨迹内
Polyak-window drift < τ_e=0.1 **或**窗口 SE（最后 200 轮，std/√200）< τ_e=0.1）。
注意 (c) 是 OR 判据、两个量都是轨迹内量（不是 cross-seed）：实测 4/6 cell 的 drift
超过 τ_e（0.105–0.227），(c) 实际经 SE 分支（0.016–0.068）通过——SE 分支只检验
polish 轨迹的稳定性，约束力弱，因此 (c) 应视为 polish 的收敛 sanity check；承重的
验收是独立 fresh-seed 的 (a)+(b)。Non-circularity 约束：leg (b) 禁止用 FOC-root-find 做 polish，
否则 (b) 会因 by-construction 而 auto-pass（`SESSION_STATE.md` §B 逐字记录了这条设计
理由）。输入是已有 r5_sampled 30 runs（3P/dc/da × q∈{35,55} × seeds 42–46，da 用
`_std` tag，经 effort 交叉验证排除 `_v2`）。输出落盘于
`results/phase0_verify_20260701_1941.log`（unbuffered，exit 0）。

**OVERALL: 6 main cells all PASS = True，max polished error-vs-e* = 0.319（3P q35）**

| Cell   | e*            | polished (mean) | error                          | verdict  | (a) EXP / (b) \|FOC\| / (c) drift, SE |
| ------ | ------------- | ---------------- | ------------------------------- | -------- | -------------------------------------- |
| 3P q35 | 25.00         | 24.68            | −0.32 (1.28%)                   | **PASS** | 0.0008 / 0.00044 / 0.159, 0.028        |
| 3P q55 | 15.91         | 15.82            | −0.09 (0.57%)                   | **PASS** | 0.0007 / 0.00047 / 0.105, 0.016        |
| dc q35 | 38.03 / 27.66 | 38.04 / 27.66    | +0.01 / +0.01 (~0.02%)          | **PASS** | 0.0006 / 0.00027 / 0.078, 0.031        |
| dc q55 | 26.54 / 19.30 | 26.56 / 19.30    | +0.02 (0.09%) / −0.00 (0.02%)   | **PASS** | 0.0002 / 0.00025 / 0.052, 0.018        |
| da q35 | 46.43 / 46.43 | 46.45 / 46.45    | +0.02 / +0.02 (~0.05%)          | **PASS** | 0.0010 / 0.00036 / 0.227, 0.068        |
| da q55 | 30.37 / 30.37 | 30.36 / 30.37    | −0.01 / +0.00 (~0.02%)          | **PASS** | 0.0005 / 0.00024 / 0.127, 0.029        |

3P q35 的 1.28% 上界是 sampled-vs-analytic gap（sampled 均衡本就在 analytic 25 下方
~0.3），不是 learning failure。

**2P do-no-harm**（sanity check，不计入 6-cell 表；`7e593b4` 后限 Set 1，排除
`wh8_wl4`）：q35 raw-mean err 1.875 → polished err 0.508（no-harm=True）；q55
raw-mean err 0.793 → polished err 0.168（no-harm=True）。polish 在两个 cell 均改善，
无 regression。

dc 按 revision plan §7 判 payoff-loss + exploitability，而非单看 effort error
（weak-identification；mixed-sign raw error 是 weak-id 指纹，不是 bug）。

---

## 4. 两处自我修正（供未来 session 参考，禁止复活被证伪的结论）

**修正 #1 —— "argmax-BR 有 ~0.17 downward bias"，已证伪。** debias smoke 几乎没动
均值（24.821→24.811，预期~0.13，实测~0.01）。vertex debias 的真实作用是**降方差**
（e=25 处 argmax 24.89±0.37 → vertex 24.75±0.27），不是去偏。polish-vs-rootfind 的
gap 是 **flat-plateau 宽度**，不是 argmax bias。

**修正 #2 —— "confirmed FOC=0 point = 24.94 ± 0.04"，过度精确，已修正。** ±0.04
只是 3-seed spread；实测 |FOC|<~0.0003 横跨 [24.7, 25.0]，与 FOC estimator 自身噪声
地板同量级。3P q35 的 sampled 均衡只能钉到一个**平台区间**，不是一个 sharp 点——因此
polished 24.68 过 leg (b) 与"24.94 是均衡点"这个旧说法并不矛盾，验收判据是
|FOC|<τ_g，不是逼近某个过度精确的点估计。

（附带一处中途修正："polishing 帮不了 dc" 因 dc 干净 polish 到 e*（err 0.02–0.05%）
而被推翻——MC-BR 是直接求解器，能定位 flat optimum，即便学习 agent 不能；这不影响
dc 仍按 weak-identification 口径报告。）

---

## 5. 组件最终处置

| Comp | 处置 | 依据 |
| --- | --- | --- |
| (1) MC reward avg (K) | **DROPPED，K=1** | 唯一触碰 training reward 者，不需要；K>1 会软化"single realized outcome"的论文措辞。 |
| (2) mode-conc + conditional ramp | **测试后 DROPPED（见第 6 节 Component-2 结果）** | 唯一能区分 Claim A/B 的实验；已授权执行，结果不支持 Claim A。 |
| (3) mode extraction | **INERT，保留为 diagnostic 列** | Finding A：3P mean≈mode（Δ≈−0.00）；dc/da 只存 effort，mode N/A。 |
| (4) MC-BR polish | **LOAD-BEARING** | 零 GPU、sampled-only；第 3 节全部结果由它得出。 |
| (5) exploitability + FOC stop | **finalized** | exploit-streak stop 已 production；新增 frozen polished profile 上 fresh-seed 二次 exploit。 |

---

## 6. Component-2 mode-conc retrain — Claim A 的直接实验检验（2026-07-02 执行）

### 6.1 为什么必须做这个实验

第 2 节的 attribution finding 已经说明：MC-BR polish 是 global solver，第 3 节的 6/6
PASS **不能**归因于"PPO 学到了均衡"。这把论文的中心论点变成一个必须回答的二选一：

- **Claim A**："PPO self-play 学习过程本身收敛到均衡 effort" —— 若成立，需要 PPO
  自己（不靠 polish）的 raw 输出落在 e* 附近。
- **Claim B**："PPO 到达 basin；sampled MC-BR + exploitability 证明均衡" —— polish
  和 exploitability 检验承担"证明"的角色，raw/polished 保持独立两列。

Owner 于 2026-07-02 授权跑 Component-2 retrain 来直接检验 Claim A，而不是凭 6/6 PASS
默认支持某一方。

### 6.2 实验设计

新增 `ActorCriticModeConc` 头（`agents/ppo_three_players.py`）：
`s=sigmoid(mode_head)`、`κ=clamp(softplus(conc_head)·scale+κ_min, max=κ_max)`、
`α=1+s·κ`、`β=1+(1−s)·κ`，`+1` 底线保证 α,β≥1（interior mode）。配合
`run/run_three_players.py` 里新的 exploitability-触发式斜坡状态机（`--mode-conc-ramp`
及 4 个配套 flag：`--kappa-schedule`、`--ramp-trigger-exp`、`--ramp-trigger-patience`、
`--kappa-stage-hold`）：

- **Explore**：κ 钉宽 [1, 20]，entropy-driven，正常训练。
- **Trigger**：raw exploitability `EXP_raw < 0.05` 连续 3 次 in-loop eval。
- **Ramp**：κ 走 [20, 50, 100, 200]，每档 20 updates。
- **Done**：κ=200 后交给正常 exploit-stop（eps_eq=0.03, patience 5）——**在 ramp 完成
  之前，正常 stop 被主动 gate 掉**，避免在宽 κ、undershoot 的探索阶段过早停止。

报告口径：按 CLAUDE.md 不变量，headline 报 **Beta mean**（不是组件 2 原规格里的
mode）；mode 作为诊断列同时记录，高 κ 下两者应接近。

规模：3P q35，seeds 42–46，spec 参数（`--ramp-trigger-exp 0.05
--ramp-trigger-patience 3 --kappa-stage-hold 20`），full episodes（6,144,000），5 GPU
并行（GPU 0–4），tag `c2_mode_conc`。

代码在 CPU smoke（loose params）+ GPU smoke 两轮验证过状态机本身正确（explore→
ramping→done 按 stage_hold 步进，α+β−2 精确 pin 到 κ，JSON schema 无误），才进入
5-seed 全跑，过程中还发现并修复了一个环境问题：vector2 的 GPU 是 V100（compute
capability 7.0 / sm_70），`uv pip install` 默认拉到的 `torch==2.12.1+cu130` 不含
sm_70 kernel，所有 GPU 算子报 `no kernel image available`——重装
`torch==2.5.1+cu121`（venv 内，无 sudo）后验证 sm_70 在 arch list 里且真实 matmul
可跑通，`requirements.lock` 已重新 pin。

### 6.3 结果——Claim A 不被支持

**raw PPO effort（无 polish）vs e*=25，5 seeds：**

| seed | raw effort | err | 备注 |
| --- | --- | --- | --- |
| 42 | 22.698 | −9.21% | ramp 完整跑完（trigger@879 → done@959） |
| 43 | 22.989 | −8.04% | ramp 完整跑完（trigger@1058 → done@1138），几乎与 r5 raw mean 完全一致 |
| 44 | 25.663 | +2.65% | ramp 完整跑完（trigger@877 → done@957） |
| 45 | 20.865 | −16.54% | ramp 完整跑完（trigger@876 → done@956） |
| 46 | 21.417 | −14.33% | **ramp 全程未触发**，跑满 1500-update 预算，全程停在 explore（κ=20） |

**vs r5 baseline（旧 `theory_align_v2` ramp，同样是 raw PPO effort，5 seeds
[23.421, 22.756, 23.125, 22.745, 22.918]）：**

| | mean | std | mean\|err\| | range |
| --- | --- | --- | --- | --- |
| r5（旧 ramp） | 22.993 | **0.255**（紧） | 8.03% | [22.75, 23.42] |
| Component-2 | 22.726 | **1.666**（6.5× 大） | 10.16% | [20.86, 25.66] |

三条独立证据方向一致：
1. **mean 没有更靠近 e***，反而略差（10.16% vs 8.03% 的 mean absolute error）。
2. **方差暴涨 6.5×**——r5 的紧密聚集本身就是"真实、可复现的学习上限"的证据；
   Component-2 的散布（20.86–25.66）说明这个 retrain **不是可靠收敛，是碰运气散布**。
3. **1/5 seed（46）在整个训练预算内从未触发该机制**，说明连"激活"这一步都不可靠，
   遑论收敛。

### 6.4 机制诊断（2026-07-02 依据 mode/mean 轨迹修订）

`EXP_raw<0.05` 的触发条件本身没有绑定"effort 接近 e*"——它绑定的是"局部偏离不划算"。
在 payoff 平台很平的区域（Finding B 已证实 3P q35 附近 FOC 接近 0 横跨相当宽的区间），
这个条件在远离 e* 的位置就被满足了：四个触发的 seed 触发时 mode ≈ 17.9–18.2、
mean ≈ 20.8–21.1，离 e*=25 还差约 7 个 effort units，且触发位置跨 seed 几乎相同。

触发之后发生的事（由 convergence JSON 的 mode/mean 轨迹核实）**不是冻结**：在约
80 updates 的 ramp 窗口内，mode 继续朝 e* 方向移动了 +2.7 到 +7.2 units（seed 42:
17.9→22.4；43: 18.2→22.7；44: 18.2→25.4，越过 e*；45: 17.9→20.6）。真实失败模式是
**过早触发 + 定长斜坡窗口走不完剩余距离**：κ 到顶后正常 exploit-stop 很快生效，最终
effort ≈ 触发位置 + 各 seed 在窗口内恰好走了多远——6.5× 的跨 seed 方差正是在这段
窗口内产生的（各 seed 走过的距离不同），而不是"在随机时间点冻结"造成的。

因此这与 Finding A 里对旧 `theory_align_v2` ramp 的批评（"lock 只是冻结了 undershoot，
没有制造它"——那里 effort 在 lock 前就已降到 ~23.5 附近并被钉住）**并不是同一个失败
模式**，共同点只在结果层面：两种机制都没有让 PPO 自身可靠收敛到 e*，而 Component-2
还额外引入了更大的跨 seed 方差。

（修正记录：本节初稿曾断言"κ 斜坡冻结触发那一刻的均值、此后不再推向 e*"，并称其
与旧 ramp 是同一失败模式——该说法与轨迹数据矛盾（mode 触发后移动 +2.7–7.2 units，
方向朝 e*），已按上述修订，禁止复活；旧表述曾同步存在于 `SESSION_STATE.md` 与
`docs/tasks/component2-mode-conc-retrain/STATE.md`，已一并修正。）

### 6.5 处置

代码保留（不删除——这是一个有完整 provenance 的真实负结果，不是失败的尝试需要清理）。
默认关闭（`agents/ppo_three_players.py` 的 `mode_conc_ramp: bool = False`；
`run/run_three_players.py` 需要显式 `--mode-conc-ramp`），不影响任何既有 r5 /
`theory_align_v2` 路径的复现性。5 个 seed 的完整 convergence JSON：
`results/three_players/convergence/ppo_3p_q35.0_seed{42..46}_c2_mode_conc_convergence.json`。
过程记录见 `docs/tasks/component2-mode-conc-retrain/{CLAUDE.md,STATE.md}`。

---

## 7. §H — Claim A vs B：最终 framing 建议（待 owner 拍板）

**第 2 节的 attribution finding 已经说明 polish 不能支撑 Claim A；第 6 节的
Component-2 实验现在给出了直接、独立的负面证据：PPO 自身的学习过程（即便配上专门为此
设计的 mode-conc head + exploitability 触发式浓度斜坡）也没有可靠地收敛到 e* 附近——
均值没有改善，方差反而暴涨 6.5×，20% 的 seed 连触发都没发生。**

**建议：论文的中心论点采用 Claim B ——"PPO self-play 训练到达均衡的 basin；一个
sampled-only、zero-GPU 的 MC best-response refinement，配合两条独立的 exploitability/FOC
检验，证明该 basin 与理论均衡一致"。** 具体写法上：

- raw（PPO 自身输出）与 polished（MC-BR 精修后）保持**独立两列**，不能只报 polished
  数字，否则会被审稿人抓到"post-hoc polish ≠ proof of convergence"（决策 2 的
  审稿人预防条款，`SESSION_STATE.md` 已记录）。
- 3P/da/dc 的 raw undershoot（第 3 节表格里"raw"列，约 5–17% 视 cell 而定）应如实
  报告为"PPO 学习动态收敛到一个宽的、次优的 basin"，而不是掩盖或删除。
- Component-2 的负结果本身是论文可以引用的一条**方法论证据**：它排除了"只要给
  policy head 换个参数化 + 调整浓度机制就能让 PPO 自己学到 e*"这条路径，从而加强
  "需要一个独立于训练动态的 refinement + verification 步骤"这个设计选择的必要性。

**若 owner 仍想追 Claim A**：当前这版 Component-2 设计已经被数据证伪，需要一个
**根本不同**的 retrain 设计（例如把触发条件换成"effort 落在某个已知邻域"而非纯
exploitability、或大幅拉长/放缓斜坡、或改用不同的正则化去压低跨 seed 方差）——这超出
本轮授权范围，需要重新决策与授权。

---

## Provenance index

- Phase-0 脚本 / commit：`8df54d0`（三脚本）、`7e593b4`（2P do-no-harm Set-1 收窄），
  分支 `fix/audit-remediation`。
- 定义性 6-cell 表 + 2P do-no-harm 原始输出：`results/phase0_verify_20260701_1941.log`
  （unbuffered，exit 0）。
- Phase-0 整合（§A–§E + self-correction + open decisions）：`SESSION_STATE.md`
  "PHASE 0" 系列章节（本文第 3–5、7 节的数字与此逐格核对一致）。
- Phase-0 重建规格手册（原始叙述记录提炼版）：`Phase0_reconstruction.md`
  （§1 Finding A/B/attribution、§5 pre-registered gate、§9 Component-2 spec 存档）。
- Component-2 代码：`agents/ppo_three_players.py`（`ActorCriticModeConc`）、
  `run/run_three_players.py`（`--mode-conc-ramp` 状态机 + 4 flag）。
- Component-2 结果原始 JSON：
  `results/three_players/convergence/ppo_3p_q35.0_seed{42,43,44,45,46}_c2_mode_conc_convergence.json`。
- Component-2 过程记录：`docs/tasks/component2-mode-conc-retrain/{CLAUDE.md,STATE.md}`。
- r5 baseline（3P q35 raw，5 seeds）：
  `results/three_players/convergence/ppo_3p_q35.0_seed{42,43,44,45,46}_r5_sampled_convergence.json`。
- 环境修复（V100 sm_70 / torch cu130→cu121）：`requirements.lock`（重新 pin）、
  memory 条目 `vector2-gpu-torch-sm70`。
