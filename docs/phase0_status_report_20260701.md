# Phase-0 Status Report — 2026-07-01 (vector2 reconstruction)

Prepared 2026-07-02. Single deliverable, not committed — for owner review.

---

## Part 0 — 来源:历史 Claude.ai 会话,非本仓库文件

> 这份 revision 最初来自 .docx(06192026TEL-PPO_one_stage_revision_plan_1_.docx),提出
> 五个串联组件(非互斥可选)压低 3P/dc/da 在 q∈{35,55} 上的 effort 误差。6/23 的会话里做了
> 风险评估(拦下"直接全跑",指出 K-averaging 的 sampled-vs-closed-form 方法论风险),裁决:
> dc 按§7 报告(weak-identification);Component 2(mode-concentration + conditional ramp)
> 保留,定位 retrain-gated Phase-1 lever,κ schedule 20→50→100→200、EXP_raw<0.05 触发
> (避开 theory-align-v2 warmup=200 的过早方差压制);Component 3 降级为纯 diagnostic。
> 该会话以"执行 Phase 0:对现有 r5 3P/dc/da 做 post-hoc MC-BR-polish,零 GPU"收尾。
>
> 6/25 的后续会话:针对 argmax-BR 在 flat surface 上的偏差,设计 quadratic-vertex BR debias
> (zeroth-order,避免 leg (b) 因 FOC-root-find 而 auto-pass 的 circularity),6 cell 全部
> 重跑,达成 Findings A/B、Component 1/2 drop、两处 self-correction、commit 8df54d0。该会话
> 起草了 docs/phase0_response_to_revision_plan.md,§H 专门讨论 Claim A/B 这个 framing
> question——这份文档目前在仓库任何地方查无踪迹,是本轮丢失的最有价值材料。

(以上原样保留,未做仓库文件核对或改写。)

---

## Part 1 — vector2 恢复与验证过程(逐条溯源仓库文件)

### 1. Git 层面能恢复的 vs. 必丢的 uncommitted 工作

- **可恢复(已在远端)**:`utils/mc_br_polish.py` + `tools/phase0_verify.py` + `tools/phase0_decomposition.py`
  提交于 `8df54d0`("feat: add MC-BR polishing module and Phase-0 verification drivers",
  3 files changed, 544 insertions,作者 GSU-FrankJ,2026-06-24)。2P do-no-harm 的 Set-1
  收窄修复在 `7e593b4`("fix: restrict Phase-0 2P do-no-harm to Set 1 (exclude wh8_wl4)",
  1 file changed +5/-1,2026-06-25)。
  验证:`git log --oneline main..origin/fix/audit-remediation` 只返回这两个 commit
  (`git rev-list --count` = 2);`git log --oneline origin/fix/audit-remediation..fix/audit-remediation`
  为空 —— 本地分支与 origin 完全同步,**这两个 commit 已推送但尚未合并进 main**。
  `git reflog` 只有两条记录(`clone: from https://github.com/GSU-FrankJ/tournament_experiment.git`
  → `checkout: moving from main to fix/audit-remediation`),证实 vector2 上这是一次**全新
  clone**,不是从旧盘复制,脚本恢复完全靠这次 `git checkout`,不靠任何本地残留文件。

- **必丢、已确认丢失、需在 vector2 上重建的 uncommitted 工作**(`Phase0_reconstruction.md`
  §0 明确列出):
  - `SESSION_STATE.md` 的整合(verdict + provenance + superseded 映射 + 两处 self-correction)
  - `docs/phase0_response_to_revision_plan.md`(response 文档,§H 讨论 Claim A/B——**这份文档
    在当前仓库任何路径下都不存在**,`find . -iname "*phase0_response_to_revision_plan*"` 零命中)
  - 旧的一次性脚本 `tools/phase0_mc_br_polish.py`(已被 decomposition/verify 取代,`Phase0_reconstruction.md`
    §0 明确"不必恢复")

### 2. venv 而非 `--break-system-packages` 的原因

- `results/phase0_verify_20260701_1929.log`(193 字节,时间戳 19:29)内容是:
  ```
  Traceback (most recent call last):
    File "/home/fjiang4/tournament_experiment/tools/phase0_verify.py", line 16, in <module>
      import numpy as np
  ModuleNotFoundError: No module named 'numpy'
  ```
  即新 clone 的系统 Python 环境里没有 numpy。CLAUDE.md 明确规定"Never install packages
  without `--break-system-packages` flag"仅适用于系统 Python 直接安装;本次改走 `.venv`
  路线 —— 仓库根目录下确认存在 `/home/fjiang4/tournament_experiment/.venv`,`requirements.lock`
  (未跟踪,1215 字节)锁定了 vector2 上实际装的版本组合(`numpy==2.5.0`、
  `torch` 相关 cuda 包等)。`docs/tasks/component2-mode-conc-retrain/STATE.md` 第 21-24 行
  独立记录了同一台机器上后续发现的 GPU 兼容问题并作为一次单独修复:V100(sm_70)与
  `torch 2.12.1+cu130` 不兼容("no kernel image"),改装 `torch==2.5.1+cu121` 后验证
  sm_70 kernel 存在 + 真实 GPU matmul + ramp 行为一致,并"Re-pinned requirements.lock"。
  这与我在 `MEMORY.md` 里已有的 [vector2-gpu-torch-sm70](../../.claude/projects/-home-fjiang4-tournament-experiment/memory/vector2-gpu-torch-sm70.md)
  记忆条目一致,是同一件事在同一时间线的两处独立记录,不是新发现。

### 3. r5 JSON 存续 + `_std`/`_v2` 排歧依据

- `SESSION_STATE.md` §A(第 118 行):Phase-0 输入是 r5_sampled convergence JSON,
  3P/dc/da × q∈{35,55} × seeds 42–46 = 30 runs,"全部已入库、磁盘实文件"。
  da 用 `_std` tag,依据是 effort 交叉验证:q35 mean 43.993 ≈ 手工核对的 43.99、
  q55 mean 29.703 ≈ 29.70;`_v2` 偏离 ~0.94/0.57,故不采用。这与 `phase0_verify_20260701_1941.log`
  第 24-27 行的实际输出一致(da q35 raw 43.99、da q55 raw 29.70)。
- `Phase0_reconstruction.md` §8(第 220-222 行)补充了排歧的**根因**:3P/dc 的 r5 用了
  `--theory-align-v2 --override-conc-ramp-warmup 200`,这套 forced ramp 把
  `conc_min` 100→1000、`conc_scale` 100→10000、`conc_max` 100000 拉爆,导致收敛时
  α+β≈25,000–33,000(near-spike),而 da 存在 `_std` vs `_v2` 两个版本正是同一问题在
  different_ability 实验上的体现 —— `_std` 是标准配置,`_v2` 是这套 theory-align-v2 ramp
  的产物。

### 4. `tools/phase0_verify.py` 的两个真实故障点

- **故障点 1(有文件证据)**:19:29 的 numpy ModuleNotFoundError(见上一节),根因是
  系统 Python 环境缺依赖,而非脚本逻辑问题;修复路径是切到 `.venv` + `requirements.lock`。
  19:41 的第二次运行(`results/phase0_verify_20260701_1941.log`,3129 字节,exit 0)成功。
- **故障点 2(仅有部分文件证据,如实标注证据缺口)**:`SESSION_STATE.md` §A 把
  `phase0_verify_20260701_1941.log` 标注为"**unbuffered**,已落盘",这个措辞本身暗示
  在此之前存在过一次因**块缓冲**(stdout 被 tee 到文件但未 flush)导致输出文件为空/不完整
  的失败尝试。但是:仓库里现存的两个 log 文件(1929、1941)分别对应"numpy 报错立即退出"
  和"完整 6-cell 成功输出",**没有第三个空文件或截断文件留存来独立佐证块缓冲故障**;
  `tools/phase0_verify.py` 源码里也没有 `PYTHONUNBUFFERED`/`stdbuf`/`flush()` 的痕迹能证明
  这一具体修复动作。**结论:这一故障点只能作为 SESSION_STATE.md 的既有标注引用,无法在
  当前仓库文件里独立复核到"发生过""如何修复"的直接证据,不应当作已核实的事实呈现。**
- 关于"tmux vs 裸后台":`Phase0_reconstruction.md` §8(第 224 行)重申了 CLAUDE.md 的既有规则
  ("`CLAUDE.md` 禁止 `nohup`/裸后台,即便是分析任务 → 用 harness-tracked 后台机制"),
  但这是**规则重申**,不是"这次 phase0_verify 运行本身违反过这条规则又被修正"的具体事故记录 ——
  `phase0_verify.py` 是一次同步跑完的验证脚本(两个 log 时间戳相差 12 分钟),不属于
  CLAUDE.md 定义的"long-running"训练/sweep 场景。**同样标注为证据不足,不作为已发生故障呈现。**

### 5. definitive 6-cell 表 + 2P do-no-harm(逐格核对 log 原文,不只信 SESSION_STATE 转录)

对照 `results/phase0_verify_20260701_1941.log` 原文与 `SESSION_STATE.md` §C 表格,逐格一致,
仅一处**呈现方式**上的差异需要指出:

| Cell | log 原文(P1/P2) | SESSION_STATE §C 表格 |
|---|---|---|
| dc q55 | P1 +0.02(0.09%) / P2 -0.00(0.02%) | "+0.02 / −0.00 (~0.05%)" |

SESSION_STATE 表格把两个 player 的百分比合并写成单一"~0.05%"(约等于 0.09% 与 0.02% 的
粗略中间值/平均),log 原文是分开的 0.09%/0.02%。数值方向和量级都对(两者都 ≪1%,不影响
6/6 PASS 结论),但这是一处**转录时的取整/合并**,建议 review 时留意 —— 这正是用户要求
"不要只信 SESSION_STATE 转录"要抓的那类问题。其余 5 个主 cell + 2P do-no-harm 两行,
逐字段核对(polished 值、误差、(a)/(b)/(c) 三条 leg 数字)均与 log 完全一致,无出入。

Overall verdict 核对:log 第 41 行"OVERALL: 6 main cells all PASS = True   max polished
error-vs-e* = 0.319"与 SESSION_STATE §C 引用逐字一致。

### 6. SESSION_STATE.md 合并(108→220 行)

- `git show HEAD:SESSION_STATE.md | wc -l` = **108**(对应最后一次提交 `2ee175a`,
  "docs: add SESSION_STATE.md handoff and record cleanup verification",2026-06-12)。
- 当前工作树 `wc -l SESSION_STATE.md` = **220**。
- `git diff --numstat SESSION_STATE.md` = `112  0  SESSION_STATE.md` —— **112 行新增,
  0 行删除**,即这是一次纯追加式合并,没有覆盖或删改此前 108 行的既有内容
  (§A-§开头的"What this whole effort did"/"Adopted canonical state"/"THE ONE OPEN TODO"
  等段落原样保留)。
- 新增的 112 行就是"PHASE 0 — DEFINITIVE CONSOLIDATION"整段(§A-§E)+
  "PHASE 0 — SELF-CORRECTION HISTORY"+ "OPEN DECISIONS" 三大块,对照
  `SESSION_STATE_phase0_consolidation.md`(未跟踪的合并草稿,顶部注释写明
  "这是要**合并进**vector2 上已有的 108 行 SESSION_STATE.md 的 Phase-0 整合块，不是整体替换")
  逐段核对,内容一致(仅去掉了草稿里给 Claude Code 自己看的集成说明 HTML 注释)。
  **这次落盘是"第一次落盘",不是"从损坏状态里恢复出一份已有文档"** —— 上一节已确认
  这份整合内容本来就是 uncommitted、随 vector 服务器损坏而丢失的工作,这次是凭
  `Phase0_reconstruction.md` 的规格重新撰写,而不是找回了原文件。

---

## Part 2 — 未来计划

### 1. Claim A vs Claim B —— 已拍板:Claim A(2026-07-02 owner 确认)

**更新(2026-07-02,owner 确认)**:Owner 已确认"Owner authorized Component-2 retrain
(walk toward Claim A)"这条记录准确 —— 即 §1 下面列的矛盾已解决,以
`docs/tasks/component2-mode-conc-retrain/STATE.md`/`CLAUDE.md` 的措辞为准:
**2026-07-01 已授权朝 Claim A 方向走,SESSION_STATE.md 里"待 Frank 拍板"的表述已过时**。
按你的要求本次仍未改动 `SESSION_STATE.md` 本身 —— 如果需要,下一步可以专门开一次
"更新 SESSION_STATE.md OPEN DECISIONS 段落"的任务来同步这个决策,而不是顺带改。

在此确认之前,这个决策已经在至少三个时间点被提出但迟迟未拍板,记录如下(历史脉络,
供追溯):

- **6/23 会话**(Part 0,历史会话记忆):裁决 Component 2 保留、定位为 retrain-gated
  Phase-1 lever —— 隐含了"如果 raw PPO 够不到 e*,就用 retrain 去够"的方向,但当时
  未明确表述为"Claim A vs B"这个二分框架。
- **6/25 会话 + `docs/phase0_response_to_revision_plan.md` §H**(Part 0,历史会话记忆):
  据背景陈述,§H"专门讨论 Claim A/B 这个 framing question"—— 但这份文档本身在仓库中
  查无踪迹,无法核实 §H 具体写了什么、是否已给出倾向性结论。
- **本次 vector2 重建**:`SESSION_STATE.md` 末尾"OPEN DECISIONS(待 Frank 拍板)"明确写着
  "决策 1(headline)Claim A vs B"仍待拍板,只是注明"当前证据(6/6 PASS + attribution)
  指向 B"。`Phase0_reconstruction.md` §9 同样把这列为"悬而未决的决策(重建后要你拍板)"。

`docs/tasks/component2-mode-conc-retrain/STATE.md`(第 9 行)和
`docs/tasks/component2-mode-conc-retrain/CLAUDE.md`("Authorized by owner 2026-07-01")
两份文件都写着"Owner authorized Component-2 retrain (walk toward Claim A), 2026-07-01"——
这曾与 `SESSION_STATE.md` 里"待 Frank 拍板"的措辞矛盾;**owner 已于 2026-07-02 确认前者
准确,矛盾已解决,决策 = Claim A**。

以下按"已选 A"整理现状/代价/下一步,不再中立并列(Claim B 一栏保留仅作历史对照):

| | Claim A(已选定 —— PPO 自己学到均衡) | Claim B(备选,未选中) |
|---|---|---|
| **现状** | Component-2 代码已在 `agents/ppo_three_players.py`、`run/run_three_players.py`(均为 uncommitted 修改,`git diff --numstat` 显示 +81/+159 行)里实现完成并 CPU smoke-tested 通过;GPU 环境已修复(torch 2.5.1+cu121);**尚未跑真实 5-seed GPU 实验** | §C 6/6 PASS + attribution finding 已齐备,但不再是当前路线 |
| **代价 / 风险** | 需要跑 3P q35 5 seeds × 全量 episodes(6,144,000)GPU 实验,per-seed 约 30min–2h;若 raw policy mean 在 κ=200 仍达不到 e*=25,则"stays Claim B, report to owner (no massaging)"(`STATE.md` 第 30-31 行原话)—— 这个实验有**明确的可能失败结果**,选 A 不等于保证 A 成立,只是授权去验证 A | — |
| **下一步** | 确认参数(`ramp-trigger-exp 0.05`, `patience 3`, `stage-hold 20`, `exploit-every 10`),tmux 起 5-seed GPU 跑(`STATE.md` 第 27-29 行清单可直接执行);跑完后按 §2 的两种结果分支处理 | 若 A 实验结果不支持(仍 undershoot),回落到 B,再执行 B 栏下一步 |

### 2. Component-2 retrain spec —— 已有两个独立来源印证,若选 Claim A 可直接用

`Phase0_reconstruction.md` §9(第 249-259 行)与 `docs/tasks/component2-mode-conc-retrain/CLAUDE.md`
("Locked decisions"一节)对以下参数**独立给出且完全一致**,不需要凭猜测重建:

- Mode-conc head:`s=sigmoid(mode_head)`,`κ=clamp(softplus(conc_head)·scale+κ_min, max=κ_max)`,
  `α=1+s·κ`,`β=1+(1−s)·κ`
- κ schedule:Explore(κ∈[1,20])→ trigger on `EXP_raw<0.05` 连续 3 次 in-loop eval →
  Ramp κ∈[20,50,100,200],每 stage 20 updates → hold 至 κ=200 后正常 exploit stop
  (eps_eq=0.03, patience 5)
- 4 个新 CLI flags:`--mode-conc-ramp`、`--kappa-schedule 20,50,100,200`、
  `--ramp-trigger-exp 0.05`、`--kappa-stage-hold 20`
- 规模:3P q35,seeds 42–46,episodes=6,144,000(full),K=1

且 `docs/tasks/component2-mode-conc-retrain/STATE.md` 显示这套 spec **已经落地为代码**
(uncommitted,`agents/ppo_three_players.py` + `run/run_three_players.py` 的当前 diff),
CPU smoke test 已通过(explore→ramping→done,κ 按 20→50→100→200 在 stage_hold 处步进,
α+β−2 精确 pin 到 κ,mean+mode 都写入 JSON,无崩溃)。即如果拍板 Claim A,不需要重写这套
spec,可以直接确认参数后在 GPU 上跑 5-seed 实验。

### 3. `docs/phase0_response_to_revision_plan.md` —— 确认在本仓库不存在,且 owner 已确认找不到原稿

`find . -iname "*phase0_response_to_revision_plan*"` 在整个仓库范围内零命中,确认这份文档
(据 Part 0 背景,曾在 6/25 会话中起草,§H 讨论过 Claim A/B 这个 framing question)确实
没有随任何形式(git 提交、工作树文件)留存在当前仓库里。

**更新(2026-07-02,owner 确认)**:owner 已确认找过、确实找不到原稿了 —— 不再是
"待搜索",而是**确认永久丢失,需要重新起草**。既然 §1 的 Claim A/B 决策已经拍板为
A,重新起草这份文档时:

- 不必再保留 §H 原本"讨论 Claim A/B 该怎么选"的悬而未决口吻 —— 直接以"已选 Claim A,
  Component-2 retrain 正在验证"为前提写,§H 的角色从"提出问题"变成"记录选择 A 的理由
  和已知风险(retrain 可能不支持 A 的情形)"。
- 可以复用的、已核实存在于本仓库的素材(避免凭空补内容):`Phase0_reconstruction.md`
  全文(Finding A/B、attribution finding、Component 1-5 disposition、两处 self-correction、
  §9 决策记录)、`SESSION_STATE.md` §A-§E + self-correction 段、
  `results/phase0_verify_20260701_1941.log` 原始数字、
  `docs/tasks/component2-mode-conc-retrain/CLAUDE.md`+`STATE.md` 的 Component-2 spec 与进度。
  这些都已在 Part 1 逐条核对过,可以直接引用,不需要重新推导。
- 建议:等 Component-2 的 5-seed GPU 结果出来后再写这份文档的结论部分(§H 的核心是
  "raw policy mean 是否够到 e*"这个尚未产生的实验结果),现在起草的话只能先写方法论
  和背景部分,结论部分留空/待补。
