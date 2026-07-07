# TEL-PPO Phase-0 重建手册 (Reconstruction Handoff)

**用途**：原 Vector 服务器在完整 push 前损坏。这份文档从 `Revision_Plan.md`（Claude Code 的**纯文字叙述记录**，不含源码/bash 输出）中提炼出所有**可用于在 vector2 新仓库重新实现**的内容——方法规格、已锁定的决策、阈值、验证过的数值、以及两处**绝不能复活的自我修正**。

> ⚠️ 关键限制：`Revision_Plan.md` 里**没有任何源码**。因此下面是**重建规格 (spec)**，不是逐行代码。脚本本身要么靠 `git` 恢复（见 §0），要么按 §4–§5 的规格由 Claude Code 在 vector2 重写。所有数值均取自该叙述记录，**未编造任何数字**；未被记录进叙述的定义性数值明确标为"叙述未捕获，需重跑"。

分支：所有工作都在 `fix/audit-remediation` 上。

---

## §0. 第一步 —— 先确认 `git` 能恢复多少

三个脚本曾被**本地提交**为 commit `8df54d0`（"feat: add MC-BR polishing module and Phase-0 verification drivers"，3 files, 544 insertions），message 里记录了 seeds / M / FD step (δ=0.75) / thresholds。但 push 是否成功**未知**。clone 到 vector2 后先跑：

```bash
git log --oneline --all | grep 8df54d0
git branch -a | grep audit-remediation

```

- **若** `8df54d0` **在远端** → `git checkout fix/audit-remediation` 即可恢复以下三个脚本，**不需按规格重写**：
  - `utils/mc_br_polish.py`
  - `tools/phase0_verify.py`
  - `tools/phase0_decomposition.py`
- **若没 push** → 按 §4–§5 让 Claude Code 重写这三个文件。

**无论哪种情况，以下两项都是 uncommitted 工作树改动，一定丢了，必须重建**（见 §7–§8、§10）：

- `SESSION_STATE.md` 的整合（verdict + provenance + superseded 映射 + 两处 self-correction）
- `docs/phase0_response_to_revision_plan.md`（response 文档）

另外 `tools/phase0_mc_br_polish.py`（最早的一次性脚本）已被 decomposition/verify 取代，**不必恢复**。

---

## §1. Phase-0 三个核心实证发现（重新构建实验前必须理解）

这三点决定了整个 pipeline 的取舍。全部是在**已有 r5 runs 上做的零 GPU probe** 得出的。

### Finding A —— mode extraction 完全 inert

- 收敛时 3P Beta 的 concentration α+β ≈ **25,000–33,000**（near-spike），因此 mean ≈ mode 精确到小数点后 3 位。
- 10 runs：mean-extraction 的 mean|err| = **1.531**，mode-extraction = **1.532**（几乎相同）。
- 这个 over-concentration 是 `--theory-align-v2 --override-conc-ramp-warmup 200` **强制造成的 artifact**，不是内在收敛：α+β 在 update 200→250 之间**暴涨 30×（918 → 28,909）**，正好在 warmup=200 边界。
- 但 **lock 只是冻结了 undershoot，没有制造它**：effort 在 update ~100–150、concentration 还只有 ~300 时就已降到 ~23.5–24（低于 e*=25）；ramp 随后把它钉在 23.4 并阻断回升。
- 全轨迹上 **max |mean−mode| = 0.20 effort units**（在 conc≈230 处），对比 ~2.0 的 gap → 任何 PPO 实际停留的 concentration 下 Beta 都太对称，mode 修不了误差。

### Finding B —— undershoot 是真实的 best-response gap，e* 是对的目标

复用已有 sampled MC-FD 梯度（`_stochastic_fd_gradients_3p`，CRN）：


| q   | e_conv | sampled FOC @ e_conv | sampled FOC @ e* |
| --- | ------ | -------------------- | ---------------- |
| 35  | 22.99  | **+0.00398**（上坡）     | −0.00023 (≈0)    |
| 55  | 15.31  | **+0.00150**（上坡）     | −0.00011 (≈0)    |


→ analytic e* ≈ **sampled game 的均衡**；policy 冻结在低约 2 units 处。这正是 **MC-BR polishing（Component 4）能起作用的条件**，且 polishing **零 GPU、不需要 policy net**（只要 efforts + game params + sampled-payoff helper）。

### Attribution finding —— polish 是 global solver（对论文 claim 至关重要）

测试 polish 是否依赖 PPO 的 warm start：**不依赖**。MC-BR 从任意初值都能到 e*：

- 3P q35 (e*=25)：inits 10/25/40/60 → 24.75 / 24.81 / 25.03 / 24.78
- dc q35 (e*=[38.03, 27.66])：inits (15,15)/(50,50)/(38,28) → 全部 → [≈38.1, 27.66]
- da q35 (e*=46.43)：inits 20/46/70 → 全部 → ≈46.5

**结论**：polish 是 global solver，PPO 输出对"polished 数字"而言只是 cosmetic。因此"TEL-PPO 通过 polishing 恢复 e* 到 <1%"这句话若直白说会**误导**。Phase-0 合法确立的是：(i) sampled-game 均衡 ≈ analytic e*（所有 cell <0.2，一个干净的 certificate）；(ii) PPO 落在那个 basin 里；(iii) 一个 sampled、无 closed-form 的 refinement + exploitability 检验证明了 proximity。**这直接引出 §10 的 Claim A/B 决策。**

---

## §2. 五个 Component 的最终处置


| Comp                                 | 处置                          | 依据                                                                                                                                                              |
| ------------------------------------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(1) MC reward avg (K)**            | **DROPPED，K=1**             | 唯一触碰 training reward 的组件。not "tried and failed"，而是**根本不需要**。若 K>1 会软化论文"single realized tournament outcome"的说法（需 owner 签字），上限 K≤4。                              |
| **(2) mode-conc + conditional ramp** | **DROPPED**（spec 存档，见 §10）  | α+β≈25k–33k 的 near-spike 使"width"前提失效；且 polish 是 init-independent，retrain 动不了 polished 落点。**但它是唯一能区分"PPO learned it"(Claim A) 与"MC-BR found it"(Claim B) 的实验**。 |
| **(3) mode extraction**              | **INERT**（保留为 diagnostic 列） | L1 mode Δ ≈ −0.00。3P 的 α,β 在 JSON 里 → 可 post-hoc；**dc/da 只存 effort，不存 α,β → mode N/A**。                                                                         |
| **(4) MC-BR polish**                 | **LOAD-BEARING（核心可复用产物）**   | 零 GPU、sampled-only、不需 policy net。见 §4。                                                                                                                          |
| **(5) exploitability + FOC stop**    | **finalized**               | exploit-streak stop 已是 production（`stop_reason="exploitability"`, eps_eq=0.03, patience 5）。新增：在 frozen polished profile 上用 fresh seed 做第二次 exploit。             |


**最终 pipeline（Claim B 路线）**：post-hoc MC-BR polish 单独就把 3P/dc/da 的 undershoot 收到 verified sampled 均衡，零 GPU。Components **1 & 2 DROPPED，3 INERT（保留为诊断），4 LOAD-BEARING，5 finalized**。

---

## §3. `utils/mc_br_polish.py` —— 方法规格

这是最重要的重建目标。核心是一个 **sampled-only 的 MC best-response polishing** 模块。

**不变量 / 硬约束（务必在实现里 verify，不是 assert）：**

1. **Sampled-only**：计算路径**只**能 import `stdlib + numpy + utils.theory`。`utils.theory` 里的 closed-form e* **仅**用于 error reporting（验收 leg d），**绝不**进入 polishing/BR/exploit/FOC 路径。**严禁** import 任何 closed-form win-prob / expected-utility：`win_prob_`*、`p_from`、`expected_utility`、`utils.prob`。—— 一旦 BR 用了 closed-form win-prob，就等于在数值求解 analytic FOC，sampled-training 的 claim 就死了。
2. **等价性自检**：模块内置 equivalence check，确认自己的 `sampled_payoff_player`（向量化 n-player sampled payoff）与仓库里 canonical 的 `_payoff_player`（在 `exploit_asymmetric.py`）数值一致 —— 用这个把"sampled-only"**证明**出来而非断言。

**算法要点：** 3. **固定 deterministic 对手**：BR 是针对**固定确定性对手 efforts**，**不是** policy samples。（注意：已有 `eval_exploitability`* 是对 stochastic policy 采样的 —— 这是区别，polishing 要的是对 fixed deterministic 对手的 BR。） 4. **per-player MC-BR + damped simultaneous update**，stop on |Δeffort| < τ_e。 5. **Iterate-averaging（Polyak–Ruppert，post-climb averaged）是必须的**：3P payoff peak 极其 flat，单次 MC best-response 即使 M=200k 也有 ±1.8 噪声（e*=25 处 BR 被采成 [23.0, 23.65, 25.65]）。**不能读最后一个 iterate**（会 oscillate，|Δe|~0.5–1.5/round）—— 要对 iterate 做平均，并用 proper convergence-drift 度量。 6. **Quadratic-vertex BR（默认 OFF，用** `bias_correct` **flag 打开）**：在 argmax 附近的窗口上对 **mean-payoff 值**拟合抛物线取顶点。**真实收益是降方差**（例如 e=25 处 argmax 24.89±0.37 → vertex 24.75±0.27），**不是去偏**（见 §8 自我修正 #1）。默认 OFF 以保持 module self-test 不变。保持 vertex BR 是"收益最大化器"这个 functional，与验收 leg (b) 的 first-order FD-FOC **不同**，从而不制造循环（见 §6 non-circularity）。

**复用来源（在 clone 的仓库里定位）：**

- sampled payoff：`_payoff_player`（`exploit_asymmetric.py`）
- BR grid：`eval_exploitability`* 里的 coarse-to-fine grid（`find_best_response_player1/2`）
- MC-FD ascent w/ CRN：已有

---

## §4. `phase0_decomposition.py` 与 `phase0_verify.py` —— 规格

### Decomposition driver (`tools/phase0_decomposition.py`)

逐 cell 输出 **L0 → L1 → L2 分层表 + 每层 signed Δ + 验收 verdict**：

```
L0 baseline (r5 raw-mean)  →  L1 +mode-extract  →  L2 +MC-BR-polish   [全 POST-HOC, 0 GPU]

```

（L3 conditional-ramp retrain / L4 K-avg 是 GPU 层，仅当 L2 留残差才触发 —— 结论是**没触发**。）

### Verification driver (`tools/phase0_verify.py`)

**definitive 版本**：所有 6 cells 用 **debiased (vertex) BR + independent 验收 legs**，外加 2P do-no-harm。

**Cells & 参数：**

- **6 主 cells**：3P, dc, da × q∈{35,55}
- **2P {35,55}** 作为 do-no-harm 对照
- **seeds**：已有 5 个（42–46）；root-find 用 fresh seeds 501/502/503
- **M**：verify M=200k（部分 300k）；root-find M=2M
- **max_rounds** ≈ 220，**n_avg** = 80，**FD step δ = 0.75**
- 验收 legs 在 **frozen polished profile 上用 fresh independent seeds** 评（不复用 polish 的 seed 族）

**dc 特别处理**：dc 按 plan §7 判 **payoff-loss + exploitability**，不是单纯 effort error（weak-identification；mixed-sign raw error 就是 weak-id 的指纹）。

---

## §5. 预注册验收门 (Pre-registered Acceptance Gate)

**先记录规则、后碰数字**（规则独立于数据）。5 条 + 阈值：


| leg                                                                                 | 条件                                | 阈值                   | 理由                                                                                                                                              |
| ----------------------------------------------------------------------------------- | --------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **(a)** post-polish exploitability（fresh draws，constant efforts + independent seed） | EXP < **τ_E**                     | **0.005**            | Phase-0 EXP_pol ∈ [0.0002, 0.0009]，estimator floor ~0.0002–0.001（max-over-grid bias）。0.005 = raw stop(0.03) 的 6× 以下、floor 的 ~10× 以上。别低于 ~0.002。 |
| **(b)** interior FOC（boundary → projected/KKT）                                      | |FOC| < **τ_g**                   | **0.001**            | |FOC@pol| ∈ [0.00006, 0.00089]，MC-FD noise floor σ≈0.0003–0.0007。0.001 刚好在 floor 之上。                                                            |
| **(c)** polish converged                                                            | |Δe| < **τ_e**                    | **0.1** effort units | cross-seed σ(e_pol)=0.02–0.22（紧）。用 cross-seed σ 或更长 averaging window，别用 raw within-run drift（那个 under-resourced）。                               |
| **(d)** payoff loss small                                                           | = u_br − u_policy = exploit delta | —                    | 已计算，复用。                                                                                                                                         |
| **(e)** abs + rel error                                                             | `utils/evaluation.py`             | —                    | 复用。                                                                                                                                             |


**Non-circularity（§3.6 的动机，务必守住）**：leg (b) **不能**用 FOC-root-find 做 polish —— 那样 polished 点 by construction 就是 polish 自己 FOC=0 的地方，(b) 会退化成"FOC-estimator 噪声 < τ_g"即 **auto-pass**。正确设计：**零阶 quadratic-vertex polish + 一阶 fresh-seed/different-step/larger-M 的 (b) + 独立零阶 (a)**。(a)+(b) 承担验证权重；(a) 结构上无法被 polish 方法 game。

**Directional undershoot guard**：不要在 grazing-0.03 处接受"所有 seed 都 undershoot 且 MC-FD 指上坡"的情况（当前 code 只按 first 5-streak under 0.03 停，需要新增这个方向性守卫）。

---

## §6. 验证结果（含 provenance 诚实标注）

⚠️ **provenance 警告**：definitive verify run 的**完整 6-cell per-player 表**是在一次 tool 调用里 pull 的，**该输出没进叙述记录 → 丢失**。叙述里**只**确切保留了 3P q35 的 definitive 数字。其余 5 cell 的 definitive 数字**需在 vector2 重跑 verify 得到**。下面严格区分"definitive（叙述已捕获）"与"SUPERSEDED（仅作 sanity 目标）"。

### DEFINITIVE（叙述已捕获，可信）

- 总结论：**all 6 cells PASS (100% seeds)，2P do-no-harm 两个 q 都 pass**。
- **3P q35**（definitive verify，vertex BR + independent legs）：
  - polished = **24.68**，err **−0.32 / 1.28%** vs analytic 25
  - legs：(a) EXP **0.0008** / (b) |FOC| **0.00044** / (c) SE **0.028**
  - **VERDICT PASS (100% seeds)**
  - 这 1.28% 是 **sampled-vs-analytic gap**（sampled 均衡本就在 analytic 25 下方 ~0.3），不是学习失败、不触发 Phase-1。

### SUPERSEDED —— 仅作重跑后的 sanity 目标，**不是最终数字**

下表来自 `phase0_decomposition.py`（**argmax BR**，被 vertex-BR verify 取代）。3P q35 在这版里因 leg (b) FAIL，后经诊断确认是 (c) 停在 (b) 满足之前的 artifact，非真实 gap：


| cell   | L0 raw-mean (err) | L1 +mode Δ | L2 polished (err vs e*) | 备注                              |
| ------ | ----------------- | ---------- | ----------------------- | ------------------------------- |
| 3P q35 | 22.99 (−2.01)     | −0.00      | 24.75±.05 (−0.25, 1.0%) | 此版 (b) FAIL → definitive 版 PASS |
| 3P q55 | 15.31 (−0.60)     | −0.00      | 15.80±.06 (−0.11, 0.7%) | PASS                            |
| dc q35 | 37.71/27.24       | N/A        | 37.99/27.68 (−.04/+.02) | PASS                            |
| dc q55 | 26.39/19.03       | N/A        | 26.58/19.32 (+.04/+.02) | PASS                            |
| da q35 | 43.99 (−2.44)     | N/A        | 46.41/46.46 (−.02/+.03) | PASS                            |
| da q55 | 29.70 (−0.67)     | N/A        | 30.35/30.34 (−.02/−.03) | PASS                            |


**2P do-no-harm（较早一版，argmax BR；definitive 版仅确认"两 q pass"）**：


| cell                      | baseline mean (err) | polished (err) | no-harm |
| ------------------------- | ------------------- | -------------- | ------- |
| 2P q35 (e*=45.45, σ=1.09) | 44.09 (1.36)        | 44.99 (0.47)   | ✓       |
| 2P q55 (e*=28.93, σ=1.09) | 30.51 (1.61)        | 28.78 (0.15)   | ✓       |


（polish 在 2P 两 cell 都朝 e* 改善、在 cross-seed σ 内 → 无 regression、无 polish bug；mean–mode gap 0.06/0.15 极小，印证 Finding A。）

**SUPERSEDED 值映射（保留 provenance，别删）：**

- 3P q35 → 24.81（首个一次性脚本）、24.75（decomposition）、"24.94±0.04 FOC=0 point" → **全部被 definitive 24.68 取代**。
- "24.94 confirmed FOC=0 point" 与 argmax-bias 的说法 → 见 §8，**已证伪，勿复活**。

---

## §7. 两处自我修正（写进 SESSION_STATE，防止任何未来 session 复活）

措辞用"先前结论 X 被证据 Y 证伪，替换为 Z"：

**修正 #1 —— "argmax-BR 有 ~0.17 向下偏差" → 证伪。** 证据：debias smoke 几乎没动均值（24.821 → 24.811，而非预期的 ~0.13）。vertex debias 的真实收益是**降方差**（e=25 处 24.89±0.37 → 24.75±0.27），**不是去偏**。polish-vs-rootfind 的 gap 不是 argmax bias，而是 **flat-plateau 宽度**。

**修正 #2 —— "confirmed FOC=0 point = 24.94 ± 0.04" → 过度精确。** ±0.04 只是 3-seed spread，但底层曲面太平：|FOC| < ~0.0003 横跨 [24.7, 25.0]，与 FOC estimator 自身噪声地板相当。**3P q35 的 sampled 均衡只能钉到 ~[24.8, 25.0]**，其中任一点都 pass |FOC| < τ_g。

**（附带一处 mid-course 修正）—— "polishing 帮不了 dc（flat FOC / weak-id）" → 错。** dc 干净 polish 到 e*（±0.03）。MC-BR 是重平均**直接求解器**（~40 rounds × 150k draws），能定位 flat optimum，即便单次采样奖励下的**学习**agent 不能。Weak-identification 讲的是 PPO 的学习动态与经济解读（payoff-flat ⇒ effort 弱识别、tiny EXP），不是暴力求解器能否找到 e*。dc 仍按 §7 报告，mixed-sign raw error 是 weak-id 指纹。

---

## §8. 基础设施 & 复现注意事项

- **无 checkpoint**：`find … -name '*.pt'` → 0，runners 里没有 `torch.save`。因此 Components 3/4/5 都在 **efforts/params（来自 JSON）** 上操作，不碰 net。
- **JSON schema**：
  - 3P PPO JSON 记录 per-update 的 `alpha_mean` / `beta_mean` 历史（len ≈ 310）→ **3P 的 mode 可 post-hoc 算**。
  - **dc/da JSON 只存 effort，不存 α,β** → dc/da 的 mode 需 re-eval，**不可 post-hoc**。
- **旧 r5 运行的陷阱**：3P/dc r5 用了 `--theory-align-v2 --override-conc-ramp-warmup 200`。这个 v2 ramp 把 conc_min 100→1000、conc_scale 100→**10000**、conc_max **100000** 拉爆 → α+β≈30k。**若重跑训练，避免这套 forced ramp**（Component 2 是 κ≤200 的更温和 regime，见 §10）。
- **GPU**：8 CUDA devices 可用；NVML / `nvidia-smi` mismatch **只**破坏监控，不破坏 compute（torch 正常看到设备）。bandit model 很小。
- **后台作业规则**：`CLAUDE.md` **禁止** `nohup` / 裸后台，即便是分析任务 → 用 harness-tracked 后台机制。
- **config 默认 episodes = 6,144,000**（完整运行值，非 quick-run）。
- 仓库里可复用/需定位的文件与函数：
  - `run_two_players.py:1038-1059`、`run_three_players.py:886`（rollout store loop，Comp 1 若需要则改这里）
  - `ppo_two_players_clean.py:62` → `ActorCriticMeanConc`（**mean**-conc head，`alpha=mean*conc`，**不是** Comp 2 想要的 mode head）
  - `exploit_asymmetric.py` → `_payoff_player`（realized sampled payoff，无 closed-form）、`find_best_response_player1/2`
  - `_stochastic_fd_gradients_3p`、`_compute_gradients_different_cost` / `_compute_gradients_different_ability`（sampled FD 梯度）
  - `eval_exploitability_3p`（`run_three_players.py:1000` 附近，in-loop EXP，给 Comp-2 的 conditional trigger 用 `EXP_raw`）
  - `utils/theory.py`（closed-form e*，仅报告用）、`utils/evaluation.py`（abs+rel error）

---

## §9. 悬而未决的决策（重建后要你拍板）

### 决策 1（headline）：论文的中心 claim 是 A 还是 B？

- **Claim A**："PPO self-play **学到**均衡 effort" → polish **不能**支撑（polish 是 global solver，从哪都到 e*）；**只有** PPO 自身（raw 或 conditionally-ramped）输出到达 ~e* 才行 → **必须**跑 Component-2 retrain（§10）。polish 降级为 certificate。
- **Claim B**："PPO 到达 basin；sampled MC-BR + exploitability **证明**均衡" → polish 够用，但论文**必须明确把最终数字归功于 MC-BR**，raw / polished 保持**独立两列**。当前证据（6/6 PASS、attribution）**指向 Claim B**。

### 决策 2：审稿人预防 —— "post-hoc polish ≠ proof of convergence"

必须在论文里主动回应这一点（polish 是 global solver 的事实会被审稿人抓）。

### 决策 3（仅当选 Claim A）：授权已存档的 Component-2 retrain

**Component-2 spec（存档，别丢）：**

- **Mode-conc head**（新 `ActorCriticModeConc`）：`s = sigmoid(mode_head)`；`κ = clamp(softplus(conc_head)·scale + κ_min, max=κ_max)`；`α = 1 + s·κ`，`β = 1 + (1−s)·κ`。保证 α,β ≥ 1（interior mode）；deterministic effort = mode = s 映射到 bounds。
- **Conditional ramp 状态机**：
  - Explore（trigger 前）：κ 钉宽，`κ_min=1, κ_max=20`，entropy-driven，无 concentration 压力（让 mean 自由爬升，23.4 处 FOC 还 +0.004 上坡）。
  - Trigger：raw exploitability `EXP_raw < 0.05` **连续 3 次** in-loop eval（防瞬时 dip）。
  - Ramp（trigger 后）：κ 走 stages **[20, 50, 100, 200]**，每 stage 钉 `κ_min=κ_max=stage`，每 **stage_hold = 20 updates** 进一档；到 κ=200 后 hold 到正常 exploitability stop（eps_eq=0.03, patience 5）。
- **新 CLI flags**（3P runner）：`--mode-conc-ramp`、`--kappa-schedule 20,50,100,200`、`--ramp-trigger-exp 0.05`、`--kappa-stage-hold N`。
- **配置**：episodes = 6,144,000，K=1，asymmetric warmup 保持原样。
- **实验规模**：3P q35，5 seeds（42–46），smoke test（1 seed ~30 updates 验证 ramp 触发/κ 步进/mode 提取/不崩）→ 5 seeds 并行 tmux，GPU-pinned via `CUDA_VISIBLE_DEVICES`，per-seed ~30–45 min，并行 <1h。
- **注意**：这与旧 `--theory-align-v2` 是**根本不同**的 regime（κ 上限 200 vs 旧的 conc_scale=10000 / conc_max=100000），需**新代码**，不是 flag flip。

---

## §10. 重建执行顺序建议（给 vector2 上的 Claude Code）

1. clone 仓库，跑 §0 的 `git` 检查 → 确定三脚本是否需要重写。
2. 若需重写：先 `utils/mc_br_polish.py`（§3），**先过 equivalence self-test 与 import-hygiene 检查**再往下。
3. 写 `tools/phase0_decomposition.py` 与 `tools/phase0_verify.py`（§4），阈值按 §5 wire 成验收规则。
4. **先写 gate（§5，不含任何数字）进 SESSION_STATE.md，再重跑**。
5. 重跑 verify（6 cells + 2P，vertex BR + independent legs），拿回丢失的 definitive 全表（§6 只保住了 3P q35）。
6. 把 verified 结果 + verdict + §7 两处 self-correction 整合进 SESSION_STATE.md（superseded 值标 `SUPERSEDED → <replacement>`，**别删**）。
7. 之后再写 `docs/phase0_response_to_revision_plan.md`，每个数字标 source（SESSION_STATE section / JSON / run path / seed / M / FD step），可独立复算。
8. 拿 §9 找你定 Claim A/B —— 若 A，才动 Component-2 retrain。

**全程铁律**：no fabricated numbers（一切数字溯源到 SESSION_STATE）；single deliverable per prompt + human review gate；no self-approving commits；main.tex 在你签字前不碰。