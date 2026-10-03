# Executive Summary

旧框架的主要目标是通过 dReach/ΔW≤0.01 的 development/final verifier 条件找到 candidate；新目标则要求先在 T=2 中同时验证 policy recovery 与 strategic optimality，再把同一套训练与验证原则用于 T=3。

旧方案本身没有高精度恢复解析 effort function，而且在训练组织、phase progression、verifier schedule、stage parameter sharing、KL 控制、checkpoint branchability 等方面仍有需要修正或升级的地方。

新方法组织为：Stagewise TEL--PPO search + [frozen continuation]{.mark} + scheduled development DP-BR + locked [full-domain]{.mark} final DP-BR。T=2 作为算法开发与校准环境；T=3 只在 T=2 validation gate 通过、v2 protocol 完全锁定后进行。

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **层次**                     **核心问题**                                         **建议主指标**                                 **作用**
  ---------------------------- ---------------------------------------------------- ---------------------------------------------- -------------------------------------
  Learning                     是否能搜索出稳定的 stage-dependent effort policy？   PPO / stagewise training logs                  候选策略生成

  T=2 Recovery                 学到的 effort function 是否接近解析均衡？            Stage-1 error; Stage-2 RMSE; peak/tail error   验证 action-level accuracy

  Strategic Quality            是否仍存在有利的动态单边偏离？                       [Gmax_full/ΔW]{.mark}                          MPE-oriented full-domain assessment

  Supplementary Verification   root / reachable / one-step 情况如何？               EXP_root; dReach; Δmax_all; dFull              补充诊断与可解释性

  Numerical Reliability        结论是否依赖 verifier 分辨率？                       dev-final refinement                           数值可信度
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. 研究目标的重新定义

## 1.1 新目标

把 T=2 变成真正的 solver validation environment：closed-form equilibrium 不只是画图参考，而是用于开发阶段诊断算法为何在 action space 中存在系统偏差；方法锁定后，再使用 fresh held-out seeds 做 confirmation。[closed form 不应作为 formal learner 的 supervised target]{.mark}。

$$T = 2\ closed - form\ validation\ \  \rightarrow \ \ lock\ solver\ \  \rightarrow \ \ T = 3\ verified\ computation$$

## 1.2 三层方法论结构

1.  Learning layer：PPO/self-play 只负责生成 candidate policy；RL 本身不作为 equilibrium proof。

2.  Calibration layer：T=2 同时检查 analytical recovery 与 strategic deviation。

3.  Verification layer：T=3 没有完整 closed form，因此可信度来自已在 T=2 校准的 solver、独立 dynamic BR verifier 和 numerical refinement。

# 2. 理论基础与新的验证对象

## 2.1 Markov policy 与 symmetric strategy

经济状态为 stage 与当前 score gap。共享对称策略不意味着 e_t(d)=e_t(-d)；它意味着两名玩家使用同一条函数，但对手在玩家视角下使用 e_t(-d)。

$$s_{t}\  = \ (t,\ d_{t})$$

$$opponent\ effort\ at\ state\ d\ \  = \ \ ê_{t}( - d)$$

## 2.2 Two-stage closed-form benchmark

在当前模型参数化与适用条件下，T=2 的解析均衡提供 stage-1 scalar target 和 stage-2 full state-dependent effort curve。它们只用于开发诊断和冻结后的 recovery evaluation。

$$e_{1}^{*}(0)\  = \ \frac{\Delta W}{6kq}$$

$$e_{2}^{*}(d)\  = \ \frac{\Delta W}{2k}f_{\xi}(d)$$

对于 uniform performance noise，f_ξ(d) 在其支持区间内是 triangular density，因此最终阶段解析 effort curve 是以 d=0 为峰值、向两侧线性下降并在理论 tail 区域归零的 hump-shaped function。

## 2.3 One-step deviation 与 full dynamic deviation

对一份冻结的 candidate policy ê，one-step deviation 只允许当前 stage 改 action，之后重新回到 candidate continuation；full dynamic deviation 则允许从当前 state 开始在所有后续阶段持续重新优化。

$$\Delta_{t}(d)\  = \ maxₑ\ Q_{t}^{ê}(d,e)\  - \ Q_{t}^{ê}(d,ê_{t}(d))$$

$$G_{t}(d)\  = \ V_{t}^{BR}(d)\  - \ V_{t}^{ê}(d)$$

最后一期没有后续 continuation，因此 full dynamic deviation 与 one-step deviation 完全相同。

$$G_{T}(d)\  = \ \Delta_{T}(d)$$

## 2.4 [由 dReach 过渡到 full-domain MPE-oriented metric]{.mark}

旧版 dReach 在 BR-reachable state region 上逐 stage 取最大的 one-step deviation，再把这些最大值相加。它仍应保留，因为它提供可解释的 conservative reachable-state diagnostic。

$$d_{Reach}\  = \ \Sigma_{t}\ \ \max_{d \in Rₜᴮᴿ}\ \ \Delta_{t}(d)$$

[如果目标是 full-domain approximate MPE assessment]{.mark}，更直接的主量是：从任意 feasible stage/state 开始，完整动态偏离能够增加的最大 continuation payoff。

$$G_{\max}^{full}\  = \ \max_{t}\ \max_{d \in Dₜ}\ \lbrack V_{t}^{BR}(d)\  - \ V_{t}^{ê}(d)\rbrack$$

$$\frac{G_{\max}^{full}}{\Delta W}\  \leq \ \varepsilon_{MPE}$$

  ---------------------------------------------------------------------------------------------------------------------------------
  **指标**    **状态范围**       **偏离方式**                       **聚合**                          **新框架角色**
  ----------- ------------------ ---------------------------------- --------------------------------- -----------------------------
  EXP_root    root d₁=0          从 root 完整动态偏离               单一 root value gap               补充主路径诊断

  dReach      BR-reachable R_t   每个 state 做 one-step deviation   逐 stage 最大值相加               保留的 reachable diagnostic

  Δmax_all    完整 D_t           one-step deviation                 全 stage/state 最大值             局部 full-domain diagnostic

  Gmax_full   完整 D_t           从该 state 起完整动态偏离          全 stage/state 最大值             新的 MPE-oriented 主指标

  dFull       完整 D_t           one-step deviation                 逐 stage full-domain 最大值相加   保守累计诊断
  ---------------------------------------------------------------------------------------------------------------------------------

# 3. 为什么低偏离不等于高精度 policy recovery

旧 T=2 结果已经表明，策略可以满足较小的 strategic-deviation criterion，却仍有明显的 peak underestimation、positive-region RMSE 和 nonzero tail effort。这不是逻辑矛盾，而是 payoff geometry 与 action identification 的区别。

PPO 优化的是 expected payoff，而不是 analytical action distance。如果某些 state 的 payoff 对 effort 比较平坦，较大的 action error 可能只对应很小的 payoff loss。因此，降低 dReach 或 Gmax_full 可能改善 strategic quality，却不会自动保证 RMSE 或 peak error 同比例下降。

$$strategic\ accuracy\ \  \neq \ \ action\ recovery\ accuracy$$

[所以 T=2 新的 validation gate 必须同时包含两类指标。]{.mark}

  ---------------------------------------------------------------------------------------------------------
  **维度**            **建议指标**                       **解释**
  ------------------- ---------------------------------- --------------------------------------------------
  Policy recovery     Stage-1 relative error             初始 tied state effort 是否接近解析值

  Policy recovery     Stage-2 positive-region RMSE       理论 positive-effort region 内的 curve recovery

  Policy recovery     Peak relative error                d=0 处是否仍系统性低估峰值

  Policy recovery     Tail mean / max effort             理论 zero-effort tails 是否仍有残余投入

  Policy recovery     Symmetry error at terminal stage   最终阶段是否恢复解析 benchmark 的 even structure

  Strategic quality   Gmax_full/ΔW                       所有 feasible states 上最严重的动态偏离收益

  Numerics            dev-final differences              结论是否受 grid/quadrature resolution 影响
  ---------------------------------------------------------------------------------------------------------

# 4. 新的训练框架：[Stagewise TEL--PPO with Frozen Continuation]{.mark}

## 4.1 从 backward curriculum 到真正的 [backward learning]{.mark}

旧框架中的 curriculum 顺序本身是合理的，但在引入更早阶段后，later-stage actor 仍继续与 earlier-stage 数据一起更新。因此它是"按后向顺序扩展 joint training"，而不是真正的 stagewise backward learning。对于高精度 recovery，这种参数漂移可能破坏已经学好的 continuation。

新版本的核心调整是：每个 stage 有独立 actor 或等价的可冻结完整映射；后期 policy 达到 precision criterion 后冻结；更早阶段训练时，后期 continuation 仍参与 rollout 和 payoff calculation，但不再被 optimizer 改变。

$$learn\ stage\ T\ \  \rightarrow \ \ freeze\ \  \rightarrow \ \ learn\ stage\ T - 1\ \  \rightarrow \ \ freeze\ \  \rightarrow \ \ \ldots\ \  \rightarrow \ \ learn\ stage\ 1$$

## 4.2 Stage-wise residual 与动态偏离的关系

当后续 continuation 固定时，可以针对当前 stage 控制 full-domain one-step residual。令 η_t 表示该 stage 的 normalized residual budget：

$$\max_{d \in Dₜ}\ \Delta_{t}(d)\  \leq \ \eta_{t}\ \Delta W$$

有限时域下，完整动态偏离可以由后续 stage residual 的累计量控制。实际 verifier 还要额外考虑 discretization、interpolation、quadrature 和 action search error，因此不能把网格上的 residual 直接称作连续状态空间的严格证明。

$$G_{t}(d)\  \leq \ \Delta W\ \Sigma_{s = t,\ldots,T}\eta_{s}$$

## 4.3 Terminal reward variance reduction

当前 sampled terminal reward 使用一次 realized performance shock 决定高低奖。建议在 T=2 terminal-only **[pilot]{.mark}** 中加入 conditional expected terminal reward 作为单变量对照。给定 state 与双方 actions，直接使用已知 shock distribution 对 terminal prize 做条件期望。

$${r\bar{}}_{i,T}\  = \ W_{L}\  + \ \Delta W\ F_{\xi}(d_{T}\  + \ e_{i,T}\  - \ e_{j,T})\  - \ k{e^{2}}_{i,T}$$

$$E\lbrack r_{i,T}\ |\ state,\ actions\rbrack\  = \ {r\bar{}}_{i,T}$$

这不是把 closed-form equilibrium action 喂给 learner，而是减少 terminal reward estimator 的抽样噪声。第一轮只改变 reward estimator，不同时改变 network、learning rate、action mode 或 BR target。

## 4.4 Deterministic mean continuation 与 frozen continuation 的区别

deterministic mean 表示给定一个网络与状态，直接执行 Beta policy 的 conditional mean effort，而不再从 Beta distribution 抽 action；frozen 表示以后训练不再改变该 state-to-effort mapping。这两个概念必须分开做实验。

  ------------------------------------------------------------------------------------------------------------------
  **概念**                          **含义**                           **是否改变参数？**   **是否保留环境噪声？**
  --------------------------------- ---------------------------------- -------------------- ------------------------
  Stochastic continuation           后期从 Beta policy 抽 action       可变或冻结均可       是

  Deterministic mean continuation   后期直接用 mean effort             可变或冻结均可       是

  Frozen continuation               后期完整 policy mapping 不再更新   否                   是

  Frozen deterministic mean         固定 mapping 且执行 mean action    否                   是
  ------------------------------------------------------------------------------------------------------------------

[因果对照的顺序建议是：先保持 continuation action mode 不变，只比较 joint update 与 true freeze；确认 freeze 有效后，再单独比较 stochastic 与 deterministic mean continuation。]{.mark}

# [5. Revised T=2 Protocol]{.mark}

## 5.1 Phase T2-A: Terminal Precision Learning

-   只训练 Stage 2 actor；[从完整 D₂ 做]{.mark} state-balanced exploring starts。

-   baseline reward 与 conditional expected terminal reward 做 paired pilot。

-   development verifier 检查 max\_{d∈D₂} Δ₂(d)/ΔW；terminal stage 中 G₂(d)=Δ₂(d)。

-   不得因 budget exhaustion 自动把未达标 Stage-2 policy 当作合格 continuation。

-   development phase [可先探索 0.002、0.001、0.0005 等 residual level]{.mark}，正式 threshold 必须在 held-out confirmation 前锁定。

## 5.2 Phase T2-B: Freeze Stage 2

保存独立的 Stage-2 actor / output mapping。之后固定 grid 上的 output-drift test 应当为 0（允许浮点误差）。freeze 不能只冻结最后一个 head，而让 shared trunk 继续更新。

$$\max_{d \in D₂}\ |ê_{2,after}(d)\  - \ ê_{2,freeze}(d)|\  \approx \ 0$$

## 5.3 Phase T2-C: Stage 1 Learning

-   root start 为 (t=1,d=0)。

-   Stage 2 继续参与 rollout 与 continuation payoff，但其 actor 不更新。

-   PPO 只更新 Stage-1 actor；critic 可以继续利用完整 trajectory return。

-   training-time frozen continuation 不限制 final verifier 中 deviator 在 Stage 2 重新优化。

## 5.4 Phase T2-D: Freeze Whole Candidate and Final Evaluation

组合 ê={ê₁, ê₂} 并整体冻结。之后再用 locked final verifier 进行 analytical recovery、full-domain dynamic BR、dReach/EXP_root、dense concentration 与 numerical refinement。

## 5.5 建议的 T2 Validation Gate

下面 recovery thresholds 先作为 method-development targets，[通过 pilot 判断可达性后再锁定]{.mark}。

  --------------------------------------------------------------------------------------------------------
  **类别**               **候选标准/输出**                            **建议状态**
  ---------------------- -------------------------------------------- ------------------------------------
  Recovery               Stage-1 relative error ≤ 5%                  development target，待 pilot 校准

  Recovery               Stage-2 peak relative error ≤ 5%             development target，待 pilot 校准

  Recovery               positive-region normalized RMSE ≤ 5%         development target，待 pilot 校准

  Recovery               tail mean effort ≤ 2                         development target，待 pilot 校准

  Strategic              Gmax_full/ΔW ≤ 0.01                          首选 overall tolerance；正式前锁定

  Numerical              Gmax_full dev-final refinement               必须新增正式阈值

  Verifier calibration   exact equilibrium ≈ numerical floor          必须

  Discrimination         misspecified policies 有明显更大 deviation   必须

  Reliability            fresh held-out seeds 可重复                  必须
  --------------------------------------------------------------------------------------------------------

# 6. Revised T=3 Protocol

只有在 v2 T=2 validation gate 通过并锁定 solver 后，才进入新的 T=3。T=3 不应继续沿用"后期没有学好也 budget-forced 进入下一阶段"的逻辑。

  --------------------------------------------------------------------------------------------------------------------------------------------------
  **阶段**   **Trainable actor**   **Frozen continuation**   **Development criterion**                    **失败处理**
  ---------- --------------------- ------------------------- -------------------------------------------- ------------------------------------------
  T3-A       Stage 3               无                        terminal full-domain residual                预算耗尽仍未达标 → stage-3 failure，停止

  T3-B       Stage 2               Stage 3                   max\_{d∈D₂} G₂(d)/ΔW                         未达标 → stage-2 failure，停止

  T3-C       Stage 1               Stages 2--3               root/full-policy strategic check             未达标 → no candidate

  Final      无，全部冻结          全部冻结                  Gmax_full + dReach + EXP_root + refinement   只做验证，不反向调参
  --------------------------------------------------------------------------------------------------------------------------------------------------

## 6.1 T3 的 full-domain verification

[最终 candidate 的主要 MPE-oriented assessment 使用 full-domain dynamic continuation gain]{.mark}。dReach 不删除，而是作为 reachable-state accumulated diagnostic 保留。

$$G_{\max}^{full}\  = \ \max_{t = 1,2,3}\ \max_{d \in Dₜ}\ \lbrack V_{t}^{BR}(d)\  - \ V_{t}^{ê}(d)\rbrack$$

# 7. 旧计划实现中的问题与新目标升级点

[修正现有代码的protocol 一致性、训练组织和高精度目标]{.mark}。

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **问题**                                **类型**                **旧代码情况**                                                **v2 改进**
  --------------------------------------- ----------------------- ------------------------------------------------------------- -------------------------------------------------------------------------
  Phase A fixed-budget 语义不完全一致     protocol mismatch       文档称 fixed-budget，但 T2 runner 仍保留可提前 exit 的 gate   明确 phase_mode=fixed_budget 或 precision_gate；删除无效双重语义

  T2/T3 verifier warm-up 不同             protocol mismatch       T2 有 warm-up；T3 B/C 实际从 update 25 开始检查               manifest 显式保存 first_check/check_every/trigger

  旧 runner defaults 残留                 reproducibility risk    旧 threshold/k_stop 可能被误运行                              单独 versioned v2 entry point；拒绝 stale config

  所有 stage 共用一个 actor               design limitation       earlier-stage gradients 会改变 later-stage policy             stage-specific actor bank + freeze_stage()

  无 explicit trainable-stage mask        design limitation       active stages 主要由 rollout 结构间接控制                     collector 保存 stage_id/trainable_mask；optimizer 只更新目标 stage

  budget-forced continuation              objective mismatch      未学好 later stage 仍可进入更早阶段                           v2 formal 中未过 precision gate 则停止并标记 stage failure

  KL 仅监测                               optimization risk       记录 approx KL，但不阻止同一 batch 后续 PPO epochs            先诊断；必要时加入 target-KL actor early stop

  Beta sample clipping likelihood         numerical/policy risk   clip 后仍按普通 Beta log-prob                                 新增 lower/upper clip fraction；precision phase 必要时改一致 likelihood

  dReach reach mask 与插值 PMF 支持口径   verifier rigor          可能存在数值支持边界差异                                      新增 PMF-outside-reach / bound consistency test

  recovery 仅 post-hoc                    objective mismatch      不参与 method validation                                      提升为 T2 validation gate

  Gmax_full 未成为正式主指标              objective mismatch      旧 final 主要看 dReach                                        保存完整 G_t(d) arrays、stage maxima 与 full maximum

  branch 只 load weights 不够             experiment-control      optimizer/RNG 不一定恢复                                      paired branch 保存完整 optimizer + RNG + snapshot state
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 7.1 旧结果如何处理

新的 v2 protocol 一旦根据这些结果开发，就必须使用 fresh seeds 做独立 confirmation。

# 8. 实验执行顺序

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **步骤**   **实验设计**                                                                               **主要问题**                                                      **进入下一步条件**
  ---------- ------------------------------------------------------------------------------------------ ----------------------------------------------------------------- --------------------------------------------------------
  P0         Baseline code correction / regression                                                      旧 protocol implementation 是否干净可复现？                       core tests + numerical audit 通过

  P1         T2 terminal-only: sampled vs conditional expected reward；2 q × 3 paired seeds × 2 modes   降低 terminal reward noise 是否改善 Stage-2 recovery？            至少在 recovery/strategic 指标上出现稳定改善或明确无效

  P2         同一 terminal checkpoint 分支：joint update vs true frozen continuation                    跨 stage 更新是否破坏已学好的 Stage 2？                           选择更稳定的 continuation scheme

  P3         precision target study                                                                     多严格的 stage residual 才能得到稳定 action recovery？            锁定可达且有解释力的 target

  P4         optional target-KL / variance reduction                                                    只有 diagnostics 指向 PPO over-update 时才测试                    单变量 ablation 支持

  P5         Freeze v2 protocol                                                                         所有 architecture / budgets / thresholds / grids / cadence 固定   协议版本锁定

  P6         Fresh T2 confirmation                                                                      新 seeds；不再调参                                                T2 validation gate 通过

  P7         T3 implementation pilot                                                                    fresh diagnostic seeds                                            stagewise solver 在 T3 能产生低偏离 candidate

  P8         T3 formal                                                                                  fresh formal seeds                                                报告全部 runs，包括失败

  P9         HCC reporting                                                                              policy functions + strategic verification + reliability           形成最终 manuscript evidence
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 8.1 推荐的 P1 最小实验规模

$$2\ q\ values\  \times \ 3\ paired\ diagnostic\ seeds\  \times \ 2\ reward\ modes\  = \ 12\ terminal - only\ runs$$

这一轮不因为第一次达到旧 1% dReach criterion 就提前停止。所有组应在相同计算预算或相同 precision-gate 规则下比较，主要输出为 Stage-2 RMSE、peak error、tail effort、max Δ₂/ΔW、KL/clip diagnostics 和 training cost。

# 9. Decision Rules：如何根据 pilot 结果决定下一步

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------
  **观察结果**                                         **解释**                                       **下一步**
  ---------------------------------------------------- ---------------------------------------------- -------------------------------------------------------------
  Expected terminal reward 显著改善 Stage-2 recovery   terminal reward variance 是重要因素            纳入 v2；再测试 frozen continuation

  Reward estimator 无改善                              主要问题不是 terminal prize noise              保持 sampled reward，继续 P2

  Frozen continuation 明显优于 joint update            跨 stage parameter drift 是关键机制            采用 stage-specific actors + true freeze

  Freeze 不改善                                        不能把 failure 主要归因于 continuation drift   进一步看 KL、critic/advantage、policy capacity

  Gmax_full 很小但 action RMSE 仍大                    payoff flatness / weak action identification   继续以 recovery target 开发，但不错误宣称 strategic failure

  Action recovery 好但 Gmax_full 大                    局部 curve 接近不等于 sequential optimality    检查 off-path states / earlier-stage continuation

  T2 held-out gate 不通过                              solver 仍未被验证                              不进入正式 T3；回到 development

  T2 gate 通过但 T3 pilot 失败                         可能是 horizon-specific search difficulty      只在 pilot 中继续定位；不直接 formal rerun
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------
