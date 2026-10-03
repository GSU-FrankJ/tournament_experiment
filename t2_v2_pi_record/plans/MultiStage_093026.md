我们现在主要的目标是提高two-stage实验的准确率（recovery和verification），确保可以作为算法开发和校准的环境，然后进行three-stage的实验。

1.  新方法的主要改动：

b\. Full- domain MPE（grid-based）

> 从任意 feasible stage/state 开始，完整动态偏离能够增加的最大 continuation payoff。

$${\widehat{G}}_{\max}^{full}\  = \ \max_{t}\ \max_{d \in Dₜ}\ \lbrack V_{t}^{BR}(d)\  - \ V_{t}^{ê}(d)\rbrack$$

> $$\frac{{\widehat{G}}_{\max}^{full}}{\Delta W}\  \leq \ \varepsilon_{MPE}$$

训练中请依旧保存以下数据：

  ---------------------------------------------------------------------------------------------------------------------------------
  **指标**    **状态范围**       **偏离方式**                       **聚合**                          **新框架角色**
  ----------- ------------------ ---------------------------------- --------------------------------- -----------------------------
  EXP_root    root d₁=0          从 root 完整动态偏离               单一 root value gap               补充主路径诊断

  dReach      BR-reachable R_t   每个 state 做 one-step deviation   逐 stage 最大值相加               保留的 reachable diagnostic

  Δmax_all    完整 D_t           one-step deviation                 全 stage/state 最大值             局部 full-domain diagnostic

  Gmax_full   完整 D_t           从该 state 起完整动态偏离          全 stage/state 最大值             新的 MPE-oriented 主指标

  dFull       完整 D_t           one-step deviation                 逐 stage full-domain 最大值相加   保守累计诊断
  ---------------------------------------------------------------------------------------------------------------------------------

b\. joint training改为stagewise backward learning：

$$learn\ stage\ T\ \  \rightarrow \ \ freeze\ \  \rightarrow \ \ learn\ stage\ T - 1\ \  \rightarrow \ \ freeze\ \  \rightarrow \ \ \ldots\ \  \rightarrow \ \ learn\ stage\ 1$$

每个 stage 有独立 actor 或等价的可冻结完整映射；后期 policy 达到 precision criterion 后冻结；更早阶段训练时，后期 continuation 仍参与 rollout 和 payoff calculation，但不再被 optimizer 改变。

2.  [Pilot实验（三组分开做，分析见上一份文件）：]{.mark}

```{=html}
<!-- -->
```
a.  为了下一步的比较，先明确one-step deviation gain的计算：

$$\Delta_{2}(d) = \max_{e \in \left\lbrack 0,100 \right\rbrack}Q_{2}^{\widehat{e}}\left( d,e \right) - Q_{2}^{\widehat{e}}\left( d,{\widehat{e}}_{2}(d) \right)$$

在stage 2， $G_{2}(d) = \Delta_{2}(d)$

令 η_2 表示该 stage 的 normalized residual budget：

$$\max_{d \in Dₜ}\ \Delta_{2}(d)\  \leq \ \eta_{2}\ \Delta W$$

b- 第一个pilot实验：进行一个小规模的 **[T=2 terminal-only]{.mark} pilot**，这个阶段只训练stage 2，暂不训练 stage1，看一下降低 terminal reward 的抽样噪声是否能够提高 Stage 2 effort policy 的学习准确率。

用 paired seeds 比较 sampled reward 和 conditional expected reward。conditional expected reward：给定 state 与双方 actions，直接使用已知 shock distribution 对 terminal prize 做条件期望。

$${r\bar{}}_{i,T}\  = \ W_{L}\  + \ \Delta W\ F_{\xi}(d_{T}\  + \ e_{i,T}\  - \ e_{j,T})\  - \ k{e^{2}}_{i,T}$$

$$E\lbrack r_{i,T}\ |\ state,\ actions\rbrack\  = \ {r\bar{}}_{i,T}$$

其他设置全部不变。[比较 peak error、RMSE、tail effort 和]{.mark} $\eta_{2}$。

实验设置两个 treatment。

**Baseline:** 使用现有 sampled terminal reward。每个 episode 实际抽取 performance shocks，并根据最终输赢确定W_H 或 W_L。

**Treatment:** 使用 conditional expected terminal reward。给定当前 score gap 和双方已经抽取的 effort，直接根据已知 shock distribution 计算 terminal prize 的条件期望。

两种方法的 expected payoff 相同，区别只在于 Treatment 去掉了最后一次 performance shock 带来的 prize sampling noise（也就是把最后一步的 Monte Carlo noise 积分掉了）。这一轮实验**只改变 terminal reward estimator**，其他设置保持完全相同。

c- 第二个pilot 实验：[frozen continuation]{.mark}是否可以避免在训练stage1的时候破坏之前学到的结果。

选择上面的实验中estimator结果好的一种，作为后续development 的 reward方式，比较joint update和frozen stage 2

从同一个已经学好的 Stage-2 checkpoint 分成两个 branch：

-   Branch A：训练 Stage 1 时，Stage 1 和 Stage 2 都继续更新；

-   Branch B：冻结 Stage 2，只更新 Stage 1。

然后比较：

-   Stage 2 recovery 是否漂移；

-   Stage 1 recovery；

-   $G_{\max}^{full}$；

-   dReach / EXP_root；

-   frozen-stage output drift。

d- Pilot 3: [Continuation action mode]{.mark}，比较stochastic continuation 和deterministic mean continuation

这个实验是要分析Stage 2 已经固定后，训练 Stage 1 时，Stage 2 action 是采用随机抽样还是直接使用 Stage 2 policy mean，看一下哪一种能提高 Stage 1 的准确性和稳定性。

在前面两个 pilot 已经确定下来的方法基础上：对每个 seed，保存同一个 Stage-2 checkpoint，然后从这个 checkpoint 分成两个 branch。Stochastic 这组是从freeze的beta policy中抽effort，另外一组直接使用Beta distribution的mean。

3.  如果pilot实验可以的到更高的准确率，我们就可以正式进行two-stage的实验。
