> 公开归档说明：原始服务器路径已改为可移植路径。密集策略表以全部种子的稀疏曲线代替，访问表保留代表性玩家的根状态自对弈分布；完整原始数组未随仓库提供。所有策略均为未认证的诊断策略。

# T=3 正式实验记录：固定预算下的搜索、验证与政策诊断


日期：2026-09-26。研究对象为q=50与q=60，各20个预先指定且未用于调试的独立随机种子。本记录仅使用formal cohort，所有40个种子均纳入；不合并pilot或后续诊断实验。

[40个种子的逐项附表](T3_FORMAL_EXPERIMENT_RECORD_20260926_tables.md) · [完整精度汇总CSV](T3_FORMAL_EXPERIMENT_RECORD_20260926.csv) · [原始正式报告](../../experiments/three_stage_implementation_pilot_20260924/reports/FORMAL_REPORT.md)

## 实验设定与结果概述


本实验评估既定随机数值求解器能否在预先规定的预算内找到并认证三阶段候选策略。q50使用11001–11020，q60使用11101–11120。经济参数为w_h=6、w_l=2、ΔW=4、k=1/3500，努力范围[0,100]。采用共享对称策略及3 → (2,3) → (1,2,3)的训练顺序。

| Phase | 训练内容 | 每次update起点组成 | 预算与检查 |
| --- | --- | --- | --- |
| A | Stage 3 terminal pretraining | 512 ES3 | 固定400；每100诊断 |
| B | Stages 2–3 continuation | 512 ES2，均继续到Stage 3 | 上限600；每25检查；连续3次eligible退出 |
| C | 完整三阶段candidate search | 256 root + 85 ES2 + 171 ES3 | 上限1800；每25检查；first eligible freeze |

D2/D3分别为q50的[-200,200]/[-400,400]、q60的[-220,220]/[-440,440]；bin width=10。训练与认证设置沿用运行前选定的formal_settings，不引入A800、center sampling、K8、额外restart筛选或事后预算扩展。

B的eligible要求全D2动态continuation gain最大值除以ΔW≤0.02，并且Stages 2–3的normalized std≤0.04；C要求development dReach/ΔW≤0.01和三个stage的normalized std≤0.04。两者均要求verifier有效。A为固定预算预训练，不以诊断值早停。

40个run全部正常完成，但两个q都没有发现candidate。四类结果的收集与报告均完成；求解器在本次预算及判据下没有获得可认证候选策略。

## 1. Search


| q | 预定/完成runs | candidate / 全部runs | certified / candidates | certified / 全部runs | discovery与end-to-end的95% Wilson区间 |
| --- | --- | --- | --- | --- | --- |
| 50 | 20 / 20 | 0 / 20 | N/A（无candidate） | 0 / 20 | [0, 0.1611] |
| 60 | 20 / 20 | 0 / 20 | N/A（无candidate） | 0 / 20 | [0, 0.1611] |

所有run的A均在400结束，B均在600以budget_forced退出，C均在1800以no_candidate_budget_exhausted结束。没有verifier_passed的B，也没有任何C eligible检查。每个run实际停止于global update 2800，完成1,433,600 episodes及2,815,400 joint environment steps。

全批合计57,344,000 episodes、112,616,000 joint environment steps。一个joint step指一个episode中的一次同时行动阶段转移；不将两名玩家的动作重复计为两个环境步。经济评估及verifier计算不计入训练步数。

训练中共调用development verifier 4000次：A阶段160次、B阶段960次、C阶段2880次，全部valid。q50的480次B检查中战略指标从未≤0.02；q60有10次≤0.02，但对应normalized std为0.04045–0.04324，高于0.04，因此eligible仍为零。C的2880次检查中，dReach/ΔW从未≤0.01。

40个进程均exit code 0，执行错误、中断、重跑及跳过均为0；完整性检查40/40通过。全局最多10个单线程worker，训练批次实际耗时84分31秒，单run峰值内存约0.5 GiB。已有正式执行记录包含60个单元测试通过。

观察到的0/20是该配置在本批独立初始化中的经验结果。Wilson区间描述有限样本不确定性，不意味着总体成功概率已被证明等于零。conditional certification没有候选作为分母，因此为N/A，而非0%。

## 2. Verification


验证对象为每个run实际C1800终点保存并重新载入的mean policy；全部标记为diagnostic_terminal。对手冻结后，final verifier重新计算完整三阶段动态最佳响应。development使用state_step=4、effort_step=1、gl_half=16；final使用2、0.5、32。

三个收益指标分别为：

\[
EXP_{root}=V_1^{BR}(0)-V_1^{mean}(0),\qquad
d_{Reach}=\sum_{t=1}^{3}\max_{d\in R_t}\delta_t(d),\qquad
\Delta_{\max}^{all}=\max_{t,d\in D_t}\delta_t(d).
\]

其中δ_t是当前阶段单步偏离、以后回到mean policy的收益增益；R_t是BR interval-reachable集合。EXP_root是从root出发的完整动态偏离收益，不是Stage-1单步偏离。Δ_max_all是最大值，不能与各阶段全域最大值之和dfull混用。下表均除以ΔW；raw收益值在CSV中同时保留。

| q | 实际终点final指标／refinement差 | 跨seed均值 | 中位数 | 范围 |
| --- | --- | --- | --- | --- |
| 50 | EXP_root/ΔW | 0.02062 | 0.01788 | [0.00847, 0.04858] |
| 50 | dReach/ΔW | 0.05987 | 0.05375 | [0.02403, 0.14479] |
| 50 | Δ_max_all/ΔW | 0.03583 | 0.02936 | [0.01436, 0.09446] |
| 50 | abs(final−dev dReach)/ΔW | 0.000402849 | 0.000128857 | [8.07968e-06, 0.00233056] |
| 50 | abs(final−dev EXP_root)/ΔW | 4.8468e-05 | 3.13003e-05 | [3.1013e-07, 0.000166122] |
| 60 | EXP_root/ΔW | 0.01329 | 0.01312 | [0.00431, 0.02717] |
| 60 | dReach/ΔW | 0.03910 | 0.03653 | [0.02055, 0.08238] |
| 60 | Δ_max_all/ΔW | 0.02305 | 0.02282 | [0.01268, 0.04902] |
| 60 | abs(final−dev dReach)/ΔW | 8.47764e-05 | 5.52838e-05 | [3.35041e-06, 0.000452229] |
| 60 | abs(final−dev EXP_root)/ΔW | 3.36626e-05 | 2.19581e-05 | [2.11797e-06, 0.000114158] |

| 检查 | q50 | q60 | 解释 |
| --- | --- | --- | --- |
| dev/final validity | 20/20 | 20/20 | 两层均完成数值有效性检查 |
| main：final dReach/ΔW≤0.01 | 0/20 | 0/20 | 全部高于预定战略阈值 |
| dReach refinement≤0.002 | 19/20 | 20/20 | q50 s11008未通过 |
| EXP_root refinement≤0.002 | 20/20 | 20/20 | 最大差0.0001661 |
| dense normalized std≤0.04 | 20/20 | 20/20 | 全D1–D3，step=0.05；最大0.03523 |
| numeric_thresholds_pass | 0/20 | 0/20 | 所有run至少main失败 |
| final_joint_pass | 0/20 | 0/20 | 无candidate且未满足全部数值条件 |
| certification | N/A | N/A | 所有run为not_applicable_no_candidate |

**s11008的refinement例外。** q50 s11008的abs(final−dev dReach)/ΔW=0.0023306，超过0.002。保存的数组显示，Stage-2可达集合在dev网格为[-104,92]，在final网格为[-106,92]；两层的最大reachable偏离都位于左端。更细网格使Stage-2贡献增加0.0021682，结合其他阶段变化后形成上述总差。这是一项真实的refinement未通过，两层verifier仍valid，结果予以保留。该run没有candidate，因此不改变“无可认证候选”的结论。

**收益偏离的量级。** root动态收益偏离相对奖品差为q50的0.85%–4.86%、q60的0.43%–2.72%；均值分别为2.06%和1.33%。可以将其描述为相对奖品差处于较小百分比量级，但不能据此笼统认定所有战略偏离很小或已经接近认证。终点dReach/ΔW为q50的2.40%–14.48%、q60的2.05%–8.24%，全部高于1%判据。root收益偏离、可达状态最大偏离的累加和全域最大偏离回答不同的问题。

认证规则也要求candidate存在。此处不是“40个candidate被final拒绝”，而是搜索阶段没有产生candidate，终点验证仅提供诊断。结果不构成全域MPE认证。

## 3. Economic policies


本节描述全部40个C1800诊断终点。certified candidate与uncertified candidate两组均为空，diagnostic_terminal每q各20个。所有跨seed汇总保留s11008，没有排除表现差的run。

### 3.1 策略曲线与平局状态努力


保存了完整的ê1(0)、ê2(d)、ê3(d)，其中Stage 2/3曲线在全D2/D3上以0.05间距评估，含0和端点。下表给出平局状态的跨seed均值±样本标准差；这个标准差反映不同训练seed的差异，不是模拟MCSE。

| q | ê1(0) | ê2(0) | ê3(0) |
| --- | --- | --- | --- |
| 50 | 44.12 ± 5.95 | 52.44 ± 7.59 | 57.68 ± 10.09 |
| 60 | 33.80 ± 4.97 | 44.14 ± 5.90 | 48.33 ± 8.01 |

完整曲线见 [policy_profiles.csv（公开版：稀疏策略曲线，含对手均值）](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/policy_curves_compact.csv)，使用checkpoint_kind=diagnostic_terminal筛选实际终点，避免混入minimum-development曲线。原图：[q50策略曲线](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/figures/policy_curves_q50.pdf)、[q60策略曲线](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/figures/policy_curves_q60.pdf)。

q50 s11008的政策几乎平坦：Stage-2全域effort range为2.279，Stage-3全域为1.274，Stage-3中心|d|≤B内为0.636。其余39个run的中心effort range在Stage 2、Stage 3分别至少为34.344、34.041。该观察描述政策形状，不能独立解释训练失败的原因。

### 3.2 Stage-wise expected effort


从root状态d=0开始独立self-play评估。mean-policy与stochastic-policy分开，每个run、每种mode均为3×200,000 episodes。阶段期望努力使用代表性玩家口径X_t=(e_{0,t}+e_{1,t})/2。下表为20个seed间的均值±样本标准差，单run的MCSE另列在附表和CSV。

| q | self-play mode | E[e1] | E[e2] | E[e3] |
| --- | --- | --- | --- | --- |
| 50 | mean | 44.12 ± 5.95 | 31.09 ± 3.42 | 30.86 ± 3.62 |
| 50 | stochastic | 44.12 ± 5.95 | 31.02 ± 3.42 | 30.80 ± 3.60 |
| 60 | mean | 33.80 ± 4.97 | 26.54 ± 3.67 | 29.88 ± 4.15 |
| 60 | stochastic | 33.80 ± 4.97 | 26.50 ± 3.67 | 29.84 ± 4.15 |

例如，q50 mean-policy的E[e2]跨seed标准差为3.418，而单run rollout MCSE最大仅0.0222。两种不确定性需分别解释。q50与q60的均值差异描述本批未认证训练终点，不能当作已验证均衡的比较静态。

### 3.3 State visitation


训练访问分布按phase、start stage和current stage记录；mean/stochastic root self-play与BR-chain节点分布单独保存。BR-chain PMF不替代self-play visitation。以下用已有pre-action physical gap的标准差概括self-play分布：先对各run计算gap SD，再对20个seed取均值，括号为这些run的最小与最大值。

| q | mode | Stage-2 gap SD：均值[范围] | Stage-3 gap SD：均值[范围] |
| --- | --- | --- | --- |
| 50 | mean | 40.829 [40.785, 40.891] | 67.534 [57.475, 75.063] |
| 50 | stochastic | 41.040 [40.962, 41.148] | 67.758 [57.643, 75.413] |
| 60 | mean | 48.987 [48.937, 49.066] | 74.954 [69.410, 85.687] |
| 60 | stochastic | 49.130 [49.026, 49.237] | 75.160 [69.555, 85.890] |

Stage 1的gap固定为0。完整分布见 [state_visitation.csv（公开版：根状态自对弈访问）](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/state_visitation_compact.csv)；图见[q50访问分布](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/figures/state_histogram_q50.pdf)、[q60访问分布](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/figures/state_histogram_q60.pdf)。

### 3.4 Leader/follower asymmetry


状态曲线的不对称定义为ê_t(x)−ê_t(−x)，x>0；占用分布下的统计采用同一episode内领先者减落后者的配对努力差，排除平局。下面给出mean-policy的跨seed统计。

| q | stage | 均值±seed SD | 范围 | 正差seeds | 负差seeds |
| --- | --- | --- | --- | --- | --- |
| 50 | 2 | 13.947 ± 9.166 | [-2.821, 24.759] | 18 | 2 |
| 50 | 3 | 3.347 ± 3.532 | [-1.653, 10.087] | 16 | 4 |
| 60 | 2 | 7.858 ± 7.571 | [-6.395, 24.498] | 17 | 3 |
| 60 | 3 | 0.041 ± 3.159 | [-8.872, 3.931] | 11 | 9 |

Stage 1全部为tie，条件leader/follower统计为N/A，不能记成“差为0”。q60 Stage 3的跨seed平均差约0.041，同时存在11个正差与9个负差seed，平均接近零不表示每条策略都没有不对称。stochastic对应统计、每项MCSE及完整曲线见附表、CSV与 [policy_asymmetry.csv（公开版：稀疏策略曲线，含对手均值）](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/policy_curves_compact.csv)。

## 4. Failure diagnosis


所有40个run均为no-candidate。每个run另外选取其C阶段最小valid development dReach检查点，仅用于诊断；它未被晋升为candidate，也未替代实际C1800终点的final评估。三个stage贡献必须来自同一检查点，不能分别挑选各stage的历史最小值。

| q | 每run min dReach/ΔW的均值 | 中位数 | 范围 | 同检查点t1贡献均值 | t2贡献均值 | t3贡献均值 |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.03572 | 0.03069 | [0.01543, 0.11678] | 0.000993 | 0.015746 | 0.018976 |
| 60 | 0.02586 | 0.02435 | [0.01557, 0.04573] | 0.000806 | 0.012473 | 0.012585 |

最小development dReach仍全部高于0.01。终点development dReach是各run历史最小值的1.00–4.83倍；例如q50 s11002在global update 2200达到0.01892，终点development值却为0.09137。因此不能把过程中曾经下降写成持续收敛。

| q | min-C最大reachable偏离在t2/t3 | min-C全域最大偏离在t2/t3 | B退出 |
| --- | --- | --- | --- |
| 50 | 7 / 13 | 7 / 13 | 20/20 budget_forced |
| 60 | 9 / 11 | 10 / 10 | 20/20 budget_forced |

合计，BR-reachable最大偏离24/40在Stage 3、16/40在Stage 2；按全域最大偏离则为23/40和17/40。差异来自q60 s11103：其reachable最大在t3,d=124，全域最大在t2,d=208，后者位于当时R2之外。40个run在minimum-C处的Stage-1贡献均≤0.004561。

最大reachable偏离状态所在宽度10的bin，其截至该检查点的累计训练访问量为q50的14,355–50,998、q60的14,828–36,351。这里统计的是所在bin的访问量，并非连续状态点被精确访问的次数。所有直接ES bins均非空；这些描述未显示缺失bin，但不能证明当前训练曝光充分，也不能据此排除其他优化问题。

B存在战略指标与集中度不能同时满足的情况；C的每次失败都含战略判据失败，concentration从未单独挡住一次本可eligible的C检查。终点的40条策略都通过dense concentration检查，但仍未达到dReach阈值。失败分解并不把原因归结为某一项未经干预验证的机制。

## 5. 总结


本次正式实验完成了预先指定的40个独立初始化，在q50和q60下分别获得0/20的candidate-discovery和end-to-end success；由于没有candidate，conditional certification无定义。全部run完整执行A400/B600/C1800，Search、Verification、Economic policies和Failure diagnosis均有记录。实现与数值链条的运行完整性不等于均衡认证成功。

终点root动态收益偏离相对奖品差处于约0.43%–4.86%的量级，其中q50、q60的中位数分别为1.79%和1.31%；但全部run的可达状态累计偏离dReach仍超过1%阈值。39/40通过dev–final refinement，40/40通过dense concentration，这些检查不能代替战略判据。记录因而应表述为“固定预算下未找到满足预定判据的候选策略”，不声称every-seed convergence，也不据此否定均衡的存在。

经济表展示了这些未认证终点的策略、努力与访问分布，包含明显的seed间差异。它们可以用于描述求解器输出及失败形态，不能作为已认证均衡政策的证据。本轮正式实验的执行及四类结果交付已经完成；本记录不启动或建议追加实验。

## 数据来源与复核说明


本文从已保存的正式CSV和数组核对、汇总，不重新训练、不改变verifier、阈值或候选选择规则。所有收益量保留raw与/ΔW口径；正文显示值适度四舍五入，精确值见汇总CSV。跨seed标准差、单run MCSE和Wilson区间各有不同分母，未混用。

| 内容 | 原始文件 |
| --- | --- |
| 运行与预算 | [runs.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/runs.csv)；[phases.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/phases.csv)；[rates.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/rates.csv) |
| 验证 | [verification_by_seed.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/verification_by_seed.csv)；[verifier_stage_metrics.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/verifier_stage_metrics.csv) |
| 经济政策 | [policy_profiles.csv（公开版：稀疏策略曲线，含对手均值）](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/policy_curves_compact.csv)；[stage_metrics.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/stage_metrics.csv)；[economics_by_group.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/economics_by_group.csv) |
| 访问与不对称 | [state_visitation.csv（公开版：根状态自对弈访问）](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/state_visitation_compact.csv)；[policy_asymmetry.csv（公开版：稀疏策略曲线，含对手均值）](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/policy_curves_compact.csv) |
| 失败诊断 | [failure_diagnostics.csv](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/failure_diagnostics.csv) |
| 完整性 | [completeness.json](../../experiments/three_stage_implementation_pilot_20260924/reports/formal/completeness.json) |

原AUTO_SUMMARY中的“解释见PILOT_REPORT”是模板文字，正式解释以 [FORMAL_REPORT.md](../../experiments/three_stage_implementation_pilot_20260924/reports/FORMAL_REPORT.md) 和本记录为准。
