# T=3 科学实施协议（2026-09-24）

这是原实施计划中与科学设置、测量和判据有关的部分。内部执行流程已移除。可移植运行方式见 [实验 README](../../experiments/three_stage_implementation_pilot_20260924/README.md)。实际 formal 配置见 [formal settings](../../experiments/three_stage_implementation_pilot_20260924/reports/FORMAL_SETTINGS.md) 与保存的 manifests。原始输出字典描述完整实验；公开归档仅保留 README 列出的紧凑结果。

## 1. 先固定本次 debug 的明确设置

### 1.1 Game、domains 和网格

沿用 w_h=6、w_l=2、k=1/3500、effort∈[0,100]、T=3。DW=w_h−w_l=4；B=(e_max−e_min)+2q；D_t=[−(t−1)B,(t−1)B]。从 GameSpec 计算派生量，不从模板复制旧值。

| 值 | q60 | q50 |
|---|---:|---:|
| B | 220 | 200 |
| D2 | [−220,220] | [−200,200] |
| D3 | [−440,440] | [−400,400] |
| bin width | 10 | 10 |
| D2/D3 bins | 44 / 88 | 40 / 80 |
| dev state grid 点数 t1/t2/t3 | 1 / 111 / 221 | 1 / 101 / 201 |
| final state grid 点数 | 1 / 221 / 441 | 1 / 201 / 401 |
| dense step 0.05 总点数 | 26403 | 24003 |

D1={0}，不得对其调用普通区间 ES sampler。observations 继续用 tau=(t−1)/(T−1)，d_norm=d/((t−1)B)，root d_norm=0。网络 float32；gaps、rewards、数值 verifier、统计累加 float64。

Development：state_step=4、effort_step=1、gl_half=16。
Final：state_step=2、effort_step=0.5、gl_half=32。
保留 grid_tol=1e−9、mass_tol=1e−10、pdl_tol_over_dw=1e−10、既有 GL 归一化要求。不得用粗网格替代 formal final。

### 1.2 三个 phase，只有三个 phase

| Phase | Active stages | 每 update starts | cap | dev cadence | 退出 |
|---|---|---|---:|---|---|
| A | [3] | 512 × ES3 | 400 | local 100,200,300,400，诊断用途 | 固定 400，fixed_budget_completed |
| B | [2,3] | 512 × ES2，全部继续到 t3 | 600 | local 25,50,…,600 | 3 次连续 eligible；否则 budget_forced |
| C | [1,2,3] | 256 root + 85 ES2 + 171 ES3 | 1800 | local 25,50,…,1800 | 第一次 eligible 立即保存并停止；否则 no_candidate |

明确移除旧 B1。C=1800 是用户选择的 debug 设置，将旧 B1 的 800 与旧 C 的 1000 合并，总 cap=2800。

B/C 从 local=25 开始检查，不保留旧 warmup=100 或 stability-triggered 检查。因此 B 最早在 75 退出，C 最早在 25 发现 candidate。所有检查在对应 update 完成后执行。cap 本身是 25 的倍数，不再额外重复 phase-end verifier。

可保留每 20 updates 的 drift/KL 诊断，但它不触发 verifier、不影响退出。A diagnostics 的 favorable metrics 也不允许 early exit。B invalid 或任一失败检查将 consecutive_eligible 归零；B budget-forced 仍进入 C，记录真实原因。

Eligible：

- B strategic：max_{d∈D2}[V2_BR(d)−V2_mean(d)]/DW ≤ 0.02。
- B concentration：在 dev grid 的 stages 2、3 上 max(std_effort/e_range) ≤ 0.04。
- C strategic：dReach_dev/DW ≤ 0.01。
- C concentration：在 dev grid 的 stages 1、2、3 上同一上界 ≤ 0.04。
- 以上都要求 verifier valid、concentration valid、相关数值 finite。
- A 记录 stage3 continuation gain 和 concentration，但无 eligibility exit。

可继续调用通用全 T=3 verifier 再取 stage2 值作为 B 指标；backward recursion 的 V2 本身不依赖 stage1。不能用 stage2 单步 delta 替代 V2_BR−V2_mean。也不要擅自改变通用 validity 定义来让 B 通过。

### 1.3 其余训练参数保持现有值

PPO：hidden=64（现有两层 tanh 网络），lr=3e−4 constant，Adam betas=(0.9,0.999)、eps=1e−8、weight_decay=0；
clip_eps=0.2、value_coef=0.5、entropy_coef=0、max_grad_norm=0.5；
epochs=10、minibatch=256、gamma=1、gae_lambda=1、c_min=100；
mu_clamp=action_clamp=1e−6、adv_norm_eps=1e−8。

不新增 annealing；跨 phase 保留 actor、critic 与 Adam state。初始化、phase entry 与每 global update 的 20 倍数刷新 frozen opponent，顺序沿用旧 runner：完成 PPO update 后 periodic refresh。每 global update 25 倍数可沿用 weights 导出，但 candidate/minimum 的保存必须独立于这个导出 cadence。

保留训练六个 RNG namespaces：init=0、env_noise=1、learner_action=2、opponent_action=3、starts_roles=4、minibatch=5。SeedSequence([seed,q,namespace])；角色与 ES sampler 继续用 starts_roles 流。增加日志/评估不能消耗这些随机流。

默认 CPU、torch_threads=1，OMP/MKL/OPENBLAS_NUM_THREADS=1。pilot 最多 3 个并发 worker。

### 1.4 预先指定 seeds

本计划生成时扫描现有 manifests 未见下列 pilot/formal seeds 冲突；执行前必须再查，尚未消耗的 seed 不能仅凭旧扫描假定安全。

| cohort | q | seeds |
|---|---:|---|
| smoke | 60 | 10400（执行前检查未使用） |
| smoke | 50 | 10410（执行前检查未使用） |
| pilot | 60 | 10401,10402,10403 |
| pilot | 50 | 10411,10412,10413 |
| formal 默认 N=10 | 50 | 11001–11010 |
| formal 默认 N=10 | 60 | 11101–11110 |
| formal 若预先选择 N=20 | 50 | 11001–11020 |
| formal 若预先选择 N=20 | 60 | 11101–11120 |

不是先做 10 个，看结果后再决定加到 20。pilot 后根据时间和内存测量，FORMAL_SETTINGS.md 给出 N=10 与 N=20 的资源估算；默认 N=10/每 q；N=20 是用户已允许的资源充足选项，执行者根据 pilot 测得的资源作出选择并记录依据，在任何 formal run 之前写入 manifest。两组使用相同 N 和同一非 q 参数配置。

若发现预先指定 seed 已用于训练/调试，不覆盖、不悄悄复用；在尚未看新结果前重列该 cohort 的未用 seeds 并记录原因。禁止删除失败 seed 或换 seed 来改善成功率。

## 2. 训练覆盖、计数与资源：必须测量什么

StartSampler.balanced 实际按等概率独立抽 bin，并非保证每 bin 恰好相同次数。报告实际整数 counts，包括 0；不能把 expected counts 当实测结果。

每条 episode 保留 start_stage∈{1,2,3}，每个到达 stage 记录 action 之前的 learner-signed gap。按 (phase,start_stage,current_stage,bin) 统计：

- start1→t1,t2,t3；
- start2→t2,t3；
- start3→t3。

新增的 start2→t3 是必须修复的旧遗漏。记录角色翻转前的 ES start counts 与 learner-signed start counts；主要 exposure 图采用 learner-signed 口径，二者字段明确。对每个 bin 先检查 gap 确在 domain±已有数值 tolerance 内，再做边界映射；不要用 clipping 隐藏真实越界。只允许数值 tolerance 内的端点 roundoff 归端点 bin。

每 update 精确总量：

| Phase | episodes | t1/t2/t3 learner transitions | joint environment steps |
|---|---:|---|---:|
| A | 512 | 0 / 0 / 512 | 512 |
| B | 512 | 0 / 512 / 512 | 1024 |
| C | 512 | 256 / 341 / 512 | 1109 |

joint environment step = 一个 episode 的一个 simultaneous stage transition = 一个 learner transition；两位 physical players 的 actions 数为其两倍。训练 total_environment_steps 不包含 verifier、economic evaluation；后两者单列费用。

若无早停，A400/B600/C1800 共 2800 updates、1,433,600 episodes、2,815,400 joint environment steps、5,630,800 physical-player actions。早停时必须从实际日志累加。

A400×512=204800 个 direct ES3：
q50 每 bin expected 2560；q60 expected 2327.2727。T2 同预算 terminal 每 bin 的 expected exposure 是此值两倍，这是采样量比较，不是“足够训练”的证明。

C 每 update direct ES expected：
q50 ES2=85/40=2.125、ES3=171/80=2.1375；
q60 ES2=85/44≈1.931818、ES3=171/88≈1.943182。
root/ES 正好 256/256。以上只对 direct starts 成立，不能强迫 continuation visits 均匀。

coverage.csv 至少输出实际 min、p05、median、max、mean、CV、zero_bin_count；同时保存所有 bin 的 counts 和 expected。tail 口径固定为 abs(bin_midpoint) ≥ 0.8×domain_half；分别输出正负 tails，包括 direct starts 和 continuation。保存失败最大偏离状态所在 bin 的 exposure，连接覆盖和误差，不把两者相关性当因果结论。

内存：Linux /proc/self/status 的 VmRSS/VmHWM，统一存 bytes；无数据则 null+reason，不能填 0。resource.getrusage 可辅助，但注意 Linux ru_maxrss 单位 KiB。记录训练、dev verifier、final verifier、dense profiles、economic rollout 各阶段 wall_sec、cpu_sec、rss_before/after、rss_peak_bytes。流式评估默认 chunk_size=10000，避免保留全部轨迹。

## 3. 精确定义 verifier 与 certification

只有 endpoint（candidate 或 diagnostic_terminal）保存后，以相同权重重新计算 development 与 final。minimum-development checkpoint 仅保留该次 development 诊断，不送入 final 候选筛选。候选来自 C 的第一次 eligible，不允许 final 失败后恢复训练或寻找另一个“更好” checkpoint。

对任意 t,d：

- mean_effort m_t(d)=e_min+e_range×alpha/(alpha+beta)。
- V_t_mean：后续双方都执行 frozen mean policy 的值。
- V_t_BR：对 frozen mean opponent 的完整动态 best response 值。
- continuation_gain_t=V_t_BR−V_t_mean。
- delta_t：只在当前 stage 最优偏离，后续回到 mean policy 的 gain。它不等于 continuation_gain_t。
- EXP_root=V1_BR(0)−V1_mean(0)。
- reach_contribution_t=max_{d∈R_t}delta_t(d)，dReach=sum_t reach_contribution_t。
- full_contribution_t=max_{d∈D_t}delta_t(d)。
- Delta_max_all=max_t full_contribution_t；旧 dfull=sum_t full_contribution_t。必须分别命名。

所有收益量同时保存 raw 和 over_dw；policy efforts 单位保持 effort。R_t 是现有 BR interval-reachable mask；BR pmf 是 BR-chain quadrature/interpolation 分布。不能把 pmf>0 当 R_t，也不能把 BR pmf 当 self-play visitation。

Final 数值检查保持：

1. development 与 final 都 valid；
2. dReach_final/DW ≤ 0.01；
3. abs(dReach_final−dReach_dev)/DW ≤ 0.002；
4. abs(EXP_root_final−EXP_root_dev)/DW ≤ 0.002；
5. 在完整 D1,D2,D3、step=0.05 上 dense concentration ≤ 0.04 且 valid；
6. final_joint_pass 必须同时要求 has_candidate=true。

额外报告 Delta_max_all 的 dev-final difference 和 stage-wise differences，但不新增其 pass threshold。EXP_root 不新增未经指定的绝对阈值。不要声称全域 MPE certification；Delta_max_all 正是用于显示未被 reach criterion 覆盖的偏离。

无 candidate 的 terminal checkpoint 也做同样的数值评估，类型为 diagnostic_terminal；其 numeric_thresholds_pass 可以为 true，但 certification 必须是 not_applicable_no_candidate，final_joint_pass=false。不存在候选时不能称 final certifier “拒绝了 candidate”。

Final 无效/异常：记录 invalid reasons、已取得的数组与日志。不能通过再训练补救该 run。最终 evaluator 可在原始权重上修复纯输出 bug 后重新运行，记录 attempt 与原因，禁止选择性保留最有利的评估值。

## 4. 经济指标：可解释、可重算、与 verifier 一致

所有 completed runs 的 endpoint 都保存经济结果，包括无 candidate；summary 将 certified candidates、uncertified candidates、diagnostic terminals 分组，不混成“equilibrium average”。

### 4.1 完整策略曲线

在 dense step=0.05 的 full domains 保存：
stage、d、d_normalized、alpha、beta、mean_effort、effort_variance、std_effort、std_norm、opponent_mean_effort=m_t(−d)。

variance=e_range²×alpha×beta/[(alpha+beta)²(alpha+beta+1)]。
e1(0) 是 stage1 唯一行，e2(d)/e3(d) 含 0 与两端。dense concentration 与 profiles 使用同一组 alpha/beta。

对 x>0：
leader_mean_effort=m_t(x)，follower_mean_effort=m_t(−x)，lead_follow_policy_difference=m_t(x)−m_t(−x)。
x=0 是 tie，差值定义可为 0，但 leader/follower 条件样本数为 0。不要拿 physical player0−player1 的差代替领先者−落后者。

### 4.2 独立 root self-play 评估

主结果为 mean-policy self-play：双方均采用 m_t(d)，从 stage1,d=0 开始，环境噪声按现有 uniform shocks 采样。这与现有 deterministic-mean verifier 的对象一致。

另存 stochastic-policy self-play：双方用冻结 actor 的 Beta sampling 和既有 action clamp，完全独立动作流。它用于说明实际 stochastic policy 行为，不是 stochastic exploitability certificate。两种 mode 严格分开报告，不能合并 occupancy 或 payoff。

每 mode：3 replicates，每 replicate 200000 episodes，chunk_size=10000。smoke 减为每 mode 2 replicates×200 episodes。使用独立 numpy Generator，例如：

    SeedSequence([9005000, seed, q, 100, mode_id, rep_id, stream_id])

mode_id=0 mean、1 stochastic；rep_id=0,1,2；stream_id=0 environment、1 player0 actions、2 player1 actions。记录全部 components。不借用训练 RNG；使用 eval()、no_grad()，评估后恢复原 mode。mode1 的条件理论 moments 与 clamp 后实测 action moments 分别标注。

对每 episode、每 stage 的 pre-action physical gap d0，player1 signed gap=−d0。保存：

- 两位 physical players 各自 effort 的 n,sum,sum_sq；E[e]、E[e²]、SD、MCSE。
- 代表性玩家平均令 X=(e0+e1)/2，保存 X 的 n,sum,sum_sq 来计算努力均值及 MCSE。其 E[e²] 另用 Y=(e0²+e1²)/2 的均值，绝不能用 E[X²] 代替；保存 Y 的 n,sum,sum_sq，代表性 expected cost=k×E[Y]。MCSE 的独立单位是 episode，不把同一 episode 两名玩家当独立样本。
- 成本 k×e² 的 moments 与每 stage expected cost；随机成本不可用 k×(E[e])² 替代。
- leader effort、follower effort、paired leader−follower difference：按 stage 开始时 d0>0 或 d0<0 确定，逐 episode 成对累加 n,sum,sum_sq。exact d0=0 单列 ties。
- root tie stage 的 leader/follower 均值与 SE 为 null，附 n=0；不是填 0。
- physical player0/1 的差，作为独立的 role-symmetry diagnostic。
- state visitation：各自 signed-gap 全 bin counts、probability_mass，包含空 bin；另存每 episode 两位玩家平均 histogram，可用于代表性分布。
- 每个 bin 的 effort_sum、effort_sq_sum、cost_sum，避免用 bin midpoint policy 代替实际积分。
- 各 stage pre-action gap mean/SD、positive/negative/tie fractions；保存 gap 的 n,sum,sum_sq。gap 的 p05/p50/p95 固定使用上述 width10 histogram 近似，保存 edges/counts 与方法 histogram_linear_width10：找到累计 mass 首次达到 p 的非空 bin，在该 bin 内按 (p−F_left)/mass_bin 线性插值；root 分位数均为0。明确 approximate，不声称精确分位数。
- final payoff0/1、win/tie rates，episode 总 payoff 与各 stage cost 的会计一致性。

主 E[e_t] 是对 root mean-policy occupancy 的期望；stochastic E[e_t] 使用自己的 occupancy，二者一般不同。训练 mean_effort_by_stage 来自混合 starts、learner 对 snapshot，只能名为 training_mean_effort，不能替代这些指标。

mean-policy rollout payoff 的主比较使用每 episode 的 U=(payoff0+payoff1)/2；保存 U 的 n,sum,sum_sq，以其均值与 final DP 的 V1_mean(0) 比较（当前模型双方对称、root=0）。另保存两位玩家各自 payoff。保存 difference、配对 episode MCSE、difference/MCSE（SE=0 时 null+reason）；数值差包含 MC noise 和 discretization，不据此自动挑选 checkpoint。stochastic-minus-mean payoff 单列为描述值，不声称是 exploitability。

每 replicate 保存 moments。对每个实际分析变量，n>0 时 mean=S/n；n>1 时 sample_variance=(SS−S²/n)/(n−1)、MCSE=sqrt(sample_variance/n)。n=0 的 mean 与 n<2 的 variance/MCSE 为 null+reason；只有浮点舍入级负方差可归零，实质负值视为统计实现错误。明确区分代表性努力变量 X 的方差与随机抽取玩家 effort 的二阶矩 E[Y]。独立 replicate 合并可累加对应变量的 n,sum,sum_sq 得到 pooled mean/MCSE；另报告 replicate means 和它们的 spread。不同训练 seeds 的差异与同一 checkpoint 的 rollout MCSE 分开。

## 5. 输出数据字典与缺失值规则

每 run 目录必有：

- config.json：完整 resolved game/PPO/phase/verifier/eval/seeds、cohort、run_id、Git版本、依赖路径、软件版本、thread settings。
- status.json：pending/running/done/failed/interrupted、PID、开始/结束时间、最后完成 update、可用 checkpoint、failure_reason。
- history.jsonl：每个成功完成的 PPO update 一行，立即 flush；包含时间、资源、loss、episodes、steps、start_counts、stage counts。
- verifier_calls.jsonl：每次调用一行，即使 invalid/exception；包含 phase/local/global update、criterion、eligible components、consecutive、full summary、concentration stage/location、costs。
- events.jsonl：phase entry/exit、snapshot refresh、checkpoint 保存、final/econ 开始结束、error。
- coverage.npz + coverage.csv：完整 start_stage×current_stage×bin 的整数 counts，含 phase/end 与 verifier 检查时累计值；CSV 是派生，NPZ 可重算。
- checkpoints/endpoint.pt + endpoint_weights.npz：candidate 或 diagnostic_terminal；config/JSON 说明其 identity 与 update。
- checkpoints/min_dev.pt + min_dev_weights.npz + min_dev_arrays.npz + min_dev_record.json：达到新的 valid C minimum 时同步保存同次权重、stage_result_arrays(result,'minimum_development')、call identity、完整 summary、concentration 与 coverage snapshot；替换 minimum 时一并替换并最后写 metadata。仅供 diagnosis，不改变 candidate 选择，不另跑 final；minimum profiles 从这里生成，不从 endpoint arrays 生成。
- final_eval.json + arrays.npz：两级 verifier scalars/arrays、dense profiles、flags、checkpoint identity。
- economics.json + economics_arrays.npz：每 mode/replicate 的 moments、histograms、MCSE、seeds、runtime。
- traceback.txt（发生异常时），不要因异常丢失已经完成的数据。

不必保存所有原始训练动作或全部 evaluation trajectories；上述 sufficient statistics 足够重算本计划要求的值。若未来需要逐轨迹研究，应另提需求，不能声称当前 moments 能恢复原始路径。

统一键：run_id、cohort、q、seed、checkpoint_kind、checkpoint_global_update。per-check 再加 phase、local_update、global_update、tier。run_id 如 t3_pilot_q60_s10401，q 与 seed 联合识别；不可只按 seed join。

| 用户要的值 | 具体字段/定义 | 文件 |
|---|---|---|
| candidate found/no candidate | has_candidate；candidate_update；search_outcome | runs.csv、final_eval.json |
| phase exits | phase、local_updates、exit_reason、eligible_count | phases.csv、events.jsonl |
| stopping update | stop_global_update、stop_C_local_update | runs.csv |
| total env steps | total_train_episodes、total_environment_steps、physical_action_count | runs.csv、history.jsonl |
| EXP_root/DW | exp_root、exp_root_over_dw、tier | runs.csv、verifier_summary.csv |
| dReach/DW | dreach、dreach_over_dw、tier | 同上 |
| Delta_max_all/DW | delta_max_all、delta_max_all_over_dw；另列 dfull | 同上 |
| dev-final differences | refine_dreach_diff_over_dw、refine_exp_diff_over_dw、refine_delta_max_all_diff_over_dw | runs.csv、final_eval.json |
| final pass/fail | valid_dev/final、main_pass、refine_*_pass、dense_conc_pass、numeric_thresholds_pass、certification、final_joint_pass | runs.csv |
| e1(0) | e1_mean_effort | runs.csv、policy_profiles.csv |
| e2(d)、e3(d) | stage,d,mean_effort,alpha,beta,std_norm | policy_profiles.csv |
| stage-wise expected effort | mode,stage,player_or_representative,n,effort_sum/sumsq,mean,MCSE,E_e2,E_cost | stage_metrics.csv |
| state visitation | origin=training/mean_selfplay/stochastic_selfplay/br_chain；bin counts/mass | state_visitation.csv |
| leader/follower asymmetry | curve difference；occupancy paired difference+MCSE+sample count | policy_asymmetry.csv、stage_metrics.csv |
| minimum dev dReach | min_valid_C_dreach_over_dw、min_C_global/local_update、min_C_conc、min_checkpoint_ref | runs.csv、failure_diagnostics.csv |
| stage-wise contribution | min_C_reach_contribution_t_raw/over_dw、full_contribution_t、argmax_d | failure_diagnostics.csv |
| maximum-deviation state | min_C_max_reach_stage/d/delta、min_C_max_all_stage/d/delta | failure_diagnostics.csv |
| B passed/budget-forced | phase_B_exit_reason、B_first_eligible、B_third_consecutive_eligible、B_checks/counts | phases.csv、runs.csv |
| payoff deviation | delta_t(d)、continuation_gain_t(d)、EXP_root 与 normalizations | deviations.csv、verifier_summary.csv |
| coverage audit | actual+expected bin counts、tails、zero bins、exposure near argmax | coverage.csv、PILOT_REPORT.md |
| memory/runtime/cadence/log completeness | peak_RSS_bytes、wall/CPU、verifier call schedule、missing fields/events | resources.csv、REPORT |

表格补充规则：

- deviations.csv 的 tier∈development/final/minimum_development；列 stage,d,delta_raw,delta_over_dw,continuation_gain_raw/over_dw,a_dev,a_br,e_hat,e_opp,reach_mask,br_pmf。不必输出全部 Q matrix 为 CSV；NPZ 可保留 stage_result_arrays。
- verifier_summary.csv 每 run/checkpoint/tier 一行；per-stage 明细可用 verifier_stage_metrics.csv，不把 list 随意塞入难解析单元格。
- state_visitation.csv 的 root 用单独 d=0、bin_left=bin_right=0，mass=1；BR pmf 使用 verifier state grid，grid_kind=verifier_nodes，不伪装成 width10 training bins。
- 每行标明 stage/current_stage 与 perspective（learner/player0/player1/representative），严禁将两套不同口径 join 后当同一分布。
- 缺失或非有限数值在 JSON 为 null，CSV 为空并配 unavailable_reason；strict JSON 禁止 NaN/Infinity。valid=false 的 verifier 已产生的有限诊断值与 arrays 仍保留，同时带 invalid_reasons；不能仅因 invalid 就抹去其有限数据。
- partial run 也必须在 runs.csv 中占一行。reporter 根据 manifest 枚举 runs，不能只扫描成功 final_eval.json。
- manifest 预定但尚未启动的 run 保留 pending，不能报告成已观察到的失败；完成报告列 N_planned/N_started/N_completed/N_operational_failed。
- cohort 全部尝试后，主要可靠性分母为所有预定且已尝试 runs，包括 operational failures。candidate 分子依据已持久化的 has_candidate；发现 candidate 后 final/econ 操作失败不能抹去发现记录。认证尚未完成计未认证；纯经济输出失败不能反向抹去已完成的认证。另给 completed-only sensitivity，并明确原因。不能替换失败 seed。
- candidate 存在但 final evaluator 中断：certification=error，仍计入 conditional certification 的 candidate 分母，未认证，不当成通过。

### Minimum C failure diagnosis 的精确选择

只从 valid 且 finite 的 C development 调用中选最小 dReach/DW；相等取最早 global update。记录其同一调用的 stage contributions、argmax、concentration、coverage、weights。不要用 terminal arrays 解释 earlier minimum。

max_reach_state 是在该记录各 stage reach_delta_max 中取最大；max_all_state 是 full_delta_max 中取最大；并列按 stage 升序、d 升序。dReach 的三项 contributions 求和应复原同一记录 dReach。

无 valid C 调用时：minimum 和相关 state 为 null，reason=no_valid_C_verifier；不能调用 min([])，也不能把正常搜索失败误报成 reporter exception。发生在进入 C 之前的操作错误记录 C_not_reached。
