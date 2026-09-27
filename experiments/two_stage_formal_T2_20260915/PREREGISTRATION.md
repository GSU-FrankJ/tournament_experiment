## 四、正式 T=2 验证计划

本轮选定唯一正式 protocol：**G1 A400 constant LR + C-stage first eligible**。q=50、q=60 各20个全新训练 seed，共40条运行。当前状态为方案与输入清单已准备，尚未启动正式训练。

选择 G1 是为了沿用此前停止规则实验的训练设置，将新验证的行为改动限定为 C 阶段 stopping / candidate selection。旧调参记录中 G2 A600 constant 的首次候选获得率是10/10，G1为8/10；本方案不据此声称 A400 最优，也不把原5个seed上的差异当成可靠的预算优劣结论。正式结果只解释本次选定的单一 protocol。

### 固定设置与 held-out seeds

以下设置在本轮40条运行期间保持一致。正式输入清单逐条保存全部 game、PPO、采样、LR、verifier、dtype 与 runtime 设置，不依赖启动时的默认值补齐：

experiments/two_stage_formal_T2_20260915/manifest.json

| 项目 | 本轮固定值 |
| --- | --- |
| 模型与博弈 | T=2；shared Beta actor与独立critic，均为2→64 Tanh→64 Tanh；actor双输出、critic单输出；w_h=6、w_l=2、k=1/3500、努力范围[0,100] |
| Curriculum budgets | A≤400、B≤600、C≤1000 PPO updates；每run总上限2000。A/B可按原规则提前推进，上限不是固定训练长度 |
| A/B progression | 保留 k_phase=3；A用完整D2的最大第二期偏离/ΔW≤0.02，B用EXP_root/ΔW≤0.02，并同时满足数值有效性及浓度标准。达到阶段上限后的推进处理沿用原实现 |
| Learning rate | actor与critic全程3×10^-4；保持Adam状态，不使用C阶段预设衰减 |
| Snapshot rule | hard snapshot：初始化、每次进入阶段，以及global update为20的倍数时刷新；沿用原先在该次PPO更新后刷新的顺序 |
| 训练采样 | 每update 512 episodes；A为第二期exploring starts，B为root starts；C为256 root + 256第二期exploring starts；ES bin width=10 |
| Development调用日程 | 每阶段local update=100强制检查；每20 updates监测策略漂移及KL；drift≤0.01且KL≤0.01连续2次可触发检查；距上次verifier达到100 updates时timeout触发；阶段上限保留末次检查。各触发条件和重置顺序沿用原runner |
| Development verifier | state step=4、effort step=1、gl_half=16；网格覆盖完整D2，B/C同时包括第一期root |
| First-eligible rule | 仅将C的k_stop从5改为1。首次有效development call同时满足dReach/ΔW≤0.01及C_dev≤0.04时，立即保存当前完整权重并结束正式PPO更新 |
| Concentration threshold | C_dev=max Std[e|t,d]/100≤0.04；α、β来自同一checkpoint，即max√[μ(1−μ)/(α+β+1)]≤0.04。最终完整网格C_all同样以0.04评价，单列结果 |
| Final verifier | 同一冻结候选使用state step=2、effort step=0.5、gl_half=32复核；不据final结果恢复训练或更换候选 |
| Final certification threshold | final数值有效且dReach_final/ΔW≤0.01；同时要求development数值有效及原numerical-refinement条件 |
| Numerical-refinement tolerance | 同一权重的两档结果满足abs(dReach_final−dReach_dev)/ΔW≤0.002，且abs(EXP_root_final−EXP_root_dev)/ΔW≤0.002 |
| Verifier内部容差 | 两档均为grid_tol=10^-9、mass_tol=10^-10、pdl_tol_over_dw=10^-10 |
| 最终浓度与恢复网格 | C_all使用root及完整D2上0.05步长网格；analytical recovery使用0.5步长网格，正努力域严格为abs(d)<2q，边界归tail |
| 计算设置 | CPU；网络float32，环境gap/reward和verifier float64；torch/OMP/MKL/OPENBLAS每进程1线程；最多10个并行进程 |
| 软件环境 | 沿用已核验环境Python3.12.3、torch2.5.1+cu121、NumPy2.5.0；解释器为python |

PPO其余参数固定为：Adam betas=(0.9,0.999)、eps=10^-8、weight decay=0、clip epsilon=0.2、value coefficient=0.5、entropy coefficient=0、max gradient norm=0.5、每update 10 epochs、minibatch=256、gamma=1、GAE lambda=1、c_min=100、mu/action clamp=10^-6、advantage normalization epsilon=10^-8。保留原动作采样、观测编码和随机数namespace。KL仅用于原稳定性判断，不新增KL early stopping。

保留原离线直接rollout诊断：每次200,000 episodes、3次重复、seed base=9005000、sensitivity阈值0.03；它不参与checkpoint选择或原overall_pass。这里的numerical refinement仅比较冻结权重的两档数值结果，与“继续训练的verifier-triggered refinement”是不同操作。EMA、stage-specific heads及后者均不加入本轮。

训练seed固定为 **10001–10020**，两个q使用相同列表形成20个seed pair。每个q有20条运行；跨q共40条运行不能当作40个独立seed做合并推断。2026-09-15只读检查根项目的experiments、MultiStage/Discussion、results/run/config/tools及指定worktree的results/run/config/tools，未发现这20个值被用作seed配置或结果路径中的seed。该结论限于已检查记录，不声称覆盖项目外或已删除的历史。

原seeds47–51和此前35个first-eligible候选仅用于开发与方案选择，不进入本轮正式分母。不依据新seed的中途结果更换配置、增删seed或提前结束整批运行。当前runner不支持保存原随机数状态的中途续跑；仅对已确认基础设施故障而未完成的run，允许同一seed从起点完整重跑，另存attempt并保留异常记录。按预定attempt顺序采用首次完整完成者，不看结果选优；已经完整完成但未通过搜索或认证的run不因失败重跑，也不补抽seed替换。

### Search success

对每条run，定义：

$$
S_{\mathrm{search}}
= \mathbf{1}\{\text{C预算内出现同点数值有效、BR通过、浓度通过的首次development检查}\}.
$$

记录首次候选的global update、C-local update、累计episodes/transitions、检查触发原因、development偏离、浓度及数值有效性。候选必须来自上述既有调用日程，不插入新的20-update BR检查来扩大搜索机会。

预算内包括C上限处的末次development检查：若首次eligible恰在该点出现，仍算search success；不能用原输出的strictly_before_cap字段代替该定义。

没有候选而用尽C预算时，记录search failure与budget exhaustion。对这类run汇总全部C检查中的有效/BR/浓度/联合通过次数，分别报告最小偏离、最小浓度以及各自同点的另一项和update；直接判据原因与训练机制解释分开。

原runner在预算耗尽后仍可能保存并评估末点。该末点只能作为无候选run的诊断输出，不能追认为first eligible，也不能凭其final结果改写search success。

### Final certification

仅以first-eligible时冻结的同一θ作为正式候选。保留三个明确字段：

| 字段 | 含义 |
| --- | --- |
| final_overall_pass | 沿用原final verifier：development/final数值有效、final dReach/ΔW≤0.01，且两项数值细化差均≤0.002 |
| final_concentration_pass | 同一θ在完整稠密网格上的C_all有效且≤0.04 |
| final_joint_pass | 上述两项同时通过；作为本轮正式交付候选的联合结果 |

原overall_pass不含浓度，不能将其与joint pass混称。EXP_root_final单独报告；除原细化差要求外，不再添加EXP_root绝对阈值。最终验证未通过时保留失败候选和原结果，不恢复PPO、不另挑checkpoint。

没有first-eligible候选的run，其候选认证字段为not_applicable_no_candidate；端到端联合成功指标记0。基础设施异常或尚未完成的run明确标记，不假装是一个已完成的verifier失败。

### Analytical recovery

对**所有first-eligible候选**计算恢复误差，包括final未通过者；同时另列final joint通过子集及其样本数。无候选run的正式候选恢复指标记NA，末点诊断与候选统计分开。

沿用第三节已经明确的连续指标：

- 第一期努力：真值、估计值、有符号误差、绝对误差及相对误差；
- 第二期完整D2与正努力域：MAE、RMSE、最大绝对误差；
- 峰值：d=0处的绝对/相对误差，另报实际网格最大值和位置；
- tail：abs(d)≥2q的平均及最大努力；
- evenness：完整D2上的最大及平均abs(ehat2(d)−ehat2(−d))；
- on-path：三角分布加权MAE、RMSE、E[ehat2]及abs(E[ehat2]−g1)；
- 跨期关系：ehat1−E[ehat2]的符号、绝对值及相对理论g1的比例。

on-path主表保留原分半64点Gauss–Legendre求积，并按既有离线脚本对冻结策略使用0.5/0.25/0.125/0.0625步长细化，单列求积敏感性，不据此更换候选。两角色是同一actor在d与−d的映射，不增加独立样本数。

本轮没有独立的analytical-recovery通过门槛，因此不定义“recovery成功率”，也不把非零误差直接记为失败。可以报告“通过原偏离认证的候选具有怎样的解析恢复精度”，不能据认证通过宣称策略已逐点恢复解析均衡。

### Experimental Evaluation

以q=50、q=60分别报告，每个q的计划分母固定为N=20：

| 结果 | 分子 / 分母 |
| --- | --- |
| Search-success rate | 找到first-eligible的run数 / 20 |
| Conditional final-certification rate | final_overall_pass数 / 找到候选的run数 |
| Conditional final-joint rate | final_joint_pass数 / 找到候选的run数 |
| End-to-end final-joint success rate | final_joint_pass数 / 20 |
| Budget-exhaustion rate | C预算耗尽且无候选的run数 / 20 |
| Operational status | 完成、待处理、中断及技术失败分别计数，不从计划分母中静默删除 |

最终报告同时给出计数与比例；在相应样本集合及全部状态明确后，比例报告95% Wilson区间。条件率分母为0时记NA。先报告每个q，跨q的总体计数仅作描述，不把配对运行当成40个独立样本计算区间。

连续指标报告均值、样本SD、中位数、IQR和范围。成本报告实际A/B/C updates、episodes、transitions、development calls、训练耗时和离线验证耗时；首次候选的取得时间与预算耗尽分别展示，不能只对成功run的较短训练时间取平均后代表整个方案。

输出一张40条run的主表、两份按q的汇总、恢复曲线与误差图，以及每个无候选run的C检查时间表。主表至少包括seed、q、运行状态、首次候选update、search success、final三个字段、偏离/浓度/数值细化量、恢复误差及计算成本。retention after first eligible不属于主成功指标；正式run在首次eligible即停止，本轮不为了计算retention延长训练。

完整输入清单沿用原round2 manifest结构，只有run/seed/group/output等身份字段及C的k_stop=1与对应q的G1模板不同；不新增训练机制。执行时使用该清单中的正式记录，保持smoke_overrides为空。方案及清单的准备和校验不代表40条训练已经执行，结果报告须以实际完成记录为准。
