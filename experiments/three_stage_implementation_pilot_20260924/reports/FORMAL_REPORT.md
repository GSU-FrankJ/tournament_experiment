# T=3 formal 报告（q50 / q60，各 20 个预定 seeds）

日期 2026-09-26（时间均为 UTC），主机 vector2。
计划：[T3 科学实施协议](../../../MultiStage/three_stage/T3_IMPLEMENTATION_PLAN_20260924.md)。
配置：`reports/formal_settings.json`，2026-09-24 在任何 formal run 之前选定，本次未修改。
本报告只含 formal cohort。pilot、A-sampling、ABC 等 debug 结果与比例不计入。

## 0. 直接回答

1. **计划 40 个 run，实际完成 40 个。正常 no-candidate 40 个，错误 0 个。**
   - 40/40 的 `state=done`。launcher 记录 40 条 `exited`，exit code 全为 0。`make_report --check-completeness` 对 40/40 判定 `ok`。
   - 40 个都是正常搜索结束：`no_candidate_budget_exhausted`。
   - 执行错误 0，未完成或中断 0，重跑 0，`skipped_existing_dir` 0。
2. **q50 与 q60 都是 candidate 0/20，认证 0。**
   - 两个 q 的 candidate discovery 与 end-to-end 都是 0/20，Wilson 95% 区间 [0, 0.161]。
   - conditional certification 为 N/A：没有 candidate，所以不写成 0%。
3. **A/B/C 的退出情况：**

   | phase | 退出 | 检查结果 |
   |---|---|---|
   | A | 40/40 `fixed_budget_completed`（400） | 仅诊断 |
   | B | 40/40 `budget_forced`（600），`verifier_passed` 0 | 960 次检查，eligible 0 |
   | C | 40/40 `no_candidate_budget_exhausted`（1800） | 2880 次检查，eligible 0，dReach/DW ≤ 0.01 的检查 0 次 |

   每个 run 都在 global update 2800 停止，训练量相同：1,433,600 episodes、2,815,400 joint environment steps、5,630,800 physical actions。
4. **四类结果齐全，没有缺项。** 40/40 都有：
   - Search；
   - Verification（dev、final 两层均 valid）；
   - Economic policy（mean、stochastic 两种 mode，各 3×200000 episodes）；
   - Failure diagnosis（40 个都无 candidate，每个都有 minimum-C 诊断）。

   没有缺失字段。以下两点不是缺失数据：
   - economics 的 certified candidate 与 uncertified candidate 组在两个 q 都是 n=0，因为没有 candidate。只有 diagnostic_terminal 组，每 q 20 个。
   - `reports/formal/AUTO_SUMMARY.md` 开头写着“interpretation is in PILOT_REPORT.md”。这是 make_report.py 的固定模板文字，formal 的解释以本文件为准（见 §2.6）。

## 1. 主结果：三类比率（按 q）

主分母是每 q 预定并已启动的全部 20 个 seeds，操作失败也会计入分母；本次没有操作失败。
completed-only sensitivity 与主结果相同（20/20 完成），见 `reports/formal/rates.csv`。

#### R. Rates per q (primary population = all started runs)

| q | population | rate | k/n = p | Wilson 95% | note |
|---|---|---|---|---|---|
| 50 | primary | candidate_discovery | 0/20 = 0 | [0, 0.1611] |  |
| 50 | primary | conditional_certification | N/A | N/A | no candidates (N/A) |
| 50 | primary | end_to_end | 0/20 = 0 | [0, 0.1611] |  |
| 50 | completed_only_sensitivity | candidate_discovery | 0/20 = 0 | [0, 0.1611] |  |
| 50 | completed_only_sensitivity | conditional_certification | N/A | N/A | no candidates (N/A) |
| 50 | completed_only_sensitivity | end_to_end | 0/20 = 0 | [0, 0.1611] |  |
| 60 | primary | candidate_discovery | 0/20 = 0 | [0, 0.1611] |  |
| 60 | primary | conditional_certification | N/A | N/A | no candidates (N/A) |
| 60 | primary | end_to_end | 0/20 = 0 | [0, 0.1611] |  |
| 60 | completed_only_sensitivity | candidate_discovery | 0/20 = 0 | [0, 0.1611] |  |
| 60 | completed_only_sensitivity | conditional_certification | N/A | N/A | no candidates (N/A) |
| 60 | completed_only_sensitivity | end_to_end | 0/20 = 0 | [0, 0.1611] |  |

解释：

- 预算（A400/B600/C1800）与阈值都是预先规定的。在此设置下，40 个独立初始化的 run 没有一次 C 检查 eligible，所以没有 candidate，也就没有可以认证的对象。
- 0/20 的 Wilson 上界 0.161，是该协议下每 q candidate-discovery 概率的 95% 上界。它是区间估计，不是“不可能”的证明。
- 本结果不声称任何 seed 收敛，也不对均衡是否存在下结论。

## 2. 科学执行记录与来源

- 正式实验日期为 2026-09-26。40 个 seeds（q50: 11001–11020；q60: 11101–11120）在正式训练前仅作预留，未用于此前 smoke、pilot 或诊断训练。没有替换或重跑 seed。
- 运行环境为 Python 3.12.3、NumPy 2.5.0、PyTorch 2.5.1+cu121，使用 CPU；每 worker 的 OMP/MKL/OPENBLAS 与 torch 线程数均为 1。原始源代码来源版本为 d1b8443；本公开仓库包含当时使用的数值实现。
- 正式配置与 pilot 的非 q 设置相同：A400/B600/C1800，economics 每模式 3×200000 episodes。未使用 smoke overrides、center-ES 或其他诊断设置。每个 run 的 resolved 配置保存在 config.json。
- 全部 40 个 run 正常完成，没有中断、失败或重跑；Phase B 都由预算上限退出，Phase C 都未发现 candidate。全部终点为未认证的诊断策略。
- 批量并发上限为 10，每 worker 单线程；整批运行 84 分 31 秒。单 run launcher wall 为 1193.7–1293.9 秒，均值 1254.2 秒。资源明细保存在 compact reports。
- 历史完整原始数据的完整性检查通过。公开版保留其检查结果和全部种子的结论，但省略大型原始数组；公开版 saved-weight replay 可独立重算 endpoint development/final 与 minimum-development verifier。完整原始报告的重新生成需要省略的数据。

## 3. Search（逐 seed）

#### S. Search (per seed)

| q | seed | state | outcome | A exit (local) | B exit (local) | B eligible/checks | C exit (local) | candidate update | stop update | episodes | joint env steps | physical actions | run wall s (launcher) | exit code | train wall s | train CPU s | final wall s | econ wall s | peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1247.6 | 0 | 1238.8 | 674.37 | 0.28 | 3.01 | 512.8 |
| 50 | 11002 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1220 | 0 | 1211.8 | 639.73 | 0.273 | 2.56 | 516.2 |
| 50 | 11003 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1193.7 | 0 | 1184 | 625.59 | 0.29 | 2.65 | 514.2 |
| 50 | 11004 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1247.7 | 0 | 1238.5 | 657.8 | 0.282 | 3.35 | 511.5 |
| 50 | 11005 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1245.8 | 0 | 1236.3 | 663.37 | 0.264 | 3.65 | 514.1 |
| 50 | 11006 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1226.2 | 0 | 1218.3 | 638.79 | 0.242 | 2.52 | 509.2 |
| 50 | 11007 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1214.1 | 0 | 1205.3 | 630.85 | 0.318 | 2.7 | 512.5 |
| 50 | 11008 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1236.5 | 0 | 1227.7 | 651.44 | 0.267 | 2.97 | 515.3 |
| 50 | 11009 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1245.3 | 0 | 1236.3 | 666.59 | 0.258 | 3.02 | 516.9 |
| 50 | 11010 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1241.3 | 0 | 1232.3 | 653.77 | 0.276 | 3.04 | 515.7 |
| 50 | 11011 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1244 | 0 | 1235.9 | 647.31 | 0.304 | 2.99 | 516.5 |
| 50 | 11012 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1230.7 | 0 | 1222.9 | 628.55 | 0.298 | 2.66 | 515.5 |
| 50 | 11013 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1272.4 | 0 | 1264.4 | 650.74 | 0.319 | 2.59 | 519.1 |
| 50 | 11014 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1282.6 | 0 | 1273.1 | 669.74 | 0.325 | 2.8 | 515 |
| 50 | 11015 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1281.5 | 0 | 1271.9 | 662.07 | 0.318 | 3.18 | 514.3 |
| 50 | 11016 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1274.8 | 0 | 1266.6 | 650.52 | 0.291 | 2.66 | 516 |
| 50 | 11017 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1217.4 | 0 | 1208.8 | 622.29 | 0.309 | 2.48 | 513.7 |
| 50 | 11018 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1268.5 | 0 | 1258.5 | 655.73 | 0.309 | 3.27 | 519.1 |
| 50 | 11019 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1266.9 | 0 | 1256.6 | 657.83 | 0.287 | 3.57 | 523.1 |
| 50 | 11020 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1228.4 | 0 | 1220.4 | 626.16 | 0.25 | 2.59 | 516 |
| 60 | 11101 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1262.7 | 0 | 1254.8 | 668.22 | 0.356 | 2.55 | 515.3 |
| 60 | 11102 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1199.4 | 0 | 1189.2 | 611.77 | 0.349 | 3.09 | 517.6 |
| 60 | 11103 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1264.1 | 0 | 1255.4 | 652.18 | 0.331 | 3.1 | 521.5 |
| 60 | 11104 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1248.2 | 0 | 1240.3 | 638.79 | 0.326 | 2.69 | 522.3 |
| 60 | 11105 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1245.4 | 0 | 1237 | 635.38 | 0.366 | 2.68 | 512.6 |
| 60 | 11106 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1262.6 | 0 | 1254.4 | 648.86 | 0.311 | 2.66 | 515.2 |
| 60 | 11107 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1240.3 | 0 | 1231.9 | 631.23 | 0.269 | 2.72 | 516.2 |
| 60 | 11108 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1250.6 | 0 | 1242.1 | 640.98 | 0.338 | 2.61 | 517 |
| 60 | 11109 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1232.1 | 0 | 1223.4 | 628.48 | 0.361 | 2.73 | 516.8 |
| 60 | 11110 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1270.5 | 0 | 1261.9 | 669.01 | 0.349 | 2.56 | 515 |
| 60 | 11111 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1257.9 | 0 | 1249.9 | 638.95 | 0.336 | 2.89 | 517.2 |
| 60 | 11112 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1291.2 | 0 | 1282.7 | 639.43 | 0.351 | 3.13 | 519.3 |
| 60 | 11113 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1288.2 | 0 | 1280.3 | 635.04 | 0.312 | 2.62 | 515.4 |
| 60 | 11114 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1291.1 | 0 | 1283.2 | 641.54 | 0.329 | 2.68 | 519.4 |
| 60 | 11115 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1262.4 | 0 | 1254 | 620.03 | 0.337 | 2.78 | 520 |
| 60 | 11116 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1281.8 | 0 | 1274.2 | 633.29 | 0.345 | 2.52 | 519.4 |
| 60 | 11117 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1293.9 | 0 | 1286.7 | 661.48 | 0.339 | 2.43 | 515.1 |
| 60 | 11118 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1269.6 | 0 | 1260.9 | 622.6 | 0.312 | 3.22 | 517.5 |
| 60 | 11119 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1289.1 | 0 | 1281.6 | 641.13 | 0.333 | 2.59 | 517 |
| 60 | 11120 | done | no_candidate | fixed_budget_completed (400) | budget_forced (600) | 0/24; longest 0 | no_candidate_budget_exhausted (1800) | — | 2800 | 1433600 | 2815400 | 5630800 | 1281.7 | 0 | 1274.3 | 640.01 | 0.319 | 2.48 | 515.1 |

- **updates 与停止点**：所有 run 的实际 updates 都是 A 400 + B 600 + C 1800 = 2800。没有 candidate，所以 candidate update 全为空，停止点全为 2800。
- **资源**：
  - 每 run 训练 wall 1184–1287 s，训练 CPU 612–674 s；40 个 run 合计 7.16 CPU·h。
  - 每 run 训练中 100 次 dev verifier 合计 5.3–6.1 s；endpoint final 两层 0.24–0.37 s；economics（2 mode × 3 × 200000）2.4–3.7 s。
  - 每 run 峰值 RSS 509–523 MiB，10 个并发约 5.2 GB。
- **wall 与 CPU 的差**：训练 wall 比 CPU 每 run 多约 600 s（pilot 在 3 个并发时约 390 s）。这与每个 update 两次 fsync 的 I/O 等待相符（10 个进程共用 /home md0），不影响数值。
- **磁盘**：40 个 run 目录共 1.4 GB（每 run 约 33 MB，含 per-run tables）；`reports/formal` 651 MB。

### Phase B 与 C 的检查统计

#### P. Phase B and C development checks (per seed)

Failure types per check: S = strategic only, K = concentration only, SK = both, I = invalid. B strategic value = max_D2 (V2_BR − V2_mean)/DW (threshold 0.02); C = dReach/DW (threshold 0.01); concentration = max std_norm on the dev grids of the phase's stages (threshold 0.04).

| q | seed | B exit (local) | B eligible | B S/K/SK/I | B strategic min / last | B conc min / last | C exit (local) | C eligible | C S/K/SK/I | C dReach min / last | C conc min / last |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 3/0/21/0 | 0.04659 / 0.04659 (≤thr 0×) | 0.03959 / 0.03959 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 70/0/2/0 | 0.02161 / 0.02396 (≤thr 0×) | 0.0283 / 0.0284 |
| 50 | 11002 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.04164 / 0.09339 (≤thr 0×) | 0.04309 / 0.04318 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 41/0/31/0 | 0.01892 / 0.09137 (≤thr 0×) | 0.03461 / 0.03461 |
| 50 | 11003 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 4/0/20/0 | 0.0264 / 0.0314 (≤thr 0×) | 0.03616 / 0.04207 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 46/0/26/0 | 0.01772 / 0.0536 (≤thr 0×) | 0.03523 / 0.03523 |
| 50 | 11004 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.03151 / 0.06635 (≤thr 0×) | 0.0415 / 0.04183 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 55/0/17/0 | 0.02306 / 0.0463 (≤thr 0×) | 0.02989 / 0.02989 |
| 50 | 11005 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 2/0/22/0 | 0.04834 / 0.04834 (≤thr 0×) | 0.03769 / 0.04063 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 62/0/10/0 | 0.05811 / 0.07527 (≤thr 0×) | 0.02886 / 0.02899 |
| 50 | 11006 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.02745 / 0.0582 (≤thr 0×) | 0.04078 / 0.04118 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 59/0/13/0 | 0.02355 / 0.03492 (≤thr 0×) | 0.03021 / 0.03021 |
| 50 | 11007 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.05476 / 0.067 (≤thr 0×) | 0.04167 / 0.04167 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 55/0/17/0 | 0.04109 / 0.04109 (≤thr 0×) | 0.03076 / 0.03213 |
| 50 | 11008 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.1275 / 0.1275 (≤thr 0×) | 0.02592 / 0.02808 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 72/0/0/0 | 0.1168 / 0.1425 (≤thr 0×) | 0.02716 / 0.02723 |
| 50 | 11009 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.05639 / 0.05762 (≤thr 0×) | 0.03086 / 0.03867 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 68/0/4/0 | 0.0447 / 0.05872 (≤thr 0×) | 0.02586 / 0.02586 |
| 50 | 11010 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 13/0/11/0 | 0.05061 / 0.05061 (≤thr 0×) | 0.03127 / 0.04081 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 55/0/17/0 | 0.04713 / 0.0526 (≤thr 0×) | 0.02956 / 0.02956 |
| 50 | 11011 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 19/0/5/0 | 0.05963 / 0.07541 (≤thr 0×) | 0.02987 / 0.03921 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 70/0/2/0 | 0.0521 / 0.06169 (≤thr 0×) | 0.02884 / 0.02902 |
| 50 | 11012 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 1/0/23/0 | 0.03722 / 0.0436 (≤thr 0×) | 0.03867 / 0.04043 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 65/0/7/0 | 0.02342 / 0.05688 (≤thr 0×) | 0.033 / 0.033 |
| 50 | 11013 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 8/0/16/0 | 0.03502 / 0.0578 (≤thr 0×) | 0.03766 / 0.03857 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 72/0/0/0 | 0.01543 / 0.02401 (≤thr 0×) | 0.02655 / 0.02655 |
| 50 | 11014 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.02972 / 0.03447 (≤thr 0×) | 0.04193 / 0.04193 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 52/0/20/0 | 0.0238 / 0.0466 (≤thr 0×) | 0.03018 / 0.03018 |
| 50 | 11015 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.05165 / 0.05165 (≤thr 0×) | 0.0412 / 0.0412 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 50/0/22/0 | 0.03354 / 0.04014 (≤thr 0×) | 0.03186 / 0.03186 |
| 50 | 11016 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 14/0/10/0 | 0.04056 / 0.04889 (≤thr 0×) | 0.03606 / 0.03796 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 72/0/0/0 | 0.03524 / 0.03801 (≤thr 0×) | 0.02895 / 0.02895 |
| 50 | 11017 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 10/0/14/0 | 0.03644 / 0.03644 (≤thr 0×) | 0.0389 / 0.03905 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 71/0/1/0 | 0.03089 / 0.04699 (≤thr 0×) | 0.02994 / 0.02994 |
| 50 | 11018 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 1/0/23/0 | 0.03118 / 0.03223 (≤thr 0×) | 0.03823 / 0.04168 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 58/0/14/0 | 0.03144 / 0.09502 (≤thr 0×) | 0.03453 / 0.03453 |
| 50 | 11019 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.03802 / 0.04463 (≤thr 0×) | 0.04014 / 0.04032 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 70/0/2/0 | 0.03049 / 0.09145 (≤thr 0×) | 0.03049 / 0.03077 |
| 50 | 11020 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 7/0/17/0 | 0.02756 / 0.02876 (≤thr 0×) | 0.03857 / 0.04012 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 72/0/0/0 | 0.02529 / 0.06916 (≤thr 0×) | 0.03006 / 0.03012 |
| 60 | 11101 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 16/3/5/0 | 0.01893 / 0.02097 (≤thr 3×) | 0.03197 / 0.0403 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 67/0/5/0 | 0.02531 / 0.0383 (≤thr 0×) | 0.02684 / 0.02714 |
| 60 | 11102 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 8/0/16/0 | 0.03379 / 0.03595 (≤thr 0×) | 0.03854 / 0.03854 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 72/0/0/0 | 0.04573 / 0.04642 (≤thr 0×) | 0.02641 / 0.02641 |
| 60 | 11103 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 0/0/24/0 | 0.02523 / 0.03081 (≤thr 0×) | 0.04274 / 0.04303 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 41/0/31/0 | 0.01595 / 0.02053 (≤thr 0×) | 0.03241 / 0.03284 |
| 60 | 11104 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.02705 / 0.02705 (≤thr 0×) | 0.03424 / 0.03771 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 71/0/1/0 | 0.02061 / 0.02453 (≤thr 0×) | 0.02869 / 0.02892 |
| 60 | 11105 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.07725 / 0.08583 (≤thr 0×) | 0.02807 / 0.03252 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 64/0/8/0 | 0.02721 / 0.03297 (≤thr 0×) | 0.02872 / 0.02951 |
| 60 | 11106 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 3/0/21/0 | 0.02015 / 0.02015 (≤thr 0×) | 0.03981 / 0.03981 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 63/0/9/0 | 0.01557 / 0.05321 (≤thr 0×) | 0.02717 / 0.02717 |
| 60 | 11107 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 7/4/13/0 | 0.0165 / 0.0165 (≤thr 4×) | 0.03257 / 0.04053 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 66/0/6/0 | 0.02309 / 0.04186 (≤thr 0×) | 0.02681 / 0.02687 |
| 60 | 11108 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.04807 / 0.04817 (≤thr 0×) | 0.03111 / 0.03396 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 71/0/1/0 | 0.03397 / 0.03553 (≤thr 0×) | 0.02643 / 0.02643 |
| 60 | 11109 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 12/0/12/0 | 0.02281 / 0.037 (≤thr 0×) | 0.02852 / 0.04003 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 66/0/6/0 | 0.02222 / 0.03463 (≤thr 0×) | 0.02607 / 0.02607 |
| 60 | 11110 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.05105 / 0.05567 (≤thr 0×) | 0.02699 / 0.03325 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 72/0/0/0 | 0.0238 / 0.02696 (≤thr 0×) | 0.02585 / 0.02585 |
| 60 | 11111 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 6/0/18/0 | 0.02436 / 0.02909 (≤thr 0×) | 0.0378 / 0.04066 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 65/0/7/0 | 0.02489 / 0.03617 (≤thr 0×) | 0.02792 / 0.02792 |
| 60 | 11112 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 14/0/10/0 | 0.02871 / 0.04767 (≤thr 0×) | 0.03686 / 0.03926 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 66/0/6/0 | 0.03812 / 0.04501 (≤thr 0×) | 0.02627 / 0.02627 |
| 60 | 11113 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.05398 / 0.05398 (≤thr 0×) | 0.03155 / 0.03593 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 72/0/0/0 | 0.03473 / 0.03691 (≤thr 0×) | 0.02588 / 0.02588 |
| 60 | 11114 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 1/0/23/0 | 0.02041 / 0.02111 (≤thr 0×) | 0.03995 / 0.03995 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 61/0/11/0 | 0.0176 / 0.08223 (≤thr 0×) | 0.02764 / 0.02801 |
| 60 | 11115 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 24/0/0/0 | 0.04478 / 0.04478 (≤thr 0×) | 0.03531 / 0.03749 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 70/0/2/0 | 0.03107 / 0.0358 (≤thr 0×) | 0.02634 / 0.02812 |
| 60 | 11116 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 21/0/3/0 | 0.02039 / 0.02039 (≤thr 0×) | 0.03542 / 0.03986 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 68/0/4/0 | 0.02694 / 0.0305 (≤thr 0×) | 0.02558 / 0.02558 |
| 60 | 11117 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 3/0/21/0 | 0.02185 / 0.02324 (≤thr 0×) | 0.03753 / 0.04098 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 65/0/7/0 | 0.02079 / 0.04772 (≤thr 0×) | 0.02898 / 0.02908 |
| 60 | 11118 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 8/3/13/0 | 0.01711 / 0.03717 (≤thr 3×) | 0.03432 / 0.04192 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 60/0/12/0 | 0.02146 / 0.04509 (≤thr 0×) | 0.0312 / 0.03132 |
| 60 | 11119 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 1/0/23/0 | 0.02564 / 0.0302 (≤thr 0×) | 0.0397 / 0.0432 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 47/0/25/0 | 0.01838 / 0.02554 (≤thr 0×) | 0.03276 / 0.03276 |
| 60 | 11120 | budget_forced (600) | 0/24; first none; longest 0; 3rd consec. none | 3/0/21/0 | 0.02292 / 0.02647 (≤thr 0×) | 0.03971 / 0.03971 | no_candidate_budget_exhausted (1800) | 0/72; first none; longest 0 | 67/0/5/0 | 0.02983 / 0.04044 (≤thr 0×) | 0.02686 / 0.02686 |

汇总：

- **B（每 q 480 次检查）**：eligible 0。没有一个 run 出现过 1 次 eligible，自然也没有 3 连。
  - q50：strategic 值从未 ≤ 0.02。失败类型为 strategic-only 130、both 350、conc-only 0、invalid 0。每 run B strategic 最小值 0.0264–0.1275，中位 0.0393。
  - q60：strategic 值 ≤ 0.02 共 10 次（s11101 3 次、s11107 4 次、s11118 3 次）。这 10 次的 concentration 在 0.0405–0.0432，都超过 0.04，所以记为 conc-only 失败。其余为 strategic-only 247、both 223、invalid 0。每 run 最小值 0.0165–0.0773，中位 0.0248。
- **C（每 q 1440 次检查）**：eligible 0，dReach/DW ≤ 0.01 的检查 0 次。
  - dReach/DW ≤ 0.02 的检查：q50 10 次，来自 3 个 run（11002、11003、11013）；q60 19 次，来自 4 个 run（11103、11106、11114、11119）。
  - 每次失败都含 strategic 失败：strategic-only q50 1235、q60 1294；both q50 205、q60 146；conc-only 0；invalid 0。

## 4. Verification（逐 seed）

阅读要点：

- **valid 与 refinement 都不等于战略阈值通过。**
  - “verifier valid” 只表示数值检查成立：网格、质量、PDL、有限值。
  - “refinement 通过” 只表示 dev 与 final 两层网格的数值差 ≤ 0.002·DW。
  - 战略阈值是 main：dReach_final/DW ≤ 0.01。
  - 本次 80 个 endpoint 层（40 个 run 的 dev 与 final）全部 valid，refinement 39/40 通过，但 main 0/40 通过。
- **Delta_max_all 是最大值，不是和。** Delta_max_all = max_t full_t，即各 stage 全域最大一步偏离中最大的那个。各 stage 最大值之和是 dfull。
- **endpoint 与最小检查点分开。** 每个 run 的 endpoint 都是 C1800 末的权重（diagnostic_terminal），从 `endpoint.pt` 重读后重新计算 dev 与 final。最小 development 检查点另列在 V4，不做 final，也不晋升。
- 表中数值都是 /DW（DW = 4），raw = 4 × /DW。逐项 raw 值见 `verification_by_seed.csv`、`runs.csv`、`verifier_stage_metrics.csv`。

#### V1. Verification of the saved endpoint (development and final tiers re-run on the reloaded weights)

Values /DW. `kind` = candidate or diagnostic_terminal (C-cap weights). Raw = 4 × /DW.

| q | seed | kind (update) | valid dev/final | invalid reasons | EXP_root dev / final | dReach dev / final | Delta_max_all dev / final | dfull final | abs ΔdReach dev−final | abs ΔEXP dev−final | abs ΔDelta_max_all dev−final | dense conc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | diagnostic_terminal (2800) | true/true | none | 0.009787 / 0.009881 | 0.02396 / 0.02403 | 0.01436 / 0.01436 | 0.02707 | 7.4e-05 | 9.4e-05 | 0 | 0.0284 |
| 50 | 11002 | diagnostic_terminal (2800) | true/true | none | 0.04854 / 0.04858 | 0.09137 / 0.09193 | 0.04577 / 0.04637 | 0.09193 | 0.00055 | 3.4e-05 | 0.00061 | 0.03461 |
| 50 | 11003 | diagnostic_terminal (2800) | true/true | none | 0.01882 / 0.01888 | 0.0536 / 0.0548 | 0.02769 / 0.02882 | 0.0548 | 0.0012 | 5.8e-05 | 0.0011 | 0.03523 |
| 50 | 11004 | diagnostic_terminal (2800) | true/true | none | 0.01726 / 0.01719 | 0.0463 / 0.0462 | 0.02337 / 0.02337 | 0.0462 | 9.3e-05 | 7.7e-05 | 1.1e-16 | 0.02989 |
| 50 | 11005 | diagnostic_terminal (2800) | true/true | none | 0.03164 / 0.03166 | 0.07527 / 0.07537 | 0.057 / 0.05704 | 0.07537 | 9.6e-05 | 2e-05 | 4.4e-05 | 0.02901 |
| 50 | 11006 | diagnostic_terminal (2800) | true/true | none | 0.01064 / 0.01067 | 0.03492 / 0.0351 | 0.0213 / 0.02153 | 0.0351 | 0.00018 | 2.7e-05 | 0.00022 | 0.03021 |
| 50 | 11007 | diagnostic_terminal (2800) | true/true | none | 0.008632 / 0.008581 | 0.04109 / 0.04112 | 0.02125 / 0.02129 | 0.04112 | 3.3e-05 | 5.1e-05 | 3.8e-05 | 0.03213 |
| 50 | 11008 | diagnostic_terminal (2800) | true/true | none | 0.04468 / 0.04466 | 0.1425 / 0.1448 | 0.09446 / 0.09446 | 0.1765 | 0.0023 | 1.6e-05 | 3.3e-09 | 0.02723 |
| 50 | 11009 | diagnostic_terminal (2800) | true/true | none | 0.01888 / 0.01886 | 0.05872 / 0.05873 | 0.02964 / 0.02965 | 0.05873 | 8.1e-06 | 2.1e-05 | 1.1e-05 | 0.02586 |
| 50 | 11010 | diagnostic_terminal (2800) | true/true | none | 0.01319 / 0.01314 | 0.0526 / 0.0527 | 0.02993 / 0.02995 | 0.0527 | 9.7e-05 | 5.2e-05 | 1.8e-05 | 0.02956 |
| 50 | 11011 | diagnostic_terminal (2800) | true/true | none | 0.02059 / 0.02056 | 0.06169 / 0.06202 | 0.03215 / 0.03215 | 0.06202 | 0.00034 | 2.9e-05 | 0 | 0.02903 |
| 50 | 11012 | diagnostic_terminal (2800) | true/true | none | 0.013 / 0.0131 | 0.05688 / 0.05771 | 0.03606 / 0.03673 | 0.05771 | 0.00083 | 0.0001 | 0.00067 | 0.033 |
| 50 | 11013 | diagnostic_terminal (2800) | true/true | none | 0.008475 / 0.008475 | 0.02401 / 0.02422 | 0.01836 / 0.01855 | 0.02422 | 0.0002 | 3.1e-07 | 0.00019 | 0.02655 |
| 50 | 11014 | diagnostic_terminal (2800) | true/true | none | 0.0144 / 0.01429 | 0.0466 / 0.04649 | 0.02819 / 0.02819 | 0.04649 | 0.00011 | 0.00011 | 7.3e-06 | 0.03018 |
| 50 | 11015 | diagnostic_terminal (2800) | true/true | none | 0.01062 / 0.01063 | 0.04014 / 0.04001 | 0.02539 / 0.02534 | 0.04001 | 0.00013 | 1.4e-05 | 4.7e-05 | 0.03186 |
| 50 | 11016 | diagnostic_terminal (2800) | true/true | none | 0.02155 / 0.02138 | 0.03801 / 0.03815 | 0.01898 / 0.01917 | 0.03815 | 0.00013 | 0.00017 | 0.00019 | 0.02895 |
| 50 | 11017 | diagnostic_terminal (2800) | true/true | none | 0.01659 / 0.01659 | 0.04699 / 0.0469 | 0.02911 / 0.02907 | 0.0469 | 8.9e-05 | 1.1e-06 | 3.6e-05 | 0.02994 |
| 50 | 11018 | diagnostic_terminal (2800) | true/true | none | 0.01857 / 0.01858 | 0.09502 / 0.09625 | 0.07117 / 0.07206 | 0.09625 | 0.0012 | 1.7e-05 | 0.00089 | 0.03453 |
| 50 | 11019 | diagnostic_terminal (2800) | true/true | none | 0.03239 / 0.03238 | 0.09145 / 0.09143 | 0.05593 / 0.05593 | 0.09143 | 2.6e-05 | 1.5e-05 | 1.1e-16 | 0.03077 |
| 50 | 11020 | diagnostic_terminal (2800) | true/true | none | 0.0344 / 0.03434 | 0.06916 / 0.06945 | 0.03255 / 0.03255 | 0.06945 | 0.00029 | 6.7e-05 | 1.1e-06 | 0.03012 |
| 60 | 11101 | diagnostic_terminal (2800) | true/true | none | 0.01314 / 0.01313 | 0.0383 / 0.0383 | 0.02529 / 0.02529 | 0.0383 | 3.4e-06 | 3.1e-06 | 7.5e-06 | 0.02714 |
| 60 | 11102 | diagnostic_terminal (2800) | true/true | none | 0.01213 / 0.01212 | 0.04642 / 0.0465 | 0.02858 / 0.02857 | 0.0465 | 8.7e-05 | 1.3e-05 | 1e-05 | 0.02641 |
| 60 | 11103 | diagnostic_terminal (2800) | true/true | none | 0.004322 / 0.004306 | 0.02053 / 0.02055 | 0.01476 / 0.01477 | 0.02594 | 2e-05 | 1.7e-05 | 9.2e-06 | 0.03284 |
| 60 | 11104 | diagnostic_terminal (2800) | true/true | none | 0.005758 / 0.005741 | 0.02453 / 0.0246 | 0.0127 / 0.01268 | 0.0246 | 7.2e-05 | 1.6e-05 | 1.9e-05 | 0.02892 |
| 60 | 11105 | diagnostic_terminal (2800) | true/true | none | 0.008804 / 0.008788 | 0.03297 / 0.033 | 0.0234 / 0.02341 | 0.033 | 3.3e-05 | 1.7e-05 | 7.3e-06 | 0.02952 |
| 60 | 11106 | diagnostic_terminal (2800) | true/true | none | 0.02717 / 0.02717 | 0.05321 / 0.05327 | 0.02993 / 0.02993 | 0.05327 | 5.3e-05 | 6.7e-06 | 2.2e-16 | 0.02721 |
| 60 | 11107 | diagnostic_terminal (2800) | true/true | none | 0.01746 / 0.01741 | 0.04186 / 0.04186 | 0.02101 / 0.02104 | 0.04186 | 4e-06 | 4.8e-05 | 2.7e-05 | 0.02687 |
| 60 | 11108 | diagnostic_terminal (2800) | true/true | none | 0.01311 / 0.0131 | 0.03553 / 0.03556 | 0.01831 / 0.0183 | 0.03556 | 3.2e-05 | 9.3e-06 | 3.5e-06 | 0.02643 |
| 60 | 11109 | diagnostic_terminal (2800) | true/true | none | 0.008153 / 0.008137 | 0.03463 / 0.03479 | 0.02318 / 0.02327 | 0.03479 | 0.00016 | 1.6e-05 | 8.6e-05 | 0.02607 |
| 60 | 11110 | diagnostic_terminal (2800) | true/true | none | 0.007238 / 0.007188 | 0.02696 / 0.02697 | 0.01646 / 0.01653 | 0.02697 | 1.5e-05 | 4.9e-05 | 6.9e-05 | 0.02586 |
| 60 | 11111 | diagnostic_terminal (2800) | true/true | none | 0.0136 / 0.01354 | 0.03617 / 0.03614 | 0.0172 / 0.0172 | 0.03614 | 3.6e-05 | 5.5e-05 | 1.1e-16 | 0.02792 |
| 60 | 11112 | diagnostic_terminal (2800) | true/true | none | 0.01795 / 0.01792 | 0.04501 / 0.04511 | 0.02976 / 0.02987 | 0.04511 | 0.00011 | 2.7e-05 | 0.00011 | 0.02627 |
| 60 | 11113 | diagnostic_terminal (2800) | true/true | none | 0.01713 / 0.0171 | 0.03691 / 0.03692 | 0.01926 / 0.01927 | 0.03692 | 9.4e-06 | 2.9e-05 | 1.4e-05 | 0.02588 |
| 60 | 11114 | diagnostic_terminal (2800) | true/true | none | 0.024 / 0.02389 | 0.08223 / 0.08238 | 0.04897 / 0.04902 | 0.08238 | 0.00015 | 0.0001 | 4.8e-05 | 0.02801 |
| 60 | 11115 | diagnostic_terminal (2800) | true/true | none | 0.01331 / 0.01326 | 0.0358 / 0.03586 | 0.02447 / 0.02449 | 0.04932 | 5.8e-05 | 5e-05 | 1.9e-05 | 0.02812 |
| 60 | 11116 | diagnostic_terminal (2800) | true/true | none | 0.01099 / 0.01099 | 0.0305 / 0.03063 | 0.01756 / 0.01766 | 0.03063 | 0.00013 | 2.7e-06 | 0.00011 | 0.02558 |
| 60 | 11117 | diagnostic_terminal (2800) | true/true | none | 0.01473 / 0.01469 | 0.04772 / 0.04787 | 0.0223 / 0.02237 | 0.04787 | 0.00015 | 4.8e-05 | 6.8e-05 | 0.02908 |
| 60 | 11118 | diagnostic_terminal (2800) | true/true | none | 0.01736 / 0.01724 | 0.04509 / 0.04554 | 0.02382 / 0.02436 | 0.04554 | 0.00045 | 0.00011 | 0.00054 | 0.03132 |
| 60 | 11119 | diagnostic_terminal (2800) | true/true | none | 0.01002 / 0.00998 | 0.02554 / 0.02556 | 0.01574 / 0.01575 | 0.02556 | 2.2e-05 | 4.5e-05 | 1.4e-05 | 0.03277 |
| 60 | 11120 | diagnostic_terminal (2800) | true/true | none | 0.01018 / 0.01018 | 0.04044 / 0.04054 | 0.02726 / 0.02728 | 0.04054 | 0.0001 | 2.1e-06 | 1.6e-05 | 0.02686 |

#### V2. Certification components (thresholds: main dReach_final ≤ 0.01; refine ≤ 0.002; dense conc ≤ 0.04)

| q | seed | kind | valid dev/final | main | refine dReach | refine EXP | dense conc | numeric (1-5) | has_candidate | certification | final_joint_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11002 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11003 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11004 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11005 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11006 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11007 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11008 | diagnostic_terminal | true/true | false | false | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11009 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11010 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11011 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11012 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11013 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11014 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11015 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11016 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11017 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11018 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11019 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 50 | 11020 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11101 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11102 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11103 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11104 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11105 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11106 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11107 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11108 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11109 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11110 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11111 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11112 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11113 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11114 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11115 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11116 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11117 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11118 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11119 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |
| 60 | 11120 | diagnostic_terminal | true/true | false | true | true | true | false | false | not_applicable_no_candidate | false |

#### V3. Stage-wise final-tier deviations at the endpoint (/DW)

reach_t = max over the BR-reachable mask R_t of the one-step gain δ_t; dReach = Σ_t reach_t. full_t = max over D_t of δ_t; Delta_max_all = max_t full_t (a maximum, not a sum). cg_t = max over D_t of the continuation gain V_t^BR − V_t^mean.

| q | seed | reach t1 @ d | reach t2 @ d | reach t3 @ d | full t2 @ d | full t3 @ d | cg t2 @ d | cg t3 @ d |
|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | 0.002978 @ 0 | 0.006692 @ 12 | 0.01436 @ -48 | 0.009728 @ 166 | 0.01436 @ -48 | 0.0178 @ 160 | 0.01436 @ -48 |
| 50 | 11002 | 0.032 @ 0 | 0.04637 @ -6 | 0.01356 @ 104 | 0.04637 @ -6 | 0.01356 @ 104 | 0.05133 @ -6 | 0.01356 @ 104 |
| 50 | 11003 | 0.006868 @ 0 | 0.02882 @ -6 | 0.01911 @ -20 | 0.02882 @ -6 | 0.01911 @ -20 | 0.03382 @ -6 | 0.01911 @ -20 |
| 50 | 11004 | 0.001119 @ 0 | 0.02171 @ 52 | 0.02337 @ -52 | 0.02171 @ 52 | 0.02337 @ -52 | 0.02961 @ -38 | 0.02337 @ -52 |
| 50 | 11005 | 0.00556 @ 0 | 0.05704 @ 28 | 0.01277 @ 100 | 0.05704 @ 28 | 0.01277 @ 100 | 0.06416 @ 28 | 0.01277 @ 100 |
| 50 | 11006 | 0.0007556 @ 0 | 0.01282 @ -32 | 0.02153 @ -54 | 0.01282 @ -32 | 0.02153 @ -54 | 0.02054 @ -32 | 0.02153 @ -54 |
| 50 | 11007 | 0.001311 @ 0 | 0.02129 @ 24 | 0.01853 @ 102 | 0.02129 @ 24 | 0.01853 @ 102 | 0.02479 @ 24 | 0.01853 @ 102 |
| 50 | 11008 | 0.001359 @ 0 | 0.06277 @ -106 | 0.08066 @ -14 | 0.09446 @ -200 | 0.08066 @ -14 | 0.1645 @ -200 | 0.08066 @ -14 |
| 50 | 11009 | 0.003897 @ 0 | 0.02518 @ 64 | 0.02965 @ -10 | 0.02518 @ 64 | 0.02965 @ -10 | 0.02825 @ 64 | 0.02965 @ -10 |
| 50 | 11010 | 2.839e-07 @ 0 | 0.02995 @ 60 | 0.02275 @ -10 | 0.02995 @ 60 | 0.02275 @ -10 | 0.03216 @ 60 | 0.02275 @ -10 |
| 50 | 11011 | 0.003027 @ 0 | 0.02685 @ 22 | 0.03215 @ 104 | 0.02685 @ 22 | 0.03215 @ 104 | 0.03622 @ 24 | 0.03215 @ 104 |
| 50 | 11012 | 0.001957 @ 0 | 0.03673 @ -2 | 0.01903 @ 102 | 0.03673 @ -2 | 0.01903 @ 102 | 0.03891 @ -2 | 0.01903 @ 102 |
| 50 | 11013 | 0.001303 @ 0 | 0.004363 @ 24 | 0.01855 @ -10 | 0.004363 @ 24 | 0.01855 @ -10 | 0.01039 @ -24 | 0.01855 @ -10 |
| 50 | 11014 | 0.00137 @ 0 | 0.02819 @ 16 | 0.01692 @ -60 | 0.02819 @ 16 | 0.01692 @ -60 | 0.03029 @ 14 | 0.01692 @ -60 |
| 50 | 11015 | 0.0003292 @ 0 | 0.02534 @ 12 | 0.01434 @ -60 | 0.02534 @ 12 | 0.01434 @ -60 | 0.02911 @ 12 | 0.01434 @ -60 |
| 50 | 11016 | 0.007949 @ 0 | 0.01917 @ -6 | 0.01103 @ -58 | 0.01917 @ -6 | 0.01103 @ -58 | 0.02433 @ -6 | 0.01103 @ -58 |
| 50 | 11017 | 0.001164 @ 0 | 0.02907 @ -40 | 0.01667 @ -60 | 0.02907 @ -40 | 0.01667 @ -60 | 0.03441 @ -40 | 0.01667 @ -60 |
| 50 | 11018 | 0.001846 @ 0 | 0.07206 @ -2 | 0.02234 @ -14 | 0.07206 @ -2 | 0.02234 @ -14 | 0.08021 @ -2 | 0.02234 @ -14 |
| 50 | 11019 | 0 @ 0 | 0.0355 @ -36 | 0.05593 @ -40 | 0.0355 @ -36 | 0.05593 @ -40 | 0.06483 @ -36 | 0.05593 @ -40 |
| 50 | 11020 | 0.02349 @ 0 | 0.03255 @ -4 | 0.01341 @ -58 | 0.03255 @ -4 | 0.01341 @ -58 | 0.03633 @ -4 | 0.01341 @ -58 |
| 60 | 11101 | 0.0004361 @ 0 | 0.02529 @ 60 | 0.01257 @ -8 | 0.02529 @ 60 | 0.01257 @ -8 | 0.02704 @ 60 | 0.01257 @ -8 |
| 60 | 11102 | 7.869e-05 @ 0 | 0.02857 @ 36 | 0.01786 @ 122 | 0.02857 @ 36 | 0.01786 @ 122 | 0.03214 @ 36 | 0.01786 @ 122 |
| 60 | 11103 | 5.779e-08 @ 0 | 0.009384 @ 60 | 0.01116 @ -116 | 0.01477 @ 220 | 0.01116 @ -116 | 0.01953 @ 218 | 0.01116 @ -116 |
| 60 | 11104 | 0.0008382 @ 0 | 0.01268 @ 4 | 0.01108 @ 126 | 0.01268 @ 4 | 0.01108 @ 126 | 0.01326 @ 2 | 0.01108 @ 126 |
| 60 | 11105 | 8.262e-05 @ 0 | 0.009508 @ -54 | 0.02341 @ -10 | 0.009508 @ -54 | 0.02341 @ -10 | 0.01698 @ -54 | 0.02341 @ -10 |
| 60 | 11106 | 0.004347 @ 0 | 0.01899 @ -34 | 0.02993 @ 36 | 0.01899 @ -34 | 0.02993 @ 36 | 0.03711 @ -34 | 0.02993 @ 36 |
| 60 | 11107 | 0.0002385 @ 0 | 0.02058 @ -56 | 0.02104 @ -10 | 0.02058 @ -56 | 0.02104 @ -10 | 0.03107 @ -56 | 0.02104 @ -10 |
| 60 | 11108 | 0.003511 @ 0 | 0.0183 @ 52 | 0.01375 @ 126 | 0.0183 @ 52 | 0.01375 @ 126 | 0.02208 @ 54 | 0.01375 @ 126 |
| 60 | 11109 | 0.0001015 @ 0 | 0.02327 @ 74 | 0.01142 @ -6 | 0.02327 @ 74 | 0.01142 @ -6 | 0.02447 @ 74 | 0.01142 @ -6 |
| 60 | 11110 | 0.000226 @ 0 | 0.01653 @ 78 | 0.01022 @ -116 | 0.01653 @ 78 | 0.01022 @ -116 | 0.01833 @ 78 | 0.01022 @ -116 |
| 60 | 11111 | 0.004342 @ 0 | 0.0146 @ 24 | 0.0172 @ -8 | 0.0146 @ 24 | 0.0172 @ -8 | 0.01809 @ 24 | 0.0172 @ -8 |
| 60 | 11112 | 0.002229 @ 0 | 0.02987 @ 34 | 0.01301 @ 120 | 0.02987 @ 34 | 0.01301 @ 120 | 0.03252 @ 34 | 0.01301 @ 120 |
| 60 | 11113 | 0.001107 @ 0 | 0.01655 @ -46 | 0.01927 @ 34 | 0.01655 @ -46 | 0.01927 @ 34 | 0.02599 @ 54 | 0.01927 @ 34 |
| 60 | 11114 | 0.00151 @ 0 | 0.03185 @ -2 | 0.04902 @ -22 | 0.03185 @ -2 | 0.04902 @ -22 | 0.05006 @ -2 | 0.04902 @ -22 |
| 60 | 11115 | 0.004106 @ 0 | 0.01103 @ -46 | 0.02072 @ 122 | 0.02449 @ 220 | 0.02072 @ 122 | 0.03594 @ 220 | 0.02072 @ 122 |
| 60 | 11116 | 0.0005683 @ 0 | 0.01766 @ 34 | 0.0124 @ -8 | 0.01766 @ 34 | 0.0124 @ -8 | 0.02187 @ 34 | 0.0124 @ -8 |
| 60 | 11117 | 0.004099 @ 0 | 0.0214 @ 6 | 0.02237 @ -118 | 0.0214 @ 6 | 0.02237 @ -118 | 0.02311 @ 4 | 0.02237 @ -118 |
| 60 | 11118 | 0.0106 @ 0 | 0.02436 @ -2 | 0.01059 @ 124 | 0.02436 @ -2 | 0.01059 @ 124 | 0.02596 @ -2 | 0.01059 @ 124 |
| 60 | 11119 | 0.002427 @ 0 | 0.007381 @ -42 | 0.01575 @ -10 | 0.007381 @ -42 | 0.01575 @ -10 | 0.01407 @ -42 | 0.01575 @ -10 |
| 60 | 11120 | 0.0006852 @ 0 | 0.02728 @ 60 | 0.01257 @ -118 | 0.02728 @ 60 | 0.01257 @ -118 | 0.02883 @ 60 | 0.01257 @ -118 |

#### V4. Minimum valid C development call vs the actual endpoint (reported separately)

The minimum is diagnostic only: it is never promoted and gets no final evaluation. `endpoint dev call` is the development call made during training at the endpoint update (for a diagnostic terminal, the C1800 check).

| q | seed | endpoint kind | min valid C dev dReach/DW @ update | EXP_root/DW at min | Delta_max_all/DW at min | conc at min | endpoint dev call dReach/DW @ update | endpoint re-run dev dReach/DW | endpoint final dReach/DW |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | diagnostic_terminal | 0.02161 @ 2775 | 0.007754 | 0.01569 | 0.02859 | 0.02396 @ 2800 | 0.02396 | 0.02403 |
| 50 | 11002 | diagnostic_terminal | 0.01892 @ 2200 | 0.006657 | 0.01149 | 0.03708 | 0.09137 @ 2800 | 0.09137 | 0.09193 |
| 50 | 11003 | diagnostic_terminal | 0.01772 @ 2575 | 0.004959 | 0.009763 | 0.03631 | 0.0536 @ 2800 | 0.0536 | 0.0548 |
| 50 | 11004 | diagnostic_terminal | 0.02306 @ 2525 | 0.009252 | 0.01214 | 0.03272 | 0.0463 @ 2800 | 0.0463 | 0.0462 |
| 50 | 11005 | diagnostic_terminal | 0.05811 @ 2625 | 0.01291 | 0.03458 | 0.03101 | 0.07527 @ 2800 | 0.07527 | 0.07537 |
| 50 | 11006 | diagnostic_terminal | 0.02355 @ 2650 | 0.009249 | 0.01485 | 0.0303 | 0.03492 @ 2800 | 0.03492 | 0.0351 |
| 50 | 11007 | diagnostic_terminal | 0.04109 @ 2800 | 0.008632 | 0.02125 | 0.03213 | 0.04109 @ 2800 | 0.04109 | 0.04112 |
| 50 | 11008 | diagnostic_terminal | 0.1168 @ 1350 | 0.0478 | 0.08804 | 0.0324 | 0.1425 @ 2800 | 0.1425 | 0.1448 |
| 50 | 11009 | diagnostic_terminal | 0.0447 @ 2175 | 0.01212 | 0.02457 | 0.02874 | 0.05872 @ 2800 | 0.05872 | 0.05873 |
| 50 | 11010 | diagnostic_terminal | 0.04713 @ 2750 | 0.01106 | 0.02415 | 0.03014 | 0.0526 @ 2800 | 0.0526 | 0.0527 |
| 50 | 11011 | diagnostic_terminal | 0.0521 @ 2425 | 0.01771 | 0.0335 | 0.03069 | 0.06169 @ 2800 | 0.06169 | 0.06202 |
| 50 | 11012 | diagnostic_terminal | 0.02342 @ 1800 | 0.009741 | 0.01132 | 0.03689 | 0.05688 @ 2800 | 0.05688 | 0.05771 |
| 50 | 11013 | diagnostic_terminal | 0.01543 @ 2750 | 0.005598 | 0.01039 | 0.02674 | 0.02401 @ 2800 | 0.02401 | 0.02422 |
| 50 | 11014 | diagnostic_terminal | 0.0238 @ 2050 | 0.007063 | 0.0133 | 0.03513 | 0.0466 @ 2800 | 0.0466 | 0.04649 |
| 50 | 11015 | diagnostic_terminal | 0.03354 @ 2150 | 0.009139 | 0.01815 | 0.03592 | 0.04014 @ 2800 | 0.04014 | 0.04001 |
| 50 | 11016 | diagnostic_terminal | 0.03524 @ 2525 | 0.01621 | 0.01747 | 0.03014 | 0.03801 @ 2800 | 0.03801 | 0.03815 |
| 50 | 11017 | diagnostic_terminal | 0.03089 @ 2025 | 0.01213 | 0.01977 | 0.03284 | 0.04699 @ 2800 | 0.04699 | 0.0469 |
| 50 | 11018 | diagnostic_terminal | 0.03144 @ 2175 | 0.01328 | 0.01678 | 0.03622 | 0.09502 @ 2800 | 0.09502 | 0.09625 |
| 50 | 11019 | diagnostic_terminal | 0.03049 @ 1425 | 0.01135 | 0.01644 | 0.03746 | 0.09145 @ 2800 | 0.09145 | 0.09143 |
| 50 | 11020 | diagnostic_terminal | 0.02529 @ 2575 | 0.00726 | 0.01479 | 0.03016 | 0.06916 @ 2800 | 0.06916 | 0.06945 |
| 60 | 11101 | diagnostic_terminal | 0.02531 @ 2100 | 0.009342 | 0.01452 | 0.03395 | 0.0383 @ 2800 | 0.0383 | 0.0383 |
| 60 | 11102 | diagnostic_terminal | 0.04573 @ 2400 | 0.0109 | 0.02609 | 0.02902 | 0.04642 @ 2800 | 0.04642 | 0.0465 |
| 60 | 11103 | diagnostic_terminal | 0.01595 @ 2300 | 0.005749 | 0.01855 | 0.03777 | 0.02053 @ 2800 | 0.02053 | 0.02055 |
| 60 | 11104 | diagnostic_terminal | 0.02061 @ 2775 | 0.004954 | 0.01015 | 0.02869 | 0.02453 @ 2800 | 0.02453 | 0.0246 |
| 60 | 11105 | diagnostic_terminal | 0.02721 @ 2775 | 0.007573 | 0.02041 | 0.02982 | 0.03297 @ 2800 | 0.03297 | 0.033 |
| 60 | 11106 | diagnostic_terminal | 0.01557 @ 2525 | 0.007004 | 0.01054 | 0.02954 | 0.05321 @ 2800 | 0.05321 | 0.05327 |
| 60 | 11107 | diagnostic_terminal | 0.02309 @ 2250 | 0.007545 | 0.01288 | 0.02857 | 0.04186 @ 2800 | 0.04186 | 0.04186 |
| 60 | 11108 | diagnostic_terminal | 0.03397 @ 2575 | 0.01143 | 0.01969 | 0.02712 | 0.03553 @ 2800 | 0.03553 | 0.03556 |
| 60 | 11109 | diagnostic_terminal | 0.02222 @ 1950 | 0.007753 | 0.01111 | 0.03033 | 0.03463 @ 2800 | 0.03463 | 0.03479 |
| 60 | 11110 | diagnostic_terminal | 0.0238 @ 2725 | 0.008588 | 0.01378 | 0.02621 | 0.02696 @ 2800 | 0.02696 | 0.02697 |
| 60 | 11111 | diagnostic_terminal | 0.02489 @ 2625 | 0.009964 | 0.0115 | 0.02891 | 0.03617 @ 2800 | 0.03617 | 0.03614 |
| 60 | 11112 | diagnostic_terminal | 0.03812 @ 2675 | 0.01386 | 0.02514 | 0.02657 | 0.04501 @ 2800 | 0.04501 | 0.04511 |
| 60 | 11113 | diagnostic_terminal | 0.03473 @ 2775 | 0.01558 | 0.01738 | 0.02599 | 0.03691 @ 2800 | 0.03691 | 0.03692 |
| 60 | 11114 | diagnostic_terminal | 0.0176 @ 1850 | 0.006488 | 0.01012 | 0.0343 | 0.08223 @ 2800 | 0.08223 | 0.08238 |
| 60 | 11115 | diagnostic_terminal | 0.03107 @ 1800 | 0.01152 | 0.01887 | 0.03145 | 0.0358 @ 2800 | 0.0358 | 0.03586 |
| 60 | 11116 | diagnostic_terminal | 0.02694 @ 2725 | 0.01135 | 0.0136 | 0.02679 | 0.0305 @ 2800 | 0.0305 | 0.03063 |
| 60 | 11117 | diagnostic_terminal | 0.02079 @ 1825 | 0.007819 | 0.01025 | 0.0336 | 0.04772 @ 2800 | 0.04772 | 0.04787 |
| 60 | 11118 | diagnostic_terminal | 0.02146 @ 2100 | 0.007377 | 0.01226 | 0.03245 | 0.04509 @ 2800 | 0.04509 | 0.04554 |
| 60 | 11119 | diagnostic_terminal | 0.01838 @ 2500 | 0.008248 | 0.01299 | 0.03385 | 0.02554 @ 2800 | 0.02554 | 0.02556 |
| 60 | 11120 | diagnostic_terminal | 0.02983 @ 2125 | 0.00777 | 0.01577 | 0.02949 | 0.04044 @ 2800 | 0.04044 | 0.04054 |

汇总与说明：

- **有效性**：训练中共 4000 次 development 调用（A 160、B 960、C 2880），全部 valid，没有异常。endpoint 的 dev 与 final 两层 40/40 valid。
- **main**：endpoint final dReach/DW 在 q50 为 0.0240–0.1448（中位 0.0537），q60 为 0.0205–0.0824（中位 0.0365）。全部 > 0.01，main_pass 0/40。
- **refinement**：
  - |ΔEXP|/DW ≤ 1.7e−4，40/40 通过。
  - |ΔdReach|/DW 除 q50 s11008 外都 ≤ 1.24e−3（最大为 q50 s11018）。
  - s11008 为 0.00233 > 0.002，refine_dreach_pass=false。原因是它的 stage-2 reach 最大值落在 BR-reachable 集 R_2 的左端点：dev 网格上 R_2=[−104, 92]，final 网格上 R_2=[−106, 92]。δ2 越接近 D2 边界越大，final 网格多出的端点 d=−106 让 stage-2 reach 贡献增加 0.00217·DW。
  - 这是 R_2 边界上的分辨率效应，两层都 valid，不是 verifier 无效。该 run 本来就没有 candidate，不影响任何认证结论。
- **dense concentration**（D1–D3 全域，step 0.05）：最大 0.0352，40/40 ≤ 0.04。
- **认证状态**：numeric_thresholds_pass 0/40，不满足的都是 main，s11008 另有 refine dReach。certification 40/40 为 `not_applicable_no_candidate`，final_joint_pass 0/40。
- **R_t 之外的偏离**：4 个 run 的 final 层 dfull > dReach（q50 s11001、s11008，q60 s11103、s11115）。这些 run 的全域最大偏离在 D2 边界或 R_2 之外，例如 s11008 的 full_2 在 d=−200 为 0.0945，而 reach_2 为 0.0628。Delta_max_all 正是用来显示这类 reach 准则覆盖不到的偏离。本报告不声称全域 MPE 认证。
- **最小 dev 检查点与 endpoint（V4）**：
  - 最小 valid C dReach/DW：q50 为 0.0154–0.1168（中位 0.0307），q60 为 0.0156–0.0457（中位 0.0243）。它们出现在 global update 1350–2800。
  - endpoint 的 dev dReach 是这个最小值的 1.00–4.83 倍。例如 q50 s11002 最小值 0.0189 出现在 u2200，而终点是 0.0914。C 中 dReach 不单调下降。
- **一致性**（均由 completeness 逐 run 检查）：
  - 40/40 重读的 endpoint 与内存中的 actor 逐位相同；
  - endpoint 上重算的 dev 与训练中 C1800 那次调用的差 ≤ 1e−12；
  - min_dev 权重的 replay 与记录的差 ≤ 1e−12。

## 5. Economic policies（实际终点）

**分组**：两个 q 的 certified candidate 都是 n=0，uncertified candidate 也都是 n=0；diagnostic_terminal 在 q50、q60 各 n=20。因此下面描述的都是 **diagnostic_terminal（C1800 末权重）** 的策略。没有一个是 candidate，更不能称为已认证的均衡。

**设置**：

- mean-policy 与 stochastic-policy self-play 分开计算。每种 mode 3 个 replicates × 200000 episodes（chunk 10000）。
- 评估 RNG 独立：SeedSequence([9005000, seed, q, 100, mode, rep, stream])，不消耗训练 RNG。
- E[e_t] 是从 root（d=0）出发的 self-play occupancy 下，代表性努力 X=(e0+e1)/2 的期望。MCSE 以 episode 为单位。
- self-play visitation 与 BR-chain（verifier 节点上的 PMF）在 `state_visitation.csv` 中用 origin 区分，不合并。

#### E. Economic policy of the actual endpoint (per seed)

E[e_t]: representative X=(e0+e1)/2 under root self-play from d=0; pooled 3 × 200000 episodes per mode; (MCSE, episode unit). LF = paired leader − follower effort at stage t (ties excluded). sd(gap) = SD of the pre-action physical gap under mean self-play.

| q | seed | group | e1(0) | e2(0) | e3(0) | mean E[e2] | mean E[e3] | stoch E[e2] | stoch E[e3] | LF t2 (mean mode) | LF t3 (mean mode) | sd(gap) t2/t3 | MC U − DP V1 (z) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | diagnostic_terminal | 41 | 49.52 | 62.3 | 31.196 (0.016) | 29.262 (0.023) | 31.111 (0.016) | 29.243 (0.023) | 19.69 (0.0094) | 10.09 (0.0059) | 40.8 / 70.1 | -0.00062 (-0.99) |
| 50 | 11002 | diagnostic_terminal | 55.73 | 60.37 | 66.41 | 38.495 (0.019) | 37.253 (0.028) | 38.418 (0.019) | 37.165 (0.028) | 20.09 (0.0093) | -1.65 (0.007) | 40.9 / 70.7 | 0.00053 (0.63) |
| 50 | 11003 | diagnostic_terminal | 51.06 | 57 | 71.85 | 34.294 (0.018) | 34.860 (0.029) | 34.174 (0.018) | 34.775 (0.029) | 20.73 (0.013) | 4.10 (0.0028) | 40.8 / 70.7 | -0.00039 (-0.46) |
| 50 | 11004 | diagnostic_terminal | 40.91 | 52.67 | 56.18 | 27.824 (0.02) | 28.233 (0.023) | 27.740 (0.02) | 28.190 (0.023) | 17.39 (0.013) | 0.48 (0.0023) | 40.8 / 67.9 | 0.00048 (0.78) |
| 50 | 11005 | diagnostic_terminal | 32.7 | 52.78 | 58.35 | 27.092 (0.02) | 36.338 (0.023) | 27.008 (0.02) | 36.250 (0.023) | -2.82 (0.011) | 10.04 (0.0056) | 40.8 / 59.5 | -0.0008 (-1.21) |
| 50 | 11006 | diagnostic_terminal | 43.77 | 55.13 | 56.54 | 29.649 (0.018) | 26.507 (0.024) | 29.559 (0.018) | 26.441 (0.024) | 22.37 (0.013) | 3.85 (0.0039) | 40.9 / 72.0 | -0.00033 (-0.54) |
| 50 | 11007 | diagnostic_terminal | 43.84 | 49.13 | 61.52 | 31.945 (0.016) | 35.747 (0.024) | 31.838 (0.016) | 35.593 (0.024) | 2.29 (0.014) | 6.22 (0.0035) | 40.8 / 63.3 | -0.00065 (-0.98) |
| 50 | 11008 | diagnostic_terminal | 42.75 | 35.22 | 30.93 | 35.220 (3.9e-06) | 30.928 (1.3e-06) | 35.220 (0.0023) | 30.929 (0.0022) | -0.38 (0.00035) | -0.15 (0.00014) | 40.9 / 57.5 | -1.5e-07 (-1.24) |
| 50 | 11009 | diagnostic_terminal | 46.26 | 47.36 | 46.94 | 29.771 (0.019) | 31.108 (0.018) | 29.734 (0.019) | 31.077 (0.018) | 5.83 (0.0044) | -0.95 (0.0012) | 40.9 / 61.1 | 0.0004 (0.82) |
| 50 | 11010 | diagnostic_terminal | 42.09 | 44.38 | 50.74 | 28.217 (0.016) | 31.616 (0.018) | 28.124 (0.016) | 31.528 (0.019) | 10.57 (0.011) | 0.92 (0.0036) | 40.8 / 62.8 | -0.00046 (-0.96) |
| 50 | 11011 | diagnostic_terminal | 34.24 | 42.85 | 54.71 | 28.920 (0.014) | 32.610 (0.02) | 28.864 (0.014) | 32.567 (0.02) | 1.14 (0.017) | 4.74 (0.009) | 40.8 / 64.1 | -0.00039 (-0.75) |
| 50 | 11012 | diagnostic_terminal | 40.22 | 60.67 | 64.91 | 36.254 (0.018) | 33.250 (0.025) | 36.173 (0.018) | 33.158 (0.025) | 22.55 (0.01) | -0.30 (0.0087) | 40.8 / 75.1 | -0.00027 (-0.35) |
| 50 | 11013 | diagnostic_terminal | 39.69 | 49.17 | 53.09 | 30.746 (0.016) | 30.482 (0.02) | 30.698 (0.016) | 30.422 (0.02) | 10.28 (0.0065) | 4.33 (0.002) | 40.8 / 65.7 | -0.00037 (-0.70) |
| 50 | 11014 | diagnostic_terminal | 41.76 | 52.99 | 59 | 29.730 (0.02) | 26.874 (0.026) | 29.632 (0.02) | 26.862 (0.026) | 23.83 (0.018) | 0.79 (0.0041) | 40.8 / 72.6 | 8.6e-05 (0.13) |
| 50 | 11015 | diagnostic_terminal | 45.21 | 52.89 | 68.86 | 31.001 (0.018) | 29.332 (0.03) | 30.955 (0.018) | 29.316 (0.03) | 24.76 (0.016) | 6.34 (0.0053) | 40.9 / 73.2 | 0.00035 (0.44) |
| 50 | 11016 | diagnostic_terminal | 48.6 | 57.28 | 55.85 | 30.603 (0.022) | 32.193 (0.024) | 30.582 (0.022) | 32.149 (0.024) | 4.08 (0.016) | 0.44 (0.0068) | 40.8 / 64.7 | 0.0006 (0.83) |
| 50 | 11017 | diagnostic_terminal | 43.33 | 56.95 | 55.26 | 27.453 (0.021) | 26.620 (0.024) | 27.362 (0.022) | 26.556 (0.025) | 21.63 (0.018) | 5.45 (0.0051) | 40.9 / 71.4 | 0.0002 (0.31) |
| 50 | 11018 | diagnostic_terminal | 48.84 | 67.59 | 74.81 | 34.579 (0.022) | 30.761 (0.032) | 34.511 (0.022) | 30.745 (0.032) | 19.79 (0.01) | 7.86 (0.0058) | 40.8 / 72.2 | 0.0016 (1.60) |
| 50 | 11019 | diagnostic_terminal | 44.13 | 43.7 | 43.31 | 25.192 (0.015) | 23.030 (0.017) | 25.135 (0.015) | 22.988 (0.017) | 14.82 (0.012) | 1.77 (0.0023) | 40.8 / 66.0 | 0.00016 (0.43) |
| 50 | 11020 | diagnostic_terminal | 56.18 | 61.23 | 62.15 | 33.648 (0.022) | 30.183 (0.026) | 33.566 (0.022) | 30.108 (0.026) | 20.28 (0.014) | 2.57 (0.0019) | 40.8 / 70.2 | 0.00056 (0.71) |
| 60 | 11101 | diagnostic_terminal | 31.94 | 40.31 | 43.93 | 22.205 (0.015) | 27.519 (0.016) | 22.163 (0.015) | 27.461 (0.016) | 7.74 (0.0084) | 1.42 (0.0033) | 49.0 / 72.6 | -5.5e-05 (-0.15) |
| 60 | 11102 | diagnostic_terminal | 29.91 | 33.52 | 50.37 | 24.180 (0.013) | 31.215 (0.019) | 24.149 (0.013) | 31.175 (0.019) | -6.40 (0.015) | 1.36 (0.0078) | 49.0 / 70.4 | -4.8e-05 (-0.11) |
| 60 | 11103 | diagnostic_terminal | 34.51 | 43.18 | 55.85 | 27.762 (0.015) | 33.408 (0.021) | 27.710 (0.015) | 33.360 (0.021) | 10.81 (0.0052) | -0.02 (0.00082) | 49.0 / 75.8 | -0.00018 (-0.35) |
| 60 | 11104 | diagnostic_terminal | 37.61 | 50.75 | 52.31 | 32.376 (0.015) | 30.487 (0.019) | 32.318 (0.016) | 30.435 (0.019) | 12.18 (0.0057) | 1.82 (0.0017) | 49.0 / 76.7 | -0.00054 (-1.05) |
| 60 | 11105 | diagnostic_terminal | 34.27 | 39.34 | 38.72 | 26.463 (0.013) | 28.065 (0.012) | 26.406 (0.013) | 27.996 (0.013) | 11.69 (0.0053) | 0.95 (0.0018) | 49.0 / 77.9 | -0.00033 (-1.08) |
| 60 | 11106 | diagnostic_terminal | 25.54 | 38.63 | 38.19 | 21.270 (0.011) | 21.723 (0.014) | 21.227 (0.011) | 21.711 (0.014) | 5.54 (0.0025) | -4.22 (0.0073) | 49.0 / 73.2 | -9e-06 (-0.03) |
| 60 | 11107 | diagnostic_terminal | 32.7 | 40.02 | 38.74 | 23.955 (0.015) | 22.870 (0.014) | 23.952 (0.015) | 22.849 (0.014) | 13.41 (0.01) | -3.90 (0.0046) | 49.0 / 77.2 | 5.8e-05 (0.18) |
| 60 | 11108 | diagnostic_terminal | 39.19 | 47.25 | 52.2 | 24.745 (0.017) | 32.010 (0.019) | 24.665 (0.018) | 31.941 (0.019) | 3.29 (0.0047) | -0.16 (0.0048) | 48.9 / 72.4 | -0.001 (-2.04) |
| 60 | 11109 | diagnostic_terminal | 35.64 | 41.92 | 44.74 | 25.735 (0.016) | 28.160 (0.017) | 25.728 (0.016) | 28.101 (0.017) | 11.56 (0.012) | 0.52 (0.004) | 49.0 / 75.3 | 0.00013 (0.32) |
| 60 | 11110 | diagnostic_terminal | 33.53 | 43.04 | 47.31 | 27.427 (0.017) | 30.525 (0.018) | 27.362 (0.017) | 30.508 (0.018) | 14.48 (0.012) | -0.53 (0.0029) | 48.9 / 78.2 | -0.00013 (-0.29) |
| 60 | 11111 | diagnostic_terminal | 28.72 | 43.85 | 41.36 | 27.228 (0.015) | 26.767 (0.016) | 27.158 (0.015) | 26.699 (0.016) | 17.05 (0.013) | 2.38 (0.0013) | 49.0 / 79.2 | -0.00098 (-2.54) |
| 60 | 11112 | diagnostic_terminal | 27.32 | 40.93 | 49.53 | 21.343 (0.017) | 30.275 (0.019) | 21.325 (0.017) | 30.260 (0.02) | -2.46 (0.011) | 3.93 (0.0049) | 49.0 / 71.2 | -0.00011 (-0.24) |
| 60 | 11113 | diagnostic_terminal | 34.9 | 43.54 | 42.11 | 23.609 (0.016) | 30.556 (0.014) | 23.579 (0.016) | 30.540 (0.014) | 2.39 (0.0043) | -8.87 (0.0097) | 49.0 / 71.4 | -0.00065 (-1.75) |
| 60 | 11114 | diagnostic_terminal | 31.81 | 54.28 | 71.85 | 34.608 (0.015) | 37.529 (0.028) | 34.579 (0.016) | 37.513 (0.028) | 24.50 (0.0091) | -1.59 (0.0086) | 49.0 / 85.7 | -0.00051 (-0.61) |
| 60 | 11115 | diagnostic_terminal | 40.52 | 49.3 | 50.36 | 27.203 (0.018) | 33.970 (0.018) | 27.179 (0.018) | 33.910 (0.018) | 6.19 (0.0069) | 3.84 (0.0044) | 49.0 / 75.0 | 0.00037 (0.72) |
| 60 | 11116 | diagnostic_terminal | 33.81 | 36.65 | 43.69 | 25.104 (0.013) | 28.316 (0.016) | 25.073 (0.013) | 28.248 (0.016) | -4.57 (0.0099) | 0.23 (0.0029) | 49.0 / 69.5 | -0.00029 (-0.83) |
| 60 | 11117 | diagnostic_terminal | 40.02 | 52.86 | 55.72 | 32.677 (0.017) | 35.611 (0.019) | 32.728 (0.017) | 35.592 (0.019) | 10.42 (0.0091) | -2.43 (0.0018) | 49.1 / 74.3 | 0.0012 (2.07) |
| 60 | 11118 | diagnostic_terminal | 45.45 | 55.42 | 52.18 | 30.351 (0.018) | 28.017 (0.02) | 30.259 (0.018) | 27.939 (0.02) | 11.66 (0.0079) | 3.07 (0.0028) | 49.0 / 78.8 | 0.00016 (0.28) |
| 60 | 11119 | diagnostic_terminal | 29.11 | 42.75 | 42.8 | 26.547 (0.013) | 25.017 (0.016) | 26.506 (0.013) | 25.015 (0.016) | 7.08 (0.006) | 3.85 (0.0019) | 49.0 / 75.0 | 1e-05 (0.03) |
| 60 | 11120 | diagnostic_terminal | 29.45 | 45.25 | 54.74 | 25.958 (0.016) | 35.567 (0.02) | 25.962 (0.016) | 35.522 (0.02) | 0.60 (0.001) | -0.84 (0.0014) | 49.1 / 69.4 | 0.00096 (1.87) |

#### G. Economics by endpoint group (across seeds; rollout MCSE kept separate)

| q | group | n runs | mode | stage | quantity | mean across runs | SD across runs | min | max | max rollout MCSE | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | certified_candidate | 0 | policy_curve | 1 | e1_at_0 | — | — | — | — | — | no run in this group |
| 50 | uncertified_candidate | 0 | policy_curve | 1 | e1_at_0 | — | — | — | — | — | no run in this group |
| 50 | diagnostic_terminal | 20 | policy_curve | 1 | e1_at_0 | 44.12 | 5.946 | 32.7 | 56.18 | — |  |
| 50 | diagnostic_terminal | 20 | policy_curve | 2 | e2_at_0 | 52.44 | 7.593 | 35.22 | 67.59 | — |  |
| 50 | diagnostic_terminal | 20 | policy_curve | 3 | e3_at_0 | 57.68 | 10.09 | 30.93 | 74.81 | — |  |
| 50 | diagnostic_terminal | 20 | mean | 1 | E_effort_representative | 44.12 | 5.946 | 32.7 | 56.18 | 5.1e-09 |  |
| 50 | diagnostic_terminal | 20 | mean | 2 | E_effort_representative | 31.09 | 3.418 | 25.19 | 38.5 | 0.022 |  |
| 50 | diagnostic_terminal | 20 | mean | 3 | E_effort_representative | 30.86 | 3.616 | 23.03 | 37.25 | 0.032 |  |
| 50 | diagnostic_terminal | 20 | mean | episode | U_mean_payoff | 2.699 | 0.2554 | 2.068 | 3.005 | 0.001 |  |
| 50 | diagnostic_terminal | 20 | stochastic | 1 | E_effort_representative | 44.12 | 5.945 | 32.7 | 56.18 | 0.0032 |  |
| 50 | diagnostic_terminal | 20 | stochastic | 2 | E_effort_representative | 31.02 | 3.419 | 25.13 | 38.42 | 0.022 |  |
| 50 | diagnostic_terminal | 20 | stochastic | 3 | E_effort_representative | 30.8 | 3.596 | 22.99 | 37.16 | 0.032 |  |
| 50 | diagnostic_terminal | 20 | stochastic | episode | U_mean_payoff | 2.695 | 0.2555 | 2.063 | 3.001 | 0.001 |  |
| 60 | certified_candidate | 0 | policy_curve | 1 | e1_at_0 | — | — | — | — | — | no run in this group |
| 60 | uncertified_candidate | 0 | policy_curve | 1 | e1_at_0 | — | — | — | — | — | no run in this group |
| 60 | diagnostic_terminal | 20 | policy_curve | 1 | e1_at_0 | 33.8 | 4.969 | 25.54 | 45.45 | — |  |
| 60 | diagnostic_terminal | 20 | policy_curve | 2 | e2_at_0 | 44.14 | 5.899 | 33.52 | 55.42 | — |  |
| 60 | diagnostic_terminal | 20 | policy_curve | 3 | e3_at_0 | 48.33 | 8.015 | 38.19 | 71.85 | — |  |
| 60 | diagnostic_terminal | 20 | mean | 1 | E_effort_representative | 33.8 | 4.969 | 25.54 | 45.45 | 4e-09 |  |
| 60 | diagnostic_terminal | 20 | mean | 2 | E_effort_representative | 26.54 | 3.665 | 21.27 | 34.61 | 0.018 |  |
| 60 | diagnostic_terminal | 20 | mean | 3 | E_effort_representative | 29.88 | 4.151 | 21.72 | 37.53 | 0.028 |  |
| 60 | diagnostic_terminal | 20 | mean | episode | U_mean_payoff | 3.093 | 0.203 | 2.742 | 3.489 | 0.00084 |  |
| 60 | diagnostic_terminal | 20 | stochastic | 1 | E_effort_representative | 33.8 | 4.97 | 25.54 | 45.45 | 0.003 |  |
| 60 | diagnostic_terminal | 20 | stochastic | 2 | E_effort_representative | 26.5 | 3.666 | 21.23 | 34.58 | 0.018 |  |
| 60 | diagnostic_terminal | 20 | stochastic | 3 | E_effort_representative | 29.84 | 4.15 | 21.71 | 37.51 | 0.028 |  |
| 60 | diagnostic_terminal | 20 | stochastic | episode | U_mean_payoff | 3.089 | 0.2031 | 2.737 | 3.485 | 0.00084 |  |

说明：

- **跨 seed 差异远大于 rollout MCSE**：q50 mean mode 的 E[e2] 跨 seed SD 为 3.42，而单 run 的 MCSE ≤ 0.022；U 的跨 seed SD 为 0.26，MCSE ≤ 0.001。
- **e1(0)**：q50 为 32.7–56.2（均值 44.1），q60 为 25.5–45.5（均值 33.8）。完整的 e2(d)、e3(d) 曲线（step 0.05，含 0 与两端）在 `policy_profiles.csv`，图为 `figures/policy_curves_q50|q60`。
- **leader−follower**：mean 模式下的配对差，stage 2 在 q50 为 −2.8…+24.8、q60 为 −6.4…+24.5；stage 3 在 q50 为 −1.7…+10.1、q60 为 −8.9…+3.9。符号随 seed 变化。root 是 tie（n=0，均值为 null）。曲线差 m_t(x)−m_t(−x) 在 `policy_asymmetry.csv`（kind=curve），图为 `figures/asymmetry_q*`。
- **mean vs stochastic**：两种 mode 的 E[e_t] 最多相差 0.154（q50 s11007 的 stage 3），仅作描述。stochastic − mean 的 U 在 q50 为 −0.0056…−0.0019、q60 为 −0.0068…−0.0024，是描述值，不是 exploitability。
- **MC vs DP**：mean mode 的 U 与 final 层 V1_mean(0) 之差除以 MCSE，q50 为 −1.24…+1.60，q60 为 −2.54…+2.07；|z|>2 的有 3 个（q60 s11108、s11111、s11117）。差值同时含 MC 噪声与 DP 离散化，只作描述，没有用于任何选择。
- **q50 s11008 的终点策略几乎不随 d 变化**（仅描述）：
  - e2(d) 在 D2 上的变化范围只有 2.3 effort 单位，e3(d) 为 1.3（|d|≤B 内 0.64）；其余 39 个 run 在 |d|≤B 内的范围都 ≥ 34。
  - 因此它的 E[e_t] 等于 e_t(0)（MCSE 约 1e−6），leader−follower 约为 0。
  - 该 run 的 B strategic 值全程 ≥ 0.1275，C 的 dReach 也最大（final 0.1448）。
- **state visitation**：`state_visitation.csv` 分别保存三类分布。
  - self-play（mean 与 stochastic 两种 mode）：player0、player1、representative 三种口径，含全部 bin 的计数与质量，以及每个 bin 的 effort 与 cost 之和；
  - training：按 phase、start stage、stage 分；
  - BR-chain：在 verifier 节点上。

  图为 `figures/state_histogram_q*`。各 stage 的 gap 矩、正/负/tie 比例与近似分位数在各 run 的 `economics.json`。

## 6. Failure diagnosis（40 个 no-candidate run）

下表取每个 run 的最小 valid C development 调用，不是 endpoint。

#### F. Failure diagnosis of no-candidate runs (minimum valid C development call; not the endpoint)

Exposure = training visits in the 10-wide bin of the state, cumulative to that call: C direct / C continuation / A+B direct / A+B continuation.

| q | seed | B exit | min dReach/DW (global/C local) | conc at min | reach contrib t1/t2/t3 | max reach state | its exposure | max all-domain state | its exposure | C dReach min / last | C calls ≤0.02 / ≤0.03 / valid | endpoint final dReach | endpoint final Delta_max_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 11001 | budget_forced | 0.02161 (2775/1775) | 0.02859 | 0.001078 / 0.004832 / 0.01569 | t3 d=-44: 0.01569 | 3862 / 25921 / 2618 / 7513 | t3 d=-44: 0.01569 (same as reach: yes) | 3862 / 25921 / 2618 / 7513 | 0.02161 / 0.02396 | 0/11/72 | 0.02403 | 0.01436 |
| 50 | 11002 | budget_forced | 0.01892 (2200/1200) | 0.03708 | 1.86e-06 / 0.007424 / 0.01149 | t3 d=104: 0.01149 | 2676 / 8659 / 2570 / 7174 | t3 d=104: 0.01149 (same as reach: yes) | 2676 / 8659 / 2570 / 7174 | 0.01892 / 0.09137 | 2/25/72 | 0.09193 | 0.04637 |
| 50 | 11003 | budget_forced | 0.01772 (2575/1575) | 0.03631 | 0.0003263 / 0.007635 / 0.009763 | t3 d=104: 0.009763 | 3336 / 13730 / 2648 / 7957 | t3 d=104: 0.009763 (same as reach: yes) | 3336 / 13730 / 2648 / 7957 | 0.01772 / 0.0536 | 3/29/72 | 0.0548 | 0.02882 |
| 50 | 11004 | budget_forced | 0.02306 (2525/1525) | 0.03272 | 0.0006088 / 0.01031 / 0.01214 | t3 d=104: 0.01214 | 3291 / 12063 / 2613 / 7959 | t3 d=104: 0.01214 (same as reach: yes) | 3291 / 12063 / 2613 / 7959 | 0.02306 / 0.0463 | 0/14/72 | 0.0462 | 0.02337 |
| 50 | 11005 | budget_forced | 0.05811 (2625/1625) | 0.03101 | 0.00195 / 0.03458 / 0.02159 | t2 d=28: 0.03458 | 3479 / 30976 / 7701 / 0 | t2 d=28: 0.03458 (same as reach: yes) | 3479 / 30976 / 7701 / 0 | 0.05811 / 0.07527 | 0/0/72 | 0.07537 | 0.05704 |
| 50 | 11006 | budget_forced | 0.02355 (2650/1650) | 0.0303 | 1.083e-06 / 0.01485 / 0.008697 | t2 d=-4: 0.01485 | 3509 / 39614 / 7875 / 0 | t2 d=-4: 0.01485 (same as reach: yes) | 3509 / 39614 / 7875 / 0 | 0.02355 / 0.03492 | 0/7/72 | 0.0351 | 0.02153 |
| 50 | 11007 | budget_forced | 0.04109 (2800/1800) | 0.03213 | 0.001387 / 0.02125 / 0.01846 | t2 d=24: 0.02125 | 3916 / 34348 / 7731 / 0 | t2 d=24: 0.02125 (same as reach: yes) | 3916 / 34348 / 7731 / 0 | 0.04109 / 0.04109 | 0/0/72 | 0.04112 | 0.02129 |
| 50 | 11008 | budget_forced | 0.1168 (1350/350) | 0.0324 | 0.002222 / 0.02652 / 0.08804 | t3 d=-16: 0.08804 | 762 / 6542 / 2519 / 7494 | t3 d=-16: 0.08804 (same as reach: yes) | 762 / 6542 / 2519 / 7494 | 0.1168 / 0.1425 | 0/0/72 | 0.1448 | 0.09446 |
| 50 | 11009 | budget_forced | 0.0447 (2175/1175) | 0.02874 | 0.001791 / 0.01834 / 0.02457 | t3 d=-8: 0.02457 | 2617 / 21049 / 2564 / 7612 | t3 d=-8: 0.02457 (same as reach: yes) | 2617 / 21049 / 2564 / 7612 | 0.0447 / 0.05872 | 0/0/72 | 0.05873 | 0.02965 |
| 50 | 11010 | budget_forced | 0.04713 (2750/1750) | 0.03014 | 0.0001477 / 0.02284 / 0.02415 | t3 d=-8: 0.02415 | 3762 / 27645 / 2571 / 7919 | t3 d=-8: 0.02415 (same as reach: yes) | 3762 / 27645 / 2571 / 7919 | 0.04713 / 0.0526 | 0/0/72 | 0.0527 | 0.02995 |
| 50 | 11011 | budget_forced | 0.0521 (2425/1425) | 0.03069 | 0.0002411 / 0.0335 / 0.01836 | t2 d=28: 0.0335 | 3013 / 27705 / 7612 / 0 | t2 d=28: 0.0335 (same as reach: yes) | 3013 / 27705 / 7612 / 0 | 0.0521 / 0.06169 | 0/0/72 | 0.06202 | 0.03215 |
| 50 | 11012 | budget_forced | 0.02342 (1800/800) | 0.03689 | 0.001093 / 0.01132 / 0.01101 | t2 d=-4: 0.01132 | 1706 / 19147 / 7691 / 0 | t2 d=-4: 0.01132 (same as reach: yes) | 1706 / 19147 / 7691 / 0 | 0.02342 / 0.05688 | 0/3/72 | 0.05771 | 0.03673 |
| 50 | 11013 | budget_forced | 0.01543 (2750/1750) | 0.02674 | 0.0007496 / 0.004281 / 0.01039 | t3 d=104: 0.01039 | 3761 / 10135 / 2502 / 7701 | t3 d=104: 0.01039 (same as reach: yes) | 3761 / 10135 / 2502 / 7701 | 0.01543 / 0.02401 | 5/12/72 | 0.02422 | 0.01855 |
| 50 | 11014 | budget_forced | 0.0238 (2050/1050) | 0.03513 | 0.0008401 / 0.009655 / 0.0133 | t3 d=-96: 0.0133 | 2252 / 9330 / 2518 / 8093 | t3 d=-96: 0.0133 (same as reach: yes) | 2252 / 9330 / 2518 / 8093 | 0.0238 / 0.0466 | 0/6/72 | 0.04649 | 0.02819 |
| 50 | 11015 | budget_forced | 0.03354 (2150/1150) | 0.03592 | 2.065e-05 / 0.01815 / 0.01538 | t2 d=-4: 0.01815 | 2474 / 27121 / 7593 / 0 | t2 d=-4: 0.01815 (same as reach: yes) | 2474 / 27121 / 7593 / 0 | 0.03354 / 0.04014 | 0/0/72 | 0.04001 | 0.02534 |
| 50 | 11016 | budget_forced | 0.03524 (2525/1525) | 0.03014 | 0.004561 / 0.01322 / 0.01747 | t3 d=-48: 0.01747 | 3199 / 22074 / 2554 / 7410 | t3 d=-48: 0.01747 (same as reach: yes) | 3199 / 22074 / 2554 / 7410 | 0.03524 / 0.03801 | 0/0/72 | 0.03815 | 0.01917 |
| 50 | 11017 | budget_forced | 0.03089 (2025/1025) | 0.03284 | 9.379e-05 / 0.01977 / 0.01102 | t2 d=-4: 0.01977 | 2189 / 24369 / 7761 / 0 | t2 d=-4: 0.01977 (same as reach: yes) | 2189 / 24369 / 7761 / 0 | 0.03089 / 0.04699 | 0/0/72 | 0.0469 | 0.02907 |
| 50 | 11018 | budget_forced | 0.03144 (2175/1175) | 0.03622 | 0.00214 / 0.01252 / 0.01678 | t3 d=104: 0.01678 | 2477 / 7503 / 2566 / 7593 | t3 d=104: 0.01678 (same as reach: yes) | 2477 / 7503 / 2566 / 7593 | 0.03144 / 0.09502 | 0/0/72 | 0.09625 | 0.07206 |
| 50 | 11019 | budget_forced | 0.03049 (1425/425) | 0.03746 | 0.0004538 / 0.0136 / 0.01644 | t3 d=108: 0.01644 | 881 / 3172 / 2572 / 7730 | t3 d=108: 0.01644 (same as reach: yes) | 881 / 3172 / 2572 / 7730 | 0.03049 / 0.09145 | 0/0/72 | 0.09143 | 0.05593 |
| 50 | 11020 | budget_forced | 0.02529 (2575/1575) | 0.03016 | 0.0001564 / 0.01034 / 0.01479 | t3 d=104: 0.01479 | 3359 / 12226 / 2597 / 7573 | t3 d=104: 0.01479 (same as reach: yes) | 3359 / 12226 / 2597 / 7573 | 0.02529 / 0.06916 | 0/7/72 | 0.06945 | 0.03255 |
| 60 | 11101 | budget_forced | 0.02531 (2100/1100) | 0.03395 | 0.0009589 / 0.01452 / 0.009834 | t2 d=-44: 0.01452 | 2147 / 14671 / 6966 / 0 | t2 d=-44: 0.01452 (same as reach: yes) | 2147 / 14671 / 6966 / 0 | 0.02531 / 0.0383 | 0/17/72 | 0.0383 | 0.02529 |
| 60 | 11102 | budget_forced | 0.04573 (2400/1400) | 0.02902 | 0.0005912 / 0.02609 / 0.01905 | t2 d=36: 0.02609 | 2757 / 21021 / 6889 / 0 | t2 d=36: 0.02609 (same as reach: yes) | 2757 / 21021 / 6889 / 0 | 0.04573 / 0.04642 | 0/0/72 | 0.0465 | 0.02857 |
| 60 | 11103 | budget_forced | 0.01595 (2300/1300) | 0.03777 | 0.000396 / 0.006124 / 0.009435 | t3 d=124: 0.009435 | 2563 / 7051 / 2344 / 6933 | t2 d=208: 0.01855 (same as reach: no) | 2449 / 0 / 6873 / 0 | 0.01595 / 0.02053 | 6/45/72 | 0.02055 | 0.01477 |
| 60 | 11104 | budget_forced | 0.02061 (2775/1775) | 0.02869 | 0.0007485 / 0.00971 / 0.01015 | t3 d=128: 0.01015 | 3438 / 9967 / 2218 / 6528 | t3 d=128: 0.01015 (same as reach: yes) | 3438 / 9967 / 2218 / 6528 | 0.02061 / 0.02453 | 0/11/72 | 0.0246 | 0.01268 |
| 60 | 11105 | budget_forced | 0.02721 (2775/1775) | 0.02982 | 7.101e-05 / 0.006724 / 0.02041 | t3 d=-112: 0.02041 | 3440 / 11115 / 2293 / 7029 | t3 d=-112: 0.02041 (same as reach: yes) | 3440 / 11115 / 2293 / 7029 | 0.02721 / 0.03297 | 0/3/72 | 0.033 | 0.02341 |
| 60 | 11106 | budget_forced | 0.01557 (2525/1525) | 0.02954 | 1.612e-05 / 0.005012 / 0.01054 | t3 d=128: 0.01054 | 2980 / 8424 / 2356 / 6820 | t3 d=128: 0.01054 (same as reach: yes) | 2980 / 8424 / 2356 / 6820 | 0.01557 / 0.05321 | 4/20/72 | 0.05327 | 0.02993 |
| 60 | 11107 | budget_forced | 0.02309 (2250/1250) | 0.02857 | 3.173e-06 / 0.01288 / 0.0102 | t2 d=76: 0.01288 | 2428 / 9841 / 7026 / 0 | t2 d=76: 0.01288 (same as reach: yes) | 2428 / 9841 / 7026 / 0 | 0.02309 / 0.04186 | 0/29/72 | 0.04186 | 0.02104 |
| 60 | 11108 | budget_forced | 0.03397 (2575/1575) | 0.02712 | 6.258e-05 / 0.01969 / 0.01423 | t2 d=44: 0.01969 | 3086 / 20769 / 7027 / 0 | t2 d=44: 0.01969 (same as reach: yes) | 3086 / 20769 / 7027 / 0 | 0.03397 / 0.03553 | 0/0/72 | 0.03556 | 0.0183 |
| 60 | 11109 | budget_forced | 0.02222 (1950/950) | 0.03033 | 0.0004924 / 0.01111 / 0.01062 | t2 d=84: 0.01111 | 1851 / 5951 / 7026 / 0 | t2 d=84: 0.01111 (same as reach: yes) | 1851 / 5951 / 7026 / 0 | 0.02222 / 0.03463 | 0/21/72 | 0.03479 | 0.02327 |
| 60 | 11110 | budget_forced | 0.0238 (2725/1725) | 0.02621 | 0.001059 / 0.01378 / 0.008962 | t2 d=80: 0.01378 | 3384 / 10872 / 7061 / 0 | t2 d=80: 0.01378 (same as reach: yes) | 3384 / 10872 / 7061 / 0 | 0.0238 / 0.02696 | 0/6/72 | 0.02697 | 0.01653 |
| 60 | 11111 | budget_forced | 0.02489 (2625/1625) | 0.02891 | 0.002368 / 0.01103 / 0.0115 | t3 d=-8: 0.0115 | 3180 / 23843 / 2343 / 6841 | t3 d=-8: 0.0115 (same as reach: yes) | 3180 / 23843 / 2343 / 6841 | 0.02489 / 0.03617 | 0/10/72 | 0.03614 | 0.0172 |
| 60 | 11112 | budget_forced | 0.03812 (2675/1675) | 0.02657 | 0.0003423 / 0.02514 / 0.01263 | t2 d=36: 0.02514 | 3289 / 25185 / 7231 / 0 | t2 d=36: 0.02514 (same as reach: yes) | 3289 / 25185 / 7231 / 0 | 0.03812 / 0.04501 | 0/0/72 | 0.04511 | 0.02987 |
| 60 | 11113 | budget_forced | 0.03473 (2775/1775) | 0.02599 | 0.00232 / 0.01503 / 0.01738 | t3 d=128: 0.01738 | 3411 / 9307 / 2383 / 6975 | t3 d=128: 0.01738 (same as reach: yes) | 3411 / 9307 / 2383 / 6975 | 0.03473 / 0.03691 | 0/0/72 | 0.03692 | 0.01927 |
| 60 | 11114 | budget_forced | 0.0176 (1850/850) | 0.0343 | 0.0009226 / 0.006555 / 0.01012 | t3 d=-120: 0.01012 | 1645 / 6413 / 2295 / 7144 | t3 d=-120: 0.01012 (same as reach: yes) | 1645 / 6413 / 2295 / 7144 | 0.0176 / 0.08223 | 4/32/72 | 0.08238 | 0.04902 |
| 60 | 11115 | budget_forced | 0.03107 (1800/800) | 0.03145 | 0.0001406 / 0.01887 / 0.01206 | t2 d=-48: 0.01887 | 1515 / 10765 / 6947 / 0 | t2 d=-48: 0.01887 (same as reach: yes) | 1515 / 10765 / 6947 / 0 | 0.03107 / 0.0358 | 0/0/72 | 0.03586 | 0.02449 |
| 60 | 11116 | budget_forced | 0.02694 (2725/1725) | 0.02679 | 0.004226 / 0.009112 / 0.0136 | t3 d=124: 0.0136 | 3334 / 8830 / 2326 / 6928 | t3 d=124: 0.0136 (same as reach: yes) | 3334 / 8830 / 2326 / 6928 | 0.02694 / 0.0305 | 0/8/72 | 0.03063 | 0.01766 |
| 60 | 11117 | budget_forced | 0.02079 (1825/825) | 0.0336 | 0.0005861 / 0.01025 / 0.009954 | t2 d=12: 0.01025 | 1522 / 15396 / 7087 / 0 | t2 d=12: 0.01025 (same as reach: yes) | 1522 / 15396 / 7087 / 0 | 0.02079 / 0.04772 | 0/25/72 | 0.04787 | 0.02237 |
| 60 | 11118 | budget_forced | 0.02146 (2100/1100) | 0.03245 | 0.0004622 / 0.008739 / 0.01226 | t3 d=-8: 0.01226 | 2147 / 16650 / 2307 / 6773 | t3 d=-8: 0.01226 (same as reach: yes) | 2147 / 16650 / 2307 / 6773 | 0.02146 / 0.04509 | 0/31/72 | 0.04554 | 0.02436 |
| 60 | 11119 | budget_forced | 0.01838 (2500/1500) | 0.03385 | 0.0003227 / 0.005065 / 0.01299 | t3 d=-8: 0.01299 | 2879 / 23712 / 2347 / 7413 | t3 d=-8: 0.01299 (same as reach: yes) | 2879 / 23712 / 2347 / 7413 | 0.01838 / 0.02554 | 5/25/72 | 0.02556 | 0.01575 |
| 60 | 11120 | budget_forced | 0.02983 (2125/1125) | 0.02949 | 3.877e-05 / 0.01402 / 0.01577 | t3 d=124: 0.01577 | 2212 / 5737 / 2266 / 6792 | t3 d=124: 0.01577 (same as reach: yes) | 2212 / 5737 / 2266 / 6792 | 0.02983 / 0.04044 | 0/1/72 | 0.04054 | 0.02728 |

汇总（仅描述，不作因果结论）：

- **失败形态**：40 个 run 都是 B budget_forced，C 没有一次 eligible。C 的失败都包含 strategic 失败，concentration 从未单独挡住 C。
- **最大 reach 偏离所在 stage**（取最小 valid C 调用处）：
  - stage 3 为 24/40（q50 13、q60 11），stage 2 为 16/40（q50 7、q60 9）；
  - 同一调用处 stage-1 贡献 ≤ 0.0046。
- **反复出现的位置**：q50 为 t3 d=104（6 个 run）与 t2 d=−4（4 个）；q60 为 t3 d=124/128（6 个）与 t3 d=−8（3 个）。
- **reach 最大与全域最大**：只有 q60 s11103 不同。它的全域最大在 t2 d=208（R_2 之外，0.0186），reach 最大在 t3 d=124（0.0094）。
- **这些状态的训练暴露**：
  - C direct：q50 762–3916，q60 1515–3440；
  - C continuation：q50 3172–39614，q60 5737–25185；
  - A+B direct：2218–7875；
  - stage-2 状态的 A+B continuation 为 0。这是结构性的：A、B 的轨迹不会延续到 stage 2。
- **直接起点覆盖**：
  - 所有 run 的 ES 直接起点每个 bin 都非零。A 阶段 ES3 每 bin 最少 2393（q50）与 2147（q60），tail bin 最少 2427 与 2190，与期望值 2560、2327.3 一致。
  - continuation 分布有零 bin，因为 root 与 ES2 的轨迹集中在内部。
  - 与 pilot 一样，覆盖数据没有显示缺失或稀疏的 bin。这里只把暴露量与误差并列，不推断因果。
- **实际终点与轨迹**：endpoint 的 final 指标见 V1。C 阶段完整的 dReach 与各 stage 贡献轨迹在 `verifier_calls.csv`，图为 `figures/verifier_curves_q50|q60`。

## 7. 完整性检查

`make_report.py --cohort formal --check-completeness` 对 40 个 run 全部判定 ok：failures 0，notes 0，unavailable 0。逐 run 检查的项目：

- history 行数 = 2800 = 最后完成的 update；
- 各 phase 每个 update 的 episodes、steps、start counts 与定义一致；
- A 4 次、B 24 次、C 72 次计划调用全部存在，没有计划外的调用；
- 重算 eligible 与连续计数，结果一致；B 的准则确实是 stage-2 continuation gain，C 的准则确实是 dReach；
- no-candidate run 确实跑到 C cap，且没有 eligible 的 C 调用；
- episode 与 step 总数等于 history 的求和；
- coverage 质量守恒，bin 数正确；
- endpoint 身份在各文件间一致，重读逐位一致，dev 重算差 ≤ 1e−12；
- dev、final 网格点数与 dense 点数正确；
- 六项认证 flag 可以重算；
- min_dev 是同一次调用，各 stage 贡献之和等于 dReach；
- economics 的均值可由 n、sum、sum_sq 重算，n=600000，直方图质量为 1，payoff 会计检查通过。

#### C. Completeness (make_report.py --check-completeness)

| run | state | complete | failures | notes | unavailable checks |
|---|---|---|---|---|---|
| t3_formal_q50_s11001 | done | True | none | none | none |
| t3_formal_q50_s11002 | done | True | none | none | none |
| t3_formal_q50_s11003 | done | True | none | none | none |
| t3_formal_q50_s11004 | done | True | none | none | none |
| t3_formal_q50_s11005 | done | True | none | none | none |
| t3_formal_q50_s11006 | done | True | none | none | none |
| t3_formal_q50_s11007 | done | True | none | none | none |
| t3_formal_q50_s11008 | done | True | none | none | none |
| t3_formal_q50_s11009 | done | True | none | none | none |
| t3_formal_q50_s11010 | done | True | none | none | none |
| t3_formal_q50_s11011 | done | True | none | none | none |
| t3_formal_q50_s11012 | done | True | none | none | none |
| t3_formal_q50_s11013 | done | True | none | none | none |
| t3_formal_q50_s11014 | done | True | none | none | none |
| t3_formal_q50_s11015 | done | True | none | none | none |
| t3_formal_q50_s11016 | done | True | none | none | none |
| t3_formal_q50_s11017 | done | True | none | none | none |
| t3_formal_q50_s11018 | done | True | none | none | none |
| t3_formal_q50_s11019 | done | True | none | none | none |
| t3_formal_q50_s11020 | done | True | none | none | none |
| t3_formal_q60_s11101 | done | True | none | none | none |
| t3_formal_q60_s11102 | done | True | none | none | none |
| t3_formal_q60_s11103 | done | True | none | none | none |
| t3_formal_q60_s11104 | done | True | none | none | none |
| t3_formal_q60_s11105 | done | True | none | none | none |
| t3_formal_q60_s11106 | done | True | none | none | none |
| t3_formal_q60_s11107 | done | True | none | none | none |
| t3_formal_q60_s11108 | done | True | none | none | none |
| t3_formal_q60_s11109 | done | True | none | none | none |
| t3_formal_q60_s11110 | done | True | none | none | none |
| t3_formal_q60_s11111 | done | True | none | none | none |
| t3_formal_q60_s11112 | done | True | none | none | none |
| t3_formal_q60_s11113 | done | True | none | none | none |
| t3_formal_q60_s11114 | done | True | none | none | none |
| t3_formal_q60_s11115 | done | True | none | none | none |
| t3_formal_q60_s11116 | done | True | none | none | none |
| t3_formal_q60_s11117 | done | True | none | none | none |
| t3_formal_q60_s11118 | done | True | none | none | none |
| t3_formal_q60_s11119 | done | True | none | none | none |
| t3_formal_q60_s11120 | done | True | none | none | none |

## 8. 限制，以及不声称的内容

- 本结果只衡量在预先规定的预算与阈值下找到 candidate 的比例：每 q 0/20。它不说明所有 seed 收敛，也不证明或否定均衡存在。
- 没有 candidate，conditional certification 无定义（N/A）。经济结果只描述 diagnostic 终点。
- root payoff 偏离较小（final EXP_root/DW：q50 0.0085–0.0486，q60 0.0043–0.0272）不等于通过认证；本次所有 run 的 dReach 都超过阈值。
- failure diagnosis 只报告相关信息：位置、暴露量、各 stage 贡献。本次没有做区分“预算不足”与“其他优化因素”的实验。
- 按要求，本次没有调参，没有追加 seeds，也没有启动 K8、A800、初始化对照或其他新实验。
- pilot（0/6）与 ABC（0/3、0/3）的比例不并入本结果。

## 9. 文件索引（路径相对于 E）

| 内容 | 位置 |
|---|---|
| 运行前选定的配置 | `reports/formal_settings.json`、`reports/FORMAL_SETTINGS.md`（未修改） |
| formal manifests | `manifests/formal_q50.json`、`manifests/formal_q60.json` |
| 40 个 run 的数据 | `runs/t3_formal_q{50,60}_s*/`，其中包括：<br>• `config.json`、`status.json`、`history.jsonl`、`verifier_calls.jsonl`、`events.jsonl`、`resources.jsonl`、`coverage.npz`、`training_summary.json`<br>• `checkpoints/`（endpoint、min_dev 的权重、arrays 与 profiles）；`weights/`（每 25 个 update 一份）<br>• `final_eval.json`、`arrays.npz`、`economics.json`、`economics_arrays.npz`；`tables/*.csv` |
| 日志 | `logs/t3_formal_*.log`（40 个）、`logs/launch_formal.out`、`logs/formal_commands.log`、`logs/formal_unittest_20260926_005634.log`、`logs/formal_make_report.log`；launcher 记录 `launch_logs/launch_20260926_005701_formal_q50+formal_q60.json` |
| cohort 表格 | `reports/formal/` 下的 runs、phases、verifier_calls、verifier_summary、verifier_stage_metrics、deviations、policy_profiles、policy_asymmetry、stage_metrics、state_visitation、coverage、coverage_summary、failure_diagnostics、resources、rates（均为 `.csv`）；另有 `completeness.json`、`AUTO_SUMMARY.md` |
| 本报告另加的汇总 | `reports/formal/FORMAL_TABLES.md`、`verification_by_seed.csv`、`economics_by_group.csv`（由 `formal_summary.py` 生成） |
| 图（均为 png/pdf，并附对应 csv） | `reports/formal/figures/` 下的 policy_curves、asymmetry、state_histogram、deviation_profile、verifier_curves、coverage_A，各有 q50、q60 两份 |

重现命令（在 vector2，E 所在 worktree 根目录执行）：

    PY=python
    E=experiments/three_stage_implementation_pilot_20260924
    $PY -B $E/build_manifests.py --cohort formal --settings $E/reports/formal_settings.json --seeds-per-q 20
    $PY -B $E/launch.py --manifest $E/manifests/formal_q50.json --manifest $E/manifests/formal_q60.json --max-workers 10
    $PY -B $E/make_report.py --cohort formal --check-completeness
    $PY -B $E/formal_summary.py --cohort formal

launcher 不会覆盖已存在的 run 目录。报表可以从已保存的权重与日志重算，不需要重新训练。

---
逐 seed 表格由 `formal_summary.py` 从 `reports/formal/*.csv` 生成（`reports/formal/FORMAL_TABLES.md`），此处原样收录；叙述部分为手写，所有数值可由上述 CSV 重算。
