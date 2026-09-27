# FORMAL_SETTINGS — T=3 formal 配置（正式运行前选定，现已完成）

机器可读版本为 [formal_settings.json](formal_settings.json)。这是正式实验运行前选择设置的科学记录；执行结果见 [FORMAL_REPORT.md](FORMAL_REPORT.md)。

## 1. 选定配置：保留已测试默认值，不做修改

| 项目 | formal 值（与 pilot 完全相同） |
|---|---|
| game | w_h=6, w_l=2, k=1/3500, e∈[0,100], T=3；q∈{50,60} 是唯一经济差异，B/domains/bins/grid 由 GameSpec 派生 |
| Phase A | active [3]，512 ES3/update，cap 400，dev 诊断 local 100/200/300/400，无早停 |
| Phase B | active [2,3]，512 ES2/update，cap 600，每 25 检查，K=3 连续 eligible 退出，否则 budget_forced |
| Phase C | active [1,2,3]，256 root + 85 ES2 + 171 ES3，cap 1800，每 25 检查，第一次 eligible 即保存 candidate 并停止 |
| eligibility | B: max_D2 (V2_BR−V2_mean)/DW ≤ 0.02；C: dReach/DW ≤ 0.01；concentration ≤ 0.04（B: stages 2,3；C: stages 1,2,3，dev 网格）；均要求 verifier valid |
| verifier | dev: state 4 / effort 1 / GL 16；final: state 2 / effort 0.5 / GL 32；grid_tol 1e−9，mass_tol 1e−10，pdl_tol/DW 1e−10 |
| final 认证 | dev/final valid；dReach_final/DW ≤ 0.01；两层 dReach、EXP 差 /DW ≤ 0.002；全 D1–D3 step 0.05 dense concentration ≤ 0.04；final_joint_pass 另需 has_candidate |
| PPO | 现有 PPOConfig 默认（hidden 64, lr 3e−4 constant, clip 0.2, epochs 10, minibatch 256, γ=λ=1, c_min 100, entropy 0）；跨 phase 保留 Adam；snapshot 每 20、weights 每 25 |
| economics | mean 与 stochastic 各 3 replicates × 200000 episodes，chunk 10000，SeedSequence([9005000, seed, q, 100, mode, rep, stream]) |
| candidate 规则 | 第一次 eligible C 调用 = candidate；final 前保存；不恢复训练、不晋升其他 checkpoint、不做多 restart 筛选 |
| N | **20 / q**（预先确定，不是先跑 10 再加） |
| seeds | q50: 11001–11020；q60: 11101–11120 |
| 并发 | 全局最多 10 workers（两个 manifest 共用一个 launcher 与同一上限），每 worker 1 CPU |

formal manifests 不含任何 smoke override。其非 q 设置与 pilot manifests 逐项相同，q 的派生量与 pilot 相同。

## 2. 为什么不改训练设置

计划规定：只有覆盖确有明显不足时才考虑修改，且应先考虑增加 A 的 ES episodes，而不是增加 PPO updates。
任何修改都必须在 formal 前先用明确标记的 debug 配置实际验证。6 个 debug runs 的数据（`PILOT_REPORT.md` Q2、Q6）显示：

- 直接起点每 bin 实测计数与期望一致，没有零 bin 或稀疏 bin（CV 0.017–0.021，tail bin 最少 2189–2404）；
- 最大偏离位于高暴露的内部状态：最小 C 调用处 C 直接暴露 2328–3496 次，continuation 暴露 6293–27293 次。

因此数据不支持“覆盖不足”，也就不支持“增加 A ES episodes”这一首选修改。预算或阈值调整同样没有被 6 个 run 验证过：

- q60 的 C dReach 在 1800 内缓慢下降，但未到 0.01；
- q50 有两个 seed 在最后约 1000 updates 没有下降趋势；
- B 中 strategic gap 下降时 concentration 同步上升。

按计划，无法用 6 个 debug run 支持的调整不做，保留已测试默认值。

## 3. N 的选择与资源依据

实测数据：每 run 3 workers 并发，见 `reports/pilot/resources.csv` 与 `runs.csv`。

| 量 | 实测 |
|---|---|
| 每 run 总 wall | 1027–1055 s（≈17.5 min） |
| 每 run CPU | 630–649 s（训练），另 dev verifier 5.4–5.9 s、final 0.3–0.4 s、economics 3.3–4.0 s |
| wall−CPU 差 | ≈390 s/run，来自每 update 两次 fsync（原始文件系统中 history 追加 ≈50 ms，status 原子写 ≈34 ms） |
| 峰值 RSS | 513–518 MiB/run（final 与 economics 阶段各自峰值 ≤516 MiB，同一进程内顺序执行） |
| 磁盘 | 32–35 MB/run（含 17 MB per-run tables），cohort 报表约 100 MB |

formal 估算：

| N/q | runs | 总 wall（run·h） | CPU·h | 10 workers 的 wall | 峰值内存（10 并发） | 磁盘 |
|---:|---:|---:|---:|---|---|---|
| 10 | 20 | ≈5.8 | ≈3.6 | ≈35–60 min（2 波） | ≈5.2 GB | ≈0.7 GB + 报表 |
| 20 | 40 | ≈11.6 | ≈7.2 | ≈70–110 min（4 波） | ≈5.2 GB | ≈1.4 GB + 报表 |

wall 区间的上端考虑了 10 个进程同时 fsync 可能带来的争用，这一点 pilot 只测到 3 并发。

N=20 的 Wilson 区间更窄：0/20 的上界 0.161，0/10 为 0.278。所以选 **N=20/q**。这是资源充足时计划允许的选项，在任何 formal run 前确定。

## 4. 仍有的限制（如实说明）

- pilot 中 0/6 找到 candidate，最小 dev dReach/DW 是阈值的 2.2–6.1 倍。按已选配置执行的 formal，很可能得到很低甚至为 0 的 candidate discovery。
  这正是 formal 要测量的量（“在预先规定预算中找到 verified candidates 的比例”），不是可以在 formal 后再调的对象。
- 改变 phase 结构、预算或 concentration 与 c_min 的关系属于新的研究设计，不能与本正式队列混合解释。
  本 pilot 没有测试任何替代设计，不能推荐具体替代值。
- fsync 开销约占 wall 的 38%。改为只 flush 不 fsync 会缩短 wall 而不影响数值，但它是代码改动，需要重新测试。本配置未采用，formal 的资源估计按现有实现计算。
