# Handoff: message to the coworker (T=2 status pack)

Nothing has been sent. The recipient and the channel are not named; this is the text to send once they are. Links use the branch `t2-status-pack`; the permalink is fixed to one commit.

## 中文

你好，需要你为 T=2 做一个决定。T=2（两阶段锦标赛基准）的求解器 v2.0 已锁定并通过 fresh-seed 确认；之后三轮多阶段实验（MS-R1 到 MS-R3）想降低 stage-2 平局点的 peak 偏差，都没有达到预先登记的判据。我把状态整理成一个自包含的文件夹。

- 文件夹：https://github.com/GSU-FrankJ/tournament_experiment/tree/t2-status-pack/reports/t2_status_100826
- 报告（固定版本）：https://github.com/GSU-FrankJ/tournament_experiment/blob/<FULL_SHA>/reports/t2_status_100826/report.md
- 对比视图：https://github.com/GSU-FrankJ/tournament_experiment/compare/ms-r3...t2-status-pack

状态：
- 已确定：v2.0 通过门槛和确认（19/20 与 20/20）；偏差的平滑部分有精确公式；没有可采纳的 tip 修复。
- 当前精度：fresh seeds 上 |peak 误差| 均值 0.0630 / 0.0678（q=50 / 60），5/20 与 4/20 的 run 不超过 0.05；开发种子上最好的 arm 为 0.0297 / 0.0209，未经 fresh-seed 确认且有一次门槛失败。
- 未决：余项的机制；`relu` 的失败率；任何 MS 配置在 fresh seeds 上的表现。

问题（原文）：基于现有结果，我们应该在此阶段收口，还是继续提高精度？如果继续，优先改进什么、目标是什么？

请回复：(i) 收口还是继续；(ii) 若继续，先做哪项措施，目标是什么（指标、数值、种子集）。你回复之前不会开始任何实验。读报告第 0、7.2、9、10 节约十五分钟即可。

## English

Hello, I need a decision from you on T=2. The T=2 solver (protocol v2.0, a two-stage tournament benchmark) is locked and passed its fresh-seed confirmation; three later multistage rounds (MS-R1 to MS-R3) tried to lower the stage-2 peak error at the tie and none met its pre-registered criterion. I put the state into one self-contained folder.

- Folder: https://github.com/GSU-FrankJ/tournament_experiment/tree/t2-status-pack/reports/t2_status_100826
- Report (fixed to one commit): https://github.com/GSU-FrankJ/tournament_experiment/blob/<FULL_SHA>/reports/t2_status_100826/report.md
- Compare view: https://github.com/GSU-FrankJ/tournament_experiment/compare/ms-r3...t2-status-pack

Status:
- Settled: v2.0 passes its gates and confirmation (19/20 and 20/20); the smoothing part of the deficit has an exact formula; no admissible tip fix exists.
- Current accuracy: on fresh seeds the mean |peak error| is 0.0630 / 0.0678 (q = 50 / 60), with 5 of 20 and 4 of 20 runs within 0.05; the best development-seed arm reaches 0.0297 / 0.0209, unconfirmed and with one gate failure.
- Open: the mechanism of the remainder; the failure rate of `relu`; any MS configuration on fresh seeds.

Question (verbatim): Based on the current results, should we close T=2 at this stage, or continue improving its accuracy? If we continue, what should be improved first, and what is the target?

Please send back: (i) close or continue; (ii) if continue, the measure to start with and the target (metric, value, seed set). No experiment starts before your reply. Sections 0, 7.2, 9 and 10 of the report take about fifteen minutes.
