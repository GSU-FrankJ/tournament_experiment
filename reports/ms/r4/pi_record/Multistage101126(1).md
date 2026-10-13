===== Page 1 =====

we can still try to reduce the stage-2 bias, and also determine the role of development DP-BR in training. My current idea is to first do the following two attempts; after they are completed, finalize the T=2 protocol and start the T=3 experimental validation.

The specific experimental details below are for reference; you can adjust them according to your experience.

The locked v2.0 is retained as the baseline. Both experiments run q=50, 60, using new development seeds 10511--10530 (20 per q, paired between groups). 30501--30520 have been used for previous confirmation and no longer count as new seeds; 40501--40520 are reserved for confirmation after the candidate scheme is locked.

===== Page 2 =====

**1. [Try to reduce the stage-2 tie-point bias]{.mark}.** Starting from relu_st_s16, make only one pre-fixed robustness candidate: change the hidden layer to LeakyReLU, and change the mean output to a smooth, bounded mapping that does not depend on hard clamp. Keep the rest as MS-R3's stratified starting point, s16 noise landing, and 2800 stage-2 updates. Rerun relu_st_s16 and t1_st_s16 as controls with the same batch of seeds. The goal is that on both q, the mean \|peak error\| is ≤0.04, and the paired improvement relative to t1_st_s16 holds within the 95% interval; development experiments G-A, G-F, G-N, G-S show no failures. The cause of the existing ReLU failure has not yet been determined; therefore, even if the candidate is effective, only report the effect of the combined changes, and do not attribute it to any one change. If the goal is not met, stop this line.

===== Page 3 =====

**2. [Test whether development DP-BR can actually help PPO training]{.mark}.** Fix the original t1 actor, training settings, and budget (stage-2 1600 times, stage-1 600 times), disable dynamic stopping and polishing, and compare only sampling feedback: ① fixed stratified sampling (α=0); ② existing residual-guided sampling (α=0.5); ③ aggregate the per-state deviation gains of development DP-BR into each state bin, and together with the existing action residuals decide the next batch's starting points (α=0.5). The three groups use the same seeds, check frequency, and state stratification ratio, and record the sampling weights after each check; ① also runs the same checks so that computational cost can be compared. The closed-form solution is used only for offline evaluation and does not enter sampling. The goal is that relative to ①, on both q, the area under the stage-2 η₂/DW learning curve at fixed checkpoints decreases by at least 20% on average, the paired 95% interval supports improvement, and at the same time the final full thresholds and stage-1 deviation gains do not become worse. If ② has already met the goal, then ③ is not needed to prove the feedback is effective; if neither meets the goal, keep DP-BR only as a validator and do not claim that it improves training.

===== Page 4 =====

Please deliver each group's configuration and code version, per-seed results, paired intervals, peak, learning curves, final thresholds of each item, and running cost. After the development results pass, first lock **one complete scheme** to adopt, then use 40501--40520 for fresh-seed confirmation; even if the two items are each effective, one cannot directly assume that they are effective when used together; combined use must first be separately validated on development seeds. If confirmation fails, retain v2.0, and report the two results as ablation and limitations.