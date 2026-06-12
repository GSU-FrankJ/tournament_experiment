# Fig-7 component ablation collection (r5_sampled wave, 2026-06-12)

Arms (2P, q in {35,45,55}, seeds 42-46, sampled training):
- full: = A1 baseline runs (NOT copied; canonical files at results/two_players/convergence/ppo_q{Q}.0_seed{S}_r5_sampled_convergence.json)
- no_stability: 15 copies here (source: ..._r5_fig7_no_stability_...; --disable-cheap-gate)
- no_exploit: 15 copies here (source: ..._r5_fig7_no_exploit_...; --disable-exploitability; all stop_reason=max_updates@1500 by design)

Headline (rel error vs e*, mean over 5 seeds):
  full:         q35 4.12%  q45 1.54%  q55 2.74%  (stops 49-109, verified)
  no_stability: q35 5.66%  q45 2.32%  q55 4.48%  (stops 49-99, verified)
  no_exploit:   q35 5.56%  q45 9.36%  q55 5.05%  (never terminates; +/-std up to 3.53)
