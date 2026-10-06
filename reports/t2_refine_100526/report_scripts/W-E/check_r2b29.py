"""Cross-check: means over seeds of the per-run table R2B-29 equal the R2C-12 rows used in table 6.3d,
and the baseline per-run values of R2B-03 equal those of the R1 stage-2 per-run table (R1-37)."""
import sys
sys.path.insert(0, '/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-'
                   'r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/'
                   'scratchpad/report_parts/W-E')
from common import *

b29 = rd('R2B-29')
c12 = rd('R2C-12')
cols = ['peak_visit_share', 'd1_flagged_L_s2', 'tail2q_mean_e2hat', 'tail2q_mean_abs_err_over_g2_0']
m = b29.groupby(['arm', 'q'])[cols].mean().reset_index()
bad = 0
for _, r in m.iterrows():
    name = r.arm if r.arm == 'A_base' else 'R2b_' + r.arm
    t = c12[(c12.arm == name) & (c12.q == r.q)]
    if len(t) != 1:
        continue
    for c in cols:
        if abs(float(t[c].iloc[0]) - r[c]) > 1e-6 * max(1.0, abs(r[c])):
            print('MISMATCH', name, r.q, c, float(t[c].iloc[0]), r[c])
            bad += 1
print('R2B-29 vs R2C-12 rows compared:', len(m), 'mismatches:', bad)

# tail statistic equals stage2_tail_mean_over_g2_0 in R2B-03
pr = rd('R2B-03')
x = pr.merge(b29, on=['arm', 'q', 'seed'])
print('tail2q_mean_abs_err_over_g2_0 vs stage2_tail_mean_over_g2_0 max abs diff:',
      (x.tail2q_mean_abs_err_over_g2_0_x - x.stage2_tail_mean_over_g2_0).abs().max()
      if 'tail2q_mean_abs_err_over_g2_0_x' in x else
      (x.tail2q_mean_abs_err_over_g2_0 - x.stage2_tail_mean_over_g2_0).abs().max())

# R1 A_base vs R2b A_base
r1 = rd('R1-37')
a = r1[r1.arm == 'A_base'][['q', 'seed', 'stage2_peak_rel_err_signed']]
b = pr[pr.arm == 'A_base'][['q', 'seed', 'stage2_peak_rel_err_signed']]
j = a.merge(b, on=['q', 'seed'])
print('R1-37 A_base vs R2B-03 A_base signed peak error: n =', len(j), 'max abs diff =',
      (j.stage2_peak_rel_err_signed_x - j.stage2_peak_rel_err_signed_y).abs().max())
# medians
for q in (50, 60):
    print(q, 'median signed A_base (R1-37):', float(a[a.q == q].stage2_peak_rel_err_signed.median()))
