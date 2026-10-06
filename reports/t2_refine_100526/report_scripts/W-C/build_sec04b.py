"""Build sections 4.5 and 4.6 of the PI report from the frozen evidence pack.

Every number that appears in the text goes through N(), which formats it and writes one
ledger row (statement_id, section, text, value, item_id, locator).
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python build_sec04b.py
Writes: ../sec04b.md and ../sec04b_ledger.csv (relative to this script's directory).
"""
import csv
import json
import os

import numpy as np
import pandas as pd

PACK = ('/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/'
        'reports/t2_refine_100526/evidence/')
R = PACK + 'results/'
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEDGER = []
CUR = {'section': ''}


def N(val, item, loc, fmt='g4'):
    """Format val, register a ledger row, return the string."""
    if fmt == 'g4':
        s = '{:.4g}'.format(val)
    elif fmt == 'd':
        s = str(int(round(val)))
    elif fmt == 'g3':
        s = '{:.3g}'.format(val)
    elif fmt == 'comma':
        s = '{:,}'.format(int(round(val)))
    elif fmt == 'f2':
        s = '{:.2f}'.format(val)
    elif fmt == 'f1':
        s = '{:.1f}'.format(val)
    elif fmt == 'raw':
        s = str(val)
    else:
        raise ValueError(fmt)
    LEDGER.append({'statement_id': 'S%04d' % (len(LEDGER) + 1), 'section': CUR['section'],
                   'text': s, 'value': repr(float(val)) if fmt != 'raw' else s,
                   'item_id': item, 'locator': loc})
    return s


def CI(m, lo, hi, item, loc):
    """mean [lo, hi] with three ledger rows."""
    return '%s [%s, %s]' % (N(m, item, loc + ' (mean)'), N(lo, item, loc + ' (ci lo)'),
                            N(hi, item, loc + ' (ci hi)'))


def rd(path):
    return pd.read_csv(R + path)


# ---------------------------------------------------------------- data
m5 = rd('v2_refine/analysis/decision_method5_pairs.csv')          # R1-08
dm_sum = rd('v2_refine/analysis/stage2_detmean_summary.csv')       # R1-33
crit_b = rd('v2_refine_r2b/analysis/criterion.csv')                # R2B-02
pair_b = rd('v2_refine_r2b/analysis/paired.csv')                   # R2B-08
per_b = rd('v2_refine_r2b/analysis/per_run.csv')                   # R2B-03
traj_b = rd('v2_refine_r2b/analysis/trajectory_per_run.csv')       # R2B-05
cost_p = rd('v2_refine_r2b/analysis/cost_P.csv')                   # R2B-11

s1crit = rd('v2_refine/analysis/stage1_criterion.csv')             # R1-02
s1disp = rd('v2_refine/analysis/stage1_dispersion.csv')            # R1-03
s1adv = rd('v2_refine/analysis/stage1_adv_ratio.csv')              # R1-04
s1pair = rd('v2_refine/analysis/stage1_paired.csv')                # R1-18
s1cost = rd('v2_refine/analysis/stage1_cost.csv')                  # R1-22
s1gate = rd('v2_refine/analysis/stage1_gate_counts.csv')           # R1-24
d1arms = rd('v2_refine/analysis/decision_d1_flags_arms.csv')       # R1-30
s1dec = rd('v2_refine/analysis/stage1_decomposition.csv')          # R1-26
chk2 = rd('v2_refine/analysis/decision_method6_check_ii.csv')      # R1-32
decin = rd('v2_refine/analysis/decision_inputs.csv')               # R1-01
cf08 = json.load(open(R + 'v2_T2_locked/rehearsal_v2_0_checks.json'))
pl05 = json.load(open(R + 'v2_T2_locked/v2_0/continuation_check_v2_0.json'))
cf01 = json.load(open(R + 'v2_T2_locked/confirmation_v2_0_analysis/verdict.json'))
cf02 = rd('v2_T2_locked/confirmation_v2_0_analysis/pass_counts.csv')
cf03 = rd('v2_T2_locked/confirmation_v2_0_analysis/per_run.csv')
cf04 = rd('v2_T2_locked/confirmation_v2_0_analysis/s1_summary.csv')
cf10 = rd('v2_T2_locked/confirmation_analysis/per_run.csv')
cf11 = rd('v2_T2_locked/confirmation_analysis/s1_summary.csv')

# ---------------------------------------------------------------- consistency checks (printed)
chk = []
end = traj_b[traj_b['update'] == 1800].groupby(['arm', 'q']).foc_mean.mean()
end3 = per_b.dropna(subset=['foc_offline_end_mean']).groupby(['arm', 'q']).foc_offline_end_mean.mean()
for k in end.index:
    chk.append('foc end R2B-05 vs R2B-03 %s %s: %.6g %.6g' % (k[0], k[1], end[k], end3[k]))
    assert abs(end[k] - end3[k]) < 1e-12
par_b = per_b[per_b.arm == 'parent_u1600'].groupby('q').stage2_rmse_pos_over_g2_0.mean()
abase_b = per_b[per_b.arm == 'A_base'].groupby('q').stage2_rmse_pos_over_g2_0.mean()
chk.append('parent rmse == A_base rmse: %s' % (np.allclose(par_b.values, abase_b.values)))
chk.append('phaseP_checks_all_true: ' + str(per_b.groupby('arm').phaseP_checks_all_true.agg(
    lambda s: int((s == True).sum())).to_dict()))  # noqa: E712
chk.append('phaseP flags (head/critic/adam): ' + str(per_b[per_b.arm.isin(['A_detmean', 'P20_lr3e-5', 'P20_lr3e-4'])].groupby('arm')[
    ['phaseP_head_bit_identical', 'phaseP_critic_bit_identical',
     'phaseP_critic_adam_bit_identical']].agg(lambda s: int((s == True).sum())).to_dict()))  # noqa: E712
print('\n'.join(chk))

# ---------------------------------------------------------------- helpers for 4.5


def m5row(comp, arm, base, q, metric):
    r = m5[(m5.comparison == comp) & (m5.arm == arm) & (m5.baseline == base) & (m5.q == q)
           & (m5.metric == metric)]
    assert len(r) == 1, (comp, arm, base, q, metric)
    return r.iloc[0]


def pbrow(arm, base, q, metric):
    r = pair_b[(pair_b.arm == arm) & (pair_b.baseline == base) & (pair_b.q == q)
               & (pair_b.metric == metric)]
    assert len(r) == 1, (arm, base, q, metric)
    return r.iloc[0]


def critrow(arm):
    r = crit_b[crit_b.arm == arm]
    assert len(r) == 1
    return r.iloc[0]


def mrow_cell(r, item, loc):
    """mean [CI] ; n_better/10 for a paired row."""
    return '%s; %s/10' % (CI(r['mean'], r['ci_mean_lo'], r['ci_mean_hi'], item, loc),
                          N(r['n_better'], item, loc + ' n_better', 'd'))


# ================================================================ SECTION 4.5
CUR['section'] = '4.5'
L = []
L.append('### 4.5 Method 5: deterministic-mean (pathwise) terminal fine-tuning\n')

p1 = critrow('P20_lr3e-5')
p2 = critrow('P20_lr3e-4')
loc_c = 'criterion.csv row arm='
L.append(
    '**Outcome.** At the same learning rate and the same number of optimiser steps, the pathwise '
    'arms do not beat PPO on the stage-2 peak error. `P20_lr3e-5` minus its PPO control '
    '`A_ctrl200` has intervals that contain 0 at both q (%s at q = 50; %s at q = 60) [R2B-02]. '
    '`P20_lr3e-4` minus its PPO control `A_ctrl200_lr3e-4` is worse at q = 50 (%s) and contains 0 '
    'at q = 60 (%s) [R2B-02]. The pathwise arms do lower the offline first-order-condition (FOC) '
    'residual and RMSE_pos/e2*(0) relative to their controls (table 3), but the peak error does '
    'not separate from the matched control. In R1 the single-step arm `A_detmean` also has '
    'intervals containing 0 against `A_ctrl200` at both q [R1-08]. **Decision (PI publication '
    'prompt, D1): method 5 (pathwise fine-tuning) is closed as negative at matched budgets.** '
    'The method is model-based and was pre-registered as ablation and diagnostic only, not as a '
    'candidate for the locked protocol [RR-09, section 2 table, row 3; RR-02, section 2; RR-05, '
    'column "protocol eligibility (D2)"].\n'
    % (CI(p1.mean_q50, p1.ci_mean_lo_q50, p1.ci_mean_hi_q50, 'R2B-02', loc_c + 'P20_lr3e-5 q50'),
       CI(p1.mean_q60, p1.ci_mean_lo_q60, p1.ci_mean_hi_q60, 'R2B-02', loc_c + 'P20_lr3e-5 q60'),
       CI(p2.mean_q50, p2.ci_mean_lo_q50, p2.ci_mean_hi_q50, 'R2B-02', loc_c + 'P20_lr3e-4 q50'),
       CI(p2.mean_q60, p2.ci_mean_lo_q60, p2.ci_mean_hi_q60, 'R2B-02', loc_c + 'P20_lr3e-4 q60')))

L.append('**What the method does as implemented** [RR-08, section 4 item 5; RR-09, section 4.3]. '
         'Phase P continues the stage-2 policy from the end-of-Phase-A state (update 1600, the '
         '"parent u1600 candidate") for 200 updates. Each update uses 512 fresh exploring-start '
         'rows and draws no shock and no action. The objective is the exact conditional expected '
         'terminal payoff R(d, e, e_opp) = w_L + DW F_xi(d + e - e_opp) - k e^2, where the '
         'learner effort e(d) is the mean of its Beta policy, e_min + range alpha/(alpha+beta), '
         'and the opponent is the lagged copy at -d, also at its Beta mean and without gradient. '
         'The loss is -mean R. The step is an Adam step on the actor (state preserved, clip 0.5), '
         'with no critic update, no PPO ratio, no clipping and no entropy term. Output row 1 of the '
         'actor (the concentration head) has its gradient zeroed and its weight and bias restored '
         'bit-exactly after every step; at the end of the phase the run asserts that the head, '
         'the critic and the critic\'s Adam state are bit-identical to the parent\'s (recorded '
         'per run: true in %s of %s runs for each of `A_detmean`, `P20_lr3e-5` and `P20_lr3e-4`; '
         'R2B-03 `phaseP_checks_all_true`). The payoff '
         'function is the known game, as in the locked expected-reward estimator; the closed-form '
         'equilibrium is not used. R1\'s `A_detmean` takes one full-batch step per update '
         '(200 steps). R2b\'s `P20` arms take E = 10 epochs over the 512 rows in minibatches of 256, '
         'i.e. 20 exact-gradient steps per update (4,000 in 200 updates), the optimiser budget '
         'of the PPO control (10 epochs x 2 minibatches) [RR-09, sections 3.2 and 4.3]. '
         'The R1 pair `A_detmean` / `A_ctrl200` matched updates and episodes (%s updates, %s '
         'episodes) but not optimiser steps (%s against %s) [R2B-11].\n'
         % (N(int(min((per_b[per_b.arm == a].phaseP_checks_all_true == True).sum()  # noqa: E712
                      for a in ('A_detmean', 'P20_lr3e-5', 'P20_lr3e-4'))), 'R2B-03',
              'per_run.csv phaseP_checks_all_true count True per arm (20 in each of the three arms)', 'd'),
            N(20, 'R2B-03', 'per_run.csv rows per arm (10 seeds x 2 q)', 'd'),
            N(cost_p[cost_p.arm == 'A_detmean'].mean_phase_local_updates.iloc[0], 'R2B-11',
              'cost_P.csv A_detmean mean_phase_local_updates', 'd'),
            '{:,}'.format(int(cost_p[cost_p.arm == 'A_detmean'].mean_phase_episodes.iloc[0])),
            N(cost_p[cost_p.arm == 'A_detmean'].mean_n_minibatch_steps_total.iloc[0], 'R2B-11',
              'cost_P.csv A_detmean mean_n_minibatch_steps_total', 'd'),
            N(cost_p[cost_p.arm == 'A_ctrl200'].mean_n_minibatch_steps_total.iloc[0], 'R2B-11',
              'cost_P.csv A_ctrl200 mean_n_minibatch_steps_total', 'comma')))
LEDGER.append({'statement_id': 'S%04d' % (len(LEDGER) + 1), 'section': '4.5', 'text': '102,400',
               'value': '102400.0', 'item_id': 'R2B-11',
               'locator': 'cost_P.csv A_detmean mean_phase_episodes'})

# ---- Table 1 (R1)
L.append('**Table 1. R1 (bootstrap seed 20261003): peak-error pairs of the single-step arm.** '
         'Primary metric = absolute signed peak error |(e2_hat(0) - e2*(0)) / e2*(0)|; the '
         'difference is arm minus comparator, paired by (q, seed), n = 10 development seeds per q; '
         'negative = arm better. `parent` = the u1600 candidate.\n')
L.append('| comparison | q | mean [95% CI] | median [95% CI] | n_better |')
L.append('|---|---|---|---|---|')
for lab, comp, arm, base in [
        ('`A_detmean` - `A_ctrl200` (ablation vs matched control)', 'ablation vs matched control',
         'A_detmean', 'A_ctrl200'),
        ('`A_detmean` - parent', 'vs parent u1600 candidate', 'A_detmean', 'parent_u1600'),
        ('`A_ctrl200` - parent', 'vs parent u1600 candidate', 'A_ctrl200', 'parent_u1600')]:
    for q in (50, 60):
        r = m5row(comp, arm, base, q, 'stage2_peak_rel_err_abs')
        loc = 'decision_method5_pairs.csv %s/%s q=%d stage2_peak_rel_err_abs' % (arm, base, q)
        L.append('| %s | %d | %s | %s | %s/10 |' % (
            lab, q, CI(r['mean'], r.ci_mean_lo, r.ci_mean_hi, 'R1-08', loc),
            CI(r['median'], r.ci_median_lo, r.ci_median_hi, 'R1-08', loc.replace('stage2', 'median stage2')),
            N(r.n_better, 'R1-08', loc + ' n_better', 'd')))
L.append('')
L.append('Source: R1-08 (`results/v2_refine/analysis/decision_method5_pairs.csv`, rows with metric '
         '`stage2_peak_rel_err_abs`; columns `mean`, `ci_mean_lo`, `ci_mean_hi`, `median`, '
         '`ci_median_lo`, `ci_median_hi`, `n_better`). The same table is section 2 of R1\'s '
         '`06_decision_inputs.md` [RR-02]. The R1 summary records the outcome as "`A_detmean` - '
         '`A_ctrl200` has an interval containing 0 at both q" [RR-01]. `A_detmean` has no D1 row '
         'because phase P draws no action [RR-08, section 7 "Facts for reading the results"].\n')

# ---- Table 2 (R2b)
L.append('**Table 2. R2b wave P, matched budget (bootstrap seed 20261004).** Arm minus comparator '
         'on the same primary metric, paired by (q, seed), n = 10 per q; `n_better` = seeds with '
         'a strictly lower |peak error|. Part (a): the interval of the mean difference lies below 0 '
         'at both q. Part (b): no run that passed G-A and its G-N part under the comparator fails '
         'it under the arm. The last row is the PPO control at LR 3e-4 against the parent.\n')
L.append('| arm | comparator | q = 50: mean [95% CI]; n_better | q = 60: mean [95% CI]; n_better '
         '| part (a) | part (b) |')
L.append('|---|---|---|---|---|---|')
t2rows = [('P20_lr3e-5', 'A_ctrl200', 'matched control'),
          ('P20_lr3e-4', 'A_ctrl200_lr3e-4', 'matched control'),
          ('A_ctrl200_lr3e-4', 'parent_u1600', 'parent u1600 candidate')]
for arm, base, lab in t2rows:
    c = critrow(arm)
    assert c.baseline == base
    r50 = pbrow(arm, base, 50, 'stage2_peak_rel_err_abs')
    r60 = pbrow(arm, base, 60, 'stage2_peak_rel_err_abs')
    # part (a) wording as recorded in R2B-05 / RR-05: "not met" plus per-q detail from a_q50/a_q60
    def side(lo, hi):
        return 'below 0' if hi < 0 else ('above 0' if lo > 0 else 'contains 0')
    pa = 'not met' if not c.a_met else 'met'
    pa += ' (q = 50: interval %s; q = 60: %s)' % (side(c.ci_mean_lo_q50, c.ci_mean_hi_q50),
                                                 side(c.ci_mean_lo_q60, c.ci_mean_hi_q60))
    if c.b_status == 'holds':
        pb = 'holds'
    else:
        viol = str(c.b_violations)
        assert viol == 'q50/10510'
        pb = ('violated: q = 50 seed 10510 (eta_2/DW %s > 0.005)' % N(0.005501, 'RR-05',
              'row "A_ctrl200_lr3e-4 vs parent_u1600" part (b) text (report only)', 'raw'))
    L.append('| `%s` | `%s` (%s) | %s | %s | %s | %s |' % (
        arm, base, lab,
        mrow_cell(r50, 'R2B-08', 'paired.csv %s vs %s q=50 stage2_peak_rel_err_abs' % (arm, base)),
        mrow_cell(r60, 'R2B-08', 'paired.csv %s vs %s q=60 stage2_peak_rel_err_abs' % (arm, base)),
        pa, pb))
L.append('')
L.append('Source: R2B-02 (`results/v2_refine_r2b/analysis/criterion.csv`: `a_met`, `a_q50`, '
         '`a_q60`, `b_status`, `b_violations`) for the verdicts; R2B-08 (`paired.csv`, rows with '
         'metric `stage2_peak_rel_err_abs`: `mean`, `ci_mean_lo`, `ci_mean_hi`, `n_better`) for '
         'the cells, equal to the `mean_q*`, `ci_mean_*_q*` columns of R2B-02. The eta_2/DW value '
         'of the part-(b) violation (0.005501) is from the R2b decision-inputs report [RR-05, '
         'observation on `A_ctrl200_lr3e-4` vs `parent_u1600`] (report only). The verdict '
         'wording is that of RR-05 and RR-04 ("not met (worse at q50)" for `P20_lr3e-4`).\n')

r_ctl_par = pbrow('A_ctrl200_lr3e-4', 'parent_u1600', 50, 'stage2_peak_rel_err_abs')
r_ctl_ctl = pbrow('A_ctrl200_lr3e-4', 'A_ctrl200', 50, 'stage2_peak_rel_err_abs')
L.append('**The control at LR 3e-4.** The new PPO control `A_ctrl200_lr3e-4` improves the peak '
         'at q = 50 against the parent candidate (%s; %s of 10 seeds) and against R1\'s '
         '`A_ctrl200` (%s; %s of 10) [R2B-08], with one G-A failure (q = 50 seed 10510) among its 20 '
         'runs; at q = 60 its interval against the parent contains 0 (%s) [R2B-08]. R2b records '
         'a higher LR over the same 200 updates as "a confound that the pathwise arms must be read '
         'against" [RR-04]; for that reason `P20_lr3e-4` is compared with this control and not with '
         '`A_ctrl200`. Against its own control `P20_lr3e-4` has %s of 10 seeds better at q = 50 '
         'and %s of 10 at q = 60 [R2B-08].\n'
         % (CI(r_ctl_par['mean'], r_ctl_par.ci_mean_lo, r_ctl_par.ci_mean_hi, 'R2B-08',
               'paired.csv A_ctrl200_lr3e-4 vs parent_u1600 q=50 stage2_peak_rel_err_abs'),
            N(r_ctl_par.n_better, 'R2B-08', 'paired.csv A_ctrl200_lr3e-4 vs parent q=50 n_better', 'd'),
            CI(r_ctl_ctl['mean'], r_ctl_ctl.ci_mean_lo, r_ctl_ctl.ci_mean_hi, 'R2B-08',
               'paired.csv A_ctrl200_lr3e-4 vs A_ctrl200 q=50 stage2_peak_rel_err_abs'),
            N(r_ctl_ctl.n_better, 'R2B-08', 'paired.csv A_ctrl200_lr3e-4 vs A_ctrl200 q=50 n_better', 'd'),
            (lambda r: CI(r['mean'], r.ci_mean_lo, r.ci_mean_hi, 'R2B-08',
                          'paired.csv A_ctrl200_lr3e-4 vs parent_u1600 q=60 stage2_peak_rel_err_abs'))(
                pbrow('A_ctrl200_lr3e-4', 'parent_u1600', 60, 'stage2_peak_rel_err_abs')),
            N(pbrow('P20_lr3e-4', 'A_ctrl200_lr3e-4', 50, 'stage2_peak_rel_err_abs').n_better,
              'R2B-08', 'paired.csv P20_lr3e-4 vs A_ctrl200_lr3e-4 q=50 n_better', 'd'),
            N(pbrow('P20_lr3e-4', 'A_ctrl200_lr3e-4', 60, 'stage2_peak_rel_err_abs').n_better,
              'R2B-08', 'paired.csv P20_lr3e-4 vs A_ctrl200_lr3e-4 q=60 n_better', 'd')))

# ---- Table 3 FOC / RMSE
L.append('**The FOC residual and RMSE observation.** The offline FOC residual is the mean '
         '|dR/de| of the exact payoff gradient on the bin centres of a fixed grid, evaluated from '
         'the weight exports (mean over the 10 seeds of each q; global update 1600 = the parent, '
         '1800 = end of the 200 updates) [RR-05, "Wave P" observations; FG-17]. RMSE_pos/e2*(0) is '
         'the G-A recovery component (limit 0.05): the root-mean-square error of e2_hat(d) against e2*(d) '
         'over the recovery-grid nodes with |d| < 2q, divided by e2*(0) [PL-01, gates.G-A]. The parent value '
         'is the mean of the `parent_u1600` rows.\n')
L.append('**Table 3. Offline FOC residual and RMSE_pos/e2*(0), before and after the 200 updates.** '
         'The two right-hand columns are paired differences against the matched control '
         '(`A_detmean`: R1-08, bootstrap seed 20261003; `P20` arms: R2B-08, seed 20261004); '
         'n_better = seeds with lower RMSE_pos/e2*(0).\n')
L.append('| q | arm | FOC residual, 1600 -> 1800 | RMSE_pos/e2*(0), parent -> end | '
         'RMSE_pos: arm - matched control [95% CI]; n_better | peak error: arm - matched control '
         '[95% CI] |')
L.append('|---|---|---|---|---|---|')
arms3 = [('A_ctrl200', None), ('A_detmean', 'A_ctrl200'), ('P20_lr3e-5', 'A_ctrl200'),
         ('A_ctrl200_lr3e-4', None), ('P20_lr3e-4', 'A_ctrl200_lr3e-4')]
for q in (50, 60):
    t = traj_b[traj_b.q == q]
    rm_par = per_b[(per_b.arm == 'parent_u1600') & (per_b.q == q)].stage2_rmse_pos_over_g2_0.mean()
    foc0 = t[(t.arm == 'A_ctrl200') & (t['update'] == 1600)].foc_mean.mean()
    for arm, ctl in arms3:
        f1 = t[(t.arm == arm) & (t['update'] == 1800)].foc_mean.mean()
        rm_end = per_b[(per_b.arm == arm) & (per_b.q == q)].stage2_rmse_pos_over_g2_0.mean()
        if ctl is None:
            c_r, c_p = '(is a control)', '(is a control)'
        else:
            if arm == 'A_detmean':
                rr = m5row('ablation vs matched control', 'A_detmean', 'A_ctrl200', q,
                           'stage2_rmse_pos_over_g2_0')
                pp = m5row('ablation vs matched control', 'A_detmean', 'A_ctrl200', q,
                           'stage2_peak_rel_err_abs')
                it, loc0 = 'R1-08', 'decision_method5_pairs.csv A_detmean/A_ctrl200 q=%d' % q
            else:
                rr = pbrow(arm, ctl, q, 'stage2_rmse_pos_over_g2_0')
                pp = pbrow(arm, ctl, q, 'stage2_peak_rel_err_abs')
                it, loc0 = 'R2B-08', 'paired.csv %s vs %s q=%d' % (arm, ctl, q)
            c_r = '%s; %s/10' % (CI(rr['mean'], rr.ci_mean_lo, rr.ci_mean_hi, it,
                                    loc0 + ' stage2_rmse_pos_over_g2_0'),
                                 N(rr.n_better, it, loc0 + ' rmse n_better', 'd'))
            c_p = CI(pp['mean'], pp.ci_mean_lo, pp.ci_mean_hi, it, loc0 + ' stage2_peak_rel_err_abs')
        L.append('| %d | `%s` | %s -> %s | %s -> %s | %s | %s |' % (
            q, arm,
            N(foc0, 'R2B-05', 'trajectory_per_run.csv q=%d update=1600 foc_mean mean over seeds' % q),
            N(f1, 'R2B-05', 'trajectory_per_run.csv q=%d arm=%s update=1800 foc_mean mean over seeds' % (q, arm)),
            N(rm_par, 'R2B-03', 'per_run.csv arm=parent_u1600 q=%d stage2_rmse_pos_over_g2_0 mean over seeds' % q),
            N(rm_end, 'R2B-03', 'per_run.csv arm=%s q=%d stage2_rmse_pos_over_g2_0 mean over seeds' % (arm, q)),
            c_r, c_p))
L.append('')
L.append('Source: R2B-05 (`results/v2_refine_r2b/analysis/trajectory_per_run.csv`, column '
         '`foc_mean`, rows `update` = 1600 and 1800, averaged over the 10 seeds of each q here; '
         'the 1800 values equal the mean of `foc_offline_end_mean` in R2B-03, checked here); R2B-03 '
         '(`per_run.csv`, column `stage2_rmse_pos_over_g2_0`, mean over seeds; rows `arm` = '
         '`parent_u1600` for the parent); R1-08 and R2B-08 (`stage2_rmse_pos_over_g2_0` and '
         '`stage2_peak_rel_err_abs` rows) for the paired columns. Means over seeds are computed '
         'here; the values agree with the R2b decision-inputs report [RR-05] and summary [RR-04], '
         'which give the FOC means to 2 or 4 significant digits.\n')

# FOC/RMSE sentence + check against RR-04
fo = lambda arm, q: traj_b[(traj_b.arm == arm) & (traj_b.q == q) & (traj_b['update'] == 1800)].foc_mean.mean()
f0 = lambda q: traj_b[(traj_b.arm == 'A_ctrl200') & (traj_b.q == q) & (traj_b['update'] == 1600)].foc_mean.mean()
dm_foc50 = dm_sum[(dm_sum.q == 50) & (dm_sum.arm == 'A_detmean - A_ctrl200') & (dm_sum.metric == 'foc_offline_end_mean')].iloc[0]
dm_foc60 = dm_sum[(dm_sum.q == 60) & (dm_sum.arm == 'A_detmean - A_ctrl200') & (dm_sum.metric == 'foc_offline_end_mean')].iloc[0]
# n seeds with lower end FOC than the matched control (computed here)
pe = per_b.dropna(subset=['foc_offline_end_mean'])


def n_lower(arm, ctl, q):
    a = pe[(pe.arm == arm) & (pe.q == q)].set_index('seed').foc_offline_end_mean
    c = pe[(pe.arm == ctl) & (pe.q == q)].set_index('seed').foc_offline_end_mean
    return int((a.sort_index() < c.sort_index()).sum()), len(a)


nl = {(a, q): n_lower(a, c, q) for a, c in [('P20_lr3e-5', 'A_ctrl200'), ('P20_lr3e-4', 'A_ctrl200_lr3e-4'),
                                           ('A_detmean', 'A_ctrl200')] for q in (50, 60)}
L.append('At q = 50 the FOC residual falls from %s to %s for `P20_lr3e-5`, to %s for `P20_lr3e-4` and '
         'to %s for `A_detmean`; it falls to %s for `A_ctrl200` and rises to %s for `A_ctrl200_lr3e-4` '
         '[R2B-05]. The R2b summary gives the same q = 50 values to two digits (5.2e-4 to 3.9e-4, '
         '4.3e-4, 4.4e-4, 4.9e-4 and 6.6e-4) [RR-04], and they equal R2B-05 at that '
         'precision (checked here); the q = 60 values are in the table and in RR-05. Computed here from R2B-03, the '
         'end-of-phase FOC residual of `P20_lr3e-5` is below `A_ctrl200`\'s in %s of %s seeds at q = 50 '
         'and %s of %s at q = 60, that of `P20_lr3e-4` is below `A_ctrl200_lr3e-4`\'s in %s of %s and '
         '%s of %s, and that of `A_detmean` is below `A_ctrl200`\'s in %s of %s and %s of %s. In R1 the '
         'paired difference of the end-of-phase FOC residual, `A_detmean` - `A_ctrl200`, is %s (q = 50) '
         'and %s (q = 60) [R1-33].\n'
         % (N(f0(50), 'R2B-05', 'trajectory q=50 update=1600 foc_mean mean'),
            N(fo('P20_lr3e-5', 50), 'R2B-05', 'trajectory q=50 P20_lr3e-5 1800 mean'),
            N(fo('P20_lr3e-4', 50), 'R2B-05', 'trajectory q=50 P20_lr3e-4 1800 mean'),
            N(fo('A_detmean', 50), 'R2B-05', 'trajectory q=50 A_detmean 1800 mean'),
            N(fo('A_ctrl200', 50), 'R2B-05', 'trajectory q=50 A_ctrl200 1800 mean'),
            N(fo('A_ctrl200_lr3e-4', 50), 'R2B-05', 'trajectory q=50 A_ctrl200_lr3e-4 1800 mean'),
            N(nl[('P20_lr3e-5', 50)][0], 'R2B-03', 'computed here foc_offline_end_mean seedwise', 'd'),
            N(nl[('P20_lr3e-5', 50)][1], 'R2B-03', 'computed here n seeds', 'd'),
            N(nl[('P20_lr3e-5', 60)][0], 'R2B-03', 'computed here', 'd'),
            N(nl[('P20_lr3e-5', 60)][1], 'R2B-03', 'computed here n seeds', 'd'),
            N(nl[('P20_lr3e-4', 50)][0], 'R2B-03', 'computed here', 'd'),
            N(nl[('P20_lr3e-4', 50)][1], 'R2B-03', 'computed here n seeds', 'd'),
            N(nl[('P20_lr3e-4', 60)][0], 'R2B-03', 'computed here', 'd'),
            N(nl[('P20_lr3e-4', 60)][1], 'R2B-03', 'computed here n seeds', 'd'),
            N(nl[('A_detmean', 50)][0], 'R2B-03', 'computed here', 'd'),
            N(nl[('A_detmean', 50)][1], 'R2B-03', 'computed here n seeds', 'd'),
            N(nl[('A_detmean', 60)][0], 'R2B-03', 'computed here', 'd'),
            N(nl[('A_detmean', 60)][1], 'R2B-03', 'computed here n seeds', 'd'),
            CI(dm_foc50['mean'], dm_foc50.ci_mean_lo, dm_foc50.ci_mean_hi, 'R1-33',
               'stage2_detmean_summary.csv q=50 A_detmean - A_ctrl200 foc_offline_end_mean'),
            CI(dm_foc60['mean'], dm_foc60.ci_mean_lo, dm_foc60.ci_mean_hi, 'R1-33',
               'stage2_detmean_summary.csv q=60 A_detmean - A_ctrl200 foc_offline_end_mean')))

pp1 = pbrow('P20_lr3e-5', 'A_ctrl200', 50, 'stage2_rmse_pos_over_g2_0')
L.append('**What moves and what does not (observation).** Against their matched controls the pathwise '
         'arms lower the offline FOC residual and RMSE_pos/e2*(0) (RMSE_pos is lower in %s of 10 seeds '
         'for both `P20` arms at both q, with intervals below 0; R2B-08), while the paired difference '
         'of the peak error contains 0 for `P20_lr3e-5` at both q and is above 0 for `P20_lr3e-4` at '
         'q = 50 (table 2). Two further readings of the same tables: the PPO control at LR 3e-4 raises '
         'RMSE_pos/e2*(0) against the parent (%s at q = 50 [R2B-08]) while it lowers the peak error at '
         'q = 50; and R1\'s `A_detmean` has an RMSE_pos interval below 0 at q = 60 and containing 0 at '
         'q = 50 against `A_ctrl200` (table 3). The logged per-update loss and FOC curves of the P20 '
         'arms and of `A_detmean` are on different time bases and are not compared (RR-09 section 4.3; '
         'FG-18); the offline curves are in `figures/FG-17_waveP_offline_trajectory.png`, and R1\'s are '
         'in `figures/FG-05_stage2_A_detmean_foc.png` and `figures/FG-06_stage2_A_detmean_objective.png`.\n'
         % (N(10, 'R2B-08', 'paired.csv P20_lr3e-5/P20_lr3e-4 rmse n_better = 10 in all four rows', 'd'),
            CI(*[pbrow('A_ctrl200_lr3e-4', 'parent_u1600', 50, 'stage2_rmse_pos_over_g2_0')[k]
                 for k in ('mean', 'ci_mean_lo', 'ci_mean_hi')], 'R2B-08',
               'paired.csv A_ctrl200_lr3e-4 vs parent q=50 rmse')))
# sanity: all four P20 rmse rows have n_better 10 and CI below 0
for arm, ctl in [('P20_lr3e-5', 'A_ctrl200'), ('P20_lr3e-4', 'A_ctrl200_lr3e-4')]:
    for q in (50, 60):
        r = pbrow(arm, ctl, q, 'stage2_rmse_pos_over_g2_0')
        assert r.n_better == 10 and r.ci_mean_hi < 0
a50 = m5row('ablation vs matched control', 'A_detmean', 'A_ctrl200', 50, 'stage2_rmse_pos_over_g2_0')
a60 = m5row('ablation vs matched control', 'A_detmean', 'A_ctrl200', 60, 'stage2_rmse_pos_over_g2_0')
assert a50.ci_mean_hi > 0 and a60.ci_mean_hi < 0

# ---- cost
cA = cost_p.set_index('arm')
L.append('**Cost.** Per run, the mean phase wall time is %s s for `P20_lr3e-5` and %s s for '
         '`P20_lr3e-4`, against %s s for `A_ctrl200` and %s s for `A_ctrl200_lr3e-4`, for the same %s '
         'optimiser steps (ratios %s and %s to the PPO control of the same LR); `A_detmean` takes %s s '
         'for %s steps [R2B-11, 20 runs per arm]. This agrees with the R2b summary ("17 s against '
         '27-28 s") [RR-04]. Wall time depends on machine load: waves A and P ran together with up to '
         '40 processes, and the R1 comparator `A_ctrl200` ran on 2026-10-03, so only the step counts '
         'are comparable across arms [RR-05]. The control `A_ctrl200_lr3e-4` ran in the R2b waves.\n'
         % (N(cA.loc['P20_lr3e-5', 'mean_phase_wall_sec'], 'R2B-11', 'cost_P.csv P20_lr3e-5 mean_phase_wall_sec', 'f2'),
            N(cA.loc['P20_lr3e-4', 'mean_phase_wall_sec'], 'R2B-11', 'cost_P.csv P20_lr3e-4 mean_phase_wall_sec', 'f2'),
            N(cA.loc['A_ctrl200', 'mean_phase_wall_sec'], 'R2B-11', 'cost_P.csv A_ctrl200 mean_phase_wall_sec', 'f2'),
            N(cA.loc['A_ctrl200_lr3e-4', 'mean_phase_wall_sec'], 'R2B-11', 'cost_P.csv A_ctrl200_lr3e-4 mean_phase_wall_sec', 'f2'),
            '{:,}'.format(int(cA.loc['P20_lr3e-5', 'mean_n_minibatch_steps_total'])),
            N(cA.loc['P20_lr3e-5', 'wall_ratio_vs_same_lr_ppo_control'], 'R2B-11', 'cost_P.csv P20_lr3e-5 wall_ratio_vs_same_lr_ppo_control'),
            N(cA.loc['P20_lr3e-4', 'wall_ratio_vs_same_lr_ppo_control'], 'R2B-11', 'cost_P.csv P20_lr3e-4 wall_ratio_vs_same_lr_ppo_control'),
            N(cA.loc['A_detmean', 'mean_phase_wall_sec'], 'R2B-11', 'cost_P.csv A_detmean mean_phase_wall_sec', 'f2'),
            N(cA.loc['A_detmean', 'mean_n_minibatch_steps_total'], 'R2B-11', 'cost_P.csv A_detmean mean_n_minibatch_steps_total', 'd')))
for v, loc in [(4000, 'cost_P.csv P20_lr3e-5 mean_n_minibatch_steps_total')]:
    LEDGER.append({'statement_id': 'S%04d' % (len(LEDGER) + 1), 'section': '4.5', 'text': '4,000',
                   'value': '4000.0', 'item_id': 'R2B-11', 'locator': loc})
L.append('**Status.** Method 5 (pathwise fine-tuning) is closed as negative at matched budgets '
         '(PI publication prompt, D1). It is an ablation and a diagnostic [RR-09, section 2]; the locked '
         'protocol v2.0 contains no pathwise setting [PL-01, searched here for "pathwise", "phase_P" '
         'and "detmean": no match].\n')

SEC45 = '\n'.join(L)

# ================================================================ SECTION 4.6
CUR['section'] = '4.6'
L = []
L.append('### 4.6 Method 6: expected continuation\n')


def s1c(arm, q, col, base=None):
    pass


def row_first(df, **kw):
    d = df
    for k, v in kw.items():
        d = d[d[k] == v]
    assert len(d) == 1, kw
    return d.iloc[0]


ex = {q: row_first(decin, arm='B_expcont') for q in (50, 60)}
dx = decin[decin.arm == 'B_expcont'].iloc[0]
crt = s1crit[s1crit.arm == 'B_expcont'].iloc[0]
g50b = row_first(s1gate, arm='B_base', q=50)
g60b = row_first(s1gate, arm='B_base', q=60)
g50e = row_first(s1gate, arm='B_expcont', q=50)
g60e = row_first(s1gate, arm='B_expcont', q=60)

# fresh-seed numbers computed here
def s1stats(df, q):
    d = df[(df.q == q) & (df.status == 'completed')]
    return len(d), float(np.median(d.s1)), float(d.s1.max()), int((d.s1 <= 0.05).sum()), int((d.s1 <= 0.10).sum())


fs = {}
for q in (50, 60):
    fs[('v1.1', q)] = s1stats(cf10, q)
    fs[('v2.0', q)] = s1stats(cf03, q)
sd11 = {q: float(cf11[cf11.q == q].sd_signed.iloc[0]) for q in (50, 60)}
sd20 = {q: float(cf04[(cf04.q == q) & (cf04.criterion == 'G-S')].sd_signed.iloc[0]) for q in (50, 60)}
gs_pass = {q: int(cf04[(cf04.q == q) & (cf04.criterion == 'G-S')].n_pass.iloc[0]) for q in (50, 60)}
runpass = {q: (int(cf02[cf02.q == q].n_pass.iloc[0]), int(cf02[cf02.q == q].n_expected.iloc[0])) for q in (50, 60)}
v11_s1pass = {q: int(cf11[cf11.q == q].S1_pass.iloc[0]) for q in (50, 60)}

L.append(
    '**Outcome.** Of the six R1 candidates, method 6 is the only one that met the pre-registered '
    'criterion; the PI adopted it into protocol v2.0, and nothing else from R1. The v2.0 '
    're-rehearsal on the development seeds reproduces the R1 pilot\'s Phase B in %s of %s runs, and '
    'the fresh-seed confirmation (seeds 30501-30520) gives the verdict "%s" under the rule "%s", with '
    'gate G-S satisfied by %s of %s runs at q = 50 and %s of %s at q = 60 [CF-01, CF-04, CF-08]. '
    'The chain has seven links.\n'
    % (N(cf08['R2']['n_identical'], 'CF-08', 'rehearsal_v2_0_checks.json R2.n_identical', 'd'),
       N(cf08['R2']['n'], 'CF-08', 'rehearsal_v2_0_checks.json R2.n', 'd'),
       cf01['overall'], cf01['rule'],
       N(gs_pass[50], 'CF-04', 's1_summary.csv q=50 criterion=G-S n_pass', 'd'),
       N(20, 'CF-04', 's1_summary.csv q=50 criterion=G-S n', 'd'),
       N(gs_pass[60], 'CF-04', 's1_summary.csv q=60 criterion=G-S n_pass', 'd'),
       N(20, 'CF-04', 's1_summary.csv q=60 criterion=G-S n', 'd')))

L.append('1. **The idea and where it enters.** In Phase B the stage-1 return uses the shock-integrated '
         'table value of the frozen stage-2 policy instead of the sampled continuation; Phase A '
         'and the stage-2 gates are untouched. The definition and the fixed integration rule are in '
         'section 3.1 and are not repeated here [PL-01 `pipeline.continuation_table`; RR-08, '
         'section 4 item 6].\n')

# ---- step 2: R1 pilot table
L.append('2. **R1 pilot, arm `B_expcont`** (stage 1, 20 runs = 10 development seeds x 2 q, paired with '
         '`B_base`; bootstrap seed 20261003). The S1 error is |e1_hat(0) - e1*| / e1*. Table 4 gives '
         'the numbers. The paired S1-error difference lies below 0 at both q: criterion part (a) is '
         'met; part (b) holds, with %s violations; the recorded overall outcome is "%s" [R1-02]. '
         'No other stage-1 arm has an interval below 0 at both q [RR-02, observations]. The signed error does '
         'not move: the paired signed difference has intervals containing 0 at both q. What changes is '
         'the spread: the across-seed SD of the signed error is %s and %s times the baseline\'s, '
         'the within-run SD of e1_hat(0) over the last five weight exports falls, the SD of the raw '
         'stage-1 advantages falls to %s and %s times the baseline\'s, and the number of runs within '
         '0.05 rises from %s and %s to %s and %s of 10. The phase wall time is %s times the baseline\'s '
         'including the table build [R1-22]. For phase B, D1 flags M1 and M2 are "%s" and "%s" '
         '(%s runs) [R1-30]. The induced target e~1 and its band are shared by every arm of a (q, '
         'seed) because every stage-1 run starts from a bit-identical frozen stage-2 snapshot '
         '[RR-02, observations "Run hygiene"; R1-26: median e~1 %s (q = 50) and %s (q = 60), median '
         'band width %s and %s, identical for `B_base` and `B_expcont`].\n'
         % (N(crt.b_n_violations, 'R1-02', 'stage1_criterion.csv B_expcont b_n_violations', 'd'),
            crt.overall,
            N(dx.disp_ratio_q50, 'R1-03', 'stage1_dispersion.csv B_expcont q=50 stage1_rel_err_signed ratio'),
            N(dx.disp_ratio_q60, 'R1-03', 'stage1_dispersion.csv B_expcont q=60 stage1_rel_err_signed ratio'),
            N(row_first(s1adv, arm='B_expcont', q=50).mean_ratio, 'R1-04', 'stage1_adv_ratio.csv B_expcont q=50 mean_ratio'),
            N(row_first(s1adv, arm='B_expcont', q=60).mean_ratio, 'R1-04', 'stage1_adv_ratio.csv B_expcont q=60 mean_ratio'),
            N(g50b.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_base q=50 n_target0929_pass', 'd'),
            N(g60b.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_base q=60 n_target0929_pass', 'd'),
            N(g50e.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_expcont q=50 n_target0929_pass', 'd'),
            N(g60e.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_expcont q=60 n_target0929_pass', 'd'),
            N(row_first(s1cost, arm='B_expcont').phase_wall_ratio_vs_base, 'R1-22',
              'stage1_cost.csv B_expcont phase_wall_ratio_vs_base'),
            row_first(d1arms, group='stage1/B_expcont').M1_outcome,
            row_first(d1arms, group='stage1/B_expcont').M2_outcome,
            N(row_first(d1arms, group='stage1/B_expcont').n_runs, 'R1-30', 'decision_d1_flags_arms.csv stage1/B_expcont n_runs', 'd'),
            N(row_first(s1dec, arm='B_expcont', q=50).median_e_tilde, 'R1-26', 'stage1_decomposition.csv B_expcont q=50 median_e_tilde'),
            N(row_first(s1dec, arm='B_expcont', q=60).median_e_tilde, 'R1-26', 'stage1_decomposition.csv B_expcont q=60 median_e_tilde'),
            N(row_first(s1dec, arm='B_expcont', q=50).median_band_width, 'R1-26', 'stage1_decomposition.csv B_expcont q=50 median_band_width'),
            N(row_first(s1dec, arm='B_expcont', q=60).median_band_width, 'R1-26', 'stage1_decomposition.csv B_expcont q=60 median_band_width')))
for a in ('B_base',):
    for q in (50, 60):
        assert row_first(s1dec, arm='B_base', q=q).median_e_tilde == row_first(s1dec, arm='B_expcont', q=q).median_e_tilde
        assert row_first(s1dec, arm='B_base', q=q).median_band_width == row_first(s1dec, arm='B_expcont', q=q).median_band_width

pg = lambda q, met, col: row_first(s1pair, arm='B_expcont', q=q, metric=met)[col]
disp = lambda q: row_first(s1disp, arm='B_expcont', q=q, metric='stage1_rel_err_signed')
L.append('**Table 4. R1 pilot, `B_expcont` - `B_base` (stage 1; bootstrap seed 20261003; n = 10 seeds per q).**\n')
L.append('| quantity | q = 50 | q = 60 |')
L.append('|---|---|---|')


def cellp(met, q, it='R1-18', file='stage1_paired.csv'):
    r = row_first(s1pair, arm='B_expcont', q=q, metric=met)
    return CI(r['mean'], r.ci_mean_lo, r.ci_mean_hi, it, '%s B_expcont q=%d %s' % (file, q, met))


def cellmed(met, q):
    r = row_first(s1pair, arm='B_expcont', q=q, metric=met)
    return CI(r['median'], r.ci_median_lo, r.ci_median_hi, 'R1-18', 'stage1_paired.csv B_expcont q=%d %s median' % (q, met))


L.append('| S1 error, paired difference of the mean [95%% CI] | %s | %s |' % (
    cellp('stage1_rel_err_abs', 50), cellp('stage1_rel_err_abs', 60)))
L.append('| S1 error, paired difference of the median [95%% CI] | %s | %s |' % (
    cellmed('stage1_rel_err_abs', 50), cellmed('stage1_rel_err_abs', 60)))
L.append('| seeds with a lower S1 error (n_better) | %s/10 | %s/10 |' % (
    N(pg(50, 'stage1_rel_err_abs', 'n_better'), 'R1-18', 'stage1_paired.csv q=50 stage1_rel_err_abs n_better', 'd'),
    N(pg(60, 'stage1_rel_err_abs', 'n_better'), 'R1-18', 'stage1_paired.csv q=60 stage1_rel_err_abs n_better', 'd')))
L.append('| criterion part (a) / (b) | %s / %s | %s / %s |' % (
    'met' if crt.a_q50 else 'not met', crt.b_status, 'met' if crt.a_q60 else 'not met', crt.b_status))
L.append('| signed error, paired difference of the mean [95%% CI] | %s | %s |' % (
    cellp('stage1_rel_err_signed', 50), cellp('stage1_rel_err_signed', 60)))
L.append('| SD of the signed error across seeds: arm; baseline | %s; %s | %s; %s |' % (
    N(disp(50).sd_arm, 'R1-03', 'stage1_dispersion.csv q=50 sd_arm'), N(disp(50).sd_base, 'R1-03', 'stage1_dispersion.csv q=50 sd_base'),
    N(disp(60).sd_arm, 'R1-03', 'stage1_dispersion.csv q=60 sd_arm'), N(disp(60).sd_base, 'R1-03', 'stage1_dispersion.csv q=60 sd_base')))
L.append('| dispersion ratio (arm / baseline) [95%% CI] | %s | %s |' % (
    CI(disp(50).ratio, disp(50).ci_lo, disp(50).ci_hi, 'R1-03', 'stage1_dispersion.csv q=50 ratio'),
    CI(disp(60).ratio, disp(60).ci_lo, disp(60).ci_hi, 'R1-03', 'stage1_dispersion.csv q=60 ratio')))
L.append('| within-run SD of e1_hat(0), last five exports: paired difference [95%% CI], effort units | %s | %s |' % (
    cellp('within_run_sd_e1_last5', 50), cellp('within_run_sd_e1_last5', 60)))
L.append('| stage-1 advantage SD, mean per-pair ratio (arm / baseline) | %s | %s |' % (
    N(row_first(s1adv, arm='B_expcont', q=50).mean_ratio, 'R1-04', 'stage1_adv_ratio.csv q=50 mean_ratio'),
    N(row_first(s1adv, arm='B_expcont', q=60).mean_ratio, 'R1-04', 'stage1_adv_ratio.csv q=60 mean_ratio')))
L.append('| runs with S1 error <= 0.05 (`n_target0929_pass`): `B_base` -> `B_expcont`, of 10 | %s -> %s | %s -> %s |' % (
    N(g50b.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_base q=50 n_target0929_pass', 'd'),
    N(g50e.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_expcont q=50 n_target0929_pass', 'd'),
    N(g60b.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_base q=60 n_target0929_pass', 'd'),
    N(g60e.n_target0929_pass, 'R1-24', 'stage1_gate_counts.csv B_expcont q=60 n_target0929_pass', 'd')))
L.append('| runs with S1 error <= 0.10 (`n_S1_pass`): `B_base` -> `B_expcont`, of 10 | %s -> %s | %s -> %s |' % (
    N(g50b.n_S1_pass, 'R1-24', 'stage1_gate_counts.csv B_base q=50 n_S1_pass', 'd'),
    N(g50e.n_S1_pass, 'R1-24', 'stage1_gate_counts.csv B_expcont q=50 n_S1_pass', 'd'),
    N(g60b.n_S1_pass, 'R1-24', 'stage1_gate_counts.csv B_base q=60 n_S1_pass', 'd'),
    N(g60e.n_S1_pass, 'R1-24', 'stage1_gate_counts.csv B_expcont q=60 n_S1_pass', 'd')))
L.append('| phase wall time ratio to `B_base`, incl. table build (both q pooled, 20 runs) | %s | %s |' % (
    N(row_first(s1cost, arm='B_expcont').phase_wall_ratio_vs_base, 'R1-22', 'stage1_cost.csv B_expcont phase_wall_ratio_vs_base'), 'same'))
L.append('| D1 flags in phase B (M1 / M2, 20 runs pooled) | %s / %s | %s / %s |' % (
    row_first(d1arms, group='stage1/B_expcont').M1_outcome, row_first(d1arms, group='stage1/B_expcont').M2_outcome,
    'same', 'same'))
L.append('')
L.append('Source: R1-18 (`stage1_paired.csv`, rows `arm` = `B_expcont`: `mean`, `ci_mean_lo`, `ci_mean_hi`, '
         '`median`, `ci_median_lo`, `ci_median_hi`, `n_better`, for metrics `stage1_rel_err_abs`, '
         '`stage1_rel_err_signed`, `within_run_sd_e1_last5`); R1-02 (`stage1_criterion.csv`: `a_q50`, '
         '`a_q60`, `b_status`, `overall`); R1-03 (`stage1_dispersion.csv`, metric `stage1_rel_err_signed`: '
         '`sd_arm`, `sd_base`, `ratio`, `ci_lo`, `ci_hi`); R1-04 (`stage1_adv_ratio.csv`: `mean_ratio`); '
         'R1-24 (`stage1_gate_counts.csv`: `n_target0929_pass`, `n_S1_pass`); R1-22 (`stage1_cost.csv`: '
         '`phase_wall_ratio_vs_base`); R1-30 (`decision_d1_flags_arms.csv`, group `stage1/B_expcont`). The same '
         'values are in R1-01 (`decision_inputs.csv`) and in RR-02, section 1 and the observations. The R1 '
         'report states a table build of 7 to 8 s per run inside the wall ratio [RR-02] (report only).\n')
# step 3
tab = pl05
L.append('3. **Check (ii) of the continuation table** (section 3.2 has the full account). The literal '
         'pre-registered criterion, a table-to-verifier difference of at most 1e-6 DW on the verifier\'s '
         'standard final tier, was not met in R1: %s DW at q = 50 and %s DW at q = 60; with the '
         'verifier refined (state step 0.25, 64 Gauss-Legendre nodes) the gap is %s and %s DW [R1-32]. The PI '
         'decided to keep method 6 in the round and record both results [RR-02, section 3; RR-08, section 4 item 6]. '
         'v2.0 replaced the literal criterion by tests (ii-a) self-convergence (%s and %s DW, limit 1e-8), '
         '(ii-b) refined-verifier agreement (%s and %s DW, limit 1e-6) and (ii-c) training-side '
         'sensitivity (largest shift of the stage-1 optimum %s of e1*, limit 1e-3), all passing [PL-05].\n'
         % (N(row_first(chk2, tier='final', q=50).max_abs_diff_over_dw, 'R1-32', 'decision_method6_check_ii.csv final q=50 max_abs_diff_over_dw'),
            N(row_first(chk2, tier='final', q=60).max_abs_diff_over_dw, 'R1-32', 'decision_method6_check_ii.csv final q=60 max_abs_diff_over_dw'),
            N(row_first(chk2, tier='final', q=50).iloc[6] if False else float(chk2[(chk2.tier == 'final') & (chk2.q == 50)].iloc[0, 6]), 'R1-32', 'decision_method6_check_ii.csv final q=50 refined_verifier column'),
            N(float(chk2[(chk2.tier == 'final') & (chk2.q == 60)].iloc[0, 6]), 'R1-32', 'decision_method6_check_ii.csv final q=60 refined_verifier column'),
            N(tab['ii_a']['q50']['max_abs_change_over_dw'], 'PL-05', 'continuation_check_v2_0.json ii_a.q50.max_abs_change_over_dw'),
            N(tab['ii_a']['q60']['max_abs_change_over_dw'], 'PL-05', 'continuation_check_v2_0.json ii_a.q60.max_abs_change_over_dw'),
            N(tab['ii_b']['q50']['max_abs_diff_over_dw'], 'PL-05', 'continuation_check_v2_0.json ii_b.q50.max_abs_diff_over_dw'),
            N(tab['ii_b']['q60']['max_abs_diff_over_dw'], 'PL-05', 'continuation_check_v2_0.json ii_b.q60.max_abs_diff_over_dw'),
            N(max(tab['ii_c']['q50']['fixed_opponent']['shift_over_e1_star'], tab['ii_c']['q50']['opponent_at_e1_star']['shift_over_e1_star'],
                  tab['ii_c']['q60']['fixed_opponent']['shift_over_e1_star'], tab['ii_c']['q60']['opponent_at_e1_star']['shift_over_e1_star']),
              'PL-05', 'continuation_check_v2_0.json ii_c.*.*.shift_over_e1_star, maximum over q and opponent setting (computed here)')))
assert all(tab[k][q]['pass'] for k in ('ii_a', 'ii_b', 'ii_c') for q in ('q50', 'q60'))
# step 4
L.append('4. **The PI decision after R1.** Method 6 enters protocol v2.0, and no other R1 arm does; '
         'target-KL is not adopted because it changes cost, not accuracy [RR-01, addendum item 2; PL-01 '
         '`change_log`, entry "version 2.0", decided by "PI, after the R1 round and before any confirmation '
         'seed was run"].\n')
# step 5
L.append('5. **Re-rehearsal on the development seeds.** The v2.0 entry point, run on the ten development '
         'seeds at each q, reproduces the pilot Phase B bit-identically: check R2, "Phase B equals the R1 pilot '
         '`B_expcont`", %s of %s identical, including the gate-metric values and the table rebuilt from the pilot\'s '
         'frozen snapshot [CF-08 `R2`; RR-03, section 3, row R2 of the checks table]. The re-rehearsal passes G-S in %s of %s runs '
         '(`R3`; the pre-registered condition was at least %s) [CF-08].\n'
         % (N(cf08['R2']['n_identical'], 'CF-08', 'rehearsal_v2_0_checks.json R2.n_identical', 'd'),
            N(cf08['R2']['n'], 'CF-08', 'rehearsal_v2_0_checks.json R2.n', 'd'),
            N(cf08['R3']['R3b']['pooled_n_G-S_pass'], 'CF-08', 'rehearsal_v2_0_checks.json R3.R3b.pooled_n_G-S_pass', 'd'),
            N(20, 'CF-08', 'rehearsal_v2_0_checks.json R3.n', 'd'),
            N(cf08['R3']['R3b']['needed'], 'CF-08', 'rehearsal_v2_0_checks.json R3.R3b.needed', 'd')))
assert cf08['R3']['R3b']['pass'] and cf08['R3']['n'] == 20

# step 6 table
L.append('6. **Fresh-seed confirmation** (seeds 30501-30520, 20 runs per q; sections 3.5 and 3.6 give '
         'the verdict and the failed run). Verdict "%s". Run pass counts: %s of %s at q = 50 and %s of %s at '
         'q = 60 [CF-02]. Table 5 gives the stage-1 error on the fresh seeds under v1.1 (seeds 20501-20520) '
         'and v2.0 (seeds 30501-30520). The medians and maxima are computed here from the `s1` column of '
         'the per-run tables (completed runs), as in section 3.6; the SD of the signed error is read from '
         'the S1 summary tables. Figure: `figures/FG-01_stage1_error_v1_1_vs_v2_0.png`.\n'
         % (cf01['overall'],
            N(runpass[50][0], 'CF-02', 'pass_counts.csv q=50 n_pass', 'd'), N(runpass[50][1], 'CF-02', 'pass_counts.csv q=50 n_expected', 'd'),
            N(runpass[60][0], 'CF-02', 'pass_counts.csv q=60 n_pass', 'd'), N(runpass[60][1], 'CF-02', 'pass_counts.csv q=60 n_expected', 'd')))
L.append('**Table 5. Stage-1 error |e1_hat(0) - e1*| / e1* on fresh seeds.**\n')
L.append('| q | protocol (seeds) | n | median | max | SD of signed error | runs <= 0.05 | runs <= 0.10 |')
L.append('|---|---|---|---|---|---|---|---|')
for q in (50, 60):
    for pr, seeds, it, perf, sdd in [('v1.1', '20501-20520', 'CF-10', 'confirmation_analysis/per_run.csv', sd11),
                                     ('v2.0', '30501-30520', 'CF-03', 'confirmation_v2_0_analysis/per_run.csv', sd20)]:
        n, med, mx, w05, w10 = fs[(pr, q)]
        sd_item, sd_loc = (('CF-11', 'confirmation_analysis/s1_summary.csv q=%d sd_signed' % q) if pr == 'v1.1'
                           else ('CF-04', 'confirmation_v2_0_analysis/s1_summary.csv q=%d criterion=G-S sd_signed' % q))
        L.append('| %d | %s (%s) | %s | %s | %s | %s | %s | %s |' % (
            q, pr, seeds, N(n, it, perf + ' q=%d count of completed runs' % q, 'd'),
            N(med, it, perf + ' q=%d median of s1 (computed here)' % q),
            N(mx, it, perf + ' q=%d max of s1 (computed here)' % q),
            N(sdd[q], sd_item, sd_loc),
            N(w05, it, perf + ' q=%d count s1<=0.05 (computed here)' % q, 'd'),
            N(w10, it, perf + ' q=%d count s1<=0.10 (computed here)' % q, 'd')))
L.append('')
L.append('Source: CF-10 and CF-03 (`per_run.csv`, column `s1`, `status` = completed; median, maximum and counts '
         'computed here); CF-11 and CF-04 (`s1_summary.csv`, column `sd_signed`; the v2.0 file has one row per '
         'criterion, G-S and S1 carry the same SD); the v2.0 G-S pass counts equal CF-04 `n_pass` and CF-02 '
         '`n_G-S_pass`. Different seeds, so the two protocols are unpaired.\n')
assert (gs_pass[50], gs_pass[60]) == (20, 20)
assert fs[('v2.0', 50)][3] == 20 and fs[('v2.0', 60)][3] == 20
# step 7
sg50 = cellp('stage1_rel_err_signed', 50)
L.append('7. **What the chain shows and does not show (observations).** The R1 improvement is a reduction of '
         'spread, as R1 states it [RR-02, observations]: the paired signed difference contains 0 at both q '
         '(Table 4) while the across-seed SD of the signed error is %s and %s times the baseline\'s. '
         'On the fresh seeds the v2.0 median and maximum of '
         'the S1 error are below the v1.1 values at both q (Table 5). The fresh-seed block is a different '
         'seed block from the pilot (30501-30520 against 10501-10510) and from the v1.1 confirmation '
         '(20501-20520), so the comparison with v1.1 is between seed blocks, not paired, and the '
         'chain contains no paired fresh-seed comparison of v1.1 and v2.0. The change is made in Phase B only (in the re-rehearsal, '
         'check R1 "R1 Phase A equals `rehearsal_v1_1`" passes [CF-08 `R1`; RR-03, section 3]); the one failed confirmation run '
         '(q = 50, seed 30510) has `outcome` = stage2_failure, `G-A_pass` = False and `G-S_pass` = True '
         '[CF-03] (section 3.5; section 6).\n'
         % (N(disp(50).ratio, 'R1-03', 'stage1_dispersion.csv q=50 ratio'), N(disp(60).ratio, 'R1-03', 'stage1_dispersion.csv q=60 ratio')))
for q in (50, 60):
    assert fs[('v2.0', q)][1] < fs[('v1.1', q)][1] and fs[('v2.0', q)][2] < fs[('v1.1', q)][2]
L.append('**Decision (recorded).** Protocol v2.0 adopts method 6: "Phase B runs with '
         'continuation_value_mode=expected" [PL-01 `change_log`, version 2.0; RR-01 addendum item 2]. No '
         'other R1 method is in the locked protocol.\n')

SEC46 = '\n'.join(L)

# check that the failed run failed a stage-2 gate and is q50 seed 30510
fr = cf03[(cf03.q == 50) & (cf03.seed == 30510)].iloc[0]
print('failed run 30510: G-A_pass', fr['G-A_pass'], 'G-S_pass', fr['G-S_pass'], 'eta_final', fr['eta_final'], 'outcome', fr['outcome'])
print('table_build_sec range CF-03:', cf03.table_build_sec.min(), cf03.table_build_sec.max())

# hand-typed constants (definitions, thresholds, seed labels, quoted report numbers)
MANUAL = [
    ('4.5', '512', 512, 'RR-08', 'section 4 item 5 / R2B-11 phase_episodes 102400 / 200 updates = 512 rows (report only)'),
    ('4.5', '10 epochs', 10, 'RR-09', 'section 4.3 and section 3.2: pathwise_epochs = 10 (report only)'),
    ('4.5', '256', 256, 'RR-09', 'section 3.2: pathwise_minibatch = 256 (report only)'),
    ('4.5', '20 exact-gradient steps per update', 20, 'RR-09', 'section 4.3: 512/256 = 2 steps per pass x 10 = 20 (report only)'),
    ('4.5', '10 epochs x 2 minibatches', 20, 'RR-09', 'section 4.3: PPO update, minibatch stream position equal to E = 10 (report only)'),
    ('4.5', 'clip 0.5', 0.5, 'RR-08', 'section 4 item 5: clip 0.5 (report only)'),
    ('4.5', '0.05', 0.05, 'PL-01', 'gates.G-A rmse_pos_over_g2_0 limit (RMSE_pos/e2*(0) <= 0.05)'),
    ('4.5', '0.005', 0.005, 'PL-01', 'gates.G-A eta_2/DW limit'),
    ('4.5', '20261003', 20261003, 'RR-02', 'bootstrap seed of the R1 round (default_rng(20261003))'),
    ('4.5', '20261004', 20261004, 'RR-05', 'bootstrap seed of the R2b round (default_rng(20261004))'),
    ('4.5', '2026-10-03', 20261003, 'RR-05', 'text: the R1 comparators ran on 2026-10-03 (report only)'),
    ('4.5', 'up to 40 processes', 40, 'RR-05', 'text: wave A and wave P ran together, up to 40 processes (report only)'),
    ('4.5', '17 s against 27-28 s', 17, 'RR-04', 'quoted from the R2b summary (report only)'),
    ('4.5', '1800', 1800, 'R2B-05', 'trajectory_per_run.csv update column'),
    ('4.5', '1600', 1600, 'R2B-05', 'trajectory_per_run.csv update column (parent)'),
    ('4.5', '200 updates', 200, 'R2B-11', 'cost_P.csv mean_phase_local_updates'),
    ('4.5', '10 development seeds', 10, 'PL-01', 'development_seeds (10501-10510)'),
    ('4.5', 'seed 10510 (eta_2/DW 0.005501 > 0.005)', 10510, 'RR-05', 'row A_ctrl200_lr3e-4 vs parent_u1600, part (b) (report only); criterion.csv b_violations = q50/10510'),
    ('4.5', '5.2e-4', 0.00052, 'RR-04', 'quoted from the R2b summary FOC sentence (report only)'),
    ('4.5', '3.9e-4', 0.00039, 'RR-04', 'quoted from the R2b summary FOC sentence (report only)'),
    ('4.5', '4.3e-4', 0.00043, 'RR-04', 'quoted from the R2b summary FOC sentence (report only)'),
    ('4.5', '4.4e-4', 0.00044, 'RR-04', 'quoted from the R2b summary FOC sentence (report only)'),
    ('4.5', '4.9e-4', 0.00049, 'RR-04', 'quoted from the R2b summary FOC sentence (report only)'),
    ('4.5', '6.6e-4', 0.00066, 'RR-04', 'quoted from the R2b summary FOC sentence (report only)'),
    ('4.6', '1e-6 DW', 1e-6, 'R1-32', 'decision_method6_check_ii.csv spec_tolerance_over_dw'),
    ('4.6', '1e-8', 1e-8, 'PL-05', 'continuation_check_v2_0.json ii_a.q50.threshold_over_dw'),
    ('4.6', '1e-3', 1e-3, 'PL-05', 'continuation_check_v2_0.json ii_c.q50.threshold_over_e1_star'),
    ('4.6', 'state step 0.25', 0.25, 'PL-05', 'continuation_check_v2_0.json ii_b.q50.verifier.state_step'),
    ('4.6', '64 Gauss-Legendre nodes', 64, 'PL-05', 'continuation_check_v2_0.json ii_b.q50.verifier.gl_half'),
    ('4.6', '0.05', 0.05, 'PL-01', 'gates.G-S threshold'),
    ('4.6', '0.10', 0.10, 'PL-01', 'secondary.S1 threshold'),
    ('4.6', '7 to 8 s', 7, 'RR-02', 'section 1 observations, table build (report only); CF-03 table_build_sec spans 7.726 to 10.04 s for the v2.0 confirmation runs'),
    ('4.6', 'seeds 30501-30520', 30501, 'CF-01', 'verdict.json seeds'),
    ('4.6', 'seeds 20501-20520', 20501, 'CF-10', 'confirmation_analysis/per_run.csv seed range'),
    ('4.6', '10501-10510', 10501, 'PL-01', 'development_seeds'),
    ('4.6', '30510', 30510, 'CF-03', 'per_run.csv q=50 seed=30510 outcome=stage2_failure'),
    ('4.6', '20 runs', 20, 'R1-22', 'stage1_cost.csv B_expcont n_runs'),
    ('4.6', 'seven links', 7, 'RR-01', 'count of links in this list (author)'),
    ('4.6', '20261003', 20261003, 'RR-02', 'bootstrap seed of the R1 round'),
    ('4.6', 'six R1 candidates', 6, 'RR-01', 'round scope: six candidate changes (report only)'),
]
for sec, txt, val, it, loc in MANUAL:
    LEDGER.append({'statement_id': 'S%04d' % (len(LEDGER) + 1), 'section': sec, 'text': txt,
                   'value': repr(float(val)), 'item_id': it, 'locator': loc})

with open(os.path.join(OUT, 'sec04b.md'), 'w') as f:
    f.write(SEC45 + '\n\n' + SEC46 + '\n')
with open(os.path.join(OUT, 'sec04b_ledger.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['statement_id', 'section', 'text', 'value', 'item_id', 'locator'])
    w.writeheader()
    w.writerows(LEDGER)
print('ledger rows', len(LEDGER))
