"""Build section 6 (the stage-2 peak) of 100526report.md from the evidence pack.

Writes SCRATCH/sec06.md and SCRATCH/sec06_ledger.csv. Every number in the text is produced by
this script from a pack table (or from a pack JSON / report quote, marked in the ledger).
Run: OMP_NUM_THREADS=1 python build_sec06.py
"""
import re
import sys

sys.path.insert(0, '/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-'
                   'r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/'
                   'scratchpad/report_parts/W-E')
from common import *  # noqa: F401,F403  (helper module of this script only)

OPEN_ISSUES = []     # filled while building; printed at the end


def one(df, **flt):
    sub = df
    for k, v in flt.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, (flt, len(sub))
    return sub.iloc[0]


def fl(fname, col, **flt):
    return '%s: %s; col %s' % (fname, ','.join('%s=%s' % kv for kv in flt.items()), col)


def met(b):
    return 'met' if bool(b) else 'not met'


# ------------------------------------------------------------------ data
tail_b = rd('R2B-04')
arm_b = rd('R2B-13')
pr_b = rd('R2B-03')
crit_b = rd('R2B-02')
pair_b = rd('R2B-08')
r1crit = rd('R1-06')
r1arm = rd('R1-40')
r1cost = pd.read_csv(EV / 'results/v2_refine/analysis/stage2_cost.csv')   # R1-23
r2b_dec = rd('R2B-01')
spec_b = rd('R2B-29')
conf = rd('R2B-18')
crit_c = rd('R2C-02')
tail_c = rd('R2C-04')
sel_c = rd('R2C-08')
resp_c = rd('R2C-09')
spec_c = rd('R2C-12')
pair_c = rd('R2C-07')
sel_json = rj('R2C-03')
chk_json = rj('R2C-06')
num = rj('R2B-15')['values']
hyp = rd('R2B-16')
dec50 = rd('R2B-17')
late = rd('R2B-24')
eta_exp = rd('R2B-21')
clamp = rd('R2B-27')

F_TAIL_B = 'results/v2_refine_r2b/analysis/tail.csv'
F_ARM_B = 'results/v2_refine_r2b/analysis/arm_summary.csv'
F_PR_B = 'results/v2_refine_r2b/analysis/per_run.csv'
F_CRIT_B = 'results/v2_refine_r2b/analysis/criterion.csv'
F_PAIR_B = 'results/v2_refine_r2b/analysis/paired.csv'
F_CRIT_C = 'results/v2_refine_r2c/analysis/criterion.csv'
F_TAIL_C = 'results/v2_refine_r2c/analysis/tail.csv'
F_SEL_C = 'results/v2_refine_r2c/analysis/selection_inputs.csv'
F_SPEC_C = 'results/v2_refine_r2c/analysis/waveS_specifics_mean.csv'
F_RESP_C = 'results/v2_refine_r2c/analysis/waveS_response.csv'
F_R1C = 'results/v2_refine/analysis/stage2_criterion.csv'

out = []   # markdown pieces


def emit(s=''):
    out.append(s)


# ------------------------------------------------------------------ design arithmetic (RR-10 s.7)
BINS = {50: 40, 60: 44}      # bins of width 10 on D_2 (RR-10 section 7)
N_PEAK, N_TAIL = 4, 20       # peak set = 4 bins; bins with |d| >= 2q = 20 at both q


def tail_bin_share(s, q):
    if s == 0:
        return N_TAIL / BINS[q]
    return (1 - s) * N_TAIL / (BINS[q] - N_PEAK)


# values quoted in RR-10 section 7 (to 4 decimals); the script asserts that the recomputation agrees
RR10 = {0: (0.5000, 0.4545), 0.25: (0.4167, 0.3750), 0.35: (0.3611, 0.3250),
        0.40: (0.3333, 0.3000), 0.50: (0.2778, 0.2500)}
for s, (a, b) in RR10.items():
    assert round(tail_bin_share(s, 50), 4) == a and round(tail_bin_share(s, 60), 4) == b, s

# =================================================================== 6 header and framing
LED.section = '6.0'
emit('## 6. The stage-2 peak')
emit()
dw_note = C('eta_2/DW <= 0.005', 'PL-02', 'protocols/v2_T2_locked_v2_0.md section 1, gate G-A (same in PL-01)')
emit('The stage-2 peak error is the signed relative error of the learned stage-2 effort at the '
     'state d = 0 against the closed form, (e2_hat(0) - e2*(0)) / e2*(0), taken from the stage-2 '
     'last iterate at the end of Phase A (global update 1600, final verifier tier); negative means '
     'the learned effort is below the closed form. Its absolute value is the primary metric of '
     'the stage-2 arms in R1, R2b and R2c (`stage2_peak_rel_err_abs`). It is not a component of a '
     'v2.0 gate: G-A consists of %s, RMSE_pos / e2*(0) <= 0.05 and a tail mean / e2*(0) <= %s '
     '(tail mean = mean of e2_hat over the recovery-grid nodes with |d| >= 2q) [PL-02, section 1]. The count '
     '"runs with |peak error| <= %s" in the R2b and R2c tables is a descriptive threshold of those '
     'reports.' % (dw_note, C('0.02', 'PL-02', 'protocols/v2_T2_locked_v2_0.md section 1, G-A tail-mean limit (same in PL-01)'),
                   C('0.05', 'RR-10', 'section 7 and R2B-04 column n_abs_peak_le_0.05')))
emit()
emit('**The PI\'s reading.** The PI summarises the remaining peak gap as a cusp-representation / '
     'weighting limit. The wording recorded in the R2c pre-registration is: "the remaining peak '
     'gap is a weighting problem (payoff flatness at the cusp, 10% of the samples)", stated by '
     'the owner as the R2b reading [RR-10, section 7]. This section reports what was measured '
     'for and against that reading. The reading is the PI\'s; none of the measurements below '
     'establishes it as the cause, and for each piece of evidence the text says what it shows '
     'and what it leaves open. I found no experiment in the round reports (RR-04, RR-05, RR-06, '
     'RR-10) that was designed to separate a weighting explanation from the alternatives for the '
     'whole peak gap; the experiments the reports do name for their own open questions are in '
     'sections 6.4 and 6.5.')
emit()
emit('**Evidence map.** The numbers are in the subsections named.')
emit()
emit('| evidence | what was measured | what it leaves open |')
emit('|---|---|---|')
emit('| pathwise fine-tuning, annealing, polishing, batch, target-KL, censored likelihood '
     '(6.2) | none of these arms has a 95% interval of the mean paired difference below 0 at '
     'both q | each was tested at one setting with 10 seeds per q; an interval containing 0 '
     'does not show that a mechanism has no effect |')
emit('| peak-focused exploring starts (6.3, 6.5) | raising the share of stage-2 starts in the '
     'peak set moves the point estimate of the mean peak error in the improving direction at both '
     'q for every share from update 1; the interval excludes 0 at q = 50 for all but the smallest '
     'share and at q = 60 only for the largest | the arms also change '
     'the share of starts in the tail bins, the clamp exposure and the tail mean, so the share is '
     'not varied in isolation; whether the response is a trade-off or noise at 10 seeds is not '
     'decided [RR-06] |')
emit('| q-asymmetry (6.1) | in the development baseline the q = 50 error is larger than the '
     'q = 60 error | the fresh-seed confirmation runs do not show the same ordering |')
emit('| tail trade-off (6.3) | the largest share from update 1 crosses the G-A tail limit in some '
     'q = 60 runs | no pilot arm tested a shape of the start distribution other than a share '
     'on the four peak bins |')
emit('| seed 30510 (6.4) | one run with a peak error far outside the others | one run; none of the '
     'pre-registered hypotheses H1-H4 is supported; the cause is not decided |')
emit('| R2c (6.5) | no arm selected under the pre-registered rule | whether any share or timing '
     'would meet the criterion with more seeds or another rule was not tested |')
emit()

# =================================================================== 6.1
LED.section = '6.1'
emit('### 6.1 What the baseline peak error looks like')
emit()
base_rows = {}
for q in (50, 60):
    t = one(tail_b, arm='A_base', q=q)
    a = one(arm_b, arm='A_base', q=q)
    r1 = one(r1arm, arm='A_base', q=q)
    assert abs(a.median_stage2_peak_rel_err_signed - r1.median_stage2_peak_rel_err_signed) < 1e-12
    n_neg = int((pr_b[(pr_b.arm == 'A_base') & (pr_b.q == q)].stage2_peak_rel_err_signed < 0).sum())
    g2 = float(pr_b[(pr_b.arm == 'A_base') & (pr_b.q == q)].g2_at_0.iloc[0])
    base_rows[q] = dict(t=t, a=a, n_neg=n_neg, g2=g2)

med50 = N(base_rows[50]['a'].median_stage2_peak_rel_err_signed, 'R2B-13',
          fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm='A_base', q=50), '.4f')
med60 = N(base_rows[60]['a'].median_stage2_peak_rel_err_signed, 'R2B-13',
          fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm='A_base', q=60), '.4f')
n_neg_all = base_rows[50]['n_neg'] + base_rows[60]['n_neg']
emit('**Outcome.** The locked baseline (`A_base`, which is R1\'s `parents_A`: the Phase-A end state '
     'of the 20 development-seed runs, seeds 10501-10510 at each q) has a stage-2 peak below the '
     'closed form in every run, and only %s of the 20 runs come within 0.05 although all of them '
     'pass the stage-2 gate components (table 6.1a). The signed error is negative in %s of 20 runs '
     '(computed here from R2B-03, column `stage2_peak_rel_err_signed`); the medians are %s (q = 50) '
     'and %s (q = 60) [R2B-13; the same numbers as the R1 stage-2 arm summary, R1-40, and as the '
     'sentence "The baseline medians of the signed peak error are -0.0625 (q = 50) and -0.0497 '
     '(q = 60)" in the "Observations" of the R1 decision-inputs report, RR-02].'
     % (N(int(base_rows[50]['t']['n_abs_peak_le_0.05'] + base_rows[60]['t']['n_abs_peak_le_0.05']),
          'R2B-04', 'computed here: sum over q of n_abs_peak_le_0.05, A_base'),
        N(n_neg_all, 'R2B-03', fl(F_PR_B, 'stage2_peak_rel_err_signed<0 count, computed here',
                                  arm='A_base')), med50, med60))
emit()
emit('**Table 6.1a. Baseline (`A_base`) peak error by q, development seeds, 10 runs per q.**')
emit()
emit('| q | e2*(0) | mean e2_hat(0) | median signed peak error | mean \\|peak error\\| | runs with '
     '\\|peak error\\| <= 0.05 | max \\|peak error\\| | runs passing G-A and G-N | median share of the '
     'd = 0 gap predicted by the policy\'s own noise |')
emit('|---|---|---|---|---|---|---|---|---|')
for q in (50, 60):
    b = base_rows[q]
    t, a = b['t'], b['a']
    emit('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        C(str(q), 'R2B-04', 'q'),
        N(b['g2'], 'R2B-03', fl(F_PR_B, 'g2_at_0', arm='A_base', q=q), '.4g'),
        N(a.mean_e2_at_0, 'R2B-13', fl(F_ARM_B, 'mean_e2_at_0', arm='A_base', q=q)),
        N(a.median_stage2_peak_rel_err_signed, 'R2B-13',
          fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm='A_base', q=q)),
        N(t.mean_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'mean_abs_peak_error', arm='A_base', q=q)),
        N(int(t['n_abs_peak_le_0.05']), 'R2B-04',
          fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm='A_base', q=q)) + ' of 10',
        N(t.max_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'max_abs_peak_error', arm='A_base', q=q)),
        N(int(t.n_gate_pass), 'R2B-04', fl(F_TAIL_B, 'n_gate_pass', arm='A_base', q=q)) + ' of 10',
        N(a.median_smoothed_share_peak_gap_d0, 'R2B-13',
          fl(F_ARM_B, 'median_smoothed_share_peak_gap_d0', arm='A_base', q=q))))
emit()
emit('Source: R2B-04 (`results/v2_refine_r2b/analysis/tail.csv`, rows `A_base`; columns '
     '`mean_abs_peak_error`, `n_abs_peak_le_0.05`, `max_abs_peak_error`, `n_gate_pass`), R2B-13 '
     '(`arm_summary.csv`, rows `A_base`; columns `mean_e2_at_0`, `median_stage2_peak_rel_err_signed`, '
     '`median_smoothed_share_peak_gap_d0`), R2B-03 (`per_run.csv`, column `g2_at_0` = e2*(0)). '
     'The R2C-04 table repeats the `A_base` rows of R2B-04 unchanged. "Share of the gap predicted '
     'by the policy\'s own noise" is the repository\'s smoothed-game decomposition of the d = 0 gap '
     '[RR-11, section 6]; it is a measured quantity and is not a statement about cause.')
emit()

dq = N(base_rows[50]['t'].mean_abs_peak_error - base_rows[60]['t'].mean_abs_peak_error,
       'R2B-04', 'computed here: mean_abs_peak_error(q=50) - mean_abs_peak_error(q=60), A_base')
emit('**q-asymmetry in the baseline.** The q = 50 baseline is worse than the q = 60 baseline on '
     'every statistic in the table: mean |peak error| %s against %s (difference %s, computed here), '
     'median signed error %s against %s, maximum %s against %s, and %s against %s runs within 0.05 '
     '[R2B-04, R2B-13]. Observation: the peak-set share of the bin-balanced start draws is '
     'the larger at q = 50 (4 of 40 bins, %s) than at q = 60 (4 of 44 bins, %s) [RR-10, section 7; '
     'measured visitation %s and %s in the baseline runs, R2C-12]; the larger error occurs with the '
     'larger peak-set share.' % (
         N(base_rows[50]['t'].mean_abs_peak_error, 'R2B-04',
           fl(F_TAIL_B, 'mean_abs_peak_error', arm='A_base', q=50)),
         N(base_rows[60]['t'].mean_abs_peak_error, 'R2B-04',
           fl(F_TAIL_B, 'mean_abs_peak_error', arm='A_base', q=60)), dq,
         N(base_rows[50]['a'].median_stage2_peak_rel_err_signed, 'R2B-13',
           fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm='A_base', q=50)),
         N(base_rows[60]['a'].median_stage2_peak_rel_err_signed, 'R2B-13',
           fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm='A_base', q=60)),
         N(base_rows[50]['t'].max_abs_peak_error, 'R2B-04',
           fl(F_TAIL_B, 'max_abs_peak_error', arm='A_base', q=50)),
         N(base_rows[60]['t'].max_abs_peak_error, 'R2B-04',
           fl(F_TAIL_B, 'max_abs_peak_error', arm='A_base', q=60)),
         N(int(base_rows[50]['t']['n_abs_peak_le_0.05']), 'R2B-04',
           fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm='A_base', q=50)),
         N(int(base_rows[60]['t']['n_abs_peak_le_0.05']), 'R2B-04',
           fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm='A_base', q=60)),
         N(4 / 40, 'RR-10', 'section 7: 4 peak bins of 40 (computed here)', '.4f'),
         N(4 / 44, 'RR-10', 'section 7: 4 peak bins of 44 (computed here)', '.4f'),
         N(one(spec_c, arm='A_base', q=50).peak_visit_share, 'R2C-12',
           fl(F_SPEC_C, 'peak_visit_share', arm='A_base', q=50)),
         N(one(spec_c, arm='A_base', q=60).peak_visit_share, 'R2C-12',
           fl(F_SPEC_C, 'peak_visit_share', arm='A_base', q=60))))
emit()

emit('**Table 6.1b. The same statistics in the 20 fresh-seed v2.0 confirmation runs per q '
     '(seeds 30501-30520; solver v2.0; not paired with the baseline above).**')
emit()
emit('| q | median signed peak error | mean \\|peak error\\| | runs with \\|peak error\\| <= 0.05 | '
     'most negative signed error | runs with a negative signed error |')
emit('|---|---|---|---|---|---|')
conf_stats = {}
for q in (50, 60):
    s = conf[conf.q == q].peak_rel_err
    conf_stats[q] = dict(med=float(s.median()), mean_abs=float(s.abs().mean()),
                         n_le=int((s.abs() <= 0.05).sum()), mn=float(s.min()), n_neg=int((s < 0).sum()),
                         n=len(s))
    cs = conf_stats[q]
    loc = 'tab_decomposition_all_runs.csv: q=%d, col peak_rel_err, %%s over %d runs (computed here)' % (q, cs['n'])
    emit('| %s | %s | %s | %s of %s | %s | %s of %s |' % (
        q, N(cs['med'], 'R2B-18', loc % 'median'), N(cs['mean_abs'], 'R2B-18', loc % 'mean |.|'),
        N(cs['n_le'], 'R2B-18', loc % 'count |.|<=0.05'), N(cs['n'], 'R2B-18', loc % 'n'),
        N(cs['mn'], 'R2B-18', loc % 'min'), N(cs['n_neg'], 'R2B-18', loc % 'count <0'),
        N(cs['n'], 'R2B-18', loc % 'n')))
emit()
emit('Source: R2B-18 (`results/v2_refine_r2b/diag_30510/tables/tab_decomposition_all_runs.csv`, column '
     '`peak_rel_err`, the signed peak error of each confirmation run); median, mean, count and '
     'minimum computed here by `build_sec06.py`. The minimum at q = 50 is seed 30510 (section 6.4).')
emit()
assert conf_stats[50]['med'] > conf_stats[60]['med']       # q=60 median is more negative
emit('Observation (computed here): the ordering of the development baseline is not present in the '
     'confirmation runs. There the q = 60 median error (%s) is the more negative one and %s of 20 runs '
     'lie within 0.05 at q = 50 against %s of 20 at q = 60 (table 6.1b). No test of the asymmetry is '
     'recorded in the pack; with 10 and 20 runs per q I treat it as an observation of the development '
     'seeds only.' % (
         N(conf_stats[60]['med'], 'R2B-18', 'tab_decomposition_all_runs.csv: q=60 median peak_rel_err'),
         N(conf_stats[50]['n_le'], 'R2B-18', 'count |peak|<=0.05, q=50'),
         N(conf_stats[60]['n_le'], 'R2B-18', 'count |peak|<=0.05, q=60')))
dec50_ = rd('R2B-17')
floor_med = float(dec50_.floor_median.iloc[0])
assert (dec50_.floor_median == floor_med).all()
emit()
emit('**Representation floor.** The pack holds one measurement on whether the actor class can represent the '
     'closed-form peak: fitting the same actor class to e2* by least squares (5 initialisations, 300,000 '
     'steps; the plateau rule never fired, so the values are upper bounds) leaves a d = 0 error with median '
     '%s effort units [R2B-17, column `floor_median`; RR-11, section 6 for the fit settings]. It comes from '
     'pilot 4 and was not recomputed here; it is not a fit of any run of R1, R2b or R2c. What it shows: a '
     'supervised fit of this class reaches the d = 0 height to about 1e-3 effort units, while the d = 0 gaps '
     'of the 20 confirmation runs at q = 50 range from %s to %s effort units (median %s) [R2B-17, column '
     '`rl_gap`, computed here]. What it does not show: whether the policy-gradient objective, with its '
     'sample weighting, reaches that fit.' % (
         N(floor_med, 'R2B-17', 'tab_decomposition_q50.csv: col floor_median (all rows equal)'),
         N(float(dec50_.rl_gap.min()), 'R2B-17', 'min of col rl_gap over the 20 q=50 runs (computed here)'),
         N(float(dec50_.rl_gap.max()), 'R2B-17', 'max of col rl_gap over the 20 q=50 runs (computed here)'),
         N(float(dec50_.rl_gap.median()), 'R2B-17', 'median of col rl_gap over the 20 q=50 runs (computed here)')))
OPEN_ISSUES.append('q-asymmetry: in the development baseline (A_base, n=10 per q) the q=50 peak error '
                   'is larger than q=60 (mean |peak| 0.065596 vs 0.053347; 2 vs 5 runs within 0.05), '
                   'but in the 20 fresh-seed v2.0 confirmation runs per q the order is reversed in the '
                   'median (q50 %.4f, q60 %.4f) and the within-0.05 counts are 5 vs 4. The spec lists '
                   'the q-asymmetry as supporting evidence; the pack supports it only for the development '
                   'seeds. Also the peak-set share by design is larger at q=50 (0.1) than q=60 (0.0909), '
                   'the same direction as the larger error, i.e. not the ordering a pure sample-share '
                   'explanation would give.' % (conf_stats[50]['med'], conf_stats[60]['med']))
emit()

# =================================================================== 6.2
LED.section = '6.2'
emit('### 6.2 Mechanisms that did not move the peak')
emit()
emit('**Outcome.** No mechanism other than peak-focused starts has a 95% interval of the mean paired '
     'difference of |peak error| below 0 at both q. Two rows have an interval below 0 at q = 50 only: '
     'the batch variant `A_batch_mb256` and a PPO control with a higher learning rate, which is not a '
     'candidate. The details of methods 1-5 are in sections 4.1-4.5; the table gives the primary paired '
     'difference (arm minus comparator, paired by (q, seed), n = 10 development seeds per q, negative = '
     'arm better) and whether part (a) of the criterion was met at each q.')
emit()
emit('**Table 6.2. Primary paired difference of |peak error| and criterion part (a), by mechanism.**')
emit()
emit('| method | arm | comparator | bootstrap seed | q = 50: mean [95% CI] | q = 60: mean [95% CI] | '
     '(a) at q = 50 / q = 60 | (b) |')
emit('|---|---|---|---|---|---|---|---|')

rows62 = [
    ('1 polish', 'A_polish1', 'A_base', 'R1'), ('1 polish', 'A_polish2', 'A_base', 'R1'),
    ('2 batch', 'A_batch', 'A_base', 'R1'), ('2 batch', 'A_batch_mb256', 'A_base', 'R1'),
    ('3 target-KL', 'A_kl005', 'A_base', 'R1'), ('3 target-KL', 'A_kl010', 'A_base', 'R1'),
    ('4 annealing', 'A_anneal2', 'A_base', 'R1'), ('4 annealing', 'A_anneal4', 'A_base', 'R1'),
    ('5 pathwise, 1 step per update (R1)', 'A_detmean', 'A_base', 'R1'),
    ('5 pathwise, 1 step per update (R1)', 'A_detmean', 'A_ctrl200', 'R1'),
    ('5 pathwise, 20 steps per update (R2b)', 'P20_lr3e-5', 'A_ctrl200', 'R2b'),
    ('5 pathwise, 20 steps per update (R2b)', 'P20_lr3e-4', 'A_ctrl200_lr3e-4', 'R2b'),
    ('(control, LR 3e-4, 200 updates)', 'A_ctrl200_lr3e-4', 'parent_u1600', 'R2b'),
    ('censored likelihood', 'A_censored', 'A_base', 'R2b'),
]
SEED = {'R1': C('20261003', 'RR-02', '06_decision_inputs.md line 3: default_rng(20261003)'),
        'R2b': C('20261004', 'RR-09', '01_preregistration.md section 6: default_rng(20261004)')}
ledger_b_notes = []
for meth, arm, comp, rnd in rows62:
    if rnd == 'R1':
        r = one(r1crit, arm=arm, baseline=comp)
        F, item = F_R1C, 'R1-06'
    else:
        r = one(crit_b, arm=arm, baseline=comp)
        F, item = F_CRIT_B, 'R2B-02'
    cells = []
    for q in (50, 60):
        cells.append(CI(r['mean_q%d' % q], r['ci_mean_lo_q%d' % q], r['ci_mean_hi_q%d' % q], item,
                        fl(F, 'mean/ci q%d' % q, arm=arm, baseline=comp)))
    a_txt = '%s / %s' % (met(r.a_q50), met(r.a_q60))
    b_txt = str(r.b_status)
    if b_txt == 'violated':
        viol = str(r.b_violations)
        b_txt = 'violated (%s)' % viol
    emit('| %s | `%s` | `%s` | %s | %s | %s | %s | %s |' % (meth, arm, comp, SEED[rnd], cells[0],
                                                      cells[1], a_txt, b_txt))
emit()
emit('Source: R1-06 (`results/v2_refine/analysis/stage2_criterion.csv`; columns `mean_q50`, '
     '`ci_mean_lo_q50`, `ci_mean_hi_q50`, the same for q60, `a_q50`, `a_q60`, `b_status`) for the R1 '
     'rows, and R2B-02 (`results/v2_refine_r2b/analysis/criterion.csv`; the same columns and '
     '`b_violations`) for the R2b rows. Bootstrap seed 20261003 (R1) and 20261004 (R2b): 10,000 '
     'resamples of the 10 paired seeds [RR-08 and RR-09, section 6]. Part (a): the interval lies '
     'below 0 at both q.')
emit()
# facts used in prose
a_ctrl = one(crit_b, arm='A_ctrl200_lr3e-4', baseline='parent_u1600')
eta_c = float(one(pr_b, arm='A_ctrl200_lr3e-4', q=50, seed=10510).eta_T_over_dw)
bm = one(r1crit, arm='A_batch_mb256', baseline='A_base')
emit('Reading the table:')
emit()
emit('- **Part (a) is not met by any row.** The interval excludes 0 on the improving side at q = 50 only '
     'for `A_batch_mb256` (%s) and for the control `A_ctrl200_lr3e-4` against its parent (%s), and '
     'in neither case at q = 60 [R1-06, R2B-02]. `P20_lr3e-4` is worse than its control at q = 50 '
     '(%s; the interval excludes 0 on the worsening side) [R2B-02]. The control `A_ctrl200_lr3e-4` '
     'also violates part (b) at q = 50 seed 10510 (eta_2/DW %s against the G-A limit 0.005) '
     '[R2B-03, R2B-02].'
     % (CI(bm.mean_q50, bm.ci_mean_lo_q50, bm.ci_mean_hi_q50, 'R1-06',
           fl(F_R1C, 'q50 mean/ci', arm='A_batch_mb256', baseline='A_base')),
        CI(a_ctrl.mean_q50, a_ctrl.ci_mean_lo_q50, a_ctrl.ci_mean_hi_q50, 'R2B-02',
           fl(F_CRIT_B, 'q50 mean/ci', arm='A_ctrl200_lr3e-4', baseline='parent_u1600')),
        CI(one(crit_b, arm='P20_lr3e-4').mean_q50, one(crit_b, arm='P20_lr3e-4').ci_mean_lo_q50,
           one(crit_b, arm='P20_lr3e-4').ci_mean_hi_q50, 'R2B-02',
           fl(F_CRIT_B, 'q50 mean/ci', arm='P20_lr3e-4')),
        N(eta_c, 'R2B-03', fl(F_PR_B, 'eta_T_over_dw', arm='A_ctrl200_lr3e-4', q=50, seed=10510))))
emit('- **The pathwise arms are read against their matched PPO controls.** R1\'s `A_detmean` takes %s '
     'optimiser steps in its 200 updates against %s for its control `A_ctrl200`; R2b\'s `P20` arms take 20 '
     'exact-gradient steps per update, %s steps, the optimiser budget of their PPO controls [R1-23, R2B-01; '
     'section 4.5]. A higher LR over the same 200 updates is a confound that the pathwise arms '
     'must be read against: the PPO control `A_ctrl200_lr3e-4` alone moves the peak at q = 50 against the '
     'parent candidate (row `A_ctrl200_lr3e-4`), and it is the matched control for `P20_lr3e-4` '
     '[RR-04]. The offline first-order-condition residual of the pathwise arms '
     'falls over the phase while the peak error does not separate from the control (section 4.5 and '
     'figure FG-17 there). What this shows: reducing that residual at the bin centres by exact-gradient '
     'steps did not lower |peak error| relative to PPO at the same optimiser budget. What it leaves open: '
     'one budget (200 updates) and the bin-centre residual only.' % (
         N(float(r1cost[r1cost.arm == 'A_detmean'].mean_n_minibatch_steps_total.iloc[0]), 'R1-23',
           'stage2_cost.csv: arm=A_detmean col mean_n_minibatch_steps_total', '.4g'),
         N(float(r1cost[r1cost.arm == 'A_ctrl200'].mean_n_minibatch_steps_total.iloc[0]), 'R1-23',
           'stage2_cost.csv: arm=A_ctrl200 col mean_n_minibatch_steps_total', '.4g'),
         N(float(one(r2b_dec, arm='P20_lr3e-5')['mean optimiser steps/run']), 'R2B-01',
           'decision_inputs.csv: arm=P20_lr3e-5 col mean optimiser steps/run', '.4g')))
emit('- **Annealing, polishing, batch and target-KL** each have an interval containing 0 at both q, with '
     'the single exception noted above (sections 4.1-4.4). Annealing lowers the policy noise sigma_2(0) '
     'by design; the smoothing-predicted part of the d = 0 gap falls with it, while the observed gap '
     'does not fall in the same direction at every (scale, q) (section 4.4, R1-07).')
emit('- **Censored likelihood** (`A_censored`): both intervals contain 0 (%s at q = 50, %s at q = 60) '
     '[R2B-02].'
     % (CI(*[one(crit_b, arm='A_censored')[k] for k in ('mean_q50', 'ci_mean_lo_q50', 'ci_mean_hi_q50')],
           'R2B-02', fl(F_CRIT_B, 'q50', arm='A_censored')),
        CI(*[one(crit_b, arm='A_censored')[k] for k in ('mean_q60', 'ci_mean_lo_q60', 'ci_mean_hi_q60')],
           'R2B-02', fl(F_CRIT_B, 'q60', arm='A_censored'))))
emit()
emit('**Decisions recorded.** Method 5 (pathwise fine-tuning) is closed as negative at matched budgets, '
     'and the censored likelihood is not adopted (PI publication prompt, D1). Methods 1-4 are reported '
     'in section 4; the decisions recorded for them are given there.')
emit()

# =================================================================== 6.3
LED.section = '6.3'
emit('### 6.3 Peak-focused exploring starts (R2b `A_peak25`, `A_peak50`)')
emit()
emit('**Outcome.** Peak-focused starts at share 0.50 from update 1 are the only R2b arm whose interval '
     'lies below 0 at both q (part (a) met), and they violate part (b) at q = 60 by three runs that '
     'cross the G-A tail-mean limit; share 0.25 does not meet part (a).')
emit()
emit('**Mechanism.** The sampler draws the exploring starts of the final stage in two groups: a share s '
     'from the four bins of width 10 that intersect (-20, 20), uniformly over those bins, and the '
     'remaining share 1 - s uniformly over the other bins; the locked sampler draws bin-balanced '
     '(every bin equally likely) [RR-09, section 4.1]. The arms are `A_peak25` (s = 0.25) and '
     '`A_peak50` (s = 0.50), both from update 1 of Phase A, compared with `A_base` (`parents_A`) paired '
     'by (q, seed), n = 10 development seeds per q, bootstrap seed 20261004 [RR-09].')
emit()
emit('**Table 6.3a. Paired difference of |peak error|, arm minus `A_base`, and criterion verdicts '
     '(R2b, bootstrap seed 20261004).**')
emit()
emit('| arm | q = 50: mean [95% CI] | seeds better, q = 50 | q = 60: mean [95% CI] | seeds better, '
     'q = 60 | (a) at q = 50 / q = 60 | (b) |')
emit('|---|---|---|---|---|---|---|')
for arm in ('A_peak25', 'A_peak50'):
    r = one(crit_b, arm=arm, baseline='A_base')
    nb = {}
    for q in (50, 60):
        pp = one(pair_b, arm=arm, baseline='A_base', q=q, metric='stage2_peak_rel_err_abs')
        nb[q] = N(int(pp.n_better), 'R2B-08', fl(F_PAIR_B, 'n_better', arm=arm, q=q,
                                                 metric='stage2_peak_rel_err_abs'))
    b_txt = 'holds' if r.b_status == 'holds' else (
        'violated (q = 60 seeds 10503, 10504, 10510: tail mean / e2*(0) > 0.02)')
    emit('| `%s` | %s | %s of 10 | %s | %s of 10 | %s / %s | %s |' % (
        arm,
        CI(r.mean_q50, r.ci_mean_lo_q50, r.ci_mean_hi_q50, 'R2B-02', fl(F_CRIT_B, 'q50', arm=arm)),
        nb[50],
        CI(r.mean_q60, r.ci_mean_lo_q60, r.ci_mean_hi_q60, 'R2B-02', fl(F_CRIT_B, 'q60', arm=arm)),
        nb[60], met(r.a_q50), met(r.a_q60), b_txt))
emit()
emit('Source: R2B-02 (`criterion.csv`; columns `mean_q50`, `ci_mean_lo_q50`, `ci_mean_hi_q50`, q60 '
     'likewise, `a_q50`, `a_q60`, `b_status`, `b_violations`), R2B-08 (`paired.csv`, metric '
     '`stage2_peak_rel_err_abs`, column `n_better`). The violating seeds are the entries of '
     '`b_violations` for `A_peak50`.')
emit()
emit('**Table 6.3b. Levels per arm and q (10 runs each).**')
emit()
emit('| arm | q | mean \\|peak error\\| | median signed peak error | runs with \\|peak error\\| <= 0.05 | '
     'max \\|peak error\\| | mean tail mean / e2*(0) | max tail mean / e2*(0) | runs passing G-A and G-N |')
emit('|---|---|---|---|---|---|---|---|---|')
lvl = {}
for arm in ('A_base', 'A_peak25', 'A_peak50'):
    for q in (50, 60):
        t = one(tail_b, arm=arm, q=q)
        a = one(arm_b, arm=arm, q=q)
        lvl[(arm, q)] = (t, a)
        emit('| `%s` | %d | %s | %s | %s of 10 | %s | %s | %s | %s of 10 |' % (
            arm, q,
            N(t.mean_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'mean_abs_peak_error', arm=arm, q=q)),
            N(a.median_stage2_peak_rel_err_signed, 'R2B-13',
              fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm=arm, q=q)),
            N(int(t['n_abs_peak_le_0.05']), 'R2B-04', fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm=arm, q=q)),
            N(t.max_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'max_abs_peak_error', arm=arm, q=q)),
            N(a.mean_stage2_tail_mean_over_g2_0, 'R2B-13',
              fl(F_ARM_B, 'mean_stage2_tail_mean_over_g2_0', arm=arm, q=q)),
            N(t.max_tail_mean_over_g2_0, 'R2B-04', fl(F_TAIL_B, 'max_tail_mean_over_g2_0', arm=arm, q=q)),
            N(int(t.n_gate_pass), 'R2B-04', fl(F_TAIL_B, 'n_gate_pass', arm=arm, q=q))))
emit()
emit('Source: R2B-04 (`tail.csv`; columns `mean_abs_peak_error`, `n_abs_peak_le_0.05`, '
     '`max_abs_peak_error`, `max_tail_mean_over_g2_0` (a maximum over the 10 seeds), `n_gate_pass`) '
     'and R2B-13 (`arm_summary.csv`; columns `median_stage2_peak_rel_err_signed` and '
     '`mean_stage2_tail_mean_over_g2_0`, the mean over the 10 seeds). `n_gate_pass` counts runs that '
     'pass G-A and its G-N part.')
emit()
t50, t60 = lvl[('A_peak50', 50)][0], lvl[('A_peak50', 60)][0]
b50, b60 = lvl[('A_base', 50)][0], lvl[('A_base', 60)][0]
emit('What the tables show for `A_peak50`: the number of runs within 0.05 rises from %s to %s at q = 50 and '
     'from %s to %s at q = 60, and the maximum |peak error| falls from %s to %s and from %s to %s '
     '[R2B-04]. The signed error stays negative in the median (%s and %s) [R2B-13]. At q = 60 the '
     'number of runs passing G-A and G-N falls from %s to %s [R2B-04].' % (
         N(int(b50['n_abs_peak_le_0.05']), 'R2B-04', fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm='A_base', q=50)),
         N(int(t50['n_abs_peak_le_0.05']), 'R2B-04', fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm='A_peak50', q=50)),
         N(int(b60['n_abs_peak_le_0.05']), 'R2B-04', fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm='A_base', q=60)),
         N(int(t60['n_abs_peak_le_0.05']), 'R2B-04', fl(F_TAIL_B, 'n_abs_peak_le_0.05', arm='A_peak50', q=60)),
         N(b50.max_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'max_abs_peak_error', arm='A_base', q=50)),
         N(t50.max_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'max_abs_peak_error', arm='A_peak50', q=50)),
         N(b60.max_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'max_abs_peak_error', arm='A_base', q=60)),
         N(t60.max_abs_peak_error, 'R2B-04', fl(F_TAIL_B, 'max_abs_peak_error', arm='A_peak50', q=60)),
         N(lvl[('A_peak50', 50)][1].median_stage2_peak_rel_err_signed, 'R2B-13',
           fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm='A_peak50', q=50)),
         N(lvl[('A_peak50', 60)][1].median_stage2_peak_rel_err_signed, 'R2B-13',
           fl(F_ARM_B, 'median_stage2_peak_rel_err_signed', arm='A_peak50', q=60)),
         N(int(b60.n_gate_pass), 'R2B-04', fl(F_TAIL_B, 'n_gate_pass', arm='A_base', q=60)),
         N(int(t60.n_gate_pass), 'R2B-04', fl(F_TAIL_B, 'n_gate_pass', arm='A_peak50', q=60))))
emit()

emit('**The tail trade-off at q = 60.** Part (b) is violated by three runs whose G-A tail component '
     '(tail mean / e2*(0), limit 0.02) crosses the limit; in each of them the other G-A components hold.')
emit()
emit('**Table 6.3c. The three violating q = 60 runs, `A_peak50` against `A_base` at the same seeds.**')
emit()
emit('| seed | arm | tail mean / e2*(0) | eta_2/DW | RMSE_pos / e2*(0) | \\|peak error\\| | G-A tail component |')
emit('|---|---|---|---|---|---|---|')
for seed in (10503, 10504, 10510):
    for arm in ('A_base', 'A_peak50'):
        r = one(pr_b, arm=arm, q=60, seed=seed)
        emit('| %d | `%s` | %s | %s | %s | %s | %s |' % (
            seed, arm,
            N(r.stage2_tail_mean_over_g2_0, 'R2B-03', fl(F_PR_B, 'stage2_tail_mean_over_g2_0', arm=arm, q=60, seed=seed)),
            N(r.eta_T_over_dw, 'R2B-03', fl(F_PR_B, 'eta_T_over_dw', arm=arm, q=60, seed=seed)),
            N(r.stage2_rmse_pos_over_g2_0, 'R2B-03', fl(F_PR_B, 'stage2_rmse_pos_over_g2_0', arm=arm, q=60, seed=seed)),
            N(r.stage2_peak_rel_err_abs, 'R2B-03', fl(F_PR_B, 'stage2_peak_rel_err_abs', arm=arm, q=60, seed=seed)),
            'passes' if bool(r.G_A_tail_pass) else 'fails (> 0.02)'))
emit()
emit('Source: R2B-03 (`results/v2_refine_r2b/analysis/per_run.csv`; rows `arm` = `A_base` / `A_peak50`, '
     '`q` = 60, the three seeds; columns `stage2_tail_mean_over_g2_0`, `eta_T_over_dw`, '
     '`stage2_rmse_pos_over_g2_0`, `stage2_peak_rel_err_abs`, `G_A_tail_pass`). The limits are '
     'eta_2/DW <= 0.005, RMSE_pos / e2*(0) <= 0.05, tail mean / e2*(0) <= 0.02 [PL-02, section 1].')
emit()
others = pr_b[(pr_b.arm == 'A_peak50') & (pr_b.q == 60) & (~pr_b.seed.isin([10503, 10504, 10510]))]
emit('In the other seven q = 60 runs of `A_peak50` the tail mean / e2*(0) lies between %s and %s (computed '
     'here from R2B-03), and the mean over all ten seeds is %s against %s for `A_base` [R2B-13]; at q = 50 '
     'the largest value of `A_peak50` is %s against the limit 0.02 [R2B-04].' % (
         N(others.stage2_tail_mean_over_g2_0.min(), 'R2B-03', 'computed here: min of stage2_tail_mean_over_g2_0, A_peak50 q=60, seeds other than 10503/10504/10510'),
         N(others.stage2_tail_mean_over_g2_0.max(), 'R2B-03', 'computed here: max of stage2_tail_mean_over_g2_0, A_peak50 q=60, seeds other than 10503/10504/10510'),
         N(lvl[('A_peak50', 60)][1].mean_stage2_tail_mean_over_g2_0, 'R2B-13', fl(F_ARM_B, 'mean_stage2_tail_mean_over_g2_0', arm='A_peak50', q=60)),
         N(lvl[('A_base', 60)][1].mean_stage2_tail_mean_over_g2_0, 'R2B-13', fl(F_ARM_B, 'mean_stage2_tail_mean_over_g2_0', arm='A_base', q=60)),
         N(t50.max_tail_mean_over_g2_0, 'R2B-04', fl(F_TAIL_B, 'max_tail_mean_over_g2_0', arm='A_peak50', q=50))))
emit()

# design arithmetic table
emit('**Design arithmetic of the tail share.** The bins have width 10; the state space has 40 bins at '
     'q = 50 and 44 at q = 60; the peak set is four bins; the bins with |d| >= 2q are 20 at both q, and '
     'since 2q = 100 and 120 is a bin edge their union is exactly the region |d| >= 2q of the G-A tail '
     'mean [RR-10, section 7]. The share of the exploring starts that fall in those 20 tail bins is '
     '20 / (number of bins) under bin-balanced draws and (1 - s) x 20 / (number of bins - 4) under '
     'peak-focused draws with share s. These are shares of the exploring starts, not of the visited '
     'states. The values below were recomputed here and agree with the four-decimal values quoted in '
     'RR-10, section 7. The measured columns are means over the 10 seeds.')
emit()
emit('**Table 6.3d. Design share of exploring starts in the |d| >= 2q bins, and what was measured.**')
emit()
emit('| arm | s | tail-bin share, design (q = 50 / q = 60) | peak-set share of learner starts, measured '
     '(q = 50 / q = 60) | mean stage-2 effort on \\|d\\| >= 2q, effort units (q = 50 / q = 60) | mean tail '
     'mean / e2*(0) (q = 50 / q = 60) | raw draws clipped at the Beta clamp per run (q = 50 / q = 60) |')
emit('|---|---|---|---|---|---|---|')
share_arms = [('A_base', 0.0, 'bin-balanced'), ('R2b_A_peak25', 0.25, '0.25'),
              ('A_peak35', 0.35, '0.35'), ('A_peak40', 0.40, '0.40'), ('R2b_A_peak50', 0.50, '0.50')]
for arm, s, lab in share_arms:
    sp = {q: one(spec_c, arm=arm, q=q) for q in (50, 60)}
    nm = {'A_base': '`A_base`', 'R2b_A_peak25': '`A_peak25` (R2b)', 'A_peak35': '`A_peak35` (R2c, 6.5)',
          'A_peak40': '`A_peak40` (R2c, 6.5)', 'R2b_A_peak50': '`A_peak50` (R2b)'}[arm]
    sl = 'peak_visit_share'
    emit('| %s | %s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s |' % (
        nm, lab,
        N(tail_bin_share(s, 50), 'RR-10', 'section 7 arithmetic recomputed here: (1-s)*20/(40-4); s=0: 20/40', '.4f'),
        N(tail_bin_share(s, 60), 'RR-10', 'section 7 arithmetic recomputed here: (1-s)*20/(44-4); s=0: 20/44', '.4f'),
        N(sp[50].peak_visit_share, 'R2C-12', fl(F_SPEC_C, sl, arm=arm, q=50), '.4f'),
        N(sp[60].peak_visit_share, 'R2C-12', fl(F_SPEC_C, sl, arm=arm, q=60), '.4f'),
        N(sp[50].tail2q_mean_e2hat, 'R2C-12', fl(F_SPEC_C, 'tail2q_mean_e2hat', arm=arm, q=50)),
        N(sp[60].tail2q_mean_e2hat, 'R2C-12', fl(F_SPEC_C, 'tail2q_mean_e2hat', arm=arm, q=60)),
        N(sp[50].tail2q_mean_abs_err_over_g2_0, 'R2C-12', fl(F_SPEC_C, 'tail2q_mean_abs_err_over_g2_0', arm=arm, q=50)),
        N(sp[60].tail2q_mean_abs_err_over_g2_0, 'R2C-12', fl(F_SPEC_C, 'tail2q_mean_abs_err_over_g2_0', arm=arm, q=60)),
        N(sp[50].d1_flagged_L_s2, 'R2C-12', fl(F_SPEC_C, 'd1_flagged_L_s2', arm=arm, q=50)),
        N(sp[60].d1_flagged_L_s2, 'R2C-12', fl(F_SPEC_C, 'd1_flagged_L_s2', arm=arm, q=60))))
emit()
emit('Source: design columns computed here from the bin counts of RR-10, section 7 (40 and 44 bins, 4 '
     'peak bins, 20 tail bins). Measured columns: R2C-12 (`results/v2_refine_r2c/analysis/'
     'waveS_specifics_mean.csv`; columns `peak_visit_share`, `tail2q_mean_e2hat`, '
     '`tail2q_mean_abs_err_over_g2_0`, `d1_flagged_L_s2`, mean over seeds); the rows of the R2b arms '
     'agree with the per-run table R2B-29 (checked by `check_r2b29.py`). Rows s = 0.35 and 0.40 are the R2c '
     'arms (section 6.5). `tail2q_mean_abs_err_over_g2_0` is the G-A tail-mean statistic (e2* = 0 on '
     '|d| >= 2q). "Raw draws clipped" are the learner\'s stage-2 raw Beta draws outside [1e-6, 1 - 1e-6] '
     'summed over Phase A (diagnostic D1, section 5.1).')
emit()
emit('**What this shows and what it does not.** It shows that moving sampling weight onto the four peak bins '
     'lowers the |peak error| of the end-of-A policy, and that the same change moves the tail mean toward '
     'the G-A limit: as the peak share rises, the design share of starts on the |d| >= 2q bins falls and the '
     'measured tail mean / e2*(0) and mean effort on |d| >= 2q rise (table 6.3d). It does not show that the sample share is the cause of the '
     'baseline peak error: the arms change several measured quantities at once (the peak-set share, the '
     'tail-bin share, the mean stage-2 effort on |d| >= 2q and the clipped raw draws, table 6.3d), and no R2b '
     'arm varies one of them alone. The censored likelihood (section 6.2) changes the likelihood of the '
     'clipped draws without changing the start distribution and has intervals containing 0 at both q; the '
     'reports draw no inference from the contrast and neither do I. The figure shows the per-seed pairs '
     '(the R2b summary restates the figures of table 6.3d, RR-04).')
emit()
emit('![R2b wave A, `A_peak50` minus `A_base`: paired difference of |peak error| for each of the 10 seeds '
     '(points), the median (bar) and the mean with its 95% bootstrap interval (bootstrap seed 20261004), '
     'at q = 50 (left) and q = 60 (right); negative = `A_peak50` better. Manifest ID FG-15.]'
     '(figures/FG-15_waveA_A_peak50_vs_A_base.png)')
emit()

# =================================================================== 6.4
LED.section = '6.4'
emit('### 6.4 The seed-30510 case')
emit()
tgt = one(dec50, seed=30510)
oth = dec50[dec50.seed != 30510]
assert len(oth) == 19
assert float(conf.peak_rel_err.min()) == float(tgt.peak_rel_err)   # most negative of the 40
lt = one(late, seed=30510)
lo_ = late[late.seed != 30510]
eta_v = num['owner.eta2_over_dw']
emit('**Outcome.** The one failed run of the v2.0 confirmation, q = 50 seed 30510, fails G-A through '
     'eta_2/DW = %s against the limit 0.005 and has the most negative stage-2 peak error of all 40 '
     'confirmation runs; none of the pre-registered hypotheses H1-H4 is supported for it. It is one run, '
     'the diagnostic is descriptive, and the cause was not decided.' % (
         N(eta_v, 'R2B-15', 'numbers.json values owner.eta2_over_dw', '.6g')))
emit()
emit('Context: the confirmation passed with %s of 20 runs at q = 50 and %s of 20 at q = 60 (rule: at least '
     '18 of 20); the one failed run was not re-run [RR-03, "Verdict"]. Seed 30510 is a confirmation seed '
     '(30501-30520), not one of the development seeds 10501-10510, so its peak error is not paired with a '
     'baseline run. The diagnostic (R2b P2) is read-only: no run was repeated and no threshold changed '
     '[RR-11].' % (C('19', 'RR-03', 'Verdict table, q=50 passes (report only; CF-02 holds it)'),
                   C('20', 'RR-03', 'Verdict table, q=60 passes (report only; CF-02 holds it)')))
emit()
emit('**Table 6.4a. Seed 30510 against the other 19 q = 50 confirmation runs (end of Phase A).**')
emit()
emit('| quantity | seed 30510 | other 19: median [min, max] | rank of 30510 (1 = lowest) |')
emit('|---|---|---|---|')


def rank_low(col, df):
    v = float(one(df, seed=30510)[col])
    return int((df[col] < v).sum()) + 1


def row64(label, col, df=dec50, item='R2B-17', fmt=None, f=None):
    f = f or 'tab_decomposition_q50.csv'
    t_ = float(one(df, seed=30510)[col])
    o_ = df[df.seed != 30510][col].astype(float)
    emit('| %s | %s | %s [%s, %s] | %s |' % (
        label, N(t_, item, '%s: seed=30510, col %s' % (f, col), fmt),
        N(o_.median(), item, '%s: median over the 19 other seeds of col %s (computed here)' % (f, col), fmt),
        N(o_.min(), item, '%s: min over the 19 other seeds of col %s (computed here)' % (f, col), fmt),
        N(o_.max(), item, '%s: max over the 19 other seeds of col %s (computed here)' % (f, col), fmt),
        N(rank_low(col, df), item, '%s: rank of seed 30510 in col %s (computed here)' % (f, col))))


row64('signed peak error', 'peak_rel_err')
row64('eta_2/DW = maximum over on-path states (G-A limit 0.005)', 'on_max')
row64('eta_2/DW maximum over off-path states', 'off_max')
row64('share of the d = 0 gap predicted by the policy\'s noise', 'smooth_share')
row64('d = 0 gap, e2*(0) - e2_hat(0), effort units', 'rl_gap')
row64('  of which predicted by the policy\'s noise', 'smooth_gap')
row64('  remainder (smoothing-free part)', 'remainder_gap')
row64('policy noise sigma_2(0), effort units', 'sigma0')
row64('RMSE_pos / e2*(0) (G-A limit 0.05)', 'rmse_over_g20')
row64('tail mean / e2*(0) (G-A limit 0.02)', 'tail_mean_over_g20')
# late mean from R2B-24
t_ = float(lt.peak_late_mean)
o_ = lo_.peak_late_mean.astype(float)
emit('| mean signed peak error over updates 900-1600 | %s | %s [%s, %s] | %s |' % (
    N(t_, 'R2B-24', 'tab_late_means_per_seed_q50.csv: seed=30510, col peak_late_mean'),
    N(o_.median(), 'R2B-24', 'median over the 19 other seeds of col peak_late_mean (computed here)'),
    N(o_.min(), 'R2B-24', 'min over the 19 other seeds (computed here)'),
    N(o_.max(), 'R2B-24', 'max over the 19 other seeds (computed here)'),
    N(int((late.peak_late_mean < t_).sum()) + 1, 'R2B-24', 'rank of seed 30510 in col peak_late_mean (computed here)')))
emit()
emit('Source: R2B-17 (`results/v2_refine_r2b/diag_30510/tables/tab_decomposition_q50.csv`; the columns named '
     'in the rows) and R2B-24 (`tab_late_means_per_seed_q50.csv`, column `peak_late_mean`); median, minimum, '
     'maximum and rank over the 19 other seeds computed here. The same values are in R2B-15 '
     '(`numbers.json`; keys `owner.*`, `end.*`, `seedstat.late_mean.*`) and in RR-11, sections 0, 2.4 and 6. '
     'Rank 1 is the lowest value, which for the signed peak error and the late mean is the most negative.')
emit()
on_ = num['owner.on_max']
off_ = num['owner.off_max']
emit('In the PI\'s numbers (reproduced from the files [RR-11, section 0]): eta_2/DW %s on the path against '
     '%s off the path, the stage-2 peak error %s, and a smoothed-game share of the d = 0 gap of %s '
     '[R2B-15, keys `owner.on_max`, `owner.off_max`, `owner.peak_err_signed`, `owner.smoothed_share`]. '
     'RMSE_pos and the tail mean pass for this run, so the failure is the eta_2 component alone '
     '[RR-03, "Verdict"; R2B-17].' % (
         N(on_, 'R2B-15', 'numbers.json values owner.on_max', '.6g'),
         N(off_, 'R2B-15', 'numbers.json values owner.off_max', '.4g'),
         N(num['owner.peak_err_signed'], 'R2B-15', 'numbers.json values owner.peak_err_signed', '.4f'),
         N(num['owner.smoothed_share'], 'R2B-15', 'numbers.json values owner.smoothed_share', '.3f')))
emit()

# hypotheses
ht = one(hyp, is_target=True)
emit('**The pre-registered hypotheses.** R2b fixed four rules before any diagnostic output existed, in terms '
     'of the signed peak error S(u) of the weight exports (every 25 updates) against the 10th percentile '
     'p10_pack(S, u) of the 19 other q = 50 confirmation runs: H1, the peak never rose to the pack\'s level; '
     'H2, it rose and decayed in the LR-decay window (departure at update 1225 or later); H3, a late '
     'excursion after a stable plateau; H4, a plateau at the pack\'s peak height with an unusually rounded '
     'cusp [RR-11, section 7; RR-09, section 5]. Applied literally to seed 30510 they give:')
emit()
emit('| hypothesis | computed inputs | verdict |')
emit('|---|---|---|')
emit('| H1 | max over the 64 exports of S = %s (at update %s); p10_pack(S, 1600) = %s | %s |' % (
    N(ht.S_max_all_exports, 'R2B-16', 'hypothesis_rules.csv: is_target=True, col S_max_all_exports'),
    N(int(ht.u_of_S_max), 'R2B-16', 'col u_of_S_max'),
    N(ht.p10_pack_S_1600, 'R2B-16', 'col p10_pack_S_1600'), ht.H1))
emit('| H2 | departure update u_leave = %s (needs >= 1225) | %s |' % (
    N(int(ht.u_leave), 'R2B-16', 'col u_leave'), ht.H2))
emit('| H3 | exports before u_leave that are not below the band: %s (needs >= %s) | %s |' % (
    N(int(ht.plateau_exports_before_u_leave), 'R2B-16', 'col plateau_exports_before_u_leave'),
    C('16', 'RR-11', 'section 7, H3 rule: 16 consecutive exports (report only)'), ht.H3))
emit('| H4 | location-free peak below the pack\'s p10 at %s of %s exports in 1225-1600; cusp rounding L - S = %s '
     'against p90_pack = %s | %s |' % (
         N(int(ht.n_L_below_p10_1225_1600), 'R2B-16', 'col n_L_below_p10_1225_1600'),
         C('16', 'RR-11', 'section 7 H4 row: 16 exports in 1225-1600 (report only)'),
         N(ht.cusp_rounding_L_minus_S_1600, 'R2B-16', 'col cusp_rounding_L_minus_S_1600'),
         N(ht.p90_pack_cusp_1600, 'R2B-16', 'col p90_pack_cusp_1600'), ht.H4))
emit()
emit('Source: R2B-16 (`results/v2_refine_r2b/diag_30510/tables/hypothesis_rules.csv`, row `is_target` = True; '
     'columns `S_max_all_exports`, `u_of_S_max`, `p10_pack_S_1600`, `u_leave`, `plateau_exports_before_u_leave`, '
     '`n_L_below_p10_1225_1600`, `cusp_rounding_L_minus_S_1600`, `p90_pack_cusp_1600`, `H1`-`H4`, `label`).')
emit()
others_h = hyp[~hyp.is_target.astype(bool)]
n_h2 = int((others_h.H2 == 'supported').sum())
n_h2_50 = int(((others_h.H2 == 'supported') & (others_h.block_q == 50) & (others_h.seed != 30510)).sum())
n_h2_60 = int(((others_h.H2 == 'supported') & (others_h.block_q == 60)).sum())
n_oth = len(hyp) - 1
n_h2_pass = int(((others_h.H2 == 'supported') & (others_h.G_A_pass.astype(bool))).sum())
assert n_oth == 39
emit('The label is "%s" [R2B-16, column `label`]. The literal rules discriminate poorly: H2 is "supported" '
     'for %s of the other %s confirmation runs (%s at q = 50, %s at q = 60), and all %s of those runs '
     'passed G-A [R2B-16, columns `H2` and `G_A_pass`, counted here; RR-11, section 7.1; RR-04].' % (
         str(ht.label), N(n_h2, 'R2B-16', 'count of rows H2=supported among the other 39 runs (computed here)'),
         N(n_oth, 'R2B-16', 'rows of hypothesis_rules.csv other than the target'),
         N(n_h2_50, 'R2B-16', 'same count, block q=50'), N(n_h2_60, 'R2B-16', 'same count, block q=60'),
         N(n_h2_pass, 'R2B-16', 'count with G_A_pass among them (computed here)')))
emit()

# late level, footprint
n_lower = int((lo_.peak_late_mean.astype(float) <= t_).sum())
emit('**Descriptive facts (not pre-registered readings).** The mean signed peak error over updates 900-1600 is '
     '%s for seed 30510 against a median of %s over the other 19 seeds (range %s to %s); %s of the 19 have a '
     'late mean at or below it [R2B-24; RR-11, section 2.4]. The peak-set share of its learner starts is '
     '%s against a median of %s for the others, design value %s, so the sampling of starts shows no '
     'difference [R2B-15, keys `vis.peak_share.*`; RR-11, section 5]. Its stage-2 raw Beta draws clipped at '
     'the lower clamp number %s in Phase A against a median of %s and a maximum of %s for the other 19 '
     '[R2B-27, column `d1_L_s2_lo_sum`, q = 50, computed here]. Several optimisation series leave the other '
     'seeds\' band for good early: the concentration alpha + beta at d = 0 at update %s, the actor '
     'gradient norm at %s, sigma_2(0) at %s, and the advantage SD and critic loss at %s [R2B-15, keys '
     '`optexit.*.sustained_exit_u`]; the pre-registered departure of the peak is at update %s. Whether '
     'the early footprint is a cause or a co-symptom of the low peak cannot be decided from the files '
     '[RR-11, section 8].' % (
         N(t_, 'R2B-24', 'peak_late_mean seed 30510'),
         N(o_.median(), 'R2B-24', 'median of the other 19 (computed here)'),
         N(o_.min(), 'R2B-24', 'min of the other 19'), N(o_.max(), 'R2B-24', 'max of the other 19'),
         N(n_lower, 'R2B-24', 'count of other seeds with peak_late_mean <= seed 30510 (computed here)'),
         N(num['vis.peak_share.target'], 'R2B-15', 'values vis.peak_share.target', '.4f'),
         N(num['vis.peak_share.others_median'], 'R2B-15', 'values vis.peak_share.others_median', '.4f'),
         N(0.1, 'R2B-15', 'RR-11 section 5 design value 4 of 40 bins', '.4f'),
         N(int(clamp[(clamp.q == 50) & (clamp.seed == 30510)].d1_L_s2_lo_sum.iloc[0]), 'R2B-27',
           'tab_d1_clamp_counts_phaseA_sum.csv: q=50 seed=30510 col d1_L_s2_lo_sum'),
         N(float(clamp[(clamp.q == 50) & (clamp.seed != 30510)].d1_L_s2_lo_sum.median()), 'R2B-27',
           'median over the other 19 q=50 seeds of d1_L_s2_lo_sum (computed here)', '.4g'),
         N(int(clamp[(clamp.q == 50) & (clamp.seed != 30510)].d1_L_s2_lo_sum.max()), 'R2B-27',
           'max over the other 19 q=50 seeds of d1_L_s2_lo_sum (computed here)'),
         N(int(num['optexit.conc0.sustained_exit_u']), 'R2B-15', 'values optexit.conc0.sustained_exit_u'),
         N(int(num['optexit.grad_norm_actor_mean.sustained_exit_u']), 'R2B-15',
           'values optexit.grad_norm_actor_mean.sustained_exit_u'),
         N(int(num['optexit.sigma0.sustained_exit_u']), 'R2B-15', 'values optexit.sigma0.sustained_exit_u'),
         N(int(num['optexit.adv_sd.sustained_exit_u']), 'R2B-15', 'values optexit.adv_sd.sustained_exit_u'),
         N(int(ht.u_leave), 'R2B-16', 'col u_leave')))
emit()
eta_t = eta_exp[(eta_exp.u >= 900)]
n_eta_above = int((eta_t.target > 0.005).sum())
emit('The development-tier eta_2/DW of seed 30510 is above 0.005 at %s of the %s weight exports from update '
     '900 on (recomputed in the diagnostic; R2B-21, column `target`, counted here), so the final-iterate '
     'value %s is not an isolated draw [RR-11, section 8, item 3 states the same count].' % (
         N(n_eta_above, 'R2B-21', 'tab_eta2_every_export_q50.csv: count of u>=900 with target > 0.005 (computed here)'),
         N(len(eta_t), 'R2B-21', 'number of exports u>=900 (computed here)'),
         N(eta_v, 'R2B-15', 'values owner.eta2_over_dw', '.6g')))
emit()
emit('**What the report states could not be decided** [RR-11, section 8; none of these experiments was run]: '
     '(1) whether the early concentration and noise divergence causes the low peak or both follow from an '
     'earlier state of the network (a branching experiment from a checkpoint at update 100 with reseeded '
     'minibatch and sampling streams, and a branch with the concentration manipulated, would test it); '
     '(2) whether the plateau is permanent (a Phase-A continuation from the stored end-of-A state would show '
     'whether it moves); (3) how much of the G-A failure is the end-iterate draw (a tail-averaged candidate '
     'would be the experiment); (4) whether the larger clamp count is more than a co-occurrence (a re-run '
     'with the clip level moved is the named test; the files show only the co-occurrence); (5) whether the '
     'export at update 825, the one export near the pack\'s median, is a transient of the policy mean or a '
     'reversed move (the exports are 25 updates apart).')
emit()
emit('**What the case does and does not show.** It shows one confirmation run whose stage-2 peak error and '
     'late level lie below those of every other q = 50 run, with the largest remainder (smoothing-free part) '
     'of the d = 0 gap in table 6.4a, and that the pre-registered rules classify it as "none of H1-H4" while '
     'the same rules also label five passing runs as H2. It does not show a mechanism: it is a single run, '
     'the comparisons with the other 19 seeds are descriptive and association is not cause, and the '
     'diagnostic offers no manipulation that separates the readings. The one measurement in it that bears on '
     'a sampling-weight explanation is the peak-set share of its learner starts, which does not differ from '
     'the other seeds (above); the diagnostic did not vary the weighting.')
emit()
emit('![Seed 30510 (q = 50, v2.0 confirmation): signed relative error of e2_hat(0) at the weight exports every '
     '25 updates (orange), against the median (blue line) and the 10-90% band of the other 19 q = 50 seeds; '
     'left column e2_hat(0), right column the location-free peak; the top row is the whole of Phase A, the '
     'middle row a zoom from update 250, the bottom row the rank of seed 30510 among the 20 runs; dashed '
     'orange line = update 900 (pre-registered departure), dotted = start of the LR decay (update 1201). '
     'Manifest ID FG-12.](figures/FG-12_fig1_peak_trajectory_q50.png)')
emit()
emit('![Seed 30510 and two other q = 50 seeds (30513, 30506) at the end of Phase A: (a) learned stage-2 '
     'effort against the closed form, (b, c) its error on the whole grid and near the cusp, (d) the one-step '
     'deviation gain divided by DW against the G-A limit 0.005, with the on-path maximum of seed 30510 at '
     'd = -4, (e) policy noise sigma_2(d), (f) symmetry error. Manifest ID FG-13.]'
     '(figures/FG-13_fig3_endA_profile.png)')
emit()

# =================================================================== 6.5
LED.section = '6.5'
emit('### 6.5 R2c: the start-distribution pilot and the selection rule\'s verdict')
emit()
emit('**Outcome.** No arm is selected: every arm meets criterion part (b), and no arm meets part (a) at both '
     'q, because at q = 60 the 95% interval of the mean paired difference contains 0 for all four arms '
     '[RR-06; R2C-02]. The pre-registered rule was applied mechanically and not relaxed; there is no '
     'protocol v2.1.')
emit()
emit('**Arms.** Four peak-focused start distributions against the baseline `parents_A`, development seeds '
     '10501-10510 at q = 50 and 60, 20 runs per arm, all with the half-width 20 of R2b [RR-10, section 2]: '
     '`A_peak35` (share 0.35 from update 1) and `A_peak40` (share 0.40 from update 1), which change the '
     'sampler for the whole of Phase A, and `A_peak50_late400` and `A_peak50_late800` (share 0.50 only from '
     'update 1201 and from update 801, i.e. the last 400 and the last 800 of the 1600 Phase-A updates; '
     'before that the locked bin-balanced sampler). The Phase-A learning rate is constant to update 1200 and '
     'then decays linearly to 3e-5 over updates 1201-1600 [RR-10, section 2]. The bootstrap seed is %s '
     '[R2C-02, column `boot_seed`].' % N(int(crit_c.boot_seed.iloc[0]), 'R2C-02',
                                          fl(F_CRIT_C, 'boot_seed', arm='A_peak35')))
emit()
emit('**Criterion and rule (pre-registered, RR-10 section 3; R2C-03, key `rule`).** Per arm: (a) the 95% '
     'percentile bootstrap interval of the mean paired difference of |peak error| against `parents_A` excludes '
     '0 in the improving direction at both q; (b) no run that passed G-A and its G-N part under the baseline '
     'fails it under the arm. Selection among the arms meeting (a) and (b): the largest total number of runs '
     'with |peak error| <= 0.05 over both q; tie: the smaller mean |peak error| over both q; further tie: the '
     'smaller change from the locked sampler. If no arm meets (a) and (b): write the pilot reports and '
     'stop; do not relax the rule and do not add arms.')
emit()
emit('**Table 6.5a. R2c arms against `A_base` (arm minus baseline of |peak error|, bootstrap seed 20261005).**')
emit()
emit('| arm | share, from update | q = 50: mean [95% CI] | q = 60: mean [95% CI] | (a) at q = 50 / q = 60 | (b) | '
     'runs with \\|peak error\\| <= 0.05: q50 + q60 | mean \\|peak error\\|, 20 runs | largest tail mean / e2*(0): '
     'q50 / q60 |')
emit('|---|---|---|---|---|---|---|---|---|')
arms_c = [('A_peak35', '0.35, 1'), ('A_peak40', '0.40, 1'), ('A_peak50_late400', '0.50, 1201'),
          ('A_peak50_late800', '0.50, 801')]
for arm, lab in arms_c:
    cr = one(crit_c, arm=arm)
    si = one(sel_c, arm=arm)
    tl = {q: one(tail_c, arm=arm, q=q) for q in (50, 60)}
    emit('| `%s` | %s | %s | %s | %s / %s | %s | %s + %s = %s | %s | %s / %s |' % (
        arm, lab,
        CI(cr.mean_q50, cr.ci_mean_lo_q50, cr.ci_mean_hi_q50, 'R2C-02', fl(F_CRIT_C, 'q50', arm=arm)),
        CI(cr.mean_q60, cr.ci_mean_lo_q60, cr.ci_mean_hi_q60, 'R2C-02', fl(F_CRIT_C, 'q60', arm=arm)),
        met(cr.a_q50), met(cr.a_q60), cr.b_status,
        N(int(si['n_abs_peak_le_0.05_q50']), 'R2C-08', fl(F_SEL_C, 'n_abs_peak_le_0.05_q50', arm=arm)),
        N(int(si['n_abs_peak_le_0.05_q60']), 'R2C-08', fl(F_SEL_C, 'n_abs_peak_le_0.05_q60', arm=arm)),
        N(int(si['n_abs_peak_le_0.05_total']), 'R2C-08', fl(F_SEL_C, 'n_abs_peak_le_0.05_total', arm=arm)),
        N(si.mean_abs_peak_error, 'R2C-08', fl(F_SEL_C, 'mean_abs_peak_error', arm=arm)),
        N(tl[50].max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm=arm, q=50)),
        N(tl[60].max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm=arm, q=60))))
# baseline row
bt = {q: one(tail_c, arm='A_base', q=q) for q in (50, 60)}
emit('| `A_base` (baseline) | bin-balanced | | | | | %s + %s = %s | %s | %s / %s |' % (
    N(int(bt[50]['n_abs_peak_le_0.05']), 'R2C-04', fl(F_TAIL_C, 'n_abs_peak_le_0.05', arm='A_base', q=50)),
    N(int(bt[60]['n_abs_peak_le_0.05']), 'R2C-04', fl(F_TAIL_C, 'n_abs_peak_le_0.05', arm='A_base', q=60)),
    N(int(bt[50]['n_abs_peak_le_0.05'] + bt[60]['n_abs_peak_le_0.05']), 'R2C-04',
      'computed here: sum of n_abs_peak_le_0.05 over q for A_base'),
    N((bt[50].mean_abs_peak_error + bt[60].mean_abs_peak_error) / 2, 'R2C-04',
      'computed here: mean of the two per-q mean_abs_peak_error of A_base (n = 10 per q)'),
    N(bt[50].max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm='A_base', q=50)),
    N(bt[60].max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm='A_base', q=60))))
emit()
emit('Source: R2C-02 (`results/v2_refine_r2c/analysis/criterion.csv`; columns `mean_q50`, `ci_mean_lo_q50`, '
     '`ci_mean_hi_q50`, q60 likewise, `a_q50`, `a_q60`, `b_status`), R2C-08 (`selection_inputs.csv`; columns '
     '`n_abs_peak_le_0.05_q50`, `_q60`, `_total`, `mean_abs_peak_error`), R2C-04 (`tail.csv`; column '
     '`max_tail_mean_over_g2_0`, a maximum over 10 seeds; baseline rows). The G-A tail limit is 0.02. The '
     'baseline mean |peak error| over both q is computed here as the mean of the two per-q means; it equals '
     'the value in R2C-09 (column `mean |peak|, both q`).')
emit()
r9 = one(resp_c, arm='A_base')
assert abs(float(r9['mean |peak|, both q']) - (bt[50].mean_abs_peak_error + bt[60].mean_abs_peak_error) / 2) < 1e-9
reason = str(sel_json['reason'])
emit('**Verdict, as recorded.** Outcome `%s`; selected arm: none (`selected` = %s) [R2C-03]. The reason string '
     'of the selection record reads: "%s" [R2C-03, key `reason`]. The selection report states: "No arm meets '
     'both parts of the criterion; the round stops after the pilot report" [RR-07, "Outcome"]. The ineligible '
     'arms and reasons (`ineligible_reasons`): %s.' % (
         sel_json['outcome'], 'None' if sel_json['selected'] is None else sel_json['selected'],
         reason,
         '; '.join('`%s`: %s' % (a['arm'], a['ineligible_reasons'][0] if len(a['ineligible_reasons']) == 1
                              else '; '.join(a['ineligible_reasons'])) for a in sel_json['arms'])))
emit()
# prefix identities
late_chk = {}
for run in chk_json['runs']:
    if run['arm'] in ('A_peak50_late400', 'A_peak50_late800'):
        c5 = run['checks']['C5']
        late_chk.setdefault(run['arm'], []).append(c5)
pre = {}
for arm, lst in late_chk.items():
    ok = sum(1 for c in lst if c['ok'])
    fd = set(c['exports']['first_diff_update'] for c in lst)
    pre[arm] = (ok, len(lst), fd)
assert pre['A_peak50_late400'][2] == {1225} and pre['A_peak50_late800'][2] == {825}
emit('**Checks.** The D2 prefix identities of the two late arms hold in %s of %s runs each: all weight exports '
     'up to update %s (`A_peak50_late400`) and up to update %s (`A_peak50_late800`) are bit-identical to '
     '`parents_A`, and the first differing export is update %s and %s in every run [R2C-06, key `runs[*].checks.C5`, '
     'counted here; RR-06]. The selection was recomputed independently by a script that does not import the '
     'analysis tool: all four arms\' means, intervals, counts and (b) verdicts agree to 1e-12 and the selected '
     'arm is none [R2C-10].' % (
         N(pre['A_peak50_late400'][0], 'R2C-06', 'launch_checks.json runs[arm=A_peak50_late400].checks.C5.ok count (computed here)'),
         N(pre['A_peak50_late400'][1], 'R2C-06', 'number of A_peak50_late400 runs in launch_checks.json'),
         N(1200, 'R2C-06', 'launch_checks.json summary.prefix_bounds.A_peak50_late400'),
         N(800, 'R2C-06', 'launch_checks.json summary.prefix_bounds.A_peak50_late800'),
         N(1225, 'R2C-06', 'runs[*].checks.C5.exports.first_diff_update, A_peak50_late400'),
         N(825, 'R2C-06', 'runs[*].checks.C5.exports.first_diff_update, A_peak50_late800')))
emit()

# share response table
emit('**Table 6.5b. Share response, shares 0.25 to 0.50 from update 1 (R2b reference rows re-bootstrapped with '
     'seed 20261005).** The R2b rows are references, not eligible, and not inputs of the selection rule.')
emit()
emit('| arm | share | q = 50: mean diff [95% CI] | q = 60: mean diff [95% CI] | mean diff / baseline mean '
     '\\|peak error\\| (q50 / q60) | runs with \\|peak error\\| <= 0.05 (q50 / q60) | largest tail mean / e2*(0) '
     '(q50 / q60) | tail-bin share, design (q50 / q60) |')
emit('|---|---|---|---|---|---|---|---|')


def parse_ci(sv):
    nums = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', sv)]
    assert len(nums) == 3, sv
    return nums


F9 = F_RESP_C
for arm, s in (('R2b_A_peak25', 0.25), ('A_peak35', 0.35), ('A_peak40', 0.40), ('R2b_A_peak50', 0.50)):
    rr = one(resp_c, arm=arm)
    cis = {}
    for q in (50, 60):
        m, lo, hi = parse_ci(rr['q%d mean diff [95%% CI]' % q])
        cis[q] = (m, lo, hi)
    # use the numeric criterion table for R2c arms (same values)
    if arm in ('A_peak35', 'A_peak40'):
        cr = one(crit_c, arm=arm)
        cis[50] = (cr.mean_q50, cr.ci_mean_lo_q50, cr.ci_mean_hi_q50)
        cis[60] = (cr.mean_q60, cr.ci_mean_lo_q60, cr.ci_mean_hi_q60)
        itm, fcol = 'R2C-02', F_CRIT_C
    else:
        itm, fcol = 'R2C-09', F9
    tl = {q: one(tail_c, arm=arm, q=q) for q in (50, 60)}
    rel = {q: cis[q][0] / bt[q].mean_abs_peak_error for q in (50, 60)}
    emit('| `%s` | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s |' % (
        arm, C('%.2f' % s, 'RR-10', 'section 2 arm table / R2C-09 col share'),
        CI(*cis[50], itm, fl(fcol, 'q50 mean diff [95% CI]', arm=arm)),
        CI(*cis[60], itm, fl(fcol, 'q60 mean diff [95% CI]', arm=arm)),
        N(rel[50], 'R2C-09', 'computed here: q50 mean diff / A_base mean_abs_peak_error (R2C-04)'),
        N(rel[60], 'R2C-09', 'computed here: q60 mean diff / A_base mean_abs_peak_error (R2C-04)'),
        N(int(rr['q50 n<=0.05']), 'R2C-09', fl(F9, 'q50 n<=0.05', arm=arm)),
        N(int(rr['q60 n<=0.05']), 'R2C-09', fl(F9, 'q60 n<=0.05', arm=arm)),
        N(tl[50].max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm=arm, q=50)),
        N(tl[60].max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm=arm, q=60)),
        N(tail_bin_share(s, 50), 'RR-10', 'section 7 arithmetic recomputed here', '.4f'),
        N(tail_bin_share(s, 60), 'RR-10', 'section 7 arithmetic recomputed here', '.4f')))
emit()
emit('Source: R2C-09 (`results/v2_refine_r2c/analysis/waveS_response.csv`; the R2b reference rows, columns '
     '`q50 mean diff [95% CI]`, `q60 mean diff [95% CI]`, `q50 n<=0.05`, `q60 n<=0.05`), R2C-02 for the '
     'R2c arms\' intervals, R2C-04 (`tail.csv`; `max_tail_mean_over_g2_0`), design column recomputed '
     'from RR-10, section 7 (table 6.3d). The column "mean diff / baseline mean |peak error|" is computed '
     'here as the point estimate divided by the baseline\'s per-q mean |peak error| (R2C-04, rows `A_base`). '
     'The R2b arms\' own intervals with bootstrap seed 20261004 are in table 6.3a; the point estimates are '
     'identical.')
emit()
emit('Observations, as labelled in the R2c summary (each restates a table; none is a pre-registered test) '
     '[RR-06]:')
emit()
pc = {a: one(pair_c, arm=a, q=60, metric='stage2_peak_rel_err_abs') for a in
      ('A_peak35', 'A_peak40', 'A_peak50_late400', 'A_peak50_late800')}
emit('- **q = 50:** three of the four arms meet part (a); `A_peak50_late400` misses it, the upper bound of '
     'its interval being %s (above 0). The number of runs within 0.05 rises from %s (baseline) to %s, %s, %s '
     'and %s for the four arms in table order [R2C-02, R2C-04].' % (
         N(one(crit_c, arm='A_peak50_late400').ci_mean_hi_q50, 'R2C-02', fl(F_CRIT_C, 'ci_mean_hi_q50', arm='A_peak50_late400')),
         N(int(bt[50]['n_abs_peak_le_0.05']), 'R2C-04', fl(F_TAIL_C, 'n_abs_peak_le_0.05', arm='A_base', q=50)),
         *[N(int(one(tail_c, arm=a, q=50)['n_abs_peak_le_0.05']), 'R2C-04',
             fl(F_TAIL_C, 'n_abs_peak_le_0.05', arm=a, q=50))
           for a in ('A_peak35', 'A_peak40', 'A_peak50_late400', 'A_peak50_late800')]))
emit('- **q = 60:** no arm\'s interval excludes 0. The point estimates are %s, %s, %s and %s for `A_peak35`, '
     '`A_peak40`, `A_peak50_late400`, `A_peak50_late800` (%s, %s, %s and %s of the 10 seeds improve), and the '
     'number of runs within 0.05 stays at the baseline\'s %s for the three arms that start at update 1 or '
     '1201 and falls to %s for `A_peak50_late800` [R2C-02, R2C-07, R2C-04]. The baseline\'s q = 60 mean error '
     '(%s) is already smaller than at q = 50 (%s), and the q = 60 intervals are wide, so an interval '
     'containing 0 does not show that an arm has no effect at q = 60.' % (
         *[N(one(crit_c, arm=a).mean_q60, 'R2C-02', fl(F_CRIT_C, 'mean_q60', arm=a))
           for a in ('A_peak35', 'A_peak40', 'A_peak50_late400', 'A_peak50_late800')],
         *[N(int(pc[a].n_better), 'R2C-07', fl('results/v2_refine_r2c/analysis/paired.csv', 'n_better', arm=a, q=60,
                                              metric='stage2_peak_rel_err_abs'))
           for a in ('A_peak35', 'A_peak40', 'A_peak50_late400', 'A_peak50_late800')],
         N(int(bt[60]['n_abs_peak_le_0.05']), 'R2C-04', fl(F_TAIL_C, 'n_abs_peak_le_0.05', arm='A_base', q=60)),
         N(int(one(tail_c, arm='A_peak50_late800', q=60)['n_abs_peak_le_0.05']), 'R2C-04',
           fl(F_TAIL_C, 'n_abs_peak_le_0.05', arm='A_peak50_late800', q=60)),
         N(bt[60].mean_abs_peak_error, 'R2C-04', fl(F_TAIL_C, 'mean_abs_peak_error', arm='A_base', q=60)),
         N(bt[50].mean_abs_peak_error, 'R2C-04', fl(F_TAIL_C, 'mean_abs_peak_error', arm='A_base', q=50))))
emit('- **Part (b) holds for all four arms** and the tail means stay below the limit 0.02: the largest is %s '
     '(`A_peak35`, q = 60) and %s (`A_peak50_late800`, q = 50); for comparison R2b\'s `A_peak50` reached %s '
     'and the baseline %s [R2C-04].' % (
         N(one(tail_c, arm='A_peak35', q=60).max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm='A_peak35', q=60)),
         N(one(tail_c, arm='A_peak50_late800', q=50).max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm='A_peak50_late800', q=50)),
         N(one(tail_c, arm='R2b_A_peak50', q=60).max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm='R2b_A_peak50', q=60)),
         N(one(tail_c, arm='A_base', q=60).max_tail_mean_over_g2_0, 'R2C-04', fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm='A_base', q=60))))
emit('- **Share response (table 6.5b):** at q = 60 the point estimate of the paired difference moves with '
     'the share while the largest tail mean over the seeds is highest at the largest share; at q = 50 the '
     'point estimates are not monotone in the share. Whether this is a trade-off or noise at 10 seeds cannot be decided from these '
     'tables [RR-06].')
emit('- **Timing:** restricting the share-0.50 draw to the last 400 or 800 updates does not reproduce the '
     'effect of the full schedule (q = 50: %s and %s against R2b\'s %s; q = 60: %s and %s against %s) '
     '[R2C-02, R2C-09].' % (
         N(one(crit_c, arm='A_peak50_late400').mean_q50, 'R2C-02', fl(F_CRIT_C, 'mean_q50', arm='A_peak50_late400')),
         N(one(crit_c, arm='A_peak50_late800').mean_q50, 'R2C-02', fl(F_CRIT_C, 'mean_q50', arm='A_peak50_late800')),
         N(parse_ci(one(resp_c, arm='R2b_A_peak50')['q50 mean diff [95% CI]'])[0], 'R2C-09', fl(F9, 'q50 mean diff', arm='R2b_A_peak50')),
         N(one(crit_c, arm='A_peak50_late400').mean_q60, 'R2C-02', fl(F_CRIT_C, 'mean_q60', arm='A_peak50_late400')),
         N(one(crit_c, arm='A_peak50_late800').mean_q60, 'R2C-02', fl(F_CRIT_C, 'mean_q60', arm='A_peak50_late800')),
         N(parse_ci(one(resp_c, arm='R2b_A_peak50')['q60 mean diff [95% CI]'])[0], 'R2C-09', fl(F9, 'q60 mean diff', arm='R2b_A_peak50'))))
emit()
emit('**What R2c does not show.** Whether any share or timing would meet the criterion with more seeds or '
     'under a different rule; those were not tested and no arm was added [RR-06; RR-10, section 3]. The seeds '
     '40501-40520 were not used.')
emit()
emit('![R2c wave S and the R2b reference arms: paired difference of |peak error| against `A_base` per arm '
     '(diamond and bar: mean and 95% bootstrap interval, tick: median; bootstrap seed 20261005, the R2b rows '
     're-bootstrapped with it) at q = 50 (left) and q = 60 (right); negative = better. Manifest ID FG-19.]'
     '(figures/FG-19_waveS_overview.png)')
emit()

# =================================================================== 6.6
LED.section = '6.6'
emit('### 6.6 What is carried to T=3')
emit()
emit('**Decision recorded (PI publication prompt, D1).** The four rounds are closed as reported. R2c selected '
     'no arm under its pre-registered rule, so there is no v2.1 and the locked T=2 solver is v2.0. Method 5 '
     '(pathwise fine-tuning) is closed as negative at matched budgets; the censored likelihood is not '
     'adopted; the peak-focused start distribution is not adopted at T=2 (it meets the criterion at q = 50 '
     'only, and the tail constraint binds at q = 60) and is carried as a design input for the T=3 terminal '
     'stage.')
emit()
emit('**Reading the shorthand against the tables.** The table gives, for every peak-focused arm, which part '
     'of the criterion was met at which q. "Criterion" is the pre-registered pair (a) and (b); part (a) '
     'needs the interval below 0 at both q.')
emit()
emit('**Table 6.6. Criterion parts of every peak-focused arm.**')
emit()
emit('| arm | round, share and start | (a) at q = 50 | (a) at q = 60 | (b) | criterion overall | largest tail mean / '
     'e2*(0), q50 / q60 (limit 0.02) |')
emit('|---|---|---|---|---|---|---|')
rows66 = [('A_peak25', 'R2b, 0.25 from 1', crit_b, F_CRIT_B, 'R2B-02', tail_b, F_TAIL_B, 'R2B-04', 'A_peak25'),
          ('A_peak50', 'R2b, 0.50 from 1', crit_b, F_CRIT_B, 'R2B-02', tail_b, F_TAIL_B, 'R2B-04', 'A_peak50'),
          ('A_peak35', 'R2c, 0.35 from 1', crit_c, F_CRIT_C, 'R2C-02', tail_c, F_TAIL_C, 'R2C-04', 'A_peak35'),
          ('A_peak40', 'R2c, 0.40 from 1', crit_c, F_CRIT_C, 'R2C-02', tail_c, F_TAIL_C, 'R2C-04', 'A_peak40'),
          ('A_peak50_late800', 'R2c, 0.50 from 801', crit_c, F_CRIT_C, 'R2C-02', tail_c, F_TAIL_C, 'R2C-04', 'A_peak50_late800'),
          ('A_peak50_late400', 'R2c, 0.50 from 1201', crit_c, F_CRIT_C, 'R2C-02', tail_c, F_TAIL_C, 'R2C-04', 'A_peak50_late400')]
for arm, lab, cdf, cf, citem, tdf, tf, titem, key in rows66:
    cr = one(cdf, arm=arm)
    tl = {q: one(tdf, arm=key, q=q) for q in (50, 60)}
    b_txt = 'holds' if cr.b_status == 'holds' else 'violated (q = 60: seeds 10503, 10504, 10510)'
    emit('| `%s` | %s | %s | %s | %s | %s | %s / %s |' % (
        arm, lab, met(cr.a_q50), met(cr.a_q60), b_txt, str(cr.overall),
        N(tl[50].max_tail_mean_over_g2_0, titem, fl(tf, 'max_tail_mean_over_g2_0', arm=key, q=50)),
        N(tl[60].max_tail_mean_over_g2_0, titem, fl(tf, 'max_tail_mean_over_g2_0', arm=key, q=60))))
emit()
emit('Source: R2B-02 and R2C-02 (`criterion.csv`; columns `a_q50`, `a_q60`, `b_status`, `b_violations`, '
     '`overall`), R2B-04 and R2C-04 (`tail.csv`; column `max_tail_mean_over_g2_0`). R2b used bootstrap seed '
     '20261004, R2c 20261005.')
emit()
emit('Read against this table, the two halves of the shorthand "meets the criterion at q = 50 only, and '
     'the tail constraint binds at q = 60" refer to different arms:')
emit()
emit('- "Meets the criterion at q = 50 only" is the record of part (a) for the R2c arms `A_peak35`, `A_peak40` '
     'and `A_peak50_late800`; `A_peak50_late400` meets (a) at neither q. Part (b) holds for all four R2c arms, '
     'and no R2c arm meets the criterion overall [R2C-02, R2C-03].')
emit('- "The tail constraint binds at q = 60" is the record of R2b\'s `A_peak50`: it meets part (a) at both '
     'q, and the criterion is not met because part (b) is violated at q = 60 by seeds 10503, 10504 and 10510 '
     '(tail mean / e2*(0) above 0.02) [R2B-02].')
emit('- In the R2c arms the tail mean stays below 0.02 at both q (largest %s, `A_peak35`, q = 60): the tail '
     'limit is approached but not crossed, and the criterion fails there on part (a) at q = 60, where the '
     'interval contains 0 [R2C-04, R2C-02].' % N(
         one(tail_c, arm='A_peak35', q=60).max_tail_mean_over_g2_0, 'R2C-04',
         fl(F_TAIL_C, 'max_tail_mean_over_g2_0', arm='A_peak35', q=60)))
emit()
emit('What is carried is therefore the start distribution, with the measurements in tables 6.3d and 6.5b '
     '(the share of exploring starts on the peak bins and on the |d| >= 2q bins, and the resulting peak and '
     'tail statistics at T=2). The reports record no decision about a share or schedule for T=3. The '
     'implications for T=3 are in section 8.')

# ------------------------------------------------------------------ identifiers and constants quoted in text
LED.section = 'constants'
for txt, item, loc in [
        ('95', 'RR-09', '01_preregistration.md section 6: 95% percentile bootstrap interval (also RR-08, RR-10 section 3)'),
        ('10000', 'R2C-02', 'criterion.csv col n_boot (R1 and R2b: RR-08 / RR-09 section 6)'),
        ('10501-10510', 'RR-10', 'section 1: development seeds'),
        ('10501', 'RR-10', 'section 1: development seeds'), ('10510', 'R2B-02', 'criterion.csv col b_violations, A_peak50 (seed) and RR-10 section 1'),
        ('10503', 'R2B-02', 'criterion.csv col b_violations, A_peak50'), ('10504', 'R2B-02', 'criterion.csv col b_violations, A_peak50'),
        ('30510', 'R2B-15', 'numbers.json: the failed confirmation run (q=50 seed)'),
        ('30501', 'RR-03', 'confirmation seed block 30501-30520 (PL-01 / PL-02 section 9)'),
        ('30520', 'RR-03', 'confirmation seed block 30501-30520'),
        ('40501-40520', 'RR-06', 'summary: seeds not used'),
        ('30513', 'FG-13', 'figure legend: seeds shown'), ('30506', 'FG-13', 'figure legend: seeds shown'),
        ('1600', 'RR-10', 'section 2: Phase A has 1600 updates'),
        ('1200', 'RR-10', 'section 2: LR constant to update 1200'),
        ('1201', 'RR-10', 'section 2 arm table: local_first of A_peak50_late400'),
        ('801', 'RR-10', 'section 2 arm table: local_first of A_peak50_late800'),
        ('400', 'RR-10', 'section 2: A_peak50_late400 = last 400 updates'),
        ('800', 'RR-10', 'section 2: A_peak50_late800 = last 800 updates'),
        ('1225', 'RR-11', 'section 7, H2 rule: u_leave >= 1225'),
        ('25', 'RR-11', 'section 2 / R2B-15 meta.n_exports: weight exports every 25 updates'),
        ('64', 'R2B-15', 'values meta.n_exports = 64'),
        ('900', 'RR-11', 'late window updates 900-1600 (section 2.4)'),
        ('3e-5', 'RR-10', 'section 2: LR decays to 3e-5 over 1201-1600'),
        ('3e-4', 'RR-10', 'section 2: constant LR 3e-4 to update 1200'),
        ('18', 'RR-03', 'confirmation rule: at least 18 of 20 runs pass per q'),
        ('1e-12', 'R2C-10', 'blind_recomputation.txt: agreement to 1e-12'),
        ('1e-6', 'RR-11', 'section 5: D1 clamp at 1e-6 and 1 - 1e-6'),
        ('44', 'RR-10', 'section 7: 44 bins at q = 60'), ('40', 'RR-10', 'section 7: 40 bins at q = 50'),
        ('100', 'RR-10', 'section 7: 2q = 100 at q = 50'), ('120', 'RR-10', 'section 7: 2q = 120 at q = 60'),
        ('10', 'RR-10', 'section 7: bins of width 10; also n = 10 seeds per q'),
        ('20', 'RR-10', 'section 7: 20 bins with |d| >= 2q; also 20 runs per arm'),
        ('4', 'RR-10', 'section 7: the peak set is four bins'),
        ('5', 'RR-11', 'section 6: supervised floor, 5 initialisations (report only)'),
        ('300,000', 'RR-11', 'section 6: supervised floor, 300,000 steps (report only)'),
        ('1e-3', 'R2B-17', 'col floor_median = 0.00133, order of magnitude (computed here)')]:
    C(txt, item, loc)

# ------------------------------------------------------------------ write
LED.section = ''
text = '\n'.join(out) + '\n'
(OUT / 'sec06.md').write_text(text)
LED.write(OUT / 'sec06_ledger.csv')
print('words:', len(text.split()), ' ledger rows:', len(LED.rows))
print('OPEN ISSUES')
for o in OPEN_ISSUES:
    print('-', o)
