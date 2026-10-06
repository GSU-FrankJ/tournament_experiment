"""Shared helpers for the section-6 builder (W-E): data loading, number formatting, ledger."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

EV = Path('/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/'
          'reports/t2_refine_100526/evidence')
OUT = Path('/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-'
           'r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/scratchpad/'
           'report_parts')

# pack item -> path inside evidence/
P = {
    'R1-06': 'results/v2_refine/analysis/stage2_criterion.csv',
    'R1-07': 'results/v2_refine/analysis/stage2_annealing.csv',
    'R1-37': 'results/v2_refine/analysis/stage2_per_run.csv',
    'R1-40': 'results/v2_refine/analysis/stage2_arm_summary.csv',
    'R2B-01': 'results/v2_refine_r2b/analysis/decision_inputs.csv',
    'R2B-02': 'results/v2_refine_r2b/analysis/criterion.csv',
    'R2B-03': 'results/v2_refine_r2b/analysis/per_run.csv',
    'R2B-04': 'results/v2_refine_r2b/analysis/tail.csv',
    'R2B-08': 'results/v2_refine_r2b/analysis/paired.csv',
    'R2B-12': 'results/v2_refine_r2b/analysis/gate_counts.csv',
    'R2B-13': 'results/v2_refine_r2b/analysis/arm_summary.csv',
    'R2B-15': 'results/v2_refine_r2b/diag_30510/numbers.json',
    'R2B-16': 'results/v2_refine_r2b/diag_30510/tables/hypothesis_rules.csv',
    'R2B-17': 'results/v2_refine_r2b/diag_30510/tables/tab_decomposition_q50.csv',
    'R2B-18': 'results/v2_refine_r2b/diag_30510/tables/tab_decomposition_all_runs.csv',
    'R2B-21': 'results/v2_refine_r2b/diag_30510/tables/tab_eta2_every_export_q50.csv',
    'R2B-24': 'results/v2_refine_r2b/diag_30510/tables/tab_late_means_per_seed_q50.csv',
    'R2B-27': 'results/v2_refine_r2b/diag_30510/tables/tab_d1_clamp_counts_phaseA_sum.csv',
    'R2B-29': 'results/v2_refine_r2b/analysis/waveA_specifics.csv',
    'R2C-02': 'results/v2_refine_r2c/analysis/criterion.csv',
    'R2C-03': 'results/v2_refine_r2c/analysis/selection.json',
    'R2C-04': 'results/v2_refine_r2c/analysis/tail.csv',
    'R2C-06': 'results/v2_refine_r2c/launch_checks.json',
    'R2C-07': 'results/v2_refine_r2c/analysis/paired.csv',
    'R2C-08': 'results/v2_refine_r2c/analysis/selection_inputs.csv',
    'R2C-09': 'results/v2_refine_r2c/analysis/waveS_response.csv',
    'R2C-10': 'results/v2_refine_r2c/analysis/blind_recomputation.txt',
    'R2C-12': 'results/v2_refine_r2c/analysis/waveS_specifics_mean.csv',
    'R2C-01': 'results/v2_refine_r2c/analysis/per_run.csv',
}


def rd(item):
    return pd.read_csv(EV / P[item])


def rj(item):
    return json.load(open(EV / P[item]))


class Ledger:
    """Collects one row per number that appears in the prose or tables."""

    def __init__(self):
        self.rows = []
        self.section = ''

    def add(self, text, value, item, loc):
        self.rows.append((self.section, str(text), value, item, loc))

    def write(self, path):
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['statement_id', 'section', 'text', 'value', 'item_id', 'locator'])
            for i, (sec, text, val, item, loc) in enumerate(self.rows, 1):
                w.writerow(['E%04d' % i, sec, text, repr(val) if isinstance(val, float) else val,
                            item, loc])


LED = Ledger()


def g4(x):
    """Format like the round reports: 4 significant digits."""
    return format(float(x), '.4g')


def N(v, item, loc, fmt=None):
    """Format a number, log it, return the string."""
    if isinstance(v, (int, np.integer)):
        s, val = str(int(v)), int(v)
    else:
        s, val = format(float(v), fmt or '.4g'), float(v)
    LED.add(s, val, item, loc)
    return s


def CI(m, lo, hi, item, loc):
    """mean [lo, hi] with three logged numbers."""
    return '%s [%s, %s]' % (N(m, item, loc + ' (mean)'), N(lo, item, loc + ' (ci lo)'),
                            N(hi, item, loc + ' (ci hi)'))


def C(text, item, loc, value=None):
    """Log a constant or a verbatim string (thresholds, verdict words, counts of bins)."""
    LED.add(text, text if value is None else value, item, loc)
    return text
