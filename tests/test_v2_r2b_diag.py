"""Tests of the seed-30510 diagnostic's pre-registered H1-H4 rules (tools/v2/diag_seed30510.py).

The rules are the ones of ``reports/v2/refine_r2b/01_preregistration.md`` section 5, applied literally to
synthetic trajectories whose classification is known by construction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("diag_seed30510", ROOT / "tools" / "v2" / "diag_seed30510.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["diag_seed30510"] = D
_spec.loader.exec_module(D)

US = np.arange(25, 1601, 25)
N = len(US)
# 19 pack runs with constant S = -0.06 + 0.001 i (p10 = -0.0582, p90 = -0.0418 at every export) and L = S + 0.001
S_PACK = np.tile((-0.06 + 0.001 * np.arange(19))[:, None], (1, N))
L_PACK = S_PACK + 0.001
P10 = float(np.percentile(S_PACK[:, 0], 10))


def traj(high_until=None, high=-0.03, low=-0.2):
    """S that is ``high`` (in the band) at the exports u <= ``high_until`` and ``low`` (out of it) afterwards."""
    s = np.full(N, low)
    if high_until is not None:
        s[US <= high_until] = high
    return s


def rules(s_t, l_t=None):
    return D.apply_rules(US, s_t, s_t + 0.001 if l_t is None else l_t, S_PACK, L_PACK)


def test_pack_band_edge_is_the_linear_percentile():
    assert P10 == pytest.approx(-0.0582)


def test_h1_peak_never_rose():
    r = rules(traj())                                  # always far below the pack's end level
    assert r["H1"] and not r["H2"] and not r["H3"] and r["out_S_1600"] and r["u_leave"] == 25
    assert r["verdict_H1"] == "supported" and r["verdict_H2"] == "not supported"


def test_h1_is_the_maximum_over_all_exports_including_the_transient():
    s = traj()
    s[3] = P10 + 1e-6                                  # one early export touches the pack's lower edge
    assert not rules(s)["H1"]


def test_h2_decay_in_the_lr_window():
    r = rules(traj(high_until=1200))                   # in the band through 1200, out from 1225 on
    assert r["u_leave"] == 1225 and r["H2"] and not r["H1"] and not r["H3"]


def test_h3_late_excursion_after_a_plateau():
    r = rules(traj(high_until=600))                    # 24 in-band exports, then out from 625
    assert r["u_leave"] == 625 and r["plateau_len"] == 24 and r["H3"] and not r["H2"] and not r["H1"]


def test_short_plateau_is_none_of_h1_h4():
    r = rules(traj(high_until=100))                    # 4 in-band exports only
    assert r["plateau_len"] == 4 and not any(r[h] for h in ("H1", "H2", "H3", "H4"))
    assert r["label"] == "none of H1-H4 (early or unstable departure)"


def test_plateau_of_exactly_sixteen_exports_counts():
    assert rules(traj(high_until=16 * 25))["H3"] and not rules(traj(high_until=15 * 25))["H3"]


def test_not_out_at_the_end_supports_none():
    r = rules(traj(high_until=1600))
    assert not r["out_S_1600"] and r["u_leave"] is None and not any(r[h] for h in ("H1", "H2", "H3", "H4"))


def test_h4_needs_the_peak_height_in_the_band_and_a_rounded_cusp():
    s = traj()
    r = rules(s, l_t=np.full(N, -0.03))                # height at the pack's level, S far below: cusp 0.17
    assert r["H4"] and r["cusp_exceeds_p90"]
    assert not rules(s, l_t=s + 0.001)["H4"]           # height low too
    l_late_low = np.full(N, -0.03)
    l_late_low[US == 1300] = -0.5                      # one export of the window below the pack's L band
    assert not rules(s, l_t=l_late_low)["H4"]
    assert not rules(s, l_t=s + 0.0005)["H4"] and not rules(s, l_t=np.full(N, -0.03) * 0 + P10 - 1e-3)["H4"]


def test_missing_input_is_undetermined():
    s = traj()
    s[10] = np.nan
    r = rules(s)
    assert r["inputs_ok"] is False and all(r[f"verdict_{h}"] == "undetermined" for h in ("H1", "H2", "H3", "H4"))
