"""Build SCRATCH/sec07.md and SCRATCH/sec07_ledger.csv (read-only on the pack and the worktree)."""
import re
import sys

sys.path.insert(0, "/tmp/claude-1331199693/-home-fjiang4-tournament-experiment--claude-worktrees-"
                   "r2c-sampler-protocol-v2-1-c4c0f1/152f0306-5492-45a1-94d5-59493fa0d141/"
                   "scratchpad/report_parts/W-F")
from common import *  # noqa: F401,F403

L = Ledger("07")
WT = "/home/fjiang4/tournament_experiment/.claude/worktrees/t2-refine-pack/"


def N(text, value, item, locator):
    """Register a number as written."""
    return L.add(text, value, item, locator)


def text_of(item):
    with open(EV + PATH[item], errors="replace") as fh:
        return fh.read()


def must(item, sub):
    assert sub in text_of(item), (item, sub)


# ---------------------------------------------------------------- item 1: D1 flags
d1 = rd("R1-29")
a_all = d1[(d1.q.astype(str) == "all") & (d1.phase == "A")].iloc[0]
b_all = d1[(d1.q.astype(str) == "all") & (d1.phase == "B")].iloc[0]
assert a_all.M1_outcome == "exceeded" and a_all.M2_outcome == "exceeded"
assert b_all.M1_outcome == "not exceeded" and b_all.M2_outcome == "not exceeded"
P29 = PATH["R1-29"]
m1 = N(g4(a_all.M1_median_over_runs), a_all.M1_median_over_runs, "R1-29", P29 + "; q=all phase=A; M1_median_over_runs")
m1t = N(g4(a_all.M1_threshold), a_all.M1_threshold, "R1-29", P29 + "; q=all phase=A; M1_threshold")
m2n = N("%d of %d" % (a_all.M2_runs_share_above_threshold, a_all.n_runs), "3 of 20", "R1-29",
        P29 + "; q=all phase=A; M2_runs_share_above_threshold, n_runs")
m2x = N(g4(a_all.M2_max_share), a_all.M2_max_share, "R1-29", P29 + "; q=all phase=A; M2_max_share")
bg = rd("R1-36")
x = bg[(bg.group == "v11_reproduction") & (bg.phase == "A")]
inn = x[x.category == "learner_final_in"]
out_ = x[x.category == "learner_final_out"]
assert (inn.hit_sum == 0).all()
h50 = int(out_[out_.q == 50].hit_sum.iloc[0])
h60 = int(out_[out_.q == 60].hit_sum.iloc[0])
t_h50 = N("%d" % h50, h50, "R1-36", PATH["R1-36"] + "; group=v11_reproduction q=50 phase=A category=learner_final_out; hit_sum")
t_h60 = N("%d" % h60, h60, "R1-36", PATH["R1-36"] + "; group=v11_reproduction q=60 phase=A category=learner_final_out; hit_sum")
t_h0 = N("0", 0, "R1-36", PATH["R1-36"] + "; group=v11_reproduction phase=A category=learner_final_in; hit_sum (both q)")

r2b = rd("R2B-02")
rc = r2b[(r2b.arm == "A_censored")].iloc[0]
assert (not rc.a_met) and rc.b_status == "holds"
ci_c50 = N(ci(rc.mean_q50, rc.ci_mean_lo_q50, rc.ci_mean_hi_q50), "%s|%s|%s" % (rc.mean_q50, rc.ci_mean_lo_q50, rc.ci_mean_hi_q50),
           "R2B-02", PATH["R2B-02"] + "; arm=A_censored; mean_q50, ci_mean_lo_q50, ci_mean_hi_q50")
ci_c60 = N(ci(rc.mean_q60, rc.ci_mean_lo_q60, rc.ci_mean_hi_q60), "%s|%s|%s" % (rc.mean_q60, rc.ci_mean_lo_q60, rc.ci_mean_hi_q60),
           "R2B-02", PATH["R2B-02"] + "; arm=A_censored; mean_q60, ci_mean_lo_q60, ci_mean_hi_q60")

sp = rd("R2B-29")
cen = sp[sp.arm == "A_censored"]
n_noflag = int((cen.d1_flagged_L_s2 == 0).sum())
assert n_noflag == 2, n_noflag
t_nf = N("%d of %d" % (n_noflag, len(cen)), "2 of 20", "R2B-29", PATH["R2B-29"] + "; arm=A_censored; rows with d1_flagged_L_s2 == 0 / all rows")


def mean_flag(arm, q):
    v = sp[(sp.arm == arm) & (sp.q == q)].d1_flagged_L_s2.mean()
    return N(g4(v), v, "R2B-29", PATH["R2B-29"] + "; arm=%s q=%d; mean over seeds of d1_flagged_L_s2" % (arm, q))


fb50, fb60 = mean_flag("A_base", 50), mean_flag("A_base", 60)
fp50, fp60 = mean_flag("A_peak50", 50), mean_flag("A_peak50", 60)

item1 = (
    "**1. D1 clamp flags in Phase A (M1 and M2 exceeded; recorded, not fixed).** In the locked pipeline's Phase A, raw Beta draws "
    "clipped at `action_clamp` give a median clamp fraction of %s over the runs (both q pooled; threshold %s) and, in the pooled reading, %s runs "
    "with a clamped-row gradient share above 1%% (maximum %s). Phase B has no hit. All Phase-A hits are on stage-2 rows with |d| >= 2q: %s hits inside "
    "|d| < 2q, %s (q = 50) and %s (q = 60) outside [R1-29, R1-36]. Under `clamp_likelihood = density` a flagged row keeps the density of its clipped action, "
    "and the censored mode replaces it by the censored log-mass (RR-09 section 4.2); this likelihood inconsistency remains and `clamp_likelihood` stays `density` "
    "(RR-12 section 1.3, recording the PI's D1 of the R2c prompt).<br>"
    "*Where:* PL-01 `known_issues` (second entry); RR-01 headline numbers (D1) and Decisions item 7; RR-02 section 4 and Observations (D1); "
    "`reports/v2/refine/02_d1_clamp.md` (needs item: that file; not in the pack); RR-03 section 6 deviation 7.<br>"
    "*Status:* known; Phase A is unchanged in v2.0 (PL-01). The censored likelihood is not adopted (PI publication prompt, D1).<br>"
    "*Effect on results:* none recorded. The one test, R2b `A_censored` (section 2), has CIs that contain 0 at both q: %s (q = 50), %s (q = 60) [R2B-02]; "
    "%s `A_censored` runs never have a flagged row and equal their baseline exactly [R2B-29]. Observation, not a test: in R2b wave A, `A_peak50` runs show %s "
    "and %s flagged raw draws per run at q = 50 and q = 60, against %s and %s for `A_base` (means over 10 seeds, R2B-29)." % (
        m1, m1t, m2n, m2x, t_h0, t_h50, t_h60, ci_c50, ci_c60, t_nf, fp50, fp60, fb50, fb60))

# ---------------------------------------------------------------- item 2: verifier numerics
pl5 = rj("PL-05")["verifier_numerics"]
g50 = pl5["q50"]["final"]["max_abs_gap_over_dw"]
g60 = pl5["q60"]["final"]["max_abs_gap_over_dw"]
s50 = pl5["q50"]["final"]["implied_shift_over_e1_star"]
s60 = pl5["q60"]["final"]["implied_shift_over_e1_star"]
assert pl5["q50"]["final"]["gated"] is False
c2 = rd("R1-32")
rf50 = float(c2[(c2.tier == "final") & (c2.q == 50)].iloc[0]["refined_verifier (state step 0.25, 64 GL nodes) max abs diff / DW"])
rf60 = float(c2[(c2.tier == "final") & (c2.q == 60)].iloc[0]["refined_verifier (state step 0.25, 64 GL nodes) max abs diff / DW"])
P5 = PATH["PL-05"]
t_g50 = N(g4(g50), g50, "PL-05", P5 + "; verifier_numerics.q50.final.max_abs_gap_over_dw")
t_g60 = N(g4(g60), g60, "PL-05", P5 + "; verifier_numerics.q60.final.max_abs_gap_over_dw")
t_s50 = N(g4(s50), s50, "PL-05", P5 + "; verifier_numerics.q50.final.implied_shift_over_e1_star")
t_s60 = N(g4(s60), s60, "PL-05", P5 + "; verifier_numerics.q60.final.implied_shift_over_e1_star")
t_r50 = N(g4(rf50), rf50, "R1-32", PATH["R1-32"] + "; tier=final q=50; refined_verifier column")
t_r60 = N(g4(rf60), rf60, "R1-32", PATH["R1-32"] + "; tier=final q=60; refined_verifier column")
t_lim = N("1e-6", 1e-6, "R1-32", PATH["R1-32"] + "; column spec_tolerance_over_dw")
t_lim3 = N("1e-3", 1e-3, "PL-06", PATH["PL-06"] + "; Reading: the limit of 1e-3 e1* that (ii-c) applies")
item2 = (
    "**2. Verifier numerics item (standard final tier).** On the verifier's standard final tier (state step 2) its stage-1 value differs from the "
    "continuation table by %s DW at q = 50 and %s DW at q = 60. Were this a table error it would move the stage-1 optimum by %s e1* and %s e1* "
    "(limit %s e1* in (ii-c)); on the refined verifier configuration the gap is %s and %s DW (limit %s DW) [PL-05, PL-06, R1-32]. "
    "The gap is on the verifier's side (its state-grid interpolation of the stage-2 value) and bounds the precision of the verifier's own stage-1 optimum on that "
    "tier; it is not evidence about the table (PL-06, Reading).<br>"
    "*Where:* PL-01 `known_issues` (third entry); PL-06; PL-05; RR-01 Decisions item 1 (\"open as stated\") and Addendum item 3; RR-03 section 2.<br>"
    "*Status:* known; reported, not gated. The literal R1 criterion (1e-6 DW on the standard tier) was replaced by the three pre-registered tests (ii-a), (ii-b), (ii-c) "
    "in the v2.0 round (RR-01 Addendum, item 3).<br>"
    "*Effect on results:* none recorded." % (t_g50, t_g60, t_s50, t_s60, t_lim3, t_r50, t_r60, t_lim))

# ---------------------------------------------------------------- item 3: stale docstring
lock = text_of("PL-03")
assert "513fef4b304e46606fd2bc436c628c90a4893b36979a454cb00924163bb50b4a" in lock
item3 = (
    "**3. Stale module docstring of `utils/v2_continuation.py`.** The docstring predates decision D3 and still calls check (ii) open. The file is "
    "byte-identical to its version at `32a8c21` and its SHA-256 is recorded in the v2.0 lock record (PL-03, `continuation_module_sha256`), "
    "so it was not edited; `protocols/v2_T2_locked_v2_0.md` and `results/v2_T2_locked/v2_0/continuation_check_v2_0.json` (PL-05) supersede it.<br>"
    "*Where:* PL-01 `known_issues` (fourth entry); PL-03.<br>"
    "*Status:* known.<br>"
    "*Effect on results:* none recorded (documentation text only).")

# ---------------------------------------------------------------- item 4: report pack
item4 = (
    "**4. The v1.1 report pack `reports/v2/t2_report`: entries T05, T56, T57 are stale; the refresh was waived.** The pack builder does not run on the current code, for two "
    "independent causes: (1) R1's commit `32a8c21` changed `Run.lr_windows` from one window per phase to a list of windows, and `tools/v2/report/sec_locked_a._schedule` "
    "still builds one window (`TypeError: string indices must be integers`); (2) the builder calls the entry point's `build_config` with the v1.0 and v1.1 protocols, and since "
    "the v2.0 lock the entry point accepts only v2.0-shaped protocols (`KeyError: 'continuation_value_mode'`). Further modules may fail after these two. "
    "T57 cites `docs/STATE.md` by SHA-256 and that file was edited in R1 (RR-01 Decisions item 5); T56 cites `tests/test_v2_locked.py`, which the v2.0 lock replaced (item 18 below); "
    "T05 counts `reports/v2/*.md`, which R2b changed (RR-13 Addendum 2, item 4).<br>"
    "*Where:* RR-13 section 1.2 and Addendum 2 item 4; RR-12 section 1.3; RR-01 Decisions item 5; RR-03 section 6 deviation 5 and section 7 (\"Housekeeping\").<br>"
    "*Status:* waived by the PI (R2c D1, RR-12 section 1.3); the pack under `reports/v2/t2_report/` is the v1.1 pack. No repair was made.<br>"
    "*Effect on results:* none recorded (the pack is a report artifact; no run or table of the four rounds is read from it).")

# ---------------------------------------------------------------- item 5: registry test and suite counts
logs = [
    ("R1 code commit tree (`32a8c21`)", "results/v2_refine/code_pytest_full.txt"),
    ("v2.0 lock content, before the lock commit", "results/v2_T2_locked/v2_0/fullsuite_prelock.txt"),
    ("v2.0 launch tree (`f2d616c`, R7 of the re-rehearsal)", "results/v2_T2_locked/rehearsal_v2_0_pytest.txt"),
    ("R2b code commit tree (`1ff99bd`)", "results/v2_refine_r2b/code_pytest_full.txt"),
    ("R2b after the audit (`87fc8dd`)", "results/v2_refine_r2b/post_audit_pytest_full.txt"),
    ("R2c code commit tree (`58c26716`)", "results/v2_refine_r2c/code_pytest_full.txt"),
]
rx = re.compile(r"=+ (?:(\d+) failed, )?(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) xfailed)?")
rows = ["| suite | passed | failed | skipped | xfailed | log |", "|---|---|---|---|---|---|"]
parsed = {}
for label, rel in logs:
    with open(WT + rel, errors="replace") as fh:
        lines = fh.read().splitlines()
    summ = [ln for ln in lines if re.search(r"\d+ passed", ln) and " in " in ln][-1]
    mm = re.search(r"(?:(\d+) failed, )?(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) xfailed)?", summ)
    fl, ps, sk, xf = [int(v) if v else 0 for v in (mm.group(1), mm.group(2), mm.group(3), mm.group(4))]
    failed_lines = [ln for ln in lines if ln.startswith("FAILED")]
    assert failed_lines == ["FAILED tests/test_registry_canonicalization.py::test_registry_canonicalization"], failed_lines
    parsed[rel] = (ps, fl, sk, xf)
    loc = "%s (needs item: %s); last pytest summary line: %s" % (rel, rel, summ.strip())
    rows.append("| %s | %s | %s | %s | %s | `%s` (needs item) |" % (
        label, N(str(ps), ps, "(needs item: %s)" % rel, loc), N(str(fl), fl, "(needs item: %s)" % rel, loc),
        N(str(sk), sk, "(needs item: %s)" % rel, loc), N(str(xf), xf, "(needs item: %s)" % rel, loc), rel))
assert parsed["results/v2_refine/code_pytest_full.txt"] == (328, 1, 0, 4)
assert parsed["results/v2_T2_locked/rehearsal_v2_0_pytest.txt"] == (391, 1, 0, 2)
assert parsed["results/v2_refine_r2b/post_audit_pytest_full.txt"] == (486, 1, 0, 2)
assert parsed["results/v2_refine_r2c/code_pytest_full.txt"] == (567, 1, 0, 2)
suite_tab = "\n".join(rows)
fail_msg = "expected 15 Set-2 gradient runs, got 0"
must("PL-01", "expected 15 Set-2 gradient runs, got 0")
item5 = (
    "**5. The registry test fails as before.** `tests/test_registry_canonicalization.py::test_registry_canonicalization` (paper registry against the data on disk: "
    "\"%s\") fails on `main` and on the v2 branch; it is pre-existing and unrelated to v2, and test and data are unchanged (PL-01 `known_issues`, first entry). "
    "It is the only failure in every full-suite log below. The two `xfailed` of v2.0 onwards are pre-existing; the R1 suite has four because it also held the two strict "
    "expected failures of the literal check (ii), replaced in v2.0 by the tests (ii-a) to (ii-c) (RR-08 section 7; RR-03 section 1.4).<br>"
    "*Where:* PL-01; RR-01 (verification list), RR-03 section 1.4, RR-04 (what was checked), RR-06 (what was checked), RR-08 section 7, RR-09 section 8, RR-10 section 10; "
    "the logs below were read here from the worktree and are not in the pack (needs item: each path).<br>"
    "*Status:* known.<br>"
    "*Effect on results:* none recorded.\n\n%s\n\nSource: the last summary line of each log file named in the table (read from the worktree, not in the pack: "
    "needs item), consistent with the counts in RR-01, RR-03 section 1.4, RR-04, RR-09 section 8 and RR-10 section 10. Script `W-F/gen_sec07.py`." % (fail_msg, suite_tab))

# ---------------------------------------------------------------- item 6: pytest tests
must("RR-09", "a bare run also collects `experiments/*/tests`")
item6 = (
    "**6. Run `pytest tests`, not a bare `pytest`.** A bare `pytest` also collects `experiments/*/tests`, which puts a different `common` module on the path and "
    "breaks the collection of two existing test files; an early bare invocation in R2b failed that way and nothing else was affected.<br>"
    "*Where:* RR-04 Deviations and open items, item 6; RR-06 Deviations and open items, item 7; RR-09 section 8.<br>"
    "*Status:* known (usage note).<br>"
    "*Effect on results:* none recorded.")

# ---------------------------------------------------------------- accepted deviations
cf9 = text_of("CF-09")
assert "IDENTICAL" in cf9 and "the only place its checkpoint.pt exists" in cf9
chk = rj("CF-08")
assert chk["R7"]["pass"] is False and chk["ALL_PASS"] is False
item7 = (
    "**7. D-R7: R7 of the v2.0 re-rehearsal checks false as the tool computed it.** The checks tool compared the C7 candidate with the reference run through a path relative "
    "to the worktree, where the reference's gitignored `checkpoint.pt` is absent; C7 crashed with `FileNotFoundError` after the six other files compared with 0 differences, "
    "and the tool recorded `c7_identical: false`. Against the canonical reference the comparison is IDENTICAL, including `checkpoint.pt` (0 differing tensors) [CF-09]. "
    "Owner decision D-R7 (2026-10-04): R7 accepted as met. The tool was fixed after the confirmation (`3fedaa2`, reference path only) and R7 re-run: pass true. "
    "The tool's own `R7.pass` and `ALL_PASS` stay false in the record [CF-08].<br>"
    "*Where:* RR-03 section 6 deviation 1; CF-09; CF-08. Note: the header of CF-09 says the reference's `checkpoint.pt` exists \"only\" in the canonical worktree; RR-03 section 6 "
    "deviation 1 states that this overstates it (an older worktree holds a byte-identical copy).<br>"
    "*Status:* accepted (owner decision D-R7).<br>"
    "*Effect on results:* none recorded; RR-03 calls it \"a reference-path defect of the checks tool, not of the pipeline\".")

item8 = (
    "**8. The 17 R1 stage-1 re-runs.** 17 stage-1 runs started while an analysis-tool edit was uncommitted, so their manifests recorded `dirty: true`; they were moved (not deleted) to "
    "`results/v2_refine/stage1/dirty_rerun/` and re-run once from a clean tree; the analysis uses the clean re-runs. Re-runs and originals are bit-identical in the "
    "training-relevant state in 17 of 17 (`results/v2_refine/stage1_dirty_rerun_comparison.json`, needs item: that file).<br>"
    "*Where:* RR-08 Addendum 1 item 3; RR-01 Decisions item 4 and Addendum item 1; RR-01 \"What was verified\" (Runs).<br>"
    "*Status:* accepted (RR-01 Addendum, item 1).<br>"
    "*Effect on results:* none; the dirty flag \"had no effect on any result\" (RR-08 Addendum 1, item 3).")

m2_50 = d1[(d1.q.astype(str) == "50") & (d1.phase == "A")].iloc[0]
m2_60 = d1[(d1.q.astype(str) == "60") & (d1.phase == "A")].iloc[0]
q50n = N("%d of %d" % (m2_50.M2_runs_share_above_threshold, m2_50.n_runs), "2 of 10", "R1-29", P29 + "; q=50 phase=A; M2_runs_share_above_threshold, n_runs")
q60n = N("%d of %d" % (m2_60.M2_runs_share_above_threshold, m2_60.n_runs), "1 of 10", "R1-29", P29 + "; q=60 phase=A; M2_runs_share_above_threshold, n_runs")
item9 = (
    "**9. Pooled reading of the D1 flag M2.** \"More than 2 of the 20 runs\" was read as the 20 locked-pipeline runs pooled over both q. The pooled count in phase A is %s, which exceeds the "
    "limit; the per-q counts are %s (q = 50) and %s (q = 60), which do not, so the reading decides the flag's outcome [R1-29].<br>"
    "*Where:* RR-08 Addendum 1 item 2; RR-01 Decisions item 7; RR-02 section 4.<br>"
    "*Status:* accepted (RR-01 Addendum, item 1).<br>"
    "*Effect on results:* none on any run; the flag is report-only." % (m2n, q50n, q60n))

item10 = (
    "**10. Stray early copies of R1 files.** Four files of R1 (`utils/v2_continuation.py`, `agents/ppo_pathwise.py`, `tests/test_v2_refine_pathwise.py`, "
    "`tools/v2/refine_preflight.py`) were written by workers into the original session worktree before the session was switched to `v2-t2-refine`; R1 reported them as "
    "untracked duplicates and left them in place. The v2.0 round recorded them and then deleted the four untracked copies (authorised): two are byte-identical to the tracked "
    "versions at `155cdec`, two are not (`utils/v2_continuation.py`, with 1 insertion and 5 deletions against the tracked file, and `tools/v2/refine_preflight.py`). "
    "The locked module is the tracked one, whose SHA-256 is the one in the LOCK record (PL-03).<br>"
    "*Where:* RR-01 Decisions item 6; RR-03 section 6 deviation 6 (record `V20/stray_files_record.json`, needs item: that file).<br>"
    "*Status:* closed by the authorised deletion.<br>"
    "*Effect on results:* none recorded.")

item11 = (
    "**11. `main` is not fast-forwarded; local `main` was stale.** In R1, local `main` was stale (`d1b8443`, 52 commits behind) and the branch was taken from `f02a256` = `origin/main` = "
    "tag `t2-v2-main`. In R2b and R2c the fast-forward of `main` was not done: in R2b the tooling refused git on the primary checkout, the owner ran it in a terminal and "
    "`origin/main` stayed at `f02a256` while local `main` stayed at `d1b8443`; in R2c the primary checkout showed a tracked modification (` M SESSION_STATE.md`), so the step was skipped by the rule of the prompt. "
    "`v2-t2-refine`, the three tags and `v2-t2-r2b` are on `origin`; `v2-t2-r2c` was local only at the end of R2c.<br>"
    "*Where:* RR-01 Decisions item 2; RR-13 section 1.6 and Addenda 1-2; RR-12 sections 1.1-1.2; RR-04 Deviations item 2; RR-06 Deviations items 1-2 (the commands that close it are in RR-12 section 1.2).<br>"
    "*Status:* open, owner-side. Nothing in R2b or R2c depends on `main`.<br>"
    "*Effect on results:* none recorded.")

item12 = (
    "**12. Two analysis-side commits after the R1 launches.** Two commits touch `tools/v2/d1_clamp_analysis.py` and its test after the launches began (the pooled M2 reading and the "
    "headline section of the D1 report); the run code is unchanged since `32a8c21`.<br>"
    "*Where:* RR-01 Decisions item 4; RR-08 Addendum 1 item 1.<br>"
    "*Status:* accepted (RR-01 Addendum, item 1).<br>"
    "*Effect on results:* none recorded.")

item13 = (
    "**13. The R2b worktree repair.** The owner ran the `main` step of the R2b P0 in a terminal; that left the `v2-t2-r2b` worktree inside another worktree on a wrong base "
    "(`d1b8443`, none of the v2.0 code). It was repaired on request: removed (clean) and re-created from `b55d389`, `main` untouched.<br>"
    "*Where:* RR-13 Addendum (after the P0 stop); RR-04 Deviations item 2.<br>"
    "*Status:* accepted (RR-12 section 1.3).<br>"
    "*Effect on results:* none recorded; the R2b branch is based on `b55d389`.")

# H1-H4
hr = rd("R2B-16")
nt = hr[~hr.is_target]
n_h2 = int((nt.H2 == "supported").sum())
assert n_h2 == 5 and bool(nt[nt.H2 == "supported"].G_A_pass.all()) and len(nt) == 39
tg = hr[hr.is_target].iloc[0]
assert tg.label == "none of H1-H4 (early or unstable departure)"
t_h2 = N("%d" % n_h2, n_h2, "R2B-16", PATH["R2B-16"] + "; is_target=False; count of rows with H2 == supported")
t_h39 = N("%d" % len(nt), len(nt), "R2B-16", PATH["R2B-16"] + "; is_target=False; number of rows")
item14 = (
    "**14. R2b: the literal H1-H4 rules classify seed 30510 as none and fire on passing runs.** The pre-registered operational rules for the seed-30510 diagnostic (fixed before "
    "any diagnostic output existed) are all \"not supported\" for q = 50 seed 30510 (label \"none of H1-H4 (early or unstable departure)\"), and H2 is \"supported\" for %s of the other %s "
    "confirmation runs, all of which passed G-A [R2B-16]. The rules were not changed; the descriptive readings are labelled post hoc in the diagnostic report.<br>"
    "*Where:* RR-04 Deviations item 3 and \"P2\"; RR-11 sections 7, 7.1 and 8; RR-09 section 5.<br>"
    "*Status:* known; the diagnostic stays descriptive (RR-12 section 1.3).<br>"
    "*Effect on results:* none on the pilots; the diagnostic does not discriminate seed 30510 from passing runs by the literal rules." % (t_h2, t_h39))

item15 = (
    "**15. R2b: two names for the code commit, commit order, and commit subjects.** Reports call `1ff99bd` the code commit (the run code); the pilot launch records and "
    "`tools/v2/r2b_launch_checks.py` use `0fa2196`, the last commit before the launches that touches anything outside `results/` and `reports/v2/refine_r2b/`; the run code is "
    "identical (`git diff --stat 1ff99bd 0fa2196 -- run agents utils envs protocols` is empty). Pilot manifests record `d581b3c`. The analysis tool was committed after the runs, as in R1. "
    "Six of the 17 commits between `b55d389` and `87fc8dd` have subjects longer than 72 characters; they were not reworded because they are cited by hash.<br>"
    "*Where:* RR-04 Deviations item 4; RR-09 Addendum 2 item 3; RR-12 section 1.3.<br>"
    "*Status:* known; the over-long subjects stay (PI, R2c D1, RR-12 section 1.3).<br>"
    "*Effect on results:* none recorded.")

item16 = (
    "**16. R2b: \"a missing reference file stops the analysis\" was built as record-and-continue.** The pre-registration says a missing reference file stops the analysis; "
    "`tools/v2/refine_r2b_analysis.py` records an incomplete or reference row with an anomaly text and continues. In R2b nothing was missing: 200 of 200 analysed rows are "
    "complete or reference rows and the anomalies column is empty in 200 of 200 (`results/v2_refine_r2b/analysis/per_run.csv`).<br>"
    "*Where:* RR-09 Addendum 2 item 1.<br>"
    "*Status:* known.<br>"
    "*Effect on results:* none recorded.")
pr = rd("R2B-03")
assert len(pr) == 200 and (pr.anomalies.fillna("") == "").all()
t200 = N("200 of 200", 200, "R2B-03", PATH["R2B-03"] + "; rows with empty anomalies / all rows")
item16 = item16.replace("200 of 200 analysed rows are complete or reference rows and the anomalies column is empty in 200 of 200", "200 of 200 analysed rows are complete or reference rows and the anomalies column is empty in %s [R2B-03]" % t200)

item17 = (
    "**17. R2c: tool-commit order and the commit named in the manifests.** The analysis tool and the launch-check tool are in the code commit `58c26716`, before any run, and were "
    "validated on R2b data; in R1 and R2b the analysis tool was committed after the runs. The wave-S manifests record the results-only commit `3ad1b07d`, not `58c26716`; "
    "the launch-check tool accepts a manifest commit whose diff from the code commit touches only `results/` and `reports/v2/refine_r2c/`, and the launch record gives both hashes.<br>"
    "*Where:* RR-06 Deviations item 3; RR-10 section 1 (\"Choice (order)\") and Addendum 1 item 1; R2C-06 (`launch_checks.json`).<br>"
    "*Status:* known (a choice recorded in the pre-registration).<br>"
    "*Effect on results:* none recorded; 80 of 80 runs pass the post-launch checks [R2C-06].")
lc = rj("R2C-06")["summary"]
assert lc["n_runs_ok"] == 80 and lc["n_runs"] == 80
t80 = N("80 of 80", 80, "R2C-06", PATH["R2C-06"] + "; summary.n_runs_ok / n_runs")
item17 = item17.replace("80 of 80 runs pass", "%s runs pass" % t80)

item18 = (
    "**18. The renamed test file.** The v2.0 lock commit `1d6d4d0` deleted `tests/test_v2_locked.py` and added `tests/test_v2_locked_v2_0.py`; all 15 v1.1 test functions are ported "
    "(13 keep their name, 2 renamed) and the v2.0 file has 34. The report-pack builder still reads the old path (`tools/v2/report/sec_stage1.py`), which is one cause of the stale entry T56 "
    "(item 4).<br>"
    "*Where:* RR-03 section 6 deviation 5 and section 1.4; RR-13 section 1.2 (step 1).<br>"
    "*Status:* accepted (RR-03 section 6: deviations accepted by the owner unless marked otherwise).<br>"
    "*Effect on results:* none recorded.")

item19 = (
    "**19. R2c: two existing R2b tests were edited, and an informational difference at the late arms' branch point.** `tests/test_v2_r2b_gaps.py::test_phase_loops_draw_their_starts_through_run_draw_starts` "
    "(its stub of `Run.draw_starts` gains the parameter `local`) and `tests/test_v2_r2b.py::test_r2b_waves_do_not_change_the_r1_launcher_tables` (the set of `WAVE_DIR` keys gains the R2c "
    "waves) were edited because R2c changes those things on purpose. Separately, the three process-global RNG states in `state_u01200.pt` of `A_peak50_late400` differ from `parents_A`'s in 20 of 20 runs "
    "(the stagewise runner does not seed them, and the training-relevant-state definition excludes them; the launch checks report them as informational).<br>"
    "*Where:* RR-06 Deviations item 4 and \"What was checked\"; RR-10 section 1.<br>"
    "*Status:* known.<br>"
    "*Effect on results:* none recorded; the D2 prefix identities of the training-relevant state hold in 20 of 20 for each late arm (RR-06).")

item20 = (
    "**20. Session and tool housekeeping.** The v2.0 round's sessions started in other worktrees and entered `v2-t2-refine` (RR-03 section 6 deviation 2); the R2c session started in a harness "
    "worktree and switched into `v2-t2-r2c` (RR-06 Deviations item 5). The v2.0 lock added an opt-in `--exclude` option to `tools/v2/seed_inventory.py` so that the inventory does not count the protocol's own "
    "declaration of the seed block (stock scan: 20 hits, all from `protocols/v2_T2_locked_v2_0.json`; with the exclusion: 0 collisions) (RR-03 section 6 deviation 4). Several record files of the v2.0 round were "
    "written by short scripts that are not in the repository, and the pre-lock review ledger is generated from two workflow journals that are not in the repository (RR-03 section 7 and section 6, 'Other anomalies'). "
    "The R2c launch-check tool's hidden options can shrink the check; the pilot was checked with the defaults (RR-10 section 10).<br>"
    "*Status:* known; accepted with the other v2.0 deviations (RR-03 section 6 preamble).<br>"
    "*Effect on results:* none recorded.")

item21 = (
    "**21. One failed run in the v2.0 confirmation.** q = 50 seed 30510 has outcome `stage2_failure` (G-A eta_2/DW above 0.005; see section 2) and was not re-run: D5 re-runs only an infrastructure kill. "
    "It is reported as computed in the pass count 19 / 20 [CF-02, CF-03].<br>"
    "*Where:* RR-03 verdict and section 6 'Other anomalies'; RR-04 (P2, diagnostic).<br>"
    "*Status:* known; the confirmation verdict is PASS (rule >= 18 of 20 per q).<br>"
    "*Effect on results:* it is part of the result.")

item22 = (
    "**22. Check (ii): the table's lookup error between nodes is not covered.** The tests (ii-a) to (ii-c) cover the node values of the continuation table; none covers the interpolation error "
    "of the step-0.05 linear lookup between nodes, so \"converged\" refers to the node values. The locked texts were not changed to say so.<br>"
    "*Where:* RR-03 section 2 (note on test T7) and section 6 deviation 3 (finding T7, \"not_changed\").<br>"
    "*Status:* known.<br>"
    "*Effect on results:* none recorded; no between-node error figure appears in the check record.")
must("RR-03", "do not cover the interpolation error")

# ---------------------------------------------------------------- limitations
b = rd("R2B-02")
c = rd("R2B-08")
c2 = rd("R2C-07")
a = c[(c.arm == "A_peak50") & (c.q == 50) & (c.metric == "stage2_peak_rel_err_abs")]
bb = c2[(c2.arm == "R2b_A_peak50") & (c2.q == 50) & (c2.metric == "stage2_peak_rel_err_abs")]
assert len(a) == 1 and len(bb) == 1
a = a.iloc[0]
bb = bb.iloc[0]
assert abs(a["mean"] - bb["mean"]) < 1e-12
ci_a = N(ci(a["mean"], a.ci_mean_lo, a.ci_mean_hi), "%s|%s|%s" % (a["mean"], a.ci_mean_lo, a.ci_mean_hi), "R2B-08",
         PATH["R2B-08"] + "; arm=A_peak50 q=50 metric=stage2_peak_rel_err_abs; mean, ci_mean_lo, ci_mean_hi")
ci_b = N(ci(bb["mean"], bb.ci_mean_lo, bb.ci_mean_hi), "%s|%s|%s" % (bb["mean"], bb.ci_mean_lo, bb.ci_mean_hi), "R2C-07",
         PATH["R2C-07"] + "; arm=R2b_A_peak50 q=50 metric=stage2_peak_rel_err_abs; mean, ci_mean_lo, ci_mean_hi")
item23 = (
    "**23. Limitations (descriptive).**<br>"
    "(a) Every pilot has n = 10 seeds per q, so the intervals are wide; an interval that contains 0 does not show that a mechanism has no effect (RR-02, RR-05, RR-06).<br>"
    "(b) The pilots use the development seeds 10501-10510 only; each protocol had one fresh-seed confirmation (v1.1: 20501-20520, v2.0: 30501-30520), and the reserved seeds 40501-40520 were not used "
    "in R2b or R2c (RR-04, RR-06).<br>"
    "(c) The closed-form equilibrium enters evaluation only, never training; q is 50 and 60 only (RR-03, protocol PL-01).<br>"
    "(d) The criterion is descriptive and not a gate (RR-02, RR-05, RR-09 section 6); R2c applied its selection rule mechanically and selected no arm (R2C-03).<br>"
    "(e) The bootstrap seed differs by round, so the same data give slightly different intervals: R2b `A_peak50` at q = 50 is %s with seed 20261004 and %s with seed 20261005 "
    "(point estimate identical) [R2B-08, R2C-07].<br>"
    "(f) Wall time depends on machine load (R2b waves A and P ran together, up to 40 processes); only optimiser-step counts are comparable across arms (RR-05 header). "
    "The logged per-update loss and FOC curves of the R2b pathwise arms and of R1's `A_detmean` are on different time bases (RR-09 section 4.3)." % (ci_a, ci_b))

items = [item1, item2, item3, item4, item5, item6, item7, item8, item9, item10, item11, item12, item13, item14,
         item15, item16, item17, item18, item19, item20, item21, item22, item23]

lead = (
    "No round report states that any item below changed a reported number. Status words: \"accepted\" = an acceptance is recorded (the source is named); "
    "\"known\" = recorded in the round reports, and the rounds are closed as reported (PI publication prompt, D1); \"open\" = an action is still pending. "
    "\"Effect on results: none recorded\" means that no round report states an effect. Items 1-6 are known issues, items 7-22 deviations, item 23 limitations.")

parts = ["## 7. Known issues, limitations and deviations\n", lead + "\n"]
parts.append("### Known issues\n")
for it in items[:6]:
    parts.append(it + "\n")
parts.append("### Accepted and recorded deviations\n")
for it in items[6:22]:
    parts.append(it + "\n")
parts.append("### Limitations\n")
parts.append(items[22] + "\n")
text = "\n".join(parts)

# settings and identifiers quoted in the text (report only; each string occurs in the text)
SET = [
    ("1%", 0.01, "RR-08", "reports/v2/refine/01_preregistration.md section 6: M2 gradient-share threshold"),
    ("|d| >= 2q", "2q", "RR-08", "reports/v2/refine/01_preregistration.md section 6 / RR-01 headline D1: hits on rows with |d| >= 2q of the final stage"),
    ("state step 2", 2, "PL-06", "results/v2_T2_locked/v2_0/verifier_numerics_note.md table: final tier state step 2"),
    ("17 stage-1 runs", 17, "RR-08", "reports/v2/refine/01_preregistration.md Addendum 1 item 3"),
    ("17 of 17", "17/17", "RR-08", "reports/v2/refine/01_preregistration.md Addendum 1 item 3: bit-identical 17/17 (record file not in the pack)"),
    ("52 commits behind", 52, "RR-01", "reports/v2/refine/summary.md Decisions item 2"),
    ("six other files", 6, "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 1"),
    ("0 differing tensors", 0, "CF-09", "results/v2_T2_locked/rehearsal_v2_0_checks/c7_canonical_reference.txt line 'checkpoint.pt: 0 differing tensors'"),
    ("all 15 v1.1 test functions", 15, "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 5"),
    ("13 keep their name, 2 renamed", "13,2", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 5"),
    ("the v2.0 file has 34", 34, "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 5"),
    ("1 insertion and 5 deletions", "1,5", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 6 (stray files table, git diff --stat)"),
    ("stock scan: 20 hits", 20, "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 4"),
    ("0 collisions", 0, "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 4"),
    ("Six of the 17 commits", "6 of 17", "RR-04", "reports/v2/refine_r2b/summary.md Deviations item 4"),
    ("72 characters", 72, "RR-04", "reports/v2/refine_r2b/summary.md Deviations item 4 (project rule)"),
    ("step-0.05", 0.05, "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 2 (note on test T7)"),
    ("up to 40 processes", 40, "RR-05", "reports/v2/refine_r2b/05_decision_inputs.md header"),
    ("20 of 20 runs", 20, "RR-06", "reports/v2/refine_r2c/summary.md 'What was checked' (RNG states informational)"),
    ("0.005", 0.005, "PL-01", "protocols/v2_T2_locked_v2_0.json gates.G-A eta_T_over_dw threshold"),
    ("19 / 20", "19/20", "CF-02", PATH["CF-02"] + " q=50 n_pass/n_expected"),
    ("rule >= 18 of 20", 18, "PL-01", "protocols/v2_T2_locked_v2_0.json confirmation rule"),
    ("2026-10-04", "2026-10-04", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 1 (D-R7)"),
    ("seed 30510", 30510, "CF-03", PATH["CF-03"] + " row q=50 seed=30510"),
    ("20501-20520", "20501-20520", "CF-10", PATH["CF-10"] + " (v1.1 confirmation seeds)"),
    ("30501-30520", "30501-30520", "PL-01", "protocols/v2_T2_locked_v2_0.json confirmation.seed_block"),
    ("10501-10510", "10501-10510", "PL-01", "protocols/v2_T2_locked_v2_0.json development_seeds"),
    ("40501-40520", "40501-40520", "RR-04", "reports/v2/refine_r2b/summary.md (stay reserved); RR-06 summary (not used)"),
    ("n = 10 seeds per q", 10, "R1-02", PATH["R1-02"] + " column n_pairs_q50"),
    ("two strict expected failures", 2, "RR-08", "reports/v2/refine/01_preregistration.md section 7 (the 4 xfailed)"),
    ("four because", 4, "RR-08", "reports/v2/refine/01_preregistration.md section 7"),
    ("0.05 step", 0.05, "RR-03", "n/a"),
    ("`32a8c21`", "32a8c21", "RR-01", "reports/v2/refine/summary.md header"),
    ("`155cdec`", "155cdec", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 6 table"),
    ("`1d6d4d0`", "1d6d4d0", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 1.5"),
    ("`f2d616c`", "f2d616c", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 1.5"),
    ("`3fedaa2`", "3fedaa2", "RR-03", "reports/v2/protocol_v2_0_confirmation.md section 6 deviation 1"),
    ("`1ff99bd`", "1ff99bd", "RR-04", "reports/v2/refine_r2b/summary.md"),
    ("`0fa2196`", "0fa2196", "RR-09", "reports/v2/refine_r2b/01_preregistration.md Addendum 2 item 3"),
    ("`d581b3c`", "d581b3c", "RR-04", "reports/v2/refine_r2b/summary.md Deviations item 4"),
    ("`87fc8dd`", "87fc8dd", "RR-04", "reports/v2/refine_r2b/summary.md 'What was checked before the pilots'"),
    ("`b55d389`", "b55d389", "RR-13", "reports/v2/refine_r2b/00_housekeeping.md Addendum 2 item 1"),
    ("`58c26716`", "58c26716", "RR-10", "reports/v2/refine_r2c/01_preregistration.md Addendum 1 item 1"),
    ("`3ad1b07d`", "3ad1b07d", "RR-10", "reports/v2/refine_r2c/01_preregistration.md Addendum 1 item 1"),
    ("`d1b8443`", "d1b8443", "RR-01", "reports/v2/refine/summary.md Decisions item 2"),
    ("`f02a256`", "f02a256", "RR-01", "reports/v2/refine/summary.md Decisions item 2"),
]
SET = [t for t in SET if t[0] != "0.05 step"]

for t, v, it, loc in SET:
    L.add(t, v, it, loc + " (quoted from the report, not a result)")
bad = [r for r in L.rows if r[2] not in text]
print("LEDGER TEXTS NOT FOUND IN TEXT:", [(r[0], r[2]) for r in bad])
with open(os.path.join(SCRATCH, "sec07.md"), "w") as fh:
    fh.write(text)
L.save("sec07_ledger.csv")
print(text)
print("ledger rows:", len(L.rows))
