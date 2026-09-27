"""Focused reporting regressions; fixtures exercise files, not mocked readers."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
try:
    import summarize_restarts as report
except ModuleNotFoundError:
    report = None


class RestartReportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(report, "Restart reporting implementation is missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, seed=1, candidate=True, final_pass=True):
        d = self.root / str(seed)
        d.mkdir()
        rec = {"seed": seed, "run": f"run{seed}", "q": 50, "T": 2,
               "protocol": {"phase_caps": {"C": 1000}}, "output_dir": str(d)}
        v = {"phase": "C", "update": 2, "local": 100 if candidate else 1000,
             "valid": True, "criterion_value_over_dw": .009 if candidate else .02,
             "concentration": {"valid": True, "max_std_norm": .03},
             "br_pass": candidate, "conc_pass": True, "eligible": candidate}
        final_dreach = .0095 if final_pass else .012
        f = {"smoke": False, "tiers": {
                "development": {"valid": True, "dreach_over_dw": .009, "exp_root_over_dw": .003},
                "final": {"valid": True, "dreach_over_dw": final_dreach, "exp_root_over_dw": .003}},
             "pass_flags": {"overall_pass": final_pass,
                            "refine_dreach_diff_over_dw": abs(final_dreach - .009),
                            "refine_exp_diff_over_dw": 0},
             "concentration": {"C_all": {"valid": True, "max_std_norm": .03}},
             "stopping_record": {"stop_update": 2, "phase_C_local_at_stop": v["local"],
                    "development_stopping_criterion_satisfied": candidate,
                    "reason": "k_stop_passes" if candidate else "budget_exhausted",
                    "total_episodes": 1024, "total_transitions": 1536},
             "costs": {"total_updates": 2, "training_wall_sec": 8,
                       "final_eval_sec": 2, "total_wall_sec": 10,
                       "total_process_cpu_sec": 9},
             "recovery": {"stage1": {"abs_error": seed}}}
        data = {"status.json": {"state": "done"}, "config.json": {"record": rec, "smoke": False},
                "train_history.json": {"history": [{"update": 1}, {"update": 2}], "verifier_calls": [v]},
                "final_eval.json": f}
        for name, value in data.items():
            (d / name).write_text(json.dumps(value))
        return rec, {"seed": seed, "run": rec["run"], "returncode": 0, "wall_sec": 11}

    def inspect(self, seed=1, candidate=True, final_pass=True):
        rec, launch = self.fixture(seed, candidate, final_pass)
        return report.read_run(rec, launch)

    def manifest(self, seeds):
        return {"runs": [{"seed": s} for s in seeds],
                "restart_evaluation": {"groups": [
                    {"group_id": i // 3 + 1, "seeds": seeds[i:i+3]}
                    for i in range(0, len(seeds), 3)]}}

    def test_candidate_rejected_by_final_is_not_success(self):
        row = self.inspect(final_pass=False)
        self.assertTrue(row["candidate"])
        self.assertFalse(row["joint_certified"])
        self.assertEqual(row["certification"], "failed")

    def test_no_candidate_terminal_pass_is_only_diagnostic(self):
        row = self.inspect(candidate=False)
        self.assertFalse(row["candidate"])
        self.assertFalse(row["joint_certified"])
        self.assertEqual(row["certification"], "not_applicable_no_candidate")
        self.assertTrue(row["terminal_final_overall_pass"])

    def test_all_failures_have_zero_success_and_na_conditional(self):
        rows = [self.inspect(s, candidate=False) for s in (1, 2, 3)]
        result = report.summarize(self.manifest([1, 2, 3]), rows)
        self.assertEqual(result["single_run"]["end_to_end"]["k"], 0)
        self.assertEqual(result["single_run"]["end_to_end"]["n"], 3)
        self.assertIsNone(result["single_run"]["conditional_certification"]["rate"])
        report.write_outputs(self.root, result)
        self.assertEqual(len((self.root / "runs.csv").read_text().splitlines()), 4)
        self.assertTrue((self.root / "REPORT.md").exists())

    def test_missing_and_infrastructure_failure_stay_in_denominator(self):
        rec, launch = self.fixture(1)
        launch["returncode"] = 1
        errored_candidate = report.read_run(rec, launch)
        missing = report.read_run({"seed": 2, "run": "missing", "q": 50,
                                   "output_dir": str(self.root / "missing")}, None)
        rows = [errored_candidate, missing, self.inspect(3)]
        result = report.summarize(self.manifest([1, 2, 3]), rows)
        self.assertEqual(result["single_run"]["end_to_end"]["n"], 3)
        self.assertEqual(result["single_run"]["end_to_end"]["k"], 1)
        self.assertEqual(result["operation_errors"], 2)
        self.assertTrue(errored_candidate["candidate"])
        self.assertFalse(errored_candidate["joint_certified"])
        self.assertEqual(result["decision"], "inconclusive_operation_errors")

    def test_fixed_groups_are_not_sorted_by_outcome_or_input_row_order(self):
        rows = [self.inspect(3), self.inspect(1, candidate=False), self.inspect(2)]
        result = report.summarize(self.manifest([1, 2, 3]), rows)
        group = result["groups"][0]
        self.assertEqual(group["seeds"], [1, 2, 3])
        self.assertEqual(group["first_joint_success_position"], 2)
        self.assertFalse(group["k1_joint"])
        self.assertTrue(group["k2_joint"])
        self.assertEqual(result["rescued_groups"], 1)

    def test_actual_all_run_cost_is_distinct_from_simulated_first_success(self):
        rows = [self.inspect(1), self.inspect(2), self.inspect(3)]
        result = report.summarize(self.manifest([1, 2, 3]), rows)
        self.assertEqual(result["actual_evaluation_costs"]["total_wall_seconds"]["sum"], 30)
        self.assertEqual(result["by_k"]["3"]["simulated_costs"]["total_wall_seconds"]["sum"], 10)
        self.assertEqual(result["by_k"]["3"]["simulated_attempts"], 1)
        self.assertEqual(result["actual_runs"], 3)

    def test_dense_concentration_failure_is_not_joint_success(self):
        rec, launch = self.fixture()
        p = Path(rec["output_dir"]) / "final_eval.json"
        f = json.loads(p.read_text())
        f["concentration"]["C_all"]["max_std_norm"] = .041
        p.write_text(json.dumps(f))
        self.assertFalse(report.read_run(rec, launch)["joint_certified"])

    def test_nonfirst_candidate_and_config_mismatch_are_explicit_errors(self):
        rec, launch = self.fixture()
        p = Path(rec["output_dir"]) / "train_history.json"
        h = json.loads(p.read_text())
        v = copy.deepcopy(h["verifier_calls"][0])
        v["update"] = 1
        h["verifier_calls"].insert(0, v)
        p.write_text(json.dumps(h))
        self.assertEqual(report.read_run(rec, launch)["operation"], "error")
        rec2, launch2 = self.fixture(2)
        p = Path(rec2["output_dir"]) / "config.json"
        cfg = json.loads(p.read_text())
        cfg["record"]["q"] = 60
        p.write_text(json.dumps(cfg))
        self.assertEqual(report.read_run(rec2, launch2)["operation"], "error")


    def test_numerically_invalid_final_is_reportable_failure_not_nan_json(self):
        rec, launch = self.fixture(1)
        p = Path(rec["output_dir"]) / "final_eval.json"
        f = json.loads(p.read_text())
        f["tiers"]["final"]["valid"] = False
        f["tiers"]["final"]["dreach_over_dw"] = float("nan")
        f["pass_flags"]["overall_pass"] = False
        p.write_text(json.dumps(f))
        row = report.read_run(rec, launch)
        self.assertTrue(row["candidate"])
        self.assertFalse(row["joint_certified"])
        result = report.summarize(self.manifest([1, 2, 3]), [row, self.inspect(2), self.inspect(3)])
        report.write_outputs(self.root, result)
        self.assertIsNone(json.loads((self.root / "summary.json").read_text())["rows"][0]["final_dreach"])

    def test_missing_final_keeps_observed_candidate_in_conditional_denominator(self):
        rec, launch = self.fixture(1)
        (Path(rec["output_dir"]) / "final_eval.json").unlink()
        row = report.read_run(rec, launch)
        result = report.summarize(self.manifest([1, 2, 3]),
                                  [row, self.inspect(2, candidate=False), self.inspect(3, candidate=False)])
        self.assertEqual(result["single_run"]["conditional_certification"]["n"], 1)
        self.assertEqual(result["single_run"]["conditional_certification"]["k"], 0)
        self.assertEqual(result["operation_errors"], 1)

    def test_invalid_final_tier_without_metrics_is_certification_failure(self):
        rec, launch = self.fixture(1)
        p = Path(rec["output_dir"]) / "final_eval.json"
        f = json.loads(p.read_text())
        f["tiers"]["final"] = {"valid": False, "error": "numerical failure", "time_sec": .1}
        f["pass_flags"]["overall_pass"] = False
        p.write_text(json.dumps(f))
        row = report.read_run(rec, launch)
        self.assertEqual(row["operation"], "done")
        self.assertEqual(row["certification"], "failed")
        self.assertFalse(row["joint_certified"])

    def test_finished_failed_batch_is_reported_with_all_scheduled_denominators(self):
        runs, launches = zip(*(self.fixture(seed) for seed in range(10201, 10231)))
        launches[0].clear()
        launches[0].update(run=runs[0]["run"], returncode=None, error="launcher could not open log")
        manifest = self.manifest(list(range(10201, 10231)))
        manifest["runs"] = list(runs)
        manifest["source_manifest"] = str(self.root / "source.json")
        (self.root / "source.json").write_text(json.dumps({"runs": [runs[0]]}))
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "launch_status.json").write_text(json.dumps({"state": "failed", "runs": launches}))
        saved_root = report.ROOT
        try:
            report.ROOT = self.root
            with contextlib.redirect_stdout(io.StringIO()):
                report.main()
        finally:
            report.ROOT = saved_root
        self.assertTrue((self.root / "summary.json").exists())
        result = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(result["single_run"]["end_to_end"]["n"], 30)
        self.assertEqual(result["operation_errors"], 1)
        self.assertEqual(result["decision"], "inconclusive_operation_errors")

if __name__ == "__main__":
    unittest.main(verbosity=2)
