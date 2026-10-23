"""CPU tests for temporal relations and false-pass prevention, without a GPU."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import check_temporal
import reference


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dlss5-temporal-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        rows = []
        colors = {"fresh_a": 10, "fresh_b": 50, "fresh_b_f1": 60, "static_reset_f2": 10,
                  "cut_reset_f2": 50, "cut_reset_f3": 60, "cut_reset_f8": 50}
        for case in reference.suite_cases("temporal"):
            folder = self.folder / case["id"]
            folder.mkdir()
            output = folder / "nr-lab-output-model1-srgb.ppm"
            output.write_bytes(b"P6\n2 1\n255\n" + bytes([colors.get(case["id"], 30)]) * 6)
            rows.append({"id": case["id"], "outputs": {output.name: reference.digest(output)}})
        self.manifest = {"suite": "temporal", "reference_kind": "port-sm86", "runtime_sha256": {}, "cases": rows}
        reference.write_json(self.folder / "manifest.json", self.manifest)

    def test_reset_relations_and_distinct_controls_pass(self):
        report = check_temporal.analyze(self.folder, self.manifest)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["checks"]), 6)

    def test_reset_regression_remains_a_failure(self):
        path = self.folder / "cut_reset_f2" / "nr-lab-output-model1-srgb.ppm"
        path.write_bytes(b"P6\n2 1\n255\n" + bytes([51]) * 6)
        report = check_temporal.analyze(self.folder, self.manifest)
        self.assertFalse(report["passed"])
        self.assertFalse(next(c for c in report["checks"] if c["name"] == "cut_reset_first")["passed"])

    def test_identical_outputs_cannot_vacuously_pass(self):
        for case in self.manifest["cases"]:
            (self.folder / case["id"] / next(iter(case["outputs"]))).write_bytes(b"P6\n2 1\n255\n" + bytes(6))
        report = check_temporal.analyze(self.folder, self.manifest)
        self.assertFalse(report["passed"])
        self.assertEqual([c["name"] for c in report["checks"] if not c["passed"]],
                         ["distinct_scenes", "reset_has_effect"])

    def test_missing_capture_overwrites_previous_success(self):
        report = self.folder / "previous.json"
        reference.write_json(report, {"passed": True})
        result = subprocess.run([sys.executable, str(Path(check_temporal.__file__)),
                                 "--capture", str(self.folder / "missing"), "--report", str(report)],
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(report.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
