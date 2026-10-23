"""CPU-only guards for the new capture-backed motion matrix checker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import check_motion2d
from check import color_pattern


class MotionTests(unittest.TestCase):
    def test_input_pattern_vertical_and_negative_wrap(self):
        import numpy as np
        base = color_pattern(53, 37)
        self.assertTrue(np.array_equal(color_pattern(53, 37, -3, 5), np.roll(base, (5, -3), axis=(0, 1))))

    def test_report_cannot_alias_source_or_capture_hardlink(self):
        with tempfile.TemporaryDirectory(prefix="dlss5-motion-guard-") as d:
            folder = Path(d)
            capture = folder / "capture"
            capture.mkdir()
            source = capture / "evidence.json"
            source.write_text("preserve")
            alias = folder / "outside.json"
            os.link(source, alias)
            for target in (capture / "report.json", alias, Path(check_motion2d.__file__)):
                with self.assertRaises(ValueError):
                    check_motion2d.validate_report_target(target, capture)
            self.assertEqual(source.read_text(), "preserve")
            check_motion2d.validate_report_target(folder / "safe.json", capture)

    def test_missing_capture_invalidates_old_pass_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory(prefix="dlss5-motion-missing-") as d:
            folder = Path(d)
            report = folder / "previous.json"
            report.write_text('{"passed":true}')
            result = subprocess.run([sys.executable, check_motion2d.__file__, "--capture", str(folder / "missing"),
                                     "--report", str(report)], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(report.read_text())["passed"])

    def test_invalid_threshold_invalidates_old_pass_before_gpu(self):
        with tempfile.TemporaryDirectory(prefix="dlss5-motion-threshold-") as d:
            folder = Path(d)
            report = folder / "previous.json"
            report.write_text('{"passed":true}')
            result = subprocess.run([sys.executable, check_motion2d.__file__, "--capture", str(folder / "missing"),
                                     "--report", str(report), "--min-correlation", "nan"],
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(report.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
