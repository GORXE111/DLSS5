"""CPU checks for acceptance failures, missing references and explicit partial coverage."""
import json
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from acceptance import Report, execute, metrics, parser, ppm


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = parser("test").parse_args(["--reference-dir", str(self.root), "--report", str(self.root / "report.json")])

    def test_cli_rejects_nonfinite_and_out_of_range_thresholds(self):
        for flag in ("--min-correlation", "--max-mae", "--max-nrmse"):
            for value in ("nan", "NaN", "inf", "-inf", "-0.1"):
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parser("test").parse_args([f"{flag}={value}"])
                    self.assertNotEqual(error.exception.code, 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser("test").parse_args(["--min-correlation=1.01"])
        valid = parser("test").parse_args(["--min-correlation=1", "--max-mae=0", "--max-nrmse=0"])
        self.assertEqual((valid.min_correlation, valid.max_mae, valid.max_nrmse), (1, 0, 0))

    def test_approximate_is_not_bit_exact(self):
        a = np.linspace(.1, .9, 12).reshape(2, 2, 3)
        b = a + 1 / 255
        report = Report("approx", self.args)
        report.compare("pixels", a, b, quantize=True)
        self.assertEqual(report.finish(), 0)
        self.assertLess(report.data["checks"][0]["exact_fraction"], 1)
        self.args.bit_exact = True
        exact = Report("exact", self.args)
        exact.compare("pixels", a, b, quantize=True)
        self.assertEqual(exact.finish(), 1)

    def test_large_error_and_nonfinite_fail(self):
        report = Report("regression", self.args)
        report.compare("bad", np.zeros((2, 2, 3)), np.ones((2, 2, 3)), quantize=True)
        self.assertEqual(report.finish(), 1)
        for bad in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                metrics([bad], [0])

    def test_partial_cannot_pass_strict(self):
        report = Report("partial", self.args)
        report.compare("available", [1, 2], [1, 2])
        report.skip("missing", "capture unavailable")
        self.assertEqual(report.finish(), 1)
        self.assertEqual(report.data["status"], "incomplete")
        self.assertEqual(report.data["coverage"]["fraction"], .5)
        self.assertEqual(report.finish(allow_partial=True), 0)
        self.assertFalse(report.data["passed"])

    def test_no_comparisons_and_missing_references_fail(self):
        self.assertEqual(Report("empty", self.args).finish(), 1)
        self.assertEqual(execute("missing", self.args, lambda _: ppm(self.root / "missing.ppm", 2, 2)), 1)
        report = json.loads((self.root / "report.json").read_text())
        self.assertIn("FileNotFoundError", report["errors"][0])

    def test_ppm_validates_shape_raster_and_comments(self):
        path = self.root / "ref.ppm"
        path.write_bytes(b"P6\n# reference\n2 1\n255\n" + bytes([10, 32, 13, 0, 255, 7]))
        self.assertEqual(ppm(path, 2, 1).shape, (1, 2, 3))
        with self.assertRaises(ValueError):
            ppm(path, 1, 2)
        path.write_bytes(b"P6\n2 1\n255\n\x00")
        with self.assertRaises(ValueError):
            ppm(path, 2, 1)

    def test_entrypoints_missing_reference_exit_nonzero_and_write_json(self):
        here = Path(__file__).resolve().parent
        for script in ("check", "check_frames", "check_motion", "check_res", "check_levels", "check_blocks"):
            with self.subTest(script=script):
                path = self.root / f"{script}.json"
                arguments = [sys.executable, str(here / f"{script}.py"), "--reference-dir", str(self.root),
                             "--report", str(path)]
                if script in ("check_levels", "check_blocks"):
                    arguments += ["--capture", str(self.root / "missing-capture")]
                result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(path.exists(), result.stdout + result.stderr)
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertFalse(data["passed"])
                self.assertEqual(data["coverage"]["compared"], 0)


if __name__ == "__main__":
    unittest.main()
