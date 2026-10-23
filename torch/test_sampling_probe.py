"""CPU rejection checks for the reciprocal probe's output entry point."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import probe_sampling_rcp as probe


class ProbeReportTests(unittest.TestCase):
    def test_unknown_option_invalidates_old_pass_without_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "project"
            root.mkdir()
            report = Path(folder) / "report.json"
            report.write_text('{"passed":true}')
            with patch.object(probe, "ROOT", root), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU only")):
                with self.assertRaises(SystemExit) as error:
                    probe.main(["--report", str(report), "--invalid-option"])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(json.loads(report.read_text())["passed"])

    def test_report_and_temporary_hardlinks_preserve_input(self):
        for temporary in (False, True):
            with self.subTest(temporary=temporary), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "project"
                source = root / "torch/dlss5/ptx_math.py"
                source.parent.mkdir(parents=True)
                source.write_bytes(b"preserve authored input")
                report = Path(folder) / "report.json"
                (report.with_suffix(".json.tmp") if temporary else report).hardlink_to(source)
                with patch.object(probe, "ROOT", root), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU only")):
                    self.assertEqual(probe.main(["--report", str(report)]), 2)
                self.assertEqual(source.read_bytes(), b"preserve authored input")


if __name__ == "__main__":
    unittest.main()
