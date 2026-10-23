"""CPU-only preservation tests for shared report paths and actual CLI adapters."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from .report_paths import ReportPath, aliases
from . import verify_assets, trace_transfer_host

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch"))
sys.path.insert(0, str(ROOT / "harness"))
import acceptance
import check_temporal


class ReportPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dlss5-report-safety-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "torch/model.py"
        self.source.parent.mkdir()
        self.original = b"source must survive\n"
        self.source.write_bytes(self.original)

    def test_source_report_and_temporary_aliases_rejected_without_writes(self):
        for mode in ("direct", "hardlink", "temporary_hardlink", "case_dotdot"):
            with self.subTest(mode=mode):
                report = self.root / (mode + ".json")
                if mode == "direct":
                    report = self.source
                elif mode == "hardlink":
                    report.hardlink_to(self.source)
                elif mode == "temporary_hardlink":
                    report.with_suffix(".json.tmp").hardlink_to(self.source)
                else:
                    report = self.source.parent / ".." / "torch" / "MODEL.PY"
                with self.assertRaisesRegex(ValueError, "aliases"):
                    ReportPath(report, root=self.root)
                self.assertEqual(self.source.read_bytes(), self.original)

    def test_runtime_binaries_and_package_model_are_project_inputs(self):
        for relative in ("harness/rt_sm86/nvngx.dll", "harness/batch3/nr-lab.exe", "plugin/dist/payload/model.cubin", "plugin/dist/payload/nvngx_dlssnr.dll"):
            with self.subTest(relative=relative):
                source = self.root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"MZ private fixture")
                outside = self.root / (source.name + ".json")
                if outside.exists():
                    outside.unlink()
                outside.hardlink_to(source)
                with self.assertRaisesRegex(ValueError, "aliases"):
                    ReportPath(outside, root=self.root)
                self.assertEqual(source.read_bytes(), b"MZ private fixture")

    def test_capture_containment_and_outside_hardlink_are_both_rejected(self):
        capture = self.root / "capture"
        capture.mkdir()
        source = capture / "manifest.json"
        source.write_bytes(self.original)
        for output in (capture / "new_report.json", self.root / "outside.json"):
            if output.parent != capture:
                output.hardlink_to(source)
            with self.assertRaises(ValueError):
                ReportPath(output, input_dirs=(capture,), root=self.root)
        self.assertEqual(source.read_bytes(), self.original)

    def test_report_and_temporary_cannot_alias_each_other(self):
        output = self.root / "report.json"
        output.write_text('{"passed":true}')
        output.with_suffix(".json.tmp").hardlink_to(output)
        with self.assertRaisesRegex(ValueError, "temporary"):
            ReportPath(output, root=self.root)

    def test_dynamic_input_registration_and_late_alias_are_revalidated(self):
        output = self.root / "report.json"
        guard = ReportPath(output, root=self.root)
        output.hardlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "aliases"):
            guard.write_json({"passed": True})
        self.assertEqual(self.source.read_bytes(), self.original)
        dynamic = self.root / "dynamic.dat"
        dynamic.write_bytes(self.original)
        guard = ReportPath(dynamic, root=self.root)
        with self.assertRaisesRegex(ValueError, "aliases"):
            guard.add_inputs(dynamic)
        self.assertEqual(dynamic.read_bytes(), self.original)

    def test_safe_failure_replaces_old_pass_and_serialization_can_retry(self):
        output = self.root / "report.json"
        output.write_text('{"passed":true}')
        guard = ReportPath(output, root=self.root)
        with self.assertRaises(ValueError):
            guard.write_json({"bad": float("nan")})
        self.assertFalse(guard.temporary.exists())
        guard.write_json({"passed": False, "status": "failed"})
        self.assertFalse(json.loads(output.read_text())["passed"])
        self.assertFalse(guard.temporary.exists())

    def test_existing_user_temporary_file_is_preserved(self):
        output = self.root / "report.json"
        guard = ReportPath(output, root=self.root)
        guard.temporary.write_bytes(self.original)
        with self.assertRaises(FileExistsError):
            guard.write_json({"passed": True})
        self.assertEqual(guard.temporary.read_bytes(), self.original)
        self.assertFalse(output.exists())

    def test_write_failure_removes_only_its_own_temporary_file(self):
        guard = ReportPath(self.root / "report.json", root=self.root)
        with patch.object(guard, "validate", side_effect=[guard.path, RuntimeError("after write fixture")]):
            with self.assertRaisesRegex(RuntimeError, "after write fixture"):
                guard.write_json({"passed": True})
        self.assertFalse(guard.temporary.exists())
        guard.write_json({"passed": False})


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dlss5-report-adapters-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_verify_assets_rejects_input_before_validation_and_preserves_it(self):
        source = self.root / "weights_map.json"
        original = b'{"source_metadata":"fixture"}\n'
        source.write_bytes(original)
        for output in (source, self.root / "outside.json"):
            if output != source:
                output.hardlink_to(source)
            with patch.object(verify_assets, "verify", side_effect=AssertionError("must not validate unsafe output")), \
                 patch.object(sys, "argv", ["verify_assets", "--root", str(self.root), "--report", str(output)]):
                with self.assertRaisesRegex(ValueError, "aliases"):
                    verify_assets.main()
            self.assertEqual(source.read_bytes(), original)

    def test_verify_assets_safe_validation_failure_writes_failed_report(self):
        output = self.root / "report.json"
        output.write_text('{"passed":true}')
        with patch.object(verify_assets, "verify", side_effect=ValueError("fixture")), \
             patch.object(sys, "argv", ["verify_assets", "--root", str(self.root), "--report", str(output)]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verify_assets.main(), 1)
        self.assertFalse(json.loads(output.read_text())["passed"])

    def test_transfer_host_rejects_custom_dll_alias_before_loading(self):
        source = self.root / "original.dll"
        source.write_bytes(b"MZfixture")
        outside = self.root / "outside.json"
        outside.hardlink_to(source)
        with patch.object(trace_transfer_host, "Image", side_effect=AssertionError("must not load unsafe output")), \
             patch.object(sys, "argv", ["transfer", "--dll", str(source), "--port", str(source), "--report", str(outside)]):
            with self.assertRaisesRegex(ValueError, "aliases"):
                trace_transfer_host.main()
        self.assertEqual(source.read_bytes(), b"MZfixture")

    def test_temporal_outside_hardlink_cannot_overwrite_capture(self):
        capture = self.root / "capture"
        capture.mkdir()
        source = capture / "manifest.json"
        source.write_bytes(b"original evidence")
        output = self.root / "outside.json"
        output.hardlink_to(source)
        with patch.object(sys, "argv", ["temporal", "--capture", str(capture), "--report", str(output)]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                check_temporal.main()
        self.assertNotEqual(error.exception.code, 0)
        self.assertEqual(source.read_bytes(), b"original evidence")

    def test_acceptance_constructor_registration_and_finish_protect_inputs(self):
        source = self.root / "reference.ppm"
        source.write_bytes(b"P6\n1 1\n255\n\x01\x02\x03")
        args = acceptance.parser("fixture").parse_args(["--reference-dir", str(self.root), "--report", str(source)])
        with self.assertRaisesRegex(ValueError, "aliases"):
            acceptance.Report("fixture", args)
        original = source.read_bytes()
        metadata = self.root / "metadata.json"
        metadata.write_bytes(b'{"metadata":"preserve"}')
        args.report = metadata
        report = acceptance.Report("fixture", args)
        with self.assertRaisesRegex(ValueError, "aliases"):
            report.reference(metadata)
        self.assertEqual(metadata.read_bytes(), b'{"metadata":"preserve"}')
        args.report = self.root / "report.json"
        report = acceptance.Report("fixture", args)
        report.reference(source)
        report.data["checks"].append({"passed": True})
        args.report.hardlink_to(source)
        with self.assertRaisesRegex(ValueError, "aliases"):
            report.finish()
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
