"""CPU failure-path tests for the strict schedule reference contract."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import schedule_refs as refs


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.reference_plan = refs.plan()
        description = dict(self.reference_plan["buffers"]["s00_out"], bytes=16, shape_hwc=[1, 1, 16])
        self.reference_plan["buffers"] = {"s00_out": description}
        self.reference_plan["steps"] = []
        self.reference_plan["bridges"] = []
        self.patches = [patch.object(refs, "plan", return_value=self.reference_plan),
                        patch.object(refs, "validate_bindings"), patch.object(refs, "trace_rows", return_value={1: {"params": bytes(248) + struct.pack("<Q", 0x10000)}})]
        for item in self.patches:
            item.start()
        run = self.folder / "run000"
        run.mkdir()
        request = refs.chunks_for_plan(self.reference_plan)[0][0]
        request.update(path=str(Path("run000") / "tap_s1_10000.bin"), address="0x10000", base_pointer="0x10000")
        (run / "tap_s1_10000.bin").write_bytes(bytes(range(16)))
        request["sha256"] = refs.digest(run / "tap_s1_10000.bin")
        (run / refs.OUTPUT).write_bytes(b"P6\n640 360\n255\n" + bytes([1, 2, 3]) * (640 * 360))
        from harness.reference import case_contract
        status, _ = case_contract({"args": refs.COMMAND})
        refs.save(run / "nr-lab-result.json", dict(status, outputCoverageValidated=True))
        (run / "nr-trace.tsv").write_text("fixture")
        (run / "stdout.txt").write_text("[tap] frame 0 seq 1 0x10000 +16 -> off 0 (launch=0)\n")
        input_path = self.folder / "color_rgb8.bin"
        input_path.write_bytes(refs.color_rgb8())
        (self.folder / "buffers").mkdir()
        (self.folder / "buffers/s00_out.bin").write_bytes(bytes(range(16)))
        artifacts = {p.name: {"path": str(p.relative_to(self.folder)), "bytes": p.stat().st_size, "sha256": refs.digest(p)} for p in run.iterdir()}
        self.runtime = {"nvngx_dlssnr.dll": refs.PORT_SHA256, "nr-lab.exe": "fixture-harness"}
        self.manifest = {"schema": refs.SCHEMA, "kind": "schedule-reference-sm86", "passed": True, "errors": [],
                         "plan": self.reference_plan, "original_dll_sha256": refs.ORIGINAL_SHA256,
                         "runtime_sha256": self.runtime, "official_reference": False, "reference_kind": "port-sm86",
                         "input": {"path": input_path.name, "bytes": input_path.stat().st_size, "sha256": refs.digest(input_path)},
                         "runs": [{"passed": True, "returncode": 0, "command": ["fixture.exe", *refs.COMMAND], "requests": [request], "artifacts": artifacts}],
                         "buffers": {"s00_out": {"path": "buffers/s00_out.bin", "bytes": 16,
                                                    "sha256": refs.digest(self.folder / "buffers/s00_out.bin"), "chunks": [copy.deepcopy(request)]}},
                         "final_output": artifacts[refs.OUTPUT]}
        self.write()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def write(self):
        refs.save(self.folder / "manifest.json", self.manifest)

    def validate(self):
        return refs.validate(self.folder, expected_runtime=self.runtime, verify_sources=False)

    def test_valid_minimal_provenance_fixture(self):
        self.assertTrue(self.validate()["passed"])

    def test_missing_named_buffer(self):
        (self.folder / "buffers/s00_out.bin").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/exact-size/hash"):
            self.validate()

    def test_wrong_size_even_if_hash_is_updated(self):
        path = self.folder / "buffers/s00_out.bin"
        path.write_bytes(path.read_bytes() + b"extra")
        self.manifest["buffers"]["s00_out"].update(bytes=21, sha256=refs.digest(path))
        self.write()
        with self.assertRaisesRegex(ValueError, "exact-size"):
            self.validate()

    def test_tampered_capture_artifact(self):
        (self.folder / "run000/tap_s1_10000.bin").write_bytes(bytes([255]) * 16)
        with self.assertRaisesRegex(ValueError, "size/hash mismatch"):
            self.validate()

    def test_wrong_dll_identity(self):
        self.manifest["original_dll_sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "DLL identity"):
            self.validate()

    def test_wrong_harness_identity(self):
        self.manifest["runtime_sha256"] = dict(self.runtime, **{"nr-lab.exe": "different"})
        self.write()
        with self.assertRaisesRegex(ValueError, "runtime identity"):
            self.validate()

    def test_wrong_schedule_identity(self):
        self.manifest["plan"] = dict(self.reference_plan, schedule_sha256="different")
        self.write()
        with self.assertRaisesRegex(ValueError, "plan identity"):
            self.validate()

    def test_unbound_or_reordered_chunks(self):
        self.manifest["buffers"]["s00_out"]["chunks"] = []
        self.write()
        with self.assertRaisesRegex(ValueError, "chunks"):
            self.validate()

    def test_traversal(self):
        self.manifest["buffers"]["s00_out"]["path"] = "../elsewhere.bin"
        self.write()
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.validate()

    def test_failed_execution_cannot_be_promoted_by_manifest_pass(self):
        path = self.folder / "run000/nr-lab-result.json"
        refs.save(path, {"created": True, "evaluationsSucceeded": 0, "outputCoverageValidated": True})
        self.manifest["runs"][0]["artifacts"][path.name].update(bytes=path.stat().st_size, sha256=refs.digest(path))
        self.write()
        with self.assertRaisesRegex(ValueError, "Execution result"):
            self.validate()

    def test_wrong_execution_contract_with_rehashed_artifact(self):
        path = self.folder / "run000/nr-lab-result.json"
        status = json.loads(path.read_text())
        status.update(model=999, input={"width": 1, "height": 1}, motionVector=[100, 100])
        refs.save(path, status)
        self.manifest["runs"][0]["artifacts"][path.name].update(bytes=path.stat().st_size, sha256=refs.digest(path))
        self.write()
        with self.assertRaisesRegex(ValueError, "requested contract"):
            self.validate()


class PlanAndCliTests(unittest.TestCase):
    def test_numerics_help_is_cpu_only(self):
        result = subprocess.run([sys.executable, str(refs.ROOT / "torch/check_numerics.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--capture", result.stdout)
        self.assertIn("--levels", result.stdout)

    def test_numerics_missing_capture_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(refs.ROOT / "torch/check_numerics.py"), "--capture", str(Path(folder) / "missing"),
                                     "--levels", "1h"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("before GPU initialization", result.stderr)

    def test_plan_has_all_steps_and_bounded_explicit_requests(self):
        p = refs.plan()
        self.assertEqual([step["step"] for step in p["steps"]], list(range(71)))
        self.assertEqual(len(p["buffers"]), 78)
        for job in refs.chunks_for_plan(p):
            self.assertLessEqual(sum(request["bytes"] for request in job), refs.MAX_TAP_BYTES)
        self.assertIn("s00_skip", p["buffers"])
        self.assertEqual(p["buffers"]["s00_skip"]["bytes"], 7_864_320)

    def test_missing_capture_invalidates_old_success_before_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = folder / "report.json"
            refs.save(report, {"passed": True})
            result = subprocess.run([sys.executable, str(refs.ROOT / "torch/check_schedule.py"),
                                     "--capture", str(folder / "missing"), "--harness", str(folder / "missing.exe"),
                                     "--report", str(report)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            current = json.loads(report.read_text())
            self.assertFalse(current["passed"])
            self.assertTrue(current["errors"])
            self.assertNotIn("gpu", current)

    def test_nonfinite_cli_invalidates_old_success(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = folder / "report.json"
            refs.save(report, {"passed": True})
            result = subprocess.run([sys.executable, str(refs.ROOT / "torch/check_schedule.py"),
                                     "--capture", str(folder / "missing"), "--harness", "unneeded.exe",
                                     "--report", str(report), "--min-correlation", "nan"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(report.read_text())["passed"])

    def test_report_inside_capture_cannot_destroy_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = folder / "manifest.json"
            original = b'{"passed":true,"original_evidence":true}'
            report.write_bytes(original)
            result = subprocess.run([sys.executable, str(refs.ROOT / "torch/check_schedule.py"),
                                     "--capture", str(folder), "--harness", "unneeded.exe", "--report", str(report)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(report.read_bytes(), original)

    def test_external_hardlink_cannot_destroy_capture_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            capture = folder / "capture"
            capture.mkdir()
            manifest = capture / "manifest.json"
            original = b'{"passed":true,"original_evidence":true}'
            manifest.write_bytes(original)
            report = folder / "outside.json"
            report.hardlink_to(manifest)
            result = subprocess.run([sys.executable, str(refs.ROOT / "torch/check_schedule.py"),
                                     "--capture", str(capture), "--harness", "unneeded.exe", "--report", str(report)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(manifest.read_bytes(), original)

    def test_missing_capture_with_resolved_runtime_invalidates_old_pass(self):
        sys.path.insert(0, str(refs.ROOT / "torch"))
        import check_schedule
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = folder / "report.json"
            refs.save(report, {"passed": True})
            with patch.object(check_schedule, "runtime_files", return_value={}), patch.object(sys, "argv", [
                    "check_schedule", "--capture", str(folder / "missing"), "--harness", "resolved-fixture", "--report", str(report)]):
                self.assertEqual(check_schedule.main(), 1)
            current = json.loads(report.read_text())
            self.assertFalse(current["passed"])
            self.assertIn("manifest.json", current["errors"][0])
            self.assertNotIn("gpu", current)

    def trace_fixture(self, folder):
        launches = json.loads(refs.EXEC_ORDER.read_text())["launches"]
        lines = ["\t".join([str(row["seq"]), "0/0/1", row["kernel"], row["grid"], row["block"],
                            str(row["smem"]), str(row["param_bytes"]), bytes(row["param_bytes"]).hex()]) for row in launches]
        path = Path(folder) / "trace.tsv"
        path.write_text("\n".join(lines) + "\n")
        return path, lines

    def test_wrong_trace_kernel_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path, lines = self.trace_fixture(folder)
            self.assertEqual(len(refs.trace_rows(path)), 156)
            lines[31] = lines[31].replace("cc_split_swin", "wrong_kernel")
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "Launch identity"):
                refs.trace_rows(path)

    def test_missing_trace_sequence(self):
        with tempfile.TemporaryDirectory() as folder:
            path, lines = self.trace_fixture(folder)
            path.write_text("\n".join(lines[:41] + lines[42:]) + "\n")
            with self.assertRaisesRegex(ValueError, "Launch identity"):
                refs.trace_rows(path)


if __name__ == "__main__":
    unittest.main()
