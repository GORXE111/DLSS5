"""CPU-only regression checks for release data preservation and HDR rejection paths."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


package = module("build_plugin", HERE / "build_plugin.py")
hdr = module("check_hdr", HERE / "fallback" / "check_hdr.py")


class ReleaseSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dlss5-release-safety-")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def fixture_package(self):
        out = self.root / "dist" / "fixture"
        out.mkdir(parents=True)
        (out / "payload.dat").write_bytes(b"original payload")
        artifacts = package.file_inventory(out)
        info = {"package": "DLSS5-RTX30", "schema_version": 2, "artifacts": artifacts,
                "build": {"sources": [], "source_fingerprint": package.fingerprint([]),
                          "artifacts_fingerprint": package.fingerprint(artifacts)}}
        (out / "package.json").write_text(json.dumps(info), encoding="utf-8")
        return out

    def test_reject_workspace_and_parent_outputs(self):
        with patch.object(package, "HERE", str(self.root)):
            for path in (self.root, self.root / "dist", self.root / "src", self.root / "dist" / "nested" / "package"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    package.validate_output(path)

    def test_reject_unowned_existing_directory_without_changes(self):
        out = self.root / "dist" / "fixture"
        out.mkdir(parents=True)
        sentinel = out / "user-save.txt"
        sentinel.write_bytes(b"keep me")
        with self.assertRaises(ValueError):
            package.validate_existing(out)
        self.assertEqual(sentinel.read_bytes(), b"keep me")

    def test_reject_modified_package_file(self):
        out = self.fixture_package()
        (out / "payload.dat").write_bytes(b"modified payload")
        with self.assertRaisesRegex(ValueError, "modified"):
            package.validate_existing(out)
        self.assertEqual((out / "payload.dat").read_bytes(), b"modified payload")

    def test_reject_extra_user_file_without_removing_it(self):
        out = self.fixture_package()
        (out / "user-save.txt").write_bytes(b"keep me")
        with self.assertRaisesRegex(ValueError, "unrecognized files"):
            package.validate_existing(out)
        self.assertEqual((out / "user-save.txt").read_bytes(), b"keep me")

    def test_reject_source_drift(self):
        out = self.fixture_package()
        with patch.object(package, "source_inventory", return_value=[{"path": "changed.cpp", "sha256": "different"}]), self.assertRaisesRegex(ValueError, "source fingerprint"):
            package.verify_package(out)

    def test_failed_build_preserves_previous_package(self):
        out = self.fixture_package()
        opti = self.root / "opti"
        opti.mkdir()
        (opti / "OptiScaler.ini").write_text("[DlssNr]\n", encoding="utf-8")
        dll = self.root / "model.dll"
        dll.write_bytes(b"MZfixture")
        before = package.file_inventory(out)
        with patch.object(package, "HERE", str(self.root)), patch.object(package, "source_inventory", return_value=[]), patch.object(package.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "build")), patch.object(sys, "argv", ["build_plugin", "--opti", str(opti), "--dll", str(dll), "--out", str(out)]):
            with self.assertRaises(subprocess.CalledProcessError):
                package.main()
        self.assertEqual(package.file_inventory(out), before)

    def test_hdr_rejects_nonfinite_and_truncated_dumps(self):
        p = self.root / "test.f16"
        p.write_bytes(struct.pack("<4e", float("nan"), 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            hdr.read_hdr(p, 1, 1, "f16")
        p.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "expected"):
            hdr.read_hdr(p, 1, 1, "f16")

    def test_hdr_rejects_transparent_output(self):
        p = self.root / "test.pq"
        p.write_bytes(struct.pack("<I", 0x1fffffff))
        with self.assertRaisesRegex(ValueError, "alpha"):
            hdr.read_hdr(p, 1, 1, "pq")

    def test_hdr_rejects_clipped_highlights_and_noop(self):
        before = [(800, 600, 500)] * 16
        with self.assertRaisesRegex(ValueError, "highlights lost"):
            hdr.validate(before, [(200, 200, 200)] * 16, 400)
        with self.assertRaisesRegex(ValueError, "not processed"):
            hdr.validate(before, before, 400)
        metrics = hdr.validate(before, [(790, 590, 490)] * 16, 400)
        self.assertGreater(metrics["highlight_mean_ratio"], 0.98)

    def test_missing_reference_exits_before_running_gpu(self):
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                 str(HERE / "fallback" / "run_tests.ps1"), "-ReferenceDirectory", str(self.root / "absent")],
                                capture_output=True, text=True, errors="replace")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required reference", result.stdout + result.stderr)
        self.assertNotIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
