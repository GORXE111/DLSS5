"""CPU-only cache invalidation checks; no CUDA compiler or DLL build is run."""
from pathlib import Path
import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from . import build_dll as build


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work, self.fatbins, self.bin, self.code = [self.root / p for p in ("work", "fatbins", "bin", "code")]
        for directory in (self.work, self.fatbins, self.bin, self.code):
            directory.mkdir()
        for name in ("build_dll", "rewrite", "fuse", "flow"):
            (self.code / f"{name}.py").write_text(name)
        for name in ("ptxas.exe", "fatbinary.exe", "cuobjdump.exe"):
            (self.bin / name).write_bytes(name.encode())
        (self.fatbins / "fatbin_01.1.sm_120.ptx").write_text("source")
        (self.fatbins / "fatbin_01.fatbin").write_bytes(b"original container")
        self.calls = []
        for name, value in (("WORK", str(self.work)), ("FATBINS", str(self.fatbins)),
                            ("BIN", str(self.bin)), ("__file__", str(self.code / "build_dll.py"))):
            p = patch.object(build, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(build, "run", self.fake_run)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(build, "rewrite", lambda text: (text + " rewritten", {}))
        p.start()
        self.addCleanup(p.stop)
        p = patch.dict(build.rw.ACC, {"mode": "f16"}, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def fake_run(self, *args, cwd=None):
        tool = Path(args[0]).name
        self.calls.append(tool)
        if tool == "cuobjdump.exe":
            (Path(cwd) / "fatbin_01.1.sm_120.cubin").write_bytes(Path(args[-1]).read_bytes() + b" cubin")
        elif tool == "ptxas.exe":
            Path(args[-1]).write_bytes(Path(args[2]).read_bytes() + b" compiled")
        elif tool == "fatbinary.exe":
            Path(next(a.split("=", 1)[1] for a in args if a.startswith("--create="))).write_bytes(b"packed")

    def test_content_and_options_invalidate_cache_without_mtime_reliance(self):
        first = build.build_fatbin(1)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(build.build_fatbin(1), first)
        self.assertEqual(len(self.calls), 3)
        paths = [self.fatbins / "fatbin_01.1.sm_120.ptx", self.fatbins / "fatbin_01.fatbin",
                 *(self.code / f"{n}.py" for n in ("build_dll", "rewrite", "fuse", "flow")),
                 *(self.bin / n for n in ("ptxas.exe", "fatbinary.exe", "cuobjdump.exe"))]
        for path in paths:
            with self.subTest(path=path.name):
                old = self.calls.count("ptxas.exe")
                path.write_bytes(path.read_bytes() + b" changed")
                build.build_fatbin(1)
                self.assertEqual(self.calls.count("ptxas.exe"), old + 1)
        old = self.calls.count("ptxas.exe")
        build.rw.ACC["mode"] = "f32"
        build.build_fatbin(1)
        self.assertEqual(self.calls.count("ptxas.exe"), old + 1)

    def test_corrupt_original_cubin_is_reextracted(self):
        first = build.build_fatbin(1)
        (self.work / "fatbin_01.1.sm_120.cubin").write_bytes(b"tampered original")
        build.build_fatbin(1)
        self.assertEqual(self.calls.count("cuobjdump.exe"), 2)
        # Restoring exactly the original bytes makes the already-built outputs valid again.
        self.assertEqual(self.calls.count("ptxas.exe"), 1)
        self.assertEqual(build.build_fatbin(1), first)

    def test_corrupt_or_missing_outputs_are_rebuilt(self):
        build.build_fatbin(1)
        for name in ("fatbin_01.sm86.ptx", "fatbin_01.sm86.cubin", "fatbin_01.sm86.fatbin"):
            old = self.calls.count("ptxas.exe")
            path = self.work / name
            path.write_bytes(b"damaged")
            build.build_fatbin(1)
            self.assertEqual(self.calls.count("ptxas.exe"), old + 1)
        manifest = self.work / "fatbin_01.sm86.fatbin.cache.json"
        manifest.unlink()
        old = self.calls.count("ptxas.exe")
        build.build_fatbin(1)
        self.assertEqual(self.calls.count("ptxas.exe"), old + 1)


class BuildReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.report = Path(self.tmp.name) / "report.json"
        self.report.write_text('{"status": "passed", "old_run": true}')
        self.addCleanup(setattr, build, "BUILD_REPORT", None)
        self.addCleanup(setattr, build, "BUILD_REPORT_PATH", None)
        guard = patch.object(build, "run", side_effect=AssertionError("report tests must never invoke a CUDA tool"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_systemexit_invalidates_stale_report_and_fails(self):
        with patch.object(build, "build", side_effect=SystemExit(7)), self.assertRaises(SystemExit) as error:
            build.main(["--report", str(self.report)])
        report = json.loads(self.report.read_text())
        self.assertEqual(error.exception.code, 7)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "SystemExit")
        self.assertEqual(report["stages"][-1]["status"], "failed")
        self.assertNotIn("old_run", report)

    def test_invalid_arguments_invalidate_stale_report(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build.main(["--report", str(self.report), "--acc", "invalid"])
        self.assertEqual(json.loads(self.report.read_text())["status"], "failed")

    def test_build_exception_is_reported(self):
        with patch.object(build, "build", side_effect=RuntimeError("fixture compiler failure")), self.assertRaises(RuntimeError):
            build.main(["--report", str(self.report)])
        report = json.loads(self.report.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("fixture compiler failure", report["error"])

    def test_explicit_work_and_relative_output_do_not_change_cwd(self):
        initial = os.getcwd()
        source = Path(self.tmp.name) / "wrong-source.dll"
        source.write_bytes(b"wrong source must never be patched")
        os.chdir(self.tmp.name)
        try:
            args = argparse.Namespace(work="isolated-cache", out="output/model.dll", enc="v1", dec="v1",
                                      acc="f16", maxnreg=0, maxnreg_floor=0)
            with patch.object(build, "SRC_DLL", str(source)), self.assertRaises(ValueError):
                build.build(args)
            self.assertEqual(Path.cwd(), Path(self.tmp.name))
            self.assertEqual(Path(build.WORK), Path(self.tmp.name) / "isolated-cache")
            self.assertEqual(Path(args.out), Path(self.tmp.name) / "output" / "model.dll")
            self.assertEqual(source.read_bytes(), b"wrong source must never be patched")
        finally:
            os.chdir(initial)

if __name__ == "__main__":
    unittest.main()
