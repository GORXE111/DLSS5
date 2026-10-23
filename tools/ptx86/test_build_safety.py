"""CPU-only file-preservation and forged-evidence regressions; every artifact is a fixture."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import build_dll as build
from . import verify_build as verify


class SafetyFixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dlss5-build-safety-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.work, self.code, self.inputs, self.bin = [self.root / name for name in ("work", "code", "inputs", "bin")]
        for path in (self.work, self.code, self.inputs, self.bin):
            path.mkdir()
        for name in ("build_dll", "rewrite", "fuse", "flow", "verify_build"):
            (self.code / (name + ".py")).write_text(name, encoding="utf-8")
        for name in ("ptxas.exe", "fatbinary.exe", "cuobjdump.exe"):
            (self.bin / name).write_bytes(name.encode())
        self.source = self.root / "original.dll"
        self.source.write_bytes(b"original fixture must survive")
        for name, value in (("SRC_DLL", str(self.source)), ("ROOT", str(self.root)), ("FATBINS", str(self.inputs)),
                            ("BIN", str(self.bin)), ("__file__", str(self.code / "build_dll.py"))):
            context = patch.object(build, name, value)
            context.start()
            self.addCleanup(context.stop)
        guard = patch.object(build, "run", side_effect=AssertionError("tests must never run real compilers"))
        guard.start()
        self.addCleanup(guard.stop)
        for name, value in (("BUILD_REPORT", None), ("BUILD_REPORT_PATH", None), ("BUILD_OUTPUT", None), ("BUILD_PROTECTED", [])):
            self.addCleanup(setattr, build, name, value)
        self.payloads, self.blocks, self.fatbins, self.rows = [], [], [], []
        for number in range(1, 16):
            tag = f"fatbin_{number:02d}"
            original = ("original-cubin-" + tag).encode()
            compiled = ("compiled-cubin-" + tag).encode()
            packed = b"packed:" + original + compiled
            block = ("source-fatbin-" + tag).encode()
            (self.inputs / (tag + ".fatbin")).write_bytes(block)
            (self.inputs / (tag + ".1.sm_120.ptx")).write_text("source PTX " + tag)
            (self.work / (tag + ".1.sm_120.cubin")).write_bytes(original)
            outputs = {}
            for kind, suffix, content in (("ptx", ".sm86.ptx", b"rewritten PTX"), ("cubin", ".sm86.cubin", compiled), ("fatbin", ".sm86.fatbin", packed)):
                path = self.work / (tag + suffix)
                path.write_bytes(content)
                outputs[kind] = {"path": str(path), "bytes": len(content), "sha256": verify.digest(content)}
            for suffix in (".sm86.fatbin.cache.json", ".sm86.fatbin.cache.json.tmp"):
                (self.work / (tag + suffix)).write_bytes(b"cache metadata sentinel")
            inputs = build.cache_inputs(self.inputs / (tag + ".1.sm_120.ptx"), self.inputs / (tag + ".fatbin"),
                                        self.work / (tag + ".1.sm_120.cubin"), options=verify.DEFAULT_OPTIONS)
            self.rows.append({"tag": tag, "status": "passed", "cache": "miss", "extracted_original": False,
                              "inputs": inputs, "outputs": outputs, "original_cubin_sha256": verify.digest(original)})
            self.payloads.append(packed)
            self.blocks.append(block)
            self.fatbins.append({"sha256": verify.digest(packed)})
        self.first, self.second = self.root / "first.dll", self.root / "second.dll"
        self.first.write_bytes(b"MZfixture-output")
        self.second.write_bytes(self.first.read_bytes())
        self.report1, self.report2 = self.root / "first.json", self.root / "second.json"
        self.runs = [self.make_run(index) for index in (0, 1)]
        for path, run in zip((self.report1, self.report2), self.runs):
            path.write_text(json.dumps(run), encoding="utf-8")

    def make_run(self, index):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["cache"] = "miss" if index == 0 else "hit"
        stages = [{"name": "full_dll_build", "status": "passed", "wall_seconds": 10,
                   "started_utc": "2026-10-23T00:00:00+00:00"}]
        for row in rows:
            names = [row["tag"] + "/total"]
            if index == 0:
                names += [row["tag"] + "/" + name for name in ("rewrite", "ptxas", "fatbinary")]
            for name in names:
                stages.append({"name": name, "status": "passed", "wall_seconds": 0.5 if name.endswith("/total") else 0.1,
                               "started_utc": "2026-10-23T00:00:00+00:00"})
        output = self.first if index == 0 else self.second
        return {"schema": 2, "report_kind": "full-dll-build", "status": "passed", "numerical_options": dict(verify.DEFAULT_OPTIONS),
                "source_dll_sha256": verify.DLL_SHA256, "output": str(output), "output_sha256": build.file_sha256(output),
                "output_bytes": output.stat().st_size, "work": str(self.work), "cache_hits": 0 if index == 0 else 15,
                "cache_misses": 15 if index == 0 else 0, "fatbins": rows, "stages": stages, "wall_seconds": 11}

    def snapshot(self):
        return {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def assert_build_rejected_without_writes(self, output, report):
        before = self.snapshot()
        with patch.object(build, "build", side_effect=lambda args: Path(args.out).write_bytes(b"MZnew-output")) as operation:
            with self.assertRaises(ValueError):
                build.main(["--work", str(self.work), "--out", str(output), "--report", str(report)])
            operation.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_build_report_equals_output_rejected_before_any_write(self):
        self.assert_build_rejected_without_writes(self.first, self.first)

    def test_preflight_normalizes_equivalent_integer_spellings(self):
        default_output = self.root / "sm86_port" / "nvngx_dlssnr.dll"
        default_output.parent.mkdir()
        default_output.write_bytes(b"MZdefault fixture must survive")
        for spelling in ("00", "-0", "+0"):
            before = self.snapshot()
            with patch.object(build, "build", return_value=None) as operation, self.assertRaises(ValueError):
                build.main(["--maxnreg", spelling, "--report", str(default_output)])
            operation.assert_not_called()
            self.assertEqual(self.snapshot(), before)

    def test_build_report_and_temporary_cannot_alias_inputs_or_caches(self):
        protected = [self.source, self.code / "build_dll.py", self.bin / "ptxas.exe",
                     self.inputs / "fatbin_01.1.sm_120.ptx", self.inputs / "fatbin_01.fatbin", *build.cache_paths(self.work)[:6]]
        for path in protected:
            with self.subTest(path=path):
                self.assert_build_rejected_without_writes(self.first, path)
        report = self.root / "temporary-alias.json"
        report.with_suffix(".json.tmp").hardlink_to(self.source)
        self.assert_build_rejected_without_writes(self.first, report)

    def test_build_output_cannot_alias_inputs_or_cache_products(self):
        for path in [self.source, self.code / "rewrite.py", self.bin / "fatbinary.exe",
                     self.inputs / "fatbin_01.fatbin", *build.cache_paths(self.work)[:6]]:
            with self.subTest(path=path):
                self.assert_build_rejected_without_writes(path, self.root / "safe-report.json")

    def test_build_alias_checks_include_case_dotdot_and_hardlinks(self):
        self.assert_build_rejected_without_writes(self.first, str(self.source).upper())
        self.assert_build_rejected_without_writes(self.first, self.root / "unused" / ".." / "original.dll")
        alias = self.root / "hardlink-report.json"
        alias.hardlink_to(self.source)
        self.assert_build_rejected_without_writes(self.first, alias)
        output_alias = self.root / "hardlink-output.dll"
        output_alias.hardlink_to(self.source)
        self.assert_build_rejected_without_writes(output_alias, self.root / "safe-report.json")

    def test_verifier_report_cannot_replace_any_input_or_referenced_artifact(self):
        protected = [self.source, self.first, self.second, self.report1, self.report2, self.code / "flow.py",
                     self.work / "fatbin_01.sm86.cubin", self.work / "fatbin_01.sm86.fatbin.cache.json"]
        for path in protected:
            with self.subTest(path=path):
                before = self.snapshot()
                with patch.object(verify, "verify", return_value={}) as operation, self.assertRaises(ValueError):
                    verify.main(["--candidate", str(self.first), "--second", str(self.second), "--original", str(self.source),
                                 "--first-report", str(self.report1), "--second-report", str(self.report2), "--report", str(path)])
                operation.assert_not_called()
                self.assertEqual(self.snapshot(), before)
        alias = self.root / "verify.json.tmp"
        alias.hardlink_to(self.first)
        before = self.snapshot()
        with patch.object(verify, "verify", return_value={}), self.assertRaises(ValueError):
            verify.main(["--candidate", str(self.first), "--report", str(self.root / "verify.json")])
        self.assertEqual(self.snapshot(), before)

    def check_run(self, run, index=0):
        output = self.first if index == 0 else self.second
        return verify.validate_run(run, index, output, build.file_sha256(output), self.blocks, self.fatbins, self.payloads)

    def test_complete_fixture_reports_are_accepted(self):
        self.check_run(self.runs[0])
        self.check_run(self.runs[1], 1)

    def test_reports_require_independent_second_dll_and_report(self):
        with self.assertRaises(ValueError):
            verify.validate_selection(self.first, None, self.report1, self.report2)
        with self.assertRaises(ValueError):
            verify.validate_selection(self.first, self.first, self.report1, self.report2)
        with self.assertRaises(ValueError):
            verify.validate_selection(self.first, self.first, None, None)
        alias = self.root / "same-artifact.dll"
        alias.hardlink_to(self.first)
        with self.assertRaises(ValueError):
            verify.validate_selection(self.first, alias, self.report1, self.report2)
        with self.assertRaises(ValueError):
            verify.validate_selection(self.first, self.second, self.report1, self.report1)

    def test_per_row_cache_and_aggregate_must_agree(self):
        for mutate in (lambda run: run["fatbins"][0].update(cache="hit"), lambda run: run.update(cache_hits=15, cache_misses=0)):
            run = copy.deepcopy(self.runs[0])
            mutate(run)
            with self.assertRaises(ValueError):
                self.check_run(run)

    def test_missing_duplicate_failed_and_extra_stages_are_rejected(self):
        for kind in ("empty", "duplicate", "failed", "missing_full", "missing_total", "second_rewrite"):
            index = 1 if kind == "second_rewrite" else 0
            run = copy.deepcopy(self.runs[index])
            if kind == "empty": run["stages"] = []
            elif kind == "duplicate": run["stages"].append(copy.deepcopy(run["stages"][3]))
            elif kind == "failed": run["stages"][3]["status"] = "failed"
            elif kind == "missing_full": run["stages"].pop(0)
            elif kind == "missing_total": run["stages"].pop(1)
            else: run["stages"].append({"name": "fatbin_01/rewrite", "status": "passed", "wall_seconds": 1})
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check_run(run, index)

    def test_forged_current_input_hashes_and_options_are_rejected(self):
        for key in ("code/build_dll.py", "tool/ptxas.exe", "source_ptx", "source_fatbin", "original_cubin"):
            run = copy.deepcopy(self.runs[0])
            run["fatbins"][0]["inputs"]["files"][key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_run(run)
        run = copy.deepcopy(self.runs[0])
        run["fatbins"][0]["inputs"]["rewrite_options"]["mode"] = "f32"
        with self.assertRaises(ValueError):
            self.check_run(run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
