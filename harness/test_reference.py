"""CPU-only tests for evidence integrity and false-pass prevention."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="dlss5-reference-test-")
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def bundle(self, name, profile=None):
        root = self.root / name
        root.mkdir()
        rows = []
        for case in reference.profiled_cases("smoke", profile):
            folder = root / case["id"]
            folder.mkdir()
            fields, names = reference.case_contract(case)
            self.assertEqual(len(names), 1)
            output = folder / next(iter(names))
            output.write_bytes(b"P6\n640 360\n255\n" + b"\x0a\x20\xff" * (640 * 360))
            reference.write_json(folder / "nr-lab-result.json", {**fields, "outputCoverageValidated": True})
            for name in ("nr-lab.log", "stdout.log", "stderr.log"):
                (folder / name).write_text("GPU: NVIDIA GeForce RTX 5090 (vendor=10DE)\n" if profile else "fixture\n")
            evidence = {name: reference.digest(folder / name) for name in ("nr-lab-result.json", "nr-lab.log", "stdout.log", "stderr.log")}
            rows.append({**case, "passed": True, "outputs": {output.name: reference.digest(output)}, "evidence": evidence})
        manifest = {"schema": reference.SCHEMA, "passed": True, "suite": "smoke", "reference_kind": "port-sm86",
                    "runtime_sha256": {key: "test" for key in ("nr-lab.exe", "nvngx.dll", "nvngx_dlss.dll", "nvngx_dlssg.dll")},
                    "cases": rows}
        if profile:
            manifest.update(capture_profile=profile, target=f"DLSS-NR {reference.OCTOBER_VERSION}",
                            reference_kind="official-sm120",
                            gpus=[{"compute_capability": "12.0", "name": "NVIDIA GeForce RTX 5090"}])
            manifest["runtime_sha256"]["nvngx_dlssnr.dll"] = reference.OCTOBER_SHA256
        reference.write_json(root / "manifest.json", manifest)
        return root, manifest

    def test_ppm_preserves_whitespace_valued_first_pixels(self):
        folder, _ = self.bundle("ppm")
        image = reference.read_ppm(folder / "reset" / "nr-lab-output-model1-srgb.ppm")
        self.assertEqual(image[0, 0].tolist(), [10, 32, 255])

    def test_incomplete_suite_is_rejected(self):
        folder, manifest = self.bundle("incomplete")
        manifest["cases"].pop()
        reference.write_json(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "coverage"):
            reference.load_capture(folder)

    def test_tampered_pixels_are_rejected(self):
        folder, _ = self.bundle("tampered")
        (folder / "reset" / "nr-lab-output-model1-srgb.ppm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        with self.assertRaisesRegex(ValueError, "modified"):
            reference.load_capture(folder)

    def test_native_label_requires_original_dll_and_blackwell(self):
        ampere = [{"compute_capability": "8.6", "name": "NVIDIA GeForce RTX 3060"}]
        with self.assertRaisesRegex(ValueError, "unmodified"):
            reference.verify_native(ampere, "modified")
        with self.assertRaisesRegex(ValueError, "Blackwell"):
            reference.verify_native(ampere, reference.ORIGINAL_SHA256)
        blackwell = [{"compute_capability": "12.0", "name": "NVIDIA GeForce RTX 5090"}]
        with self.assertRaisesRegex(ValueError, "execution log"):
            reference.verify_native(blackwell, reference.ORIGINAL_SHA256, "GPU: NVIDIA GeForce RTX 3060 (vendor=10DE)")

    def test_october_requires_explicit_model_and_native_kind(self):
        for preset in (None, True, 0, 3):
            with self.assertRaisesRegex(ValueError, "explicit --model-preset"):
                reference.capture_profile("official-sm120", reference.OCTOBER_VERSION, preset)
        with self.assertRaisesRegex(ValueError, "no validated sm86 port"):
            reference.capture_profile("port-sm86", reference.OCTOBER_VERSION, 2)

    def test_october_accepts_only_its_original_dll_on_blackwell(self):
        blackwell = [{"compute_capability": "12.0", "name": "NVIDIA GeForce RTX 5090"}]
        reference.verify_native(blackwell, reference.OCTOBER_SHA256, nr_version=reference.OCTOBER_VERSION)
        with self.assertRaisesRegex(ValueError, "unmodified"):
            reference.verify_native(blackwell, reference.ORIGINAL_SHA256, nr_version=reference.OCTOBER_VERSION)
        with self.assertRaisesRegex(ValueError, "unmodified"):
            reference.verify_native(blackwell, reference.OCTOBER_SHA256)

    def test_october_preset_2_capture_rejects_model_1_case(self):
        profile = reference.capture_profile("official-sm120", reference.OCTOBER_VERSION, 2)
        root, manifest = self.bundle("october", profile)
        valid = reference.load_capture(root)
        self.assertEqual(valid["capture_profile"]["statically_mapped_weight_resource"], "WEIGHTS_CC_SMALL_FP8")
        self.assertEqual(valid["cases"][0]["args"][-2:], ["--model", "2"])
        manifest["cases"][0]["args"][-1] = "1"
        reference.write_json(root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "coverage"):
            reference.load_capture(root)

    def test_october_profile_cannot_be_relabelled_or_omitted(self):
        profile = reference.capture_profile("official-sm120", reference.OCTOBER_VERSION, 2)
        root, manifest = self.bundle("october", profile)
        manifest["capture_profile"]["statically_mapped_weight_resource"] = "WEIGHTS_HT"
        reference.write_json(root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "inconsistent capture profile"):
            reference.load_capture(root)
        manifest.pop("capture_profile")
        manifest["target"] = "DLSS-NR 310.8.0"
        manifest["reference_kind"] = "port-sm86"
        reference.write_json(root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "October DLL"):
            reference.load_capture(root)

    def test_october_comparison_rejects_different_presets(self):
        left, right = [self.bundle(f"preset{n}", reference.capture_profile("official-sm120", reference.OCTOBER_VERSION, n))[0]
                       for n in (1, 2)]
        report = self.root / "comparison.json"
        args = argparse.Namespace(reference=left, candidate=right, report=report, criterion="bit-exact", max_mae=None, min_psnr=None)
        with self.assertRaisesRegex(ValueError, "requested model presets differ"):
            reference.compare(args)
        self.assertFalse(json.loads(report.read_text())["passed"])

    def test_nonfinite_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reference.metrics(np.array([float("nan")]), np.array([0.0]))

    def test_signed_zero_is_not_bit_exact(self):
        self.assertFalse(reference.metrics(np.array([-0.0], dtype=np.float16), np.array([0.0], dtype=np.float16))["bit_exact"])

    def test_port_comparison_cannot_claim_official_reference(self):
        left, _ = self.bundle("left")
        right, _ = self.bundle("right")
        report = self.root / "comparison.json"
        args = argparse.Namespace(reference=left, candidate=right, report=report, criterion="bit-exact", max_mae=None, min_psnr=None)
        self.assertEqual(reference.compare(args), 0)
        self.assertFalse(json.loads(report.read_text())["official_reference"])

    def test_real_pixel_regression_exits_failure(self):
        left, _ = self.bundle("left")
        right, manifest = self.bundle("right")
        path = right / "reset" / "nr-lab-output-model1-srgb.ppm"
        path.write_bytes(b"P6\n640 360\n255\n" + b"\x0a\x20\xfe" * (640 * 360))
        manifest["cases"][0]["outputs"][path.name] = reference.digest(path)
        reference.write_json(right / "manifest.json", manifest)
        args = argparse.Namespace(reference=left, candidate=right, report=self.root / "difference.json", criterion="bit-exact", max_mae=None, min_psnr=None)
        self.assertEqual(reference.compare(args), 1)

    def test_wrong_report_dimensions_cannot_pass(self):
        case = reference.suite_cases("full")[0]
        fields, names = reference.case_contract(case)
        fields["output"] = {"width": 1, "height": 1}
        fields["outputCoverageValidated"] = True
        with self.assertRaisesRegex(ValueError, "requested contract"):
            reference.validate_report(case, fields, names)

    def test_hdr_requires_raw_output(self):
        case = reference.suite_cases("full")[-1]
        fields, names = reference.case_contract(case)
        fields["outputCoverageValidated"] = True
        with self.assertRaisesRegex(ValueError, "inventory"):
            reference.validate_report(case, fields, {name for name in names if name.endswith(".ppm")})

    def test_modified_result_sidecar_cannot_pass(self):
        folder, _ = self.bundle("sidecar")
        (folder / "reset" / "nr-lab-result.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence was modified"):
            reference.load_capture(folder)

    def test_invalid_capture_invalidates_previous_success_report(self):
        left, _ = self.bundle("left")
        right, _ = self.bundle("right")
        report = self.root / "old_success.json"
        reference.write_json(report, {"passed": True})
        (right / "reset" / "nr-lab.log").unlink()
        args = argparse.Namespace(reference=left, candidate=right, report=report, criterion="bit-exact", max_mae=None, min_psnr=None)
        with self.assertRaises(OSError):
            reference.compare(args)
        self.assertFalse(json.loads(report.read_text())["passed"])

    def test_temporal_suite_requires_scene_and_reset_fields(self):
        case = next(c for c in reference.suite_cases("temporal") if c["id"] == "cut_reset_f3")
        fields, names = reference.case_contract(case)
        fields["outputCoverageValidated"] = True
        reference.validate_report(case, fields, names)
        fields["resetFrame"] = -1
        with self.assertRaisesRegex(ValueError, "requested contract"):
            reference.validate_report(case, fields, names)

    def test_temporal_events_reject_missing_reset_and_wrong_scene(self):
        case = next(c for c in reference.suite_cases("temporal") if c["id"] == "cut_reset_f3")
        log = "\n".join(f"temporal input: frame={f} scene={int(f >= 2)} reset={int(f in (0, 2))}" for f in range(4))
        reference.validate_temporal_log(case, log)
        for bad in (log.replace("frame=2 scene=1 reset=1", "frame=2 scene=1 reset=0"),
                    log.replace("frame=3 scene=1", "frame=3 scene=0"), log.splitlines()[0]):
            with self.assertRaisesRegex(ValueError, "temporal input/reset"):
                reference.validate_temporal_log(case, bad)

    def test_unknown_suite_cannot_silently_become_smoke(self):
        with self.assertRaisesRegex(ValueError, "unknown suite"):
            reference.suite_cases("typo")

    def test_motion2d_contract_checks_vertical_shift_and_intensity(self):
        cases = reference.suite_cases("motion2d")
        self.assertEqual(len(cases), 19)
        self.assertEqual(len({c["id"] for c in cases}), 19)
        case = next(c for c in cases if c["id"] == "input_diagonal_i0")
        fields, names = reference.case_contract(case)
        fields["outputCoverageValidated"] = True
        self.assertEqual(fields["temporalShiftYPixels"], 2)
        self.assertEqual(fields["intensity"], 0)
        reference.validate_report(case, fields, names)
        for key, wrong in (("temporalShiftYPixels", 0), ("intensity", 1)):
            with self.assertRaisesRegex(ValueError, "requested contract"):
                reference.validate_report(case, {**fields, key: wrong}, names)


if __name__ == "__main__":
    unittest.main()
