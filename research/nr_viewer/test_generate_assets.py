"""Small CPU fixtures; no retained captures, binaries, browser, or GPU required."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from generate_assets import capture_pair, demo_pair, generate, read_ppm


def write_ppm(path, width, height, color):
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + bytes(color) * (width * height))


class ViewerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix=".nr-viewer-test-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_demo_generates_lossless_images_and_relative_dataset(self):
        spec = {"title": "合成", "datasets": [{"id": "demo", "label": "demo", "stage": "demo"}]}
        page = generate(spec, self.root, self.root / "bundle", demo=True)
        self.assertIn('src="./datasets.js"', page.read_text(encoding="utf-8"))
        js = page.with_name("datasets.js").read_text(encoding="utf-8")
        bundle = json.loads(js.removeprefix("window.NR_DATA = ").removesuffix(";\n"))
        d = bundle["datasets"][0]
        self.assertEqual(bundle["title"], "合成")
        self.assertIsNone(d["present"])
        self.assertGreater(d["mae"], 0)
        a, b, _ = demo_pair()
        for name, expected in (("before", a[0]), ("after", b[0]), ("absolute", None), ("signed", None)):
            self.assertTrue(d[name].startswith("./assets/"))
            with Image.open(page.parent / d[name]) as image:
                self.assertEqual(image.size, (384, 240))
                if expected is not None:
                    np.testing.assert_array_equal(np.asarray(image), expected)
                else:
                    self.assertGreater(len(image.getcolors()), 1)
        evidence = json.loads((page.parent / "assets/evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["datasets"][0]["origin"], "synthetic_demo")
        self.assertNotIn(str(self.root).replace("\\", "/"), js)

    def test_same_present_number_and_model_dimensions(self):
        self.root.joinpath("dlss5fb.ini").write_text("[DLSS5]\nDumpFrame=40\nDumpCount=2\n")
        log = (
            "swap chain TEST tracked (D3D12 queue TEST)\n"
            "resize: back buffer 4x2 fmt=28 -> NR 2x1 (scale 0.500)\n"
            "CreateFeature(18) 2x1 style=0 = 0x00000001 seh=0x00000000 handle=TEST\n"
            "dumped dlss5fb_in_1.ppm (4x2)\n"
            "dumped dlss5fb_out_1.ppm (4x2)\n"
            "dumped dlss5fb_lo_1.ppm (2x1)\n"
            "dumped dlss5fb_nr_1.ppm (2x1)\n"
            "swap chain TEST released (42 frames)\n"
        )
        log_path = self.root / "dlss5fb.log"
        log_path.write_text(log, encoding="utf-8")
        for key, size, color in (("in", (4, 2), (10, 20, 30)), ("out", (4, 2), (20, 30, 40)),
                                 ("lo", (2, 1), (50, 60, 70)), ("nr", (2, 1), (40, 50, 60))):
            write_ppm(self.root / f"dlss5fb_{key}_1.ppm", *size, color)
        # Stand-in bytes exercise identity handling without a real runtime.
        fixture = b"identity fixture"
        for name in ("dxgi.dll", "nvngx_dlssnr.dll", "dlss5_nvngx.dll"):
            (self.root / name).write_bytes(fixture)
        pin = hashlib.sha256(fixture).hexdigest()
        a, b, evidence = capture_pair(self.root, "model", 1, pin)
        self.assertEqual(a[0].shape, (1, 2, 3))
        self.assertEqual(b[0][0, 0].tolist(), [40, 50, 60])
        self.assertEqual(evidence["present"], 41)
        self.assertTrue(evidence["model_sha256_pinned"])
        self.assertEqual(capture_pair(self.root, "present", 1)[0][0].shape, (2, 4, 3))
        with self.assertRaisesRegex(ValueError, "expected SHA-256"):
            capture_pair(self.root, "model", 1, "0" * 64)
        log_path.write_text(log + "dumped dlss5fb_nr_1.ppm (2x1)\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            capture_pair(self.root, "model", 1)

    def test_reject_truncated_ppm_without_publishing(self):
        a = self.root / "a.ppm"
        a.write_bytes(b"P6\n1 1\n255\n\x01\x02")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            read_ppm(a)
        spec = {"datasets": [{"id": "pair", "label": "pair", "before": "a.ppm", "after": "a.ppm"}]}
        output = self.root / "bundle"
        with self.assertRaisesRegex(ValueError, "Truncated"):
            generate(spec, self.root, output)
        self.assertFalse(output.exists())

    def test_existing_output_preserves_user_file(self):
        output = self.root / "bundle"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_bytes(b"preserve")
        spec = {"datasets": [{"id": "demo", "label": "demo", "stage": "demo"}]}
        with self.assertRaises(FileExistsError):
            generate(spec, self.root, output, demo=True)
        self.assertEqual(sentinel.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
