"""Exact named buffers from one immutable capture manifest, shared by legacy gates."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.schedule_refs import digest, recorded_runtime, safe_path, validate  # noqa: E402


class ScheduleReference:
    def __init__(self, folder, harness=None):
        if folder is None:
            raise ValueError("Explicit --capture is required; legacy tap-directory guessing is disabled")
        self.folder = Path(folder).resolve()
        self.data = validate(self.folder, expected_runtime=recorded_runtime(self.folder, harness))

    def path(self, seq, size, *, shape=None, layout=None):
        names = [name for name, desc in self.data["plan"]["buffers"].items()
                 if desc["seq"] == seq and desc["bytes"] == size and
                 (shape is None or desc["shape_hwc"] == list(shape)) and
                 (layout is None or desc["layout"] == layout)]
        if len(names) != 1:
            raise ValueError(f"Expected one exact declared buffer for seq={seq}, bytes={size}; found {names}")
        return safe_path(self.folder, self.data["buffers"][names[0]]["path"])

    def annotate(self, report):
        report.data["reference_capture"] = {"path": str(self.folder), "manifest_sha256": digest(self.folder / "manifest.json"),
                                            "runtime_sha256": self.data["runtime_sha256"],
                                            "capture_source_sha256": self.data["source_sha256"],
                                            "capture_tool_sha256": self.data["capture_tool_sha256"],
                                            "official_reference": False}


def arguments(parser):
    parser.add_argument("--capture", type=Path, help="Fresh schedule_refs capture; required for execution")
    parser.add_argument("--harness", type=Path, help="Optional explicit matching executable; otherwise verify the frozen runtime copy")
    return parser
