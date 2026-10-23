"""GPU scheduling/lifetime checks for the authored reciprocal shim.

This checks stream ordering, graph replay and cold-capture rejection; it is not
an independent numerical reference for the NVIDIA sampling kernels.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch

from dlss5 import ptx_math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from report_paths import ReportPath


def probe(destination):
    report = {"passed": False, "scope": __doc__, "checks": [], "errors": []}
    def save():
        destination.write_json(report)
    save()
    try:
        report["sources"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__), ROOT / "torch/dlss5/ptx_math.py", ROOT / "tools/report_paths.py")}
        report["device"] = torch.cuda.get_device_name(0)
        source = torch.full((8193,), 2., device="cuda")
        torch.cuda.synchronize()
        if ptx_math._modules:
            raise RuntimeError("Run this probe in a fresh process to test cold capture")
        cold = torch.cuda.CUDAGraph()
        rejected = False
        try:
            with torch.cuda.graph(cold):
                ptx_math.rcp_approx(source)
        except RuntimeError as error:
            if "Warm up rcp_approx" not in str(error):
                raise
            rejected = True
        report["checks"].append(dict(name="cold_capture_rejects_JIT", passed=rejected))
        ptx_math.rcp_approx(source)  # Compile before any warm graph recording.
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.cuda._sleep(8000000)
            source.fill_(37.)
            output = ptx_math.rcp_approx(source)
        torch.cuda.current_stream().wait_stream(side)
        reference = ptx_math.rcp_approx(source)
        report["checks"].append(dict(name="nondefault_stream_waits_for_input_write",
                                     passed=bool(torch.equal(output, reference)),
                                     reciprocal_37=float(output[0])))
        # The captured function keeps the same module/function handles. New
        # values at the same input address must reach replay's current stream.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = ptx_math.rcp_approx(source)
        for value in (1.0625, 1.124755859375):
            source.fill_(value)
            graph.replay()
            reference = ptx_math.rcp_approx(source)
            report["checks"].append(dict(name="warm_graph_new_input", value=value,
                                         passed=bool(torch.equal(captured, reference)),
                                         output_sha256=hashlib.sha256(captured.cpu().numpy().tobytes()).hexdigest()))
        report["passed"] = all(row["passed"] for row in report["checks"])
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        save()
    return 0 if report["passed"] and not report["errors"] else 1


def main(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--report", type=Path, default=ROOT / "research/batch4_sampling/rcp_stream_graph.json")
    early, _ = preliminary.parse_known_args(argv)
    try:
        output = ReportPath(early.report, inputs=[ROOT / "research/batch4_sampling/post_arithmetic_static.json",
                                                ROOT / "research/validation/batch4_sampling.json"], root=ROOT)
        if not any(item in ("-h", "--help") for item in (sys.argv[1:] if argv is None else argv)):
            output.write_json(dict(passed=False, errors=["CLI or probe has not completed"]))
    except (ValueError, OSError) as error:
        print(f"FAIL unsafe report: {error}", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.parse_args(argv)
    return probe(output)


if __name__ == "__main__":
    raise SystemExit(main())
