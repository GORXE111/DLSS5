"""Force cg2r's OKLab transfer branch and compare its isolated sm_86 kernel.

This checks an independent helper, not NGX parameter binding or native RTX 50
equivalence. No neural weights or D3D runtime are loaded. Default tolerances are
0.005/255 maximum and 0.0005/255 mean: a small float32 log/exp/FMA error budget,
far below one output code value. Shader output is compared as float32 RGB.
Previous sampler boundary failures remain as regression cases; passing the named
cases does not certify other GPUs, arbitrary descriptors, or NGX configuration.
Usage: python torch/check_transfer.py --report research/validation/check_transfer.json
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

from dlss5 import style

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import klab  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def texture_descriptor(img, rect, filtering):
    h, w = img.shape[:2]
    values = (0, 0, w, h, 1 / w, 1 / h) if rect is None else tuple(rect)
    if len(values) == 4:
        values += (1 / w, 1 / h)
    if len(values) != 6:
        raise ValueError("Invalid texture descriptor")
    handle, keep = klab.texture(img, linear=filtering == "linear")
    return struct.pack("<Q6f", handle, *values), keep


def run_kernel(nr, base=None, target=None, *, grid_size=None, base_subrect=None, target_subrect=None,
               base_filter="linear", target_filter="linear", enabled=True, style_id=0,
               intensity=1, control_mask=None, ui=None, ui_alpha=None, backbuffer=None, color=None,
               nr_texture=None, nr_subrect=None, nr_filter="point"):
    h, w = nr.shape[:2]
    gh, gw = grid_size if grid_size is not None else (target.shape[:2] if target is not None else nr.shape[:2])
    params = bytearray(376)
    keep = []
    out = klab.Surface(w, h)
    struct.pack_into("<Q6f", params, 0, out.handle, 0, 0, w, h, 1 / w, 1 / h)
    # +32 is intentionally unrelated to A/B, proving it is ignored when enabled.
    images = {32: (nr if nr_texture is None else nr_texture, nr_subrect, nr_filter), 224: (base, base_subrect, base_filter),
              256: (target, target_subrect, target_filter),
              64: (backbuffer if backbuffer is not None else color, None, "point"),
              128: (color if backbuffer is not None else None, None, "point"),
              96: (control_mask, None, "point"), 160: (ui, None, "point"), 192: (ui_alpha, None, "point")}
    for offset, (img, rect, filtering) in images.items():
        if img is not None:
            descriptor, buf = texture_descriptor(img, rect, filtering)
            params[offset:offset + 32] = descriptor
            keep.append(buf)
    struct.pack_into("<fiiiii", params, 288, intensity, w, h, int(enabled), gw, gh)
    grading = style.STYLES.get(style_id, {})
    for i, key in enumerate(("black", "white", "exposure", "gamma", "contrast", "saturation", "vibrance", "warm", "tint")):
        struct.pack_into("<f", params, 316 + 4 * i, grading.get(key, 1 if key == "white" else 0))
    struct.pack_into("<5f", params, 352, *grading.get("zones", (0,) * 5))
    klab.launch(klab.function("14", "cg2r_post_process_kernel"), ((w + 15) // 16, (h + 15) // 16, 1),
                (16, 16, 1), 0, bytes(params))
    return out.read()[..., :3]


def cases():
    rng = np.random.default_rng(2026102302)
    def image(h, w, scale=1, offset=0):
        return (rng.random((h, w, 3), dtype=np.float32) * scale + offset).astype(np.float32)
    nr, base, target = image(37, 53), image(37, 53), image(19, 27)
    rows = [
        ("point_same_grid", nr, base, image(37, 53), dict(grid_size=(37, 53), base_filter="point", target_filter="point")),
        ("point_low_odd", nr, base, target, dict(grid_size=(19, 27), base_filter="point", target_filter="point")),
        ("linear_same_grid", nr, base, image(37, 53), dict(grid_size=(37, 53))),
        ("linear_low_odd", nr, base, target, dict(grid_size=(19, 27))),
        ("point_large_grid_fallback", nr, image(31, 47), image(41, 61), dict(grid_size=(81, 117), base_filter="point", target_filter="point")),
        ("linear_large_grid_fallback", nr, image(31, 47), image(41, 61), dict(grid_size=(81, 117))),
        ("point_subrect", nr, image(43, 71), image(23, 39), dict(grid_size=(17, 25), base_subrect=(3, 2, 61, 37),
            target_subrect=(4, 3, 29, 17), base_filter="point", target_filter="point")),
        ("linear_subrect", nr, image(43, 71), image(23, 39), dict(grid_size=(17, 25), base_subrect=(3, 2, 61, 37), target_subrect=(4, 3, 29, 17))),
        ("point_outside_rect", nr, base, target, dict(grid_size=(19, 27), base_subrect=(-12, -8, 76, 55),
            target_subrect=(-7, -5, 41, 31), base_filter="point", target_filter="point")),
        ("linear_outside_rect", nr, base, target, dict(grid_size=(19, 27), base_subrect=(-12, -8, 76, 55), target_subrect=(-7, -5, 41, 31))),
        ("linear_explicit_descriptor", nr, base, target, dict(grid_size=(13, 21), base_subrect=(1.25, 2.5, 45, 29, 1/53, 1/37),
            target_subrect=(0.5, 1.25, 23, 15, 1/27, 1/19))),
        ("point_different_resource_sizes", image(45, 63), image(51, 79), image(11, 21), dict(grid_size=(9, 19), base_filter="point", target_filter="point")),
        ("linear_single_pixel", image(1, 1), image(3, 5), image(5, 3), dict(grid_size=(1, 1))),
        ("linear_single_row", image(1, 37), image(1, 53), image(1, 7), dict(grid_size=(1, 5))),
        ("point_single_column", image(37, 1), image(53, 1), image(7, 1), dict(grid_size=(5, 1), base_filter="point", target_filter="point")),
        ("equal_base_target", nr, base, base, dict(grid_size=(19, 27))),
        ("equal_dark_base_target", nr, np.full_like(base, 1e-6), np.full_like(base, 1e-6), dict(grid_size=(19, 27))),
        ("clamp_out_of_range_rgb", nr, image(37, 53, 3, -1), image(19, 27, 3, -1), dict(grid_size=(19, 27))),
        ("dark_to_bright", nr, image(37, 53, 1e-4), image(19, 27), dict(grid_size=(19, 27))),
        ("bright_to_dark", nr, base, image(19, 27, 1e-4), dict(grid_size=(19, 27))),
        ("flag_disabled", nr, base, target, dict(grid_size=(19, 27), enabled=False)),
        ("missing_base_disabled", nr, None, target, dict(grid_size=(19, 27))),
        ("missing_target_disabled", nr, base, None, dict(grid_size=(19, 27))),
        ("zero_grid_disabled", nr, base, target, dict(grid_size=(0, 27))),
        ("negative_grid_disabled", nr, base, target, dict(grid_size=(19, -1))),
    ]
    for sid in (0, 1, 2):
        rows.append((f"compose_style{sid}_ui_mask", nr, base, target,
                     dict(grid_size=(19, 27), composition=dict(style_id=sid, color=image(37, 53), backbuffer=image(37, 53),
                        ui=rng.random((37,53,4),dtype=np.float32), ui_alpha=image(37,53)[...,:1],
                        control_mask=rng.random((37,53,4),dtype=np.float32), intensity=0.6))))
    for sign in (-1, 1):
        rows.append((f"point_outside_neighbor_{sign:+d}", nr, base, target,
                     dict(grid_size=(19, 27), base_subrect=(-12 + sign/1024, -8, 76, 55),
                          target_subrect=(-7 + sign/1024, -5, 41, 31), base_filter="point", target_filter="point")))
    for grid in ((80, 116), (82, 118)):
        rows.append((f"linear_large_grid_neighbor_{grid[0]}x{grid[1]}", nr, image(31, 47), image(41, 61),
                     dict(grid_size=grid)))
    rows.append(("mixed_filter_linear_point", nr, base, target,
                 dict(grid_size=(19, 27), base_filter="linear", target_filter="point")))
    rows.append(("mixed_filter_point_linear", nr, base, target,
                 dict(grid_size=(19, 27), base_filter="point", target_filter="linear")))
    return rows


def run_cases(report):
    for name, nr, base, target, options in cases():
        options = dict(options)
        composition = options.pop("composition", None)
        ref = run_kernel(nr, base, target, **options, **(composition or {}))
        tx = lambda img: None if img is None else torch.from_numpy(img).cuda()
        out = style.transfer(tx(nr), tx(base), tx(target), **options)
        if composition:
            out = style.apply(tx(composition["color"]), out, composition["style_id"], composition["intensity"],
                              tx(composition["control_mask"]), True, tx(composition["ui"]),
                              tx(composition["ui_alpha"]), tx(composition["backbuffer"]))
        out = out.cpu().numpy()
        if not np.isfinite(ref).all() or not np.isfinite(out).all():
            raise ValueError(f"Non-finite result: {name}")
        diff = np.abs(out.astype(np.float64) - ref.astype(np.float64)) * 255
        row = dict(name=name, options=options, output_shape=list(out.shape),
                   base_shape=None if base is None else list(base.shape), target_shape=None if target is None else list(target.shape),
                   mae_255=float(diff.mean()), max_255=float(diff.max()),
                   p99_255=float(np.percentile(diff, 99)), p999_255=float(np.percentile(diff, 99.9)),
                   within_max_budget_fraction=float((diff <= report["thresholds"]["max_255"]).mean()),
                   reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                   output_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                   input_sha256={key: None if img is None else hashlib.sha256(img.tobytes()).hexdigest()
                                 for key, img in (("nr", nr), ("base", base), ("target", target))})
        if composition:
            row["composition"] = {key: value for key, value in composition.items() if not isinstance(value, np.ndarray)}
            row["input_sha256"].update({key: hashlib.sha256(value.tobytes()).hexdigest()
                                         for key, value in composition.items() if isinstance(value, np.ndarray)})
        row["passed"] = row["max_255"] <= report["thresholds"]["max_255"] and row["mae_255"] <= report["thresholds"]["mae_255"]
        report["checks"].append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name:33s} max={row['max_255']:.6f}/255 mean={row['mae_255']:.6f}/255", flush=True)


def sampler_probes(report):
    """Independent original-kernel probes of axis and cross-weight half-way ties."""
    src = np.zeros((2, 2, 3), np.float32)
    src[1, 1, 0] = 1  # cross weight
    src[..., 1] = np.array([[0, 1], [0, 1]])  # x marginal
    src[..., 2] = np.array([[0, 0], [1, 1]])  # y marginal
    for kind, values in (("axis_half_up", (0, 1, 2, 127, 128, 254)), ("cross_half_up", (1, 3, 5, 127, 129, 255))):
        for value in values:
            rect = ((value + 0.5) / 256, 0.5, 1, 1) if kind.startswith("axis") else (value / 256, 0.5, 1, 1)
            ref = run_kernel(np.zeros((1, 1, 3), np.float32), enabled=False,
                             nr_texture=src, nr_subrect=rect, nr_filter="linear")
            u = torch.full((1, 1), 0.5, device="cuda")
            out = style._transfer_texture(torch.from_numpy(src).cuda(), u, u, rect, "linear").cpu().numpy()
            diff = np.abs(out.astype(np.float64) - ref.astype(np.float64)) * 255
            report["sampler_probes"].append(dict(name=f"{kind}_{value}", descriptor=rect,
                reference_rgb=ref[0, 0].tolist(), helper_rgb=out[0, 0].tolist(), max_255=float(diff.max()),
                passed=bool(diff.max() <= report["thresholds"]["max_255"]),
                reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(), output_sha256=hashlib.sha256(out.tobytes()).hexdigest()))
    ramp = np.repeat((np.arange(53, dtype=np.float32) / 64)[None, :, None], 3, axis=-1)
    points = [("minus_1_over_1024", 26 - 1/1024),
              ("minus_1_ulp", float(np.nextafter(np.float32(26), np.float32(-np.inf)))),
              ("integer", 26.0),
              ("plus_1_ulp", float(np.nextafter(np.float32(26), np.float32(np.inf)))),
              ("plus_1_over_1024", 26 + 1/1024)]
    for name, coordinate in points:
        rect = (coordinate, 0, 0, 1)
        ref = run_kernel(np.zeros((1, 1, 3), np.float32), enabled=False,
                         nr_texture=ramp, nr_subrect=rect, nr_filter="point")
        u = torch.full((1, 1), 0.5, device="cuda")
        out = style._transfer_texture(torch.from_numpy(ramp).cuda(), u, u, rect, "point").cpu().numpy()
        diff = np.abs(out.astype(np.float64) - ref.astype(np.float64)) * 255
        report["sampler_probes"].append(dict(name=f"point_boundary_53_{name}", descriptor=rect,
            reference_texel_index=float(ref[0, 0, 0] * 64), helper_texel_index=float(out[0, 0, 0] * 64),
            reference_rgb=ref[0, 0].tolist(), helper_rgb=out[0, 0].tolist(), max_255=float(diff.max()),
            passed=bool(diff.max() <= report["thresholds"]["max_255"]),
            reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(), output_sha256=hashlib.sha256(out.tobytes()).hexdigest()))


# This probe is authored here, not extracted from an NVIDIA kernel. It bypasses
# cg2r's divisions, descriptor affine transforms, and grading, exposing only TEX.
_RAW_TEX_PROBE = b""".version 8.0
.target sm_86
.address_size 64
.visible .entry raw_tex_probe(.param .align 8 .b8 cfg[32]) {
.reg .pred %p;
.reg .b32 %r<5>;
.reg .b64 %rd<8>;
.reg .f32 %f<6>;
ld.param.u64 %rd0,[cfg];
ld.param.u64 %rd1,[cfg+8];
ld.param.u64 %rd2,[cfg+16];
ld.param.u32 %r3,[cfg+24];
mov.u32 %r0,%ctaid.x;
mov.u32 %r1,%ntid.x;
mov.u32 %r2,%tid.x;
mad.lo.u32 %r0,%r0,%r1,%r2;
setp.ge.u32 %p,%r0,%r3;
@%p bra done;
mul.wide.u32 %rd3,%r0,8;
add.u64 %rd4,%rd1,%rd3;
ld.global.v2.f32 {%f0,%f1},[%rd4];
tex.2d.v4.f32.f32 {%f2,%f3,%f4,%f5},[%rd2,{%f0,%f1}];
mul.wide.u32 %rd5,%r0,16;
add.u64 %rd6,%rd0,%rd5;
st.global.v4.f32 [%rd6],{%f2,%f3,%f4,%f5};
done: ret;
}
"""


def normalized_coordinate_probes(report):
    """Compare candidate precision models and the helper against independent TEX."""
    klab.ctx()
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    klab.ck(klab.cu.cuModuleLoadData(ctypes.byref(module), _RAW_TEX_PROBE))
    rows, model_mismatches = [], {str(bits): 0 for bits in range(16, 27)}
    try:
        klab.ck(klab.cu.cuModuleGetFunction(ctypes.byref(function), module, b"raw_tex_probe"))
        for axis in (0, 1):
            for size in (2, 3, 7, 17, 31, 37, 53, 64, 127, 257, 1023):
                values = []
                for texel in sorted({1, size // 4, size // 2, size - 1}):
                    if not 0 < texel < size:
                        continue
                    bits = np.float32(texel / size).view(np.uint32).item()
                    values.extend(np.array(bits + step, dtype=np.uint32).view(np.float32).item()
                                  for step in range(-40, 41))
                positions = np.asarray(values, np.float32)
                coords = np.full((len(positions), 2), 0.5, np.float32)
                coords[:, axis] = positions
                ramp = np.repeat(np.arange(size, dtype=np.float32)[None, :, None], 3, axis=-1)
                if axis == 1:
                    ramp = ramp.transpose(1, 0, 2).copy()
                texture, keep = klab.texture(ramp, linear=False)
                tc = torch.from_numpy(coords).cuda()
                output = torch.zeros(len(coords), 4, device="cuda")
                try:
                    klab.launch(function, ((len(coords) + 127) // 128, 1, 1), (128, 1, 1), 0,
                                struct.pack("<QQQI4x", output.data_ptr(), tc.data_ptr(), texture, len(coords)))
                    ref = output[:, :3].cpu().numpy()
                    out = style._transfer_texture(torch.from_numpy(ramp).cuda(), tc[:, 0], tc[:, 1],
                                                   (0, 0, 1, 1, 1, 1), "point").cpu().numpy()
                    mismatches = int((out != ref).any(-1).sum())
                    models = {}
                    for fractional_bits in range(16, 27):
                        unit = 2 ** fractional_bits
                        expected = np.floor(np.floor(positions.astype(np.float64) * unit) / unit * size)
                        expected = expected.clip(0, size - 1)
                        models[str(fractional_bits)] = int((expected != ref[:, 0]).sum())
                        model_mismatches[str(fractional_bits)] += models[str(fractional_bits)]
                    rows.append(dict(axis="xy"[axis], resource_extent=size, samples=len(coords),
                                     helper_mismatches=mismatches, models=models, passed=mismatches == 0,
                                     input_coordinates_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
                                     reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                                     output_sha256=hashlib.sha256(out.tobytes()).hexdigest()))
                finally:
                    klab.ck(klab.cu.cuTexObjectDestroy(ctypes.c_uint64(texture)))
        report["normalized_coordinate_probes"] = dict(
            authored_ptx_sha256=hashlib.sha256(_RAW_TEX_PROBE).hexdigest(), checks=rows,
            samples=sum(row["samples"] for row in rows), model_mismatches=model_mismatches,
            passed=all(row["passed"] for row in rows) and model_mismatches["21"] == 0,
            claim="Empirical normalized-coordinate truncation to 21 fractional bits on the recorded RTX 3060; not an API guarantee")
    finally:
        klab.ck(klab.cu.cuModuleUnload(module))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "research" / "validation" / "check_transfer.json")
    parser.add_argument("--baseline", type=Path, help="Prior report to bind old/new source hashes and case results")
    args = parser.parse_args()
    report = dict(scope="Forced cg2r transfer kernel on RTX 3060; NGX resource binding/filter state and native RTX 50 parity unverified",
                  implementation_status="experimental; measured RTX 3060 sampler contract, NGX binding unverified",
                  thresholds=dict(max_255=0.005, mae_255=0.0005), checks=[], sampler_probes=[], errors=[], passed=False)
    report["sampler_model"] = dict(normalized_coordinate_fraction_bits=21,
        coordinate_rounding="truncate downward before multiplying resource dimensions", weight_fraction_bits=8,
        weight_rounding="axis and cross weight half-way ties round upward",
        scope="Empirically measured on the reported RTX 3060; not a cross-GPU or API guarantee")
    report["documentation"] = dict(
        url="https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#linear-filtering",
        documented="CUDA describes axis interpolation coefficients with eight fractional bits",
        empirical="The 21-bit coordinate model, cross-weight quantization and tie behavior are measured separately here")
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    try:
        save()  # Invalidate an earlier successful report before any CUDA access.
        report["kernel_sha256"] = digest(ROOT / "sm86_port" / "fatbin_14.sm86.cubin")
        report["helper_sha256"] = digest(ROOT / "torch" / "dlss5" / "style.py")
        report["test_sha256"] = digest(Path(__file__))
        if args.baseline:
            previous = json.loads(args.baseline.read_text(encoding="utf-8"))
            report["baseline"] = dict(path=str(args.baseline.resolve()), report_sha256=digest(args.baseline),
                                      helper_sha256=previous["helper_sha256"], test_sha256=previous["test_sha256"],
                                      previous_passed=previous["passed"])
            report["baseline"]["source_hashes_changed"] = {
                key: previous[key] != report[key] for key in ("helper_sha256", "test_sha256")}
        report["gpu"] = torch.cuda.get_device_name()
        report["torch_version"] = torch.__version__
        run_cases(report)
        sampler_probes(report)
        normalized_coordinate_probes(report)
        report["passed"] = (bool(report["checks"]) and all(row["passed"] for row in report["checks"] + report["sampler_probes"])
                            and report["normalized_coordinate_probes"]["passed"])
        report["coverage"] = {"transfer_checks_passed": sum(row["passed"] for row in report["checks"]),
                              "transfer_checks_total": len(report["checks"]),
                              "sampler_probes_passed": sum(row["passed"] for row in report["sampler_probes"]),
                              "sampler_probes_total": len(report["sampler_probes"])}
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["passed"] = False
    if not report["passed"] and not report["errors"]:
        report["errors"].append("One or more cases exceeded the fixed float32 error budget")
    try:
        save()
    except (ValueError, OSError) as error:
        print(f"FAIL: cannot save report: {error}", file=sys.stderr)
        return 1
    for error in report["errors"]:
        print(f"FAIL: {error}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
