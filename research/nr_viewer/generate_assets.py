"""Build an offline NR comparison bundle from local PPMs (CPU only)."""
import argparse
from configparser import ConfigParser, Error as ConfigError
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


RGB = np.array([.2126, .7152, .0722], dtype=np.float64)
LIMIT = 40
MAX_PIXELS = 8192 * 4320
ABS_COLORS = np.array([[12, 15, 25], [55, 34, 105], [117, 42, 131], [192, 56, 111],
                       [238, 102, 70], [253, 168, 58], [252, 239, 159]])
SIGNED_COLORS = np.array([[98, 194, 255], [28, 119, 196], [35, 60, 98], [28, 31, 41],
                          [109, 53, 35], [214, 95, 48], [255, 199, 115]])
PPM = re.compile(rb"\AP6\r?\n([1-9][0-9]{0,5}) ([1-9][0-9]{0,5})\r?\n255\r?\n")
DUMP = re.compile(r"\bdumped (dlss5fb_(?:in|out|lo|nr)(?:_\d+)?\.ppm) \((\d+)x(\d+)\)$")
SIZE = re.compile(r"\bback buffer (\d+)x(\d+) fmt=\d+ -> NR (\d+)x(\d+) \(scale [\d.]+\)$")
FEATURE = re.compile(r"\bCreateFeature\(18\) (\d+)x(\d+) .* = 0x00000001 seh=0x00000000\b")
RELEASE = re.compile(r"\bswap chain (\S+) released \((\d+) frames\)$")
TRACKED = re.compile(r"\bswap chain (\S+) tracked \(")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_ppm(path):
    # This is the format emitted by the SDR fallback dump, not a general PNM reader.
    with path.open("rb") as stream:
        header = stream.readline(64) + stream.readline(64) + stream.readline(64)
        match = PPM.fullmatch(header)
        if not match:
            raise ValueError(f"Expected an 8-bit P6 dump: {path.name}")
        width, height = map(int, match.groups())
        if width * height > MAX_PIXELS:
            raise ValueError(f"PPM is larger than {MAX_PIXELS} pixels: {path.name}")
        n = width * height * 3
        payload = stream.read(n + 1)
        if len(payload) != n:
            raise ValueError(f"Truncated or extra PPM pixels: {path.name}")
    source_hash = hashlib.sha256(header)
    source_hash.update(payload)
    return np.frombuffer(payload, np.uint8).reshape(height, width, 3), source_hash.hexdigest()


def capture_pair(folder, stage, slot, expected_model=None):
    """Check the single-chain log, dimensions and frame numbering before pairing."""
    if type(slot) is not int or slot < 0 or (stage == "model" and not slot):
        raise ValueError("Dump slot must be a nonnegative integer; model pairs require slot >= 1")
    config_path, log_path = folder / "dlss5fb.ini", folder / "dlss5fb.log"
    config_bytes, log_bytes = config_path.read_bytes(), log_path.read_bytes()
    settings = ConfigParser(interpolation=None)
    settings.read_string(config_bytes.decode("utf-8-sig"))
    cfg = settings["DLSS5"]
    first = cfg.getint("DumpFrame")
    if first < 1 or slot >= cfg.getint("DumpCount", fallback=1):
        raise ValueError("Dump slot falls outside the configured capture")
    frame = first + slot
    suffix = "" if slot == 0 else f"_{slot}"
    lines = log_bytes.decode("utf-8", errors="replace").splitlines()
    tracked = [m[1] for line in lines if (m := TRACKED.search(line))]
    released = [(m[1], int(m[2])) for line in lines if (m := RELEASE.search(line))]
    if (len(tracked) != 1 or len(released) != 1 or released[0][0] != tracked[0]
            or released[0][1] < frame or any(re.search(r"\bFAIL\b", line) for line in lines)):
        raise ValueError("Expected a successful single-chain runtime with enough frames")

    keys = ("in", "nr", "out") if slot == 0 else ("in", "out", "lo", "nr")
    names = {key: f"dlss5fb_{key}{suffix}.ppm" for key in keys}
    dumps = [(i, m[1], (int(m[2]), int(m[3]))) for i, line in enumerate(lines) if (m := DUMP.search(line))]
    matches = [i for i, row in enumerate(dumps) if row[1] in names.values()]
    if (len(matches) != len(keys) or matches != list(range(matches[0], matches[0] + len(keys)))
            or [dumps[i][1] for i in matches] != list(names.values())):
        raise ValueError("Missing, duplicate or noncontiguous same-frame dump log")
    prior = lines[:dumps[matches[0]][0]]
    sizes = [tuple(map(int, m.groups())) for line in prior if (m := SIZE.search(line))]
    features = [tuple(map(int, m.groups())) for line in prior if (m := FEATURE.search(line))]
    if not sizes or not features or features[-1] != sizes[-1][2:]:
        raise ValueError("No successful feature at the logged capture size")
    back_size, model_size = sizes[-1][:2], sizes[-1][2:]
    if any(dumps[i][2] != (model_size if key in {"lo", "nr"} else back_size)
           for key, i in zip(keys, matches)):
        raise ValueError("Dump dimensions disagree with the runtime back buffer/feature")

    pair_keys = ("lo", "nr") if stage == "model" else ("in", "out")
    expected_size = model_size if stage == "model" else back_size
    before, after = (read_ppm(folder / names[key]) for key in pair_keys)
    if any(raster.shape != (expected_size[1], expected_size[0], 3) for raster, _ in (before, after)):
        raise ValueError("PPM dimensions disagree with the dump log")
    runtime_hashes = {name: digest(folder / name) for name in ("nvngx_dlssnr.dll", "dxgi.dll", "dlss5_nvngx.dll")}
    if expected_model is not None and runtime_hashes["nvngx_dlssnr.dll"] != expected_model:
        raise ValueError("Capture model does not match the expected SHA-256")
    process_path = folder / "process.json"
    if process_path.is_file():
        process = process_path.read_bytes()
        if json.loads(process.decode("utf-8-sig"))["exit_code"] != 0:
            raise ValueError("Capture process did not exit successfully")
        runtime_hashes[process_path.name] = hashlib.sha256(process).hexdigest()
    runtime_hashes[config_path.name] = hashlib.sha256(config_bytes).hexdigest()
    runtime_hashes[log_path.name] = hashlib.sha256(log_bytes).hexdigest()
    evidence = dict(origin="fallback_dump", present=frame, dump_slot=slot, configured_first_dump=first,
                    back_buffer_size_wh=back_size, model_size_wh=model_size, released_frames=released[0][1],
                    source_names=[names[key] for key in pair_keys], runtime_sha256=runtime_hashes,
                    model_sha256_pinned=expected_model is not None)
    return before, after, evidence


def heatmap(values, signed=False):
    palette = SIGNED_COLORS if signed else ABS_COLORS
    knots = np.linspace(-LIMIT if signed else 0, LIMIT, len(palette))
    rgb = np.empty((*values.shape, 3), np.uint8)
    for i in range(3):
        channel = np.interp(values, knots, palette[:, i])
        np.rint(channel, out=channel)
        rgb[..., i] = channel
    return rgb


def region_points(points):
    rows = []
    for item in points:
        label, x, y = item["label"], item["x"], item["y"]
        if not isinstance(label, str) or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in (x, y)):
            raise ValueError("Each region needs a text label and normalized x/y in [0, 1]")
        rows.append(dict(label=label, x=x, y=y))
    return rows


def demo_pair():
    y, x = np.indices((240, 384))
    checker = ((x // 16 + y // 16) % 2) * 18
    base = np.stack((35 + x // 3 + checker, 24 + y // 2 + checker, 74 + (x + y) // 8 + checker), axis=-1)
    ellipse = ((x - 145) / 68)**2 + ((y - 108) / 55)**2 < 1
    band = (x > 257) & (y > 115)
    out = np.clip(base + 28 * ellipse[..., None] - 22 * band[..., None], 0, 255).astype(np.uint8)
    return (base.astype(np.uint8), None), (out, None), dict(origin="synthetic_demo", present=None)


def build_dataset(item, base_dir, assets):
    stage = item.get("stage", "pair")
    if stage in ("present", "model"):
        expected = item.get("expected_model_sha256")
        if expected is not None and (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected)):
            raise ValueError("expected_model_sha256 must be a 64-digit hex string")
        before, after, evidence = capture_pair(base_dir / item["capture"], stage, item.get("slot", 0),
                                               expected.lower() if expected else None)
    elif stage == "pair":
        before, after = (read_ppm(base_dir / item[key]) for key in ("before", "after"))
        evidence = dict(origin="user_supplied_pair", present=None)
    elif stage == "demo":
        before, after, evidence = demo_pair()
    else:
        raise ValueError(f"Unknown stage: {stage}")
    (a, sha_a), (b, sha_b) = before, after
    if a.shape != b.shape:
        raise ValueError("Before/after raster sizes differ")
    height, width, _ = a.shape
    delta = np.subtract(b, a, dtype=np.int16)
    absolute = np.abs(delta).mean(axis=2)
    signed = delta @ RGB
    data = dict(id=item["id"], label=item["label"], stage=stage, width=width, height=height,
                present=evidence["present"], description=item.get("description", ""),
                regions=region_points(item.get("regions", [])), mae=float(absolute.mean()),
                p95=float(np.percentile(absolute, 95)), luma=float(signed.mean()))
    evidence.update(dataset_id=item["id"], size_wh=[width, height],
                    source_ppm_sha256=dict(before=sha_a, after=sha_b) if sha_a else None,
                    max_abs_pixel_mean_255=float(absolute.max()), max_abs_channel_255=int(np.abs(delta).max()),
                    mae=data["mae"], p95=data["p95"], luma=data["luma"], pngs={})
    del delta
    for key in ("before", "after", "absolute", "signed"):
        pixels = a if key == "before" else b if key == "after" else heatmap(absolute) if key == "absolute" else heatmap(signed, True)
        name = f"{data['id']}_{key}.png"
        path = assets / name
        Image.fromarray(pixels).save(path, format="PNG", compress_level=3)
        with Image.open(path) as check:
            if check.size != (width, height) or check.mode != "RGB" or not np.array_equal(np.asarray(check), pixels):
                raise ValueError(f"PNG lossless round-trip failed: {name}")
        evidence["pngs"][name] = dict(bytes=path.stat().st_size, sha256=digest(path))
        data[key] = f"./assets/{name}"
    return data, evidence


def write_json(path, data, prefix="", suffix="\n"):
    path.write_text(prefix + json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + suffix,
                    encoding="utf-8", newline="\n")


def generate(spec, base_dir, destination, *, demo=False):
    items = spec["datasets"]
    if not isinstance(items, list) or not items:
        raise ValueError("Spec must contain a nonempty datasets list")
    ids = set()
    for item in items:
        name = item["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name) or name in ids:
            raise ValueError("Dataset IDs must be unique short lowercase ASCII slugs")
        if not isinstance(item["label"], str) or not isinstance(item.get("description", ""), str):
            raise ValueError("Labels and descriptions must be strings")
        if item.get("stage") == "demo" and not demo:
            raise ValueError("Synthetic examples are available only with --demo")
        ids.add(name)
    default = spec.get("default_dataset", items[0]["id"])
    title, meta = spec.get("title", "NR 图像对照"), spec.get("meta", "")
    if not isinstance(title, str) or not isinstance(meta, str) or not isinstance(default, str) or default not in ids:
        raise ValueError("Title/meta must be strings and default_dataset must name a dataset")

    destination = destination.absolute()
    destination = destination.parent.resolve(strict=True) / destination.name
    if os.path.lexists(destination):
        raise FileExistsError("Output must be a new directory; existing content was preserved")
    with TemporaryDirectory(prefix=".nr-viewer-", dir=destination.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        assets = bundle / "assets"
        assets.mkdir(parents=True)
        views, records = [], []
        for item in items:
            view, record = build_dataset(item, base_dir, assets)
            views.append(view)
            records.append(record)
            print(f"{view['id']}: {view['width']}x{view['height']} RGB MAE={view['mae']:.6f}/255")
        write_json(assets / "evidence.json", dict(schema=1, scope="Local 8-bit RGB pairs; no image resizing or rendering",
                   absolute_scale_255=[0, LIMIT], signed_Yprime_scale_255=[-LIMIT, LIMIT],
                   signed_Yprime_weights=RGB.tolist(), datasets=records))
        write_json(bundle / "datasets.js", dict(title=title, meta=meta, default_dataset=default, datasets=views),
                   "window.NR_DATA = ", ";\n")
        shutil.copyfile(Path(__file__).with_name("index.html"), bundle / "index.html")
        if os.path.lexists(destination):
            raise FileExistsError("Output appeared during generation; existing content was preserved")
        bundle.rename(destination)
    return destination / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spec", type=Path, help="Local JSON spec; input paths are relative to the spec")
    mode.add_argument("--demo", action="store_true", help="Make a synthetic sample without capture data")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; its parent must already exist")
    args = parser.parse_args()
    try:
        if args.demo:
            spec = dict(title="NR 查看器 · 合成示例", meta="合成像素 · 不含模型运行",
                        datasets=[dict(id="demo", label="合成对照", stage="demo",
                                       description="由代码生成的测试图，不是游戏画面或模型效果。",
                                       regions=[dict(label="提亮椭圆", x=.38, y=.45),
                                                dict(label="压暗区域", x=.83, y=.73)])])
            result = generate(spec, Path.cwd(), args.out, demo=True)
        else:
            spec_path = args.spec.resolve(strict=True)
            spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
            result = generate(spec, spec_path.parent, args.out)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ConfigError) as error:
        parser.exit(1, f"{error}\n")
    print(result)


if __name__ == "__main__":
    main()
