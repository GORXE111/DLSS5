"""Build and package DLSS5-RTX30; --verify-only checks hashes and source provenance.
Existing packages are preserved as sibling backups. --out must be a dedicated
immediate child of plugin/dist, keeping source and unrelated user directories safe.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
INI_OVERRIDES = {"DlssNr": {"Enabled": "true", "AutoCapture": "false", "WorkingScale": "0.5"}}


def patch_ini(text, overrides):
    """Edit selected keys while preserving comments and other sections."""
    out, section, seen = [], None, set()
    for line in text.splitlines():
        m = re.match(r"\s*\[(\w+)\]\s*$", line)
        if m:
            if section in overrides:
                for k, v in overrides[section].items():
                    if (section, k) not in seen:
                        out.append(f"{k}={v}")
            section = m.group(1)
        kv = re.match(r"(\w+)\s*=", line)
        if section in overrides and kv and kv.group(1) in overrides[section]:
            k = kv.group(1)
            out.append(f"{k}={overrides[section][k]}")
            seen.add((section, k))
            continue
        out.append(line)
    if section in overrides:
        for k, v in overrides[section].items():
            if (section, k) not in seen:
                out.append(f"{k}={v}")
    return "\r\n".join(out) + "\r\n"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_inventory(directory):
    directory = Path(directory)
    return [{"path": p.relative_to(directory).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(directory.rglob("*")) if p.is_file()]


def source_inventory():
    """Include native / managed sources, build scripts and the external SDK headers."""
    paths = [Path(__file__), *Path(HERE, "src").rglob("*")]
    for folder in ("manager", "fallback"):
        paths.extend(Path(HERE, folder).glob("*"))
    paths.append(Path(ROOT, "harness", "nvngx-bridge.cpp"))
    for folder in ("oss/nvidia-dlss/include", "oss/DLSS5-Reshade-AIO/addon/include"):
        paths.extend(Path(ROOT, folder).rglob("*.h"))
    extensions = {".py", ".cs", ".cpp", ".h", ".asm", ".def", ".bat", ".ps1", ".txt"}
    return [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)}
            for p in sorted(set(paths)) if p.is_file() and p.suffix.lower() in extensions]


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_output(path):
    """Reject non-dedicated targets and Windows junctions before any replacement."""
    candidate = Path(os.path.abspath(path))
    allowed = Path(HERE, "dist").resolve()
    if candidate.parent.resolve() != allowed or candidate.name.startswith("."):
        raise ValueError(f"--out must be a dedicated child directory of {allowed}")
    if candidate.exists() and (not candidate.is_dir() or candidate.resolve().parent != allowed):
        raise ValueError(f"Unsafe package output: {candidate}")
    for p in [candidate, *candidate.rglob("*")] if candidate.exists() else []:
        if getattr(p.lstat(), "st_file_attributes", 0) & 0x400 or p.is_symlink():
            raise ValueError(f"Refusing reparse point in package output: {p}")
    return candidate


def validate_existing(path):
    if not path.exists():
        return
    manifest = path / "package.json"
    try:
        info = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Existing output is not a recognized package; preserving it: {path}") from exc
    if info.get("package") != "DLSS5-RTX30":
        raise ValueError(f"Unrecognized package ownership: {path}")
    if info.get("schema_version") == 2:
        expected = {f["path"]: f for f in info["artifacts"]}
    else:
        # Legacy packages only had payload sizes. Preserve their complete tree as a backup.
        expected = {"payload/" + f["path"]: f for f in info["files"]}
        for name in ("install.ps1", "使用说明.txt", "安装.bat", "卸载.bat", "DLSS5Manager.exe", "dlss5.exe"):
            expected[name] = {}
    actual = {p.relative_to(path).as_posix(): p for p in path.rglob("*") if p.is_file() and p != manifest}
    if set(actual) != set(expected):
        raise ValueError(f"Existing output has missing or unrecognized files; preserving it: {path}")
    for name, p in actual.items():
        record = expected[name]
        if "sha256" in record and sha256(p) != record["sha256"]:
            raise ValueError(f"Existing package has a modified file; preserving it: {p}")
        if "size" in record and p.stat().st_size != record["size"]:
            raise ValueError(f"Existing package has a modified size; preserving it: {p}")


def verify_package(path):
    validate_existing(path)
    info = json.loads((path / "package.json").read_text(encoding="utf-8"))
    if info.get("schema_version") != 2:
        raise ValueError("Package must contain a version 2 manifest")
    if fingerprint(info["artifacts"]) != info["build"]["artifacts_fingerprint"]:
        raise ValueError("Artifact fingerprint mismatch")
    if source_inventory() != info["build"]["sources"]:
        raise ValueError("Package source fingerprint differs from current build inputs; rebuild the package")
    if fingerprint(info["build"]["sources"]) != info["build"]["source_fingerprint"]:
        raise ValueError("Source fingerprint mismatch")
    return info


def assemble(a, out, sources):
    payload = out / "payload"
    shutil.copytree(a.opti, payload, ignore=shutil.ignore_patterns(
        "setup_windows.bat", "setup_linux.sh", "!! EXTRACT ALL FILES TO GAME FOLDER !!"))
    ini = payload / "OptiScaler.ini"
    text = ini.read_text(encoding="utf-8", errors="surrogateescape")
    with ini.open("w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write(patch_ini(text, INI_OVERRIDES))
    shutil.copy2(a.dll, payload / "nvngx_dlssnr.dll")
    if len(re.findall(r"(?m)^WorkingScale=", ini.read_text(encoding="utf-8", errors="surrogateescape"))) != 1:
        raise ValueError("Expected exactly one WorkingScale setting")
    # BOM keeps Chinese readable in Windows PowerShell 5.1.
    for name in ("install.ps1", "使用说明.txt"):
        text = Path(HERE, "src", name).read_text(encoding="utf-8")
        with (out / name).open("w", encoding="utf-8-sig", newline="\r\n") as f:
            f.write(text)
    for src, dst in (("install.bat", "安装.bat"), ("uninstall.bat", "卸载.bat")):
        shutil.copy2(Path(HERE, "src", src), out / dst)
    for exe in ("DLSS5Manager.exe", "dlss5.exe"):
        shutil.copy2(Path(HERE, "manager", "bin", exe), out / exe)
    (payload / "fallback").mkdir()
    for dll in ("dxgi.dll", "dlss5_nvngx.dll"):
        shutil.copy2(Path(HERE, "fallback", "bin", dll), payload / "fallback" / dll)
    artifacts = file_inventory(out)
    for item in artifacts:
        p = out / item["path"]
        if p.suffix.lower() in {".dll", ".exe"}:
            with p.open("rb") as f:
                if f.read(2) != b"MZ":
                    raise ValueError(f"Invalid executable in package: {p}")
    info = {"schema_version": 2, "package": "DLSS5-RTX30", "optiscaler": Path(a.opti).name,
            "dlssnr_sha256": sha256(a.dll), "files": file_inventory(payload), "artifacts": artifacts,
            "build": {"created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
                      "source_fingerprint": fingerprint(sources), "artifacts_fingerprint": fingerprint(artifacts)}}
    (out / "package.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opti", default=os.path.join(ROOT, "_dl", "opti_020"))
    ap.add_argument("--dll", default=os.path.join(ROOT, "sm86_port", "nvngx_dlssnr.dll"))
    ap.add_argument("--out", default=os.path.join(HERE, "dist", "DLSS5-RTX30"))
    ap.add_argument("--verify-only", action="store_true", help="Verify all artifact hashes and current source fingerprint")
    a = ap.parse_args()
    out = validate_output(a.out)
    if a.verify_only:
        info = verify_package(out)
        print(f"Verified {len(info['artifacts'])} artifacts; source {info['build']['source_fingerprint']}")
        return
    validate_existing(out)
    if not Path(a.opti, "OptiScaler.ini").is_file() or not Path(a.dll).is_file():
        raise ValueError("Missing OptiScaler input or ported model DLL")
    sources = source_inventory()
    # Complete every build before staging or replacing an existing package.
    for folder in ("manager", "fallback"):
        subprocess.run(["cmd", "/c", os.path.join(HERE, folder, "build.bat")], check=True)
    if source_inventory() != sources:
        raise ValueError("Source inputs changed during build; rerun packaging")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Only TemporaryDirectory's newly created, owned path is ever recursively removed.
    with tempfile.TemporaryDirectory(prefix=".dlss5-package-", dir=out.parent) as temporary:
        staged = Path(temporary, out.name)
        staged.mkdir()
        info = assemble(a, staged, sources)
        verify_package(staged)
        validate_existing(out)
        backup = None
        if out.exists():
            backup = Path(temporary).with_name(out.name + ".previous-" + Path(temporary).name.removeprefix(".dlss5-package-"))
            out.rename(backup)
        try:
            staged.rename(out)
        except OSError:
            if backup is not None:
                backup.rename(out)
            raise
    verify_package(out)
    print(f"Package complete {out}: {len(info['artifacts'])} files; source {info['build']['source_fingerprint']}")
    if backup is not None:
        print(f"Previous package preserved: {backup}")


if __name__ == "__main__":
    main()