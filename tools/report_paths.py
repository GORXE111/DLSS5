"""Shared, stdlib-only report output protection for local validation tools.

Report and atomic temporary file may never alias declared inputs (including
hard links), nor live inside protected capture/model directories. Enumeration
is limited to the project's declared source trees and supplied input folders.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".cpp", ".c", ".h", ".hpp", ".ps1", ".bat", ".cmd", ".md", ".json", ".ptx"}
REFERENCE_SUFFIXES = {".ppm", ".png", ".npy", ".npz", ".bin", ".dll", ".exe", ".cubin", ".fatbin", ".ptx", ".tsv"}
SOURCE_SUFFIXES |= REFERENCE_SUFFIXES


def aliases(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    if str(left).casefold() == str(right).casefold():
        return True
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def project_inputs(root=ROOT):
    """Known source/configuration files and private model directories, not an E: scan."""
    root = Path(root).resolve()
    files = [root / name for name in ("README.md", "PLAN.md", ".gitignore", ".gitattributes", "weights_map.json", "WEIGHTS_HT.bin",
                                      ".git/index", ".git/HEAD", ".git/config", ".git/packed-refs")]
    for name in ("tools", "torch", "harness", "plugin"):
        folder = root / name
        if folder.is_dir():
            files.extend(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)
    research = root / "research"
    if research.is_dir():
        files.extend(research.glob("*.py"))
        # Retained authored source snapshots are immutable reference inputs.
        for folder in research.glob("**/sources"):
            if folder.is_dir():
                files.extend(path for path in folder.rglob("*") if path.is_file())
    directories = [root / name for name in ("dlss5", "fatbins", "sm86_port")]
    directories.extend(path for path in root.glob("sm86_port.*") if path.is_dir())
    return files, directories


def reference_inputs(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in REFERENCE_SUFFIXES]


class ReportPath:
    def __init__(self, target, *, inputs=(), input_dirs=(), root=ROOT):
        self.path = Path(target).resolve()
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.files, self.directories = project_inputs(root)
        self.files.extend(Path(path) for path in inputs)
        self.directories.extend(Path(path).resolve() for path in input_dirs)
        # Directory containment alone cannot catch an outside hard link.
        for directory in self.directories:
            if directory.is_dir():
                self.files.extend(path for path in directory.rglob("*") if path.is_file())
        self.files = list(dict.fromkeys(Path(path).resolve() for path in self.files))
        self.directories = list(dict.fromkeys(Path(path).resolve() for path in self.directories))
        self.validate()

    def validate(self):
        if aliases(self.path, self.temporary):
            raise ValueError("Report aliases its temporary output")
        for output in (self.path, self.temporary):
            resolved = output.resolve()
            if resolved.exists() and resolved.is_dir():
                raise ValueError("Report output must be a file: " + str(output))
            if any(resolved.is_relative_to(directory) for directory in self.directories):
                raise ValueError("Report must be outside protected input directories: " + str(output))
            for source in self.files:
                if aliases(output, source):
                    raise ValueError("Report aliases a source/reference/model input: " + str(source))
        return self.path

    def add_inputs(self, *paths):
        self.files.extend(Path(path).resolve() for path in paths)
        self.files = list(dict.fromkeys(self.files))
        self.validate()

    def write_json(self, data):
        self.validate()
        serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create preserves an unrelated pre-existing temporary file.
        created_identity = None
        try:
            with self.temporary.open("x", encoding="utf-8", newline="\n") as stream:
                identity = os.fstat(stream.fileno())
                created_identity = identity.st_dev, identity.st_ino
                stream.write(serialized)
            self.validate()
            self.temporary.replace(self.path)
        except BaseException:
            if created_identity is not None:
                try:
                    identity = self.temporary.lstat()
                    if (identity.st_dev, identity.st_ino) == created_identity:
                        self.temporary.unlink()
                except FileNotFoundError:
                    pass
            raise
