#!/usr/bin/env python3
"""Build an allowlisted source archive without data, private notes or Git history."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
import tarfile

REPO = Path(__file__).resolve().parents[1]
PREFIX = "plato-imu-0.1.0"
ROOT_FILES = ("README.md", "REPRODUCIBILITY.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "VALIDATION.md", "VALIDATION.json", "AQUALUX_VALIDATION.md", "AQUALUX_VALIDATION.json", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff", "pyproject.toml", "requirements.txt", "environment.yml", ".gitignore")
MODULES = ("dataset", "aqualux_dataset", "Interpolation", "lie_algebra", "utils", "network_for_BiadDy", "learning", "evalutation", "mainEurocPLATO", "mainAquaLuxPLATO")
TOOLS = ("train_euroc_release.py", "evaluate_euroc_release.py", "train_aqualux_release.py", "evaluate_aqualux_release.py", "check_paper_metrics.py", "build_release.py", "export_release_checkpoint.py", "verify_aqualux_layout.py")


def selected_files(root: Path = REPO) -> list[Path]:
    relative = list(ROOT_FILES)
    relative += [f"external/Debias_IMU/BiasDy/{name}.py" for name in MODULES]
    relative += [f"tools/{name}" for name in TOOLS]
    relative += ["scripts/create_env.sh", "scripts/smoke_test.sh", "external/Debias_IMU/LICENSE"]
    for directory, pattern in (("external/Debias_IMU/BiasDy/plato", "*.py"), ("configs", "*.json"), ("tests", "*.py"), (".github/workflows", "*.yml")):
        relative += [str(path.relative_to(root)) for path in (root / directory).rglob(pattern)]
    paths = sorted(root / name for name in set(relative))
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or unsafe release input: {path}")
        content = path.read_text(encoding="utf-8")
        # This is a narrow accidental-disclosure check, not a security audit.
        if re.search(r"/home/[A-Za-z0-9_.-]+/", content) or ("-----BEGIN " + "PRIVATE KEY-----") in content:
            raise ValueError(f"Host-specific path or private key in public source: {path}")
    return paths


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = mode
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def build(output: Path, root: Path = REPO) -> dict:
    paths = selected_files(root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w|") as archive:
        for path in paths:
            add_bytes(archive, f"{PREFIX}/{path.relative_to(root)}", path.read_bytes(), mode=0o755 if path.suffix == ".sh" else 0o644)
        add_bytes(archive, f"{PREFIX}/SOURCE_MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return {"file_count": len(paths), "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "dist" / f"{PREFIX}.tar.gz")
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps({"output": str(args.output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
