#!/usr/bin/env python3
"""Copy installed model source and optional weights into THIS independent checkout.

Never copies .git, virtualenvs, datasets or outputs. Source provenance is recorded
with SHA-256 hashes. Re-running accepts identical assets and rejects conflicts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".md", ".sh", ".cpp", ".h", ".cu", ".cuh"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    before = source.stat()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_hash = sha256(source)
        if destination.stat().st_size != before.st_size or sha256(destination) != source_hash:
            raise RuntimeError(f"destination differs; refusing to overwrite {destination}")
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as f:
                temporary = Path(f.name)
                with source.open("rb") as original:
                    h = hashlib.sha256()
                    for block in iter(lambda: original.read(8 * 1024**2), b""):
                        f.write(block)
                        h.update(block)
                source_hash = h.hexdigest()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"source changed while copying: {source}")
            shutil.copystat(source, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(bytes=before.st_size, sha256=source_hash)


def copy_model(source, destination, package):
    source = Path(source).resolve()
    if not (source / package).is_dir() or not (source / "LICENSE").is_file():
        raise FileNotFoundError(f"need {package}/ and LICENSE under {source}")
    sources = [source / "LICENSE"]
    sources += [source / n for n in ("README.md", "pyproject.toml", "requirements.txt") if (source/n).is_file()]
    for path in sorted((source / package).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and "license" not in path.name.lower():
            raise ValueError(f"unexpected package resource {path}; inspect it before extending the source allowlist")
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError(f"unexpectedly large source file: {path}")
        sources.append(path)
    inventory = {}
    for path in sources:
        relative = path.relative_to(source)
        inventory[str(relative)] = copy_file(path, destination / relative)
    try:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    # Hashes, rather than just HEAD, identify installed/uncommitted model edits.
    return dict(source_commit=revision, files=inventory)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--copy-weights", action="store_true")
    a = p.parse_args()
    source = a.source_root.resolve()
    if source == ROOT or source in ROOT.parents or ROOT in source.parents:
        raise ValueError("source and destination must be independent directory trees")
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    records = {}
    for name, directory, package in (("omega", "vggt-omega", "vggt_omega"), ("moge", "moge", "moge")):
        print(f"Copying {name} model source into {vendor / name}", flush=True)
        records[name] = copy_model(source / directory, vendor / name, package)
    (vendor / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    if a.copy_weights:
        weights = [(source / "vggt-omega/checkpoints/vggt_omega_1b_512.pt", ROOT / "checkpoints/vggt_omega_1b_512.pt"),
                   (source / "checkpoints/moge-2-vitl/model.pt", ROOT / "checkpoints/moge-2-vitl/model.pt")]
        needed = sum(src.stat().st_size for src, dst in weights if not dst.exists())
        if shutil.disk_usage(ROOT).free < needed + 1024**3:
            raise RuntimeError("not enough filesystem space for independent checkpoint copies")
        fingerprints = {}
        for src, dst in weights:
            print(f"Copying/checking checkpoint {dst.name} ({src.stat().st_size/1024**3:.2f} GiB)", flush=True)
            fingerprints[str(dst.relative_to(ROOT))] = copy_file(src, dst)
        (ROOT / "checkpoints/fingerprints.json").write_text(json.dumps(fingerprints, indent=2)+"\n")
    print("ASSET_IMPORT=PASS; model source is Git-trackable, checkpoints stay ignored", flush=True)


if __name__ == "__main__":
    main()
